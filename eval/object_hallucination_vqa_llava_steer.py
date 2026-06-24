"""
Fix 1: Additive Steering 평가 스크립트.

APD (직교 투영)와 달리, d_ablation 방향으로 고정량(gamma)만큼 hidden state를 이동:
    h* = h - γ·d̂_abl    (투영 계수 무관, 상수 이동)

d_ablation 방향 = language prior 방향이므로 빼는 것 = visual grounding 방향으로 push.
"""

import argparse
import gc
import torch
import os
import json
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
from transformers import set_seed
from vcd_utils.ablation_proj import register_proj_hook


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    if not os.path.exists(args.directions_file):
        raise FileNotFoundError(f"directions.pt 없음: {args.directions_file}")
    dirs   = torch.load(args.directions_file, map_location='cpu')
    d_hat  = dirs['d_ablation'][args.proj_layer].float()
    print(f"[Steer] layer={args.proj_layer}, gamma={args.proj_gamma}, ||d||={d_hat.norm():.4f}")

    questions    = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    hook_handle = register_proj_hook(model, args.proj_layer, d_hat, args.proj_gamma, mode="steer")

    for line in tqdm(questions):
        idx        = line["question_id"]
        image_file = line["image"]
        qs         = line["text"]
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids    = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image        = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        stop_str     = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM skip] question_id={idx}")
            gc.collect(); torch.cuda.empty_cache(); continue

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({
            "question_id": idx, "prompt": cur_prompt, "text": outputs,
            "model_id": model_name, "image": image_file, "metadata": {},
        }) + "\n")
        ans_file.flush()

        del output_ids, input_ids, image_tensor
        gc.collect()
        torch.cuda.empty_cache()

    hook_handle.remove()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",      type=str,   default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base",      type=str,   default=None)
    parser.add_argument("--image-folder",    type=str,   default="")
    parser.add_argument("--question-file",   type=str,   default="tables/question.jsonl")
    parser.add_argument("--answers-file",    type=str,   default="answer.jsonl")
    parser.add_argument("--conv-mode",       type=str,   default="llava_v1")
    parser.add_argument("--temperature",     type=float, default=1.0)
    parser.add_argument("--top_p",           type=float, default=1)
    parser.add_argument("--top_k",           type=int,   default=None)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--directions_file", type=str,   default="./diag_output/directions.pt")
    parser.add_argument("--proj_layer",      type=int,   default=14)
    parser.add_argument("--proj_gamma",      type=float, default=10.0)
    parser.add_argument("--max_new_tokens",  type=int,   default=8)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
