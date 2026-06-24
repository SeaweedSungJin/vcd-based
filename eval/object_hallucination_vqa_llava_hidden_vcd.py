"""
Fix 2: Hidden-state VCD 평가 스크립트.

VCD와 동일한 two-branch 구조이되, layer L hidden state 수준에서 결합:
    h_star = (1+α)·h_normal_L − α·h_noisy_L

남은 레이어(L+1...31) + norm + lm_head 를 통과해서 logit을 계산.
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
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample_hidden_vcd import evolve_vcd_sampling_hidden
evolve_vcd_sampling_hidden()


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions    = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

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

        image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    images_cd=image_tensor_cd.unsqueeze(0).half().cuda(),
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta,
                    proj_layer=args.proj_layer,
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

        del output_ids, input_ids, image_tensor, image_tensor_cd
        gc.collect()
        torch.cuda.empty_cache()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",    type=str,   default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base",    type=str,   default=None)
    parser.add_argument("--image-folder",  type=str,   default="")
    parser.add_argument("--question-file", type=str,   default="tables/question.jsonl")
    parser.add_argument("--answers-file",  type=str,   default="answer.jsonl")
    parser.add_argument("--conv-mode",     type=str,   default="llava_v1")
    parser.add_argument("--temperature",   type=float, default=1.0)
    parser.add_argument("--top_p",         type=float, default=1)
    parser.add_argument("--top_k",         type=int,   default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--noise_step",    type=int,   default=500)
    parser.add_argument("--cd_alpha",      type=float, default=1.0)
    parser.add_argument("--cd_beta",       type=float, default=0.1)
    parser.add_argument("--proj_layer",    type=int,   default=14)
    parser.add_argument("--max_new_tokens",type=int,   default=8)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
