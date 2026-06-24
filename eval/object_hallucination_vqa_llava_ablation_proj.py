"""
Ablation-Projection Decoding (APD) 평가 스크립트.

진단(diagnose_hallucination_direction.py)에서 계산한 d_ablation 방향을
중간 레이어 hidden state에서 직교 투영으로 제거하며 생성.

VCD noisy branch 없음. 표준 generation + 레이어 hook.
--directions_file: diagnose_hallucination_direction.py가 저장한 directions.pt 경로
--proj_layer    : hook을 걸 레이어 인덱스 (기본 14, AUROC 0.80)
--proj_gamma    : 제거 강도 (기본 1.0 = 완전 직교 투영)
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

    # d_ablation 로드
    if not os.path.exists(args.directions_file):
        raise FileNotFoundError(
            f"directions.pt 없음: {args.directions_file}\n"
            "먼저 diagnose_hallucination_direction.py를 실행해 directions.pt를 생성하세요."
        )
    dirs = torch.load(args.directions_file, map_location='cpu')
    d_ablation_all = dirs['d_ablation']
    if args.proj_layer not in d_ablation_all:
        raise KeyError(
            f"proj_layer={args.proj_layer}가 directions.pt에 없음. "
            f"사용 가능한 레이어: {sorted(d_ablation_all.keys())}"
        )
    d_hat = d_ablation_all[args.proj_layer].float()  # (4096,) float32
    print(f"d_ablation 로드: layer={args.proj_layer}, norm={d_hat.norm():.4f}, gamma={args.proj_gamma}")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # hook은 루프 밖에서 한 번만 등록 (per-question 등록/해제 시 GPU 텐서 누적 방지)
    hook_handle = register_proj_hook(model, args.proj_layer, d_hat, args.proj_gamma)

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

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

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
            gc.collect()
            torch.cuda.empty_cache()
            continue

        input_token_len = input_ids.shape[1]
        n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff > 0:
            print(f'[Warning] {n_diff} output_ids differ from input_ids')

        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "model_id": model_name,
            "image": image_file,
            "metadata": {},
        }) + "\n")
        ans_file.flush()

        # KV 캐시 및 중간 텐서 명시적 해제
        del output_ids, input_ids, image_tensor
        gc.collect()
        torch.cuda.empty_cache()

    hook_handle.remove()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",       type=str,   default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base",       type=str,   default=None)
    parser.add_argument("--image-folder",     type=str,   default="")
    parser.add_argument("--question-file",    type=str,   default="tables/question.jsonl")
    parser.add_argument("--answers-file",     type=str,   default="answer.jsonl")
    parser.add_argument("--conv-mode",        type=str,   default="llava_v1")
    parser.add_argument("--temperature",      type=float, default=1.0)
    parser.add_argument("--top_p",            type=float, default=1)
    parser.add_argument("--top_k",            type=int,   default=None)
    parser.add_argument("--seed",             type=int,   default=42)
    # APD 전용 인자
    parser.add_argument("--directions_file",  type=str,   default="./diag_output/directions.pt")
    parser.add_argument("--proj_layer",       type=int,   default=14)
    parser.add_argument("--proj_gamma",       type=float, default=1.0,
                        help="제거 강도: 0=변화없음, 1=완전직교투영, >1=과도보정")
    parser.add_argument("--max_new_tokens",   type=int,   default=16,
                        help="POPE는 yes/no이므로 16으로 충분. OOM 방지용.")
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
