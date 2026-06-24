"""
EPP (Evidence-Protected Projection) 평가 스크립트.

Task:
  pope  — POPE yes/no VQA  (Acc, F1, yes-ratio)
  chair — CHAIR caption eval (CHAIR_s, CHAIR_i, Recall, avg_len)

Modes:
  none     — 기본 생성 (no projection)
  ablation — d_ablation projection (정제 없음)
  epp_b1   — d_clean_B1 projection (ILVAD e_img, single-pass)
  epp_b3   — d_clean_B3 projection (image-ablation diff e_img, 2-pass)

Usage example (mini comparison — 500 POPE popular + 100 CHAIR):
  # POPE
  python eval/object_hallucination_vqa_llava_epp.py \\
    --task pope \\
    --mode epp_b1 \\
    --pope_file /home/sjkim/ILVAD/data/pope/coco/coco_pope_popular.json \\
    --image_folder /home/sjkim/datasets/coco/val2014 \\
    --directions_pt ../diag_output/directions.pt \\
    --proj_layers 16 --gamma 1.0 \\
    --n_questions 500 \\
    --answers_file ./mini_results/pope_popular_epp_b1_g1.0_l16.jsonl

  # CHAIR
  python eval/object_hallucination_vqa_llava_epp.py \\
    --task chair \\
    --mode ablation \\
    --ann_file /home/sjkim/datasets/coco/annotations/instances_val2014.json \\
    --image_folder /home/sjkim/datasets/coco/val2014 \\
    --directions_pt ../diag_output/directions.pt \\
    --proj_layers 16 --gamma 1.0 \\
    --n_images 100 --max_new_tokens 64 \\
    --answers_file ./mini_results/chair_ablation_g1.0_l16.jsonl
"""

import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from vcd_utils.epp_proj import EPPProjector

# ── COCO synonym map ──────────────────────────────────────────────────────────
COCO_SYNONYMS = {
    'person':        ['person','man','woman','people','boy','girl','child','kid','human','men','women'],
    'bicycle':       ['bicycle','bike'],
    'car':           ['car','cars','automobile','vehicle'],
    'motorcycle':    ['motorcycle','motorbike'],
    'airplane':      ['airplane','plane','aircraft','jet'],
    'bus':           ['bus','buses'],
    'train':         ['train','trains'],
    'truck':         ['truck','trucks'],
    'boat':          ['boat','boats','ship'],
    'traffic light': ['traffic light','stoplight'],
    'fire hydrant':  ['fire hydrant','hydrant'],
    'stop sign':     ['stop sign'],
    'bench':         ['bench','benches'],
    'bird':          ['bird','birds'],
    'cat':           ['cat','cats','kitten'],
    'dog':           ['dog','dogs','puppy'],
    'horse':         ['horse','horses'],
    'sheep':         ['sheep'],
    'cow':           ['cow','cows'],
    'elephant':      ['elephant','elephants'],
    'bear':          ['bear','bears'],
    'zebra':         ['zebra','zebras'],
    'giraffe':       ['giraffe','giraffes'],
    'backpack':      ['backpack','rucksack'],
    'umbrella':      ['umbrella','umbrellas'],
    'handbag':       ['handbag','purse','bag'],
    'tie':           ['tie','necktie'],
    'suitcase':      ['suitcase','luggage'],
    'frisbee':       ['frisbee'],
    'skis':          ['skis','ski'],
    'snowboard':     ['snowboard'],
    'sports ball':   ['sports ball','ball'],
    'kite':          ['kite','kites'],
    'baseball bat':  ['baseball bat'],
    'baseball glove':['baseball glove','glove'],
    'skateboard':    ['skateboard'],
    'surfboard':     ['surfboard'],
    'tennis racket': ['tennis racket','racket'],
    'bottle':        ['bottle','bottles'],
    'wine glass':    ['wine glass'],
    'cup':           ['cup','cups','mug'],
    'fork':          ['fork'],
    'knife':         ['knife','knives'],
    'spoon':         ['spoon'],
    'bowl':          ['bowl','bowls'],
    'banana':        ['banana','bananas'],
    'apple':         ['apple','apples'],
    'sandwich':      ['sandwich'],
    'orange':        ['orange','oranges'],
    'broccoli':      ['broccoli'],
    'carrot':        ['carrot','carrots'],
    'hot dog':       ['hot dog','hotdog'],
    'pizza':         ['pizza'],
    'donut':         ['donut','doughnut'],
    'cake':          ['cake'],
    'chair':         ['chair','chairs'],
    'couch':         ['couch','sofa'],
    'potted plant':  ['potted plant','plant'],
    'bed':           ['bed','beds'],
    'dining table':  ['dining table','table'],
    'toilet':        ['toilet'],
    'tv':            ['tv','television','monitor'],
    'laptop':        ['laptop','notebook'],
    'mouse':         ['mouse'],
    'remote':        ['remote','remote control'],
    'keyboard':      ['keyboard'],
    'cell phone':    ['cell phone','phone','smartphone'],
    'microwave':     ['microwave'],
    'oven':          ['oven'],
    'toaster':       ['toaster'],
    'sink':          ['sink'],
    'refrigerator':  ['refrigerator','fridge'],
    'book':          ['book','books'],
    'clock':         ['clock'],
    'vase':          ['vase'],
    'scissors':      ['scissors'],
    'teddy bear':    ['teddy bear','teddy'],
    'hair drier':    ['hair dryer','hair drier'],
    'toothbrush':    ['toothbrush'],
}
_WORD_TO_CAT = {}
for _cat, _words in COCO_SYNONYMS.items():
    for _w in _words:
        _WORD_TO_CAT[_w] = _cat
_SORTED_WORDS = sorted(_WORD_TO_CAT.keys(), key=len, reverse=True)


def extract_objects(text: str):
    cap = text.lower()
    found, covered = [], []
    for w in _SORTED_WORDS:
        for m in re.finditer(r'\b' + re.escape(w) + r'\b', cap):
            s, e = m.start(), m.end()
            if any(cs <= s < ce or cs < e <= ce for cs, ce in covered):
                continue
            covered.append((s, e))
            found.append(_WORD_TO_CAT[w])
    return found


def load_coco_gt(ann_file: str) -> dict:
    data = json.load(open(ann_file))
    cat_map = {c['id']: c['name'] for c in data['categories']}
    gt = defaultdict(set)
    for ann in data['annotations']:
        gt[ann['image_id']].add(cat_map[ann['category_id']])
    return gt


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pope(args, model, tokenizer, image_processor, projector, device):
    questions = [json.loads(q) for q in open(args.pope_file)]
    if args.n_questions > 0:
        questions = questions[:args.n_questions]

    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    ans_file = open(args.answers_file, 'w')

    for line in tqdm(questions, desc=f"POPE [{args.mode}]"):
        idx   = line["question_id"]
        imgf  = line["image"]
        qs    = line["text"]

        qs_full = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], qs_full + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep2

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0)

        image = Image.open(os.path.join(args.image_folder, imgf)).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        try:
            # Prepare per-sample d_proj (for EPP modes)
            projector.prepare(model, input_ids.cpu(), image_tensor.to(device), device)

            with torch.inference_mode(), projector:
                output_ids = model.generate(
                    input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM skip] q_id={idx}")
            gc.collect(); torch.cuda.empty_cache()
            continue

        gen = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        if gen.endswith(stop_str):
            gen = gen[:-len(stop_str)].strip()

        ans_file.write(json.dumps({
            "question_id": idx, "prompt": qs, "text": gen,
            "model_id": "llava-v1.5-7b", "image": imgf, "metadata": {},
        }) + "\n")
        ans_file.flush()

        del output_ids, input_ids, image_tensor
        gc.collect(); torch.cuda.empty_cache()

    ans_file.close()

    # In-place POPE eval
    print(f"\n{'='*60}")
    print(f"POPE 결과 [{args.mode}]")
    print(f"{'='*60}")
    gt_lines  = [json.loads(q) for q in open(args.pope_file)][:args.n_questions if args.n_questions > 0 else None]
    gen_lines = [json.loads(q) for q in open(args.answers_file)]
    tp = tn = fp = fn = yes_cnt = 0
    for gt_line, gen_line in zip(gt_lines, gen_lines):
        gt_ans  = gt_line["label"].lower().strip()
        gen_ans = gen_line["text"].lower().strip()
        is_yes  = 'yes' in gen_ans
        if gt_ans == 'yes':
            (tp := tp + 1) if is_yes else (fn := fn + 1)
        else:
            (fp := fp + 1) if is_yes else (tn := tn + 1)
        if is_yes:
            yes_cnt += 1

    total = tp + tn + fp + fn
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc   = (tp + tn) / total if total > 0 else 0
    yes_r = yes_cnt / total if total > 0 else 0

    metrics = {
        'mode': args.mode, 'task': 'pope',
        'accuracy': round(acc, 4), 'f1': round(f1, 4),
        'precision': round(prec, 4), 'recall': round(rec, 4),
        'yes_ratio': round(yes_r, 4), 'n': total,
    }
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    result_file = args.answers_file.replace('.jsonl', '_metrics.json')
    json.dump(metrics, open(result_file, 'w'), indent=2)
    print(f"  저장: {result_file}")
    return metrics


def run_chair(args, model, tokenizer, image_processor, projector, device):
    coco_gt = load_coco_gt(args.ann_file)
    img_dir = Path(args.image_folder)
    all_imgs = sorted(img_dir.glob("*.jpg"))
    if args.n_images > 0:
        all_imgs = all_imgs[:args.n_images]

    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    ans_file = open(args.answers_file, 'w')

    question = "Please describe this image in detail."
    conv_tmpl = conv_templates['llava_v1']

    n_caps = n_hall_caps = 0
    total_hall_obj = total_mention_obj = total_gt_correct = total_gt_obj = 0
    total_len = 0

    for img_path in tqdm(all_imgs, desc=f"CHAIR [{args.mode}]"):
        img_id_m = re.search(r'(\d+)\.jpg$', img_path.name)
        if not img_id_m:
            continue
        img_id  = int(img_id_m.group(1))
        gt_cats = coco_gt.get(img_id, set())

        qs_full = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = conv_tmpl.copy()
        conv.append_message(conv.roles[0], qs_full)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep2

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0)

        image = Image.open(str(img_path)).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        try:
            projector.prepare(model, input_ids.cpu(), image_tensor.to(device), device)
            with torch.inference_mode(), projector:
                output_ids = model.generate(
                    input_ids.to(device),
                    images=image_tensor.unsqueeze(0).half().to(device),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM skip] {img_path.name}")
            gc.collect(); torch.cuda.empty_cache()
            continue

        caption = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        if caption.endswith(stop_str):
            caption = caption[:-len(stop_str)].strip()

        mentioned_raw = extract_objects(caption)                  # may have repeats
        mentioned_uniq = list(dict.fromkeys(mentioned_raw))        # deduplicate, preserve order
        n_mentioned = len(mentioned_uniq)
        n_hall_in_cap = sum(1 for m in mentioned_uniq if m not in gt_cats)
        n_correct = len(set(mentioned_uniq) & gt_cats)             # GT objects correctly mentioned
        has_hall = (n_hall_in_cap > 0)

        n_caps += 1
        if has_hall:
            n_hall_caps += 1
        total_hall_obj    += n_hall_in_cap
        total_mention_obj += n_mentioned
        total_gt_correct  += n_correct
        total_gt_obj      += len(gt_cats)
        total_len         += len(caption.split())

        ans_file.write(json.dumps({
            "image_id": img_id, "image": img_path.name, "caption": caption,
            "gt_cats": list(gt_cats), "mentioned": mentioned_uniq,
            "n_hall": n_hall_in_cap, "n_mention": n_mentioned,
        }) + "\n")
        ans_file.flush()

        del output_ids, input_ids, image_tensor
        gc.collect(); torch.cuda.empty_cache()

    ans_file.close()

    chair_s  = n_hall_caps / n_caps if n_caps > 0 else 0
    chair_i  = total_hall_obj / total_mention_obj if total_mention_obj > 0 else 0
    recall   = total_gt_correct / total_gt_obj if total_gt_obj > 0 else 0
    avg_len  = total_len / n_caps if n_caps > 0 else 0

    print(f"\n{'='*60}")
    print(f"CHAIR 결과 [{args.mode}]")
    print(f"{'='*60}")
    metrics = {
        'mode': args.mode, 'task': 'chair',
        'n_captions': n_caps,
        'CHAIR_s': round(chair_s, 4),
        'CHAIR_i': round(chair_i, 4),
        'Recall':  round(recall, 4),
        'avg_len': round(avg_len, 1),
        'max_new_tokens': args.max_new_tokens,
    }
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    result_file = args.answers_file.replace('.jsonl', '_metrics.json')
    json.dump(metrics, open(result_file, 'w'), indent=2)
    print(f"  저장: {result_file}")
    return metrics


def main(args):
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path)
    )
    model.eval()
    device = next(model.parameters()).device

    # Load d_ablation
    if args.mode != 'none':
        assert os.path.exists(args.directions_pt), f"Not found: {args.directions_pt}"
        dirs = torch.load(args.directions_pt, map_location='cpu')
        d_hall_all = dirs['d_ablation']
        d_hall = {l: d_hall_all[l].float() for l in args.proj_layers if l in d_hall_all}
        missing = [l for l in args.proj_layers if l not in d_hall_all]
        if missing:
            raise KeyError(f"proj_layers {missing} not in directions.pt. "
                           f"Available: {sorted(d_hall_all.keys())}")
    else:
        d_hall = {}

    projector = EPPProjector(
        model=model,
        d_hall_by_layer=d_hall,
        mode=args.mode,
        gamma=args.gamma,
        proj_layers=args.proj_layers if args.mode != 'none' else [],
        tau=args.tau,
        T_saliency=args.T_saliency,
    )

    print(f"mode={args.mode}, proj_layers={args.proj_layers}, gamma={args.gamma}")
    print(f"task={args.task}")

    if args.task == 'pope':
        run_pope(args, model, tokenizer, image_processor, projector, device)
    elif args.task == 'chair':
        run_chair(args, model, tokenizer, image_processor, projector, device)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--task', choices=['pope', 'chair'], default='pope')
    # Mode
    parser.add_argument('--mode', choices=['none', 'ablation', 'epp_b1', 'epp_b3'], default='epp_b1')
    # Model
    parser.add_argument('--model_path', default='liuhaotian/llava-v1.5-7b')
    # Data
    parser.add_argument('--pope_file',     default='/home/sjkim/ILVAD/data/pope/coco/coco_pope_popular.json')
    parser.add_argument('--image_folder',  default='/home/sjkim/datasets/coco/val2014')
    parser.add_argument('--ann_file',      default='/home/sjkim/datasets/coco/annotations/instances_val2014.json')
    # EPP params
    parser.add_argument('--directions_pt', default='../diag_output/directions.pt')
    parser.add_argument('--proj_layers',   type=int, nargs='+', default=[16])
    parser.add_argument('--gamma',         type=float, default=1.0)
    parser.add_argument('--tau',           type=float, default=5.0)
    parser.add_argument('--T_saliency',    type=int, default=4)
    # Generation
    parser.add_argument('--max_new_tokens', type=int, default=16,
                        help='POPE: 16 sufficient; CHAIR: use 64 or 512')
    parser.add_argument('--seed', type=int, default=42)
    # Subset
    parser.add_argument('--n_questions', type=int, default=500,
                        help='POPE questions to eval (0=all 3000)')
    parser.add_argument('--n_images',    type=int, default=100,
                        help='CHAIR images to eval (0=all)')
    # Output
    parser.add_argument('--answers_file', default='./mini_results/answers.jsonl')

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
