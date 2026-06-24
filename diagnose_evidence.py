#!/usr/bin/env python3
"""
Test 0: e_img validity check for EPP (Evidence-Protected Projection).

Metrics:
  1. Distinctiveness  — within-image vs cross-image cos similarity
  2. Evidence align   — AUROC: faithful tokens have higher <h, e_img> than hallucinated
  3. (B1/B2/B3) x layer AUROC sweep table
  + Massive activation check on e_img_B2

Usage:
  cd /home/sjkim/VCD/experiments
  python diagnose_evidence.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --image_dir /home/sjkim/datasets/coco/val2014 \
    --ann_file /home/sjkim/datasets/coco/annotations/instances_val2014.json \
    --n_images 200 --layers 8 10 12 14 16 --T 4 --tau 5.0
"""

import argparse
import json
import os
import re
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from vcd_utils.evidence_extractor import (
    compute_layer_binary_attn,
    compute_saliency_from_binary,
    compute_e_img_b1,
    compute_e_img_b2,
    compute_e_img_b3,
)

# ── COCO synonyms ─────────────────────────────────────────────────────────────
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
WORD_TO_CAT = {}
for _cat, _words in COCO_SYNONYMS.items():
    for _w in _words:
        WORD_TO_CAT[_w] = _cat
SORTED_WORDS = sorted(WORD_TO_CAT.keys(), key=len, reverse=True)


def extract_objects(caption: str) -> list:
    cap = caption.lower()
    found, covered = [], []
    for w in SORTED_WORDS:
        for m in re.finditer(r'\b' + re.escape(w) + r'\b', cap):
            s, e = m.start(), m.end()
            if any(cs <= s < ce or cs < e <= ce for cs, ce in covered):
                continue
            covered.append((s, e))
            found.append((w, WORD_TO_CAT[w], s))
    return found


def load_coco_gt(ann_file: str) -> dict:
    data = json.load(open(ann_file))
    cat_map = {c['id']: c['name'] for c in data['categories']}
    gt = defaultdict(set)
    for ann in data['annotations']:
        gt[ann['image_id']].add(cat_map[ann['category_id']])
    return gt


def build_prompt(tokenizer):
    conv = conv_templates['llava_v1'].copy()
    qs = DEFAULT_IMAGE_TOKEN + '\nDescribe the image in detail.'
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv.sep2


def get_img_pos(input_ids: torch.Tensor) -> int:
    pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    assert len(pos) == 1
    return pos[0].item()


def find_token_pos(obj_word: str, gen_ids: list, tokenizer) -> int | None:
    for prefix in [' ' + obj_word, obj_word]:
        toks = tokenizer.encode(prefix, add_special_tokens=False)
        if not toks:
            continue
        n = len(toks)
        for i in range(len(gen_ids) - n + 1):
            if gen_ids[i:i+n] == toks:
                return i
    return None


def auroc_faithful(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC where label=1 is hallucinated; faithful should have higher score."""
    if len(np.unique(labels)) < 2:
        return float('nan')
    # positive = faithful (1-labels), score = e_img alignment → faithful should be higher
    try:
        return roc_auc_score(1 - labels, scores)
    except Exception:
        return float('nan')


# ── Teacher-forced forward for evidence extraction ────────────────────────────
def tf_forward_evidence(
    model, full_input_ids, image_tensor, device,
    target_layers, T, tau, prompt_len
):
    """
    Full teacher-forced forward on prompt + generated tokens.

    Returns:
        img_hidden   {l: (n_img, 4096)} CPU  — image token hidden states
        gen_hidden   {l: (gen_len, 4096)} CPU — hidden states at generated positions
        saliency     (n_img,) CPU             — ILVAD saliency using all 32 layers
        last_normal  {l: (4096,)} CPU         — last prompt pos, normal forward
        last_ablated {l: (4096,)} CPU         — last prompt pos, image-masked forward
        img_start, img_end, n_img, gen_start  (ints)
    """
    seq_len_orig = full_input_ids.shape[1]
    gen_len = seq_len_orig - prompt_len

    attn_mask = torch.ones_like(full_input_ids)
    img_pos_in_prompt = get_img_pos(full_input_ids)

    with torch.inference_mode():
        _, new_attn, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
            full_input_ids.clone(), attn_mask.clone(), None, None,
            image_tensor.unsqueeze(0).half().to(device),
        )
        if inputs_embeds is None:
            raise RuntimeError("inputs_embeds is None")

        n_img = inputs_embeds.shape[1] - seq_len_orig + 1
        img_start = img_pos_in_prompt
        img_end = img_start + n_img
        gen_start = prompt_len - 1 + n_img    # first generated token in expanded seq
        last_prompt_pos = gen_start - 1

        T_actual = min(T, max(1, gen_len))
        gen_positions = list(range(gen_start, gen_start + T_actual))

        # ── Normal forward ────────────────────────────────────────────────────
        out_normal = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attn,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

        # Compute saliency layer-by-layer, delete each attn tensor immediately
        layer_binary_list = []
        for l_idx in range(len(out_normal.attentions)):
            attn_l = out_normal.attentions[l_idx][0].float()  # (n_heads, seq, seq)
            binary = compute_layer_binary_attn(
                attn_l, gen_positions, img_start, img_end, tau
            )
            layer_binary_list.append(binary)
            del attn_l
        saliency = compute_saliency_from_binary(layer_binary_list, n_img)

        # Extract image token and gen token hidden states (target layers only)
        img_hidden, gen_hidden, last_normal = {}, {}, {}
        for l in target_layers:
            hs = out_normal.hidden_states[l + 1][0].float()  # (expanded_seq, 4096)
            img_hidden[l] = hs[img_start:img_end].cpu()
            end_gen = min(gen_start + gen_len, hs.shape[0])
            gen_hidden[l] = hs[gen_start:end_gen].cpu()
            last_normal[l] = hs[last_prompt_pos].cpu()

        del out_normal
        torch.cuda.empty_cache()

        # ── Ablated forward (image tokens masked) for B3 ─────────────────────
        abl_mask = new_attn.clone()
        abl_mask[:, img_start:img_end] = 0
        out_ablated = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=abl_mask,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
        last_ablated = {}
        for l in target_layers:
            hs_a = out_ablated.hidden_states[l + 1][0].float()
            last_ablated[l] = hs_a[last_prompt_pos].cpu()

        del out_ablated
        torch.cuda.empty_cache()

    return (
        img_hidden, gen_hidden, saliency,
        last_normal, last_ablated,
        img_start, img_end, n_img, gen_start
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    target_layers = args.layers

    print("모델 로딩...")
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path)
    )
    model.eval()
    device = next(model.parameters()).device

    print("COCO annotation 로딩...")
    coco_gt = load_coco_gt(args.ann_file)

    img_dir = Path(args.image_dir)
    all_imgs = sorted(img_dir.glob("*.jpg"))[:args.n_images]
    print(f"이미지: {len(all_imgs)}")

    prompt_str, _ = build_prompt(tokenizer)
    prompt_input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0)
    prompt_len = prompt_input_ids.shape[1]

    n_img_global = None

    # Per-image: e_img vectors
    image_e_b1 = {}   # {img_idx: {l: (4096,)}}
    image_e_b2 = {}
    image_e_b3 = {}

    # Per object token: (label, img_idx, {l: h})
    sample_records = []

    for i, img_path in enumerate(tqdm(all_imgs, desc="이미지 처리")):
        img_id_match = re.search(r'(\d+)\.jpg$', img_path.name)
        if not img_id_match:
            continue
        img_id = int(img_id_match.group(1))
        gt_cats = coco_gt.get(img_id, set())

        image = Image.open(img_path).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        pids = prompt_input_ids.clone().to(device)
        with torch.inference_mode():
            out = model.generate(
                pids,
                images=image_tensor.unsqueeze(0).half().to(device),
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
        gen_ids = out[0, prompt_len:].tolist()
        caption = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        objects = extract_objects(caption)
        if not objects:
            continue

        labeled_objs = []
        for (word, cat, _) in objects:
            tok_idx = find_token_pos(word, gen_ids, tokenizer)
            if tok_idx is None:
                continue
            label = 0 if cat in gt_cats else 1
            labeled_objs.append((word, cat, tok_idx, label))

        if not labeled_objs:
            continue

        # Teacher-forced on prompt + full caption
        full_ids = out[:, :prompt_len + len(gen_ids)].cpu()
        try:
            (img_hidden, gen_hidden, saliency,
             last_normal, last_ablated,
             img_start, img_end, n_img, gen_start) = tf_forward_evidence(
                model, full_ids, image_tensor, device,
                target_layers, args.T, args.tau, prompt_len
            )
        except Exception as e:
            print(f"[skip] {img_path.name}: {e}")
            continue

        if n_img_global is None:
            n_img_global = n_img
            print(f"image token 수: {n_img} (expect 576)")

        # Compute e_img for all three variants
        e_b1 = compute_e_img_b1(img_hidden, saliency, target_layers)
        e_b2 = compute_e_img_b2(img_hidden, target_layers)
        e_b3 = compute_e_img_b3(last_normal, last_ablated, target_layers)

        image_e_b1[i] = e_b1
        image_e_b2[i] = e_b2
        image_e_b3[i] = e_b3

        # Record object token hidden states
        for (word, cat, tok_idx, label) in labeled_objs:
            if tok_idx >= gen_hidden[target_layers[0]].shape[0]:
                continue
            h_by_layer = {l: gen_hidden[l][tok_idx].clone() for l in target_layers}
            sample_records.append({'label': label, 'img_idx': i, 'h': h_by_layer})

    print(f"\n총 object token: {len(sample_records)}")
    n_hall  = sum(s['label'] == 1 for s in sample_records)
    n_faith = sum(s['label'] == 0 for s in sample_records)
    print(f"  hallucinated={n_hall}, faithful={n_faith}")
    n_images_processed = len(image_e_b1)
    print(f"  이미지={n_images_processed}")

    if n_hall < 10 or n_faith < 10:
        print("WARNING: 샘플 수 부족. --n_images를 늘리세요.")

    # ── Metric 1: Distinctiveness ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("Metric 1: Distinctiveness (A-A vs A-B cosine similarity)")
    print("="*70)
    img_indices = list(image_e_b1.keys())
    for l in target_layers:
        aa_scores, ab_scores = [], []
        for s in sample_records:
            idx = s['img_idx']
            if idx not in image_e_b1:
                continue
            h = F.normalize(s['h'][l].float(), dim=-1)
            e_same = image_e_b1[idx][l].float()   # already unit-norm
            cos_same = float(h @ e_same)
            aa_scores.append(cos_same)

            # Cross-image: random other image
            other_indices = [j for j in img_indices if j != idx]
            if not other_indices:
                continue
            j = random.choice(other_indices)
            e_other = image_e_b1[j][l].float()
            cos_other = float(h @ e_other)
            ab_scores.append(cos_other)

        if aa_scores and ab_scores:
            print(f"  L{l:2d}: A-A={np.mean(aa_scores):.4f}  A-B={np.mean(ab_scores):.4f}  "
                  f"diff={np.mean(aa_scores)-np.mean(ab_scores):+.4f}")

    # ── Metric 2/3: Evidence alignment AUROC ─────────────────────────────────
    print("\n" + "="*70)
    print("Metric 2/3: Evidence alignment AUROC  (faithful=positive class)")
    print("Hypothesis: faithful tokens have higher <h, e_img> than hallucinated")
    print("="*70)
    labels_all = np.array([s['label'] for s in sample_records])

    variants = [
        ('B1_ILVAD', image_e_b1),
        ('B2_Mean',  image_e_b2),
        ('B3_Diff',  image_e_b3),
    ]

    # Header
    header = f"{'Method':<12}" + "".join(f"  L{l:2d} " for l in target_layers)
    print(header)
    print("-" * len(header))

    best_overall = {'auroc': 0, 'method': '', 'layer': -1}
    for vname, e_dict in variants:
        row = f"{vname:<12}"
        for l in target_layers:
            scores = np.array([
                float(F.normalize(s['h'][l].float(), dim=-1) @ e_dict[s['img_idx']][l].float())
                if s['img_idx'] in e_dict else 0.0
                for s in sample_records
            ])
            auc = auroc_faithful(scores, labels_all)
            row += f"  {auc:.3f}"
            if not np.isnan(auc) and auc > best_overall['auroc']:
                best_overall = {'auroc': auc, 'method': vname, 'layer': l}
        print(row)

    # ── Massive activation check ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("Massive activation check on e_img_B2 (top-5 dimensions by abs value)")
    print("="*70)
    if image_e_b2:
        first_idx = next(iter(image_e_b2))
        for l in target_layers:
            e2_stack = torch.stack([image_e_b2[idx][l] for idx in image_e_b2])  # (N, 4096)
            # Original (pre-unit-norm) img_hidden is no longer available here,
            # but we can check if e_img_B2 direction is dominated by few dims
            abs_mean = e2_stack.abs().mean(0)  # (4096,)
            top5_vals, top5_idx = torch.topk(abs_mean, 5)
            frac = float(top5_vals.sum() / (abs_mean.sum() + 1e-10))
            print(f"  L{l:2d}: top5_dims={top5_idx.tolist()}  "
                  f"top5_abs={[f'{v:.3f}' for v in top5_vals.tolist()]}  "
                  f"top5_fraction={frac:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("판단")
    print("="*70)
    auc_b1_by_layer = {}
    for l in target_layers:
        scores = np.array([
            float(F.normalize(s['h'][l].float(), dim=-1) @ image_e_b1[s['img_idx']][l].float())
            if s['img_idx'] in image_e_b1 else 0.0
            for s in sample_records
        ])
        auc_b1_by_layer[l] = auroc_faithful(scores, labels_all)

    best_b1_layer = max(auc_b1_by_layer, key=lambda k: auc_b1_by_layer[k] if not np.isnan(auc_b1_by_layer[k]) else 0)
    best_b1_auc = auc_b1_by_layer[best_b1_layer]

    print(f"  B1 최고 AUROC: {best_b1_auc:.4f} @ layer {best_b1_layer}")

    if best_b1_auc >= 0.60:
        print(f"  결론: e_img 쓸만함 (B1 @ L{best_b1_layer}, AUROC {best_b1_auc:.4f} ≥ 0.60) → EPP 진행")
    else:
        print(f"  결론: e_img 부족 (B1 max AUROC {best_b1_auc:.4f} < 0.60) → 재설계 필요")

    # ── Save results ──────────────────────────────────────────────────────────
    result_dict = {
        'n_images': n_images_processed,
        'n_samples': len(sample_records),
        'n_hall': int(n_hall),
        'n_faith': int(n_faith),
        'n_img_tokens': n_img_global,
        'auc_b1_by_layer': {str(l): float(v) for l, v in auc_b1_by_layer.items()},
        'best_b1': {'layer': best_b1_layer, 'auroc': float(best_b1_auc)},
    }
    out_path = os.path.join(args.output_dir, 'evidence_results.json')
    with open(out_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image_dir',   default='/home/sjkim/datasets/coco/val2014')
    parser.add_argument('--ann_file',    default='/home/sjkim/datasets/coco/annotations/instances_val2014.json')
    parser.add_argument('--n_images',    type=int, default=200)
    parser.add_argument('--layers',      type=int, nargs='+', default=[8, 10, 12, 14, 16])
    parser.add_argument('--T',           type=int, default=4, help='number of gen steps for saliency')
    parser.add_argument('--tau',         type=float, default=5.0, help='binarization threshold multiplier')
    parser.add_argument('--output_dir',  default='./diag_output')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
