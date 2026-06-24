#!/usr/bin/env python3
"""
확인 1 & 2: d_hall 정제(refinement) 검증 — EPP novelty 사전 검증.

핵심 질문:
  확인 1-a) |<d̂_hall, ê_img>| (alpha)가 충분히 큰가? ≈0 이면 정제 no-op.
  확인 1-b) cos(d̂_hall, d̂_clean) < 1 이 충분히 작은가?
  확인 1-c) d̂_clean의 환각 판별 AUROC가 d_ablation 대비 유지되는가?
  확인 2)   hallucinated 토큰의 <h, d̂_clean> > faithful 토큰? (AUROC, hall=positive)

수식:
  d_clean^(l) = d̂_hall^(l) − α · ê_img^(l)      α = ⟨d̂_hall, ê_img⟩
  d̂_clean^(l) = d_clean / ‖d_clean‖

Usage:
  cd /home/sjkim/VCD/experiments
  python diagnose_refinement.py \\
    --model_path liuhaotian/llava-v1.5-7b \\
    --image_dir /home/sjkim/datasets/coco/val2014 \\
    --ann_file /home/sjkim/datasets/coco/annotations/instances_val2014.json \\
    --directions_pt ./diag_output/directions.pt \\
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
    compute_e_img_b3,
)

# ── COCO synonym map (same as diagnose_evidence.py) ──────────────────────────
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


def extract_objects(caption):
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


def load_coco_gt(ann_file):
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
    return conv.get_prompt()


def get_img_pos(input_ids):
    pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    assert len(pos) == 1
    return pos[0].item()


def find_token_pos(obj_word, gen_ids, tokenizer):
    for prefix in [' ' + obj_word, obj_word]:
        toks = tokenizer.encode(prefix, add_special_tokens=False)
        if not toks:
            continue
        n = len(toks)
        for i in range(len(gen_ids) - n + 1):
            if gen_ids[i:i+n] == toks:
                return i
    return None


def tf_forward_evidence(model, full_input_ids, image_tensor, device,
                         target_layers, T, tau, prompt_len):
    """Reuse identical logic from diagnose_evidence.py."""
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
        gen_start = prompt_len - 1 + n_img
        last_prompt_pos = gen_start - 1

        T_actual = min(T, max(1, gen_len))
        gen_positions = list(range(gen_start, gen_start + T_actual))

        # Normal forward
        out_normal = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attn,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

        # Saliency: process all 32 layers, delete each attn tensor
        layer_binary_list = []
        for l_idx in range(len(out_normal.attentions)):
            attn_l = out_normal.attentions[l_idx][0].float()
            binary = compute_layer_binary_attn(attn_l, gen_positions, img_start, img_end, tau)
            layer_binary_list.append(binary)
            del attn_l
        saliency = compute_saliency_from_binary(layer_binary_list, n_img)

        img_hidden, gen_hidden, last_normal = {}, {}, {}
        for l in target_layers:
            hs = out_normal.hidden_states[l + 1][0].float()
            img_hidden[l] = hs[img_start:img_end].cpu()
            end_gen = min(gen_start + gen_len, hs.shape[0])
            gen_hidden[l] = hs[gen_start:end_gen].cpu()
            last_normal[l] = hs[last_prompt_pos].cpu()

        del out_normal
        torch.cuda.empty_cache()

        # Ablated forward for B3
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

    return (img_hidden, gen_hidden, saliency, last_normal, last_ablated,
            img_start, img_end, n_img, gen_start)


def compute_d_clean(d_hall, e_img, eps=1e-6):
    """
    d_clean = d̂_hall − ⟨d̂_hall, ê_img⟩ · ê_img
    Returns (d̂_clean, alpha, norm_clean).
    If norm_clean < eps, returns (d̂_hall, alpha, 0) as fallback.
    """
    alpha = float(torch.dot(d_hall.float(), e_img.float()))
    d_clean = d_hall.float() - alpha * e_img.float()
    norm_clean = float(d_clean.norm())
    if norm_clean < eps:
        return d_hall.float(), alpha, 0.0
    return d_clean / norm_clean, alpha, norm_clean


def auroc(scores, labels_hall):
    """AUROC where labels_hall=1 means hallucinated (positive class)."""
    if len(np.unique(labels_hall)) < 2:
        return float('nan')
    try:
        return roc_auc_score(labels_hall, scores)
    except Exception:
        return float('nan')


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    target_layers = args.layers

    print("d_ablation 로딩...")
    directions = torch.load(args.directions_pt, map_location='cpu')
    d_hall = {}
    for l in target_layers:
        d_hall[l] = directions['d_ablation'][l].float()
        assert abs(float(d_hall[l].norm()) - 1.0) < 1e-4, f"L{l} d_ablation not unit-norm"
    print(f"  d_ablation: layers {sorted(d_hall.keys())}")

    print("모델 로딩...")
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path)
    )
    model.eval()
    device = next(model.parameters()).device

    coco_gt = load_coco_gt(args.ann_file)
    img_dir = Path(args.image_dir)
    all_imgs = sorted(img_dir.glob("*.jpg"))[:args.n_images]
    print(f"이미지: {len(all_imgs)}")

    prompt_str = build_prompt(tokenizer)
    prompt_input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0)
    prompt_len = prompt_input_ids.shape[1]

    # Accumulators
    # per-image per-layer: alpha, cos, d̂_clean
    alpha_b1 = {l: [] for l in target_layers}
    alpha_b3 = {l: [] for l in target_layers}
    cos_b1   = {l: [] for l in target_layers}
    cos_b3   = {l: [] for l in target_layers}
    # per-object: label + scores
    records  = []   # {label, img_idx, h_by_layer}
    # per-image: d̂_clean_b1, d̂_clean_b3
    d_clean_b1_per_img = {}   # {img_idx: {l: (4096,)}}
    d_clean_b3_per_img = {}

    n_img_global = None

    for i, img_path in enumerate(tqdm(all_imgs, desc="이미지")):
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
            label = 1 if (cat not in gt_cats) else 0   # 1=hallucinated
            labeled_objs.append((word, cat, tok_idx, label))

        if not labeled_objs:
            continue

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

        e_b1 = compute_e_img_b1(img_hidden, saliency, target_layers)
        e_b3 = compute_e_img_b3(last_normal, last_ablated, target_layers)

        d_clean_b1_per_img[i] = {}
        d_clean_b3_per_img[i] = {}

        for l in target_layers:
            dc_b1, a_b1, nc_b1 = compute_d_clean(d_hall[l], e_b1[l])
            dc_b3, a_b3, nc_b3 = compute_d_clean(d_hall[l], e_b3[l])

            alpha_b1[l].append(abs(a_b1))
            alpha_b3[l].append(abs(a_b3))

            # cos(d̂_hall, d̂_clean) = analytically sqrt(1 - alpha^2), but compute directly
            cos_b1_val = float(torch.dot(d_hall[l], dc_b1)) if nc_b1 > 1e-6 else 1.0
            cos_b3_val = float(torch.dot(d_hall[l], dc_b3)) if nc_b3 > 1e-6 else 1.0
            cos_b1[l].append(cos_b1_val)
            cos_b3[l].append(cos_b3_val)

            d_clean_b1_per_img[i][l] = dc_b1
            d_clean_b3_per_img[i][l] = dc_b3

        for (word, cat, tok_idx, label) in labeled_objs:
            if tok_idx >= gen_hidden[target_layers[0]].shape[0]:
                continue
            h_by_layer = {l: gen_hidden[l][tok_idx].clone() for l in target_layers}
            records.append({'label': label, 'img_idx': i, 'h': h_by_layer})

    print(f"\n총 object token: {len(records)}")
    n_hall  = sum(r['label'] == 1 for r in records)
    n_faith = sum(r['label'] == 0 for r in records)
    print(f"  hallucinated={n_hall}, faithful={n_faith}")

    if n_hall < 10 or n_faith < 10:
        print("WARNING: 샘플 수 부족")
        return

    labels_arr = np.array([r['label'] for r in records])

    # ── 확인 1-a: |alpha| 통계 ────────────────────────────────────────────────
    print("\n" + "="*72)
    print("확인 1-a: |<d̂_hall, ê_img>| — 정제 성분 크기 (0=no-op, 0.2~0.6=실질적)")
    print("="*72)
    fmt = f"  {'Layer':<6}  {'B1 mean±std':>16}  {'B3 mean±std':>16}"
    print(fmt)
    print("  " + "-"*52)
    alpha_stats = {}
    for l in target_layers:
        a1 = np.array(alpha_b1[l])
        a3 = np.array(alpha_b3[l])
        print(f"  L{l:<5d}  {np.mean(a1):.4f} ± {np.std(a1):.4f}  "
              f"  {np.mean(a3):.4f} ± {np.std(a3):.4f}")
        alpha_stats[l] = {'b1_mean': float(np.mean(a1)), 'b1_std': float(np.std(a1)),
                          'b3_mean': float(np.mean(a3)), 'b3_std': float(np.std(a3))}

    # ── 확인 1-b: cos(d̂_hall, d̂_clean) ────────────────────────────────────────
    print("\n" + "="*72)
    print("확인 1-b: cos(d̂_hall, d̂_clean) — 방향 변화 (1.0=no-op, <0.95=실질적)")
    print("="*72)
    print(fmt)
    print("  " + "-"*52)
    cos_stats = {}
    for l in target_layers:
        c1 = np.array(cos_b1[l])
        c3 = np.array(cos_b3[l])
        print(f"  L{l:<5d}  {np.mean(c1):.4f} ± {np.std(c1):.4f}  "
              f"  {np.mean(c3):.4f} ± {np.std(c3):.4f}")
        cos_stats[l] = {'b1_mean': float(np.mean(c1)), 'b1_std': float(np.std(c1)),
                        'b3_mean': float(np.mean(c3)), 'b3_std': float(np.std(c3))}

    # ── 확인 1-c: AUROC 비교 (hall=positive) ─────────────────────────────────
    print("\n" + "="*72)
    print("확인 1-c: AUROC 비교 [hallucinated=positive] (d_clean ≥ d_ablation 이면 ✓)")
    print("="*72)
    header = f"  {'Layer':<6}  {'d_ablation':>12}  {'d_clean_B1':>12}  {'d_clean_B3':>12}"
    print(header)
    print("  " + "-"*52)
    auroc_stats = {}
    for l in target_layers:
        # d_ablation scores (global direction, same for all images)
        scores_hall = np.array([
            float(F.normalize(r['h'][l].float(), dim=-1) @ d_hall[l])
            for r in records
        ])
        auc_hall = auroc(scores_hall, labels_arr)

        # d_clean_b1 scores (per-image direction)
        scores_b1 = np.array([
            float(F.normalize(r['h'][l].float(), dim=-1) @
                  d_clean_b1_per_img[r['img_idx']][l])
            if r['img_idx'] in d_clean_b1_per_img else 0.0
            for r in records
        ])
        auc_b1 = auroc(scores_b1, labels_arr)

        scores_b3 = np.array([
            float(F.normalize(r['h'][l].float(), dim=-1) @
                  d_clean_b3_per_img[r['img_idx']][l])
            if r['img_idx'] in d_clean_b3_per_img else 0.0
            for r in records
        ])
        auc_b3 = auroc(scores_b3, labels_arr)

        delta1 = auc_b1 - auc_hall
        delta3 = auc_b3 - auc_hall
        tag1 = "✓" if delta1 >= -0.02 else "✗"
        tag3 = "✓" if delta3 >= -0.02 else "✗"
        print(f"  L{l:<5d}  {auc_hall:>12.4f}  {auc_b1:>10.4f}{tag1}  {auc_b3:>10.4f}{tag3}  "
              f"(Δb1={delta1:+.4f}, Δb3={delta3:+.4f})")
        auroc_stats[l] = {
            'd_ablation': float(auc_hall),
            'd_clean_b1': float(auc_b1), 'delta_b1': float(delta1),
            'd_clean_b3': float(auc_b3), 'delta_b3': float(delta3),
        }

    # ── 확인 2: <h, d̂_clean> faithful vs hallucinated 분포 ───────────────────
    print("\n" + "="*72)
    print("확인 2: <h, d̂_clean_B1> 분포 — hall mean > faith mean 이어야 함 (sanity)")
    print("="*72)
    hdr2 = f"  {'Layer':<6}  {'faith mean':>12}  {'hall mean':>12}  {'diff':>10}  {'AUROC':>8}"
    print(hdr2)
    print("  " + "-"*52)
    sanity_stats = {}
    for l in target_layers:
        scores_b1 = np.array([
            float(F.normalize(r['h'][l].float(), dim=-1) @
                  d_clean_b1_per_img[r['img_idx']][l])
            if r['img_idx'] in d_clean_b1_per_img else 0.0
            for r in records
        ])
        faith_scores = scores_b1[labels_arr == 0]
        hall_scores  = scores_b1[labels_arr == 1]
        faith_mean   = float(np.mean(faith_scores)) if len(faith_scores) > 0 else float('nan')
        hall_mean    = float(np.mean(hall_scores))  if len(hall_scores) > 0 else float('nan')
        diff         = hall_mean - faith_mean
        auc          = auroc(scores_b1, labels_arr)
        tag = "✓" if diff > 0 else "✗"
        print(f"  L{l:<5d}  {faith_mean:>12.5f}  {hall_mean:>12.5f}  {diff:>+10.5f}{tag}  {auc:>8.4f}")
        sanity_stats[l] = {
            'faith_mean': faith_mean, 'hall_mean': hall_mean,
            'diff': diff, 'auroc': float(auc),
        }

    # ── 최종 판단 ─────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("최종 판단")
    print("="*72)

    best_l = max(target_layers, key=lambda l: alpha_stats[l]['b1_mean'])
    mean_alpha_b1 = np.mean([alpha_stats[l]['b1_mean'] for l in target_layers])
    mean_cos_b1   = np.mean([cos_stats[l]['b1_mean']   for l in target_layers])
    best_auroc_b1 = max(auroc_stats[l]['d_clean_b1'] for l in target_layers)
    best_auroc_orig = max(auroc_stats[l]['d_ablation'] for l in target_layers)
    sanity_ok = all(sanity_stats[l]['diff'] > 0 for l in target_layers)

    print(f"  평균 |alpha_B1|        : {mean_alpha_b1:.4f}  (기준 ≥ 0.10 → 정제 비-trivial)")
    print(f"  평균 cos(d̂_hall,d̂_clean): {mean_cos_b1:.4f}  (기준 ≤ 0.995 → 방향 변화 존재)")
    print(f"  d_clean_B1 최고 AUROC  : {best_auroc_b1:.4f}  vs d_ablation {best_auroc_orig:.4f}")
    print(f"  확인 2 sanity (hall>faith): {'모두 통과' if sanity_ok else '일부 실패'}")

    trivial = mean_alpha_b1 < 0.05
    no_change = mean_cos_b1 > 0.999
    auroc_dropped = (best_auroc_orig - best_auroc_b1) > 0.05

    if trivial or no_change:
        verdict = "정제 NO-OP → EPP 중단. 설계 재고."
    elif auroc_dropped:
        verdict = "정제가 판별력을 심각히 훼손 → 설계 재고 필요."
    else:
        verdict = "정제 실질적 + 판별력 유지 → projection 본체 구현으로 진행."

    print(f"\n  판단: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        'n_images': len(d_clean_b1_per_img),
        'n_samples': len(records),
        'n_hall': int(n_hall), 'n_faith': int(n_faith),
        'alpha_stats': {str(l): alpha_stats[l] for l in target_layers},
        'cos_stats':   {str(l): cos_stats[l]   for l in target_layers},
        'auroc_stats': {str(l): auroc_stats[l] for l in target_layers},
        'sanity_stats':{str(l): sanity_stats[l] for l in target_layers},
        'verdict': verdict,
    }
    out_path = os.path.join(args.output_dir, 'refinement_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',    default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image_dir',     default='/home/sjkim/datasets/coco/val2014')
    parser.add_argument('--ann_file',
                        default='/home/sjkim/datasets/coco/annotations/instances_val2014.json')
    parser.add_argument('--directions_pt', default='./diag_output/directions.pt')
    parser.add_argument('--n_images', type=int, default=200)
    parser.add_argument('--layers',   type=int, nargs='+', default=[8, 10, 12, 14, 16])
    parser.add_argument('--T',        type=int, default=4)
    parser.add_argument('--tau',      type=float, default=5.0)
    parser.add_argument('--output_dir', default='./diag_output')
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
