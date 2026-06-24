#!/usr/bin/env python3
"""
Hallucination direction diagnostic (Step 0 → Step 4 of the design spec).

측정 대상:
  d_true    : hallucinated vs faithful object token hidden states의 평균 차이 (reference)
  d_ablation: image attention 차단 전후 hidden state 변화 ("image-ignoring" 방향)
  d_attn    : low-attention image tokens - high-attention image tokens hidden 차이

출력: 레이어별 cosine alignment + projection-separation AUROC 표 및 plot

실행 예시:
  cd /home/sjkim/VCD/experiments
  python diagnose_hallucination_direction.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --image_dir /home/sjkim/datasets/coco/val2014 \
    --ann_file /home/sjkim/datasets/coco/annotations/instances_val2014.json \
    --n_images 200 \
    --layers 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 \
    --output_dir ./diag_output
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise SystemExit("scikit-learn 필요: pip install scikit-learn")

sys.path.insert(0, str(Path(__file__).parent.parent))  # VCD root
sys.path.insert(0, str(Path(__file__).parent))          # experiments/

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

# ── COCO category synonym map ────────────────────────────────────────────────
# 각 category → 인식할 단어 목록 (소문자). 길이 긴 것부터 매칭.
COCO_SYNONYMS = {
    'person':        ['person','man','woman','people','boy','girl','child','kid','human','men','women'],
    'bicycle':       ['bicycle','bike','bicycles','bikes'],
    'car':           ['car','cars','automobile','automobiles','vehicle','vehicles'],
    'motorcycle':    ['motorcycle','motorbike','motorcycles','motorbikes'],
    'airplane':      ['airplane','plane','aircraft','jet','aeroplane','airplanes','planes'],
    'bus':           ['bus','buses','coach'],
    'train':         ['train','trains'],
    'truck':         ['truck','trucks','lorry'],
    'boat':          ['boat','boats','ship','ships','vessel'],
    'traffic light': ['traffic light','traffic lights','stoplight'],
    'fire hydrant':  ['fire hydrant','hydrant'],
    'stop sign':     ['stop sign','stop signs'],
    'parking meter': ['parking meter'],
    'bench':         ['bench','benches'],
    'bird':          ['bird','birds'],
    'cat':           ['cat','cats','kitten','kittens'],
    'dog':           ['dog','dogs','puppy'],
    'horse':         ['horse','horses'],
    'sheep':         ['sheep'],
    'cow':           ['cow','cows','cattle'],
    'elephant':      ['elephant','elephants'],
    'bear':          ['bear','bears'],
    'zebra':         ['zebra','zebras'],
    'giraffe':       ['giraffe','giraffes'],
    'backpack':      ['backpack','backpacks','rucksack'],
    'umbrella':      ['umbrella','umbrellas'],
    'handbag':       ['handbag','purse','bag'],
    'tie':           ['tie','necktie'],
    'suitcase':      ['suitcase','suitcases','luggage'],
    'frisbee':       ['frisbee'],
    'skis':          ['skis','ski'],
    'snowboard':     ['snowboard','snowboards'],
    'sports ball':   ['sports ball','ball'],
    'kite':          ['kite','kites'],
    'baseball bat':  ['baseball bat'],
    'baseball glove':['baseball glove','glove'],
    'skateboard':    ['skateboard','skateboards'],
    'surfboard':     ['surfboard','surfboards'],
    'tennis racket': ['tennis racket','racket','racquet'],
    'bottle':        ['bottle','bottles'],
    'wine glass':    ['wine glass','wineglass'],
    'cup':           ['cup','cups','mug','mugs'],
    'fork':          ['fork','forks'],
    'knife':         ['knife','knives'],
    'spoon':         ['spoon','spoons'],
    'bowl':          ['bowl','bowls'],
    'banana':        ['banana','bananas'],
    'apple':         ['apple','apples'],
    'sandwich':      ['sandwich','sandwiches'],
    'orange':        ['orange','oranges'],
    'broccoli':      ['broccoli'],
    'carrot':        ['carrot','carrots'],
    'hot dog':       ['hot dog','hotdog'],
    'pizza':         ['pizza'],
    'donut':         ['donut','donuts','doughnut','doughnuts'],
    'cake':          ['cake','cakes'],
    'chair':         ['chair','chairs'],
    'couch':         ['couch','sofa','couches','sofas'],
    'potted plant':  ['potted plant','plant','houseplant'],
    'bed':           ['bed','beds'],
    'dining table':  ['dining table','table','tables'],
    'toilet':        ['toilet','toilets'],
    'tv':            ['tv','television','monitor','screen'],
    'laptop':        ['laptop','laptops','notebook'],
    'mouse':         ['mouse'],
    'remote':        ['remote','remote control'],
    'keyboard':      ['keyboard'],
    'cell phone':    ['cell phone','phone','smartphone','mobile'],
    'microwave':     ['microwave'],
    'oven':          ['oven','ovens'],
    'toaster':       ['toaster'],
    'sink':          ['sink','sinks'],
    'refrigerator':  ['refrigerator','fridge'],
    'book':          ['book','books'],
    'clock':         ['clock','clocks'],
    'vase':          ['vase','vases'],
    'scissors':      ['scissors'],
    'teddy bear':    ['teddy bear','teddy','stuffed animal'],
    'hair drier':    ['hair dryer','hair drier','dryer'],
    'toothbrush':    ['toothbrush','toothbrushes'],
}

# 역방향 맵: word → category
WORD_TO_CAT = {}
for cat, words in COCO_SYNONYMS.items():
    for w in words:
        WORD_TO_CAT[w] = cat
# 긴 phrase를 먼저 매칭하기 위해 단어 길이순 정렬
SORTED_WORDS = sorted(WORD_TO_CAT.keys(), key=len, reverse=True)


def extract_objects(caption: str) -> list:
    """caption에서 (matched_word, category) 목록 반환. 겹치는 span 제외."""
    cap = caption.lower()
    found = []
    covered = []
    for w in SORTED_WORDS:
        for m in re.finditer(r'\b' + re.escape(w) + r'\b', cap):
            s, e = m.start(), m.end()
            if any(cs <= s < ce or cs < e <= ce for cs, ce in covered):
                continue
            covered.append((s, e))
            found.append((w, WORD_TO_CAT[w], s))  # (word, category, char_start)
    return found


# ── COCO ground truth 로드 ───────────────────────────────────────────────────

def load_coco_gt(ann_file: str):
    """
    반환: {image_id: set of category_names}
    """
    data = json.load(open(ann_file))
    cat_map = {c['id']: c['name'] for c in data['categories']}
    gt = defaultdict(set)
    for ann in data['annotations']:
        gt[ann['image_id']].add(cat_map[ann['category_id']])
    return gt


# ── 프롬프트 빌더 ────────────────────────────────────────────────────────────

def build_prompt(tokenizer):
    """LLaVA-1.5 captioning prompt. Returns (prompt_str, conv_mode)."""
    conv = conv_templates['llava_v1'].copy()
    qs = DEFAULT_IMAGE_TOKEN + '\nDescribe the image in detail.'
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv.sep2  # stop_str


# ── 이미지 토큰 위치 확인 ────────────────────────────────────────────────────

def get_img_pos(input_ids: torch.Tensor) -> int:
    """input_ids (1, L)에서 IMAGE_TOKEN_INDEX의 위치 반환."""
    pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    assert len(pos) == 1, f"IMAGE_TOKEN_INDEX가 {len(pos)}개 발견됨"
    return pos[0].item()


# ── 생성된 token에서 object word의 첫 subword 위치 찾기 ──────────────────────

def find_token_pos(obj_word: str, gen_ids: list, tokenizer) -> int | None:
    """
    gen_ids 안에서 obj_word에 해당하는 토큰 시퀀스의 첫 번째 토큰 인덱스 반환.
    못 찾으면 None.
    선택: 첫 번째 subword 토큰만 사용.
    """
    candidates = []
    for prefix in [' ' + obj_word, obj_word]:
        toks = tokenizer.encode(prefix, add_special_tokens=False)
        if toks:
            candidates.append(toks)

    for toks in candidates:
        n = len(toks)
        for i in range(len(gen_ids) - n + 1):
            if gen_ids[i:i+n] == toks:
                return i  # 0-indexed in gen_ids
    return None


# ── teacher-forced forward ───────────────────────────────────────────────────

def tf_forward(model, full_input_ids, image_tensor, device, layer_ids):
    """
    Teacher-forced forward pass.
    반환:
        hs_normal  : {layer: tensor(seq_len, 4096) float32 CPU}
        hs_ablated : {layer: tensor(seq_len, 4096) float32 CPU}
        attn       : {layer: tensor(n_heads, seq_len, seq_len) float32 CPU}
        img_start  : int (expanded seq에서 image token 시작 위치)
        img_end    : int
        n_img      : int (image token 수, sanity: ~576)
    """
    attn_mask = torch.ones_like(full_input_ids)
    img_pos = get_img_pos(full_input_ids)

    with torch.inference_mode():
        # 이미지 토큰 확장 — prepare_inputs_labels_for_multimodal 직접 호출
        _, new_attn, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
            full_input_ids.clone(), attn_mask.clone(), None, None,
            image_tensor.unsqueeze(0).half().to(device)
        )
        if inputs_embeds is None:
            raise RuntimeError("inputs_embeds가 None — 이미지 확장 실패")

        orig_len = full_input_ids.shape[1]
        expanded_len = inputs_embeds.shape[1]
        n_img = expanded_len - orig_len + 1   # IMAGE_TOKEN_INDEX 1개를 n_img개로 교체
        img_start = img_pos
        img_end   = img_pos + n_img

        # ── 정상 forward ──
        out_normal = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attn,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

        # ── ablated forward (image 위치 attention 차단) ──
        abl_mask = new_attn.clone()
        abl_mask[:, img_start:img_end] = 0
        out_ablated = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=abl_mask,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )

    hs_normal, hs_ablated, attn = {}, {}, {}
    for l in layer_ids:
        # hidden_states[0] = embedding layer, [l+1] = l번째 transformer layer 출력
        hs_normal[l]  = out_normal.hidden_states[l + 1][0].float().cpu()
        hs_ablated[l] = out_ablated.hidden_states[l + 1][0].float().cpu()
        # attention: tuple of (1, n_heads, seq, seq) per layer
        attn[l] = out_normal.attentions[l][0].float().cpu()

    return hs_normal, hs_ablated, attn, img_start, img_end, n_img


# ── 방향 벡터 정규화 ─────────────────────────────────────────────────────────

def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-10)


# ── AUROC helper ─────────────────────────────────────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """labels: 1=hallucinated(양성), 0=faithful(음성)"""
    if len(np.unique(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    layer_ids = args.layers  # e.g. [2,4,...,30]

    # 모델 로드
    print("모델 로딩 중...")
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path)
    )
    model.eval()
    device = next(model.parameters()).device

    # COCO GT 로드
    print("COCO annotation 로딩...")
    coco_gt = load_coco_gt(args.ann_file)

    # 이미지 목록
    img_dir = Path(args.image_dir)
    all_imgs = sorted(img_dir.glob("*.jpg"))[:args.n_images]
    print(f"사용 이미지: {len(all_imgs)}")

    # 프롬프트 구성
    prompt_str, stop_str = build_prompt(tokenizer)
    prompt_input_ids = tokenizer_image_token(
        prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0)  # (1, prompt_len)
    prompt_len = prompt_input_ids.shape[1]
    img_pos_in_prompt = get_img_pos(prompt_input_ids)

    # ── 데이터 수집 루프 ──────────────────────────────────────────────────────
    # 각 sample = {
    #   'label': 0(faithful) or 1(hallucinated),
    #   'hs_normal':  {l: tensor(4096)},
    #   'hs_ablated': {l: tensor(4096)},
    #   'img_hs':     {l: tensor(n_img, 4096)},   # image token hidden states
    #   'img_attn':   {l: tensor(n_heads, n_img)}, # attention to image tokens (mean over gen tokens)
    # }
    samples = []
    n_img_global = None

    for img_path in tqdm(all_imgs, desc="이미지 처리"):
        # 이미지 ID (COCO 파일명에서 숫자 추출)
        img_id_match = re.search(r'(\d+)\.jpg$', img_path.name)
        if not img_id_match:
            continue
        img_id = int(img_id_match.group(1))
        gt_cats = coco_gt.get(img_id, set())

        # 이미지 전처리
        image = Image.open(img_path).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # 1. Greedy generation
        pids = prompt_input_ids.clone().cuda()
        with torch.inference_mode():
            out = model.generate(
                pids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
        gen_ids = out[0, prompt_len:].tolist()
        caption = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 2. object word 추출 및 라벨링
        objects = extract_objects(caption)
        if not objects:
            continue

        # 3. object token 위치 찾기 → labeled (word, category, gen_idx, label)
        labeled_objs = []
        for (word, cat, _) in objects:
            tok_idx = find_token_pos(word, gen_ids, tokenizer)
            if tok_idx is None:
                continue
            label = 0 if cat in gt_cats else 1  # 1=hallucinated
            labeled_objs.append((word, cat, tok_idx, label))

        if not labeled_objs:
            continue

        # 4. teacher-forced forward
        full_ids = out[:, :prompt_len + len(gen_ids)].cpu()  # (1, prompt_len+gen_len)
        try:
            hs_n, hs_a, attn_w, img_start, img_end, n_img = tf_forward(
                model, full_ids, image_tensor, device, layer_ids
            )
        except Exception as e:
            print(f"[skip] {img_path.name}: {e}")
            continue

        if n_img_global is None:
            n_img_global = n_img
            print(f"image token 수: {n_img} (sanity check, expect ~576)")

        # expanded sequence에서 gen token 시작 위치
        # original: prompt_len tokens (IMAGE_TOKEN_INDEX at img_pos_in_prompt)
        # expanded: prompt_len - 1 + n_img tokens
        expanded_gen_start = prompt_len - 1 + n_img  # = img_end + (prompt_len - img_pos_in_prompt - 1)

        for (word, cat, tok_idx, label) in labeled_objs:
            exp_pos = expanded_gen_start + tok_idx  # position in expanded sequence
            if exp_pos >= hs_n[layer_ids[0]].shape[0]:
                continue  # out of bounds guard

            sample = {
                'label': label,
                'word': word, 'cat': cat,
                'hs_normal':  {l: hs_n[l][exp_pos]           for l in layer_ids},
                'hs_ablated': {l: hs_a[l][exp_pos]           for l in layer_ids},
                'img_hs':     {l: hs_n[l][img_start:img_end] for l in layer_ids},
                'img_attn':   {},  # filled below
            }

            # attention from this gen token to image tokens, averaged over heads
            for l in layer_ids:
                # attn_w[l]: (n_heads, seq_len, seq_len)
                # from exp_pos to image tokens
                if exp_pos < attn_w[l].shape[1]:
                    att = attn_w[l][:, exp_pos, img_start:img_end]  # (n_heads, n_img)
                    sample['img_attn'][l] = att.mean(dim=0)          # (n_img,)
                else:
                    sample['img_attn'][l] = torch.zeros(n_img)

            samples.append(sample)

    print(f"\n총 object token 수: {len(samples)}")
    n_hall = sum(s['label'] == 1 for s in samples)
    n_faith = sum(s['label'] == 0 for s in samples)
    print(f"  hallucinated: {n_hall}, faithful: {n_faith}")

    if n_hall < 20:
        print("WARNING: hallucinated token이 너무 적습니다. --n_images를 늘리세요.")

    # ── fit / eval split ──────────────────────────────────────────────────────
    np.random.seed(42)
    n_fit = int(len(samples) * 0.7)
    idx = np.random.permutation(len(samples))
    fit_samples  = [samples[i] for i in idx[:n_fit]]
    eval_samples = [samples[i] for i in idx[n_fit:]]
    print(f"fit: {len(fit_samples)}, eval: {len(eval_samples)}")

    # ── 방향 벡터 계산 (fit set 기반) ────────────────────────────────────────
    d_true     = {}
    d_ablation = {}
    d_attn     = {}

    k_attn = max(1, (n_img_global or 576) // 10)  # top/bottom 10%

    for l in layer_ids:
        hall_h = torch.stack([s['hs_normal'][l] for s in fit_samples if s['label']==1])
        faith_h = torch.stack([s['hs_normal'][l] for s in fit_samples if s['label']==0])

        if hall_h.shape[0] == 0 or faith_h.shape[0] == 0:
            d_true[l] = torch.zeros(4096)
        else:
            d_true[l] = unit(hall_h.mean(0) - faith_h.mean(0))

        # d_ablation: per-sample δ = ablated - normal (image-ignoring 방향)
        deltas = torch.stack([
            s['hs_ablated'][l] - s['hs_normal'][l]
            for s in fit_samples
        ])
        d_ablation[l] = unit(deltas.mean(0))

        # d_attn: low-att image tokens - high-att image tokens
        # 각 sample에서 image token hidden states를 attention으로 랭킹
        high_list, low_list = [], []
        for s in fit_samples:
            att = s['img_attn'][l]               # (n_img,)
            img_h = s['img_hs'][l].float()       # (n_img, 4096)
            if att.shape[0] < 2 * k_attn:
                continue
            sorted_idx = att.argsort()
            low_list.append(img_h[sorted_idx[:k_attn]].mean(0))
            high_list.append(img_h[sorted_idx[-k_attn:]].mean(0))

        if low_list:
            raw_d_attn = torch.stack(low_list).mean(0) - torch.stack(high_list).mean(0)
            d_attn[l] = unit(raw_d_attn)
        else:
            d_attn[l] = torch.zeros(4096)

    # ── 평가 (eval set) ───────────────────────────────────────────────────────
    rows = []
    for l in layer_ids:
        # projection scores: h · d_candidate
        labels_arr = np.array([s['label'] for s in eval_samples])
        h_eval = torch.stack([s['hs_normal'][l] for s in eval_samples])  # (N, 4096)

        proj_true    = (h_eval @ d_true[l]).numpy()
        proj_ablation= (h_eval @ d_ablation[l]).numpy()
        proj_attn    = (h_eval @ d_attn[l]).numpy()
        proj_attn_neg= -proj_attn  # sign 불확실이므로 양쪽 시도

        auc_true    = auroc(proj_true,     labels_arr)
        auc_ablation= auroc(proj_ablation, labels_arr)
        auc_attn    = max(auroc(proj_attn, labels_arr),
                          auroc(proj_attn_neg, labels_arr))

        cos_ablation = float(d_ablation[l] @ d_true[l])
        cos_attn     = float(d_attn[l] @ d_true[l])

        # per-sample cosine for d_ablation
        per_deltas = torch.stack([s['hs_ablated'][l] - s['hs_normal'][l] for s in eval_samples])
        per_deltas_n = F.normalize(per_deltas, dim=-1)
        per_cos = (per_deltas_n @ d_true[l]).numpy()

        rows.append({
            'layer': l,
            'cos_ablation_true': cos_ablation,
            'cos_attn_true':     cos_attn,
            'auc_dtrue':         auc_true,
            'auc_ablation':      auc_ablation,
            'auc_attn':          auc_attn,
            'per_cos_mean':      float(per_cos.mean()),
            'per_cos_std':       float(per_cos.std()),
            'per_cos_pos_frac':  float((per_cos > 0).mean()),
        })

    # ── 출력 ─────────────────────────────────────────────────────────────────
    print("\n" + "="*95)
    print(f"{'Layer':>5} | {'cos(abl,true)':>13} | {'cos(attn,true)':>14} | "
          f"{'AUROC(true)':>11} | {'AUROC(abl)':>10} | {'AUROC(attn)':>11}")
    print("-"*95)
    for r in rows:
        print(f"{r['layer']:>5} | {r['cos_ablation_true']:>13.4f} | {r['cos_attn_true']:>14.4f} | "
              f"{r['auc_dtrue']:>11.4f} | {r['auc_ablation']:>10.4f} | {r['auc_attn']:>11.4f}")
    print("="*95)

    print("\n[d_ablation per-sample cosine 분포 (eval set)]")
    print(f"{'Layer':>5} | {'mean':>8} | {'std':>8} | {'pos%':>8}")
    print("-"*40)
    for r in rows:
        print(f"{r['layer']:>5} | {r['per_cos_mean']:>8.4f} | {r['per_cos_std']:>8.4f} | "
              f"{r['per_cos_pos_frac']*100:>7.1f}%")

    # 샘플 수 보고
    print(f"\n[샘플 정보]")
    print(f"  총 이미지: {len(all_imgs)}")
    print(f"  총 object tokens: {len(samples)} (hall={n_hall}, faith={n_faith})")
    print(f"  fit={len(fit_samples)}, eval={len(eval_samples)}")
    print(f"  image tokens per image: {n_img_global}")
    print(f"  레이어 인덱싱: 0-based (transformers 기준), 레이어 0 = 첫 번째 transformer block")

    # 판단
    best_l_abl = max(rows, key=lambda r: r['auc_ablation'] if not np.isnan(r['auc_ablation']) else 0)
    best_l_atn = max(rows, key=lambda r: r['auc_attn']     if not np.isnan(r['auc_attn'])     else 0)
    print(f"\n[판단]")
    print(f"  d_ablation 최고 AUROC: {best_l_abl['auc_ablation']:.4f} @ layer {best_l_abl['layer']}")
    print(f"  d_attn     최고 AUROC: {best_l_atn['auc_attn']:.4f}     @ layer {best_l_atn['layer']}")
    thresh = 0.65
    abl_ok = best_l_abl['auc_ablation'] >= thresh
    atn_ok = best_l_atn['auc_attn']     >= thresh
    if abl_ok and atn_ok:
        winner = 'd_ablation' if best_l_abl['auc_ablation'] >= best_l_atn['auc_attn'] else 'd_attn'
        print(f"  결론: {winner}을 쓰자 (둘 다 임계치 {thresh} 이상)")
    elif abl_ok:
        print(f"  결론: d_ablation을 쓰자 (AUROC {best_l_abl['auc_ablation']:.4f} ≥ {thresh})")
    elif atn_ok:
        print(f"  결론: d_attn을 쓰자 (AUROC {best_l_atn['auc_attn']:.4f} ≥ {thresh})")
    else:
        print(f"  결론: 둘 다 실패 (max AUROC = {max(best_l_abl['auc_ablation'], best_l_atn['auc_attn']):.4f} < {thresh}), 재설계 필요")

    # JSON 저장
    out_json = os.path.join(args.output_dir, 'diag_results.json')
    with open(out_json, 'w') as f:
        json.dump({'rows': rows, 'n_hall': n_hall, 'n_faith': n_faith,
                   'n_images': len(all_imgs), 'n_img_tokens': n_img_global}, f, indent=2)
    print(f"\n결과 저장: {out_json}")

    # 방향 벡터 저장 (직교 투영 디코딩에 사용)
    dir_path = os.path.join(args.output_dir, 'directions.pt')
    torch.save({
        'd_ablation': {l: d_ablation[l] for l in layer_ids},
        'd_true':     {l: d_true[l]     for l in layer_ids},
    }, dir_path)
    print(f"방향 벡터 저장: {dir_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        layers_x = [r['layer'] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(layers_x, [r['cos_ablation_true'] for r in rows], 'b-o', label='cos(d_ablation, d_true)')
        ax.plot(layers_x, [r['cos_attn_true']     for r in rows], 'r-s', label='cos(d_attn, d_true)')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('Layer'); ax.set_ylabel('Cosine similarity')
        ax.set_title('Cosine alignment with d_true'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(layers_x, [r['auc_dtrue']    for r in rows], 'k-^', label='AUROC(d_true) [ceiling]')
        ax.plot(layers_x, [r['auc_ablation'] for r in rows], 'b-o', label='AUROC(d_ablation)')
        ax.plot(layers_x, [r['auc_attn']     for r in rows], 'r-s', label='AUROC(d_attn)')
        ax.axhline(0.65, color='orange', linestyle='--', label='threshold 0.65')
        ax.axhline(0.5,  color='gray',   linestyle=':',  label='random')
        ax.set_xlabel('Layer'); ax.set_ylabel('AUROC')
        ax.set_title('Projection-separation AUROC'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.05)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, 'diag_plot.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Plot 저장: {plot_path}")
    except Exception as e:
        print(f"Plot 실패 (무시): {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--image_dir',   default='/home/sjkim/datasets/coco/val2014')
    parser.add_argument('--ann_file',    default='/home/sjkim/datasets/coco/annotations/instances_val2014.json')
    parser.add_argument('--n_images',    type=int, default=200)
    parser.add_argument('--layers',      type=int, nargs='+', default=list(range(2, 32, 2)))
    parser.add_argument('--output_dir',  default='./diag_output')
    args = parser.parse_args()
    main(args)
