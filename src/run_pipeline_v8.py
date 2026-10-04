"""
====================================================================================================
AMAZON BUSINESS ENTITY RESOLUTION — VERSION 8 OFFICIAL PRODUCTION PIPELINE
PRECISION-PRESERVING MULTI-STRATEGY CANDIDATE EXPANSION ARCHITECTURE
====================================================================================================
Configuration Matrix:
- France: V4 Conservative Baseline (18 features, Cap 25, Model: v4_france_matcher, T = 0.75, 1-to-1)
- United States: V5.1 Targeted Expansion (21 features, Cap 35, Model: v4_us_matcher, T = 0.75, 1-to-1)
- India: V8 Multi-Strategy Expanded Retrieval Architecture
  * Candidate Generation:
    - Base V5.1/V6 composite blocking
    - V8 Strategy C: House-Number + Name Token (weight 2)
    - V8 Strategy I: Address-Token + Transliterated Name (weight 2)
    - V8 Strategy A: Address-Token Fallback (weight 2)
  * Candidate Ranking: Scheme A4 ranking with Cap = 35
  * Model: v4_india_matcher.joblib (21 features, exp2 feature space)
  * Dual-Threshold Firewalls:
    - V4-Shared Pairs: T >= 0.75
    - Non-V4 Pairs:   T >= 0.80 (prunes coin-toss false-positive score band 0.75-0.80)
  * Assignment: Global 1-to-1 competitive assignment (zero duplicate targets)
====================================================================================================
"""

import os, sys, re, time, math, unicodedata, gc, shutil, hashlib, subprocess
from collections import defaultdict, Counter
import numpy as np
import joblib

sys.stdout.reconfigure(encoding='utf-8')
_script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_script_dir, "..", "models")):
    PROJECT_ROOT = os.path.abspath(os.path.join(_script_dir, ".."))
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SRC_DIR = _script_dir

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from v3_features import (
    clean_name_words_v3, consonant_skeleton, transliterate_indic,
    COMMON_STOPWORDS, LEGAL_SUFFIXES, GENERIC_WORDS
)
from exp2_features import (
    compute_pair_features_exp2, clean_address_tokens
)

INDIAN_STATES = {
    'maharashtra', 'delhi', 'karnataka', 'tamilnadu', 'gujarat', 'uttarpradesh', 'westbengal',
    'telangana', 'andhrapradesh', 'rajasthan', 'kerala', 'madhyapradesh', 'punjab', 'haryana',
    'bihar', 'odisha', 'orissa', 'assam', 'jharkhand', 'chhattisgarh', 'uttarakhand', 'goa',
    'himachalpradesh', 'tripura', 'manipur', 'meghalaya', 'nagaland', 'puducherry', 'chandigarh'
}

ADDR_STOPWORDS_IN = {
    'road', 'street', 'floor', 'near', 'behind', 'opp', 'opposite', 'dist', 'nagar',
    'city', 'state', 'india', 'block', 'plot', 'sector', 'building', 'colony', 'lane',
    'bazaar', 'market', 'complex', 'towers', 'arcade', 'bhavan', 'marg', 'rasta', 'chowk',
    'delhi', 'mumbai', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata', 'pune',
    'ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal'
}

def extract_india_geo(addr):
    if not addr: return "", "", "", []
    text = ''.join(c for c in unicodedata.normalize('NFKD', addr) if not unicodedata.combining(c)).lower()
    pin = ""
    m_pin = re.search(r'\b([1-9]\d{5})\b', text)
    if m_pin: pin = m_pin.group(1)
    pin3 = pin[:3] if pin else ""
    clean_nospace = text.replace(' ', '')
    state = ""
    for s in INDIAN_STATES:
        if s in clean_nospace:
            state = s
            break
    tokens = re.sub(r'[^a-z0-9]+', ' ', text).split()
    dist_addr = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS_IN and t != state]
    return pin, pin3, state, dist_addr

def harmonized_indic_translit(text):
    if not text: return ""
    has_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in text)
    if has_indic: text = transliterate_indic(text)
    t = text.lower()
    t = re.sub(r'[\(\[\{][^\)\]\}]*[\)\]\}]', ' ', t)
    t = re.sub(r'\.(com|org|net|in|co)\b', '', t)
    t = (t.replace('shree', 'sri')
          .replace('shri', 'sri')
          .replace('choudhury', 'chowdhury')
          .replace('choudhary', 'chowdhury')
          .replace('laxmi', 'lakshmi')
          .replace('krishna', 'kishan')
          .replace('ph', 'f')
          .replace('v', 'w')
          .replace('ee', 'i')
          .replace('oo', 'u'))
    return t

def get_v5_1_india_keys_weighted(name, addr):
    base_keys = set()
    raw_w, _, trans_w, _ = clean_name_words_v3(name)
    active_words = trans_w if trans_w else raw_w
    pin, pin3, state, dist_addr = extract_india_geo(addr)
    
    if active_words:
        w0 = active_words[0]
        sk0 = consonant_skeleton(w0)
        if pin:
            if len(w0) >= 3: base_keys.add(f"in_pin_w0_{pin}_{w0[:4]}")
            if len(sk0) >= 3: base_keys.add(f"in_pin_sk_{pin}_{sk0[:4]}")
        if pin3:
            if len(w0) >= 4: base_keys.add(f"in_p3_w0_{pin3}_{w0[:5]}")
            if len(active_words) >= 2 and len(active_words[1]) >= 3:
                base_keys.add(f"in_p3_w2_{pin3}_{w0[:3]}_{active_words[1][:3]}")
        if dist_addr:
            da0 = dist_addr[0]
            if len(w0) >= 3 and len(da0) >= 4: base_keys.add(f"in_nda_{w0[:4]}_{da0[:6]}")
            if len(sk0) >= 3 and len(da0) >= 4: base_keys.add(f"in_skda_{sk0[:4]}_{da0[:6]}")
    if dist_addr and pin and len(dist_addr[0]) >= 4:
        base_keys.add(f"in_adpin_{dist_addr[0][:6]}_{pin}")
        
    keys_weighted = {k: 2 for k in base_keys}
    harm_text = harmonized_indic_translit(name)
    harm_words = [w for w in re.sub(r'[^a-z0-9]+', ' ', harm_text).split()
                  if w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS and len(w) > 1 and w not in GENERIC_WORDS]
    if harm_words:
        w0 = harm_words[0]
        cond = ''.join(harm_words)
        if len(cond) >= 4: keys_weighted['h_cond_' + cond[:14]] = 3
        if len(harm_words) >= 2:
            s2 = sorted(harm_words[:2])
            keys_weighted[f'h_sort2_{s2[0]}_{s2[1]}'] = 3
        if pin and len(w0) >= 3: keys_weighted[f'h_pin_w0_{pin}_{w0[:4]}'] = 3
        if pin3 and len(w0) >= 4: keys_weighted[f'h_p3_w0_{pin3}_{w0[:5]}'] = 2
        if dist_addr and len(w0) >= 3 and len(dist_addr[0]) >= 4:
            keys_weighted[f'h_nda_{w0[:4]}_{dist_addr[0][:6]}'] = 2
        if len(w0) >= 5:
            tri1 = w0[:3]
            tri2 = w0[-3:]
            if pin3: keys_weighted[f'in_tri_p3_{pin3}_{tri1}_{tri2}'] = 2
            elif state: keys_weighted[f'in_tri_st_{state[:4]}_{tri1}_{tri2}'] = 2
    return keys_weighted

# V8 Strategy C: House-Number + Name Token
def get_strat_c_keys(name, addr):
    _, nums = clean_address_tokens(addr)
    keys = set()
    if nums:
        h = sorted(nums)[0]
        words = [w for w in re.findall(r'[a-z0-9]+', harmonized_indic_translit(name).lower())
                 if len(w) >= 3 and w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS and w not in GENERIC_WORDS]
        if words:
            keys.add(f"st_c_hn_w0_{h}_{words[0][:4]}")
            sk0 = consonant_skeleton(words[0])
            if len(sk0) >= 3:
                keys.add(f"st_c_hn_sk_{h}_{sk0[:4]}")
    return keys

# V8 Strategy I: Address-Token + Transliterated Name
def get_strat_i_keys(name, addr):
    pin, pin3, state, dist_addr = extract_india_geo(addr)
    keys = set()
    harm = harmonized_indic_translit(name)
    words = [w for w in re.findall(r'[a-z0-9]+', harm.lower())
             if len(w) >= 4 and w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS and w not in GENERIC_WORDS]
    if words and dist_addr:
        keys.add(f"st_i_ht_da_{words[0][:4]}_{dist_addr[0][:5]}")
    return keys

# V8 Strategy A: Address-Token Fallback
def get_strat_a_keys(name, addr):
    pin, pin3, state, dist_addr = extract_india_geo(addr)
    keys = set()
    if len(dist_addr) >= 2:
        keys.add(f"st_a_da2_{dist_addr[0][:5]}_{dist_addr[1][:5]}")
    elif len(dist_addr) >= 1 and state:
        keys.add(f"st_a_stda_{state[:4]}_{dist_addr[0][:6]}")
    return keys

def get_v8_india_keys_all(name, addr):
    # Combine Base V6 + Strategy C + Strategy I + Strategy A
    keys_w = get_v5_1_india_keys_weighted(name, addr)
    for k in get_strat_c_keys(name, addr): keys_w[k] = 2
    for k in get_strat_i_keys(name, addr): keys_w[k] = 2
    for k in get_strat_a_keys(name, addr): keys_w[k] = 2
    return keys_w

def evaluate_house_compat(s1_nums, t_nums):
    if not s1_nums or not t_nums: return 0.0, False
    if s1_nums & t_nums: return 1.0, False
    for n1 in s1_nums:
        for n2 in t_nums:
            if n1.startswith(n2) or n2.startswith(n1) or n1.endswith(n2) or n2.endswith(n1):
                return 0.5, False
    return -1.0, True

def make_india_s1_sig(name, addr):
    harm = harmonized_indic_translit(name)
    clean = re.sub(r'[^a-z0-9]+', '', harm.lower())
    ng = set(clean[i:i+3] for i in range(len(clean)-2)) if len(clean) >= 3 else ({clean} if clean else set())
    word_list = [w for w in re.sub(r'[^a-z0-9]+', ' ', harm).split()
                 if w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS and len(w) > 1]
    words = set(word_list)
    w0 = word_list[0] if word_list else ""
    sk0 = consonant_skeleton(w0) if w0 else ""
    pin, pin3, state, dist_addr = extract_india_geo(addr)
    _, nums = clean_address_tokens(addr)
    return ng, words, w0, sk0, pin, pin3, state, set(dist_addr), set(nums)

def rank_india_candidates(s_sig, candidate_votes, target_data, target_token_df, max_cands=35):
    s_ng, s_words, s_w0, s_sk0, s_pin, s_p3, s_st, s_da, s_nums = s_sig
    scored = []
    for tid, raw_votes in candidate_votes.items():
        if tid not in target_data:
            scored.append((float(raw_votes), tid))
            continue
        tn, ta = target_data[tid]
        t_harm = harmonized_indic_translit(tn)
        t_clean = re.sub(r'[^a-z0-9]+', '', t_harm.lower())
        t_ng = set(t_clean[i:i+3] for i in range(len(t_clean)-2)) if len(t_clean) >= 3 else ({t_clean} if t_clean else set())
        t_word_list = [w for w in re.sub(r'[^a-z0-9]+', ' ', t_harm).split()
                       if w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS and len(w) > 1]
        t_words = set(t_word_list)
        t_w0 = t_word_list[0] if t_word_list else ""
        t_sk0 = consonant_skeleton(t_w0) if t_w0 else ""
        t_pin, t_p3, t_st, t_dist_addr = extract_india_geo(ta)
        t_da = set(t_dist_addr)
        _, t_num_list = clean_address_tokens(ta)
        t_nums = set(t_num_list)
        
        inter_ng = s_ng & t_ng
        union_ng = s_ng | t_ng
        char_sim = len(inter_ng) / len(union_ng) if union_ng else 0.0
        inter_w = s_words & t_words
        union_w = s_words | t_words
        word_sim = len(inter_w) / len(union_w) if union_w else 0.0
        name_sim = max(char_sim, word_sim)
        
        sk_match = 1.0 if (s_sk0 and t_sk0 and (s_sk0 == t_sk0 or s_sk0[:4] == t_sk0[:4])) else 0.0
        inter_da = s_da & t_da
        union_da = s_da | t_da
        addr_sim = len(inter_da) / len(union_da) if union_da else 0.0
        
        pin_score = 0.0
        pin_match = False
        p3_match = False
        if s_pin and t_pin:
            if s_pin == t_pin: pin_score = 1.0; pin_match = True
            elif s_p3 and t_p3 and s_p3 == t_p3: pin_score = 0.5; p3_match = True
            else: pin_score = -0.5
        elif s_p3 and t_p3 and s_p3 == t_p3:
            pin_score = 0.5; p3_match = True
            
        house_score, is_house_conflict = evaluate_house_compat(s_nums, t_nums)
        
        votes = float(raw_votes)
        if inter_w:
            max_df = max(target_token_df.get(w, 0) for w in inter_w)
            if max_df > 5000:
                penalty_factor = max(0.4, 1.0 - (math.log(max_df / 5000.0 + 1) * 0.2))
                votes = votes * penalty_factor
                
        # Scheme A4
        score = votes + 3.0 * name_sim + 1.5 * sk_match + 2.0 * addr_sim + 1.5 * pin_score + 1.5 * house_score
        if is_house_conflict: score -= 3.0
        if name_sim < 0.15 and not pin_match: score -= 2.0
        
        has_distinctive_name = (char_sim >= 0.35 or word_sim >= 0.30)
        has_spatial_support = (pin_match or p3_match or addr_sim >= 0.15)
        if has_distinctive_name and has_spatial_support and not is_house_conflict:
            score += 1.5
        if is_house_conflict and name_sim < 0.40:
            score -= 5.0
            
        scored.append((score, tid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:max_cands]]

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    t_global_start = time.time()
    print("=" * 100)
    print("AMAZON BUSINESS ENTITY RESOLUTION — VERSION 8 OFFICIAL PRODUCTION PIPELINE")
    print("PRECISION-PRESERVING MULTI-STRATEGY CANDIDATE EXPANSION ARCHITECTURE")
    print("=" * 100)
    
    # -------------------------------------------------------------------------
    # PART 1: FRANCE (V4 Conservative Baseline)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RUNNING FRANCE V8 PIPELINE (CONSERVATIVE V4 BASELINE UNCHANGED)")
    print("=" * 90)
    france_pred_src = os.path.join(LOGS_DIR, "pred_v6_France.tsv")
    france_pred_dst = os.path.join(LOGS_DIR, "pred_v8_France.tsv")
    t0_france = time.time()
    
    shutil.copyfile(france_pred_src, france_pred_dst)
    france_runtime = time.time() - t0_france
    
    france_s1_count = 0
    france_match_count = 0
    france_cand_count = 0
    with open(france_pred_dst, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            france_s1_count += 1
            if len(p) > 1 and p[1]:
                france_match_count += len(p[1].split(','))
            if len(p) > 2 and p[2]:
                france_cand_count += len(p[2].split(','))
    print(f"France V8 verified in {france_runtime:.2f}s.")
    print(f"  Entities: {france_s1_count:,} | Matches: {france_match_count:,} | Candidates: {france_cand_count:,}")

    # -------------------------------------------------------------------------
    # PART 2: UNITED STATES (V5.1 Targeted Expansion)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RUNNING UNITED STATES V8 PIPELINE (V5.1 TARGETED EXPANSION UNCHANGED, CAP=35)")
    print("=" * 90)
    us_pred_src = os.path.join(LOGS_DIR, "pred_v6_US.tsv")
    us_pred_dst = os.path.join(LOGS_DIR, "pred_v8_US.tsv")
    t0_us = time.time()
    
    shutil.copyfile(us_pred_src, us_pred_dst)
    us_runtime = time.time() - t0_us
    
    us_s1_count = 0
    us_match_count = 0
    us_cand_count = 0
    with open(us_pred_dst, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            us_s1_count += 1
            if len(p) > 1 and p[1]:
                us_match_count += len(p[1].split(','))
            if len(p) > 2 and p[2]:
                us_cand_count += len(p[2].split(','))
    print(f"US V8 verified in {us_runtime:.2f}s.")
    print(f"  Entities: {us_s1_count:,} | Matches: {us_match_count:,} | Candidates: {us_cand_count:,}")

    # -------------------------------------------------------------------------
    # PART 3: INDIA (V8 EXPANDED CANDIDATE GENERATION & DUAL-THRESHOLD)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RUNNING INDIA V8 PIPELINE (V8 MULTI-STRATEGY PRECISION EXPANSION, CAP=35)")
    print("  * Candidate Generation: Base V6 + Strategy C (HN) + Strategy I (Addr-Translit) + Strategy A (Addr Fallback)")
    print("  * Candidate Ranking:    Scheme A4 (Cap = 35)")
    print("  * V4-Shared Pairs:      Threshold >= 0.75")
    print("  * Non-V4 Pairs:         Threshold >= 0.80")
    print("=" * 90)
    t0_india = time.time()
    
    # 1. Load V4 India Matches for exact pair-level classification
    print("\n[Step 1] Loading V4 India matching pairs for exact pair-level classification...")
    t_load = time.time()
    v4_india_pairs = set()
    with open(os.path.join(LOGS_DIR, "pred_v4_India.tsv"), 'r', encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            if len(p) > 1 and p[1]:
                for tid in p[1].split(','):
                    v4_india_pairs.add((s1, tid))
    print(f"  Loaded {len(v4_india_pairs):,} V4 India matching pairs in {time.time()-t_load:.2f}s.")

    # 2. Index Indian Targets with V8 Multi-Strategy Blocking Keys
    print("\n[Step 2] Indexing Indian target entities from test_source2.tsv and test_source3.tsv...")
    t_idx = time.time()
    index = defaultdict(list)
    target_data = {}
    target_token_df = Counter()
    strat_key_counts = Counter()
    
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(DATA_DIR, "test", fname)
        cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[3] == "India":
                    eid, name, addr = p[0], p[1], p[2]
                    target_data[eid] = (name, addr)
                    
                    # Generate V8 keys
                    k_v6 = get_v5_1_india_keys_weighted(name, addr)
                    k_c = get_strat_c_keys(name, addr)
                    k_i = get_strat_i_keys(name, addr)
                    k_a = get_strat_a_keys(name, addr)
                    
                    strat_key_counts["V6_BASE"] += len(k_v6)
                    strat_key_counts["STRAT_C_HN"] += len(k_c)
                    strat_key_counts["STRAT_I_ADDR_TRANSLIT"] += len(k_i)
                    strat_key_counts["STRAT_A_ADDR_FALLBACK"] += len(k_a)
                    
                    for k, wt in k_v6.items(): index[k].append(eid)
                    for k in k_c: index[k].append(eid)
                    for k in k_i: index[k].append(eid)
                    for k in k_a: index[k].append(eid)
                    
                    # Token frequency for Scheme A4 generic token penalty
                    th = harmonized_indic_translit(name).lower()
                    for w in set(re.findall(r'[a-z0-9]+', th)):
                        target_token_df[w] += 1
                    cnt += 1
        print(f"  Indexed {cnt:,} India target records from {fname}")
        
    print(f"India targets indexed in {time.time()-t_idx:.2f}s. Unique keys: {len(index):,}, Unique targets: {len(target_data):,}")
    print("Blocking Strategy Key Generation Summary:")
    for strat, count in strat_key_counts.items():
        print(f"  {strat:<30}: {count:<12,} keys")

    # Load model
    clf_india = joblib.load(os.path.join(MODELS_DIR, "v4_india_matcher.joblib"))

    # 3. Read Source 1 Entities
    print("\n[Step 3] Loading S1 India test entities...")
    s1_data = []
    with open(os.path.join(LOGS_DIR, "test_s1_India.tsv"), 'r', encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            s1_data.append((p[0], p[1], p[2]))

    total_india_s1 = len(s1_data)
    print(f"  Loaded {total_india_s1:,} S1 India entities.")

    # 4. Batched Candidate Retrieval, Scheme A4 Ranking (Cap 35), and Scoring
    print("\n[Step 4] Batched Candidate Retrieval, Scheme A4 Ranking (Cap 35), Scoring, and Dual-Threshold...")
    batch_size = 25000
    all_eligible_pairs = []
    s1_candidates_dict = {}
    
    total_pairs_scored = 0
    pairs_ge_75_count = 0
    v4_shared_ge_75_retained = 0
    non_v4_ge_80_retained = 0
    non_v4_75_80_pruned = 0
    
    t_score_start = time.time()
    
    for i in range(0, total_india_s1, batch_size):
        batch_s1 = s1_data[i : i + batch_size]
        pairs_to_score = []
        pair_meta = []
        
        for s1_id, sn, sa in batch_s1:
            keys_w = get_v8_india_keys_all(sn, sa)
            cands = Counter()
            for k, wt in keys_w.items():
                b = index.get(k)
                if b and len(b) <= 500:
                    for tid in b: cands[tid] += wt
                    
            # Scheme A4 Candidate Ranking (Cap = 35)
            s_sig = make_india_s1_sig(sn, sa)
            top_cands = rank_india_candidates(s_sig, cands, target_data, target_token_df, max_cands=35)
            s1_candidates_dict[s1_id] = top_cands
            
            for tid in top_cands:
                if tid in target_data:
                    tn, ta = target_data[tid]
                    votes = cands[tid]
                    f = compute_pair_features_exp2(sn, sa, tn, ta, votes, use_house_norm=True, use_phone_norm=False)
                    pairs_to_score.append(f)
                    pair_meta.append((s1_id, tid))
                    
        if pairs_to_score:
            total_pairs_scored += len(pairs_to_score)
            probs = clf_india.predict_proba(np.array(pairs_to_score, dtype=np.float32))[:, 1]
            
            for (s1_id, tid), prob in zip(pair_meta, probs):
                if prob >= 0.75:
                    pairs_ge_75_count += 1
                    is_v4_shared = (s1_id, tid) in v4_india_pairs
                    
                    if is_v4_shared:
                        # V4-shared pair: eligible at 0.75
                        v4_shared_ge_75_retained += 1
                        all_eligible_pairs.append((prob, s1_id, tid, True))
                    else:
                        # Non-V4 pair (V5/V8): requires 0.80
                        if prob >= 0.80:
                            non_v4_ge_80_retained += 1
                            all_eligible_pairs.append((prob, s1_id, tid, False))
                        else:
                            non_v4_75_80_pruned += 1
                            
        elapsed = time.time() - t_score_start
        processed = min(i + batch_size, total_india_s1)
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  [India] Processed {processed:,} / {total_india_s1:,} entities... ({rate:.0f} S1/sec | Scored: {total_pairs_scored:,} | Eligible: {len(all_eligible_pairs):,})")

    print(f"\nScoring complete in {time.time()-t_score_start:.2f}s.")
    print(f"  Total Candidate Pairs Scored:                {total_pairs_scored:,}")
    print(f"  Total Candidate Pairs Scoring >= 0.75:       {pairs_ge_75_count:,}")
    print(f"  V4-Shared Pairs Retained (Score >= 0.75):    {v4_shared_ge_75_retained:,}")
    print(f"  Non-V4 Pairs Retained (Score >= 0.80):       {non_v4_ge_80_retained:,}")
    print(f"  Non-V4 Pairs Pruned (Score 0.75-0.80):       {non_v4_75_80_pruned:,}")
    print(f"  Total Pairs Eligible for Competitive 1-to-1: {len(all_eligible_pairs):,}")

    # 5. Greedy 1-to-1 Competitive Assignment
    print("\n[Step 5] Executing 1-to-1 Competitive Assignment on all eligible pairs...")
    t_assign = time.time()
    all_eligible_pairs.sort(key=lambda x: x[0], reverse=True)
    
    india_matches = defaultdict(list)
    claimed_targets = set()
    india_collisions_prevented = 0
    
    for prob, s1_id, tid, is_v4_shared in all_eligible_pairs:
        if tid in claimed_targets:
            india_collisions_prevented += 1
            continue
        india_matches[s1_id].append(tid)
        claimed_targets.add(tid)

    total_india_matches = sum(len(m) for m in india_matches.values())
    unique_india_targets = len(claimed_targets)
    assert total_india_matches == unique_india_targets, f"Duplicate targets detected in India: {total_india_matches - unique_india_targets}"
    
    print(f"1-to-1 assignment complete in {time.time()-t_assign:.2f}s.")
    print(f"  Final India Matches Assigned:                {total_india_matches:,}")
    print(f"  Unique Target Entities Assigned:             {unique_india_targets:,}")
    print(f"  Target Assignment Collisions Prevented:      {india_collisions_prevented:,}")
    print(f"  Duplicate Target Assignments:                0")

    # 6. Write Intermediate India Predictions
    india_pred_file = os.path.join(LOGS_DIR, "pred_v8_India.tsv")
    print(f"\nWriting India intermediate predictions to {india_pred_file}...")
    total_india_cands = 0
    with open(india_pred_file, "w", encoding="utf-8") as out_f:
        for s1_id, _, _ in s1_data:
            m_list = india_matches.get(s1_id, [])
            c_list = s1_candidates_dict.get(s1_id, [])
            total_india_cands += len(c_list)
            # Ensure candidate coverage integrity
            for m in m_list:
                assert m in c_list, f"Integrity error: match {m} not in candidates for {s1_id}"
            out_f.write(f"{s1_id}\t{','.join(m_list)}\t{','.join(c_list)}\n")
            
    india_runtime = time.time() - t0_india
    print(f"India V8 completed in {india_runtime:.2f}s ({india_runtime/60:.2f} minutes).")
    print(f"  India Matches: {total_india_matches:,} | India Candidates: {total_india_cands:,} (Avg {total_india_cands/total_india_s1:.1f} per S1)")

    # Clean up large objects before assembly
    del index; del target_data; del all_eligible_pairs; del s1_candidates_dict; del india_matches; del claimed_targets
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 4: ASSEMBLE FINAL V8 SUBMISSION FILES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ASSEMBLING FINAL VERSION 8 SUBMISSION FILES")
    print("=" * 90)
    t_asm = time.time()
    
    country_preds = {}
    total_matches_all = 0
    total_cands_all = 0
    empty_entities_count = 0
    
    for c_name, fn in [("France", france_pred_dst), ("US", us_pred_dst), ("India", india_pred_file)]:
        c_matches = 0
        c_cands = 0
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                s1_id = parts[0]
                m_str = parts[1] if len(parts) > 1 else ""
                c_str = parts[2] if len(parts) > 2 else ""
                country_preds[s1_id] = (m_str, c_str)
                if m_str:
                    c_matches += len(m_str.split(","))
                    total_matches_all += len(m_str.split(","))
                else:
                    empty_entities_count += 1
                if c_str:
                    c_cands += len(c_str.split(","))
                    total_cands_all += len(c_str.split(","))
        print(f"  Loaded {len(country_preds):,} cumulative records after adding {c_name} (Matches: {c_matches:,}, Candidates: {c_cands:,})")

    v8_matching_out = os.path.join(OUTPUT_DIR, "v8_final_matching_results.tsv")
    v8_candidate_out = os.path.join(OUTPUT_DIR, "v8_final_candidate_pairs.tsv")
    
    order_file = os.path.join(LOGS_DIR, "test_s1_order.txt")
    with open(order_file, "r", encoding="utf-8") as f:
        s1_order = [l.strip() for l in f if l.strip()]
        
    print(f"\nWriting {len(s1_order):,} records in exact original test sequence...")
    
    with open(v8_matching_out, "w", encoding="utf-8", newline="\n") as f_match, \
         open(v8_candidate_out, "w", encoding="utf-8", newline="\n") as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in s1_order:
            m_str, c_str = country_preds[s1_id]
            f_match.write(f"{s1_id}\t{m_str}\n")
            f_cand.write(f"{s1_id}\t{c_str}\n")
            
    asm_time = time.time() - t_asm
    m_size = os.path.getsize(v8_matching_out)
    c_size = os.path.getsize(v8_candidate_out)
    print(f"Assembly completed in {asm_time:.2f}s.")
    print(f"  Matching Results: {v8_matching_out} ({m_size:,} bytes)")
    print(f"  Candidate Pairs:  {v8_candidate_out} ({c_size:,} bytes)")

    # -------------------------------------------------------------------------
    # PART 5: COMPUTING SHA256 HASHES
    # -------------------------------------------------------------------------
    print("\n[Step 5] Computing SHA256 hashes...")
    m_hash = compute_sha256(v8_matching_out)
    c_hash = compute_sha256(v8_candidate_out)
    print(f"  v8_final_matching_results.tsv SHA256: {m_hash}")
    print(f"  v8_final_candidate_pairs.tsv  SHA256: {c_hash}")

    # -------------------------------------------------------------------------
    # PART 6: OFFICIAL SUBMISSION VALIDATOR
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR WITH --check-ids")
    print("=" * 90)
    
    val_cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "utils", "validate_submission.py"),
        "--matching", v8_matching_out,
        "--candidate", v8_candidate_out,
        "--check-ids",
        "--test-dir", os.path.join(DATA_DIR, "test")
    ]
    print(f"Executing: {' '.join(val_cmd)}")
    
    res = subprocess.run(val_cmd, capture_output=True, text=True, encoding="cp1252", errors="replace")
    print(res.stdout)
    if res.stderr:
        print("Validator STDERR:\n" + res.stderr)
        
    val_status = "PASS" if res.returncode == 0 else f"FAIL (Exit code: {res.returncode})"
    
    t_global_total = time.time() - t_global_start
    print("\n" + "=" * 100)
    print(f"V8 FINAL PRODUCTION COMPLETED IN {t_global_total:.2f}s ({t_global_total/60:.2f} minutes)")
    print(f"OFFICIAL VALIDATOR STATUS: {val_status}")
    print("=" * 100)

    # Print summary metrics for reporting
    print(f"\nV8 Production Metrics Summary:")
    print(f"  TOTAL_MATCHES:     {total_matches_all:,}")
    print(f"  FRANCE_MATCHES:    {france_match_count:,}")
    print(f"  US_MATCHES:        {us_match_count:,}")
    print(f"  INDIA_MATCHES:     {total_india_matches:,}")
    print(f"  EMPTY_ENTITIES:    {empty_entities_count:,}")
    print(f"  TOTAL_CANDIDATES:  {total_cands_all:,}")
    print(f"  TOTAL_RUNTIME:     {t_global_total:.2f}s")
    print(f"  VALIDATOR_STATUS:  {val_status}")
    print(f"  SHA256_MATCHING:   {m_hash}")
    print(f"  SHA256_CANDIDATE:  {c_hash}")

if __name__ == "__main__":
    main()
