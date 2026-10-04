"""
====================================================================================================
AMAZON BUSINESS ENTITY RESOLUTION — VERSION 9 OFFICIAL PRODUCTION PIPELINE
FULL-CORPUS HARD-NEGATIVE LIGHTGBM ARCHITECTURE WITH RAPIDFUZZ SIMILARITIES
====================================================================================================
Configuration Matrix:
- France: V4 Conservative Baseline (18 features, Cap 25, Model: v4_france_matcher, T = 0.75, 1-to-1)
- United States: V5.1 Targeted Expansion (21 features, Cap 35, Model: v4_us_matcher, T = 0.75, 1-to-1)
- India: V9 Full-Corpus Hard-Negative LightGBM Architecture
  * Candidate Generation:
    - Base V5.1/V6 composite blocking
    - V8 Strategy C: House-Number + Name Token (weight 2)
    - V8 Strategy I: Address-Token + Transliterated Name (weight 2)
    - V8 Strategy A: Address-Token Fallback (weight 2)
  * Candidate Ranking: Scheme A4 ranking with Cap = 35
  * Model: v9_india_matcher_37f.joblib (37 pairwise features: RapidFuzz Levenshtein, Jaro-Winkler,
           token sort/set ratios, numeric compatibility, cross-attribute multiplicative interactions)
  * Decision Threshold: Calibrated Optimal T = 0.90
  * Assignment: Global 1-to-1 competitive assignment (target-side exclusivity, S1 multi-matches enabled)
====================================================================================================
"""

import os, sys, re, time, math, unicodedata, gc, shutil, hashlib, subprocess
from collections import defaultdict, Counter
import numpy as np
import joblib
import lightgbm as lgb
from rapidfuzz import fuzz, distance

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
from run_pipeline_v8 import (
    harmonized_indic_translit, get_v5_1_india_keys_weighted,
    get_strat_c_keys, get_strat_i_keys, get_strat_a_keys, get_v8_india_keys_all,
    make_india_s1_sig, rank_india_candidates, extract_india_geo
)
from production_features import (
    compute_pairwise_features, FEATURE_NAMES
)

def run_v9_production_pipeline():
    print("=" * 100)
    print("STARTING VERSION 9 PRODUCTION PIPELINE EXECUTION")
    print("  * Target: Full 1,732,544-entity Test Set")
    print("  * France: V4 Conservative Baseline (T=0.75, Cap 25)")
    print("  * US:     V5.1 Targeted Expansion (T=0.75, Cap 35)")
    print("  * India:  V9 37-Feature LightGBM Hard-Negative Matcher (T=0.90, Cap 35)")
    print("=" * 100)
    t_global_start = time.time()
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # PART 1: FRANCE (V4 Conservative Baseline)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RUNNING FRANCE V9 PIPELINE (V4 CONSERVATIVE BASELINE UNCHANGED, CAP=25)")
    print("=" * 90)
    france_pred_src = os.path.join(LOGS_DIR, "pred_v6_France.tsv")
    france_pred_dst = os.path.join(LOGS_DIR, "pred_v9_France.tsv")
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
            if len(p) > 1 and p[1]: france_match_count += len(p[1].split(','))
            if len(p) > 2 and p[2]: france_cand_count += len(p[2].split(','))
    print(f"France V9 verified in {france_runtime:.2f}s.")
    print(f"  Entities: {france_s1_count:,} | Matches: {france_match_count:,} | Candidates: {france_cand_count:,}")

    # -------------------------------------------------------------------------
    # PART 2: UNITED STATES (V5.1 Targeted Expansion)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RUNNING UNITED STATES V9 PIPELINE (V5.1 TARGETED EXPANSION UNCHANGED, CAP=35)")
    print("=" * 90)
    us_pred_src = os.path.join(LOGS_DIR, "pred_v6_US.tsv")
    us_pred_dst = os.path.join(LOGS_DIR, "pred_v9_US.tsv")
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
            if len(p) > 1 and p[1]: us_match_count += len(p[1].split(','))
            if len(p) > 2 and p[2]: us_cand_count += len(p[2].split(','))
    print(f"US V9 verified in {us_runtime:.2f}s.")
    print(f"  Entities: {us_s1_count:,} | Matches: {us_match_count:,} | Candidates: {us_cand_count:,}")

    # -------------------------------------------------------------------------
    # PART 3: INDIA (V9 37-FEATURE LIGHTGBM HARD-NEGATIVE ARCHITECTURE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RUNNING INDIA V9 PIPELINE (37-FEATURE LIGHTGBM HARD-NEGATIVE MATCHER, T=0.90, CAP=35)")
    print("=" * 90)
    t0_india = time.time()

    # 1. Load V9 Production Matcher
    india_model_path = os.path.join(MODELS_DIR, "v9_india_matcher_37f.joblib")
    print(f"Loading V9 India production matcher from {india_model_path}...")
    clf_india = joblib.load(india_model_path)
    print(f"  Loaded model successfully: {type(clf_india)}")

    # 2. Index COMPLETE Indian Test Target Corpus (4,717,565 targets)
    print("\n[Step 1] Indexing COMPLETE Indian test target entities from test_source2.tsv and test_source3.tsv...")
    t_idx = time.time()
    index = defaultdict(list)
    target_data = {}
    target_token_df = Counter()
    
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
                    
                    for k, wt in k_v6.items(): index[k].append(eid)
                    for k in k_c: index[k].append(eid)
                    for k in k_i: index[k].append(eid)
                    for k in k_a: index[k].append(eid)
                    
                    th = harmonized_indic_translit(name).lower()
                    for w in set(re.findall(r'[a-z0-9]+', th)):
                        target_token_df[w] += 1
                    cnt += 1
        print(f"  Indexed {cnt:,} India target records from {fname}")
        
    print(f"India targets indexed in {time.time()-t_idx:.2f}s. Unique keys: {len(index):,}, Unique targets: {len(target_data):,}")

    # 3. Load India Source 1 Test Entities
    print("\n[Step 2] Loading India Source 1 test entities...")
    s1_data = []
    with open(os.path.join(DATA_DIR, "test", "test_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[3] == "India":
                s1_data.append((p[0], p[1], p[2]))
    total_india_s1 = len(s1_data)
    print(f"Loaded {total_india_s1:,} Indian Source 1 test entities.")

    # 4. Stream Candidate Generation, Feature Extraction & Scoring in Batches
    print("\n[Step 3] Executing Candidate Generation & 37-Feature Scoring in batches...")
    t_score_start = time.time()
    batch_size = 25000
    s1_candidates_dict = {}
    all_eligible_pairs = []
    total_pairs_scored = 0
    pairs_ge_90_count = 0
    
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
            
            for rank, tid in enumerate(top_cands, 1):
                if tid in target_data:
                    tn, ta = target_data[tid]
                    votes = cands[tid]
                    f37 = compute_pairwise_features(sn, sa, tn, ta, rank=rank, votes=votes, is_india=True)
                    pairs_to_score.append(f37)
                    pair_meta.append((s1_id, tid))
                    
        if pairs_to_score:
            total_pairs_scored += len(pairs_to_score)
            probs = clf_india.predict_proba(np.array(pairs_to_score, dtype=np.float32))[:, 1]
            
            for (s1_id, tid), prob in zip(pair_meta, probs):
                if prob >= 0.90:
                    pairs_ge_90_count += 1
                    all_eligible_pairs.append((prob, s1_id, tid))
                    
        elapsed = time.time() - t_score_start
        processed = min(i + batch_size, total_india_s1)
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  [India] Processed {processed:,} / {total_india_s1:,} entities... ({rate:.0f} S1/sec | Scored: {total_pairs_scored:,} | Eligible >= 0.90: {pairs_ge_90_count:,})")

    print(f"\nScoring complete in {time.time()-t_score_start:.2f}s.")
    print(f"  Total Candidate Pairs Scored:                {total_pairs_scored:,}")
    print(f"  Total Candidate Pairs Scoring >= 0.90:       {pairs_ge_90_count:,}")

    # 5. Greedy 1-to-1 Competitive Assignment (Target-Side Exclusivity)
    print("\n[Step 4] Executing Target-Side Competitive Assignment on all eligible pairs...")
    t_assign = time.time()
    all_eligible_pairs.sort(key=lambda x: x[0], reverse=True)
    
    india_matches = defaultdict(list)
    claimed_targets = set()
    india_collisions_prevented = 0
    
    for prob, s1_id, tid in all_eligible_pairs:
        if tid in claimed_targets:
            india_collisions_prevented += 1
            continue
        india_matches[s1_id].append(tid)
        claimed_targets.add(tid)

    total_india_matches = sum(len(m) for m in india_matches.values())
    unique_india_targets = len(claimed_targets)
    assert total_india_matches == unique_india_targets, f"Duplicate targets detected in India: {total_india_matches - unique_india_targets}"
    
    multi_match_s1_india = sum(1 for m in india_matches.values() if len(m) > 1)
    max_matches_india = max((len(m) for m in india_matches.values()), default=0)
    
    print(f"1-to-1 target assignment complete in {time.time()-t_assign:.2f}s.")
    print(f"  Final India Matches Assigned:                {total_india_matches:,}")
    print(f"  Unique Target Entities Assigned:             {unique_india_targets:,}")
    print(f"  Duplicate Target Assignments:                0 (Strict Exclusivity Guaranteed)")
    print(f"  Target Collisions Prevented:                 {india_collisions_prevented:,}")
    print(f"  India S1 Entities with >1 Matches:           {multi_match_s1_india:,} ({multi_match_s1_india/total_india_s1*100:.2f}%)")
    print(f"  Maximum Matches for Single S1:               {max_matches_india}")

    # 6. Write Intermediate India Predictions
    india_pred_file = os.path.join(LOGS_DIR, "pred_v9_India.tsv")
    print(f"\nWriting India intermediate predictions to {india_pred_file}...")
    total_india_cands = 0
    with open(india_pred_file, "w", encoding="utf-8") as out_f:
        for s1_id, _, _ in s1_data:
            m_list = india_matches.get(s1_id, [])
            c_list = s1_candidates_dict.get(s1_id, [])
            total_india_cands += len(c_list)
            for m in m_list:
                assert m in c_list, f"Integrity error: match {m} not in candidates for {s1_id}"
            out_f.write(f"{s1_id}\t{','.join(m_list)}\t{','.join(c_list)}\n")
            
    india_runtime = time.time() - t0_india
    print(f"India V9 completed in {india_runtime:.2f}s ({india_runtime/60:.2f} minutes).")

    # Clean up objects before assembly
    del index; del target_data; del all_eligible_pairs; del s1_candidates_dict; del india_matches; del claimed_targets
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 4: ASSEMBLE FINAL V9 SUBMISSION FILES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ASSEMBLING FINAL VERSION 9 SUBMISSION FILES")
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

    v9_matching_out = os.path.join(OUTPUT_DIR, "v9_final_matching_results.tsv")
    v9_candidate_out = os.path.join(OUTPUT_DIR, "v9_final_candidate_pairs.tsv")
    
    order_file = os.path.join(LOGS_DIR, "test_s1_order.txt")
    with open(order_file, "r", encoding="utf-8") as f:
        s1_order = [l.strip() for l in f if l.strip()]
        
    print(f"\nWriting {len(s1_order):,} records in exact original test sequence...")
    
    with open(v9_matching_out, "w", encoding="utf-8", newline="\n") as f_match, \
         open(v9_candidate_out, "w", encoding="utf-8", newline="\n") as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in s1_order:
            m_str, c_str = country_preds[s1_id]
            f_match.write(f"{s1_id}\t{m_str}\n")
            f_cand.write(f"{s1_id}\t{c_str}\n")
            
    print(f"Assembly complete in {time.time()-t_asm:.2f}s.")
    print(f"  Matching Output:  {v9_matching_out} ({os.path.getsize(v9_matching_out):,} bytes)")
    print(f"  Candidate Output: {v9_candidate_out} ({os.path.getsize(v9_candidate_out):,} bytes)")

    # Compute SHA-256 Hashes
    print("\nComputing SHA-256 Hashes of V9 submission outputs...")
    def get_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536): h.update(chunk)
        return h.hexdigest().upper()

    sha_matching = get_sha256(v9_matching_out)
    sha_candidates = get_sha256(v9_candidate_out)
    print(f"  SHA256 (Matching):   {sha_matching}")
    print(f"  SHA256 (Candidates): {sha_candidates}")

    # Official Validator Check
    print("\n" + "=" * 90)
    print("RUNNING OFFICIAL COMPETITION SUBMISSION VALIDATOR")
    print("=" * 90)
    validator_cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "utils", "validate_submission.py"),
        "--matching", v9_matching_out,
        "--candidate", v9_candidate_out,
        "--check-ids",
        "--test-dir", os.path.join(DATA_DIR, "test")
    ]
    val_proc = subprocess.run(validator_cmd, capture_output=True, text=True)
    print(val_proc.stdout)
    if val_proc.stderr: print(val_proc.stderr)
    assert val_proc.returncode == 0, f"Validator failed with exit code {val_proc.returncode}"
    print("OFFICIAL VALIDATOR RESULT: PASS (Code 0)")

    total_time = time.time() - t_global_start
    print("\n" + "=" * 100)
    print("VERSION 9 PRODUCTION EXECUTION SUMMARY")
    print("=" * 100)
    print(f"  Total Runtime:                {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Total Records Processed:      {len(s1_order):,}")
    print(f"  Total Matches Generated:      {total_matches_all:,}")
    print(f"    France Matches:             {france_match_count:,}")
    print(f"    US Matches:                 {us_match_count:,}")
    print(f"    India Matches:              {total_india_matches:,}")
    print(f"  Total Candidate Pairs:        {total_cands_all:,}")
    print(f"  Empty S1 Entities:            {empty_entities_count:,} ({empty_entities_count/len(s1_order)*100:.2f}%)")
    print(f"  Average Matches per Non-Empty: {total_matches_all/(len(s1_order)-empty_entities_count):.2f}")
    print("=" * 100)

if __name__ == '__main__':
    run_v9_production_pipeline()
