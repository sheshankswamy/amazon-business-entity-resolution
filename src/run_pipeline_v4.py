"""
Amazon Business Entity Resolution — Version 4 Official Pipeline Runner

Integrates validated country-specific configurations from the reproducibility audit:
- USA: High-cardinality composite blocking + soft house-number compatibility + trailing phone extraction (T=0.75)
- INDIA: V3 multi-script Brahmi phonetic transliteration + soft house-number compatibility (phone inactive, T=0.75)
- FRANCE: V3 French legal suffix normalization + address normalization + conservative rules (T=0.75)

Produces:
- output/v4_matching_results.tsv
- output/v4_candidate_pairs.tsv
- output/v4/v4_matching_results.tsv
- output/v4/v4_candidate_pairs.tsv
"""

import os
import sys
import gc
import time
from collections import defaultdict, Counter
import numpy as np
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
EXP1_DIR = os.path.join(PROJECT_ROOT, "experiments", "exp1_composite_blocking")
EXP2_DIR = os.path.join(PROJECT_ROOT, "experiments", "exp2_synthetic_normalization")

sys.path.append(PROJECT_ROOT)
sys.path.append(EXP1_DIR)
sys.path.append(EXP2_DIR)

from v3_features import (
    clean_name_words_v3, clean_address_tokens_v3, get_char_ngrams_v3, get_translit_char_ngrams_v3,
    generate_blocking_keys_v3, compute_pair_features_v3
)
from composite_blocking import generate_composite_blocking_keys
from exp2_features import (
    compute_pair_features_exp2, extract_trailing_phone, FEAT_NAMES
)

def split_s1_by_country_if_needed():
    """Verifies that split test S1 files exist; creates them if not."""
    order_file = os.path.join(LOGS_DIR, "test_s1_order.txt")
    fr_file = os.path.join(LOGS_DIR, "test_s1_France.tsv")
    us_file = os.path.join(LOGS_DIR, "test_s1_US.tsv")
    in_file = os.path.join(LOGS_DIR, "test_s1_India.tsv")
    
    if os.path.isfile(order_file) and os.path.isfile(fr_file) and os.path.isfile(us_file) and os.path.isfile(in_file):
        print(f"Pre-split test files verified in {LOGS_DIR}:")
        for fn in ["test_s1_France.tsv", "test_s1_US.tsv", "test_s1_India.tsv", "test_s1_order.txt"]:
            size = os.path.getsize(os.path.join(LOGS_DIR, fn))
            print(f"  {fn}: {size:,} bytes")
        return
        
    t0 = time.time()
    s1_path = os.path.join(DATA_DIR, "test", "test_source1.tsv")
    print(f"Splitting {s1_path} by country...")
    
    country_files = {
        "France": open(fr_file, "w", encoding="utf-8"),
        "US": open(us_file, "w", encoding="utf-8"),
        "India": open(in_file, "w", encoding="utf-8")
    }
    
    total_count = 0
    country_counts = Counter()
    
    with open(s1_path, "r", encoding="utf-8") as f, open(order_file, "w", encoding="utf-8") as ord_f:
        f.readline()
        for line in f:
            total_count += 1
            p = line.rstrip("\r\n").split("\t")
            eid, country = p[0], p[3]
            ord_f.write(eid + "\n")
            if country in country_files:
                country_files[country].write(line)
                country_counts[country] += 1

    for fh in country_files.values():
        fh.close()
        
    print(f"Split {total_count:,} records in {time.time() - t0:.2f}s:")
    for c, cnt in country_counts.items():
        print(f"  {c}: {cnt:,} records")

def index_country_targets_v4(country):
    """Indexes S2 and S3 target records using country-specific blocking logic."""
    t0 = time.time()
    print(f"\n--- [Indexing {country} Targets for Version 4] ---")
    index = defaultdict(list)
    target_data = {}
    
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(DATA_DIR, "test", fname)
        cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[3] == country:
                    eid, name, addr = p[0], p[1], p[2]
                    target_data[eid] = (name, addr)
                    
                    if country == "US":
                        clean_name = extract_trailing_phone(name)[0]
                        keys = generate_composite_blocking_keys(clean_name, addr, country)
                    elif country == "India":
                        # V3 multi-script Brahmi phonetic transliteration blocking (phone inactive)
                        keys = generate_blocking_keys_v3(name, addr)
                    else: # France
                        keys = generate_blocking_keys_v3(name, addr)
                        
                    for k in keys:
                        index[k].append(eid)
                    cnt += 1
        print(f"  Indexed {cnt:,} {country} records from {fname}")
        
    print(f"Indexing {country} complete in {time.time() - t0:.2f}s. Unique keys: {len(index):,}, Unique targets: {len(target_data):,}")
    return index, target_data

def score_batch_v4(batch_s1, index, target_data, clf, threshold, country, max_candidates, max_bucket,
                   all_scored_pairs, s1_candidates_dict):
    """Scores a batch of S1 entities using country-specific feature extractors and rules."""
    pairs_to_score = []
    pair_meta = []  # (s1_id, tid)
    
    for s1_id, name, addr in batch_s1:
        if country == "US":
            clean_s1_n = extract_trailing_phone(name)[0]
            keys = generate_composite_blocking_keys(clean_s1_n, addr, country)
        else:
            keys = generate_blocking_keys_v3(name, addr)
            
        cand_counts = Counter()
        for k in keys:
            b = index.get(k)
            if b and len(b) <= max_bucket:
                for tid in b:
                    cand_counts[tid] += 1
                    
        top_cands = [c[0] for c in cand_counts.most_common(max_candidates)]
        s1_candidates_dict[s1_id] = top_cands
        
        for tid in top_cands:
            if tid in target_data:
                t_name, t_addr = target_data[tid]
                
                if country == "US":
                    feats = compute_pair_features_exp2(
                        name, addr, t_name, t_addr, cand_counts[tid],
                        use_house_norm=True, use_phone_norm=True
                    )
                    best_name_sim = feats[7]
                    colocated = feats[17]
                    hard_conflict_low_name = feats[15]
                    
                    # Rule 1: Shared commercial complex distractor rule
                    if colocated == 1.0 and best_name_sim < 0.20:
                        continue
                    # Rule 2: Conflicting house/building numbers
                    if hard_conflict_low_name == 1.0:
                        continue
                elif country == "India":
                    feats = compute_pair_features_exp2(
                        name, addr, t_name, t_addr, cand_counts[tid],
                        use_house_norm=True, use_phone_norm=False
                    )
                    # Soft learned co-location penalty (no hard veto)
                else: # France
                    t_raw_w, t_raw_dw, t_tr_w, t_tr_dw = clean_name_words_v3(t_name)
                    t_raw_ng = get_char_ngrams_v3(t_name)
                    t_tr_ng = get_translit_char_ngrams_v3(t_name)
                    t_aw, t_nums = clean_address_tokens_v3(t_addr)
                    has_t_a = bool(t_addr.strip())
                    
                    s1_raw_w, s1_raw_dw, s1_tr_w, s1_tr_dw = clean_name_words_v3(name)
                    s1_raw_ng = get_char_ngrams_v3(name)
                    s1_tr_ng = get_translit_char_ngrams_v3(name)
                    s1_aw, s1_nums = clean_address_tokens_v3(addr)
                    has_s1_a = bool(addr.strip())
                    
                    feats = compute_pair_features_v3(
                        s1_raw_w, s1_raw_dw, s1_tr_w, s1_tr_dw, s1_raw_ng, s1_tr_ng,
                        s1_aw, s1_nums, has_s1_a,
                        t_raw_w, t_raw_dw, t_tr_w, t_tr_dw, t_raw_ng, t_tr_ng,
                        t_aw, t_nums, has_t_a,
                        cand_counts[tid]
                    )
                    colocated = feats[16]
                    num_conflict_flag = feats[14]
                    best_name_sim = feats[7]
                    
                    if colocated == 1.0 and best_name_sim < 0.20:
                        continue
                    if num_conflict_flag == 1.0:
                        continue
                        
                pairs_to_score.append(feats)
                pair_meta.append((s1_id, tid))

    if pairs_to_score:
        X = np.array(pairs_to_score, dtype=np.float32)
        probs = clf.predict_proba(X)[:, 1]
        for (s1_id, tid), prob in zip(pair_meta, probs):
            if prob >= threshold:
                all_scored_pairs.append((prob, s1_id, tid))
                
    return len(pairs_to_score)

def process_country_s1_v4(country, clf, threshold=0.75, max_candidates=25, max_bucket=500, batch_size=20000):
    """Processes candidate generation, feature scoring, and competitive 1-to-1 assignment for one country."""
    print(f"\n==================== PROCESSING {country} (VERSION 4 PIPELINE) ====================")
    print(f"Configuration for {country}:")
    print(f"  Decision Threshold: {threshold}")
    print(f"  Max Candidates per Query: {max_candidates}")
    print(f"  Max Bucket Size: {max_bucket}")
    if country == "US":
        print(f"  Blocking: High-Cardinality Composite Blocking (with phone-stripped names)")
        print(f"  Features: 21 Exp2 Features (House Norm + Trailing Phone Extraction)")
        print(f"  Rules: Competitive 1-to-1 + Co-Location Distractor Filter + House Number Compatibility")
    elif country == "India":
        print(f"  Blocking: V3 Multi-Script Brahmi Transliteration & Phonetic Consonant Skeletons")
        print(f"  Features: 21 Exp2 Features (House Norm Enabled, Phone Extraction INACTIVE)")
        print(f"  Rules: Competitive 1-to-1 + Soft Learned GBDT Co-Location (No Hard Veto)")
    elif country == "France":
        print(f"  Blocking: V3 French Legal Suffix & Address Normalization")
        print(f"  Features: 18 Baseline V3 Features")
        print(f"  Rules: Conservative Threshold 0.75 + French Stopwords + Co-Location Filter")
        
    index, target_data = index_country_targets_v4(country)
    
    input_file = os.path.join(LOGS_DIR, f"test_s1_{country}.tsv")
    output_file = os.path.join(LOGS_DIR, f"pred_v4_{country}.tsv")
    
    t_start = time.time()
    all_scored_pairs = []  # list of (prob, s1_id, tid)
    s1_candidates_dict = {}  # s1_id -> list of candidate target IDs
    
    total_s1 = 0
    total_candidate_pairs_scored = 0
    
    with open(input_file, "r", encoding="utf-8") as in_f:
        batch_s1 = []
        for line in in_f:
            total_s1 += 1
            p = line.rstrip("\r\n").split("\t")
            batch_s1.append((p[0], p[1], p[2]))
            
            if len(batch_s1) >= batch_size:
                scored_cnt = score_batch_v4(
                    batch_s1, index, target_data, clf, threshold, country, max_candidates, max_bucket,
                    all_scored_pairs, s1_candidates_dict
                )
                total_candidate_pairs_scored += scored_cnt
                batch_s1 = []
                elapsed_now = time.time() - t_start
                rate = total_s1 / elapsed_now if elapsed_now > 0 else 0.0
                print(f"  [{country}] Processed {total_s1:,} entities... ({rate:.0f} S1/sec | Scored pairs above T: {len(all_scored_pairs):,})")
                
        if batch_s1:
            scored_cnt = score_batch_v4(
                batch_s1, index, target_data, clf, threshold, country, max_candidates, max_bucket,
                all_scored_pairs, s1_candidates_dict
            )
            total_candidate_pairs_scored += scored_cnt

    print(f"\nScoring complete for {country}. Total pairs above threshold {threshold}: {len(all_scored_pairs):,}")
    
    # Competitive 1-to-1 Target Assignment:
    t_comp = time.time()
    print(f"Executing competitive 1-to-1 target assignment for {country}...")
    all_scored_pairs.sort(key=lambda x: x[0], reverse=True)
    
    claimed_targets = set()
    s1_final_matches = defaultdict(list)
    multi_claims_prevented = 0
    
    for prob, s1_id, tid in all_scored_pairs:
        # A target can only be assigned to the single S1 entity that claimed it with highest probability
        if tid not in claimed_targets:
            s1_final_matches[s1_id].append(tid)
            claimed_targets.add(tid)
        else:
            multi_claims_prevented += 1
            
    # Statistics and Metrics
    match_lengths = Counter(len(m) for m in s1_final_matches.values())
    print(f"Competitive assignment finished in {time.time() - t_comp:.2f}s:")
    print(f"  Multi-claims prevented (guaranteed false positives averted): {multi_claims_prevented:,}")
    print(f"  Total unique targets matched: {len(claimed_targets):,}")
    print(f"  Average matches per S1: {len(claimed_targets) / total_s1:.2f}")
    singletons = sum(1 for eid in s1_candidates_dict if len(s1_final_matches[eid]) == 0)
    print(f"  Singletons (empty matches): {singletons:,} ({singletons / total_s1 * 100:.2f}%)")
    print(f"  Cardinality distribution (targets per S1):")
    for k in sorted(match_lengths.keys()):
        if k <= 6:
            print(f"    {k} matches: {match_lengths[k]:,} S1 entities")
    print(f"    >6 matches: {sum(match_lengths[k] for k in match_lengths if k > 6):,} S1 entities")
    
    # Write intermediate country log
    print(f"Writing {country} predictions to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for s1_id, cands in s1_candidates_dict.items():
            matches = s1_final_matches.get(s1_id, [])
            valid_matches = [m for m in matches if m in cands]
            cands_str = ",".join(cands)
            match_str = ",".join(valid_matches)
            out_f.write(f"{s1_id}\t{match_str}\t{cands_str}\n")
            
    elapsed = time.time() - t_start
    print(f"Finished {country} pipeline in {elapsed:.2f}s.")
    
    del index
    del target_data
    del all_scored_pairs
    del s1_candidates_dict
    del s1_final_matches
    del claimed_targets
    gc.collect()

def assemble_final_v4_outputs():
    """Assembles final v4_matching_results.tsv and v4_candidate_pairs.tsv in exact test order."""
    t0 = time.time()
    print("\n--- [Assembling Final Version 4 Submission Files] ---")
    
    order_file = os.path.join(LOGS_DIR, "test_s1_order.txt")
    match_out_path = os.path.join(OUTPUT_DIR, "v4_matching_results.tsv")
    cand_out_path = os.path.join(OUTPUT_DIR, "v4_candidate_pairs.tsv")
    match_v4_subdir = os.path.join(OUTPUT_DIR, "v4", "v4_matching_results.tsv")
    cand_v4_subdir = os.path.join(OUTPUT_DIR, "v4", "v4_candidate_pairs.tsv")
    
    print("Loading country predictions...")
    pred_data = {}
    for country in ["France", "US", "India"]:
        cpath = os.path.join(LOGS_DIR, f"pred_v4_{country}.tsv")
        with open(cpath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                eid = parts[0]
                m_str = parts[1] if len(parts) > 1 else ""
                c_str = parts[2] if len(parts) > 2 else ""
                pred_data[eid] = (m_str, c_str)
                
    print(f"Loaded {len(pred_data):,} predictions. Writing V4 outputs in original test sequence...")
    written = 0
    with open(order_file, "r", encoding="utf-8") as ord_f, \
         open(match_out_path, "w", encoding="utf-8") as m_out, \
         open(cand_out_path, "w", encoding="utf-8") as c_out, \
         open(match_v4_subdir, "w", encoding="utf-8") as m_sub, \
         open(cand_v4_subdir, "w", encoding="utf-8") as c_sub:
        
        header_m = "source1_entity_id\tmatched_entity_ids\n"
        header_c = "source1_entity_id\tcandidate_entity_ids\n"
        
        m_out.write(header_m)
        m_sub.write(header_m)
        c_out.write(header_c)
        c_sub.write(header_c)
        
        for line in ord_f:
            eid = line.strip()
            if not eid: continue
            m_str, c_str = pred_data.get(eid, ("", ""))
            line_m = f"{eid}\t{m_str}\n"
            line_c = f"{eid}\t{c_str}\n"
            
            m_out.write(line_m)
            m_sub.write(line_m)
            c_out.write(line_c)
            c_sub.write(line_c)
            written += 1
            
    print(f"Successfully assembled {written:,} rows to:")
    print(f"  {match_out_path}")
    print(f"  {cand_out_path}")
    print(f"  {match_v4_subdir}")
    print(f"  {cand_v4_subdir}")
    print(f"Assembly completed in {time.time() - t0:.2f}s")
    del pred_data
    gc.collect()

def run_submission_validator_v4():
    """Runs the official utils/validate_submission.py validator with --check-ids."""
    print("\n--- [Running Official Submission Validator with --check-ids] ---")
    validator_script = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    matching_tsv = os.path.join(OUTPUT_DIR, "v4_matching_results.tsv")
    candidate_tsv = os.path.join(OUTPUT_DIR, "v4_candidate_pairs.tsv")
    test_dir = os.path.join(DATA_DIR, "test")
    
    import subprocess
    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_tsv,
        "--candidate", candidate_tsv,
        "--check-ids",
        "--test-dir", test_dir
    ]
    print("Executing command: " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)
    print(f"Validator Return Code: {res.returncode}")
    return res.returncode == 0

def run_v4_full_pipeline():
    print("=" * 100)
    print("AMAZON BUSINESS ENTITY RESOLUTION — VERSION 4 OFFICIAL FULL PIPELINE")
    print("=" * 100)
    t_global = time.time()
    
    # Step 1: Ensure test split
    split_s1_by_country_if_needed()
    
    # Step 2: Load Models
    print("\nLoading V4 Model Bundle...")
    clf_france = joblib.load(os.path.join(MODELS_DIR, "v4_france_matcher.joblib"))["model"]
    clf_us = joblib.load(os.path.join(MODELS_DIR, "v4_us_matcher.joblib"))
    clf_india = joblib.load(os.path.join(MODELS_DIR, "v4_india_matcher.joblib"))
    print("All country models loaded successfully.")
    
    # Step 3: Run France Pipeline
    process_country_s1_v4("France", clf_france, threshold=0.75, max_candidates=25, max_bucket=500, batch_size=20000)
    
    # Step 4: Run US Pipeline
    process_country_s1_v4("US", clf_us, threshold=0.75, max_candidates=25, max_bucket=500, batch_size=20000)
    
    # Step 5: Run India Pipeline
    process_country_s1_v4("India", clf_india, threshold=0.75, max_candidates=25, max_bucket=500, batch_size=20000)
    
    # Step 6: Assemble Final Outputs
    assemble_final_v4_outputs()
    
    # Step 7: Run Official Validator
    valid = run_submission_validator_v4()
    
    print("\n" + "=" * 100)
    print(f"VERSION 4 PIPELINE EXECUTION COMPLETE IN {time.time() - t_global:.2f}s")
    print(f"OFFICIAL VALIDATOR STATUS: {'PASS (EXIT CODE 0)' if valid else 'FAILED'}")
    print("=" * 100)

if __name__ == '__main__':
    run_v4_full_pipeline()
