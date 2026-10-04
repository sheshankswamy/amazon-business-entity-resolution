"""
Amazon Business Entity Resolution — Train Baseline Matcher
Trains a fast, robust matching model and calibrates threshold for Macro F0.5.
"""

import os
import sys
import re
import unicodedata
import time
from collections import defaultdict, Counter
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'enterprises', 'solutions',
    'sarl', 'sasu', 'eurl', 'sa', 'snc', 'sci', 'gmbh', 'plc', 'llp',
    'technologies', 'consulting', 'group', 'holdings', 'industries', 'international'
}

COMMON_STOPWORDS = {
    'the', 'and', 'of', 'in', 'for', 'at', 'on', 'to', 'a', 'an', 'by',
    'de', 'la', 'le', 'et', 'du', 'des', 'les'
}

def clean_name_words(text):
    if not text:
        return []
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'\.(com|org|net|in|co|io|fr|biz|info)\b', '', text)
    text = re.sub(r'[\(\[\{][^\)\]\}]*[\)\]\}]', ' ', text)
    text = re.sub(r'\b(formerly|dba|mr|mrs|ms)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\u0900-\u097f]+', ' ', text)
    return [w for w in text.split() if w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS and len(w) > 1]

def clean_address_tokens(addr):
    if not addr:
        return [], []
    text = ''.join(c for c in unicodedata.normalize('NFKD', addr) if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\u0900-\u097f]+', ' ', text)
    words = []
    nums = []
    for t in text.split():
        if t.isdigit():
            norm_num = str(int(t))
            if len(norm_num) >= 2:
                nums.append(norm_num)
        elif len(t) >= 3 and t not in COMMON_STOPWORDS:
            words.append(t)
    return words, nums

def get_char_ngrams(text, n=3):
    clean = re.sub(r'[^a-z0-9\u0900-\u097f]', '', text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def generate_blocking_keys(name, addr):
    words = clean_name_words(name)
    addr_words, nums = clean_address_tokens(addr)
    keys = set()
    
    if not words:
        raw_norm = re.sub(r'[^a-z0-9]+', '', name.lower())
        if raw_norm:
            keys.add('raw_' + raw_norm[:10])
        return keys
    
    condensed = ''.join(words)
    if len(condensed) >= 4:
        keys.add('cond_' + condensed[:14])
        
    sorted_words = sorted(words)
    keys.add('sort2_' + '_'.join(sorted_words[:2]))
    if len(sorted_words) >= 3:
        keys.add('sort3_' + '_'.join(sorted_words[:3]))
        
    if len(words[0]) >= 4:
        keys.add('w0_' + words[0])
    if len(words) >= 2 and len(words[1]) >= 4:
        keys.add('w1_' + words[1])
        
    if nums:
        keys.add(f"wnum_{words[0][:5]}_{nums[0]}")
        if len(sorted_words) >= 2:
            keys.add(f"wnum_{sorted_words[0][:5]}_{nums[0]}")

    return keys

def compute_pair_features(s1_name_words, s1_addr_words, s1_nums, s1_char_ngrams, has_s1_addr,
                          t_name_words, t_addr_words, t_nums, t_char_ngrams, has_t_addr,
                          shared_keys):
    # Name word Jaccard
    s1_w_set = set(s1_name_words)
    t_w_set = set(t_name_words)
    inter_words = s1_w_set & t_w_set
    union_words = s1_w_set | t_w_set
    name_jaccard = len(inter_words) / len(union_words) if union_words else 0.0
    
    # Name word containment
    min_len = min(len(s1_w_set), len(t_w_set))
    name_containment = len(inter_words) / min_len if min_len > 0 else 0.0
    
    # Character 3-gram Jaccard
    inter_ng = s1_char_ngrams & t_char_ngrams
    union_ng = s1_char_ngrams | t_char_ngrams
    char_jaccard = len(inter_ng) / len(union_ng) if union_ng else 0.0
    
    # Condensed match
    c1 = ''.join(s1_name_words)
    c2 = ''.join(t_name_words)
    condensed_match = 1.0 if (c1 and c2 and (c1 == c2 or c1.startswith(c2) or c2.startswith(c1))) else 0.0
    
    # Address features
    both_addr = 1.0 if (has_s1_addr and has_t_addr) else 0.0
    if both_addr:
        s1_a_set = set(s1_addr_words)
        t_a_set = set(t_addr_words)
        inter_addr = s1_a_set & t_a_set
        union_addr = s1_a_set | t_a_set
        addr_jaccard = len(inter_addr) / len(union_addr) if union_addr else 0.0
        
        s1_n_set = set(s1_nums)
        t_n_set = set(t_nums)
        if s1_n_set and t_n_set:
            num_match = 1.0 if (s1_n_set & t_n_set) else 0.0
        else:
            num_match = 0.5
    else:
        addr_jaccard = 0.0
        num_match = 0.5
        
    return [
        name_jaccard,
        name_containment,
        char_jaccard,
        condensed_match,
        both_addr,
        addr_jaccard,
        num_match,
        float(shared_keys)
    ]

def evaluate_macro_f05(gt_dict, pred_dict, s1_all):
    total_f05 = 0.0
    for s1_id in s1_all:
        true_set = gt_dict.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())
        
        if not true_set and not pred_set:
            f05 = 1.0
        elif not pred_set or not true_set:
            f05 = 0.0
        else:
            tp = len(true_set & pred_set)
            p = tp / len(pred_set)
            r = tp / len(true_set)
            denom = 0.25 * p + r
            f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
            
        total_f05 += f05
    return total_f05 / len(s1_all)

def train_and_evaluate(sample_s1_count=40000):
    print("=" * 80)
    print("TRAINING BASELINE MATCHING MODEL")
    print("=" * 80)
    
    t_start = time.time()
    
    # 1. Load S1 sample (balanced across US and India)
    print(f"Loading {sample_s1_count} S1 records from train_source1.tsv...")
    s1_dict = {}
    with open(os.path.join(DATA_DIR, "train", "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= sample_s1_count:
                break
            p = line.rstrip("\r\n").split("\t")
            s1_dict[p[0]] = (p[1], p[2], p[3])
            
    print(f"Loaded {len(s1_dict)} S1 entities.")
    
    # 2. Load Ground Truth for this sample
    print("Loading ground truth matches...")
    gt = {}
    all_needed_targets = set()
    total_true_matches = 0
    with open(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in s1_dict:
                matches = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
                gt[p[0]] = matches
                all_needed_targets.update(matches)
                total_true_matches += len(matches)
                
    print(f"Total true target matches in sample: {total_true_matches} (Unique target IDs: {len(all_needed_targets)})")
    
    # 3. Index S2 and S3 records
    print("Indexing S2 and S3 records...")
    index = defaultdict(list)
    target_data = {}  # eid -> (name_words, addr_words, nums, char_ngrams, has_addr)
    
    def load_targets(source_file, needed_set, limit=120000):
        count = 0
        with open(source_file, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid, name, addr, cntry = p[0], p[1], p[2], p[3]
                if eid in needed_set or count < limit:
                    nw = clean_name_words(name)
                    aw, nums = clean_address_tokens(addr)
                    ng = get_char_ngrams(name)
                    has_a = bool(addr.strip())
                    target_data[eid] = (nw, aw, nums, ng, has_a)
                    
                    for k in generate_blocking_keys(name, addr):
                        index[(cntry, k)].append(eid)
                    count += 1
        print(f"Indexed {count} records from {os.path.basename(source_file)}")
        
    load_targets(os.path.join(DATA_DIR, "train", "train_source2.tsv"), all_needed_targets, limit=120000)
    load_targets(os.path.join(DATA_DIR, "train", "train_source3.tsv"), all_needed_targets, limit=120000)
    
    # 4. Generate candidate pairs & extract features for training
    print("Generating candidate pairs and computing features...")
    MAX_CANDIDATES = 25
    MAX_BUCKET = 600
    
    s1_keys_list = list(s1_dict.keys())
    # Split 75% train / 25% validation
    n_train = int(len(s1_keys_list) * 0.75)
    train_s1_keys = s1_keys_list[:n_train]
    val_s1_keys = s1_keys_list[n_train:]
    
    X_train, y_train = [], []
    candidate_recall_hits = 0
    total_val_links = sum(len(gt.get(k, set())) for k in val_s1_keys)
    
    for s1_id in train_s1_keys:
        name, addr, cntry = s1_dict[s1_id]
        s1_nw = clean_name_words(name)
        s1_aw, s1_nums = clean_address_tokens(addr)
        s1_ng = get_char_ngrams(name)
        has_s1_a = bool(addr.strip())
        
        keys = generate_blocking_keys(name, addr)
        cand_counts = Counter()
        for k in keys:
            b = index.get((cntry, k))
            if b and len(b) <= MAX_BUCKET:
                for tid in b:
                    cand_counts[tid] += 1
                    
        top_cands = cand_counts.most_common(MAX_CANDIDATES)
        true_set = gt.get(s1_id, set())
        
        # Build training pairs
        for tid, shared_k in top_cands:
            if tid not in target_data:
                continue
            t_nw, t_aw, t_nums, t_ng, has_t_a = target_data[tid]
            feats = compute_pair_features(
                s1_nw, s1_aw, s1_nums, s1_ng, has_s1_a,
                t_nw, t_aw, t_nums, t_ng, has_t_a,
                shared_k
            )
            label = 1 if tid in true_set else 0
            X_train.append(feats)
            y_train.append(label)
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Training set: {len(X_train):,} pairs (Positives: {np.sum(y_train):,}, Negatives: {len(y_train) - np.sum(y_train):,})")
    
    # 5. Train Model
    print("Fitting HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # 6. Evaluate on Validation Split & Calibrate Threshold
    print("Scoring validation set and optimizing decision threshold for Macro F0.5...")
    val_preds_prob = {}  # s1_id -> list of (tid, prob)
    
    val_blocking_hits = 0
    val_total_cands = 0
    
    for s1_id in val_s1_keys:
        name, addr, cntry = s1_dict[s1_id]
        s1_nw = clean_name_words(name)
        s1_aw, s1_nums = clean_address_tokens(addr)
        s1_ng = get_char_ngrams(name)
        has_s1_a = bool(addr.strip())
        
        keys = generate_blocking_keys(name, addr)
        cand_counts = Counter()
        for k in keys:
            b = index.get((cntry, k))
            if b and len(b) <= MAX_BUCKET:
                for tid in b:
                    cand_counts[tid] += 1
                    
        top_cands = cand_counts.most_common(MAX_CANDIDATES)
        val_total_cands += len(top_cands)
        
        c_ids = [c[0] for c in top_cands]
        true_set = gt.get(s1_id, set())
        val_blocking_hits += len(true_set.intersection(c_ids))
        
        pairs_feats = []
        valid_tids = []
        for tid, shared_k in top_cands:
            if tid not in target_data:
                continue
            t_nw, t_aw, t_nums, t_ng, has_t_a = target_data[tid]
            feats = compute_pair_features(
                s1_nw, s1_aw, s1_nums, s1_ng, has_s1_a,
                t_nw, t_aw, t_nums, t_ng, has_t_a,
                shared_k
            )
            pairs_feats.append(feats)
            valid_tids.append(tid)
            
        if pairs_feats:
            probs = clf.predict_proba(np.array(pairs_feats, dtype=np.float32))[:, 1]
            val_preds_prob[s1_id] = list(zip(valid_tids, probs))
        else:
            val_preds_prob[s1_id] = []
            
    blocking_recall = (val_blocking_hits / total_val_links) * 100 if total_val_links > 0 else 0
    print(f"\n--- VALIDATION BLOCKING RESULTS ---")
    print(f"Validation S1 count: {len(val_s1_keys):,}")
    print(f"Candidate Blocking Recall: {val_blocking_hits:,} / {total_val_links:,} ({blocking_recall:.2f}%)")
    print(f"Avg candidates per S1: {val_total_cands / len(val_s1_keys):.2f}")
    
    # Grid search threshold
    print("\n--- THRESHOLD CALIBRATION FOR MACRO F0.5 ---")
    best_thresh = 0.5
    best_f05 = -1.0
    
    for thresh in np.arange(0.30, 0.85, 0.05):
        pred_dict = {}
        for s1_id in val_s1_keys:
            cand_probs = val_preds_prob.get(s1_id, [])
            matched = {tid for tid, prob in cand_probs if prob >= thresh}
            pred_dict[s1_id] = matched
            
        f05 = evaluate_macro_f05(gt, pred_dict, val_s1_keys)
        print(f"  Threshold {thresh:.2f} -> Macro F0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"\nOPTIMAL THRESHOLD: {best_thresh:.2f} with Validation Macro F0.5 = {best_f05:.4f}")
    
    # 7. Save model and metadata
    model_bundle = {
        "model": clf,
        "best_threshold": float(best_thresh),
        "validation_f05": float(best_f05),
        "blocking_recall": float(blocking_recall),
        "max_candidates": MAX_CANDIDATES,
        "max_bucket": MAX_BUCKET,
    }
    
    model_save_path = os.path.join(MODELS_DIR, "baseline_matcher.joblib")
    joblib.dump(model_bundle, model_save_path)
    print(f"Model saved to: {model_save_path}")
    print(f"Total time elapsed: {time.time() - t_start:.2f}s")
    return model_bundle

if __name__ == '__main__':
    train_and_evaluate(sample_s1_count=40000)
