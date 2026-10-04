"""
Amazon Business Entity Resolution — Version 3 Feature Extraction & Blocking
Includes:
- Brahmi ISCII Multi-Script Phonetic Transliteration (Devanagari, Bengali, Gurmukhi, Malayalam, etc.)
- Dual Script Blocking & Feature Computation (Raw Script + Transliterated)
- Co-Located Commercial Complex Distractor Penalties
- Building/House-Number Conflict Handling with Strong-Name Exceptions
"""

import os, sys, re, unicodedata
from collections import defaultdict, Counter
import numpy as np

# Reconfigure stdout if available
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Comprehensive Legal and Generic Terms (US, India, France)
LEGAL_SUFFIXES = {
    # US / UK / India
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'enterprises', 'solutions',
    'technologies', 'consulting', 'group', 'holdings', 'industries', 'international',
    'trust', 'foundation', 'associates', 'partners', 'llp', 'plc', 'gmbh',
    # France / Europe
    'sarl', 'sasu', 'eurl', 'sa', 'snc', 'sci', 'scs', 'selarl', 'gie',
    'association', 'ste', 'societe', 'cie', 'compagnie', 'ets', 'etablissements',
    # Transliterated Indic legal suffixes
    'praivet', 'privet', 'piraivet', 'limittd', 'limitet', 'praivtt'
}

GENERIC_WORDS = {
    # English
    'club', 'centre', 'center', 'school', 'academy', 'institute', 'hotel', 'restaurant',
    'bar', 'cafe', 'shop', 'store', 'market', 'agency', 'studio', 'salon', 'clinic',
    'hospital', 'pharmacy', 'lab', 'labs', 'services', 'solutions', 'enterprises',
    # French
    'ecole', 'groupe', 'agence', 'atelier', 'cabinet', 'bureau', 'boutique', 'magasin',
    'auto', 'garage'
}

COMMON_STOPWORDS = {
    # English
    'the', 'and', 'of', 'in', 'for', 'at', 'on', 'to', 'a', 'an', 'by', 'with',
    # French
    'de', 'la', 'le', 'et', 'du', 'des', 'les', 'au', 'aux', 'en', 'sur', 'pour',
    'par', 'dans', 'avec', 'd', 'l'
}

COMMON_ADDR_STOPWORDS = {
    'road', 'street', 'floor', 'near', 'behind', 'opp', 'opposite', 'dist', 'nagar',
    'city', 'state', 'india', 'pune', 'delhi', 'mumbai', 'kolkata', 'chennai',
    'bangalore', 'hyderabad', 'block', 'plot', 'sector', 'building', 'colony',
    'rue', 'avenue', 'boulevard', 'france', 'paris', 'lyon', 'marseille', 'bordeaux',
    'nantes', 'dunkerque', 'lille', 'toulouse', 'strasbourg', 'montpellier'
}

# Brahmi ISCII relative offset mapping (0x00 to 0x7F)
# Shared across Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
BRAHMI_OFFSETS = {
    # Independent vowels
    0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0A: 'u',
    0x0B: 'ri', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x12: 'o', 0x13: 'o', 0x14: 'au',
    # Consonants
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh', 0x1E: 'ny',
    0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n', 0x29: 'n',
    0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm',
    0x2F: 'y', 0x30: 'r', 0x31: 'r', 0x32: 'l', 0x33: 'l', 0x34: 'l',
    0x35: 'v', 0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    # Vowel signs (matras)
    0x3E: 'a', 0x3F: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u', 0x43: 'ri',
    0x46: 'e', 0x47: 'e', 0x48: 'ai', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au',
    # Additional signs
    0x02: 'n', 0x03: 'h', 0x4D: ''
}

def transliterate_indic(text):
    if not text: return ""
    text = text.replace('\u0D31\u0D4D\u0D31', 'tt')
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            out.append(BRAHMI_OFFSETS.get(cp & 0x7F, ''))
        else:
            out.append(ch)
    return ''.join(out)

def clean_name_words_v3(text):
    if not text: return [], [], [], []
    text_norm = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)).lower()
    text_norm = re.sub(r'\.(com|org|net|in|co|io|fr|biz|info)\b', '', text_norm)
    text_norm = re.sub(r'[\(\[\{][^\)\]\}]*[\)\]\}]', ' ', text_norm)
    text_norm = re.sub(r'\b(formerly|dba|mr|mrs|ms)\b', ' ', text_norm)
    
    # 1. Raw words
    raw_clean = re.sub(r'[^a-z0-9\u0900-\u097f]+', ' ', text_norm)
    raw_words = []
    raw_dist = []
    for w in raw_clean.split():
        if w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS and len(w) > 1:
            raw_words.append(w)
            if w not in GENERIC_WORDS:
                raw_dist.append(w)
                
    # 2. Transliterated words
    has_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in text)
    trans_words = []
    trans_dist = []
    if has_indic:
        trans_text = transliterate_indic(text_norm)
        trans_clean = re.sub(r'[^a-z0-9]+', ' ', trans_text)
        for w in trans_clean.split():
            if w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS and len(w) > 1:
                trans_words.append(w)
                if w not in GENERIC_WORDS:
                    trans_dist.append(w)
    else:
        trans_words = raw_words
        trans_dist = raw_dist
        
    return raw_words, raw_dist, trans_words, trans_dist

def clean_address_tokens_v3(addr):
    if not addr: return [], set()
    text = ''.join(c for c in unicodedata.normalize('NFKD', addr) if not unicodedata.combining(c)).lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    words = []
    nums = set()
    for t in text.split():
        if t.isdigit():
            norm_num = str(int(t))
            if len(norm_num) >= 2:
                nums.add(norm_num)
        elif len(t) >= 3 and t not in COMMON_ADDR_STOPWORDS and t not in COMMON_STOPWORDS:
            words.append(t)
    return words, nums

def get_char_ngrams_v3(text, n=3):
    clean = re.sub(r'[^a-z0-9\u0900-\u097f]', '', text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def get_translit_char_ngrams_v3(text, n=3):
    has_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in text)
    if has_indic:
        text = transliterate_indic(text)
    clean = re.sub(r'[^a-z0-9]', '', text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def extract_address_keys_v3(addr):
    if not addr: return set()
    text = ''.join(c for c in unicodedata.normalize('NFKD', addr) if not unicodedata.combining(c)).lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    tokens = text.split()
    nums = [str(int(t)) for t in tokens if t.isdigit() and len(str(int(t))) >= 2]
    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in COMMON_ADDR_STOPWORDS]
    
    keys = set()
    if nums and words:
        keys.add(f'an_{nums[0]}_{words[0]}')
        if len(words) >= 2:
            keys.add(f'an_{nums[0]}_{words[1]}')
    if len(words) >= 2:
        keys.add(f'aw_{words[0]}_{words[1]}')
        keys.add(f'aw_sort_{sorted(words[:2])[0]}_{sorted(words[:2])[1]}')
    return keys

def consonant_skeleton(word):
    if not word: return ''
    clean = re.sub(r'[^a-z0-9]', '', word.lower())
    if not clean: return ''
    clean = clean.replace('ph', 'f').replace('v', 'w').replace('c', 'k').replace('x', 'ks')
    first = clean[0]
    rest = re.sub(r'[aeiouy]', '', clean[1:])
    return first + rest

def generate_blocking_keys_v3(name, addr):
    raw_w, _, trans_w, _ = clean_name_words_v3(name)
    keys = set()
    
    # 1. Raw script keys
    if raw_w:
        condensed = ''.join(raw_w)
        if len(condensed) >= 4:
            keys.add('cond_' + condensed[:14])
        sorted_words = sorted(raw_w)
        keys.add('sort2_' + '_'.join(sorted_words[:2]))
        if len(sorted_words) >= 3:
            keys.add('sort3_' + '_'.join(sorted_words[:3]))
        if len(raw_w[0]) >= 4:
            keys.add('w0_' + raw_w[0])
        if len(raw_w) >= 2 and len(raw_w[1]) >= 4:
            keys.add('w1_' + raw_w[1])
            
    # 2. Transliterated keys & Phonetic Consonant Skeletons
    active_words = trans_w if trans_w else raw_w
    if active_words:
        if trans_w and trans_w != raw_w:
            t_condensed = ''.join(trans_w)
            if len(t_condensed) >= 4:
                keys.add('cond_' + t_condensed[:14])
            t_sorted = sorted(trans_w)
            keys.add('sort2_' + '_'.join(t_sorted[:2]))
            if len(t_sorted) >= 3:
                keys.add('sort3_' + '_'.join(t_sorted[:3]))
            if len(trans_w[0]) >= 4:
                keys.add('w0_' + trans_w[0])
            if len(trans_w) >= 2 and len(trans_w[1]) >= 4:
                keys.add('w1_' + trans_w[1])
                
        # Phonetic skeletons (invariant to vowel representations)
        sk0 = consonant_skeleton(active_words[0])
        sk1 = consonant_skeleton(active_words[1]) if len(active_words) >= 2 else ''
        if len(sk0) >= 3 and len(sk1) >= 3:
            s_pair = sorted([sk0[:5], sk1[:5]])
            keys.add(f'sk_sort_{s_pair[0]}_{s_pair[1]}')
        if len(sk0) >= 4:
            keys.add(f'sk0_{sk0[:7]}')
            
    # 3. Address keys
    addr_keys = extract_address_keys_v3(addr)
    keys.update(addr_keys)
    return keys

def compute_pair_features_v3(s1_raw_w, s1_raw_dw, s1_tr_w, s1_tr_dw, s1_raw_ng, s1_tr_ng,
                             s1_aw, s1_nums, has_s1_a,
                             t_raw_w, t_raw_dw, t_tr_w, t_tr_dw, t_raw_ng, t_tr_ng,
                             t_aw, t_nums, has_t_a,
                             shared_keys):
    """
    Computes 18 features capturing:
    1. Raw script name similarities
    2. Transliterated script name similarities
    3. Maximum/distinctive name agreement
    4. Address similarity & house-number compatibility
    5. Commercial co-location distractor flag
    """
    # 1. Raw name token Jaccard & Containment
    s1_w_set = set(s1_raw_w)
    t_w_set = set(t_raw_w)
    inter_raw_w = s1_w_set & t_w_set
    union_raw_w = s1_w_set | t_w_set
    raw_name_jaccard = len(inter_raw_w) / len(union_raw_w) if union_raw_w else 0.0
    min_raw_len = min(len(s1_w_set), len(t_w_set))
    raw_name_containment = len(inter_raw_w) / min_raw_len if min_raw_len > 0 else 0.0
    
    # 2. Raw distinctive name Jaccard
    s1_dw_set = set(s1_raw_dw)
    t_dw_set = set(t_raw_dw)
    inter_raw_dw = s1_dw_set & t_dw_set
    union_raw_dw = s1_dw_set | t_dw_set
    raw_dist_name_jaccard = len(inter_raw_dw) / len(union_raw_dw) if union_raw_dw else 0.0
    
    # 3. Raw char n-gram Jaccard
    inter_raw_ng = s1_raw_ng & t_raw_ng
    union_raw_ng = s1_raw_ng | t_raw_ng
    raw_char_jaccard = len(inter_raw_ng) / len(union_raw_ng) if union_raw_ng else 0.0
    
    # 4. Transliterated name similarities
    s1_tr_w_set = set(s1_tr_w)
    t_tr_w_set = set(t_tr_w)
    inter_tr_w = s1_tr_w_set & t_tr_w_set
    union_tr_w = s1_tr_w_set | t_tr_w_set
    trans_name_jaccard = len(inter_tr_w) / len(union_tr_w) if union_tr_w else 0.0
    min_tr_len = min(len(s1_tr_w_set), len(t_tr_w_set))
    trans_name_containment = len(inter_tr_w) / min_tr_len if min_tr_len > 0 else 0.0
    
    s1_tr_dw_set = set(s1_tr_dw)
    t_tr_dw_set = set(t_tr_dw)
    inter_tr_dw = s1_tr_dw_set & t_tr_dw_set
    union_tr_dw = s1_tr_dw_set | t_tr_dw_set
    trans_dist_name_jaccard = len(inter_tr_dw) / len(union_tr_dw) if union_tr_dw else 0.0
    
    inter_tr_ng = s1_tr_ng & t_tr_ng
    union_tr_ng = s1_tr_ng | t_tr_ng
    trans_char_jaccard = len(inter_tr_ng) / len(union_tr_ng) if union_tr_ng else 0.0
    
    # 5. Best overall name similarities across representations
    best_dist_name_jaccard = max(raw_dist_name_jaccard, trans_dist_name_jaccard)
    best_containment = max(raw_name_containment, trans_name_containment)
    best_char_jaccard = max(raw_char_jaccard, trans_char_jaccard)
    
    # Condensed match
    c1 = ''.join(s1_tr_w)
    c2 = ''.join(t_tr_w)
    condensed_match = 1.0 if (c1 and c2 and (c1 == c2 or c1.startswith(c2) or c2.startswith(c1))) else 0.0
    
    # 6. Address & House Number Features
    both_addr = 1.0 if (has_s1_a and has_t_a) else 0.0
    addr_jaccard = 0.0
    num_match = 0.5
    num_conflict = 0.0
    num_conflict_with_low_name = 0.0
    addr_strong_match = 0.0
    colocated_distractor = 0.0
    
    if both_addr:
        s1_a_set = set(s1_aw)
        t_a_set = set(t_aw)
        inter_addr = s1_a_set & t_a_set
        union_addr = s1_a_set | t_a_set
        addr_jaccard = len(inter_addr) / len(union_addr) if union_addr else 0.0
        
        # House numbers
        if s1_nums and t_nums:
            if s1_nums & t_nums:
                num_match = 1.0
                num_conflict = 0.0
            else:
                num_match = 0.0
                num_conflict = 1.0
                # Conflicting numbers with non-identical names
                if max(best_dist_name_jaccard, best_containment, condensed_match) < 0.80:
                    num_conflict_with_low_name = 1.0
        else:
            num_match = 0.5
            num_conflict = 0.0
            
        if num_match == 1.0 and addr_jaccard >= 0.35:
            addr_strong_match = 1.0
            
        # Co-located distractor: High address match, but distinctive business names do not match!
        if addr_jaccard >= 0.35 and best_dist_name_jaccard < 0.25 and best_char_jaccard < 0.30:
            colocated_distractor = 1.0
            
    return [
        raw_name_jaccard,               # 0
        raw_dist_name_jaccard,          # 1
        raw_char_jaccard,               # 2
        trans_name_jaccard,             # 3
        trans_name_containment,         # 4
        trans_dist_name_jaccard,        # 5
        trans_char_jaccard,             # 6
        best_dist_name_jaccard,         # 7
        best_char_jaccard,              # 8
        condensed_match,                # 9
        both_addr,                      # 10
        addr_jaccard,                   # 11
        num_match,                      # 12
        num_conflict,                   # 13
        num_conflict_with_low_name,     # 14
        addr_strong_match,              # 15
        colocated_distractor,           # 16
        float(shared_keys)              # 17
    ]

def evaluate_macro_f05_v3(gt_dict, pred_dict, s1_all):
    total_f05 = 0.0
    total_tp, total_fp, total_fn = 0, 0, 0
    singletons, sing_errors = 0, 0
    
    for s1_id in s1_all:
        true_set = gt_dict.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())
        if not true_set:
            singletons += 1
            if pred_set:
                sing_errors += 1
                total_fp += len(pred_set)
                f05 = 0.0
            else:
                f05 = 1.0
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            if not pred_set:
                f05 = 0.0
            else:
                p = tp / len(pred_set)
                r = tp / len(true_set)
                denom = 0.25 * p + r
                f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
        total_f05 += f05
    macro = total_f05 / len(s1_all) if s1_all else 0.0
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    return macro, p, r, total_fp, total_fn, sing_errors, singletons

print("V3 Feature and blocking definitions loaded.")
