"""
Experiment 2: Synthetic Normalization & Identifier Extraction Features
Includes:
- Synthetic House-Number Truncation/Extension Normalization (932 <-> 9327, 609 <-> 1609)
- Soft Building Number Compatibility vs Hard Conflict Separation
- Trailing 10-Digit Phone / Numeric Identifier Extraction from Entity Names
- Separate Cleaned Normalized-Name Feature Space while Preserving Raw Names
- Phone Exact Match and Phone Conflict Features
- Brahmi Multi-Script Phonetic Transliteration & Phonetic Consonant Skeletons
"""

import os, sys, re, unicodedata
from collections import defaultdict, Counter
import numpy as np

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

# -------------------------------------------------------------
# EXPERIMENT 2 COMPONENT A: HOUSE NUMBER NORMALIZATION
# -------------------------------------------------------------

def extract_house_numbers(addr):
    """
    Extracts house/building numbers from address.
    Preserves:
    - orig_nums: set of original numeric tokens
    - norm_nums: set of normalized integer string representations (length >= 2)
    """
    if not addr: return set(), set()
    text = ''.join(c for c in unicodedata.normalize('NFKD', addr) if not unicodedata.combining(c)).lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    tokens = text.split()
    orig_nums = set()
    norm_nums = set()
    for t in tokens:
        if t.isdigit():
            orig_nums.add(t)
            v = str(int(t))
            if len(v) >= 2:
                norm_nums.add(v)
    return orig_nums, norm_nums

def compute_house_number_soft_compat(s1_nums, t_nums, max_name_sim=0.0):
    """
    Evaluates house number relationship:
    - Detects prefix/suffix truncation & extension (932 <-> 9327, 609 <-> 1609)
    - Distinguishes exact match, soft compatibility, and hard conflict
    
    Returns:
      num_exact_match (float): 1.0 if s1_nums & t_nums, 0.5 if either empty, 0.0 otherwise
      num_soft_compat (float): 1.0 if exact OR prefix/suffix match (|len diff| <= 2), 0.5 if empty, 0.0 otherwise
      num_hard_conflict (float): 1.0 if both non-empty and NO soft compatibility, 0.0 otherwise
      num_conflict_with_low_name (float): 1.0 if num_hard_conflict == 1.0 and max_name_sim < 0.80, 0.0 otherwise
    """
    if not s1_nums or not t_nums:
        return 0.5, 0.5, 0.0, 0.0
    
    if s1_nums & t_nums:
        return 1.0, 1.0, 0.0, 0.0
        
    # Check prefix / suffix truncation/extension
    for n1 in s1_nums:
        for n2 in t_nums:
            if len(n1) >= 2 and len(n2) >= 2:
                # Prefix match: 932 vs 9327, 45 vs 459
                if (n1.startswith(n2) or n2.startswith(n1)) and abs(len(n1) - len(n2)) <= 2:
                    return 0.0, 1.0, 0.0, 0.0
                # Suffix match: 609 vs 1609, 15 vs 215
                if (n1.endswith(n2) or n2.endswith(n1)) and abs(len(n1) - len(n2)) <= 2:
                    return 0.0, 1.0, 0.0, 0.0
                    
    # Neither exact nor soft compatible -> Hard conflict
    num_conflict_low_name = 1.0 if max_name_sim < 0.80 else 0.0
    return 0.0, 0.0, 1.0, num_conflict_low_name


# -------------------------------------------------------------
# EXPERIMENT 2 COMPONENT B: TRAILING PHONE EXTRACTION
# -------------------------------------------------------------

def extract_trailing_phone(name):
    """
    Detects trailing 10-digit phone number in entity names.
    Preserves original name, returning (clean_name, phone_digits_or_None).
    """
    if not name: return name, None
    # 1. Match trailing 10 digits with separator (e.g. " - 7087928842", " / 7087928842")
    m = re.search(r'(?:[-–—\s/|\\]+)\b(\d{10})\b\s*$', name)
    if m:
        phone = m.group(1)
        clean_name = name[:m.start()].strip()
        return clean_name, phone
        
    # 2. Match trailing 10 digits at word boundary
    m2 = re.search(r'\b(\d{10})\b\s*$', name)
    if m2:
        phone = m2.group(1)
        clean_name = name[:m2.start()].strip()
        return clean_name, phone
        
    return name, None

def compute_phone_features(s1_phone, t_phone):
    """
    Computes phone agreement features:
    - phone_exact_match: 1.0 if both present and equal, 0.0 otherwise
    - phone_conflict: 1.0 if both present and different, 0.0 otherwise
    - phone_present_either: 1.0 if either has phone, 0.0 otherwise
    """
    if s1_phone and t_phone:
        if s1_phone == t_phone:
            return 1.0, 0.0, 1.0
        else:
            return 0.0, 1.0, 1.0
    elif s1_phone or t_phone:
        return 0.0, 0.0, 1.0
    else:
        return 0.0, 0.0, 0.0


# -------------------------------------------------------------
# TEXT CLEANING & TOKENIZATION
# -------------------------------------------------------------

def clean_name_words(text):
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

def clean_address_tokens(addr):
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

def get_char_ngrams(text, n=3):
    clean = re.sub(r'[^a-z0-9\u0900-\u097f]', '', text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def get_translit_char_ngrams(text, n=3):
    has_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in text)
    if has_indic:
        text = transliterate_indic(text)
    clean = re.sub(r'[^a-z0-9]', '', text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def consonant_skeleton(word):
    if not word: return ''
    clean = re.sub(r'[^a-z0-9]', '', word.lower())
    if not clean: return ''
    clean = clean.replace('ph', 'f').replace('v', 'w').replace('c', 'k').replace('x', 'ks')
    first = clean[0]
    rest = re.sub(r'[aeiouy]', '', clean[1:])
    return first + rest

# -------------------------------------------------------------
# MODULAR FEATURE EXTRACTOR FOR CONFIGURATIONS 1, 2, 3, 4
# -------------------------------------------------------------

def compute_pair_features_exp2(
    s1_name_raw, s1_addr,
    t_name_raw, t_addr,
    shared_keys,
    use_house_norm=True,
    use_phone_norm=True
):
    """
    Computes pair features with configurable:
    - use_house_norm: enables soft compatibility and prefix/suffix truncation features
    - use_phone_norm: strips trailing 10-digit phone for name features, adds phone match/conflict features
    """
    # 1. Phone extraction
    if use_phone_norm:
        s1_clean_name, s1_phone = extract_trailing_phone(s1_name_raw)
        t_clean_name, t_phone = extract_trailing_phone(t_name_raw)
    else:
        s1_clean_name, s1_phone = s1_name_raw, None
        t_clean_name, t_phone = t_name_raw, None
        
    # 2. Tokenize names
    s1_raw_w, s1_raw_dw, s1_tr_w, s1_tr_dw = clean_name_words(s1_clean_name)
    s1_raw_ng = get_char_ngrams(s1_clean_name)
    s1_tr_ng = get_translit_char_ngrams(s1_clean_name)
    
    t_raw_w, t_raw_dw, t_tr_w, t_tr_dw = clean_name_words(t_clean_name)
    t_raw_ng = get_char_ngrams(t_clean_name)
    t_tr_ng = get_translit_char_ngrams(t_clean_name)
    
    # 3. Tokenize addresses
    s1_aw, s1_nums = clean_address_tokens(s1_addr)
    t_aw, t_nums = clean_address_tokens(t_addr)
    has_s1_a = bool(s1_addr.strip())
    has_t_a = bool(t_addr.strip())
    
    # 4. Name similarity metrics
    s1_w_set = set(s1_raw_w)
    t_w_set = set(t_raw_w)
    inter_raw_w = s1_w_set & t_w_set
    union_raw_w = s1_w_set | t_w_set
    raw_name_jaccard = len(inter_raw_w) / len(union_raw_w) if union_raw_w else 0.0
    min_raw_len = min(len(s1_w_set), len(t_w_set))
    raw_name_containment = len(inter_raw_w) / min_raw_len if min_raw_len > 0 else 0.0
    
    s1_dw_set = set(s1_raw_dw)
    t_dw_set = set(t_raw_dw)
    inter_raw_dw = s1_dw_set & t_dw_set
    union_raw_dw = s1_dw_set | t_dw_set
    raw_dist_name_jaccard = len(inter_raw_dw) / len(union_raw_dw) if union_raw_dw else 0.0
    
    inter_raw_ng = s1_raw_ng & t_raw_ng
    union_raw_ng = s1_raw_ng | t_raw_ng
    raw_char_jaccard = len(inter_raw_ng) / len(union_raw_ng) if union_raw_ng else 0.0
    
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
    
    best_dist_name_jaccard = max(raw_dist_name_jaccard, trans_dist_name_jaccard)
    best_containment = max(raw_name_containment, trans_name_containment)
    best_char_jaccard = max(raw_char_jaccard, trans_char_jaccard)
    
    c1 = ''.join(s1_tr_w)
    c2 = ''.join(t_tr_w)
    condensed_match = 1.0 if (c1 and c2 and (c1 == c2 or c1.startswith(c2) or c2.startswith(c1))) else 0.0
    
    # 5. Address similarity
    both_addr = 1.0 if (has_s1_a and has_t_a) else 0.0
    addr_jaccard = 0.0
    colocated_distractor = 0.0
    
    if both_addr:
        s1_a_set = set(s1_aw)
        t_a_set = set(t_aw)
        inter_addr = s1_a_set & t_a_set
        union_addr = s1_a_set | t_a_set
        addr_jaccard = len(inter_addr) / len(union_addr) if union_addr else 0.0
        
        # Co-located distractor
        if addr_jaccard >= 0.35 and best_dist_name_jaccard < 0.25 and best_char_jaccard < 0.30:
            colocated_distractor = 1.0
            
    # 6. House number features
    max_name_sim = max(best_dist_name_jaccard, best_containment, condensed_match)
    
    if use_house_norm:
        ex_match, soft_compat, hard_conf, conf_low_name = compute_house_number_soft_compat(s1_nums, t_nums, max_name_sim)
        addr_strong_match = 1.0 if (soft_compat >= 1.0 and addr_jaccard >= 0.35) else 0.0
    else:
        # V3 baseline logic
        if s1_nums and t_nums:
            if s1_nums & t_nums:
                ex_match = 1.0
                soft_compat = 1.0
                hard_conf = 0.0
                conf_low_name = 0.0
            else:
                ex_match = 0.0
                soft_compat = 0.0
                hard_conf = 1.0
                conf_low_name = 1.0 if max_name_sim < 0.80 else 0.0
        else:
            ex_match = 0.5
            soft_compat = 0.5
            hard_conf = 0.0
            conf_low_name = 0.0
        addr_strong_match = 1.0 if (ex_match == 1.0 and addr_jaccard >= 0.35) else 0.0
        
    # 7. Phone features
    phone_exact, phone_conf, phone_present = compute_phone_features(s1_phone, t_phone)
    
    # Assemble feature vector
    feats = [
        raw_name_jaccard,           # 0
        raw_dist_name_jaccard,      # 1
        raw_char_jaccard,           # 2
        trans_name_jaccard,         # 3
        trans_name_containment,     # 4
        trans_dist_name_jaccard,    # 5
        trans_char_jaccard,         # 6
        best_dist_name_jaccard,     # 7
        best_char_jaccard,          # 8
        condensed_match,            # 9
        both_addr,                  # 10
        addr_jaccard,               # 11
        ex_match,                   # 12
        soft_compat,                # 13 (Experiment 2 addition)
        hard_conf,                  # 14
        conf_low_name,              # 15
        addr_strong_match,          # 16
        colocated_distractor,       # 17
        phone_exact,                # 18 (Experiment 2 addition)
        phone_conf,                 # 19 (Experiment 2 addition)
        float(shared_keys)          # 20
    ]
    
    return feats

# Feature index mapping for transparency
FEAT_NAMES = [
    "raw_name_jaccard",
    "raw_dist_name_jaccard",
    "raw_char_jaccard",
    "trans_name_jaccard",
    "trans_name_containment",
    "trans_dist_name_jaccard",
    "trans_char_jaccard",
    "best_dist_name_jaccard",
    "best_char_jaccard",
    "condensed_match",
    "both_addr",
    "addr_jaccard",
    "num_exact_match",
    "num_soft_compat",
    "num_hard_conflict",
    "num_conflict_with_low_name",
    "addr_strong_match",
    "colocated_distractor",
    "phone_exact_match",
    "phone_conflict",
    "shared_keys"
]

print(f"Loaded exp2_features with {len(FEAT_NAMES)} features defined.")
