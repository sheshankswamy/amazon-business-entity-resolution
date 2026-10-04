"""
Production Feature Engineering Module (37 Pairwise Features)
Amazon Business Entity Resolution Hackathon (ML Challenge 2026)

Includes:
- RapidFuzz Levenshtein, Jaro-Winkler, Token Sort, Token Set ratios (Names & Addresses)
- Raw & Transliterated Indic representations
- Character n-gram similarities & normalized length differences
- House number exact match, soft compatibility & conflict detection
- Geographic compatibility: PIN, PIN3, City, State
- Cross-attribute multiplicative & min/max interaction terms
- Retrieval signals: rank, reciprocal rank, channel votes, channel agreement
"""

import os, re, unicodedata
from rapidfuzz import fuzz, distance
import numpy as np

# Common stop words & legal suffixes
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'enterprises', 'solutions',
    'technologies', 'consulting', 'group', 'holdings', 'industries', 'international',
    'trust', 'foundation', 'associates', 'partners', 'llp', 'plc', 'gmbh',
    'sarl', 'sasu', 'eurl', 'sa', 'snc', 'sci', 'scs', 'selarl', 'gie',
    'association', 'ste', 'societe', 'cie', 'compagnie', 'ets', 'etablissements',
    'praivet', 'privet', 'piraivet', 'limittd', 'limitet', 'praivtt'
}

COMMON_STOPWORDS = {
    'the', 'and', 'of', 'in', 'for', 'at', 'on', 'to', 'a', 'an', 'by', 'with',
    'de', 'la', 'le', 'et', 'du', 'des', 'les', 'au', 'aux', 'en', 'sur', 'pour',
    'par', 'dans', 'avec', 'd', 'l'
}

INDIAN_STATES = {
    'maharashtra', 'delhi', 'karnataka', 'tamilnadu', 'gujarat', 'uttarpradesh', 'westbengal',
    'telangana', 'andhrapradesh', 'rajasthan', 'kerala', 'madhyapradesh', 'punjab', 'haryana',
    'bihar', 'odisha', 'orissa', 'assam', 'jharkhand', 'chhattisgarh', 'uttarakhand', 'goa',
    'himachalpradesh', 'tripura', 'manipur', 'meghalaya', 'nagaland', 'puducherry', 'chandigarh'
}

INDIAN_CITIES = {
    'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'kolkata', 'pune',
    'ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal',
    'visakhapatnam', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad',
    'meerut', 'rajkot', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad', 'amritsar', 'navi mumbai',
    'allahabad', 'prayagraj', 'ranchi', 'howrah', 'coimbatore', 'jabalpur', 'gwalior', 'vijayawada',
    'jodhpur', 'madurai', 'raipur', 'kota', 'chandigarh', 'guwahati', 'solapur', 'hubli', 'mysore'
}

BRAHMI_OFFSETS = {
    0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0A: 'u',
    0x0B: 'ri', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x12: 'o', 0x13: 'o', 0x14: 'au',
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh', 0x1E: 'ny',
    0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
    0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm',
    0x2F: 'y', 0x30: 'r', 0x32: 'l', 0x33: 'l', 0x35: 'v',
    0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    0x3E: 'a', 0x3F: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u', 0x43: 'ri',
    0x47: 'e', 0x48: 'ai', 0x4B: 'o', 0x4C: 'au',
    0x02: 'm', 0x03: 'h', 0x3D: '', 0x4D: ''
}

def transliterate_indic(text):
    if not text: return ""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            offset = cp % 0x80
            out.append(BRAHMI_OFFSETS.get(offset, ch))
        else:
            out.append(ch)
    return ''.join(out)

def get_char_ngrams(text, n=3):
    clean = re.sub(r'[^a-z0-9]+', '', text.lower())
    if len(clean) < n: return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def extract_house_numbers(addr):
    if not addr: return set()
    raw_tokens = re.findall(r'\b\d+[a-z]?\b|\b\d+[-/]\d+\b', addr.lower())
    nums = set()
    for tok in raw_tokens:
        sub = re.findall(r'\d+', tok)
        for s in sub:
            n_int = int(s)
            if 1 <= n_int <= 9999 and not (len(s) == 6 and s.startswith(('1','2','3','4','5','6','7','8','9'))):
                nums.add(s)
    return nums

def extract_india_geo_full(addr):
    if not addr: return "", "", "", "", []
    text = ''.join(c for c in unicodedata.normalize('NFKD', addr) if not unicodedata.combining(c)).lower()
    
    # PIN
    pin = ""
    m_pin = re.search(r'\b([1-9]\d{5})\b', text)
    if m_pin: pin = m_pin.group(1)
    pin3 = pin[:3] if pin else ""
    
    # State
    clean_nospace = text.replace(' ', '')
    state = ""
    for s in INDIAN_STATES:
        if s in clean_nospace:
            state = s
            break
            
    # City
    city = ""
    for c in INDIAN_CITIES:
        if re.search(r'\b' + re.escape(c) + r'\b', text):
            city = c
            break
            
    tokens = re.sub(r'[^a-z0-9]+', ' ', text).split()
    dist_addr = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t != state and t != city]
    return pin, pin3, state, city, dist_addr

FEATURE_NAMES = [
    # NAME (11)
    "name_ratio",
    "name_levenshtein",
    "name_jw",
    "name_token_sort",
    "name_token_set",
    "name_token_overlap",
    "name_jaccard",
    "name_char_ngram",
    "name_len_diff",
    "name_translit_jw",
    "best_name_sim",
    # ADDRESS (16)
    "both_addr_present",
    "addr_ratio",
    "addr_levenshtein",
    "addr_jw",
    "addr_token_sort",
    "addr_token_set",
    "addr_token_overlap",
    "addr_jaccard",
    "addr_char_ngram",
    "house_num_exact",
    "house_num_soft",
    "house_num_conflict",
    "pin_match",
    "pin3_match",
    "city_match",
    "state_match",
    # CROSS-ATTRIBUTE (6)
    "name_x_addr",
    "name_plus_addr",
    "min_name_addr",
    "max_name_addr",
    "weighted_combined",
    "colocated_distractor",
    # RETRIEVAL (4)
    "cand_rank",
    "reciprocal_rank",
    "channel_votes",
    "channel_agreement"
]

def compute_pairwise_features(s1_name, s1_addr, t_name, t_addr, rank=1, votes=1, is_india=True):
    """
    Computes 37 validated pairwise features between Source 1 and Target entity.
    """
    # ---------------------------------------------------------
    # 1. NAME SIMILARITIES
    # ---------------------------------------------------------
    s1_n_clean = re.sub(r'[\(\[\{][^\)\]\}]*[\)\]\}]', ' ', s1_name).lower().strip()
    t_n_clean = re.sub(r'[\(\[\{][^\)\]\}]*[\)\]\}]', ' ', t_name).lower().strip()
    
    # RapidFuzz standard metrics
    name_ratio = fuzz.ratio(s1_n_clean, t_n_clean) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_n_clean, t_n_clean) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_n_clean, t_n_clean) / 100.0
    name_jw = distance.JaroWinkler.similarity(s1_n_clean, t_n_clean)
    
    # Normalized Levenshtein
    max_len = max(len(s1_n_clean), len(t_n_clean), 1)
    name_lev = 1.0 - (distance.Levenshtein.distance(s1_n_clean, t_n_clean) / max_len)
    
    # Token Overlap & Jaccard
    s1_w = [w for w in re.sub(r'[^a-z0-9]+', ' ', s1_n_clean).split() if w not in COMMON_STOPWORDS and w not in LEGAL_SUFFIXES]
    t_w = [w for w in re.sub(r'[^a-z0-9]+', ' ', t_n_clean).split() if w not in COMMON_STOPWORDS and w not in LEGAL_SUFFIXES]
    s1_wset = set(s1_w)
    t_wset = set(t_w)
    
    if s1_wset and t_wset:
        inter = len(s1_wset & t_wset)
        name_token_overlap = inter / min(len(s1_wset), len(t_wset))
        name_jaccard = inter / len(s1_wset | t_wset)
    else:
        name_token_overlap = name_jaccard = 0.0
        
    # Character n-grams
    s1_ng = get_char_ngrams(s1_n_clean, n=3)
    t_ng = get_char_ngrams(t_n_clean, n=3)
    if s1_ng and t_ng:
        name_char_ngram = len(s1_ng & t_ng) / len(s1_ng | t_ng)
    else:
        name_char_ngram = 0.0
        
    name_len_diff = abs(len(s1_n_clean) - len(t_n_clean)) / max_len
    
    # Transliteration similarity
    if is_india and (any(0x0900 <= ord(c) <= 0x0D7F for c in s1_name) or any(0x0900 <= ord(c) <= 0x0D7F for c in t_name)):
        s1_tr = transliterate_indic(s1_n_clean)
        t_tr = transliterate_indic(t_n_clean)
        name_translit_jw = distance.JaroWinkler.similarity(s1_tr, t_tr)
    else:
        name_translit_jw = name_jw
        
    best_name_sim = max(name_jw, name_translit_jw, name_token_set)
    
    # ---------------------------------------------------------
    # 2. ADDRESS SIMILARITIES
    # ---------------------------------------------------------
    s1_a_clean = s1_addr.lower().strip() if s1_addr else ""
    t_a_clean = t_addr.lower().strip() if t_addr else ""
    both_addr_present = 1.0 if (s1_a_clean and t_a_clean) else 0.0
    
    if both_addr_present:
        addr_ratio = fuzz.ratio(s1_a_clean, t_a_clean) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1_a_clean, t_a_clean) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_a_clean, t_a_clean) / 100.0
        addr_jw = distance.JaroWinkler.similarity(s1_a_clean, t_a_clean)
        
        a_max_len = max(len(s1_a_clean), len(t_a_clean), 1)
        addr_lev = 1.0 - (distance.Levenshtein.distance(s1_a_clean, t_a_clean) / a_max_len)
        
        s1_awset = set(re.sub(r'[^a-z0-9]+', ' ', s1_a_clean).split())
        t_awset = set(re.sub(r'[^a-z0-9]+', ' ', t_a_clean).split())
        if s1_awset and t_awset:
            a_inter = len(s1_awset & t_awset)
            addr_token_overlap = a_inter / min(len(s1_awset), len(t_awset))
            addr_jaccard = a_inter / len(s1_awset | t_awset)
        else:
            addr_token_overlap = addr_jaccard = 0.0
            
        s1_ang = get_char_ngrams(s1_a_clean, n=3)
        t_ang = get_char_ngrams(t_a_clean, n=3)
        addr_char_ngram = len(s1_ang & t_ang) / len(s1_ang | t_ang) if (s1_ang and t_ang) else 0.0
    else:
        addr_ratio = addr_lev = addr_jw = addr_token_sort = addr_token_set = 0.0
        addr_token_overlap = addr_jaccard = addr_char_ngram = 0.0
        
    # House Numbers
    s1_nums = extract_house_numbers(s1_addr)
    t_nums = extract_house_numbers(t_addr)
    house_num_exact = house_num_soft = house_num_conflict = 0.0
    
    if s1_nums and t_nums:
        if s1_nums & t_nums:
            house_num_exact = 1.0
            house_num_soft = 1.0
        else:
            # Check prefix/suffix truncation (e.g. 932 vs 9327)
            soft = False
            for s in s1_nums:
                for t in t_nums:
                    if len(s) >= 2 and len(t) >= 2 and (s.startswith(t) or t.startswith(s) or s.endswith(t) or t.endswith(s)):
                        soft = True
                        break
                if soft: break
            if soft:
                house_num_soft = 0.8
            else:
                house_num_conflict = 1.0
    else:
        house_num_soft = 0.5
        
    # Geo matching
    if is_india:
        s_pin, s_p3, s_st, s_city, _ = extract_india_geo_full(s1_addr)
        t_pin, t_p3, t_st, t_city, _ = extract_india_geo_full(t_addr)
        
        pin_match = 1.0 if (s_pin and t_pin and s_pin == t_pin) else (0.0 if (s_pin and t_pin) else 0.5)
        pin3_match = 1.0 if (s_p3 and t_p3 and s_p3 == t_p3) else (0.0 if (s_p3 and t_p3) else 0.5)
        city_match = 1.0 if (s_city and t_city and s_city == t_city) else (0.0 if (s_city and t_city) else 0.5)
        state_match = 1.0 if (s_st and t_st and s_st == t_st) else (0.0 if (s_st and t_st) else 0.5)
    else:
        pin_match = pin3_match = city_match = state_match = 0.5
        
    # ---------------------------------------------------------
    # 3. CROSS-ATTRIBUTE INTERACTIONS
    # ---------------------------------------------------------
    name_x_addr = best_name_sim * addr_token_sort
    name_plus_addr = 0.5 * (best_name_sim + addr_token_sort)
    min_name_addr = min(best_name_sim, addr_jw)
    max_name_addr = max(best_name_sim, addr_jw)
    weighted_combined = 0.65 * best_name_sim + 0.35 * addr_token_sort
    colocated_distractor = 1.0 if (addr_token_sort >= 0.65 and best_name_sim < 0.40) else 0.0
    
    # ---------------------------------------------------------
    # 4. RETRIEVAL SIGNALS
    # ---------------------------------------------------------
    r_rank = float(rank)
    recip_rank = 1.0 / r_rank if r_rank > 0 else 0.0
    ch_votes = float(votes)
    ch_agreement = min(1.0, ch_votes / 6.0)
    
    return [
        name_ratio, name_lev, name_jw, name_token_sort, name_token_set,
        name_token_overlap, name_jaccard, name_char_ngram, name_len_diff,
        name_translit_jw, best_name_sim,
        both_addr_present, addr_ratio, addr_lev, addr_jw, addr_token_sort,
        addr_token_set, addr_token_overlap, addr_jaccard, addr_char_ngram,
        house_num_exact, house_num_soft, house_num_conflict,
        pin_match, pin3_match, city_match, state_match,
        name_x_addr, name_plus_addr, min_name_addr, max_name_addr,
        weighted_combined, colocated_distractor,
        r_rank, recip_rank, ch_votes, ch_agreement
    ]

print(f"Production features loaded: {len(FEATURE_NAMES)} features defined.")
