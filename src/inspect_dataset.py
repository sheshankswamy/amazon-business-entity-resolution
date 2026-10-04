"""
Amazon Business Entity Resolution Hackathon — Dataset Inspector
Performs memory-efficient chunked inspection of all TSV dataset files.
"""

import os
import sys
import json
import time
from collections import Counter

# Set UTF-8 encoding for stdout/stderr
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def inspect_tsv_file(file_path, expected_prefix, sample_size=5, chunk_size=100000):
    start_time = time.time()
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    total_rows = 0
    col_names = []
    sample_records = []
    
    missing_counts = {}
    country_counts = Counter()
    prefix_violations = 0
    id_seen = set()
    duplicate_ids = 0

    name_empty_count = 0
    address_empty_count = 0
    name_len_sum = 0
    address_len_sum = 0
    min_name_len = float('inf')
    max_name_len = 0
    min_addr_len = float('inf')
    max_addr_len = 0

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        header_line = f.readline()
        if not header_line:
            return {"error": "File is empty"}
        col_names = [c.strip() for c in header_line.split('\t')]
        for c in col_names:
            missing_counts[c] = 0

        # Process lines
        for line_num, line in enumerate(f, start=1):
            total_rows += 1
            parts = line.rstrip('\r\n').split('\t')
            
            # Record sample records
            if len(sample_records) < sample_size:
                sample_records.append(parts)

            # Check column count
            if len(parts) < len(col_names):
                parts.extend([''] * (len(col_names) - len(parts)))
            elif len(parts) > len(col_names):
                # Extra tabs in address or name?
                parts = parts[:len(col_names)]

            # Row dict
            row = dict(zip(col_names, parts))

            # Missing values check
            for c, val in row.items():
                if val is None or val.strip() == '' or val.strip().lower() in ['nan', 'null', 'none']:
                    missing_counts[c] += 1

            # Entity ID checks
            eid = row.get('entity_id', '').strip()
            if not eid.startswith(expected_prefix):
                prefix_violations += 1
            if eid in id_seen:
                duplicate_ids += 1
            else:
                id_seen.add(eid)

            # Country check
            country = row.get('country', '').strip()
            country_counts[country] += 1

            # Name and address quality
            name = row.get('business_name', '').strip()
            addr = row.get('business_address', '').strip()

            n_len = len(name)
            a_len = len(addr)

            if n_len == 0:
                name_empty_count += 1
            else:
                min_name_len = min(min_name_len, n_len)
                max_name_len = max(max_name_len, n_len)
                name_len_sum += n_len

            if a_len == 0:
                address_empty_count += 1
            else:
                min_addr_len = min(min_addr_len, a_len)
                max_addr_len = max(max_addr_len, a_len)
                address_len_sum += a_len

    elapsed = time.time() - start_time
    valid_name_rows = total_rows - name_empty_count
    valid_addr_rows = total_rows - address_empty_count

    return {
        "file_path": file_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "total_rows": total_rows,
        "elapsed_seconds": round(elapsed, 2),
        "columns": col_names,
        "sample_records": sample_records,
        "missing_counts": missing_counts,
        "missing_pct": {c: round((cnt / total_rows) * 100, 4) if total_rows > 0 else 0 for c, cnt in missing_counts.items()},
        "prefix_expected": expected_prefix,
        "prefix_violations": prefix_violations,
        "duplicate_ids": duplicate_ids,
        "unique_ids_count": len(id_seen),
        "country_distribution": dict(country_counts),
        "quality_metrics": {
            "name_empty_count": name_empty_count,
            "name_avg_len": round(name_len_sum / valid_name_rows, 2) if valid_name_rows > 0 else 0,
            "name_min_len": min_name_len if min_name_len != float('inf') else 0,
            "name_max_len": max_name_len,
            "address_empty_count": address_empty_count,
            "address_avg_len": round(address_len_sum / valid_addr_rows, 2) if valid_addr_rows > 0 else 0,
            "address_min_len": min_addr_len if min_addr_len != float('inf') else 0,
            "address_max_len": max_addr_len,
        },
        "id_set": id_seen  # returned for cross-referencing
    }

def inspect_ground_truth(gt_path, train_s1_ids, train_s2_ids, train_s3_ids, sample_size=5):
    start_time = time.time()
    file_size_bytes = os.path.getsize(gt_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    total_s1_rows = 0
    col_names = []
    sample_records = []
    
    empty_matches_count = 0  # singletons
    match_list_len_counts = Counter()
    total_matched_instances = 0
    all_matched_ids_set = set()
    s2_matched_ids_set = set()
    s3_matched_ids_set = set()
    
    internal_duplicates_in_row = 0
    s1_self_reference_count = 0
    unknown_prefix_count = 0
    unresolved_s2_ids = 0
    unresolved_s3_ids = 0
    gt_s1_ids_seen = set()
    duplicate_gt_s1_rows = 0

    with open(gt_path, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline()
        col_names = [c.strip() for c in header.split('\t')]
        
        for line in f:
            total_s1_rows += 1
            parts = line.rstrip('\r\n').split('\t')
            if len(sample_records) < sample_size:
                sample_records.append(parts)
            
            s1_id = parts[0].strip()
            if s1_id in gt_s1_ids_seen:
                duplicate_gt_s1_rows += 1
            else:
                gt_s1_ids_seen.add(s1_id)

            match_str = parts[1].strip() if len(parts) > 1 else ''
            if not match_str:
                empty_matches_count += 1
                match_list_len_counts[0] += 1
            else:
                matched_ids = [m.strip() for m in match_str.split(',') if m.strip()]
                match_len = len(matched_ids)
                match_list_len_counts[match_len] += 1
                total_matched_instances += match_len
                
                # Check within-row duplicates
                if len(matched_ids) != len(set(matched_ids)):
                    internal_duplicates_in_row += 1

                for mid in matched_ids:
                    all_matched_ids_set.add(mid)
                    if mid.startswith('S1-'):
                        s1_self_reference_count += 1
                    elif mid.startswith('S2-'):
                        s2_matched_ids_set.add(mid)
                        if train_s2_ids and mid not in train_s2_ids:
                            unresolved_s2_ids += 1
                    elif mid.startswith('S3-'):
                        s3_matched_ids_set.add(mid)
                        if train_s3_ids and mid not in train_s3_ids:
                            unresolved_s3_ids += 1
                    else:
                        unknown_prefix_count += 1

    elapsed = time.time() - start_time
    
    # Check coverage of train_source1.tsv
    missing_s1_in_gt = len(train_s1_ids - gt_s1_ids_seen) if train_s1_ids else 0
    extra_s1_in_gt = len(gt_s1_ids_seen - train_s1_ids) if train_s1_ids else 0

    return {
        "file_path": gt_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "total_s1_rows": total_s1_rows,
        "unique_s1_in_gt": len(gt_s1_ids_seen),
        "duplicate_gt_s1_rows": duplicate_gt_s1_rows,
        "missing_s1_in_gt": missing_s1_in_gt,
        "extra_s1_in_gt": extra_s1_in_gt,
        "columns": col_names,
        "sample_records": sample_records,
        "empty_matches_count_singletons": empty_matches_count,
        "singleton_pct": round((empty_matches_count / total_s1_rows) * 100, 4) if total_s1_rows > 0 else 0,
        "total_matched_instances": total_matched_instances,
        "unique_matched_entities": len(all_matched_ids_set),
        "unique_s2_matches": len(s2_matched_ids_set),
        "unique_s3_matches": len(s3_matched_ids_set),
        "match_list_distribution": dict(sorted(match_list_len_counts.items())),
        "internal_duplicates_in_row": internal_duplicates_in_row,
        "s1_self_reference_count": s1_self_reference_count,
        "unknown_prefix_count": unknown_prefix_count,
        "unresolved_s2_ids": unresolved_s2_ids,
        "unresolved_s3_ids": unresolved_s3_ids,
        "elapsed_seconds": round(elapsed, 2)
    }

def main():
    print("=" * 80)
    print("AMAZON BUSINESS ENTITY RESOLUTION — DATASET INSPECTION")
    print("=" * 80)

    # 1. Environment & Compute
    import platform
    import ctypes
    
    print("\n--- [1] COMPUTE & ENVIRONMENT ---")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python Version: {platform.python_version()}")
    print(f"Python Executable: {sys.executable}")
    print(f"CPU Count: {os.cpu_count()} cores")
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullExtendedVirtual', ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_ram = stat.ullTotalPhys / (1024**3)
        avail_ram = stat.ullAvailPhys / (1024**3)
        print(f"System RAM: {total_ram:.2f} GB total, {avail_ram:.2f} GB available ({stat.dwMemoryLoad}% load)")
    except Exception as e:
        print(f"System RAM check: {e}")

    files_to_inspect = [
        ("train/train_source1.tsv", "S1-"),
        ("train/train_source2.tsv", "S2-"),
        ("train/train_source3.tsv", "S3-"),
        ("test/test_source1.tsv", "S1-"),
        ("test/test_source2.tsv", "S2-"),
        ("test/test_source3.tsv", "S3-"),
    ]

    results = {}
    id_sets = {}

    print("\n--- [2] SOURCE FILES INSPECTION ---")
    for rel_path, prefix in files_to_inspect:
        fpath = os.path.join(DATA_DIR, rel_path)
        print(f"\nProcessing {rel_path}...")
        if not os.path.exists(fpath):
            print(f"ERROR: File not found: {fpath}")
            continue
        
        res = inspect_tsv_file(fpath, prefix)
        id_sets[rel_path] = res.pop("id_set")  # Extract set for memory and GT cross-referencing
        results[rel_path] = res
        
        print(f"  Path: {fpath}")
        print(f"  Size: {res['file_size_mb']} MB ({res['file_size_bytes']:,} bytes)")
        print(f"  Rows: {res['total_rows']:,}")
        print(f"  Columns: {res['columns']}")
        print(f"  Unique Entity IDs: {res['unique_ids_count']:,} | Duplicates: {res['duplicate_ids']}")
        print(f"  Prefix Violations: {res['prefix_violations']}")
        print(f"  Missing values: {res['missing_counts']} ({res['missing_pct']})")
        print(f"  Country Distribution: {res['country_distribution']}")
        print(f"  Quality: Name avg len = {res['quality_metrics']['name_avg_len']}, empty names = {res['quality_metrics']['name_empty_count']}")
        print(f"           Addr avg len = {res['quality_metrics']['address_avg_len']}, empty addrs = {res['quality_metrics']['address_empty_count']}")
        print(f"  Sample row 1: {res['sample_records'][0] if res['sample_records'] else 'N/A'}")

    # Inspect Ground Truth
    print("\n--- [3] GROUND TRUTH INSPECTION ---")
    gt_path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
    if os.path.exists(gt_path):
        gt_res = inspect_ground_truth(
            gt_path,
            id_sets.get("train/train_source1.tsv", set()),
            id_sets.get("train/train_source2.tsv", set()),
            id_sets.get("train/train_source3.tsv", set())
        )
        results["train/train_ground_truth.tsv"] = gt_res
        print(f"  Path: {gt_path}")
        print(f"  Size: {gt_res['file_size_mb']} MB ({gt_res['file_size_bytes']:,} bytes)")
        print(f"  Total Source 1 Entities in GT: {gt_res['total_s1_rows']:,}")
        print(f"  Missing S1 from train_source1: {gt_res['missing_s1_in_gt']} | Extra S1: {gt_res['extra_s1_in_gt']}")
        print(f"  Singletons (Empty Matches): {gt_res['empty_matches_count_singletons']:,} ({gt_res['singleton_pct']}%)")
        print(f"  Entities with Matches: {gt_res['total_s1_rows'] - gt_res['empty_matches_count_singletons']:,}")
        print(f"  Total Matched Instances: {gt_res['total_matched_instances']:,}")
        print(f"  Unique Matched Entities: {gt_res['unique_matched_entities']:,} (S2: {gt_res['unique_s2_matches']:,}, S3: {gt_res['unique_s3_matches']:,})")
        print(f"  Unresolved S2 IDs: {gt_res['unresolved_s2_ids']} | Unresolved S3 IDs: {gt_res['unresolved_s3_ids']}")
        print(f"  S1 Self References: {gt_res['s1_self_reference_count']} | Unknown Prefixes: {gt_res['unknown_prefix_count']}")
        print(f"  Duplicates in single match list: {gt_res['internal_duplicates_in_row']}")
        print("  Match list size distribution (top 10):")
        for k, v in list(gt_res['match_list_distribution'].items())[:10]:
            print(f"    Length {k}: {v:,} rows ({v/gt_res['total_s1_rows']*100:.2f}%)")
    else:
        print(f"ERROR: Ground truth file not found at {gt_path}")

    # Save full results as JSON
    report_json_path = os.path.join(LOGS_DIR, "dataset_inspection_results.json")
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed inspection JSON to: {report_json_path}")

if __name__ == '__main__':
    main()
