# Amazon Business Entity Resolution — Final Submission Verification Check
**ML Challenge 2026**  
**Execution Timestamp:** 2026-09-27T05:28:10+05:30  
**Selected Production Architecture:** Version 9 (V9)  
**Status:** VALIDATION COMPLETE — READY FOR SUBMISSION PACKAGING  

---

## 1. Executive Summary

- **Selected Submission Version:** Version 9 (V9)
- **Official Competition Leaderboard Score:** **0.776** (Current Best)
- **Full-Corpus Holdout Macro $F_{0.5}$:** **0.8177**
- **Total Validated Matches:** **5,956,917**
- **Total Blocking Candidate Pairs:** **49,747,767**
- **Official Validator Status:** **`PASS` (Exit Code 0, 0 Warnings, 0 Errors)**

---

## 2. File Verification & Exact Hashes

The production output files were copied from their verified V9 counterparts without modification. Hashes and exact byte lengths were checked before and after copying:

| File Name | Exact File Size (Bytes) | SHA256 Checksum | Match Status |
| :--- | :---: | :---: | :---: |
| `output/v9_final_matching_results.tsv` | 99,199,547 | `820AFB679FE8685F864EEFFFCEAB53A917826483C2DBA1D73BA1861CCB8C996B` | Master Baseline |
| `output/matching_results.tsv` | 99,199,547 | `820AFB679FE8685F864EEFFFCEAB53A917826483C2DBA1D73BA1861CCB8C996B` | **IDENTICAL (100% Byte Match)** |
| `output/v9_final_candidate_pairs.tsv` | 663,531,825 | `A264081402A52F4A9FDE0889BCAEABF65C42851722A968F2E4C5A4F752C80A2C` | Master Baseline |
| `output/candidate_pairs.tsv` | 663,531,825 | `A264081402A52F4A9FDE0889BCAEABF65C42851722A968F2E4C5A4F752C80A2C` | **IDENTICAL (100% Byte Match)** |

---

## 3. Official Validator Execution & Output

The official validator script was executed against the complete test dataset with full ID verification enabled (`--check-ids`):

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --check-ids \
    --test-dir data/test
```

### Execution Log Output:
```text
ML Challenge 2026 — submission validator
  test dir: data/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (90222 empty, 1642322 non-empty).
  candidate_pairs.tsv: 1732544 rows (1298 empty, 1731246 non-empty).

PASS — no blocking issues found. Safe to submit.
```
- **Exit Code:** `0`
- **Total S2/S3 Test Match IDs Verified:** 9,969,589
- **Required S1 Entities Checked:** 1,732,544
- **Format Warnings:** 0
- **Validation Errors:** 0

---

## 4. Comprehensive 10-Point Integrity Verification Checklist

An automated audit script was executed across all 1.73M entities and 5.95M matches:

| Check # | Requirement | Verified Result | Status |
| :---: | :--- | :--- | :---: |
| **1** | Exactly 1,732,544 S1 entities represented | Exactly 1,732,544 rows in both TSV files | **PASS** |
| **2** | Matching rows have valid IDs | All IDs match `S1-` (col 1) and `S2-`/`S3-` (col 2) | **PASS** |
| **3** | Candidate rows have valid IDs | All IDs match `S1-` (col 1) and `S2-`/`S3-` (col 2) | **PASS** |
| **4** | No invalid IDs or garbage strings | 0 unknown IDs, 0 incorrect prefixes | **PASS** |
| **5** | No missing required S1 entities | 0 missing S1 entities against `test_source1.tsv` | **PASS** |
| **6** | No duplicate target assignments (Target Exclusivity) | 5,956,917 distinct targets assigned across 5,956,917 match links (0 duplicate target assignments) | **PASS** |
| **7** | S1 can have multiple target matches | 1,466,526 S1 entities have >1 targets; 175,796 have 1 target; 90,222 singletons | **PASS** |
| **8** | Candidate coverage is valid | 100% of final matched IDs are present in `candidate_pairs.tsv` | **PASS** |
| **9** | File hashes identical before and after copying | Both SHA256 hashes match byte-for-byte | **PASS** |
| **10** | Exact file sizes and SHA256 reported | Exact sizes and SHA256 hashes verified and recorded | **PASS** |
| *Extra* | Zero self-matches | 0 `S1-` IDs present in target columns | **PASS** |

---

## 5. Country Breakdown of Matches

| Country Partition | Matched Links | Percentage of Total Matches | Operating Threshold | Candidate Cap |
| :--- | :---: | :---: | :---: | :---: |
| **United States** | 2,608,652 | 43.79% | $T = 0.75$ | 35 |
| **India** | 2,467,705 | 41.43% | $T = 0.90$ (Hard Neg Mined) | 35 |
| **France** | 880,560 | 14.78% | $T = 0.75$ | 25 |
| **Total** | **5,956,917** | **100.00%** | — | — |

---

## 6. Final Submission Directory Structure

In accordance with `README.md` (lines 156–167), the required submission bundle format is:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv            # 99,199,547 bytes | SHA256: 820AFB67...
│   └── candidate_pairs.tsv             # 663,531,825 bytes | SHA256: A2640814...
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── inspect_dataset.py
│       │   ├── production_features.py  # 37 domain features (RapidFuzz/Levenshtein/numeric)
│       │   ├── run_pipeline_v9.py      # End-to-end full production runner
│       │   └── ...
│       ├── README.md                   # Full reproduction guide
│       └── requirements.txt            # Pinned dependencies (numpy, lightgbm, rapidfuzz, etc.)
└── Documentation_template.md           # Methodology write-up
```

---

## 7. Final Packaged Submission Archive Details

The official submission archive has been compiled using `zipfile.ZIP_DEFLATED`:

| Attribute | Verified Value |
| :--- | :--- |
| **ZIP File Path** | `submission.zip` |
| **Exact Archive Size** | **323,155,649 bytes** (308.19 MB) |
| **Archive SHA256 Checksum** | `393B6CC33EDC242B5EADD47B6F54CEE5285BBFBAE823AB64B62AF48B14E0C831` |
| **Total Archive Entries** | 19 items (0 pycache / 0 .pyc files) |
| **Archived `output/matching_results.tsv` SHA256** | `820AFB679FE8685F864EEFFFCEAB53A917826483C2DBA1D73BA1861CCB8C996B` (MATCH) |
| **Archived `output/candidate_pairs.tsv` SHA256** | `A264081402A52F4A9FDE0889BCAEABF65C42851722A968F2E4C5A4F752C80A2C` (MATCH) |

### Complete Archive Manifest:
1. `output/matching_results.tsv` (99,199,547 bytes)
2. `output/candidate_pairs.tsv` (663,531,825 bytes)
3. `Documentation_template.md` (8,848 bytes)
4. `code/business_entity_resolution/README.md` (2,490 bytes)
5. `code/business_entity_resolution/requirements.txt` (388 bytes)
6. `code/business_entity_resolution/src/inspect_dataset.py` (14,661 bytes)
7. `code/business_entity_resolution/src/production_features.py` (13,089 bytes)
8. `code/business_entity_resolution/src/run_pipeline.py` (13,474 bytes)
9. `code/business_entity_resolution/src/run_pipeline_v2.py` (13,932 bytes)
10. `code/business_entity_resolution/src/run_pipeline_v3.py` (16,871 bytes)
11. `code/business_entity_resolution/src/run_pipeline_v4.py` (19,749 bytes)
12. `code/business_entity_resolution/src/run_pipeline_v5.py` (42,767 bytes)
13. `code/business_entity_resolution/src/run_pipeline_v6.py` (21,438 bytes)
14. `code/business_entity_resolution/src/run_pipeline_v8.py` (30,600 bytes)
15. `code/business_entity_resolution/src/run_pipeline_v9.py` (18,432 bytes)
16. `code/business_entity_resolution/src/test_v5_micro_subset.py` (1,263 bytes)
17. `code/business_entity_resolution/src/test_v5_production_smoke.py` (5,631 bytes)
18. `code/business_entity_resolution/src/train_baseline_matcher.py` (14,390 bytes)
19. `code/business_entity_resolution/src/v3_features.py` (15,797 bytes)

---

## 8. Operational Status

The final submission package has been generated, internally verified via streaming checksum extraction, and confirmed compliant with all rules.  
Per instructions: **STOPPED. No files have been uploaded or submitted.** Awaiting your explicit signal.
