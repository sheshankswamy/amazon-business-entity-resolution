# Pretrained Model Artifacts

This directory contains the serialized model binary for the final production matcher used in **Version 9 (V9)** of the Amazon Business Entity Resolution solution.

## Model Overview

- **Filename:** `v9_india_matcher_37f.joblib`
- **File Size:** ~2.44 MB (2,435,660 bytes)
- **Algorithm:** LightGBM (`LGBMClassifier` / Gradient Boosted Decision Trees)
- **Input Dimensionality:** 37 domain-engineered pairwise similarity features
- **Training Strategy:** Iterative Hard-Negative Mining over the full 4.13M Indian target entity space
- **Operating Decision Threshold:** $T = 0.90$ (Calibrated for maximum precision on Macro $F_{0.5}$)
- **Holdout Performance:** Full-corpus holdout Macro $F_{0.5} = 0.8177$ (Precision: 90.92%, Recall: 74.84%)

## Feature Schema (37 Features)

The model evaluates candidate pairs across four core feature categories implemented via `RapidFuzz` and NumPy in `src/production_features.py`:
1. **Name Similarity (12 features):** Token Sort Ratio, Token Set Ratio, Partial Ratio, Levenshtein Distance, Jaro-Winkler, Character 3-Gram Jaccard, Stem Overlap, Legal Suffix Compatibility, Phonetic Consonant Skeleton Match.
2. **Address & Geographic Similarity (15 features):** Token Sort/Set Ratio, Partial Ratio, Postal PIN Match Flag, 3-Digit Postal Prefix Match, Extracted House/Building Number Match, House Number Conflict Veto Flag, Street Name Jaccard.
3. **Cross-Attribute Interactions (10 features):** Multiplicative interactions (`name_sim * addr_sim`, `token_set * house_match`), Co-location penalty terms, Source-2 vs Source-3 provenance adjustments.

## Baseline & Regional Processing

For the US and France partitions, lightweight baseline classifiers and conservative rule-based thresholds ($T = 0.75$) were developed during earlier iterations. The repository contains the core feature engineering, candidate retrieval, modeling, and V9 production pipeline developed during the competition. The original competition dataset and large intermediate production artifacts are intentionally excluded due to their size and competition constraints.
