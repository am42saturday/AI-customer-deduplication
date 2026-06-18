# Entity Resolution for Marketplace Customer Profiles

This project addresses a customer-data-quality problem for a discount marketplace: duplicate profiles fragment purchase history, loyalty rewards, personalization signals, and business analytics. I designed an AI-powered entity resolution pipeline that matches noisy and incomplete customer records and recommends whether profiles should be automatically merged, manually reviewed, or kept separate.

The solution combines EDA, profile-level aggregation, similarity feature engineering, supervised matching, thresholding, and graph-based clustering into a reproducible prototype that can support cleaner customer data, more accurate personalization, and more reliable marketplace analytics.

## Current Solution Status

The current implementation is contained in `EDA_and_model.ipynb` and covers:

- exploratory data analysis of the source customer profiles;
- aggregation to the `profile_id` level;
- normalization of identity fields;
- analysis of duplicate groups through `entity_id`;
- generation of positive and negative profile pairs;
- feature engineering for profile-pair comparison;
- training a Decision Tree baseline;
- training the final CatBoost model;
- analysis of `auto_merge`, `manual_review`, and `do_not_merge` thresholds;
- graph clustering with connected components;
- validation of clustering quality.

## Business Goal

Duplicate customer profiles cause several problems:

- purchase history and customer behavior are fragmented across accounts;
- personalization and marketing campaigns become less accurate;
- one customer may receive benefits through multiple accounts;
- LTV, retention, RFM, and cohort analytics become distorted;
- additional opportunities for fraud appear.

This solution turns the problem into a controlled pipeline. The model estimates the probability that two profiles belong to the same customer, and a decision layer then selects one of three outcomes.

| Score range | Decision | Interpretation |
|---:|---|---|
| `score >= 0.85` | `auto_merge` | merge automatically when confidence is high |
| `0.60 <= score < 0.85` | `manual_review` | manually review the uncertain pair or cluster |
| `score < 0.60` | `do_not_merge` | keep the profiles separate |

## Data

The source dataset contains 68,036 rows and 12 columns. After aggregation, it contains 61,927 unique `profile_id` values and 53,369 unique `entity_id` values.

Main fields:

- `created_at`: date and time when the record appeared;
- `first_name`, `last_name`: synthetic customer names;
- `email`: profile email address;
- `phone`: phone number in various formats;
- `birthday`: date of birth;
- `sex`: sex;
- `non_processing_features`: technical and geographic features;
- `realtime_features`: JSON context such as country, city, timezone, and geoid;
- `fs_features`: behavioral features;
- `profile_id`: profile identifier;
- `entity_id`: ground-truth customer entity.

`entity_id` is used as the label. If one `entity_id` contains multiple `profile_id` values, those profiles are treated as duplicates belonging to the same customer.

## Key EDA Findings

- Duplicate groups contain 16,631 profiles, or 26.86% of all profiles.
- 8,073 `entity_id` values contain more than one `profile_id`.
- Most duplicate clusters have size 2, while the largest true cluster contains 116 profiles.
- `birthday`, `last_name`, `phone`, and `first_name` have high missing-value rates.
- Full email addresses are unique at the `profile_id` level, so exact email matching is not suitable as the primary duplicate-detection method.
- A normalized phone number is a high-precision signal, but coverage is low at approximately 7.5% of profiles.
- Technical and behavioral features substantially improve model quality, especially for sparse profiles.

## Methodology

### 1. Profile-Level Aggregation

Source rows are aggregated by `profile_id` because one profile may appear more than once. Identity fields use the first non-empty value, temporal features use the minimum and maximum `created_at`, and list-valued features combine unique tokens.

### 2. Normalization

The notebook performs the following transformations:

- converts email addresses to lowercase and separates them into `email_local` and `email_domain`;
- removes non-numeric characters from phone numbers and converts them to a consistent format;
- trims whitespace and lowercases text fields;
- parses `realtime_features` from JSON;
- converts `non_processing_features` and `fs_features` into token sets;
- calculates `identity_completeness_score`.

### 3. Pairwise Formulation

The task is formulated as binary classification of profile pairs:

- positive pair: both profiles have the same `entity_id`;
- negative pair: the profiles have different `entity_id` values.

Data is split by `entity_id` to prevent leakage between the training, validation, and test sets.

### 4. Feature Engineering

Similarity features are generated for every profile pair:

- contact: phone match, email-domain match, and email local-part similarity;
- name: first-name and full-name similarity;
- demographic: sex, birth year, and conflicts between known values;
- geographic: city, region, country, and geoid;
- technical: device, browser, operating system, and `non_processing_features` Jaccard similarity;
- behavioral: `fs_features` Jaccard similarity and `behavior_mean_sim`;
- temporal: difference between profile creation dates;
- completeness: number of exact matches and completeness score.

### 5. Matching Model

Decision Tree and CatBoost models were compared. The Decision Tree achieved acceptable quality but relied almost entirely on `behavior_mean_sim`, making it fragile.

The final model is a `CatBoostClassifier` with:

- `iterations=300`;
- `learning_rate=0.05`;
- `depth=4`;
- `loss_function='Logloss'`;
- `eval_metric='F1'`;
- `auto_class_weights='Balanced'`.

### 6. Clustering

After match probabilities are calculated, pairs with `score >= 0.85` are added to a graph as edges. Graph vertices are `profile_id` values, and connected components are interpreted as predicted entity clusters.

## Notebook Metrics

### Pairwise Validation Metrics, CatBoost, Threshold = 0.5

| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0, not duplicate | 0.98 | 0.98 | 0.98 | 9,261 |
| 1, duplicate | 0.89 | 0.88 | 0.88 | 1,323 |
| accuracy | | | 0.97 | 10,584 |
| macro avg | 0.93 | 0.93 | 0.93 | 10,584 |

### Threshold Analysis, Validation Set

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.90 | 0.967 | 0.755 | 0.848 |
| 0.85 | 0.962 | 0.794 | 0.870 |
| 0.80 | 0.957 | 0.815 | 0.880 |
| 0.60 | 0.929 | 0.854 | 0.890 |
| 0.50 | 0.915 | 0.868 | 0.891 |

### Cluster Validation Metrics

| Metric | Value |
|---|---:|
| Validation profiles / graph vertices | 9,266 |
| Auto-merge edges | 1,093 |
| Predicted clusters | 8,224 |
| Mean cluster size | 1.127 |
| Max predicted cluster size | 6 |
| ARI | 0.864 |
| NMI | 0.997 |
| Share of true entities not split | 96.75% |

## Installation

Recommended Python version: 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Notebook

1. Place the dataset at the path configured in the notebook, for example:

```text
content/split_label_train_V3.snappy.parquet
```

2. Start Jupyter Lab and open the notebook:

```bash
jupyter lab EDA_and_model.ipynb
```

## Deployment Recommendations

- Start with a batch pilot on a recent snapshot of the customer database.
- Do not merge every detected pair automatically. Use a high threshold for automatic merging and a separate manual-review range.
- Keep an audit log containing the model version, score, threshold, top features, original field values, and reviewer decision.
- Provide a rollback procedure for incorrectly merged profiles.
- Retrain the model with confirmed cases from manual review.

## Scaling Recommendations

- Move blocking into a separate layer and monitor candidate recall.
- Use multiple blocking keys: normalized phone, email domain and local prefix, geographic region, device/browser/OS, and MinHash or LSH for behavioral tokens.
- At large scale, use Spark or a SQL engine for candidate generation and batch scoring.

## Monitoring

Monitor:

- pairwise precision, recall, and F1 on a labeled control set;
- cluster metrics: ARI, NMI, cluster splitting, and cluster merging;
- the share of `auto_merge`, `manual_review`, and `do_not_merge` decisions;
- batch-processing latency;
- input-feature drift, including null rates, email domains, phones, regions, and devices.
