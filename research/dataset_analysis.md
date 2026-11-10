# Dataset Analysis: DS2OS and TON_IoT

## 1. Dataset Selection Decision

For this project we use the **DS2OS dataset** as the primary dataset.

**Reasons:**
- DS2OS is smaller (357,952 records) and computationally manageable on CPU
- It has 13 features (vs. TON_IoT's 44), making it faster to process and explain
- It is freely available via multiple sources without registration
- It has the most extreme imbalance cases (Wrong Setup: 122 samples out of 357,952 → imbalance ratio 2,851:1)
- The DS2OS dataset is confirmed as the primary evaluation dataset in the DIFT paper
- TON_IoT would be used as a secondary validation if compute allows

We will document the TON_IoT dataset structure for completeness.

---

## 2. DS2OS Dataset

### 2.1 Official Source and Access

- **Full name:** Distributed Smart Space Orchestration System IoT Dataset
- **Original publication:** Shivanjali, K., Michael, T., "Ensemble learning for detecting attacks and anomalies in IoT smart home," 3rd International Conference on Data Intelligence and Security, 2020.
- **Kaggle mirror:** https://www.kaggle.com/datasets/francoisxa/ds2osmulticlassnetworktraffic (freely accessible)
- **UCI Repository:** Also available via UCI ML repository
- **License:** Public research use; citation required

### 2.2 Dataset Structure

| Property | Value |
|---|---|
| Total records | 357,952 |
| Number of classes | 8 (1 Normal + 7 attack) |
| Number of features | 13 |
| File format | CSV |
| File name | DS2OS_traffic_traces.csv (typical) |

### 2.3 Features (13 total)

| Feature | Type | Description |
|---|---|---|
| sourceid | Nominal (categorical) | Source device identifier |
| sourceAddress | Nominal (categorical) | Source device IP/address |
| sourceType | Nominal (categorical) | Type of source device |
| sourceLocation | Nominal (categorical) | Physical location of source device |
| destinationServiceAddress | Nominal (categorical) | Destination service address |
| destinationServiceType | Nominal (categorical) | Type of destination service |
| destinationLocation | Nominal (categorical) | Physical location of destination |
| accessedNodeAddress | Nominal (categorical) | Address of accessed IoT node |
| accessedNodeType | Nominal (categorical) | Type of accessed node (**has missing values**) |
| operation | Nominal (categorical) | Operation performed (read/write/update/etc.) |
| value | Continuous (float) | Value associated with the operation (**has missing values**) |
| timestamp | Discrete (integer) | Timestamp of the event |
| normality | Nominal (target) | Class label (Normal or attack type) |

**Features used for modelling:** We drop `sourceid`, `sourceAddress`, `destinationServiceAddress`, `accessedNodeAddress`, and `timestamp` as they are identifiers/raw addresses that do not generalise. We retain: `sourceType`, `sourceLocation`, `destinationServiceType`, `destinationLocation`, `accessedNodeType`, `operation`, `value` — 7 informative features.

### 2.4 Class Distribution

| Class | Count | % of Total | Imbalance Ratio vs. Normal |
|---|---|---|---|
| Normal | 347,935 | 97.20% | 1:1 (majority) |
| DoS | 5,780 | 1.615% | 60.2:1 |
| Scan | 1,547 | 0.432% | 225.0:1 |
| Malicious Control | 889 | 0.248% | 391.4:1 |
| Malicious Operation | 805 | 0.225% | 432.5:1 |
| Spying | 532 | 0.149% | 654.2:1 |
| Data Probing | 342 | 0.096% | 1017.4:1 |
| Wrong Setup | 122 | 0.034% | 2851.1:1 |

**Key observation:** Wrong Setup (122 samples) is 2,851 times less frequent than Normal. Even DoS (5,780 samples) is 60x less frequent. This extreme imbalance validates the necessity of augmentation.

### 2.5 Known Data Quality Issues

1. **Missing values in `accessedNodeType`:** Approximately 15–20% of records have NaN in this field.
2. **Missing values in `value`:** Some records have NaN in the continuous value field.
3. **Handling strategy:**
   - `accessedNodeType`: fill with mode ("unknown_device") or separate category
   - `value`: fill with median of training set only (to avoid data leakage)

### 2.6 Train/Test Split

DIFT uses approximately 75%/25% split (260,951 train / 86,984 test for Normal class).

For our implementation: **80/20 stratified split** to maintain class proportions.

---

## 3. TON_IoT Dataset (Documentation Only)

### 3.1 Official Source

- **Official:** https://research.unsw.edu.au/projects/toniot-datasets
- **Requires registration** at UNSW official page (free for research)
- **Citation required:** Moustafa, N. et al., "Federated TON_IoT Windows Datasets," IEEE TrustCom 2020.
- **Kaggle mirrors available** but verify licence compliance

### 3.2 Structure (IoT Network Subset)

| Property | Value |
|---|---|
| Total records (IoT Network subset) | ~461,043 (as per DIFT paper) |
| Number of classes | 10 (1 Normal + 9 attack) |
| Number of features | ~44 (network flow features) |

### 3.3 Class Distribution (from DIFT paper, Table 2)

| Class | Train | Test | Total | Imbalance vs. Normal |
|---|---|---|---|---|
| Normal | 210,000 | 90,000 | 300,000 | 1:1 (majority) |
| DDoS | 14,000 | 6,000 | 20,000 | 15:1 |
| DoS | 14,000 | 6,000 | 20,000 | 15:1 |
| Injection | 14,000 | 6,000 | 20,000 | 15:1 |
| MITM | 730 | 313 | 1,043 | 288:1 |
| Password | 14,000 | 6,000 | 20,000 | 15:1 |
| Ransomware | 14,000 | 6,000 | 20,000 | 15:1 |
| Scanning | 14,000 | 6,000 | 20,000 | 15:1 |
| XSS | 14,000 | 6,000 | 20,000 | 15:1 |
| Backdoor | 14,000 | 6,000 | 20,000 | 15:1 |

**Key observation:** MITM is the only extreme minority class in TON_IoT (1,043 samples vs 300,000 Normal). Most attack classes have 20,000 samples — not highly imbalanced relative to Normal but heavily outnumbered.

### 3.4 Key Features (44 network flow features, partial list)

Flow-level features extracted by Zeek/Argus:
- src_ip, src_port, dst_ip, dst_port, proto, service
- duration, orig_bytes, resp_bytes, missed_bytes
- orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes
- conn_state, history
- ... (additional 30 derived statistical features)

---

## 4. Data Download Instructions

### 4.1 DS2OS (Primary Dataset)

**Option 1 — Kaggle (fastest):**
1. Create a Kaggle account at https://kaggle.com (free)
2. Search: "DS2OS multiclass network traffic" 
3. Download: `DS2OS_traffic_traces.csv`
4. Place in: `data/raw/DS2OS_traffic_traces.csv`

**Option 2 — UCI ML Repository:**
- Search "DS2OS" at https://archive.ics.uci.edu

### 4.2 TON_IoT (Secondary Dataset)

1. Visit: https://research.unsw.edu.au/projects/toniot-datasets
2. Register with academic email
3. Download "TON_IoT Network dataset" (IoT/IIoT network traffic subset)
4. Place files in: `data/raw/TON_IoT/`

> **Note:** Do not commit raw dataset files to GitHub. They are large and may have licence restrictions. Document the download procedure instead.

---

## 5. Dataset Size Considerations

| Dataset | File Size (approximate) | Records | Decision |
|---|---|---|---|
| DS2OS | ~15–30 MB CSV | 357,952 | ✅ Use fully (CPU-feasible) |
| TON_IoT (network) | ~500 MB – 2 GB | ~461,043 | Secondary; use subset if needed |

For DS2OS, the full dataset fits comfortably in memory (~50 MB when loaded as DataFrame).

---

## 6. Preprocessing Plan

```python
# Step 1: Load data
df = pd.read_csv('data/raw/DS2OS_traffic_traces.csv')

# Step 2: Identify and handle missing values
# - Fill 'accessedNodeType' NaN with 'Unknown'
# - Fill 'value' NaN with training median (fit on train only)

# Step 3: Drop non-informative identifier columns
drop_cols = ['sourceid', 'sourceAddress', 'destinationServiceAddress', 
             'accessedNodeAddress', 'timestamp']
df = df.drop(columns=drop_cols, errors='ignore')

# Step 4: Encode categorical features
# - LabelEncoder or OrdinalEncoder for tree-based models
# - Fit encoders on training data only

# Step 5: Encode target variable
# - Map class labels to integers: Normal=0, DoS=1, Scan=2, etc.

# Step 6: Train/test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Scale numerical features
# - StandardScaler fit on X_train only, applied to X_test

# Step 8: SHAP feature selection (Phase 1 of SHAP-DA-IDS)
# Step 9: Difficulty-adaptive augmentation (Phase 2-3 of SHAP-DA-IDS)
# Step 10: Final classifier training and evaluation
```

**Data leakage prevention:** All fitted transformers (encoder, scaler, imputer) are fitted exclusively on `X_train`. The same transforms are then applied to `X_test` without re-fitting.
