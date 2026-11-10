# Dataset README

## Datasets Used in This Project

This project uses two publicly available IoT intrusion detection datasets:

---

## 1. DS2OS Dataset (Primary)

**Full name:** Distributed Smart Space Orchestration System IoT Dataset  
**Source paper:** Shivanjali, K. and Michael, T., "Ensemble learning for detecting attacks and anomalies in IoT smart home," *3rd International Conference on Data Intelligence and Security*, 2020.

### Download

**Method 1 — Kaggle (Recommended):**
1. Go to https://www.kaggle.com
2. Search for "DS2OS multiclass network traffic" 
3. Download the dataset (CSV file, approximately 15–30 MB)
4. Place the file at: `data/raw/DS2OS_traffic_traces.csv`

**Method 2 — UCI ML Repository:**
- Visit https://archive.ics.uci.edu
- Search "DS2OS"

### Quick Stats

| Property | Value |
|---|---|
| Total records | 357,952 |
| Number of classes | 8 |
| Number of features | 13 |
| Severe imbalance? | Yes (Worst ratio 2851:1) |

### Class Labels

| Label | Samples |
|---|---|
| Normal | 347,935 |
| DoS | 5,780 |
| Scan | 1,547 |
| Malicious Control | 889 |
| Malicious Operation | 805 |
| Spying | 532 |
| Data Probing | 342 |
| Wrong Setup | 122 |

---

## 2. TON_IoT Dataset (Secondary / Reference)

**Full name:** TON_IoT (Thing-of-Network IoT) Dataset  
**Source paper:** Moustafa, N. et al., "Federated TON_IoT Windows Datasets for Evaluating AI-based Security Applications," *IEEE TrustCom 2020*.  
**Official page:** https://research.unsw.edu.au/projects/toniot-datasets

### Download

1. Visit https://research.unsw.edu.au/projects/toniot-datasets
2. Register with an academic email (free)
3. Download the **IoT/IIoT network traffic subset** (not Windows OS or Linux OS logs)
4. Place extracted CSV files at: `data/raw/TON_IoT/`

### Quick Stats

| Property | Value |
|---|---|
| Total records (IoT network subset) | ~461,043 |
| Number of classes | 10 |
| Number of features | ~44 |

---

## Important Notes

1. **Do NOT commit raw dataset CSV files to this repository.** They are large (DS2OS ~20 MB; TON_IoT can be hundreds of MB) and may have licence restrictions. Only commit processed results and analysis scripts.

2. **After downloading**, place the files at the paths specified above, then run the preprocessing scripts in `src/`.

3. **Cite the datasets in your paper** using the references above.

4. **For DS2OS:** The file is usually named `DS2OS_traffic_traces.csv`. If it has a different name, update the path in `src/01_data_loading.py`.
