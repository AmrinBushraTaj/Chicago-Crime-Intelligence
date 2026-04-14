# 🔍 Chicago Crime Intelligence
### A Dual-Task Classification and Hotspot Detection Study

> **Course:** INFO 6105 — Data Science Engineering Methods and Tools  
> **Team 13** | Amrin Bushra Taj · Anushika Balamurgan  
> **Northeastern University, College of Engineering** | April 2026

---

## 📌 Overview

This project applies a comprehensive machine learning pipeline to **553,879 Chicago crime incident records (2008–2017)** to build a three-task crime intelligence platform:

| Task | Type | Goal |
|------|------|------|
| **Task 1** — Arrest Prediction | Binary Classification | Predict whether an incident results in an arrest |
| **Task 2** — Hotspot Detection | Unsupervised Clustering | Identify geographic crime concentration zones |
| **Task 3** — Crime Type Prediction | Multi-class Classification | Predict the primary offense category (10 classes) |

---

## 🏆 Key Results

### Task 1 — Arrest Outcome Prediction

| Model | Accuracy | ROC-AUC | Arrest F1 |
|-------|----------|---------|-----------|
| **XGBoost** ⭐ | 0.8765 | **0.8780** | **0.6964** |
| Random Forest | 0.8768 | 0.8676 | 0.6913 |
| Decision Tree | 0.8715 | 0.8680 | 0.6948 |
| Logistic Regression | 0.7895 | 0.7181 | 0.5334 |

### Task 2 — Geospatial Hotspot Detection

| Algorithm | Result |
|-----------|--------|
| KMeans | 8 clusters · Silhouette Score: **0.401** |
| DBSCAN | 1 dominant hotspot · Centre: 41.84°N, 87.67°W · Top crime: Theft (arrest rate 23.8%) |

### Task 3 — Crime Type Prediction (10 classes)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| **XGBoost** ⭐ | **0.9954** | **0.9946** |
| Decision Tree | 0.9947 | 0.9937 |
| Random Forest | 0.9702 | 0.9705 |
| Logistic Regression | 0.7958 | 0.6955 |

---

## 📂 Repository Structure

```
chicago-crime-intelligence/
│
├── data/
│   ├── Chicago_Crimes_2008_to_2011.csv      # Source data (download separately)
│   └── Chicago_Crimes_2012_to_2017.csv      # Source data (download separately)
│
├── notebooks/
│   └── Chicagi_Crime_Intelligence_Team_13.ipynb   # Main analysis notebook
│
├── outputs/
│   ├── chicago_crime_heatmap.html           # Interactive Folium heatmap
│   └── figures/                             # All saved plots
│
├── report/
│   └── Chicago_Crime_Intelligence_Team_13.docx    # Full academic report
│
└── README.md
```

---

## 🗃️ Dataset

**Source:** [Chicago Data Portal — Crimes 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

Two CSV files were merged to form the consolidated dataset:
- `Chicago_Crimes_2008_to_2011.csv`
- `Chicago_Crimes_2012_to_2017.csv`

**Update the file paths** in the notebook before running:
```python
FILE_1 = '/path/to/Chicago_Crimes_2008_to_2011.csv'
FILE_2 = '/path/to/Chicago_Crimes_2012_to_2017.csv'
```

| Property | Value |
|----------|-------|
| Total records | 553,879 |
| Attributes | 23 |
| Memory footprint | ~421 MB |
| Time span | 2008–2017 |
| Missing coords | 3.90% (excluded from clustering) |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Jupyter Notebook or Google Colab

### Install dependencies

```bash
pip install seaborn xgboost shap folium scikit-learn pandas numpy matplotlib imbalanced-learn
```

Or run the first cell of the notebook which handles installation automatically:

```python
!pip install -q seaborn xgboost shap folium scikit-learn pandas numpy matplotlib imbalanced-learn
```

### Run the notebook

```bash
jupyter notebook notebooks/Chicagi_Crime_Intelligence_Team_13.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/) and upload the notebook + data files.

---

## 🔬 Methodology

### Feature Engineering
13 features used across all tasks:

| Feature | Description |
|---------|-------------|
| `Hour`, `DayOfWeek`, `Month`, `Year` | Extracted from incident timestamp |
| `Latitude`, `Longitude` | Geographic coordinates |
| `Domestic` | Boolean encoded to 0/1 |
| `Location Description` | Label-encoded (140 categories) |
| `FBI Code`, `IUCR` | Label-encoded crime classification codes |
| `Beat`, `District`, `Ward` | CPD administrative subdivisions |

### Class Imbalance Handling (Task 1)
- **SMOTE** oversampling of minority (Arrest) class
- **RandomUnderSampler** on majority class  
- Combined pipeline yields ~1.25:1 ratio (from original 3.19:1)
- Per-model **threshold optimization** via Precision-Recall F1 maximization

### Geospatial Clustering (Task 2)
- **KMeans** on StandardScaler-normalized coordinates; K=8 via Elbow method
- **DBSCAN** with Haversine metric (eps ≈ 88m, min_samples=50) via ball_tree
- Interactive heatmap generated with **Folium** on CartoDB dark-matter tiles

### Explainability
- **SHAP TreeExplainer** applied to best XGBoost models for Tasks 1 & 3
- Summary bar plots + beeswarm plots over 500-instance test subsamples
- Key drivers: geographic coordinates, hour of day, FBI/IUCR codes

---

## 📊 Pipeline Overview

```
Raw CSVs (2008–2011, 2012–2017)
        │
        ▼
   Data Merging & Inspection
        │
        ▼
   Exploratory Data Analysis
   (crime types, temporal heatmaps, arrest rates, locations)
        │
        ▼
   Cleaning & Feature Engineering
   (drop irrelevant cols, parse datetime, label encode)
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
   Task 1                 Task 2                 Task 3
   Binary Classification  Geospatial Clustering  Multi-class Classification
   (Arrest Prediction)    (KMeans + DBSCAN)      (Crime Type)
        │                      │                      │
        ▼                      ▼                      ▼
   SMOTE + Threshold      Elbow Method           4 Models Trained
   Optimization           Silhouette Score       Macro-F1 Evaluation
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                               ▼
                      SHAP Explainability
                   (Tasks 1 & 3 — XGBoost)
```

---

## 📈 EDA Highlights

- **Top crime types:** Theft (115,656) > Battery (102,195) > Criminal Damage (63,571) > Narcotics (60,824)
- **Arrest rate:** 26.7% overall — highest for Narcotics (~68%), lowest for Theft/Criminal Damage (<15%)
- **Temporal peaks:** Crime spikes between 18:00–02:00; Fridays and Saturdays highest
- **Crime trend:** Consistent year-over-year decline from ~75K incidents (2009) to ~40K (2017)

---

## 🧠 SHAP Findings

**Task 1 (Arrest Prediction):** Latitude and Longitude are the top predictors — specific Chicago zones (South/West Side) have systematically higher arrest probabilities. Hour of day shows a strong negative contribution for late-night incidents. The `Domestic` flag consistently raises arrest probability, consistent with Illinois mandatory arrest statutes.

**Task 3 (Crime Type):** IUCR and FBI Code dominate predictions due to their definitional relationship with Primary Type. Among contextual features, geographic coordinates and Hour of Day are the strongest discriminators between crime categories.

---

## ⚠️ Limitations

- **No temporal holdout:** Random splitting may overestimate prospective performance; a train-2008–2015 / test-2016–2017 split is recommended for deployment validation.
- **Definitional features in Task 3:** IUCR and FBI Code are structurally linked to the crime type target, inflating accuracy figures.
- **DBSCAN resolution:** A single cluster at 88m scale reflects city-wide density; finer-grained analysis requires smaller epsilon or HDBSCAN.
- **Missing covariates:** Socioeconomic, weather, and prior-history features are absent.

---

## ⚖️ Ethical Considerations

Predictive models trained on historical arrest data encode past policing decisions, not just underlying crime rates. Deployment of these models without bias audits risks amplifying racial and socioeconomic disparities. Any operational use should include:
- Disaggregated fairness evaluations (equalized odds, calibration by community area demographics)
- Community consultation and transparency
- Ongoing monitoring for disparate impact and concept drift

---

## 📄 References

- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016.*
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017.*
- Ester et al. (1996). A Density-Based Algorithm for Discovering Clusters. *KDD 1996.*
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR.*
- Sherman, Gartin & Buerger (1989). Hot Spots of Predatory Crime. *Criminology.*
- Chicago Data Portal. Crimes — 2001 to Present. https://data.cityofchicago.org

---

## 👥 Team

| Name | Institution |
|------|-------------|
| Amrin Bushra Taj | Northeastern University |
| Anushika Balamurgan | Northeastern University |

---

*INFO 6105 — Data Science Engineering Methods and Tools | Northeastern University | April 2026*
