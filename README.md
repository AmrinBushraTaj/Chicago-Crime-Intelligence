# Chicago Crime Intelligence
### Machine Learning Applications in Predictive Policing and Geospatial Hotspot Analysis

> **Course:** INFO 6105 — Data Science Engineering Methods and Tools  
> **Team:** 13 | **Members:** Anushika Balamurgan · Amrin Bushra Taj  
> **University:** Northeastern University | **April 2026**  
> **Dataset:** Chicago Open Data Portal 2008–2017 | **Records:** 2,509,705

---

## Overview

A full-stack machine learning pipeline applied to the Chicago Open Data Portal crime dataset across three analytically distinct tasks:

| Task | Type | Best Model | Best Score |
|------|------|-----------|-----------|
| **Task 1:** Arrest Outcome Prediction | Binary Classification | XGBoost | ROC-AUC: **0.8780** |
| **Task 2:** Crime Hotspot Detection | Geospatial Clustering | DBSCAN (haversine) | Silhouette: **0.4001** |
| **Task 3:** Crime Type Prediction | Multi-class Classification | XGBoost | Macro-F1: **0.9957** |

---

## Dataset

- **Kaggle:** https://www.kaggle.com/datasets/adelanseur/crimes-2001-to-present-chicago
- **Original:** https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
- **Files needed:** `Chicago_Crimes_2008_to_2011.csv` + `Chicago_Crimes_2012_to_2017.csv`
- **Size:** ~450 MB combined | 2,509,705 records | 23 columns

> ⚠️ **CSV files are NOT included** (too large for GitHub). See `data/README.md` for download instructions.

---

## Project Structure

```
chicago-crime-intelligence/
│
├── notebooks/
│   └── Chicago_Crime_Intelligence.ipynb   ← Full pipeline: Tasks 1, 2 & 3
│
├── reports/
│   ├── Chicago_Crime_Intelligence_Report.docx   ← Final academic report
│   └── Chicago_Crime_Presentation_Slides.pdf    ← Presentation slides
│
├── assets/
│   └── chicago_crime_heatmap.html         ← Interactive map (generated on run)
│
├── data/
│   └── README.md                          ← Download instructions for CSVs
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/chicago-crime-intelligence.git
cd chicago-crime-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download CSVs from Kaggle → place in data/
#    Then update Cell 6 in the notebook:
FILE_1 = 'data/Chicago_Crimes_2008_to_2011.csv'
FILE_2 = 'data/Chicago_Crimes_2012_to_2017.csv'

# 4. Run
jupyter notebook notebooks/Chicago_Crime_Intelligence.ipynb
```

> Or open directly in **Google Colab** — Cell 0 auto-installs all dependencies and Cell 1 mounts Google Drive.

---

## Methodology

### Pipeline
```
Raw CSV Files (2,509,705 records)
        ↓
Step 1: Import Libraries
Step 2: Data Ingestion & Merging (2 CSV files → 1 DataFrame)
Step 3: EDA (crime distributions, temporal patterns, arrest rates)
Step 4: Data Cleaning (drop irrelevant cols, filter top-10 crime types)
Step 5: Feature Engineering (label encode, extract datetime features)
Step 6: Preprocessing (sample 150k, scale, 80/20 split)
        ↓
┌──────────────────┬─────────────────┬──────────────────┐
│     TASK 1       │     TASK 2      │     TASK 3       │
│ Arrest Outcome   │  Geospatial     │  Crime Type      │
│ Binary Classif.  │  Clustering     │  Multi-class     │
│ 4 ML Models      │ DBSCAN+KMeans   │  4 ML Models     │
└──────────────────┴─────────────────┴──────────────────┘
        ↓
Step 10: SHAP Explainability (Tasks 1 & 3)
Step 11: Final Summary & Conclusion
```

### Features Used (Both Tasks 1 & 3)

| Category | Features |
|----------|----------|
| Temporal | Hour, DayOfWeek, Month, Year |
| Spatial | Latitude, Longitude, District, Ward, Beat, Community Area |
| Categorical | Location Description (encoded), Domestic |
| Classification | FBI Code (encoded) |

### Imbalance Handling — Task 1

| Technique | Applied To |
|-----------|-----------|
| `class_weight='balanced'` | Logistic Regression, Decision Tree |
| `class_weight='balanced_subsample'` | Random Forest |
| `scale_pos_weight=neg/pos ratio` | XGBoost |
| SMOTE + RandomUnderSampler | XGBoost training data |
| Precision-Recall threshold tuning | All 4 models |

---

## Results

### Task 1 — Arrest Prediction
*Test set: 30,000 records | 22,885 No Arrest (76.3%) / 7,115 Arrest (23.7%)*

| Model | Accuracy | ROC-AUC | Arrest Precision | Arrest Recall | Arrest F1 |
|-------|----------|---------|-----------------|--------------|----------|
| **XGBoost** | **0.8806** | **0.8780** | 0.90 | 0.56 | **0.69** |
| Decision Tree | 0.8802 | 0.8710 | 0.92 | 0.55 | 0.68 |
| Random Forest | 0.8755 | 0.8586 | 0.95 | 0.50 | 0.66 |
| Logistic Regression | 0.7689 | 0.7131 | 0.54 | 0.18 | 0.27 |

**SHAP Top Features:** `Hour` > `District` > `Location Description` > `FBI Code` > `Beat`

---

### Task 2 — Geospatial Hotspot Detection
*50,000 GPS coordinate points*

| Algorithm | Config | Result |
|-----------|--------|--------|
| **DBSCAN** | haversine, eps=0.0008 (~88m), min_samples=50 | 1 dense cluster, 0 noise |
| **KMeans** | K=8, n_init=10 | Silhouette = 0.4001 |

**Primary Hotspot:** 41.84°N, 87.67°W — Chicago central corridor (Loop / Near South Side)  
**Dominant crime:** THEFT | **Arrest rate in hotspot:** 23.7%

---

### Task 3 — Crime Type Prediction
*Test set: 30,000 records | 10 crime classes*

| Model | Accuracy | Macro-F1 |
|-------|----------|---------|
| **XGBoost** | **99.65%** | **0.9957** |
| Decision Tree | 99.58% | 0.9950 |
| Random Forest | 93.51% | 0.9157 |
| Logistic Regression | 80.02% | 0.6949 |

**SHAP Top Features:** `Location Description` > `FBI Code` > `District` > `Hour` > `Beat`

---

## Model Configurations

### Task 1 Models
```python
# Logistic Regression
LogisticRegression(max_iter=500, class_weight='balanced', n_jobs=-1)

# Decision Tree
DecisionTreeClassifier(max_depth=12, class_weight='balanced', min_samples_leaf=20)

# Random Forest
RandomForestClassifier(n_estimators=100, max_depth=12,
                       class_weight='balanced_subsample', min_samples_leaf=10)

# XGBoost (trained on SMOTE-resampled data)
XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8,
              scale_pos_weight=RATIO, eval_metric='aucpr')
```

### Task 3 Models
```python
# Decision Tree
DecisionTreeClassifier(max_depth=15, min_samples_leaf=10)

# Random Forest
RandomForestClassifier(n_estimators=100, max_depth=15,
                       class_weight='balanced_subsample', min_samples_leaf=5)

# XGBoost
XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss')
```

---

## Generated Outputs

Running the full notebook produces:

| Output | Location | Description |
|--------|----------|-------------|
| `chicago_crime_heatmap.html` | `assets/` | Interactive Folium heatmap |
| ROC + PR Curves | Notebook Cell 37 | All 4 Task 1 models |
| Confusion Matrix | Notebook Cell 51 | Task 3 Decision Tree |
| SHAP Summary Plots | Notebook Cells 59–61 | Bar + beeswarm for Tasks 1 & 3 |
| Final Summary | Notebook Cell 63 | All results printed |

---

## Dependencies

```
pandas, numpy, scikit-learn, xgboost, imbalanced-learn,
shap, folium, matplotlib, seaborn, jupyter
```

Full list in [`requirements.txt`](requirements.txt).

---

## Ethical Statement

This project acknowledges predictive policing limitations:
- Historical crime data reflects policing intensity, not ground truth
- No demographic or racial features used — all signal from temporal, spatial, and incident-type attributes
- SHAP values enable full audit of individual predictions
- Outputs are decision **support** tools only — not autonomous enforcement triggers

---

## Citation

```bibtex
@misc{chicago_crime_intelligence_2026,
  author    = {Balamurgan, Anushika and Taj, Amrin Bushra},
  title     = {Chicago Crime Intelligence: Machine Learning Applications
               in Predictive Policing and Geospatial Hotspot Analysis},
  year      = {2026},
  note      = {INFO 6105 Final Project, Team 13, Northeastern University},
  url       = {https://github.com/YOUR_USERNAME/chicago-crime-intelligence}
}
```

---

## License

Academic use only. Dataset subject to [Chicago Police Department data terms](https://data.cityofchicago.org/).
