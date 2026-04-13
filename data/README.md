# Data

This folder should contain the two Chicago crime CSV files.

## Download

1. Go to: https://www.kaggle.com/datasets/adelanseur/crimes-2001-to-present-chicago
2. Download both files and place them here:

```
data/
├── Chicago_Crimes_2008_to_2011.csv   (~200 MB)
└── Chicago_Crimes_2012_to_2017.csv   (~250 MB)
```

## Update Paths

Then in `notebooks/Chicago_Crime_Intelligence.ipynb`, Cell 5, update:

```python
FILE_1 = 'data/Chicago_Crimes_2008_to_2011.csv'
FILE_2 = 'data/Chicago_Crimes_2012_to_2017.csv'
```

## Original Source

Chicago Police Department CLEAR system via Chicago Open Data Portal:  
https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2

> ⚠️ CSV files are excluded from this repository via `.gitignore` due to their size (~450 MB combined).
