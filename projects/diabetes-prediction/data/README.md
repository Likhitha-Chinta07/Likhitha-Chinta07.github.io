# Dataset

The project expects the public Pima Indians Diabetes CSV at `data/diabetes.csv`.

Run from the project directory:

```bash
python download_data.py
```

The downloader adds the column names expected by the project and validates the row format before saving the file.

The dataset contains 768 records, eight predictor variables and one binary target. The benchmark is commonly distributed as a headerless CSV; this project adds explicit column names so the analysis code is self-documenting.
