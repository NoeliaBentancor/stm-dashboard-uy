# STM Dashboard UY

Dashboard for **public transport demand analysis (STM Montevideo)** featuring:
- descriptive analytics,
- demand forecasting using ML (Random Forest),
- anomaly detection based on residual analysis.

## Data Source

National Open Data Catalog (CKAN):
- Dataset: `Trips made on buses of the Metropolitan Transportation System (STM)`
- ID: `1205fc5c-b1b5-4478-b43e-c7411949ff15`

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── ingest.py
└── data/
    ├── raw/         # not versioned
    └── processed/   # aggregated parquet files
```

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Data ingestion and aggregation (last month by default)
python3 -m src.ingest --months 1

# Launch app
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. In Streamlit Cloud: **New app**.
3. Select repository: `NoeliaBentancor/stm-dashboard-uy`.
4. Branch: `main`, Main file path: `app.py`.
5. Deploy.
