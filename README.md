# STM Dashboard UY

Dashboard de **demanda de transporte (STM Montevideo)** con:
- análisis descriptivo,
- forecast de demanda con ML (Random Forest),
- detección de anomalías por residuo.

## Fuente de datos

Catálogo Nacional de Datos Abiertos (CKAN):
- Dataset: `Viajes realizados en los ómnibus del Sistema de Transporte Metropolitano - STM`
- ID: `1205fc5c-b1b5-4478-b43e-c7411949ff15`

## Estructura

```
.
├── app.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── ingest.py
└── data/
    ├── raw/         # no versionado
    └── processed/   # parquet agregado
```

## Correr local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ingesta y agregación (último mes por defecto)
python3 -m src.ingest --months 1

# App
streamlit run app.py
```

## Deploy en Streamlit Community Cloud

1. Push del repo a GitHub.
2. En Streamlit Cloud: **New app**.
3. Seleccionar repo: `NoeliaBentancor/stm-dashboard-uy`.
4. Branch: `main`, Main file path: `app.py`.
5. Deploy.

## Próxima mejora (AI novedosa)

Comparar el forecast clásico contra un foundation model de series temporales (por ejemplo Chronos desde Hugging Face) y mostrar benchmark MAE/MAPE en la UI.

