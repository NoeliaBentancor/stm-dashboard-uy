from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

DATASET_ID = "1205fc5c-b1b5-4478-b43e-c7411949ff15"
PACKAGE_SHOW_URL = "https://catalogodatos.gub.uy/api/3/action/package_show"

USECOLS = [
    "fecha_evento",
    "cod_empresa",
    "descrip_empresa",
    "linea_codigo",
    "dsc_linea",
    "cantidad_pasajeros",
]


def fetch_resources(dataset_id: str = DATASET_ID) -> list[dict]:
    response = requests.get(PACKAGE_SHOW_URL, params={"id": dataset_id}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError("CKAN package_show failed")
    return payload["result"]["resources"]


def monthly_zip_resources(resources: Iterable[dict]) -> list[dict]:
    selected: list[dict] = []
    for resource in resources:
        name = (resource.get("name") or "").lower()
        fmt = (resource.get("format") or "").lower()
        url = (resource.get("url") or "").lower()
        if not name.startswith("viajes "):
            continue
        if "zip" in fmt or url.endswith(".zip"):
            selected.append(resource)
    selected.sort(key=lambda item: item.get("position", 0))
    return selected


def download(url: str, destination: Path) -> None:
    if "imnube.montevideo.gub.uy/share/s/" in url and not url.rstrip("/").endswith("/download"):
        url = f"{url.rstrip('/')}/download"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) STM-Dashboard/1.0",
        "Accept": "*/*",
    }
    with requests.get(url, timeout=120, stream=True, headers=headers) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)


def aggregate_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found inside {zip_path.name}")
        csv_name = csv_names[0]
        with zf.open(csv_name, "r") as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding="utf-8", errors="replace")
            chunks: list[pd.DataFrame] = []
            for chunk in pd.read_csv(
                text_file,
                usecols=USECOLS,
                chunksize=400_000,
                parse_dates=["fecha_evento"],
                low_memory=False,
            ):
                chunk["cod_empresa"] = chunk["cod_empresa"].astype(str)
                chunk["descrip_empresa"] = chunk["descrip_empresa"].astype(str)
                chunk["linea_codigo"] = chunk["linea_codigo"].astype(str)
                chunk["dsc_linea"] = chunk["dsc_linea"].astype(str)
                chunk["cantidad_pasajeros"] = pd.to_numeric(chunk["cantidad_pasajeros"], errors="coerce").fillna(0)
                chunk["timestamp_hour"] = chunk["fecha_evento"].dt.floor("h")
                grouped = (
                    chunk.groupby(
                        [
                            "timestamp_hour",
                            "cod_empresa",
                            "descrip_empresa",
                            "linea_codigo",
                            "dsc_linea",
                        ],
                        dropna=False,
                        as_index=False,
                    )["cantidad_pasajeros"]
                    .sum()
                    .rename(columns={"cantidad_pasajeros": "demanda"})
                )
                chunks.append(grouped)
    if not chunks:
        return pd.DataFrame(
            columns=[
                "timestamp_hour",
                "cod_empresa",
                "descrip_empresa",
                "linea_codigo",
                "dsc_linea",
                "demanda",
            ]
        )
    merged = pd.concat(chunks, ignore_index=True)
    merged = (
        merged.groupby(
            ["timestamp_hour", "cod_empresa", "descrip_empresa", "linea_codigo", "dsc_linea"],
            dropna=False,
            as_index=False,
        )["demanda"]
        .sum()
        .sort_values("timestamp_hour")
    )
    return merged


def run(months: int, output_path: Path, raw_dir: Path) -> None:
    resources = fetch_resources()
    zipped = monthly_zip_resources(resources)
    if not zipped:
        raise RuntimeError("No monthly ZIP resources were found")
    selected = zipped[-months:]

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_aggregates: list[pd.DataFrame] = []
    for resource in selected:
        name = resource.get("name", "resource")
        url = resource.get("url")
        if not url:
            continue
        zip_name = f"{name.replace(' ', '_').lower()}.zip"
        zip_path = raw_dir / zip_name
        if zip_path.exists():
            print(f"Using cached file {zip_name}...")
        else:
            print(f"Downloading {name}...")
            download(url, zip_path)
        print(f"Aggregating {zip_name}...")
        all_aggregates.append(aggregate_zip(zip_path))

    if not all_aggregates:
        raise RuntimeError("No resources were processed")

    final_df = pd.concat(all_aggregates, ignore_index=True)
    final_df = (
        final_df.groupby(
            ["timestamp_hour", "cod_empresa", "descrip_empresa", "linea_codigo", "dsc_linea"],
            dropna=False,
            as_index=False,
        )["demanda"]
        .sum()
        .sort_values("timestamp_hour")
    )
    final_df["cod_empresa"] = final_df["cod_empresa"].astype(str)
    final_df["descrip_empresa"] = final_df["descrip_empresa"].astype(str)
    final_df["linea_codigo"] = final_df["linea_codigo"].astype(str)
    final_df["dsc_linea"] = final_df["dsc_linea"].astype(str)
    final_df.to_parquet(output_path, index=False)
    print(f"Wrote {len(final_df):,} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and aggregate STM monthly travel data")
    parser.add_argument("--months", type=int, default=1, help="How many latest months to ingest")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stm_hourly_line.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where raw ZIP files are temporarily stored",
    )
    args = parser.parse_args()
    run(months=args.months, output_path=args.output, raw_dir=args.raw_dir)


if __name__ == "__main__":
    main()
