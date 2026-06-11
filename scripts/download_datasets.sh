#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/proyecto/data"
BASE_URL="https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/subnat_fullwithnotes"

COUNTRIES=(
  AUS
  BRA
  CAN
  CHN
  GBR
  IND
  USA
)

mkdir -p "${DATA_DIR}"

for country in "${COUNTRIES[@]}"; do
  filename="OxCGRT_fullwithnotes_${country}_v1.csv"
  url="${BASE_URL}/${filename}"
  output_path="${DATA_DIR}/${filename}"
  temp_path="${output_path}.tmp"

  echo "Descargando ${filename}"
  curl --fail --location --show-error --progress-bar "${url}" --output "${temp_path}"
  mv "${temp_path}" "${output_path}"
done

echo "Datasets descargados en ${DATA_DIR}"
