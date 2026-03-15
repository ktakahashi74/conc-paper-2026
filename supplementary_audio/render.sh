#!/bin/bash
# Render all Rhai audio supplement scenarios to WAV.
# Run from the supplementary_audio/ directory.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
CONCHORDAL_MANIFEST="${ROOT_DIR}/../conchordal/Cargo.toml"
PAPER_MANIFEST="${ROOT_DIR}/Cargo.toml"
RENDER="cargo run --release --manifest-path ${CONCHORDAL_MANIFEST} --bin conchordal-render --"

cd "${SCRIPT_DIR}"

echo "Regenerating Rhai scenarios..."
(cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --audio-rhai)

echo "Rendering E3 audio..."
(cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --e3-audio)

for rhai in scenarios/*.rhai; do
  name=$(basename "$rhai" .rhai)
  echo "Rendering ${name}..."
  $RENDER "$rhai" --output "audio/${name}.wav"
  if [ "$name" = "00_quicklisten" ]; then
    (cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --postprocess-quicklisten)
  elif [ "$name" = "20_integration" ]; then
    (cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --postprocess-integration)
  fi
  echo "  → audio/${name}.wav"
done

echo "Done. All WAVs in audio/"
