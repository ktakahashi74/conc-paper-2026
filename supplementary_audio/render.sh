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
(cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --e3-audio)

for rhai in scenarios/*.rhai; do
  name=$(basename "$rhai" .rhai)
  if [[ "$name" == temporal_scaffold_* ]]; then
    echo "Skipping ${name}; temporal scaffold WAVs are rendered directly by paper --e3-audio"
    continue
  fi
  echo "Rendering ${name}..."
  $RENDER "$rhai" --output "audio/${name}.wav"
  if [ "$name" = "showcase" ]; then
    (cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --postprocess-quicklisten-showcase)
  elif [ "$name" = "controls" ]; then
    (cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --postprocess-quicklisten-controls)
  elif [ "$name" = "self_organized_polyphony" ]; then
    (cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --postprocess-polyphony)
  elif [ "$name" = "self_organized_polyphony_no_hill" ]; then
    (cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --postprocess-polyphony-no-hill)
  elif [ "$name" = "hereditary_adaptation" ]; then
    (cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --postprocess-e6b)
  elif [ "$name" = "hereditary_adaptation_controls" ]; then
    (cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --postprocess-hereditary-controls)
  elif [ "$name" = "hereditary_adaptation_candidates" ]; then
    (cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --postprocess-hereditary-candidates)
  fi
  echo "  → audio/${name}.wav"
done

echo "Done. All WAVs in audio/"
