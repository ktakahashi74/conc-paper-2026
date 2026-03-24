#!/bin/bash
# Render all Rhai audio supplement scenarios to WAV.
# Run from the supplementary_audio/ directory.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
CONCHORDAL_MANIFEST="${ROOT_DIR}/../conchordal/Cargo.toml"
CONCHORDAL_RENDER_BIN="${ROOT_DIR}/../conchordal/target/release/conchordal-render"
PAPER_MANIFEST="${ROOT_DIR}/Cargo.toml"

render_with_conchordal() {
  local script="$1"
  local output="$2"

  if [[ -x "${CONCHORDAL_RENDER_BIN}" ]]; then
    "${CONCHORDAL_RENDER_BIN}" "$script" --output "$output"
  elif command -v conchordal-render >/dev/null 2>&1; then
    conchordal-render "$script" --output "$output"
  else
    cargo run --release --manifest-path "${CONCHORDAL_MANIFEST}" --bin conchordal-render -- "$script" --output "$output"
  fi
}

cd "${SCRIPT_DIR}"

echo "Regenerating Rhai scenarios..."
(cd "${ROOT_DIR}" && cargo run --release --manifest-path "${PAPER_MANIFEST}" --bin paper -- --audio-rhai)

for rhai in scenarios/*.rhai; do
  name=$(basename "$rhai" .rhai)
  echo "Rendering ${name}..."
  render_with_conchordal "$rhai" "audio/${name}.wav"
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
  fi
  echo "  → audio/${name}.wav"
done

echo "Done. All WAVs in audio/"
