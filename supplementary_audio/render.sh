#!/bin/bash
# Render all Rhai audio supplement scenarios to WAV.
# Run from the supplementary_audio/ directory.
set -euo pipefail

RENDER="cargo run --release --manifest-path ../../conchordal/Cargo.toml --bin conchordal-render --"

for rhai in scenarios/*.rhai; do
  name=$(basename "$rhai" .rhai)
  echo "Rendering ${name}..."
  $RENDER "$rhai" --output "audio/${name}.wav"
  echo "  → audio/${name}.wav"
done

echo "Done. All WAVs in audio/"
