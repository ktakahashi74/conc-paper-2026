set shell := ["bash", "-euo", "pipefail", "-c"]

default:
	@just --list

# Run experiments (release build, default: e1,e2,e3,e6b,e7)
paper *args:
	cargo run --release --bin paper -- {{args}}

# Cargo check + test
paper-check:
	cargo check --bins
	cargo check --all-targets
	cargo test --bins

# Convert SVG plots to PDF
svg2pdf dir="experiments/plots":
	@if command -v rsvg-convert >/dev/null 2>&1; then \
		find "{{dir}}" -type f -name '*.svg' -print0 | \
		while IFS= read -r -d '' svg; do \
			pdf="${svg%.svg}.pdf"; \
			rsvg-convert -f pdf -o "$pdf" "$svg"; \
			echo "write $pdf"; \
		done; \
	elif command -v inkscape >/dev/null 2>&1; then \
		find "{{dir}}" -type f -name '*.svg' -print0 | \
		while IFS= read -r -d '' svg; do \
			pdf="${svg%.svg}.pdf"; \
			inkscape "$svg" --export-type=pdf --export-filename="$pdf" >/dev/null; \
			echo "write $pdf"; \
		done; \
	else \
		echo "error: install inkscape or rsvg-convert for vector SVG->PDF conversion"; \
		exit 1; \
	fi
	# Enforce paper rebuild whenever figures are regenerated.
	just latex

# Run experiments + convert SVGs to PDF + rebuild paper PDF
paper-pdf *args:
	just paper {{args}}
	just svg2pdf

# Build main.pdf with pdflatex
latex:
	pdflatex -interaction=nonstopmode main.tex

# Full pipeline: experiments → SVG→PDF → pdflatex
all *args:
	just paper-pdf {{args}}

# Archive plots directory
tar out="experiments/plots.tar" dir="experiments/plots":
	@if [ ! -d "{{dir}}" ]; then \
		echo "error: directory not found: {{dir}}"; \
		exit 1; \
	fi
	tar -cf "{{out}}" -C "{{dir}}" .
	@echo "write {{out}}"

# Remove plots and lock file
clean:
	rm -rf experiments/plots
	rm -f experiments/.paper_plots.lock
