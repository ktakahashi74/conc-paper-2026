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
	just latex-main

# Run experiments + convert SVGs to PDF + rebuild paper PDF
paper-pdf *args:
	just paper {{args}}
	just svg2pdf

# Build main.pdf with bibtex + reruns
latex-main:
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

# Build supplementary.pdf
supplementary:
	pdflatex -interaction=nonstopmode supplementary.tex
	pdflatex -interaction=nonstopmode supplementary.tex

# Build both paper PDFs
latex:
	just latex-main
	just supplementary

# Build an arXiv source bundle (TeX source + figures + ancillary supplement)
arxiv out="dist/conc-paper-2026-arxiv.tar.gz" stage="dist/arxiv-src":
	rm -rf "{{stage}}" "{{out}}"
	mkdir -p "{{stage}}/experiments/plots/e1"
	mkdir -p "{{stage}}/experiments/plots/e2"
	mkdir -p "{{stage}}/experiments/plots/e3"
	mkdir -p "{{stage}}/experiments/plots/e6b"
	mkdir -p "{{stage}}/experiments/plots/e7"
	mkdir -p "{{stage}}/anc"
	cp main.tex "{{stage}}/"
	printf '%s\n' '\def\isarxiv{1}' | cat - "{{stage}}/main.tex" > "{{stage}}/main.tex.tmp"
	mv "{{stage}}/main.tex.tmp" "{{stage}}/main.tex"
	cp main.bbl "{{stage}}/"
	cp alifeconf.sty "{{stage}}/"
	cp experiments/plots/e1/paper_e1_landscape_scan_anchor220.pdf "{{stage}}/experiments/plots/e1/"
	cp experiments/plots/e2/paper_e2_figure_e2_1.pdf "{{stage}}/experiments/plots/e2/"
	cp experiments/plots/e2/paper_e2_figure_e2_2.pdf "{{stage}}/experiments/plots/e2/"
	cp experiments/plots/e3/paper_e3_figure4.pdf "{{stage}}/experiments/plots/e3/"
	cp experiments/plots/e6b/paper_e6b_figure.pdf "{{stage}}/experiments/plots/e6b/"
	cp experiments/plots/e7/paper_e7_figure.pdf "{{stage}}/experiments/plots/e7/"
	cp supplementary.pdf "{{stage}}/anc/"
	(cd "{{stage}}" && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
	rm -f "{{stage}}"/main.aux "{{stage}}"/main.log "{{stage}}"/main.out "{{stage}}"/main.pdf
	tar -czf "{{out}}" -C "{{stage}}" .
	@echo "write {{out}}"

# Full pipeline: experiments → SVG→PDF → manuscript PDFs
all *args:
	just paper-pdf {{args}}
	just supplementary

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
