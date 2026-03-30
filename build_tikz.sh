#!/bin/bash
# Compile all TikZ .tex files to PDF and PNG
set -uo pipefail

TEX_DIR="resources/text"
OUT_DIR="visuals_1"
BUILD_DIR="/tmp/tikz_build"

mkdir -p "$OUT_DIR" "$BUILD_DIR"

for texfile in "$TEX_DIR"/fig_*.tex; do
    base=$(basename "$texfile" .tex)
    echo "Building $base..."

    # Compile to PDF
    pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" "$texfile" > /dev/null 2>&1
    cp "$BUILD_DIR/$base.pdf" "$OUT_DIR/$base.pdf"

    # Convert PDF to PNG (300 dpi)
    if command -v magick &> /dev/null; then
        magick -density 300 "$BUILD_DIR/$base.pdf" -quality 95 \
            "$OUT_DIR/$base.png"
    elif command -v convert &> /dev/null; then
        convert -density 300 "$BUILD_DIR/$base.pdf" -quality 95 \
            "$OUT_DIR/$base.png" 2>/dev/null || true
    fi

    # Convert PDF to SVG
    if command -v dvisvgm &> /dev/null; then
        dvisvgm --pdf "$BUILD_DIR/$base.pdf" -o "$OUT_DIR/$base.svg" 2>/dev/null || true
    fi

    echo "  Done: $base"
done

# Clean up
rm -f "$BUILD_DIR"/*.aux "$BUILD_DIR"/*.log

echo ""
echo "All TikZ diagrams compiled."
