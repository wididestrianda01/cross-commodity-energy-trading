#!/usr/bin/env bash
# Export all P17 notebooks to PDF via nbconvert webpdf (Playwright).
# Requires: nbconvert[webpdf], playwright chromium
# Setup: poetry add --group dev 'nbconvert[webpdf]' && poetry run playwright install chromium
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="docs/notebooks"
mkdir -p "$OUTDIR"

NOTEBOOKS=(
    "notebooks/01_market_landscape.ipynb"
    "notebooks/02_spread_economics.ipynb"
    "notebooks/03_correlation_crisis.ipynb"
    "notebooks/04_portfolio_risk.ipynb"
)

echo "Exporting ${#NOTEBOOKS[@]} notebooks to $OUTDIR ..."
for nb in "${NOTEBOOKS[@]}"; do
    name=$(basename "$nb" .ipynb)
    echo "  → $name ..."
    poetry run jupyter nbconvert --to webpdf --output-dir "$OUTDIR" "$nb" \
        > /dev/null 2>&1 && echo "    $OUTDIR/${name}.pdf"
done

echo ""
echo "Done. PDFs in $OUTDIR/:"
ls -lh "$OUTDIR"/*.pdf
