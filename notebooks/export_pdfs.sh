#!/usr/bin/env bash
# Export all P17 notebooks to PDF via HTML + weasyprint.
# Requires: jupyter nbconvert, weasyprint (pip install weasyprint)
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="docs/notebooks"
TMPDIR=$(mktemp -d)
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
    poetry run jupyter nbconvert --to html --output-dir "$TMPDIR" "$nb" > /dev/null 2>&1
    poetry run python -c "
from weasyprint import HTML
HTML('$TMPDIR/${name}.html').write_pdf('$OUTDIR/${name}.pdf')
" && echo "    $OUTDIR/${name}.pdf"
done

rm -rf "$TMPDIR"
echo ""
echo "Done. PDFs in $OUTDIR/:"
ls -lh "$OUTDIR"/*.pdf
