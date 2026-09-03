#!/usr/bin/env bash
# Build the Hasat 2026 pitch deck PDF from deck.html with headless Chrome,
# then verify page count, page size and that no slide overflows.
set -euo pipefail
cd "$(dirname "$0")"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT="nest-hasat-2026.pdf"

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$OUT" "file://$PWD/deck.html" 2>/dev/null

python3 - "$OUT" <<'PY'
import sys, fitz
d = fitz.open(sys.argv[1])
bad = [i for i, p in enumerate(d) if abs(p.rect.width - 960) > 1 or abs(p.rect.height - 540) > 1]
print(f"pages={len(d)} size={d[0].rect.width:.0f}x{d[0].rect.height:.0f}pt bad_pages={bad}")
sys.exit(1 if bad else 0)
PY

# Overflow check: load deck.html with ?qa=1, the page reports any slide whose
# content is taller/wider than the fixed 13.333x7.5in box.
"$CHROME" --headless=new --disable-gpu --virtual-time-budget=4000 --window-size=1280,720 \
  --dump-dom "file://$PWD/deck.html?qa=1" 2>/dev/null | grep -o '<pre id="__qa">.*</pre>' | sed 's/<[^>]*>//g'
