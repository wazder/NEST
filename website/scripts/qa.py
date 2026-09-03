#!/usr/bin/env python3
"""Layout QA for the static site: renders every page at three viewports with
headless Chrome (light + dark), collects the ?qa=1 report and checks links."""
import json, re, subprocess, sys
from pathlib import Path
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAGES = [ROOT / "index.html"] + sorted((ROOT / "pages").glob("*.html")) + [ROOT / "en" / "index.html"] + sorted((ROOT / "en" / "pages").glob("*.html"))
VIEWPORTS = [(1440, 900), (1024, 800), (390, 844)]

HOST = Path("/private/tmp/claude-501/-Users-wazder-GitHub/ee9d2005-321c-47db-b378-8490225e4f7a/scratchpad/qa")

def render(page: Path, w: int, h: int, dark: bool) -> dict:
    theme = "dark" if dark else "light"
    url = f"file://{page}?qa=1&theme={theme}"
    if w < 500:
        # headless Chrome will not go below 500px wide; emulate with a fixed-width iframe host
        HOST.mkdir(parents=True, exist_ok=True)
        host = HOST / f"host_{w}_{page.parent.name}_{page.stem}.html"
        host.write_text(f"""<!doctype html><meta charset=utf-8><body style="margin:0">
<iframe id=f src="{url}" style="width:{w}px;height:{h}px;border:0"></iframe>
<script>window.addEventListener('load',function(){{setTimeout(function(){{try{{var d=document.getElementById('f').contentDocument;var q=d&&d.getElementById('__qa');var p=document.createElement('pre');p.id='__qa';p.textContent=q?q.textContent:'{{"error":"no qa in iframe"}}';document.body.appendChild(p);}}catch(e){{var p=document.createElement('pre');p.id='__qa';p.textContent=JSON.stringify({{error:String(e)}});document.body.appendChild(p);}}}},2500);}});</script>""")
        url = f"file://{host}"
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", f"--window-size={max(w, 800)},{h}",
           "--virtual-time-budget=8000", "--hide-scrollbars", "--dump-dom", url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    m = re.search(r'<pre id="__qa">(.*?)</pre>', out, re.S)
    if not m: return {"page": str(page.relative_to(ROOT)), "vw": w, "error": "no qa block"}
    import html as _h
    return json.loads(_h.unescape(m.group(1)))

class LinkParser(HTMLParser):
    def __init__(self): super().__init__(); self.links = []; self.ids = set()
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a: self.ids.add(a["id"])
        if tag == "a" and a.get("href"): self.links.append(a["href"])
        if tag in ("link", "script", "img") and (a.get("href") or a.get("src")): self.links.append(a.get("href") or a.get("src"))

def check_links():
    problems = []
    parsed = {}
    for p in PAGES:
        lp = LinkParser(); lp.feed(p.read_text()); parsed[p] = lp
    for p, lp in parsed.items():
        for href in lp.links:
            if href.startswith(("http", "mailto:", "data:")): continue
            target, _, frag = href.partition("#")
            tpath = p if not target else (p.parent / target.split("?")[0]).resolve()
            if target and not tpath.exists(): problems.append(f"{p.relative_to(ROOT)} -> missing file {href}"); continue
            if frag:
                ids = lp.ids if not target else (parsed.get(tpath).ids if tpath in parsed else set(re.findall(r'id="([^"]+)"', tpath.read_text())))
                if frag not in ids: problems.append(f"{p.relative_to(ROOT)} -> missing anchor {href}")
    return problems

def main():
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(page, w, h, dark) for page in PAGES for (w, h) in VIEWPORTS for dark in (False, True)]
    fails = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda j: (j, render(*j)), jobs))
    for (page, w, h, dark), r in results:
        bad = r.get("error") or r.get("horizontalScroll") or r.get("overflowCount") or r.get("clipped") or r.get("brokenImages") or r.get("h1Count") != 1
        flag = "FAIL" if bad else "ok  "
        if bad: fails += 1
        print(f"{flag} {r.get('page')} @{w} {'dark' if dark else 'light'}: hscroll={r.get('horizontalScroll')} overflow={r.get('overflowCount')} clipped={len(r.get('clipped') or [])} imgs={r.get('brokenImages')} h1={r.get('h1Count')} inter={r.get('fontInter')}")
        if bad and not r.get("error"):
            for o in (r.get("overflow") or [])[:6]: print("      overflow:", o)
            for c in (r.get("clipped") or [])[:6]: print("      clipped:", c)
    links = check_links()
    for l in links: print("LINK", l)
    print(f"\nsummary: {fails} failing renders, {len(links)} link problems")
    sys.exit(1 if (fails or links) else 0)

if __name__ == "__main__":
    main()
