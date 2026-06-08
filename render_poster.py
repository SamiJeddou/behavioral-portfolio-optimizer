# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
# Re-render poster.html -> poster.png (2x) and poster.pdf, matching the original pipeline.
# Requires: pip install playwright ; python -m playwright install chromium
from playwright.sync_api import sync_playwright
import pathlib
src = pathlib.Path("poster.html").resolve().as_uri()
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(device_scale_factor=2)
    pg.goto(src)
    pg.wait_for_timeout(400)
    el = pg.query_selector(".poster")
    box = el.bounding_box()
    el.screenshot(path="poster.png")
    pg.set_viewport_size({"width": int(box["width"]), "height": int(box["height"]) + 2})
    pg.pdf(path="poster.pdf", width=f'{int(box["width"])}px',
           height=f'{int(box["height"])+2}px', print_background=True)
    b.close()
print("wrote poster.png and poster.pdf")
