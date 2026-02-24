"""
Property Price Comparator — FastAPI Backend
Run: uvicorn server:app --reload --port 8000
"""

import re
import os
import asyncio
import base64
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Property Price Comparator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────

_df: Optional[pd.DataFrame] = None
DEFAULT_CSV = Path("default_predictions.csv")  # place your CSV here

# ── Street type map ───────────────────────────────────────────────────────────

STREET_TYPE_MAP = {
    "ST": "street", "RD": "road", "AVE": "avenue", "DR": "drive",
    "CR": "crescent", "CT": "court", "PL": "place", "TCE": "terrace",
    "GR": "grove", "BLVD": "boulevard", "HWY": "highway", "LN": "lane",
    "PROM": "promenade", "CL": "close", "WY": "way", "RDGE": "ridge",
    "ESP": "esplanade", "PDE": "parade", "CCT": "circuit", "LOOP": "loop",
    "CRES": "crescent", "GDNS": "gardens", "GDN": "garden",
    "MT": "mount", "MTWY": "motorway", "BYPA": "bypass",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_url(row: dict) -> str:
    state    = str(row['region']).lower()
    locality = str(row['locality']).replace(' ', '-').lower()
    postcode = str(int(float(row['postcode'])))
    street_words = str(row['street']).replace(' ', '-').lower()
    code     = str(row['streetTypeCode']).upper()
    street_type = STREET_TYPE_MAP.get(code, code.lower())
    number   = str(int(float(row['numberFirst'])))

    unit = row.get('unit')
    has_unit = unit is not None and str(unit).strip() not in ('', 'nan', 'NaN', 'None')
    params = f"?unitNumber={int(float(unit))}&streetNumber={number}" if has_unit else f"?streetNumber={number}"

    return (
        f"https://www.onthehouse.com.au/real-estate/{state}/"
        f"{locality}-{postcode}/{street_words}-{street_type}{params}"
    )

def format_address(row: dict) -> str:
    unit = row.get('unit')
    has_unit = unit is not None and str(unit).strip() not in ('', 'nan', 'NaN', 'None')
    num = str(int(float(row['numberFirst'])))
    prefix = f"{int(float(unit))}/{num}" if has_unit else num
    return f"{prefix} {row['street']} {row['streetTypeCode']}, {row['locality']}, {row['region']} {int(float(row['postcode']))}"

def row_to_dict(row: pd.Series) -> dict:
    d = row.where(pd.notna(row), None).to_dict()
    return {k: (None if v != v else v) for k, v in d.items()}

def safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# ── Price parsing ─────────────────────────────────────────────────────────────

_EXCLUDE_KW = re.compile(
    r'land.?value|council|unimproved|capital.?value|site.?value|'
    r'annual.?value|municipal|rating|levy|tax|insurance',
    re.IGNORECASE
)

def _parse_price(text: str, max_val: float = 15_000_000) -> Optional[float]:
    if not text:
        return None
    text = text.strip().replace(',', '').replace(' ', '')
    m = re.search(r'[\$]?([\d.]+)\s*[Mm](?:illion)?', text)
    if m:
        try:
            val = float(m.group(1)) * 1_000_000
            if 50_000 < val < max_val:
                return val
        except ValueError:
            pass
    m = re.search(r'[\$]?([\d.]+)\s*[Kk]', text)
    if m:
        try:
            val = float(m.group(1)) * 1_000
            if 50_000 < val < max_val:
                return val
        except ValueError:
            pass
    cleaned = re.sub(r'[^\d.]', '', text)
    try:
        val = float(cleaned)
        if 50_000 < val < max_val:
            return val
    except ValueError:
        pass
    return None

def _parse_price_or_range(text: str) -> Optional[tuple]:
    if not text:
        return None
    range_pat = re.search(
        r'([\$]?[\d.,]+\s*[KkMm]?)\s*[-–to]+\s*([\$]?[\d.,]+\s*[KkMm]?)',
        text
    )
    if range_pat:
        lo = _parse_price(range_pat.group(1))
        hi = _parse_price(range_pat.group(2))
        if lo and hi:
            return (lo, hi)
    single = _parse_price(text)
    if single:
        return (single, single)
    return None

def _extract_estimate_from_html(html: str) -> Optional[tuple]:
    patterns = [
        (r'for.?sale|asking.?price|listed.?at|listing.?price', 'for_sale'),
        (r'estim|valuat|avm|property.?value|price.?guide|price.?range', 'estimate'),
        (r'sold.?for|recently.?sold|last.?sold|sale.?price', 'sold'),
    ]
    for pattern, label in patterns:
        for m in re.finditer(pattern, html, re.IGNORECASE):
            window = html[max(0, m.start()-150): m.end()+400]
            if _EXCLUDE_KW.search(window):
                continue
            result = _parse_price_or_range(window)
            if result:
                return (result, label)
    return None

# ── Scraper ───────────────────────────────────────────────────────────────────

async def scrape_property(url: str, anthropic_key: str = "") -> dict:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900}
        )
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(3000)
        except Exception as e:
            await browser.close()
            raise RuntimeError(f"Page load failed: {e}")

        price = None
        source_label = "estimate"

        # DOM extraction
        estimate_selectors = [
            "[class*='estimatedValue']", "[class*='estimated-value']",
            "[class*='avm']", "[class*='estimate']", "[class*='valuation']",
            "[class*='price-estimate']", "[data-testid*='estimate']",
            "[data-testid*='valuation']", "[class*='propertyValue']",
            "[class*='property-value']",
        ]
        for_sale_selectors = [
            "[class*='forSale']", "[class*='for-sale']",
            "[class*='listingPrice']", "[class*='listing-price']",
            "[class*='askingPrice']", "[class*='asking-price']",
            "[data-testid*='listing-price']",
        ]
        sold_selectors = [
            "[class*='soldPrice']", "[class*='sold-price']",
            "[class*='salePrice']", "[class*='sale-price']",
            "[class*='lastSold']", "[class*='last-sold']",
            "[data-testid*='sold']",
        ]

        for selectors, lbl in [
            (estimate_selectors, 'estimate'),
            (for_sale_selectors, 'for_sale'),
            (sold_selectors, 'sold'),
        ]:
            if price:
                break
            for sel in selectors:
                try:
                    elements = await page.query_selector_all(sel)
                    for el in elements:
                        text = await el.inner_text()
                        result = _parse_price_or_range(text)
                        if result:
                            price = result
                            source_label = lbl
                            break
                except Exception:
                    continue
                if price:
                    break

        # HTML fallback
        if not price:
            content = await page.content()
            extracted = _extract_estimate_from_html(content)
            if extracted:
                price, source_label = extracted

        # Vision fallback
        screenshot_b64 = None
        if not price and anthropic_key:
            screenshot_bytes = await page.screenshot(full_page=False)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                response = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=256,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Look at this Australian property listing page. "
                                    "First look for a property estimate or valuation range (e.g. '$650k - $750k'). "
                                    "If no estimate exists, look for a 'for sale', 'recently sold for' or 'last sold' price. "
                                    "Reply with ONLY two numbers separated by a comma: low,high. "
                                    "If single value repeat it: 850000,850000. "
                                    "No symbols or text. If nothing found reply: null"
                                )
                            }
                        ]
                    }]
                )
                raw = response.content[0].text.strip()
                if raw.lower() != "null":
                    parts = raw.split(',')
                    if len(parts) == 2:
                        lo = _parse_price(parts[0])
                        hi = _parse_price(parts[1])
                        if lo and hi:
                            price = (lo, hi)
                            source_label = "vision"
            except Exception:
                pass

        await browser.close()

        result = {"url": url, "source_label": source_label}
        if price:
            result["price_lo"] = price[0]
            result["price_hi"] = price[1]
        else:
            result["price_lo"] = None
            result["price_hi"] = None

        return result

# ── API routes ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _df
    if DEFAULT_CSV.exists():
        _df = pd.read_csv(DEFAULT_CSV)
        print(f"[startup] Loaded default CSV: {DEFAULT_CSV} ({len(_df)} rows)")
    else:
        print(f"[startup] No default CSV found at {DEFAULT_CSV}")


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global _df
    content = await file.read()
    import io
    _df = pd.read_csv(io.BytesIO(content))
    return {"rows": len(_df), "filename": file.filename}


@app.get("/api/properties")
async def get_properties(
    search: str = "",
    page: int = 1,
    page_size: int = 50
):
    if _df is None:
        raise HTTPException(status_code=404, detail="No CSV loaded")

    df = _df.copy()

    if search:
        mask = df.apply(
            lambda row: search.lower() in " ".join(str(v).lower() for v in row.values),
            axis=1
        )
        df = df[mask]

    total = len(df)
    start = (page - 1) * page_size
    chunk = df.iloc[start: start + page_size]

    rows = []
    for idx, row in chunk.iterrows():
        d = row_to_dict(row)
        d['_idx'] = int(idx)
        d['_address'] = format_address(d)
        d['_url'] = build_url(d)
        rows.append(d)

    return {"total": total, "page": page, "rows": rows}


@app.get("/api/property/{idx}")
async def get_property(idx: int):
    if _df is None:
        raise HTTPException(status_code=404, detail="No CSV loaded")
    try:
        row = _df.loc[idx]
    except KeyError:
        raise HTTPException(status_code=404, detail="Row not found")
    d = row_to_dict(row)
    d['_idx'] = idx
    d['_address'] = format_address(d)
    d['_url'] = build_url(d)
    return d


@app.get("/api/random")
async def random_property(search: str = ""):
    if _df is None:
        raise HTTPException(status_code=404, detail="No CSV loaded")
    df = _df
    if search:
        mask = df.apply(
            lambda row: search.lower() in " ".join(str(v).lower() for v in row.values),
            axis=1
        )
        df = df[mask]
    if df.empty:
        raise HTTPException(status_code=404, detail="No matching rows")
    row = df.sample(1).iloc[0]
    idx = int(row.name)
    d = row_to_dict(row)
    d['_idx'] = idx
    d['_address'] = format_address(d)
    d['_url'] = build_url(d)
    return d


class ScrapeRequest(BaseModel):
    idx: int
    anthropic_key: str = ""


@app.post("/api/scrape")
async def scrape(req: ScrapeRequest):
    if _df is None:
        raise HTTPException(status_code=404, detail="No CSV loaded")
    try:
        row = _df.loc[req.idx]
    except KeyError:
        raise HTTPException(status_code=404, detail="Row not found")
    url = build_url(row_to_dict(row))
    result = await scrape_property(url, req.anthropic_key)
    return result


@app.get("/api/status")
async def status():
    return {
        "loaded": _df is not None,
        "rows": len(_df) if _df is not None else 0,
        "default_csv": str(DEFAULT_CSV),
        "default_exists": DEFAULT_CSV.exists(),
    }


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
