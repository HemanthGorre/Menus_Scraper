import os, re, sys, json, shutil, tempfile, subprocess
from pathlib import Path
import argparse
import pandas as pd
import duckdb

# ---------- Gemini ----------
import google.generativeai as genai

# ---------- Optional OCR / PDF helpers ----------
import pdfplumber

# ---------- Heuristic keyword sets ----------
DRINK_KEYWORDS_NONALC = {
    "soda","soft drink","juice","lemonade","limeade","milkshake","smoothie","iced tea",
    "tea","coffee","latte","mocha","americano","espresso","hot chocolate","milk","water",
    "sparkling","club soda","kombucha","horchata","agua fresca"
}
DRINK_KEYWORDS_ALC = {
    "beer","lager","ipa","stout","porter","ale","pilsner","cider",
    "wine","red","white","rosé","cabernet","pinot","merlot","riesling","sauvignon",
    "champagne","prosecco","sparkling wine",
    "cocktail","margarita","martini","mojito","old fashioned","negroni","manhattan",
    "whiskey","bourbon","rye","scotch","vodka","gin","rum","tequila","mezcal","soju","sake",
    "liqueur","abv","proof"
}
SHARED_FRIER_PHRASES = [
    "shared fryer","common fryer","fried in same oil","prepared in a shared fryer",
    "shared toaster","shared equipment","shared grill","prepared on shared surfaces"
]

# ---------- Helpers ----------
def ensure_api_key(cli_key: str | None = None):
    key = cli_key or os.getenv("GEMINI_API_KEY")
    if not key:
        print("[ERROR] Provide --api_key or set GEMINI_API_KEY.")
        sys.exit(1)
    genai.configure(api_key=key)

def read_master_columns(master_xlsx: Path) -> list[str]:
    df = pd.read_excel(master_xlsx, nrows=0)
    return list(df.columns)

def normalize_units(value):
    if value is None:
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    s2 = re.sub(r"[^\d\.\-]", "", s)
    if s2 in {"", ".", "-"}:
        return ""
    try:
        f = float(s2)
        return int(f) if f.is_integer() else f
    except:
        return ""

def detect_drink_flags(category: str, dish: str) -> tuple[str, str]:
    text = f"{category or ''} {dish or ''}".lower()
    is_alc = any(k in text for k in DRINK_KEYWORDS_ALC)
    is_non = any(k in text for k in DRINK_KEYWORDS_NONALC)
    return ("Yes" if is_alc else "", "Yes" if (is_non and not is_alc) else "")

def is_scanned_pdf(pdf_path: Path, sample_pages=2) -> bool:
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for p in pdf.pages[:sample_pages]:
                tx = p.extract_text() or ""
                if tx.strip():
                    return False
    except Exception:
        pass
    return True

def try_ocr(pdf_path: Path) -> Path:
    if shutil.which("ocrmypdf") is None:
        return pdf_path
    tmpdir = Path(tempfile.mkdtemp())
    outpdf = tmpdir / f"ocr_{pdf_path.name}"
    cmd = ["ocrmypdf", "--force-ocr", "--skip-text", "--optimize", "0", str(pdf_path), str(outpdf)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return outpdf
    except Exception:
        return pdf_path

def harvest_text(pdf_path: Path, max_pages=14) -> str:
    try:
        out = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pg in pdf.pages[:max_pages]:
                tx = pg.extract_text() or ""
                out.append(tx)
        return "\n".join(out)
    except Exception:
        return ""

def detect_shared_fryer_any(texts: list[str]) -> str:
    blob = "\n".join(t.lower() for t in texts)
    return "Yes" if any(ph in blob for ph in SHARED_FRIER_PHRASES) else ""

def coerce_rows_to_master(rows: list[dict], master_cols: list[str]) -> pd.DataFrame:
    fixed = []
    for r in rows:
        fr = {c: r.get(c, "") for c in master_cols}
        for nc in ["Calories","Fat Calories","Total Fat(g)","Saturated Fat(g)","Trans Fat(g)",
                   "Cholestrol(mg)","Sodium(mg)","Total Carbs(g)","Fiber(g)","Sugars(g)","Protein(g)"]:
            if nc in fr:
                fr[nc] = normalize_units(fr[nc])
        fixed.append(fr)
    return pd.DataFrame(fixed, columns=master_cols)

# ---------- Gemini prompting ----------
GEMINI_SYSTEM_INSTRUCTIONS = """You are a meticulous data extraction agent for restaurant menus.
Goal: Convert menu/allergen/nutrition PDFs into rows (one row per menu item) with a fixed column schema.
Rules:
- Output ONLY valid JSON (no commentary). Use an array of objects.
- Keys MUST match the provided 'columns' exactly (spelling, spaces, punctuation).
- If a value is unknown or not applicable, use an empty string "".
- For boolean-like flags (e.g., Is_Vegan), use "Yes" for true, else "".
- 'Common_Frier_Toaster' must be "Yes" if the document mentions shared fryer/toaster equipment.
- Classify drinks: mark 'Is_Alcoholic_Drink' OR 'Is_NonAlcoholic_Drink' with "Yes" (at most one should be Yes).
- If 'Contains_Wheat' is "Yes" and 'Contains_Gluten' is "", set 'Contains_Gluten' to "Yes".
- Keep nutrition fields numeric where possible (no units), otherwise "".
- Use per-serving values when ambiguous.
- Do NOT add extra keys or structures.
"""

def _parse_json_strict(raw: str):
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return json.loads(s)

def _discover_models(preferred: str | None) -> list[str]:
    """
    Ask the API which models are available for generateContent and
    build a prioritized candidate list. Works across old/new client versions.
    """
    cands = []
    try:
        models = list(genai.list_models())
        names = []
        for m in models:
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                # m.name is like "models/gemini-1.5-flash-latest"
                n = getattr(m, "name", "")
                if n:
                    names.append(n.split("/")[-1])
        # Prefer 1.5 (flash first), then 1.0/pro
        names_15 = [n for n in names if "1.5" in n]
        names_10 = [n for n in names if "1.0" in n or n.endswith("-pro") or n == "gemini-pro"]
        # flash before pro
        names_15 = sorted(names_15, key=lambda x: (("flash" not in x), ("latest" not in x), x))
        names_10 = sorted(names_10, key=lambda x: (("latest" not in x), x))
        cands = names_15 + names_10
    except Exception as e:
        # Fallback static list covering old & new names
        cands = [
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro-latest",
            "gemini-pro",
        ]
    # Put user preference first if supplied
    if preferred:
        p = preferred.split("/")[-1]
        if p in cands:
            cands.remove(p)
        cands.insert(0, p)
    # Deduplicate
    seen, ordered = set(), []
    for n in cands:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered

def _generate_json_with_fallback(model_candidates: list[str], parts: list, gen_cfg: dict) -> list[dict]:
    last_exc = None
    for name in model_candidates:
        try:
            model = genai.GenerativeModel(name, system_instruction=GEMINI_SYSTEM_INSTRUCTIONS)
            resp = model.generate_content(parts, generation_config=gen_cfg)
            return _parse_json_strict(resp.text)
        except Exception as e:
            print(f"[WARN] Gemini generate failed on model '{name}': {e}")
            last_exc = e
    raise last_exc or RuntimeError("Gemini generation failed")

def gemini_extract_from_pdf(model_name: str, pdf_path: Path, columns: list[str], restaurant: str) -> list[dict]:
    """
    Text-only mode by default (robust across client versions and avoids ragStoreName).
    """
    use_pdf = try_ocr(pdf_path) if is_scanned_pdf(pdf_path) else pdf_path

    prompt = {
        "task": "Extract rows for the given restaurant menu/allergen/nutrition document.",
        "restaurant": restaurant,
        "columns": columns,
        "strict_json_schema": {c: "string" for c in columns},
        "notes": [
            "Return ONLY a JSON array of objects. No prose.",
            "Keys must match 'columns' exactly and be present in every object.",
            "Use \"Yes\" for true flags; otherwise empty string \"\".",
            "Do not fabricate nutrition; leave empty if unknown."
        ],
    }
    gen_cfg = {"response_mime_type": "application/json", "temperature": 0.2}
    model_candidates = _discover_models(model_name)

    # Prefer text mode (feed extracted text)
    try:
        text_blob = harvest_text(use_pdf, max_pages=14)
        return _generate_json_with_fallback(model_candidates, [json.dumps(prompt), text_blob], gen_cfg)
    except Exception as e_text:
        print(f"[WARN] Text mode failed on {pdf_path.name}: {e_text}")

    # Final minimal placeholder so the sheet exists for manual edits
    return [{
        "Category Name": "",
        "Dish Name": f"[Placeholder] Could not parse {pdf_path.name}",
        "Common_Frier_Toaster": "",
        "Contains_Peanuts": "", "Contains_TreeNuts": "", "Contains_Soy": "", "Contains_Eggs": "",
        "Contains_Dairy": "", "Contains_Wheat": "", "Contains_Fish": "",
        "Contains_MolluscanShellfish": "", "Contains_CrustaceanShellfish": "",
        "Contains_Sesame Seeds": "", "Contains_Gluten": "", "Contains_Garlic": "",
        "Contains_Sulphites": "", "Is_Vegan": "", "Is_Vegetarian": "", "Contains_Celery": "",
        "Gluten_Free_Alternate": "", "Vegetarian_Alternate": "", "Vegan_Alternate": "",
        "Contains_Mustard": "", "Contains_Lupen": "",
        "Is_Alcoholic_Drink": "", "Is_NonAlcoholic_Drink": "",
        "Calories": "", "Fat Calories": "", "Total Fat(g)": "", "Saturated Fat(g)": "",
        "Trans Fat(g)": "", "Cholestrol(mg)": "", "Sodium(mg)": "",
        "Total Carbs(g)": "", "Fiber(g)": "", "Sugars(g)": "", "Protein(g)": ""
    }]

def guess_doc_type(name: str) -> str:
    name = name.lower()
    if any(k in name for k in ["allergen","allergy","contains"]): return "allergen"
    if any(k in name for k in ["ingredient","ingredients"]): return "ingredient"
    return "menu"

def process_restaurant_folder(restaurant_dir: Path, master_cols: list[str], model_name: str) -> pd.DataFrame:
    restaurant = restaurant_dir.name

    # Windows-safe PDF de-duplication
    all_paths = [p for p in restaurant_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    seen = set()
    pdfs = []
    for p in all_paths:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            pdfs.append(p)
    pdfs.sort(key=lambda x: x.name.lower())

    all_rows = []
    text_blobs = []

    for pdf in pdfs:
        print(f"   - Using Gemini on: {pdf.name}")
        rows = gemini_extract_from_pdf(model_name, pdf, master_cols, restaurant)
        df_rows = coerce_rows_to_master(rows, master_cols)
        all_rows.append(df_rows)

        text_blobs.append(harvest_text(try_ocr(pdf) if is_scanned_pdf(pdf) else pdf, max_pages=14))

    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
    else:
        df = pd.DataFrame([{c: "" for c in master_cols}], columns=master_cols)
        df.loc[0, "Dish Name"] = f"[Placeholder] No PDFs parsed for {restaurant}"

    # Drinks safety pass
    for i, r in df.iterrows():
        alc, non = detect_drink_flags(r.get("Category Name",""), r.get("Dish Name",""))
        if not r.get("Is_Alcoholic_Drink",""):
            df.at[i, "Is_Alcoholic_Drink"] = alc
        if not r.get("Is_NonAlcoholic_Drink",""):
            df.at[i, "Is_NonAlcoholic_Drink"] = non

    # Shared fryer doc-level backstop ONLY
    shared = detect_shared_fryer_any(text_blobs)
    if shared == "Yes":
        df["Common_Frier_Toaster"] = df["Common_Frier_Toaster"].replace("", "Yes")

    return df

def save_excel(df: pd.DataFrame, restaurant: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9 _-]+","", restaurant).strip() or "Restaurant"
    xlsx = out_dir / f"{safe}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return xlsx

def upsert_duckdb(df: pd.DataFrame, restaurant: str, db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS menu_items (
            restaurant TEXT,
            "Category Name" TEXT,
            "Dish Name" TEXT,
            "Common_Frier_Toaster" TEXT,
            "Contains_Peanuts" TEXT,
            "Contains_TreeNuts" TEXT,
            "Contains_Soy" TEXT,
            "Contains_Eggs" TEXT,
            "Contains_Dairy" TEXT,
            "Contains_Wheat" TEXT,
            "Contains_Fish" TEXT,
            "Contains_MolluscanShellfish" TEXT,
            "Contains_CrustaceanShellfish" TEXT,
            "Contains_Sesame Seeds" TEXT,
            "Contains_Gluten" TEXT,
            "Contains_Garlic" TEXT,
            "Contains_Sulphites" TEXT,
            "Is_Vegan" TEXT,
            "Is_Vegetarian" TEXT,
            "Contains_Celery" TEXT,
            "Gluten_Free_Alternate" TEXT,
            "Vegetarian_Alternate" TEXT,
            "Vegan_Alternate" TEXT,
            "Contains_Mustard" TEXT,
            "Contains_Lupen" TEXT,
            "Is_Alcoholic_Drink" TEXT,
            "Is_NonAlcoholic_Drink" TEXT,
            "Calories" TEXT,
            "Fat Calories" TEXT,
            "Total Fat(g)" TEXT,
            "Saturated Fat(g)" TEXT,
            "Trans Fat(g)" TEXT,
            "Cholestrol(mg)" TEXT,
            "Sodium(mg)" TEXT,
            "Total Carbs(g)" TEXT,
            "Fiber(g)" TEXT,
            "Sugars(g)" TEXT,
            "Protein(g)" TEXT
        )
    """)
    dfc = df.copy()
    dfc.insert(0, "restaurant", restaurant)
    con.register("tmp_df", dfc)
    con.execute("INSERT INTO menu_items SELECT * FROM tmp_df")
    con.close()

def main():
    ap = argparse.ArgumentParser(description="Menu PDF → Excel per restaurant + DuckDB (Gemini-powered).")
    ap.add_argument("--raw_dir", required=True, help="data/raw with subfolders per restaurant")
    ap.add_argument("--master_xlsx", required=True, help="Path to Master_Template.xlsx")
    ap.add_argument("--out_dir", default="data/processed", help="Output Excels folder")
    ap.add_argument("--db_path", default="data/db/menus.duckdb", help="DuckDB path")
    ap.add_argument("--model", default="gemini-1.5-flash-latest", help="Preferred Gemini model")
    ap.add_argument("--api_key", help="Override GEMINI_API_KEY via CLI")
    args = ap.parse_args()

    ensure_api_key(args.api_key)

    raw_dir = Path(args.raw_dir)
    master_xlsx = Path(args.master_xlsx)
    out_dir = Path(args.out_dir)
    db_path = Path(args.db_path)

    if not raw_dir.exists():
        print(f"[ERROR] raw_dir not found: {raw_dir}"); sys.exit(1)
    if not master_xlsx.exists():
        print(f"[ERROR] master_xlsx not found: {master_xlsx}"); sys.exit(1)

    master_cols = read_master_columns(master_xlsx)

    restaurants = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not restaurants:
        print(f"[WARN] No restaurant folders under {raw_dir}")

    for rdir in restaurants:
        print(f"[INFO] Restaurant: {rdir.name}")
        df = process_restaurant_folder(rdir, master_cols, args.model)
        xlsx = save_excel(df, rdir.name, out_dir)
        upsert_duckdb(df, rdir.name, db_path)
        print(f"[OK] {rdir.name}: wrote {xlsx.name}, appended to {db_path}")

    print("\nDone.\nExample query:")
    print(f'  duckdb {db_path}')
    print('  SELECT restaurant, "Dish Name", "Calories" FROM menu_items WHERE "Is_Alcoholic_Drink" = \'Yes\';')

if __name__ == "__main__":
    main()
