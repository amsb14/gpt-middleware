import os
import io
import json
from typing import List, Optional, Literal, Dict, Any

import boto3
import duckdb
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ================== ENV ==================
R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET = os.getenv("R2_BUCKET")
SIGN_TTL = int(os.getenv("R2_SIGN_TTL", "3600"))  # longer TTL for big scans

CATALOG_KEY = os.getenv("CATALOG_KEY", "catalog.json")  # where the tiny catalog lives

# ================== R2 CLIENT ==================
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
)

def r2_signed_url(key: str) -> str:
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=SIGN_TTL
    )

def r2_get_text(key: str) -> Optional[str]:
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return obj["Body"].read().decode("utf-8")
    except Exception:
        return None

def r2_list(prefix: str = "") -> List[dict]:
    resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
    return [
        {"name": o["Key"], "size": o["Size"], "last_modified": o["LastModified"].isoformat()}
        for o in resp.get("Contents", [])
    ]

# ================== FASTAPI ==================
app = FastAPI(
    title="R2 Data Gateway",
    version="2.0.0",
    servers=[{"url": "https://gpt-middleware-gvo9.onrender.com", "description": "Render deployment"}],
    description="Generic, schema-agnostic query gateway for large CSV/Parquet in R2, designed for ChatGPT Actions."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to chat.openai.com in prod if you prefer
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== DUCKDB ==================
def new_conn():
    con = duckdb.connect()
    # HTTP/Parquet performance knobs
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("SET enable_http_metadata_cache=true;")
    con.execute("SET http_metadata_cache_max_entries=20000;")
    # Reasonable parallelism by default
    con.execute("PRAGMA threads=%d" % max(2, os.cpu_count() or 4))
    return con

# ================== HELPERS ==================
def load_catalog() -> Dict[str, Any]:
    txt = r2_get_text(CATALOG_KEY)
    if not txt:
        return {}
    try:
        return json.loads(txt)
    except Exception:
        return {}

def normalize_cols(df, case: Literal["lower","preserve"]="lower"):
    if df is None:
        return df
    if case == "lower":
        df.columns = [c.lower() for c in df.columns]
    return df

def quote_ident(name: str) -> str:
    # Minimal identifier quoting
    return '"' + name.replace('"', '""') + '"'

def read_table_sql(url: str, fmt: Optional[str], coerce: Literal["all_varchar","auto"]="auto") -> str:
    # Return a table expression for DuckDB
    # Prefer Parquet scan if extension/format suggests it
    if fmt == "parquet" or (fmt is None and url.lower().endswith(".parquet")):
        return f"parquet_scan('{url}')"
    # CSV fallback
    if coerce == "all_varchar":
        return f"read_csv_auto('{url}', HEADER=TRUE, ALL_VARCHAR=TRUE)"
    return f"read_csv_auto('{url}', HEADER=TRUE)"

def build_filter_sql(f: Dict[str, Any]) -> str:
    col = quote_ident(f["col"])
    op = f.get("op", "=").upper().strip()
    val = f.get("value")
    if op in ("IN","NOT IN") and isinstance(val, list):
        items = []
        for v in val:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                items.append(str(v))
            else:
                items.append("'" + str(v).replace("'", "''") + "'")
        return f"{col} {op} ({', '.join(items)})"
    if op == "BETWEEN" and isinstance(val, list) and len(val) == 2:
        a, b = val
        a_sql = str(a) if isinstance(a, (int, float)) else "'" + str(a).replace("'", "''") + "'"
        b_sql = str(b) if isinstance(b, (int, float)) else "'" + str(b).replace("'", "''") + "'"
        return f"{col} BETWEEN {a_sql} AND {b_sql}"
    # default scalar
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        v_sql = str(val)
    else:
        v_sql = "'" + str(val).replace("'", "''") + "'"
    return f"{col} {op} {v_sql}"

def ensure_order_by(select_cols: List[str], order_by: Optional[List[Dict[str,str]]]) -> str:
    if order_by:
        parts = []
        for ob in order_by:
            c = quote_ident(ob["col"])
            d = ob.get("dir","asc").upper()
            if d not in ("ASC","DESC"):
                d = "ASC"
            parts.append(f"{c} {d}")
        return " ORDER BY " + ", ".join(parts)
    # fallback to first column for determinism if exists
    if select_cols:
        return " ORDER BY " + quote_ident(select_cols[0]) + " ASC"
    return ""

# ================== BASIC ROUTES ==================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/list")
def list_files(prefix: str = ""):
    try:
        files = r2_list(prefix)
        return {"count": len(files), "files": files}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/file")
def get_file(name: str):
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=name)
        headers = {"Cache-Control": "no-store, no-cache, must-revalidate"}
        return StreamingResponse(obj["Body"], media_type="application/octet-stream", headers=headers)
    except Exception as e:
        raise HTTPException(404, str(e))

# ================== CATALOG & SCHEMA ==================
@app.get("/catalog")
def get_catalog():
    cat = load_catalog()
    # Optionally enrich with last_modified/size for the first key
    for ds, meta in cat.items():
        keys = meta.get("keys", [])
        if keys:
            try:
                # cheap-enrich: only first key
                for f in r2_list(keys[0].rsplit("/",1)[0] if "/" in keys[0] else ""):
                    if f["name"] == keys[0]:
                        meta["first_key_size"] = f["size"]
                        meta["first_key_last_modified"] = f["last_modified"]
                        break
            except Exception:
                pass
    return {"datasets": cat}

@app.get("/schema")
def schema(source: str, coerce: Literal["all_varchar","auto"]="all_varchar"):
    """
    source can be a catalog dataset name or a raw R2 key.
    Returns column names/types (types are best-effort for CSV).
    """
    cat = load_catalog()
    fmt = None
    urls = []
    if source in cat:
        fmt = cat[source].get("format")
        keys = cat[source].get("keys", [])
        if not keys:
            raise HTTPException(404, f"Catalog entry '{source}' has no keys")
        urls = [r2_signed_url(keys[0])]
    else:
        urls = [r2_signed_url(source)]

    con = new_conn()
    try:
        texpr = read_table_sql(urls[0], fmt, coerce=coerce)
        df = con.execute(f"SELECT * FROM {texpr} LIMIT 1").df()
        cols = [{"name": c, "type": str(df[c].dtype)} for c in df.columns]
        return {"columns": cols}
    except Exception as e:
        raise HTTPException(500, f"Error reading schema: {e}")
    finally:
        con.close()

# ================== PREVIEW (multi-format) ==================
@app.post("/preview")
def preview(body: Dict[str, Any] = Body(...)):
    """
    body = {
      "datasets": ["income_statements", "balance_summaries"] or R2 keys,
      "limit": 100,
      "coerce": "all_varchar"|"auto",
      "columns_case": "lower"|"preserve"
    }
    """
    datasets: List[str] = body.get("datasets", [])
    limit: int = max(1, min(int(body.get("limit", 100)), 2000))
    coerce: Literal["all_varchar","auto"] = body.get("coerce","all_varchar")
    case: Literal["lower","preserve"] = body.get("columns_case","lower")

    if not datasets:
        raise HTTPException(400, "datasets is required")

    cat = load_catalog()
    items = []
    for ds in datasets:
        if ds in cat:
            fmt = cat[ds].get("format")
            for k in cat[ds].get("keys", []):
                items.append((read_table_sql(r2_signed_url(k), fmt, coerce), cat[ds].get("aliases", {})))
        else:
            items.append((read_table_sql(r2_signed_url(ds), None, coerce), {}))

    con = new_conn()
    try:
        # Build UNION BY NAME preview
        union_sql = " UNION BY NAME ALL ".join([f"SELECT * FROM {t}" for t, _ in items])
        q = f"SELECT * FROM ({union_sql}) LIMIT {limit}"
        df = con.execute(q).df()
        df = normalize_cols(df, case)
        return {"count": len(df), "rows": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, f"Error previewing data: {e}")
    finally:
        con.close()

# ================== GENERIC QUERY ==================
@app.post("/query")
def query(body: Dict[str, Any] = Body(...)):
    """
    Generic, schema-agnostic query.

    Example body:
    {
      "datasets": ["income_statements", "finance/prices_2024.parquet"],
      "mode": "union",   // "union" or "join"
      "on": ["ticker","date"], // for join
      "select": ["ticker","date","eps","revenue"],
      "filters": [{"col":"date","op":">=","value":"2021-01-01"}, {"col":"ticker","op":"IN","value":["AAPL","MSFT"]}],
      "aggregations": [{"alias":"avg_eps","expr":"avg(eps)"}],
      "group_by": ["ticker"],
      "order_by": [{"col":"ticker","dir":"asc"}],
      "page": 1,
      "page_size": 500,
      "coerce": "auto",            // or "all_varchar" for messy CSV
      "columns_case": "lower"      // or "preserve"
    }
    """
    cat = load_catalog()

    datasets: List[str] = body.get("datasets", [])
    mode: Literal["union","join"] = body.get("mode", "union")
    on_keys: List[str] = body.get("on", [])
    select_cols: List[str] = body.get("select", [])
    filters: List[Dict[str, Any]] = body.get("filters", [])
    aggs: List[Dict[str, str]] = body.get("aggregations", [])
    group_by: List[str] = body.get("group_by", [])
    order_by: Optional[List[Dict[str,str]]] = body.get("order_by", None)
    page: int = max(1, int(body.get("page", 1)))
    page_size: int = min(10_000, max(1, int(body.get("page_size", 1000))))
    coerce: Literal["all_varchar","auto"] = body.get("coerce","auto")
    case: Literal["lower","preserve"] = body.get("columns_case","lower")

    if not datasets:
        raise HTTPException(400, "datasets is required")

    # Resolve datasets → table expressions and alias maps
    table_exprs: List[str] = []
    alias_maps: List[Dict[str,str]] = []
    for ds in datasets:
        if ds in cat:
            fmt = cat[ds].get("format")
            aliases = cat[ds].get("aliases", {})
            keys = cat[ds].get("keys", [])
            if not keys:
                raise HTTPException(404, f"Catalog entry '{ds}' has no keys")
            for k in keys:
                table_exprs.append(read_table_sql(r2_signed_url(k), fmt, coerce))
                alias_maps.append(aliases)
        else:
            table_exprs.append(read_table_sql(r2_signed_url(ds), None, coerce))
            alias_maps.append({})

    con = new_conn()
    try:
        # Build base relation
        if mode == "union":
            base_sql = " UNION BY NAME ALL ".join([f"SELECT * FROM {t}" for t in table_exprs])
            base_cte = f"base AS ({base_sql})"
            from_sql = "FROM base"
        elif mode == "join":
            if len(table_exprs) < 2:
                raise HTTPException(400, "join mode requires at least two datasets")
            if not on_keys:
                raise HTTPException(400, "join mode requires 'on' keys")
            # Simple left-to-right joins on the provided keys
            join_sql = f"SELECT * FROM {table_exprs[0]} t0 "
            for i, t in enumerate(table_exprs[1:], start=1):
                on_clause = " AND ".join([f"coalesce(t0.{quote_ident(k)}, t0.{quote_ident(k.lower())}) = coalesce(t{i}.{quote_ident(k)}, t{i}.{quote_ident(k.lower())})" for k in on_keys])
                join_sql += f"LEFT JOIN {t} t{i} ON {on_clause} "
            base_cte = f"base AS ({join_sql})"
            from_sql = "FROM base"
        else:
            raise HTTPException(400, "mode must be 'union' or 'join'")

        # Filters
        where_sql = ""
        if filters:
            parts = [build_filter_sql(f) for f in filters]
            where_sql = " WHERE " + " AND ".join(parts)

        # Projection
        proj_parts = []
        if select_cols:
            for c in select_cols:
                proj_parts.append(quote_ident(c))
        else:
            proj_parts = ["*"]

        # Aggregations / Grouping
        agg_parts = []
        if aggs:
            for a in aggs:
                alias = quote_ident(a["alias"])
                expr = a["expr"]
                agg_parts.append(f"{expr} AS {alias}")

        if aggs and not group_by:
            # If aggregations exist without group_by, everything collapses to a single row
            select_sql = ", ".join(agg_parts)
        elif aggs and group_by:
            gb_cols = ", ".join([quote_ident(c) for c in group_by])
            proj = ", ".join([quote_ident(c) for c in group_by] + agg_parts)
            group_sql = f" GROUP BY {gb_cols}"
            # We'll combine later
        else:
            select_sql = ", ".join(proj_parts)

        # Pagination
        order_sql = ensure_order_by(select_cols or group_by, order_by)
        offset = (page - 1) * page_size
        limit_sql = f" LIMIT {page_size} OFFSET {offset}"

        # Final SQL assembly
        if aggs and group_by:
            sql = f"WITH {base_cte} SELECT {proj} {from_sql} {where_sql} {group_sql} {order_sql} {limit_sql}"
        else:
            sql = f"WITH {base_cte} SELECT {select_sql} {from_sql} {where_sql} {order_sql} {limit_sql}"

        df = con.execute(sql).df()
        df = normalize_cols(df, case)
        return {
            "count": len(df),
            "page": page,
            "page_size": page_size,
            "rows": df.to_dict(orient="records"),
            "sql": sql  # helpful for debugging; remove in prod if you want
        }
    except Exception as e:
        raise HTTPException(500, f"Error executing query: {e}")
    finally:
        con.close()
