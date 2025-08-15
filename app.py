import os
import json
from typing import Any, List, Optional, Literal

import boto3
import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, conint

# ================== ENV ==================
R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET = os.getenv("R2_BUCKET")
SIGN_TTL = int(os.getenv("R2_SIGN_TTL", "3600"))  # signed URL TTL
CATALOG_KEY = os.getenv("CATALOG_KEY", "catalog.json")  # location in bucket

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
    description="Generic, schema-agnostic query gateway for large CSV/Parquet in R2."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== DUCKDB ==================
def new_conn():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("SET enable_http_metadata_cache=true;")
    con.execute("SET http_metadata_cache_max_entries=20000;")
    con.execute("PRAGMA threads=%d" % max(2, os.cpu_count() or 4))
    return con

# ================== HELPERS ==================
def load_catalog() -> dict:
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
    return '"' + name.replace('"', '""') + '"'

def read_table_sql(url: str, fmt: Optional[str], coerce: Literal["all_varchar","auto"]="auto") -> str:
    if fmt == "parquet" or (fmt is None and url.lower().endswith(".parquet")):
        return f"parquet_scan('{url}')"
    if coerce == "all_varchar":
        return f"read_csv_auto('{url}', HEADER=TRUE, ALL_VARCHAR=TRUE)"
    return f"read_csv_auto('{url}', HEADER=TRUE)"

def build_filter_sql(f: dict) -> str:
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
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        v_sql = str(val)
    else:
        v_sql = "'" + str(val).replace("'", "''") + "'"
    return f"{col} {op} {v_sql}"

def ensure_order_by(select_cols: List[str], order_by: Optional[List[dict]]) -> str:
    if order_by:
        parts = []
        for ob in order_by:
            c = quote_ident(ob["col"])
            d = ob.get("dir","asc").upper()
            if d not in ("ASC","DESC"):
                d = "ASC"
            parts.append(f"{c} {d}")
        return " ORDER BY " + ", ".join(parts)
    if select_cols:
        return " ORDER BY " + quote_ident(select_cols[0]) + " ASC"
    return ""

# ================== Pydantic models ==================
class PreviewRequest(BaseModel):
    datasets: List[str] = Field(..., description="Catalog names or raw R2 keys")
    limit: conint(ge=1, le=2000) = Field(100)
    coerce: Literal["all_varchar", "auto"] = Field("all_varchar")
    columns_case: Literal["lower", "preserve"] = Field("lower")

class Filter(BaseModel):
    col: str
    op: Literal["=", "!=", ">", ">=", "<", "<=", "IN", "NOT IN", "LIKE", "BETWEEN"] = "="
    value: Any

class OrderBy(BaseModel):
    col: str
    dir: Literal["asc", "desc"] = "asc"

class Aggregation(BaseModel):
    alias: str
    expr: str

class QueryRequest(BaseModel):
    datasets: List[str] = Field(..., description="Catalog names or raw R2 keys")
    mode: Literal["union", "join"] = "union"
    on: List[str] = Field(default_factory=list)
    select: List[str] = Field(default_factory=list)
    filters: List[Filter] = Field(default_factory=list)
    aggregations: List[Aggregation] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    order_by: Optional[List[OrderBy]] = None
    page: conint(ge=1) = 1
    page_size: conint(ge=1, le=10_000) = 1000
    coerce: Literal["all_varchar", "auto"] = "auto"
    columns_case: Literal["lower", "preserve"] = "lower"

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
    return {"datasets": cat}

@app.get("/schema")
def schema(source: str, coerce: Literal["all_varchar","auto"]="all_varchar"):
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

# ================== PREVIEW ==================
@app.post("/preview")
def preview(body: PreviewRequest):
    datasets = body.datasets
    limit = body.limit
    coerce = body.coerce
    case = body.columns_case
    if not datasets:
        raise HTTPException(400, "datasets is required")

    cat = load_catalog()
    items = []
    for ds in datasets:
        if ds in cat:
            fmt = cat[ds].get("format")
            for k in cat[ds].get("keys", []):
                items.append(read_table_sql(r2_signed_url(k), fmt, coerce))
        else:
            items.append(read_table_sql(r2_signed_url(ds), None, coerce))

    con = new_conn()
    try:
        union_sql = " UNION BY NAME ALL ".join([f"SELECT * FROM {t}" for t in items])
        q = f"SELECT * FROM ({union_sql}) LIMIT {limit}"
        df = con.execute(q).df()
        df = normalize_cols(df, case)
        return {"count": len(df), "rows": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, f"Error previewing data: {e}")
    finally:
        con.close()

# ================== QUERY ==================
@app.post("/query")
def query(body: QueryRequest):
    cat = load_catalog()
    datasets = body.datasets
    mode = body.mode
    on_keys = body.on
    select_cols = body.select
    filters = [f.dict() for f in body.filters]
    aggs = [a.dict() for a in body.aggregations]
    group_by = body.group_by
    order_by = [o.dict() for o in body.order_by] if body.order_by else None
    page = body.page
    page_size = body.page_size
    coerce = body.coerce
    case = body.columns_case

    if not datasets:
        raise HTTPException(400, "datasets is required")

    table_exprs: List[str] = []
    for ds in datasets:
        if ds in cat:
            fmt = cat[ds].get("format")
            keys = cat[ds].get("keys", [])
            if not keys:
                raise HTTPException(404, f"Catalog entry '{ds}' has no keys")
            for k in keys:
                table_exprs.append(read_table_sql(r2_signed_url(k), fmt, coerce))
        else:
            table_exprs.append(read_table_sql(r2_signed_url(ds), None, coerce))

    con = new_conn()
    try:
        if mode == "union":
            base_sql = " UNION BY NAME ALL ".join([f"SELECT * FROM {t}" for t in table_exprs])
            base_cte = f"base AS ({base_sql})"
            from_sql = "FROM base"
        elif mode == "join":
            if len(table_exprs) < 2:
                raise HTTPException(400, "join mode requires at least two datasets")
            if not on_keys:
                raise HTTPException(400, "join mode requires 'on' keys")
            join_sql = f"SELECT * FROM {table_exprs[0]} t0 "
            for i, t in enumerate(table_exprs[1:], start=1):
                on_clause = " AND ".join([f"t0.{quote_ident(k)} = t{i}.{quote_ident(k)}" for k in on_keys])
                join_sql += f"LEFT JOIN {t} t{i} ON {on_clause} "
            base_cte = f"base AS ({join_sql})"
            from_sql = "FROM base"
        else:
            raise HTTPException(400, "mode must be 'union' or 'join'")

        where_sql = ""
        if filters:
            parts = [build_filter_sql(f) for f in filters]
            where_sql = " WHERE " + " AND ".join(parts)

        proj_parts = [quote_ident(c) for c in select_cols] if select_cols else ["*"]

        agg_parts = []
        if aggs:
            for a in aggs:
                alias = quote_ident(a["alias"])
                expr = a["expr"]
                agg_parts.append(f"{expr} AS {alias}")

        if aggs and not group_by:
            select_sql = ", ".join(agg_parts)
        elif aggs and group_by:
            gb_cols = ", ".join([quote_ident(c) for c in group_by])
            proj = ", ".join([quote_ident(c) for c in group_by] + agg_parts)
            group_sql = f" GROUP BY {gb_cols}"
        else:
            select_sql = ", ".join(proj_parts)

        order_sql = ensure_order_by(select_cols or group_by, order_by)
        offset = (page - 1) * page_size
        limit_sql = f" LIMIT {page_size} OFFSET {offset}"

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
            "sql": sql
        }
    except Exception as e:
        raise HTTPException(500, f"Error executing query: {e}")
    finally:
        con.close()
