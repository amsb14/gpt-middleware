import os
import boto3
import duckdb
import io
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ====== ENV VARS ======
R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET = os.getenv("R2_BUCKET")

# ====== S3 CLIENT FOR R2 ======
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
)

# ====== FASTAPI APP ======
app = FastAPI(
    title="R2 Proxy",
    version="1.1.0",
    servers=[{"url": "https://gpt-middleware-gvo9.onrender.com", "description": "Render deployment"}],
    description="Proxy to list, preview, and query large CSV files from a private R2 bucket without loading entire files into GPT."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or lock to chat.openai.com
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== HELPERS ======
def r2_signed_url(key: str) -> str:
    """Generate a temporary signed URL for a private R2 object."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET, "Key": key},
        ExpiresIn=300
    )

# ====== BASIC ROUTES ======
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/list")
def list_files(prefix: str = ""):
    try:
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        files = [
            {
                "name": o["Key"],
                "size": o["Size"],
                "last_modified": o["LastModified"].isoformat()
            }
            for o in resp.get("Contents", [])
        ]
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

# ====== CSV ENDPOINTS ======
@app.get("/head_csv")
def head_csv(name: str, n: int = 100):
    """Preview the first N rows of a CSV file."""
    url = r2_signed_url(name)
    con = duckdb.connect()
    try:
        nrows = max(1, min(n, 1000))
        rows = con.execute(
            f"SELECT * FROM read_csv_auto('{url}', HEADER=TRUE) LIMIT {nrows}"
        ).df()
        return {"count": len(rows), "rows": rows.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, f"Error reading CSV: {e}")
    finally:
        con.close()

@app.get("/query_csv")
def query_csv(
    name: str,
    tickers: list[str] = Query(default=[]),
    ticker_col: str = "ticker",
    date_col: str = "date",
    start: str | None = None,
    end: str | None = None,
    columns: list[str] = Query(default=[]),
    limit: int = 1000
):
    """Filter CSV server-side; skips filters if columns are missing."""
    url = r2_signed_url(name)
    con = duckdb.connect()
    try:
        # Load only the header first to see available columns
        header_df = con.execute(f"SELECT * FROM read_csv_auto('{url}', HEADER=TRUE, SAMPLE_SIZE=1)").df()
        available_cols = set(header_df.columns)

        where_clauses = []

        # Apply ticker filter only if column exists and tickers provided
        if tickers and ticker_col in available_cols:
            in_list = ",".join([f"'{t.upper()}'" for t in tickers])
            where_clauses.append(f"upper({ticker_col}) IN ({in_list})")

        # Apply date filters only if column exists
        if start and date_col in available_cols:
            where_clauses.append(f"{date_col} >= '{start}'")
        if end and date_col in available_cols:
            where_clauses.append(f"{date_col} <= '{end}'")

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Columns to select
        cols_sql = ", ".join(columns) if columns else "*"
        lim = max(1, min(limit, 10000))

        q = f"SELECT {cols_sql} FROM read_csv_auto('{url}', HEADER=TRUE) {where_sql} LIMIT {lim}"
        rows = con.execute(q).df()
        return {"count": len(rows), "rows": rows.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, f"Error querying CSV: {e}")
    finally:
        con.close()
