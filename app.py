import os, boto3
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

R2_KEY     = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET  = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET  = os.getenv("R2_BUCKET")

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
)

app = FastAPI(
    title="R2 Proxy",
    version="1.0.0",
    servers=[
        {"url": "https://gpt-middleware-gvo9.onrender.com", "description": "Render deployment"}
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or lock to chat.openai.com if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/list")
def list_files(prefix: str = ""):
    try:
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        files = [o["Key"] for o in resp.get("Contents", [])]
        return {"files": files}
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
