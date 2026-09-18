"""FastAPI backend for PrivacyGuard."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.redection.privacy_firewall import privacy_guard

app = FastAPI(title="PrivacyGuard API", version="1.0.0")
class ScanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    confidence_threshold: float = Field(default=.80, ge=0, le=1)

@app.get("/health")
def health() -> dict[str, str]: return {"status": "ok"}

@app.post("/scan")
def scan(request: ScanRequest) -> dict:
    try: return privacy_guard(request.text, request.confidence_threshold)
    except (TypeError, ValueError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000)
