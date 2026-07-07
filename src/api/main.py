from fastapi import FastAPI

from src.api.websocket import router as ws_router

app = FastAPI(title="EngageIQ AI")

app.include_router(ws_router)


@app.get("/health")
def health():
    return {"status": "ok"}
