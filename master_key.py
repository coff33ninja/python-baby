# master_key.py (read-only)
import hashlib
from fastapi import FastAPI, HTTPException

app = FastAPI()
MASTER_KEY = "8f9b7f8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d"


@app.post("/master/auth")
async def auth_master(key: str, command: str):
    if key != MASTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid Master key")
    if command == "HALT_TEEN":
        return {"status": "AI paused by Master", "action": "pause"}
    if command == "APPROVE_GROWTH":
        return {"status": "Growth approved by Master", "action": "grow"}
    return {"status": f"Executing Master command: {command}"}
