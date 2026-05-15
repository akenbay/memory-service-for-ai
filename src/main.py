from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI(title="Memory Service")


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class TurnRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: List[Message]
    timestamp: str
    metadata: Dict[str, Any] = {}


class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    max_tokens: int = 1024


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 10


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/turns", status_code=status.HTTP_201_CREATED)
async def write_turn(req: TurnRequest):
    return {"id": "stub-turn-id"}


@app.post("/recall")
async def recall(req: RecallRequest):
    return {"context": "", "citations": []}


@app.post("/search")
async def search(req: SearchRequest):
    return {"results": []}


@app.get("/users/{user_id}/memories")
async def get_memories(user_id: str):
    return {"memories": []}


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    return None


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    return None