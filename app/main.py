import base64
import re
import secrets
from pathlib import Path
from uuid import UUID, uuid4

import ollama
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from app.config import settings
from app.agent import graph


ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Medical Agent Educational API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        x.strip()
        for x in settings.allowed_origins.split(",")
        if x.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Conversation-ID"],
)

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


# --------------------------------------------------
# MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    image_data: str | None = None


# --------------------------------------------------
# AUTH
# --------------------------------------------------

def auth(authorization: str | None):
    expected = f"Bearer {settings.api_token}"

    if not secrets.compare_digest(
        (authorization or "").encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Invalid API token")


# --------------------------------------------------
# EMERGENCY CHECK (runs before any model call)
# A simple heuristic that errs on the side of safety.
# --------------------------------------------------

EMERGENCY_PATTERNS = [
    r"\b(i have|i'm having|im having|i am having|having)\b.{0,20}\bchest (pain|pressure|tightness)\b",
    r"\bsevere chest pain\b",
    r"\b(can'?t|cannot|unable to|struggling to)\s+breathe\b",
    r"\bsevere (breathing|breathlessness)\b",
    r"\bface (is )?drooping\b|\bslurred speech\b|\bsudden (weakness|numbness)\b.{0,20}\bone side\b",
    r"\b(heavy|severe) bleeding\b|\bwon'?t stop bleeding\b|\bbleeding (heavily|badly)\b",
    r"\bunconscious\b|\bunresponsive\b|\bnot responding\b",
    r"\bkill myself\b|\bend my life\b|\bwant to die\b|\b(i am|i'm|im) suicidal\b",
]

EMERGENCY_RE = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)

EMERGENCY_MESSAGE = (
    "What you describe may be a medical emergency. Please contact your local "
    "emergency services right now (112 in India, 911 in the US, 999 in the UK) "
    "or go to the nearest emergency room. If someone is with you, ask them to "
    "stay with you.\n\n"
    "If you are having thoughts of ending your life, you are not alone: in India "
    "you can call Tele-MANAS at 14416, and in the US you can call or text 988.\n\n"
    "I'm an educational assistant and can't assess emergencies, so please don't "
    "wait for more information here."
)


# --------------------------------------------------
# IMAGE ANALYSIS (local Ollama vision model)
# --------------------------------------------------

def inspect_image(data: str | None):
    if not data:
        return ""

    try:
        header, payload = data.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        raw = base64.b64decode(payload, validate=True)

        if mime not in ("image/jpeg", "image/png", "image/webp"):
            raise ValueError("Only JPEG, PNG, and WebP are supported")

        if len(raw) > 5_000_000:
            raise ValueError("Image must be under 5 MB")

    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        client = ollama.Client(timeout=240)

        response = client.chat(
            model=settings.ollama_vision_model,
            messages=[{
                "role": "user",
                "content": (
                    "Describe what is visibly shown in this image in 2-3 plain "
                    "sentences. Do not diagnose or suggest treatment."
                ),
                "images": [raw],
            }],
            options={"temperature": 0.1},
        )

        return response.message.content or "Unable to describe the image."

    except Exception as exc:
        print("Image analysis error:", repr(exc), flush=True)
        raise HTTPException(
            status_code=502,
            detail=(
                "Image analysis failed. Is Ollama running and the vision model "
                f"installed? Run: ollama pull {settings.ollama_vision_model}"
            ),
        )


# --------------------------------------------------
# PAGES
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    try:
        ollama.Client(timeout=3).list()
        ollama_ok = True
    except Exception:
        ollama_ok = False

    return {
        "status": "ok",
        "configured": ollama_ok and bool(settings.pinecone_api_key),
        "ollama_running": ollama_ok,
        "pinecone_configured": bool(settings.pinecone_api_key),
        "tavily_configured": bool(settings.tavily_api_key),
        "chat_model": settings.ollama_chat_model,
    }


# --------------------------------------------------
# CHAT
# --------------------------------------------------

@app.post("/api/chat")
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias="X-Conversation-ID"),
):
    auth(authorization)

    if conversation_id:
        try:
            thread_id = str(UUID(conversation_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation ID")
    else:
        thread_id = str(uuid4())

    # Emergency shortcut: no retrieval, no model, instant answer
    if EMERGENCY_RE.search(payload.message):
        return {
            "answer": EMERGENCY_MESSAGE,
            "sources": [],
            "tools": ["emergency-check"],
            "conversation_id": thread_id,
        }

    try:
        observation = inspect_image(payload.image_data)

        state = graph.invoke(
            {
                "question": payload.message,
                "messages": [HumanMessage(content=payload.message)],
                "image_description": observation,
                "kb": [],
                "web": [],
                "web_status": "",
                "answer": "",
                "route": "",
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        sources = []

        # Only show PDF sources if the answer actually used them
        if state.get("route") == "answer":
            seen_pages = set()
            for item in state.get("kb", []):
                page = item.get("page")
                if page in seen_pages:
                    continue
                seen_pages.add(page)
                sources.append({
                    "type": "pdf",
                    "page": page,
                    "score": round(float(item.get("score") or 0), 3),
                })

        for item in state.get("web", []):
            sources.append({
                "type": "web",
                "url": item.get("url"),
                "title": item.get("title"),
            })

        tools_used = ["ollama", "pinecone"]

        if state.get("route") == "web" and settings.tavily_api_key:
            tools_used.append("tavily")

        if observation:
            tools_used.append("vision")

        return {
            "answer": state.get("answer", ""),
            "sources": sources,
            "tools": tools_used,
            "conversation_id": thread_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("Chat error:", repr(exc), flush=True)
        raise HTTPException(
            status_code=502,
            detail=(
                "AI service failed. Make sure Ollama is running and the chat "
                f"model is installed (ollama pull {settings.ollama_chat_model})."
            ),
        )