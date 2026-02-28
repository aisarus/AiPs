import os
import base64
import traceback
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

STATIC_DIR = os.getenv("STATIC_DIR", "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-2.5-flash")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image")
KEY_COLOR_DEFAULT = os.getenv("KEY_COLOR_DEFAULT", "#00FF00").upper()

_gclient = None
_gclient_err = None

def get_client():
    global _gclient, _gclient_err
    if _gclient is not None:
        return _gclient
    if not GEMINI_API_KEY:
        _gclient_err = "Missing GEMINI_API_KEY (set it in Render Environment)."
        return None
    try:
        from google import genai  # google-genai
        _gclient = genai.Client(api_key=GEMINI_API_KEY)
        return _gclient
    except Exception as e:
        _gclient_err = f"Failed to init google-genai client: {type(e).__name__}: {e}"
        return None

def extract_text(resp: Any) -> str:
    txt = getattr(resp, "text", None)
    if txt:
        return txt.strip()
    out = []
    for c in (getattr(resp, "candidates", None) or []):
        content = getattr(c, "content", None)
        for p in (getattr(content, "parts", None) or []):
            t = getattr(p, "text", None)
            if t:
                out.append(t)
    return "\n".join(out).strip()

def extract_inline_image(resp: Any) -> Dict[str, str]:
    for c in (getattr(resp, "candidates", None) or []):
        content = getattr(c, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                data = inline.data
                if isinstance(data, (bytes, bytearray)):
                    data = base64.b64encode(data).decode("utf-8")
                mime = getattr(inline, "mime_type", "image/png")
                return {"image_base64": data, "mime_type": mime}
    raise RuntimeError("No inline image in response (model might not be image-capable).")

def local_constraints(prompt: str, kind: str, key_color: str) -> str:
    p = (prompt or "").strip()
    if not p:
        return ""
    if kind == "background":
        return (
            f"{p}\n\nRules:\n"
            "- Background plate ONLY (no characters, no foreground subject).\n"
            "- Consistent lighting.\n"
            "- No text, no watermark, no logos.\n"
        )
    if kind == "subject":
        return (
            f"{p}\n\nSTRICT RULES:\n"
            "- ONE main subject only.\n"
            "- Centered, fully visible (not cropped).\n"
            f"- Solid flat chroma background ONLY: {key_color} (no gradient, no shadows, no texture).\n"
            "- No extra objects, no scenery.\n"
            "- No text, no watermark.\n"
            "- Clean edges.\n"
        )
    return (
        f"{p}\n\nOutput requirements:\n"
        "- One clear scene.\n"
        "- Specify style, lighting, camera.\n"
        "- No text, no watermark.\n"
    )

def err(message: str, status: int = 400, debug: Optional[dict] = None):
    payload = {"error": message}
    if debug:
        payload["debug"] = debug
    return JSONResponse(payload, status_code=status)

class PingResponse(BaseModel):
    ok: bool
    provider: str
    text_model: str
    image_model: str

class ImproveReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    kind: Literal["scene", "background", "subject"] = "scene"
    key_color: Optional[str] = None

class ImproveResp(BaseModel):
    improved: str
    used_provider: str
    notes: str

class BreakdownReq(BaseModel):
    scene_prompt: str = Field(..., min_length=1)
    key_color: Optional[str] = None

class BreakdownResp(BaseModel):
    background: str
    subject: str
    used_provider: str

class GenLayerReq(BaseModel):
    layer_kind: Literal["background", "subject"] = "background"
    prompt: str = Field(..., min_length=1)
    key_color: Optional[str] = None

class GenLayerResp(BaseModel):
    image_base64: str
    mime_type: str
    layer_kind: str
    final_prompt: str
    used_provider: str
    debug: Optional[Dict[str, Any]] = None

app = FastAPI(title="AI Scene Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Missing static/index.html</h1>", status_code=500)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/api/ping", response_model=PingResponse)
def ping():
    provider = "gemini" if get_client() else "unconfigured"
    return PingResponse(ok=True, provider=provider, text_model=TEXT_MODEL, image_model=IMAGE_MODEL)

@app.post("/api/improve_prompt", response_model=ImproveResp)
def improve(req: ImproveReq):
    key_color = (req.key_color or KEY_COLOR_DEFAULT).upper()
    local = local_constraints(req.prompt, "scene" if req.kind == "scene" else req.kind, key_color)

    client = get_client()
    if not client:
        return ImproveResp(improved=local, used_provider="local", notes=_gclient_err or "Gemini not configured.")

    meta = (
        "You are a prompt improver. Keep user intent. Make it more specific and production-ready. "
        "Return ONLY the improved prompt, no commentary."
    )
    hard = ""
    if "\n\nRules:" in local or "\n\nSTRICT RULES:" in local:
        hard = local.split("\n\n", 1)[1]
    prompt = f"{meta}\n\nUSER PROMPT:\n{req.prompt}\n\nHARD CONSTRAINTS TO INCLUDE:\n{hard or '(none)'}"
    try:
        resp = client.models.generate_content(model=TEXT_MODEL, contents=[prompt])
        improved = extract_text(resp) or local
        return ImproveResp(improved=improved, used_provider="gemini", notes="Gemini rewrite + constraints.")
    except Exception as e:
        return ImproveResp(improved=local, used_provider="local", notes=f"Gemini failed: {type(e).__name__}: {e}")

@app.post("/api/breakdown_scene", response_model=BreakdownResp)
def breakdown(req: BreakdownReq):
    key_color = (req.key_color or KEY_COLOR_DEFAULT).upper()
    client = get_client()
    if not client:
        bg = local_constraints(req.scene_prompt, "background", key_color)
        sub = local_constraints("Main subject of the scene (be specific).", "subject", key_color)
        return BreakdownResp(background=bg, subject=sub, used_provider="local")

    instr = (
        "Split one scene prompt into 2 prompts: BACKGROUND_PROMPT and SUBJECT_PROMPT.\n"
        "Requirements:\n"
        "- BACKGROUND_PROMPT: environment/background only, no characters, no main subject.\n"
        "- SUBJECT_PROMPT: exactly one main subject, centered, fully visible.\n"
        f"- SUBJECT_PROMPT background must be perfectly solid flat chroma color {key_color}, no gradient/shadows.\n"
        "- No text, no watermark, no logos.\n"
        'Return STRICT JSON only: {"background":"...","subject":"..."}'
    )
    try:
        resp = client.models.generate_content(model=TEXT_MODEL, contents=[instr + "\n\nSCENE:\n" + req.scene_prompt])
        txt = extract_text(resp)
        import json
        obj = json.loads(txt)
        bg = local_constraints(obj.get("background", ""), "background", key_color)
        sub = local_constraints(obj.get("subject", ""), "subject", key_color)
        return BreakdownResp(background=bg, subject=sub, used_provider="gemini")
    except Exception:
        bg = local_constraints(req.scene_prompt, "background", key_color)
        sub = local_constraints("Main subject of the scene (be specific).", "subject", key_color)
        return BreakdownResp(background=bg, subject=sub, used_provider="local")

@app.post("/api/generate_layer", response_model=GenLayerResp)
def generate_layer(req: GenLayerReq):
    key_color = (req.key_color or KEY_COLOR_DEFAULT).upper()
    kind = req.layer_kind
    final_prompt = local_constraints(req.prompt, kind, key_color)

    client = get_client()
    if not client:
        return err(_gclient_err or "Gemini not configured.", 400)

    try:
        resp = client.models.generate_content(model=IMAGE_MODEL, contents=[final_prompt])
        img = extract_inline_image(resp)
        return GenLayerResp(
            image_base64=img["image_base64"],
            mime_type=img.get("mime_type", "image/png"),
            layer_kind=kind,
            final_prompt=final_prompt,
            used_provider="gemini",
            debug={"model": IMAGE_MODEL},
        )
    except Exception as e:
        return err(
            f"{type(e).__name__}: {e}",
            500,
            debug={"trace": traceback.format_exc(limit=3), "model": IMAGE_MODEL},
        )
