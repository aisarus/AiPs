import os
import base64
from typing import Optional, Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Gemini SDK (Google GenAI)
# Requirements must include: google-genai
try:
    from google import genai
except Exception:
    genai = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Choose models (you can change later without touching frontend)
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-1.5-flash")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.0-flash-exp-image-generation")

client = None
if GEMINI_API_KEY and genai is not None:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None


app = FastAPI()


class ImproveRequest(BaseModel):
    kind: Literal["image", "object", "background", "light", "mood"]
    text: str


class ImproveResponse(BaseModel):
    text: str


class LayerRequest(BaseModel):
    kind: Literal["image", "object", "background", "light", "mood"]
    prompt: str


class LayerResponse(BaseModel):
    image_base64: str
    mime_type: str
    kind: str


def _extract_inline_image(resp) -> tuple[Optional[bytes], str]:
    """
    Tries to extract inline image bytes from google-genai response.
    Returns (bytes, mime_type). If none found, (None, "image/png").
    """
    mime = "image/png"
    try:
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None) is not None:
                    data = inline.data
                    mime = getattr(inline, "mime_type", mime) or mime
                    if isinstance(data, (bytes, bytearray)):
                        return bytes(data), mime
                    # sometimes returned as base64 string
                    if isinstance(data, str):
                        try:
                            return base64.b64decode(data), mime
                        except Exception:
                            pass
    except Exception:
        pass
    return None, mime


def _json_error(msg: str, status: int = 400):
    return JSONResponse(status_code=status, content={"error": msg})


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "configured": bool(client),
        "has_key": bool(GEMINI_API_KEY),
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL,
    }


@app.post("/api/improve_prompt", response_model=ImproveResponse)
def improve_prompt(req: ImproveRequest):
    if client is None:
        return _json_error("GEMINI_API_KEY not set or google-genai is missing.", 500)

    text = (req.text or "").strip()
    if not text:
        return _json_error("Empty text.", 400)

    # Simple, honest "improvement": make it more specific, add constraints per kind.
    rules_by_kind = {
        "image": "Improve the overall scene prompt: add concrete details (style, camera, time, atmosphere), keep it ONE scene.",
        "background": "Improve background prompt only: no main subject in foreground, coherent environment, no text/watermarks.",
        "object": "Improve main subject prompt: ONE subject, full body visible, centered, clean silhouette, bright green background for chroma key, no extra objects, no text/watermarks.",
        "light": "Improve lighting prompt: describe light direction, softness, color temperature, shadows, cinematic intent. No text.",
        "mood": "Improve mood prompt: describe mood/grade/atmosphere (fog, film look, palette), no text, no watermark."
    }

    system = (
        "You are a prompt editor. Output ONLY the improved prompt text. "
        "No quotes, no bullets, no explanations."
    )
    rule = rules_by_kind.get(req.kind, "Improve the prompt.")

    prompt = f"{system}\n\nRULE:\n{rule}\n\nINPUT:\n{text}\n\nOUTPUT:"

    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=[prompt],
        )
        out = getattr(resp, "text", None)
        if out and isinstance(out, str) and out.strip():
            return ImproveResponse(text=out.strip())

        # fallback parse
        parts = []
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            c = getattr(cand, "content", None)
            for p in getattr(c, "parts", None) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
        joined = ("\n".join(parts)).strip()
        if not joined:
            return _json_error("Text model returned empty response.", 502)
        return ImproveResponse(text=joined)

    except Exception as e:
        return _json_error(f"Text generation failed: {type(e).__name__}: {e}", 500)


@app.post("/api/generate_layer", response_model=LayerResponse)
def generate_layer(req: LayerRequest):
    if client is None:
        return _json_error("GEMINI_API_KEY not set or google-genai is missing.", 500)

    base = (req.prompt or "").strip()
    if not base:
        return _json_error("Empty prompt.", 400)

    # Compose strict layer prompts (this is the whole “principle” of the tool):
    # - it doesn't do magic; it adds constraints by layer type to make outputs usable.
    if req.kind == "background":
        final_prompt = (
            f"{base}\n\n"
            "STRICT RULES:\n"
            "- Background only. No main subject in the foreground.\n"
            "- Wide establishing shot, coherent perspective.\n"
            "- No text, no watermark, no logos.\n"
        )
    elif req.kind == "object":
        final_prompt = (
            f"{base}\n\n"
            "STRICT RULES:\n"
            "- One main subject only, centered, full body visible (not cropped).\n"
            "- Background must be perfectly solid bright green (#00FF00), flat color.\n"
            "- No additional objects, no scenery.\n"
            "- No text, no watermark, no logos.\n"
            "- Clean edges, clear silhouette.\n"
        )
    elif req.kind == "light":
        final_prompt = (
            f"{base}\n\n"
            "STRICT RULES:\n"
            "- Generate a lighting overlay for the whole frame.\n"
            "- Abstract light/shadow pattern, cinematic.\n"
            "- No text, no watermark, no logos.\n"
        )
    elif req.kind == "mood":
        final_prompt = (
            f"{base}\n\n"
            "STRICT RULES:\n"
            "- Generate a mood/color-grading overlay for the whole frame.\n"
            "- Atmospheric, film look.\n"
            "- No text, no watermark, no logos.\n"
        )
    else:  # image
        final_prompt = (
            f"{base}\n\n"
            "STRICT RULES:\n"
            "- One coherent scene.\n"
            "- No text, no watermark, no logos.\n"
        )

    try:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[final_prompt],
        )
        img_bytes, mime = _extract_inline_image(resp)
        if not img_bytes:
            return _json_error("Image not generated (no inline image data in response). Try simplifying prompt.", 502)

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return LayerResponse(image_base64=b64, mime_type=mime, kind=req.kind)

    except Exception as e:
        return _json_error(f"Layer generation failed: {type(e).__name__}: {e}", 500)


# Serve static UI
@app.get("/")
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
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
