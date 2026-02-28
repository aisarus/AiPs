# AI Scene Studio (Render + Python 3.11.9)

A small “AI Photoshop-ish” workbench:
- prompt improvement (Gemini text)
- scene → layer breakdown (BG + Subject)
- image generation per layer (Gemini image)
- cheap composite (subject over background) in the browser via canvas
- fullscreen + download
- backend keeps the API key secret (stored in Render env vars)

## Repo structure
```
.
├─ server.py
├─ requirements.txt
├─ render.yaml
├─ .python-version
├─ runtime.txt
├─ static/
│  └─ index.html
└─ .env.example
```

## Deploy to Render
1. Push this repo to GitHub.
2. In Render: **New Web Service** → connect repo.
3. Render → Environment:
   - `GEMINI_API_KEY` = your key
   - `TEXT_MODEL` = `gemini-2.5-flash` (default)
   - `IMAGE_MODEL` = `gemini-2.5-flash-image` (default)
4. Deploy. Start command:
   `uvicorn server:app --host 0.0.0.0 --port $PORT`

## Run locally
### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# put your GEMINI_API_KEY into .env
uvicorn server:app --reload --port 8000
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# put your GEMINI_API_KEY into .env
uvicorn server:app --reload --port 8000
```

Open: http://localhost:8000

## API
- GET `/api/ping`
- POST `/api/improve_prompt`  `{prompt, kind: scene|background|subject, key_color?}`
- POST `/api/breakdown_scene` `{scene_prompt, key_color?}`
- POST `/api/generate_layer`  `{layer_kind: background|subject, prompt, key_color?}`

All image responses: `{image_base64, mime_type}`.
