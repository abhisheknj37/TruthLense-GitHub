# TruthLens

**Multimodal Digital Forensics and Media Authenticity Analysis**

TruthLens is a hackathon MVP for inspecting image, video and audio evidence, preserving traceability, and communicating uncertainty clearly.

> A model score is not proof that media is real or fake. Automated findings must remain reviewable by a human.

## Current implementation
- Image decoding, dimensions, format and available EXIF metadata.
- Video metadata and representative frame sampling.
- WAV/FLAC/OGG audio metadata and signal-level statistics.
- Multi-file case intake, context notes and a structured forensic report.
- Explicit `not_configured` model state until validated authenticity models are connected.
- React dashboard with evidence intake and report visualization.

**Not yet implemented:** trained deepfake classifiers, face/voice identity matching, lip-sync classification, cross-modal authenticity scoring, and calibrated confidence. The application intentionally does not invent real/fake verdicts.

## Repository layout
- `backend/`: FastAPI API, media inspectors and detector adapter.
- `frontend/`: React + Vite investigation dashboard.
- `docs/ARCHITECTURE.md`: system design and extension points.
- `docs/EVALUATION.md`: model validation protocol.
- `docs/DEMO_CHECKLIST.md`: demo sequence.

## Run on Windows PowerShell

Backend terminal:
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

If Windows Application Control blocks an executable in your environment, use an administrator-approved Python environment; do not disable security controls to work around the policy.

Frontend terminal:
```powershell
cd frontend
npm install
npm run dev
```

Open:
- Frontend: http://localhost:5173
- API health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs

Copy `frontend/.env.example` to `frontend/.env.local` only when you need to change the API URL.

## Tests
From `backend/`:
```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Safe evidence handling
- Files are limited to 40 MB each and 8 files per case.
- Temporary uploaded media is deleted after analysis.
- Uploaded files are never executed.
- Do not treat EXIF absence, signal statistics or sampled frames as evidence of a deepfake.
- A public deployment needs isolated media workers, stronger content validation, authentication, rate limiting, private storage and retention controls.

See `docs/ARCHITECTURE.md` before adding model adapters.
