# TruthLens
Multimodal Digital Forensics and Media Authenticity Analysis.

> A hackathon MVP focused on explainable indicators, evidence traceability and human review. A model score is not proof that media is real or fake.

## MVP scope
- Image analysis endpoint and metadata observations
- Video metadata inspection and representative frame sampling
- WAV/FLAC/OGG signal metadata and level measurements
- Cross-modal evidence correlation as consistency checks, not identity proof
- Explainable report with limitations and human-review status

## Architecture
- `backend/`: FastAPI API, media validation, processing and model adapters
- `frontend/`: React + Vite dashboard
- `docs/`: architecture, model cards and evaluation protocol

## Run locally
Backend:
```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m uvicorn app.main:app --reload
```

Frontend (second terminal):
```powershell
cd frontend
npm install
npm run dev
```

## Safe handling
- Uploads are size-limited and type-checked.
- Do not execute uploaded files.
- Keep original evidence unchanged; analyze working copies.
- Reports distinguish observed indicators from model inferences.
- Never interpret confidence as calibrated probability unless validated on an appropriate dataset.

See `docs/ARCHITECTURE.md` and `docs/EVALUATION.md`.
