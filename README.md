# TruthLens

**Multimodal Digital Forensics and Media Authenticity Analysis**

TruthLens inspects image, video and speech for manipulation-related model signals, records evidence fingerprints and produces explainable reports for human review.

> A model score is not proof that media is real or fake. Every finding remains reviewable and may be inconclusive.

## Workspace capabilities

- **Dashboard:** workspace overview, saved-case metrics, recent investigations and the four-stage review workflow.
- **New investigation:** drag-and-drop evidence intake, image thumbnails, video previews, audio playback cards, per-file validation, removal and total upload-size summary.
- **Analysis pipeline:** media inspection, modality analysis, evidence correlation, then human review and reporting. Browser upload progress is shown; the API does not currently stream model-stage progress.
- **Reports:** image face regions and optional Grad-CAM-style patch attribution, sampled video-frame timeline, audio speech-window scores, metadata, SHA-256 fingerprint and cross-modal comparison.
- **Case history:** reopen server-persisted investigations, search saved cases, and delete reports and their attribution artifacts for the signed-in account.
- **Exports:** JSON download and print-friendly report that can be saved as PDF from the browser print dialog.
- **Settings:** local workspace label and API health check.

Completed reports, activity events, evidence metadata, SHA-256 verification results and generated model-attribution PNGs are persisted by the backend, scoped to the verified Firebase UID. The dashboard polls the authenticated activity feed every 8 seconds; case history and reports are loaded from the API and can be accessed across browsers/devices that sign in to the same account. Original uploaded media is processed from temporary working copies and removed after analysis; it is not retained as a case attachment. By default, the SQLite database and heatmap artifacts are stored under `backend/data/`. Back up and protect this directory in deployment.

## Authentication

- Firebase Authentication supports email/password and Google sign-in.
- FastAPI verifies Firebase ID tokens (signed JWTs) on protected requests. The API never trusts a user ID sent in the request body.
- Follow [docs/FIREBASE_AUTH_SETUP.md](docs/FIREBASE_AUTH_SETUP.md) to configure your Firebase project and local secrets.
- Copy `frontend/.env.example` to `frontend/.env.local`; copy `backend/.env.example` to `backend/.env`. Keep the Firebase Admin service-account JSON private and outside Git.
- The analysis, activity, report retrieval, heatmap retrieval and report deletion APIs require a valid Firebase ID token. Public health and model-capability checks remain unauthenticated.
- Report records and heatmap artifacts are scoped by Firebase UID in the backend database. Original uploaded media is not stored after processing.

## Current model-backed pipelines

- **Image:** face crops → DeiTFake ViT image classifier → per-face score, heuristic uncertainty band and Grad-CAM-style patch attribution.
- **Video:** up to eight evenly spaced frames → same face-image classifier → timestamped frame results and clip-level median over face-bearing frames.
- **Audio:** 16 kHz mono speech windows → Wav2Vec2 classifier → segment scores and median summary.
- **Cross-modal:** optional comparison of separate audio/video model signals only when the user confirms they belong to the same recording.
- **Evidence:** server-computed SHA-256 fingerprint compared with a browser pre-upload digest, media metadata and structured JSON report. A matching digest verifies transfer integrity only.

## Selected pretrained models

- Image: [sakshamkr1/deitfake-v2](https://huggingface.co/sakshamkr1/deitfake-v2) — Apache-2.0 model card; trained on face-cropped images from an OpenForensics-derived dataset; research/educational use.
- Audio: [Hemgg/Deepfake-audio-detection](https://huggingface.co/Hemgg/Deepfake-audio-detection) — Apache-2.0 model card; Wav2Vec2 base, multi-ethnic English audio dataset, approximately 16 kHz.

The model cards report their own evaluation metrics. Those numbers are not TruthLens benchmark results. The checkpoints have not been independently validated by this project on a representative forensic test set. The audio model card explicitly says more data are needed. See `docs/EVALUATION.md`.

## Run on Windows PowerShell

First clone or update the advanced project branch:

```powershell
git clone --branch feature/truthlens-advanced-mvp https://github.com/abhisheknj37/TruthLense.git
# or, inside an existing clone:
git switch feature/truthlens-advanced-mvp
git pull origin feature/truthlens-advanced-mvp
```

Backend terminal:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, transformers, fastapi, uvicorn, librosa, cv2; print('All TruthLens dependencies ready!')"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Frontend terminal (in a second PowerShell window):

```powershell
cd frontend
npm install
npm run dev
```

Open the frontend at http://localhost:5173 and API docs at http://127.0.0.1:8000/docs. Verify the backend at http://127.0.0.1:8000/health and http://127.0.0.1:8000/api/capabilities.

**First-run note:** PyTorch/Transformers installation and the first model download can be large. The image model loads only when an image with a detected face is analyzed; the audio model loads when an audio file is analyzed. CPU inference works but may be slow. If Windows Application Control blocks an executable, use an administrator-approved Python environment; do not disable security controls to work around the policy.

## Supported evidence formats

- Images: JPG, JPEG, PNG, WEBP, BMP, TIF, TIFF
- Video: MP4, MOV, AVI, MKV, WEBM
- Audio: WAV, FLAC, OGG
- Up to 8 files per case; maximum 40 MB per file

Document uploads such as PDF, DOCX and XLSX are not supported by the current analysis API and are therefore not offered as evidence types in this version.

## Safety and limitations

- Image model is face-crop focused; a no-face image returns inconclusive.
- Video is sampled frame classification, not temporal forensics or lip-sync analysis.
- Audio results may not generalize beyond the model's limited training scope.
- Softmax outputs and 0.35/0.65 score bands are not calibrated probabilities.
- Cross-modal agreement is not proof of authenticity or synchronization.
- The backend database and heatmap directory are persistent local server storage; production deployments should use backed-up durable storage and strict filesystem permissions.
- Uploaded files are never executed. Public deployment requires isolated processing, authentication, rate limits and stronger security controls.

See `docs/ARCHITECTURE.md`, `docs/EVALUATION.md` and `docs/DEMO_CHECKLIST.md`.
