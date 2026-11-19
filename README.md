# TruthLens

**Multimodal Digital Forensics and Media Authenticity Analysis**

TruthLens inspects image, video and speech for manipulation-related model signals, records evidence fingerprints and produces explainable reports for human review.

> A model score is not proof that media is real or fake. Every finding remains reviewable and may be inconclusive.

## Current model-backed pipelines
- **Image:** face crops → DeiTFake ViT image classifier → per-face score, heuristic uncertainty band and Grad-CAM-style patch attribution.
- **Video:** up to eight evenly spaced frames → same face-image classifier → timestamped frame results and clip-level median over face-bearing frames.
- **Audio:** 16 kHz mono speech windows → Wav2Vec2 classifier → segment scores and median summary.
- **Cross-modal:** optional comparison of separate audio/video model signals only when the user confirms they belong to the same recording.
- **Evidence:** SHA-256 fingerprint, media metadata and structured JSON report.

## Selected pretrained models
- Image: [sakshamkr1/deitfake-v2](https://huggingface.co/sakshamkr1/deitfake-v2) — Apache-2.0 model card; trained on face-cropped images from an OpenForensics-derived dataset; research/educational use.
- Audio: [Hemgg/Deepfake-audio-detection](https://huggingface.co/Hemgg/Deepfake-audio-detection) — Apache-2.0 model card; Wav2Vec2 base, multi-ethnic English audio dataset, approximately 16 kHz.

The model cards report their own evaluation metrics. Those numbers are not TruthLens benchmark results. The checkpoints have not been independently validated by this project on a representative forensic test set. The audio model card explicitly says more data are needed. See `docs/EVALUATION.md`.

## Run on Windows PowerShell

**First sync the latest project branch from the repository root.** This prevents an older local copy of `requirements.txt` from omitting ML packages such as Transformers.

```powershell
git switch feature/truthlens-advanced-mvp
git pull origin feature/truthlens-advanced-mvp
```

Backend terminal (from the repository root):

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, transformers, fastapi, uvicorn, librosa, cv2; print('All TruthLens dependencies ready!')"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Frontend terminal (from the repository root, in a separate terminal):

```powershell
cd frontend
npm install
npm run dev
```

Open the frontend at http://localhost:5173 and API docs at http://127.0.0.1:8000/docs. Verify the backend at http://127.0.0.1:8000/health and http://127.0.0.1:8000/api/capabilities.

**First-run note:** PyTorch/Transformers installation and the first model download can be large. The image model loads only when an image with a detected face is analyzed; the audio model loads when an audio file is analyzed. CPU inference works but may be slow. If Windows Application Control blocks an executable, use an administrator-approved Python environment; do not disable security controls to work around the policy.

## Safety and limitations
- Image model is face-crop focused; a no-face image returns inconclusive.
- Video is sampled frame classification, not temporal forensics or lip-sync analysis.
- Audio results may not generalize beyond the model's limited training scope.
- Softmax outputs and 0.35/0.65 score bands are not calibrated probabilities.
- Cross-modal agreement is not proof of authenticity or synchronization.
- Uploaded files are never executed. Public deployment requires isolated processing and stronger security controls.

See `docs/ARCHITECTURE.md`, `docs/EVALUATION.md` and `docs/DEMO_CHECKLIST.md`.
