# TruthLens demo checklist

## First-run setup
- Use a supported Python 3.11 environment.
- From `backend/`, create and activate a virtual environment, then install `requirements.txt`.
- Install PyTorch/Transformers dependencies before presenting; the first install and model downloads are large.
- Copy `.env.example` to `.env` inside `backend/` only when changing model IDs or heuristic score bands.
- Start FastAPI from `backend/` with `python -m uvicorn app.main:app --reload`.
- Start the frontend from `frontend/` with `npm install` and `npm run dev`.
- Test `/health` and `/api/capabilities`.
- Run a sample image, short face-containing video and speech recording once to cache model weights before the actual presentation.

## Suggested live demo
1. Upload a face-containing image. Show the image-model signal, separate class scores and Grad-CAM-style patch-attribution overlay.
2. Upload a short face-containing video. Show sampled frame indices, timestamps and frame-level model signals.
3. Upload speech in WAV/FLAC/OGG. Show segment-level audio scores and the aggregate model signal.
4. Upload the associated separate audio and video files, then explicitly select "These audio and video files belong to the same recording". Show cross-modal agreement, disagreement or inconclusive status.
5. Export the JSON report and point out SHA-256 fingerprint, model IDs, heuristic score band and limitations.

## Presentation language
- Say "model signal", not "proof", "certain" or "verified fake".
- The 0.35 / 0.65 thresholds are heuristic score bands, not calibrated confidence boundaries.
- The image model is face-crop focused; a no-face image remains inconclusive.
- The video pipeline samples frames; it is not a temporal or lip-sync detector.
- The audio model's model card describes a multi-ethnic English training set and says more data are needed. Do not generalize it to all languages, codecs, generators or noisy recordings.
- Cross-modal comparison compares independent model signals only; it does not prove synchronization, identity or provenance.
- If a model fails to download/load or the sample is outside the model scope, report inconclusive.
