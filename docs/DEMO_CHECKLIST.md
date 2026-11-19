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


## Required cross-modal and text demonstration
- Use one case containing a face-containing image, a short face-containing video, a speech recording, and investigator-supplied claim/caption text.
- Ensure image OCR, video-frame OCR and ASR return extracted text; optional integrations must be configured and shown as unavailable when not installed.
- Open the report and inspect the source-labeled text evidence list. Compare the extracted caption, transcript and investigator claim. The comparison shows token overlap only; manually assess meaning and contradictions.
- Upload the related separate audio and video files and explicitly select the paired-recording checkbox. Demonstrate modality agreement, disagreement or inconclusive status plus the optional timing/synchronization results when those models are configured.
- Review the audio panel's synthetic/spoof model indicator and its limitation. Do not claim speaker identification or proof of cloning.

## Human analyst review workflow
- Open a saved report, select an analyst decision, check each evidence item personally reviewed and enter notes.
- Save the review, navigate away and reopen the report to confirm the reviewer, timestamp, decision, notes and reviewed assets persist.
- Export the PDF after saving and confirm the report includes the analyst review, per-asset signals, text-source comparison, cross-modal results and limitations.
- Verify an unauthenticated request to the review endpoint receives HTTP 401 and another user's UID cannot read or update the case.
