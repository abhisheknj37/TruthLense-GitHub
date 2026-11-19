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
2. Upload a short face-containing video. Show up to sixteen deterministic sampled frame indices, timestamps, per-frame model scores, direction counts and score range. Explain that it is sampled frame classification, not a trained temporal deepfake detector.
3. Upload speech in WAV/FLAC/OGG. Show segment-level audio scores and the aggregate model signal.
4. Upload the associated separate audio and video files, then explicitly select "These audio and video files belong to the same recording". Show cross-modal agreement, disagreement or inconclusive status, plus optional timing/synchronization results when configured.
5. Add investigator-supplied claim/caption text to the same case. Show pairwise comparisons with image OCR, video-frame text and the audio transcript, including source labels, shared terms and previews.
6. Review at least one technical finding: choose a decision, write a rationale and save it. Navigate away and reopen the case to show that the saved review persists.
7. Export the PDF. Point out the audio synthetic/spoof indicator, text-source comparisons, analyst decisions, limitations and SHA-256 fingerprint. Export JSON as the machine-readable companion.

## Presentation language
- Say "model signal", not "proof", "certain" or "verified fake".
- The 0.35 / 0.65 thresholds are heuristic score bands, not calibrated confidence boundaries.
- The primary image model is face-crop focused. If no face is detected, explain that the face-model path is inconclusive; a separate whole-image AI-art research fallback may run, but it is not a general photograph-manipulation detector.
- The video pipeline samples frames; it is not a temporal or lip-sync detector.
- The audio model's model card describes a multi-ethnic English training set and says more data are needed. Its synthetic/spoof-direction indicator is not speaker identification and does not prove voice cloning. Do not generalize it to all languages, codecs, generators or noisy recordings.
- Text-source overlap is lexical triage only. Shared wording does not establish agreement or truth; low overlap does not establish contradiction. Review the original image, frames, audio and primary sources.
- Cross-modal comparison compares independent model signals only; it does not prove synchronization, identity or provenance.
- If a model fails to download/load or the sample is outside the model scope, report inconclusive.


## Plain-text and semantic review
- Upload a UTF-8 `.txt` or `.md` file with a claim, caption or source note alongside the image/video/audio evidence.
- Check the report retains the text file as a separate source and includes it in lexical pairwise comparison.
- When `TRUTHLENS_NLI_MODEL` is configured, demonstrate bidirectional candidate entailment/contradiction/neutral sentence relationships. State that these are uncalibrated model-generated candidates, not factual verdicts.


8. Upload a UTF-8 TXT/MD source file alongside the media. Show the imported source, lexical comparison and, only when configured, bidirectional NLI candidate relationships against transcript/OCR/claim text.
9. When available, show an OpenAlex candidate source's abstract excerpt and the claim-to-abstract NLI review. State that source relevance and NLI labels do not prove a claim.
10. Demonstrate the optional per-case original-retention choice on a non-sensitive sample, then show authenticated download and SHA-256 recheck. Explain that retention is opt-in and expires.
11. Export the chain-of-custody manifest and verify the local hash chain; show HMAC status only if a server key is configured.
