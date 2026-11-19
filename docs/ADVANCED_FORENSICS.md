# Advanced TruthLens forensic modules

The advanced modules add review evidence and optional analysis paths. They do not turn the application into an automatic authenticity adjudicator.

## Temporal video review

`app.services.advanced_forensics.temporal_video_forensics` samples frames and measures Farneback optical flow and colour-histogram changes. It reports timestamps for abrupt motion and scene-change cues. Camera movement, cuts, edits, lighting, compression and scene content can produce the same measurements. This is a heuristic triage tool, not a trained temporal deepfake detector.

An optional user-supplied TorchScript model can be set with `TRUTHLENS_TEMPORAL_MODEL`. Its input contract is RGB float32 `[1,T,3,224,224]` in [0,1]. No temporal model weights are included. Use only a checkpoint whose architecture, preprocessing, class mapping and training/evaluation provenance match that contract.

## Within-video face consistency

Install the optional requirements and configure InsightFace model weights as needed. The integration compares normalized face embeddings sampled within one clip; it does not identify a person. Pose, occlusion, lighting, multiple people and detector errors can change embeddings. Similarity thresholds are review heuristics, not validated identity decisions.

## Audio/video synchronization review

When a user confirms that separate audio and video files belong together, TruthLens runs a coarse lower-frame motion versus audio-energy envelope comparison. This is experimental activity alignment, not phoneme-to-viseme lip-sync verification.

The optional SyncNet adapter requires a compatible package, checkpoint and FFmpeg executable. Set `TRUTHLENS_SYNCNET_WEIGHTS` to the checkpoint path. When separate paired audio is selected, FFmpeg creates a temporary analysis-only mux; original uploaded files are not changed. SyncNet's offset/confidence are model-specific synchronization measures, not probabilities of manipulation. A mismatch can also arise from legitimate dubbing, editing or recording offsets.

## C2PA Content Credentials

The optional `c2pa-python` SDK can read supported manifests and return validation metadata. Install it from `requirements-forensics.txt` and configure appropriate trust anchors. Manifest presence alone is not proof that a signer is trusted or that depicted events are true. When the SDK is unavailable or the format is unsupported, TruthLens reports that state rather than inferring credentials are absent or invalid.

## OCR and speech transcription

OCR requires the optional `pytesseract` Python package and a separately installed Tesseract executable on the backend host. It runs on uploaded images and sampled video frames. OCR text can contain recognition errors.

Speech transcription is optional. Set `TRUTHLENS_ASR_MODEL` to a compatible local or approved Hugging Face ASR model. The first model load may download weights. Transcription can be inaccurate and must be checked against original audio.

## Text triage and scholarly source discovery

Investigation notes, OCR and transcription are combined for rule-based text triage (dates, URLs, repeated terms and selected language patterns). `POST /api/claims/verify` discovers related scholarly works through OpenAlex. Returned works are candidate sources only; the endpoint does not determine whether a claim is true or false. Network access is required for source lookup.

## Evidence ledger and APIs

Each completed case receives a local SQLite hash-linked event history. The report includes event hashes, and authenticated endpoints expose:

- `GET /api/cases/{report_id}/chain-of-custody`
- `GET /api/cases/{report_id}/chain-of-custody/verify`
- `POST /api/claims/verify` with JSON body `{"text":"..." }`

The ledger is scoped using the authenticated Firebase UID. Hash-chain verification detects changes to stored events, but this local database is not externally anchored, independently timestamped, or protected against full database replacement. For formal evidentiary custody, export events to an independently controlled append-only system with trusted timestamps and access logging.

## Score evaluation and calibration

Install `scikit-learn` from `requirements-forensics.txt` for the calibration utility. Prepare a labelled CSV with `label,score` columns (labels 0/1; fake-direction scores in [0,1]) and run from `backend/`:

```powershell
python evaluate_scores.py labelled_scores.csv --threshold 0.65 --output evaluation-results.json
python evaluate_calibration.py labelled_scores.csv --output calibration-results.json
```

The score evaluator reports metrics for the supplied labels and threshold. The calibration utility fits an isotonic mapping on the first 70% and evaluates on the remaining 30%. Shuffle and split by independent source/video/person before using it; a random row split can leak near-duplicate material. These utilities do not automatically alter or calibrate the deployed model outputs. No benchmark dataset or calibration results are bundled.

## Optional dependencies

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements-forensics.txt
```

Install the system Tesseract and FFmpeg executables separately where applicable. First-run model downloads may be large. All missing dependencies, weights and unsupported formats should remain visible as unavailable/inconclusive states.
