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

OCR prefers the optional `pytesseract` Python package and a separately installed Tesseract executable on the backend host. If these are missing but `rapidocr-onnxruntime` is installed, RapidOCR is used as a fallback for images and sampled video frames. It runs on uploaded images and sampled video frames. OCR text can contain recognition errors.

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


## OCR setup on Windows

1. Install the Tesseract OCR Windows installer from the UB Mannheim Tesseract distribution page (the installer is maintained separately from this Python project).
2. In a new PowerShell terminal, verify the executable:
   ```powershell
   tesseract --version
   tesseract --list-langs
   ```
3. Install the Python wrapper from the backend virtual environment:
   ```powershell
   cd E:\TruthLense-GitHub\backend
   .\.venv\Scripts\python.exe -m pip install pytesseract
   ```
4. Edit `backend/.env` and set the executable path if Tesseract is not on PATH:
   ```dotenv
   TRUTHLENS_TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
   TRUTHLENS_OCR_LANGUAGES=eng
   ```
5. Restart FastAPI and run a new investigation. Existing reports are not reprocessed automatically.

TruthLens checks both the executable and each requested language code. OCR returns `engine_not_found` or `language_data_missing` with a clear note when setup is incomplete. Add another language only after its matching traineddata is installed, for example `eng+hin`. OCR output is best-effort text extraction and must be checked against the source image/frame.

## Windows optional-integration configuration

Optional integrations are configured on the backend computer, not in the browser. From PowerShell, create the backend environment file once:

```powershell
cd E:\TruthLense-GitHub\backend
Copy-Item .env.example .env
```

Install the Python extras you intend to use. The complete optional set is listed here; some packages, particularly InsightFace and SyncNet, may need platform-specific installation work:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-forensics.txt
```

### System executables

- **Tesseract OCR:** install a Windows build, then configure its executable path in `backend/.env`. Example: `TRUTHLENS_TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe`. The environment variable is optional when `tesseract.exe` is already available on PATH.
- **FFmpeg:** install FFmpeg and add its `bin` directory to the backend process PATH. Verify from the same PowerShell session with `ffmpeg -version`. SyncNet needs FFmpeg to prepare paired audio/video inputs.
- **InsightFace:** the configured model pack (default `buffalo_l`) may download into the backend user's InsightFace cache on first use. The Settings status reports whether the Python dependencies and model cache are detected.

### Model paths and environment variables

Edit `backend/.env` using paths that exist on the backend machine:

| Variable | Purpose |
|---|---|
| `TRUTHLENS_TEMPORAL_MODEL` | Optional TorchScript temporal checkpoint. It must match the documented tensor input, architecture and class mapping. |
| `TRUTHLENS_FACE_MODEL` | InsightFace model-pack name; defaults to `buffalo_l`. |
| `TRUTHLENS_FACE_SIMILARITY_REVIEW_THRESHOLD` | Within-clip cosine-similarity review threshold; defaults to `0.35`, and is not an identity decision. |
| `TRUTHLENS_SYNCNET_WEIGHTS` | Path to a compatible SyncNet checkpoint. |
| `TRUTHLENS_ASR_MODEL` | Transformers-compatible automatic-speech-recognition checkpoint ID or local model path. |
| `TRUTHLENS_TESSERACT_CMD` | Optional absolute path to `tesseract.exe` when it is not on PATH. |
| `TRUTHLENS_LEDGER_DB` | Optional SQLite evidence-ledger location. |
| `HF_HOME` | Optional Hugging Face model cache location. |

Leave model paths blank until you have a compatible checkpoint. This repository does not bundle temporal-classifier or SyncNet weights. A filename alone does not establish checkpoint compatibility.

The current C2PA adapter supports JPEG, PNG, WebP, MP4 and MOV inputs only. WebM is explicitly reported as `unsupported_format`; the presence of a WebM container does not imply that it can carry a supported C2PA manifest. Do not rename a WebM file to MP4 or infer that credentials are absent. The adapter uses the installed SDK to read manifest and validation metadata. It does not expose a project-specific trust-store or trust-anchor path setting. Treat the returned validation data as something to review, not as a conclusion that a signer is trusted.

## Temporal model selection note

Do not point `TRUTHLENS_TEMPORAL_MODEL` at an arbitrary `.pt` or `.pth` file. The current optional adapter expects a TorchScript model accepting RGB float32 input shaped `[1,T,3,224,224]` and currently reads its first scalar output as a raw fake score. A model with a different architecture, frame sampling, normalization, output layout or label order is incompatible even if it is described as a video deepfake detector. Before integrating a checkpoint, verify its source, model card/paper, weights license, inference code, preprocessing, label mapping, supported video domain and independent cross-dataset evaluation. Keep it disabled until an adapter and tests match the exact checkpoint contract.

A current candidate reviewed during development, FakeSTormer, is the official ICCV 2025 implementation and reports cross-dataset video-level AUC results, but its repository specifies an SNT academic license and an older CUDA/PyTorch/MMCV stack. It is not automatically suitable for inclusion in this project or a general Windows CPU deployment. Obtain permission/confirm intended-use terms and build a separately tested adapter before adoption. Its reported benchmark values are the authors' results, not TruthLens results.


### Check readiness from the app

After editing `.env`, restart FastAPI from the backend folder:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/api/capabilities](http://127.0.0.1:8000/api/capabilities). The response now includes `optional_integration_status`, with readiness and missing-prerequisite information. The frontend Settings page reads the same data and provides a Refresh button.

A status such as `ready_to_attempt` only means the prerequisite checks passed. Actual inference can still fail due to checkpoint incompatibility, unsupported media, model downloads, device limits or runtime errors. Configure secrets only in local `.env` files or deployment secret storage; never commit credentials or model weights.


## Plain-text source evidence and semantic comparison

TruthLens accepts bounded UTF-8 `.txt` and `.md` evidence files (maximum 1 MB each). Text is treated as content only; no markup, links or scripts are executed. The imported source is attached to the case and participates in OCR/transcript/context comparisons.

Optional sentence-pair NLI review can be enabled by setting `TRUTHLENS_NLI_MODEL=cross-encoder/nli-deberta-v3-base`. This Apache-2.0 English NLI checkpoint exposes contradiction, entailment and neutral labels. Its model card reports 92.38% SNLI test accuracy and 90.04% MNLI mismatched accuracy; these are checkpoint-reported benchmark results, not TruthLens results. The local NLI module compares sentence pairs in both directions and returns candidate relationships with raw uncalibrated softmax scores. It does not verify a claim, establish what happened, or replace source inspection. Model weights load lazily on first semantic comparison; without the configuration, TruthLens reports NLI as not configured and retains lexical comparisons.

Model card: https://huggingface.co/cross-encoder/nli-deberta-v3-base


## Original evidence retention and signed custody

Original uploads are not retained unless the investigator checks the per-case retention option. Opted-in originals are stored under a private backend-only directory, separated by a SHA-256-derived owner folder, report UUID and asset UUID. The report records the storage key, digest, size, timestamp and expiry. Authenticated downloads re-check report ownership and the stored file digest. Deleting the case removes its originals. Configure `TRUTHLENS_ORIGINALS_DIR` to a private durable volume and `TRUTHLENS_ORIGINAL_RETENTION_DAYS` to the chosen retention period.

The chain-of-custody API also exports a JSON manifest. Set a strong server-only `TRUTHLENS_LEDGER_HMAC_KEY` and `TRUTHLENS_LEDGER_KEY_ID` to add HMAC-SHA256 signatures over event hashes. Verification checks the local hash chain and HMAC signatures. This is stronger tamper evidence against edits by actors without the secret, but it is still not an independently timestamped or externally witnessed record. Never commit secrets or retained evidence directories.


## Group-aware model-score calibration

The calibration script requires a source-group column and uses a group-separated fit/held-out partition. It exports a candidate artifact containing the dataset SHA-256, model ID, pipeline, fit/held-out group counts, raw-vs-calibrated held-out metrics and the isotonic mapping. Candidate artifacts are never applied at runtime. An authorized reviewer must explicitly mark the artifact approved and record reviewer/time; runtime then checks the exact model ID and modality pipeline and rejects mismatches or insufficient held-out evidence.

No labelled dataset or approved calibration artifact is bundled. Until project-specific validation is completed and approved, TruthLens must continue to report raw model outputs and heuristic review uncertainty as uncalibrated.


### Runtime readiness and retained-original purge

`GET /api/capabilities` reports the status of NLI, per-pipeline calibration, optional OCR/ASR, private original retention and custody HMAC signing. It does not download or execute optional models just to report readiness. Authenticated users can call `POST /api/cases/retention/purge-expired` to purge expired originals only inside their own private storage folder.


Temporal adapter contract is intentionally strict: it accepts exactly one scalar output and reports it as an unverified raw value. A value in the numerical range [0,1] is not treated as a probability unless the checkpoint's output semantics have been independently documented and validated. Multi-logit or multi-class outputs are rejected rather than silently taking the first element.
