# TruthLens architecture

## Evidence-first design
TruthLens separates technical observations, model outputs and human conclusions. Each uploaded object receives a SHA-256 digest. The original file is never modified; an ephemeral working copy is used for processing and removed afterwards.

## Modality pipelines

### Image
- Face locations are proposed by OpenCV Haar Cascade.
- Up to three largest face crops are passed to \`sakshamkr1/deitfake-v2\` (Apache-2.0 model card).
- The implementation reads the checkpoint's \`id2label\` mapping instead of assuming the class order.
- Output contains per-face real/fake softmax scores, a median summary, a heuristic inconclusive band and an optional transformer patch-attribution overlay.

**Scope:** the checkpoint is described as a face-cropped real/fake image classifier trained from OpenForensics-derived data. It is a research/educational model, not a validated forensic service. Images with no detected face remain inconclusive.

### Video
- OpenCV reads the container metadata.
- At most eight evenly spaced frames are sampled.
- Each frame is passed through the same face-crop image detector.
- The report stores frame index, timestamp, per-frame result and a median score over face-bearing frames.

**Scope:** this is sampled frame classification, not temporal deepfake detection. It does not establish lip-sync, continuity or clip authenticity.

### Audio
- Audio is resampled to 16 kHz mono and analyzed in up to five 8-second windows.
- \`Hemgg/Deepfake-audio-detection\` is loaded with Transformers (Apache-2.0 model card).
- Class indices are resolved from the model config labels (\`AIVoice\` and \`HumanVoice\`), not guessed.
- Per-window scores and median audio score are returned.

**Scope:** the model card describes a multi-ethnic English audio dataset and a self-reported evaluation score, while also noting that more data are needed. This is a narrow research signal, not general voice-clone verification. It may fail on languages, codecs, generators or recording conditions outside its training data.

## Uncertainty and decision bands
The 0.35 / 0.65 cutoffs are configurable heuristic score bands. They are not calibrated confidence thresholds. The UI labels outputs as model signals and displays an inconclusive band between thresholds. Do not describe these softmax values as probability that the media is fake.

## Grad-CAM-style patch attribution
For the DeiT image model, the last transformer encoder block is hooked. The gradient of the fake-class logit with respect to patch tokens produces a patch-importance overlay. This visualization identifies image regions that influenced this model's output; it is not a manipulation mask and does not explain causality.

## Cross-modal comparison
Cross-modal comparison runs only after the user explicitly marks separate audio and video uploads as belonging to the same recording. It compares the audio-window signal with the video face-frame signal and reports agreement, disagreement or inconclusive status. It does not calculate a fused authenticity probability and does not independently verify synchronization or provenance.

## Run-time behavior
Models are loaded lazily on first use and cached in process memory. The first call may take time and download model weights from Hugging Face. CPU inference is supported but may be slow; CUDA is used when available.

## API
- \`GET /health\`
- \`GET /api/capabilities\`
- \`POST /api/analyze\` multipart fields: \`files\` (one or more), \`context_text\` (optional), \`pair_media\` (optional boolean).

## Public deployment hardening
The local hackathon MVP needs isolated media workers, process-level CPU/memory/time limits, stronger content validation, authentication, rate limiting, private storage and deletion policies before public exposure. Do not allow uploaded media to execute.

## Supabase persistence

When `DATABASE_URL` is configured, the FastAPI backend stores reports, normalized assets, activity, heatmaps and hash-linked chain-of-custody events in Supabase PostgreSQL. Original media is retained in the private `truthlens-evidence` Storage bucket when `SUPABASE_URL` and the backend-only `SUPABASE_SERVICE_ROLE_KEY` are configured. Firebase Auth remains the user identity provider. Local development without these settings retains SQLite. See [SUPABASE_SETUP.md](SUPABASE_SETUP.md).


## Evidence text-source comparison and human review
OCR text from images and sampled video frames, optional ASR transcripts from speech audio, and investigator-provided context are preserved as separately labeled sources. The case report provides token-set Jaccard overlap comparisons, including shared and source-specific terms. This is a lexical triage aid only; it does not understand semantic contradiction or establish factual correctness.

Human review is a separate report block. An authenticated analyst records a decision, notes, reviewed asset IDs, UID and timestamp through `POST /api/reports/{report_id}/review`. SQLite and Supabase persistence update only that case owner's report. Automated model outputs are not overwritten by the reviewer decision.

The audio model additionally exposes a synthetic/spoof-direction indicator with model name and window count. It does not identify a person, establish who generated a voice, or prove voice cloning.
