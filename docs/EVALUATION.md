# TruthLens evaluation protocol

## Model outputs are not ground truth
A softmax score from a pretrained checkpoint is not a calibrated probability. The current 0.35/0.65 thresholds are heuristic separation bands only. Keep the inconclusive state visible.

## Image evaluation
The selected DeiTFake checkpoint is described as a face-cropped real/fake image classifier trained on an OpenForensics-derived dataset. Its model card reports high test metrics on its own setting, but explicitly says it is not recommended for production forensic verification without further validation. Evaluate on a held-out set of identities and generators not used during training. Include compression, resizing, lighting, demographics and no-face cases.

Record precision, recall, F1, ROC-AUC, false-positive rate at the chosen threshold, calibration (Brier score / reliability plot), and a confusion matrix. Split by subject/source, not by near-duplicate image.

## Audio evaluation
The selected Wav2Vec2 model card describes a multi-ethnic English dataset sampled at approximately 16 kHz, reports evaluation accuracy of 0.9545, and says additional data are needed. Do not generalize this to all languages, TTS engines, voice conversion systems or noisy real-world audio. Test held-out speakers, languages, synthesis engines, codecs and recording channels.

Record EER, ROC-AUC, precision, recall, false-positive rate, calibration and real-time factor. Preserve the per-window scores in the report.

## Video evaluation
The current pipeline samples at most eight evenly spaced frames and runs the image face detector on each. Evaluate clip-level results and per-frame results separately. This is not a temporal detector. Do not claim lip-sync detection or temporal consistency analysis until a corresponding model and dataset are implemented.

## Cross-modal evaluation
Only compare audio/video assets when the uploader confirms they belong to the same recording. Current cross-modal logic compares independent modality scores and reports agreement/disagreement/inconclusive; it does not establish sync, identity or authenticity. Evaluate paired, timestamp-aligned samples before adding actual sync claims.

## Reporting rules
- Keep model id, model version, source and scope in every report.
- Distinguish model signals from forensic conclusions.
- Use "inconclusive" when no face is found, weights fail to load, the clip is out of scope, or modality outputs are ambiguous.
- Never say "100% real" or "definitely fake".
- Show Grad-CAM-style visualization as model attribution, never as a ground-truth manipulation mask.
- Preserve original evidence and its SHA-256 digest.


## Voice synthesis / spoof-speech indicator
The audio report may expose a synthetic/spoof-direction research-model indicator with checkpoint ID, number of speech windows and median model score. It is not speaker identification, source attribution, or proof of voice cloning. Keep the acoustic waveform/timeline review separate from the synthetic-speech classifier indicator.

## Cross-modal demonstration
Use one case containing image OCR, sampled video-frame text, an audio transcript and investigator-supplied claim text. TruthLens compares each available pair using lexical token overlap and displays previews, shared terms and comparison scope. This is not semantic contradiction detection or fact verification. A human analyst must inspect original media and primary sources.

The investigator review UI stores a decision and bounded rationale for each technical finding, scoped to the authenticated user and report. PDF export includes saved decisions, available reviewer metadata, timestamps, per-asset outputs, cross-modal comparisons and limitations. A saved finding decision is an analyst assessment of that finding, not a media-authenticity verdict.


## Confidence and uncertainty reporting
TruthLens must keep three concepts separate: the raw model score, the model's threshold-based signal direction, and the evidence-support/uncertainty assessment. A softmax score is not a calibrated probability or a confidence percentage.

For every assessed media item, the API reports a separate uncertainty assessment based on:
- Distance of the aggregate score from the modality decision threshold and whether the score falls in the ambiguous band.
- Variation and disagreement among available face crops, sampled video frames, or speech windows.
- Number of inference samples actually assessed.

The assessment uses labels such as high, moderate and lower review uncertainty. These are heuristic evidence-review labels, not statistical confidence intervals or estimates of the probability of correctness. A single sample is marked as limited support. Mixed real-direction and fake-direction sample results increase uncertainty even when the aggregate median crosses a decision threshold.

No model is labelled calibrated. Calibration requires a representative, independently labelled held-out validation dataset for that specific model and deployment domain, followed by reliability analysis and a documented calibration method. Do not claim calibrated confidence until that evaluation is completed.


### Calibration candidate workflow

Use one CSV per model checkpoint and modality pipeline, with columns `label,score,group_id` (0=real, 1=fake). Each `group_id` must identify the independent source group such as original video, subject/session or source image—not a frame/window extracted from that source.

Example:

```powershell
python evaluate_calibration.py labelled_scores.csv --model-id "Hemgg/Deepfake-audio-detection" --pipeline audio --group-column group_id --output calibration-audio-candidate.json
```

Review the output's group split, dataset hash, raw vs calibrated held-out Brier/log-loss/ECE and reliability bins. Do not approve the artifact unless the held-out sources are independent and representative of the deployment domain. Calibration is specific to the checkpoint, pipeline and score distribution.


For a deployment that runs several model pipelines, point `TRUTHLENS_CALIBRATION_ARTIFACT` at a private directory containing one JSON artifact per checkpoint/pipeline pair. TruthLens selects only an artifact whose exact `model_id` and `pipeline` match the current evidence item; unapproved, mismatched or invalid candidates are not applied.


## Claim-to-source semantic review
When OpenAlex returns an abstract, TruthLens can compare the extracted claim against the abstract using the optional local NLI checkpoint. Both premise directions are evaluated, and the report displays source metadata, available abstract excerpts and candidate entailment/contradiction/neutral scores. NLI scores are uncalibrated and trained on general English NLI datasets; they are not evidence that the cited source directly supports the claim. Review the paper, methods, date, population, and exact cited passage manually.


## Source-group score evaluation

Use `evaluate_scores.py` with `--group-column video_id` (or another independent source ID) to report both row-level metrics and source-level metrics. Source-level scoring takes the median fake-direction score within each group so a video with many frames does not dominate a test set. The same source identifier must cover every frame/window/crop derived from one original item. Groups with conflicting labels are rejected. Prefer source-level metrics when presenting detector performance to evaluators.
