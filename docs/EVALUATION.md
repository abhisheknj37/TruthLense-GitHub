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
