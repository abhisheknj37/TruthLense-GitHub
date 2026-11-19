import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import LiveCapture from "./LiveCapture";
import {
  Activity, AlertTriangle, ArrowRight, AudioLines, Bell, CalendarDays,
  Check, CheckCircle2, ChevronRight, Clock3, Download, FileBarChart2,
  FileImage, FileJson, Fingerprint, FolderOpen, Globe, Headphones, Image as ImageIcon,
  LayoutDashboard, LoaderCircle, LockKeyhole, LogOut, Mail, Mic, Plus, Printer,
  Search, Settings, Shield, ShieldAlert, ShieldCheck, Trash2, UploadCloud,
  UserRound, Video, X, Copy
} from "lucide-react";
import {
  createEmailAccount, firebaseConfigured, observeAuth,
  signInEmail, signInGoogle, signOutUser
} from "./firebase";

const API = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(new RegExp("/+$"), "");
const STORAGE_KEY = "truthlens.saved-investigations.v1";
const MAX_FILES = 8;
const MAX_BYTES = 40 * 1024 * 1024;
const ALLOWED = {
  image: ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
  video: ["mp4", "mov", "avi", "mkv", "webm"],
  audio: ["wav", "flac", "ogg"]
};

async function sha256Hex(file) {
  if (!globalThis.crypto?.subtle) throw new Error("Secure browser hashing is unavailable.");
  const bytes = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

function formatBytes(bytes = 0) {
  if (!bytes) return "0 B";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}
function formatDate(value) {
  if (!value) return "Date unavailable";
  return new Date(value).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}
function fileType(file) {
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (ALLOWED.image.includes(ext) || file.type.startsWith("image/")) return "image";
  if (ALLOWED.video.includes(ext) || file.type.startsWith("video/")) return "video";
  if (ALLOWED.audio.includes(ext) || file.type.startsWith("audio/")) return "audio";
  return "unsupported";
}
function typeIcon(type, size = 17) {
  if (type === "video") return <Video size={size} />;
  if (type === "audio") return <AudioLines size={size} />;
  return <ImageIcon size={size} />;
}
function scorePercent(value) {
  return typeof value === "number" ? (value * 100).toFixed(1) + "%" : "—";
}
function predictionLabel(value = "inconclusive") {
  return ({
    fake_signal: "Manipulation-related model signal",
    real_signal: "Authenticity-related model signal",
    inconclusive: "Inconclusive"
  })[value] || String(value).replaceAll("_", " ");
}
function cleanReports(reports) {
  return Array.isArray(reports) ? reports.filter((item) => item && item.report_id && Array.isArray(item.assets)) : [];
}
function getReportsForUser(uid) {
  if (!uid) return [];
  try { return cleanReports(JSON.parse(localStorage.getItem(STORAGE_KEY + "." + uid) || "[]")); }
  catch { return []; }
}
function authErrorMessage(error) {
  const code = error?.code || "";
  if (code.includes("email-already-in-use")) return "An account already exists for this email. Sign in instead.";
  if (code.includes("invalid-credential") || code.includes("wrong-password") || code.includes("user-not-found")) return "Email or password is incorrect.";
  if (code.includes("weak-password")) return "Choose a password with at least 6 characters.";
  if (code.includes("invalid-email")) return "Enter a valid email address.";
  if (code.includes("popup-closed-by-user")) return "Google sign-in was cancelled.";
  if (code.includes("operation-not-allowed")) return "Enable this sign-in provider in Firebase Console → Authentication → Sign-in method.";
  if (code.includes("unauthorized-domain")) return "Add localhost to Firebase Authentication → Settings → Authorized domains.";
  return error?.message || "Authentication failed. Please try again.";
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}
function downloadJson(report) {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "truthlens-case-" + report.report_id.slice(0, 8) + ".json";
  link.click();
  URL.revokeObjectURL(url);
}
function getAssessmentStatusLabel(status) {
  const labels = {
    model_indicators_available_human_review_required: "Model indicators available",
    inconclusive: "Inconclusive",
  };
  return labels[status] || String(status || "Not available").replaceAll("_", " ");
}

function getInvestigatorReviewProgress(report) {
  const findings = (report.assets || []).flatMap((asset) => {
    const observations = asset.observations?.length ? asset.observations : [{ code: "MODEL_ASSESSMENT" }];
    return observations.map((item, index) => asset.asset_id + ":" + String(item.code || index));
  });
  const saved = new Map((report.investigator_reviews || []).map((item) => [item.finding_id, item]));
  const reviewed = findings.filter((id) => ["confirmed", "rejected", "inconclusive"].includes(saved.get(id)?.status)).length;
  return { total: findings.length, reviewed, complete: findings.length > 0 && reviewed === findings.length };
}

function exportPdf(report) {
  const reviewProgress = getInvestigatorReviewProgress(report);
  const rows = report.assets.map((asset) => {
    const model = asset.model_assessment || {};
    const meta = Object.entries(asset.metadata || {}).filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
      .map(([key, value]) => "<tr><th>" + escapeHtml(key.replaceAll("_", " ")) + "</th><td>" + escapeHtml(value) + "</td></tr>").join("");
    return "<section class='asset'><h2>" + escapeHtml(asset.filename) + "</h2><p>" +
      escapeHtml(String(asset.media_type).toUpperCase()) + " · " + escapeHtml(formatBytes(asset.size_bytes)) +
      "</p><p><b>Model signal:</b> " + escapeHtml(predictionLabel(model.prediction)) +
      "</p><p><b>Model:</b> " + escapeHtml(model.model_name || "Not available") +
      "</p><p><b>SHA-256:</b><br><code>" + escapeHtml(asset.sha256) +
      "</code></p><p><b>Evidence reference:</b> <code>" + escapeHtml(asset.asset_id || "Not assigned") + "</code></p>" +
      "<h3>Observed findings</h3>" + ((asset.observations || []).length ? "<ul>" + asset.observations.map((item) => "<li><b>" + escapeHtml(item.title || item.code || "Observation") + ":</b> " + escapeHtml(item.detail || item.note || item.interpretation || "No detail supplied") + " <span class=muted>(" + escapeHtml(item.severity || "unclassified") + ")</span></li>").join("") + "</ul>" : "<p>No structured observations were returned for this evidence item.</p>") +
      "<table>" + meta + "</table>" +
      (asset.forensic_parameters && Object.keys(asset.forensic_parameters).length ? "<h3>Image forensic parameters</h3><pre>" + escapeHtml(JSON.stringify(asset.forensic_parameters, null, 2)) + "</pre>" : "") +
      (asset.provenance ? "<h3>Evidence provenance</h3><pre>" + escapeHtml(JSON.stringify(asset.provenance, null, 2)) + "</pre>" : "") +
      (asset.text_extraction ? "<h3>OCR / transcription</h3><pre>" + escapeHtml(JSON.stringify(asset.text_extraction, null, 2)) + "</pre>" : "") +
      (asset.model_assessment?.audio_review_timeline ? "<h3>Audio waveform and acoustic review</h3><pre>" + escapeHtml(JSON.stringify(asset.model_assessment.audio_review_timeline, null, 2)) + "</pre>" : "") +
      (asset.video_analysis?.temporal_forensics ? "<h3>Temporal video review</h3><pre>" + escapeHtml(JSON.stringify(asset.video_analysis.temporal_forensics, null, 2)) + "</pre>" : "") +
      (asset.video_analysis?.face_identity_consistency ? "<h3>Face consistency review</h3><pre>" + escapeHtml(JSON.stringify(asset.video_analysis.face_identity_consistency, null, 2)) + "</pre>" : "") +
      "</section>";
  }).join("");
  const limitations = (report.limitations || []).map((item) => "<li>" + escapeHtml(item) + "</li>").join("");
  const html = "<!doctype html><html><head><title>TruthLens case report</title><style>" +
    "body{font:14px Arial;color:#172033;margin:36px}h1{margin-bottom:4px}h2{font-size:17px}p{line-height:1.5}" +
    ".muted{color:#62718a}.asset{border:1px solid #cbd5e1;border-radius:8px;padding:16px;margin:18px 0;break-inside:avoid}" +
    "code{font-size:10px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px;border-top:1px solid #e2e8f0}" +
    "th{text-transform:capitalize;width:35%}li{margin:7px 0}@media print{button{display:none}}</style></head><body>" +
    "<h1>TruthLens · Forensic case report</h1><p class='muted'>Case " + escapeHtml(report.report_id) +
    " · " + escapeHtml(formatDate(report.created_at)) + "</p><h2>Case summary</h2><p>" +
    escapeHtml(report.summary) + "</p><p><b>Automated assessment:</b> " + escapeHtml(getAssessmentStatusLabel(report.overall_status)) +
    " · <b>Human oversight:</b> " + (report.human_review_required ? "Required" : "Not specified") +
    " · <b>Investigator review:</b> " + reviewProgress.reviewed + "/" + reviewProgress.total + " findings reviewed (" + (reviewProgress.complete ? "Complete" : "Pending") + ")" +
    "</p><h2>Evidence inventory and findings</h2>" + rows +
    "<h2>Cross-modal review</h2>" + (report.cross_modal_review || []).map((item) =>
      "<p><b>" + escapeHtml(item.type || "Review") + " · " + escapeHtml(item.status) +
      ":</b> " + escapeHtml(item.detail || item.note || item.interpretation || JSON.stringify(item)) + "</p>").join("") +
    (report.text_analysis ? "<h2>Text and context review</h2><pre>" + escapeHtml(JSON.stringify(report.text_analysis, null, 2)) + "</pre>" : "") +
    (report.investigator_reviews?.length ? "<h2>Investigator decisions</h2><table>" + report.investigator_reviews.map((item) => "<tr><th>" + escapeHtml(item.finding_id) + "</th><td>" + escapeHtml(item.status) + "</td><td>" + escapeHtml(item.comment || "No analyst note") + "</td><td>" + escapeHtml(item.updated_at || "") + "</td></tr>").join("") + "</table>" : "<h2>Investigator decisions</h2><p>No saved investigator decisions were available at export time.</p>") +
    (report.evidence_events?.length ? "<h2>Hash-linked evidence events</h2><pre>" + escapeHtml(JSON.stringify(report.evidence_events, null, 2)) + "</pre>" : "") +
    "<h2>Limitations and review notes</h2><ul>" + limitations + "</ul>" +
    "<p class='muted'>Model scores are not calibrated probabilities or proof of authenticity. Human review and original-source context are required.</p><div class='footer'>TRUTHLENS · Evidence references are tied to the case record and SHA-256 values shown above. Preserve original source media separately.</div>" +
    "<script>window.onload=function(){window.print()}<\/script></body></html>";
  const win = window.open("", "_blank");
  if (!win) return false;
  win.document.open();
  win.document.write(html);
  win.document.close();
  return true;
}

function BrandGlyph({ size = 30 }) {
  return <svg width={size} height={size} viewBox="0 0 40 40" fill="none" aria-hidden="true">
    <circle cx="20" cy="20" r="13.2" stroke="currentColor" strokeWidth="2.2"/>
    <circle cx="20" cy="20" r="6.1" stroke="currentColor" strokeWidth="2.2"/>
    <circle cx="20" cy="20" r="2" fill="currentColor"/>
    <path d="M20 1.5v7M20 31.5v7M1.5 20h7M31.5 20h7M7 7l5 5M28 28l5 5M33 7l-5 5M12 28l-5 5" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
  </svg>;
}

function ScoreBar({ label, value }) {
  if (typeof value !== "number") return null;
  return <div className="score-row">
    <div><span>{label.replaceAll("_", " ")}</span><b>{scorePercent(value)}</b></div>
    <div className="score-track"><i style={{ width: Math.max(0, Math.min(100, value * 100)) + "%" }} /></div>
  </div>;
}

function ForensicParameters({ asset }) {
  const params = asset.forensic_parameters || {};
  if (asset.media_type !== "image" || !Object.keys(params).length) return null;
  const rows = [
    ["Lighting variation", params.lighting?.grid_mean_intensity_std, "Grid brightness variation"],
    ["Texture detail", params.texture?.laplacian_variance, "Laplacian variance"],
    ["Noise estimate", params.noise?.robust_high_frequency_sigma, "High-frequency residual"],
    ["Compression heuristic", params.compression?.eight_pixel_boundary_gradient_ratio, "8-pixel boundary ratio"]
  ];
  return <section className="forensic-parameters">
    <div className="section-kicker">IMAGE FORENSIC PARAMETERS</div>
    <div className="forensic-parameter-grid">
      {rows.map(([label, value, hint]) => <div className="forensic-parameter" key={label}>
        <small>{label}</small><b>{typeof value === "number" ? value.toFixed(3) : "N/A"}</b><span>{hint}</span>
      </div>)}
    </div>
    <p className="model-note">{params.interpretation || "Descriptive measurements, not standalone manipulation detectors."}</p>
  </section>;
}

function Pipeline({ loading, uploadProgress, report }) {
  const steps = [
    ["Media inspection", "File structure, metadata and SHA-256"],
    ["Modality analysis", "Image, sampled video frames or speech windows"],
    ["Evidence correlation", "Compare paired media model signals"],
    ["Human review & reporting", "Review findings, limitations and export"]
  ];
  const reviewProgress = report ? getInvestigatorReviewProgress(report) : null;
  const stateFor = (index) => {
    if (report) {
      if (index < 3) return "complete";
      return reviewProgress?.complete ? "complete" : "review";
    }
    if (!loading) return "waiting";
    if (index === 0) return uploadProgress >= 100 ? "complete" : "active";
    if (index === 1) return uploadProgress >= 100 ? "active" : "waiting";
    return "waiting";
  };
  return <div className="pipeline-grid">
    {steps.map(([title, description], index) => {
      const state = stateFor(index);
      return <div className={"pipeline-step " + state} key={title}>
        <div className="pipeline-step-top"><span className="step-number">0{index + 1}</span>
          <span className="step-state">{state === "complete" ? <Check size={13} /> : state === "active" ? <LoaderCircle className="spin" size={13} /> : state === "review" ? <Clock3 size={13} /> : <span className="state-dot" />}</span>
        </div>
        <b>{title}</b><small>{description}</small>
        <span className="state-label">{state === "complete" ? "Completed" : state === "active" ? "In progress" : state === "review" ? (reviewProgress?.total ? "Review pending" : "Review available") : "Queued"}</span>
      </div>;
    })}
  </div>;
}

function ModelAssessment({ asset }) {
  const model = asset.model_assessment || {};
  const video = asset.video_analysis || {};
  const details = asset.media_type === "video" ? video : model;
  const assessed = details.status === "model_assessed";
  const frames = video.frames || [];
  const segments = model.segments || [];
  return <div className="model-result">
    <div className="section-kicker">MODEL ASSESSMENT</div>
    <div className="model-result-head">
      <b>{predictionLabel(details.prediction)}</b>
      <span className={"model-pill " + (assessed ? "available" : "pending")}>{assessed ? "Signal available" : "Inconclusive"}</span>
    </div>
    {details.model_name && <small className="model-name">{details.model_name}</small>}
    {details.scores && <div className="score-list">
      {Object.entries(details.scores).filter(([key, value]) => typeof value === "number" && (key.includes("fake") || key.includes("real") || key.includes("ai_generated")))
        .map(([key, value]) => <ScoreBar key={key} label={key} value={value} />)}
    </div>}
    {details.note && <p className="model-note">{details.note}</p>}
    {asset.media_type === "image" && model.faces?.length > 0 && <div className="face-results">
      <b>Detected face crops · analyst review</b>
      <p className="model-note">Check that each crop contains a real face before interpreting its score. Incorrect detections can affect the median.</p>
      {model.faces.map((face) => <div className="face-result" key={face.face_index}>
        <div className="face-meta"><span>Face {face.face_index + 1} · {predictionLabel(face.prediction)}</span>
          <small>Region: {(face.bounding_box_xywh || []).join(", ")}</small></div>
        <div className="face-review-grid">
          {face.crop_preview_base64 && <img className="gradcam-image" src={"data:image/jpeg;base64," + face.crop_preview_base64} alt={"Detected face crop " + (face.face_index + 1)} />}
          <div className="score-list">
            <ScoreBar label="Fake model score" value={face.scores?.fake} />
            <ScoreBar label="Real model score" value={face.scores?.real} />
          </div>
        </div>
        {face.gradcam_overlay_base64 && <img className="gradcam-image" src={"data:image/png;base64," + face.gradcam_overlay_base64} alt="Transformer patch attribution overlay" />}
        <p className="cam-caption">Attribution is not a manipulation mask or proof. Scores are uncalibrated model outputs.</p>
      </div>)}
      {model.aggregation && <div className="provenance-note"><b>HOW THE OVERALL SCORE IS CALCULATED</b>
        <span>{model.aggregation.face_count} detected face(s) · median aggregation</span>
        <small>{model.aggregation.explanation}</small>
        <small>Individual fake scores: {(model.aggregation.individual_fake_scores || []).map(scorePercent).join(" · ")}</small>
      </div>}
    </div>}
    {asset.media_type === "video" && video.temporal_forensics && <div className="provenance-note">
      <b>TEMPORAL VIDEO REVIEW</b><span>{video.temporal_forensics.status || "not available"}</span>
      <small>{video.temporal_forensics.interpretation || video.temporal_forensics.reason}</small>
      <p>Scene changes: {(video.temporal_forensics.scene_change_timestamps_seconds || []).join(", ") || "None flagged"} seconds</p>
      <p>Motion discontinuities: {(video.temporal_forensics.motion_discontinuity_timestamps_seconds || []).join(", ") || "None flagged"} seconds</p>
      {video.temporal_forensics.trained_model_assessment?.note && <small>{video.temporal_forensics.trained_model_assessment.note}</small>}
    </div>}
    {asset.media_type === "video" && video.face_identity_consistency && <div className="provenance-note">
      <b>FACE CONSISTENCY REVIEW</b><span>{video.face_identity_consistency.status}</span><small>{video.face_identity_consistency.note}</small>
      {video.face_identity_consistency.identity_change_review_points?.map((point, index) => <p key={index}>{point.timestamp_seconds}s · similarity {point.cosine_similarity_to_previous_sample}</p>)}
    </div>}
    {asset.media_type === "video" && frames.length > 0 && <div className="frame-results">
      <b>Sampled video timeline</b>
      <div className="frame-timeline">
        {frames.map((frame) => <div className="timeline-item" key={frame.frame_index}>
          <span className={"timeline-pin " + (frame.prediction === "fake_signal" ? "signal" : frame.prediction === "real_signal" ? "low" : "")} />
          <small>{frame.timestamp_seconds == null ? "Frame " + frame.frame_index : frame.timestamp_seconds + "s"}</small>
          <b>#{frame.frame_index}</b>
          <em>{String(frame.prediction || "inconclusive").replaceAll("_", " ")}</em>
        </div>)}
      </div>
      <p className="model-note">Frames are sampled and assessed independently; this is not temporal or lip-sync detection.</p>
    </div>}
    {asset.media_type === "audio" && model.audio_review_timeline?.waveform?.length > 0 && <div className="audio-segments">
      <b>Audio waveform · acoustic review</b>
      <div className="waveform" role="img" aria-label="Audio amplitude waveform" style={{ display: "flex", alignItems: "center", gap: 2, height: 64, padding: "8px 0", overflow: "hidden" }}>
        {model.audio_review_timeline.waveform.map((point, index) => <span key={index} title={point.timestamp_seconds + "s"} style={{ flex: 1, minWidth: 2, borderRadius: 2, background: "var(--accent, #4f7cff)", height: Math.max(3, Math.min(100, Math.abs(point.amplitude_max - point.amplitude_min) * 90)) + "%" }} />)}
      </div>
      <small>{model.audio_review_timeline.duration_seconds}s · {model.audio_review_timeline.method}</small>
      <p className="model-note">{model.audio_review_timeline.interpretation}</p>
      {model.audio_review_timeline.acoustic_review_points?.map((point, index) => <p key={index}>{point.timestamp_seconds}s · unusual energy review cue · RMS {point.rms}</p>)}
    </div>}
    {asset.media_type === "audio" && segments.length > 0 && <div className="audio-segments">
      <b>Speech segment analysis</b>
      {segments.map((segment, index) => <div className="audio-segment" key={index}>
        <div><span>{segment.start_seconds}s – {segment.end_seconds}s</span><b>{scorePercent(segment.fake_model_score)}</b></div>
        <div className="score-track"><i style={{ width: Math.max(0, Math.min(100, (segment.fake_model_score || 0) * 100)) + "%" }} /></div>
        <small>{segment.fake_model_score >= 0.65 ? "Higher model signal" : segment.fake_model_score <= 0.35 ? "Lower model signal" : "Ambiguous band"}</small>
      </div>)}
    </div>}
  </div>;
}

function EvidenceCard({ asset }) {
  return <article className="evidence-card">
    <div className="asset-head"><span>{typeIcon(asset.media_type)} {String(asset.media_type).toUpperCase()} EVIDENCE</span><small>{formatBytes(asset.size_bytes)}</small></div>
    <h3>{asset.filename}</h3>
    <div className="metadata-grid">
      {Object.entries(asset.metadata || {}).filter(([key, value]) => typeof value !== "object" && value !== null).slice(0, 8)
        .map(([key, value]) => <div key={key}><small>{key.replaceAll("_", " ")}</small><b>{String(value)}</b></div>)}
    </div>
    <div className="hash-line"><small><Fingerprint size={12} /> SHA-256 EVIDENCE FINGERPRINT</small><code>{asset.sha256}</code>
      <span className={"hash-verification " + (asset.integrity_verification?.status || "recorded")}>{asset.integrity_verification?.status === "match" ? "✓ Browser and API digests match" : asset.integrity_verification?.status === "mismatch" ? "⚠ Digest mismatch — review required" : "Fingerprint recorded · not rechecked"}</span></div>
    <ForensicParameters asset={asset} />
    {asset.provenance && <div className="provenance-note"><b>PROVENANCE</b>
      <span>SHA-256 recorded · C2PA: {asset.provenance.content_credentials?.status || "not verified"}</span>
      <small>{asset.provenance.content_credentials?.note || asset.provenance.content_credentials?.detail || "No trusted provenance conclusion is available."}</small>
    </div>}
    {asset.text_extraction && <div className="provenance-note">
      <b>TEXT EXTRACTION · {String(asset.text_extraction.status || "unknown").toUpperCase()}</b>
      <span>{asset.text_extraction.note || "OCR / transcription status"}</span>
      {asset.text_extraction.text && <p className="extracted-text">{asset.text_extraction.text}</p>}
      {(asset.text_extraction.frames || []).map((frame, index) => <p key={index}><b>{frame.timestamp_seconds ?? "—"}s:</b> {frame.text}</p>)}
    </div>}
    <ModelAssessment asset={asset} />
    {(asset.observations || []).map((item) => <div className="observation" key={item.code}>
      <CheckCircle2 size={16} /><div><b>{item.title}</b><p>{item.detail || item.note || item.interpretation || JSON.stringify(item)}</p></div>
    </div>)}
  </article>;
}

function TextContextReview({ report }) {
  const result = report.text_analysis || {};
  if (!result.status || result.status === "not_provided") return null;
  return <section className="text-review-panel">
    <div className="section-kicker">TEXT & CONTEXT TRIAGE</div>
    <h3>Supplied claim / caption review</h3>
    <div className="metadata-grid">
      <div><small>WORDS</small><b>{result.word_count ?? "—"}</b></div>
      <div><small>DATES FOUND</small><b>{(result.dates || []).length}</b></div>
      <div><small>URLS FOUND</small><b>{(result.urls || []).length}</b></div>
      <div><small>LANGUAGE CUES</small><b>{(result.language_cues || []).length}</b></div>
    </div>
    {(result.language_cues || []).map((cue, index) => <div className="observation" key={cue.type + index}>
      <AlertTriangle size={16}/><div><b>{cue.type.replaceAll("_", " ")}</b><p>{(cue.matched_terms || []).join(", ")} — {cue.interpretation}</p></div>
    </div>)}
    {(result.dates || []).length > 0 && <p><b>Dates extracted:</b> {(result.dates || []).join(", ")}</p>}
    {(result.urls || []).length > 0 && <p><b>URLs supplied:</b> {(result.urls || []).join(", ")}</p>}
    {result.claim_verification?.results?.map((item, index) => <div className="observation" key={index}>
      <div><b>Claim · {item.verdict?.replaceAll("_", " ")}</b><p>{item.claim}</p>
        <small>{(item.candidate_sources || []).slice(0, 3).map((source) => source.title + (source.year ? " (" + source.year + ")" : "")).join(" · ") || "No candidate scholarly sources returned"}</small>
      </div>
    </div>)}
    <p className="model-note">Rule-based triage and scholarly-source discovery only. Candidate papers are not proof that a claim is true or false; verify original sources and relevant passages.</p>
  </section>;
}

function InvestigatorReview({ report, user, onChange }) {
  const [reviews, setReviews] = useState({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState("");
  const [error, setError] = useState("");

  const findings = report.assets.flatMap((asset) => {
    const observations = asset.observations?.length ? asset.observations : [{
      code: "MODEL_ASSESSMENT",
      title: "Model assessment",
      detail: asset.model_assessment?.note || asset.model_assessment?.prediction || "No additional explanation returned."
    }];
    return observations.map((item, index) => ({
      id: asset.asset_id + ":" + String(item.code || index),
      asset: asset.filename,
      title: item.title || item.code || "Finding",
      detail: item.detail || item.note || item.interpretation || "No explanation available."
    }));
  });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const token = await user.getIdToken();
        const response = await axios.get(API + "/api/reports/" + encodeURIComponent(report.report_id) + "/reviews", {
          headers: { Authorization: "Bearer " + token }, timeout: 15000
        });
        if (cancelled) return;
        const next = {};
        (response.data.reviews || []).forEach((item) => { next[item.finding_id] = item; });
        setReviews(next);
        onChange(Object.values(next));
        setError("");
      } catch (err) {
        if (!cancelled) setError(err.response?.data?.detail || "Could not load saved investigator decisions.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [report.report_id, user]);

  function edit(id, field, value) {
    setReviews((current) => ({
      ...current,
      [id]: { ...(current[id] || { finding_id: id, status: "unreviewed", comment: "" }), [field]: value }
    }));
  }

  async function save(id) {
    setSaving(id);
    setError("");
    try {
      const item = reviews[id] || { status: "unreviewed", comment: "" };
      const token = await user.getIdToken();
      const response = await axios.put(
        API + "/api/reports/" + encodeURIComponent(report.report_id) + "/reviews/" + encodeURIComponent(id),
        { status: item.status || "unreviewed", comment: item.comment || "" },
        { headers: { Authorization: "Bearer " + token }, timeout: 15000 }
      );
      setReviews((current) => {
        const next = { ...current, [id]: response.data };
        onChange(Object.values(next));
        return next;
      });
    } catch (err) {
      setError(err.response?.data?.detail || "Could not save this investigator decision.");
    } finally {
      setSaving("");
    }
  }

  return <section className="cross-review investigator-review">
    <div className="cross-header"><div><span className="section-kicker">HUMAN REVIEW WORKFLOW</span><h3>Investigator decisions</h3></div>
      <span className="model-pill pending">{loading ? "Loading…" : findings.length + " findings"}</span></div>
    <p className="model-note">Record an analyst decision for each finding. These decisions are stored separately from automated model output.</p>
    {findings.map((finding) => {
      const item = reviews[finding.id] || {};
      return <div className="cross-item" key={finding.id}>
        <b>{finding.title}</b><small>Evidence: {finding.asset}</small><p>{finding.detail}</p>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(150px, 220px) 1fr auto", gap: 10, alignItems: "start", marginTop: 10 }}>
          <select aria-label={"Decision for " + finding.title} value={item.status || "unreviewed"} onChange={(event) => edit(finding.id, "status", event.target.value)}>
            <option value="unreviewed">Unreviewed</option><option value="confirmed">Confirmed</option>
            <option value="rejected">Rejected</option><option value="inconclusive">Inconclusive</option>
          </select>
          <textarea aria-label={"Analyst note for " + finding.title} rows={2} maxLength={4000} placeholder="Analyst notes / rationale…" value={item.comment || ""} onChange={(event) => edit(finding.id, "comment", event.target.value)} />
          <button className="button button-quiet" disabled={saving === finding.id} onClick={() => save(finding.id)}>
            {saving === finding.id ? "Saving…" : "Save decision"}
          </button>
        </div>
        {item.updated_at && <small>Last saved: {formatDate(item.updated_at)}</small>}
      </div>;
    })}
    {error && <p className="model-note" role="alert">{error}</p>}
  </section>;
}

function ReportView({ report, onJson, onPdf, user, onReviewsChange }) {
  const assessed = report.assets.filter((asset) => asset.model_assessment?.status === "model_assessed").length;
  const findings = report.assets.reduce((sum, asset) => sum + (asset.observations || []).filter((item) => item.severity === "warning").length, 0);
  const inconclusive = report.assets.length - assessed;
  return <section className="report-section">
    <div className="report-title">
      <div><span className="section-kicker">CASE REPORT · {report.report_id.slice(0, 8).toUpperCase()}</span>
        <h2>{report.case_title || "Forensic assessment"}</h2>
        <p className="report-date"><CalendarDays size={13} /> {formatDate(report.created_at)}</p>
      </div>
      <div className="report-actions">
        <button className="button button-quiet" onClick={onJson}><FileJson size={15} /> JSON</button>
        <button className="button button-primary" onClick={onPdf}><Printer size={15} /> Export PDF</button>
      </div>
    </div>
    <TextContextReview report={report} />
    <div className="report-stats">
      <div className="report-stat"><span>FILES ANALYSED</span><b>{report.assets.length}</b><small>Evidence items received</small></div>
      <div className="report-stat"><span>MODEL ASSESSMENTS</span><b>{assessed}</b><small>Modality signals available</small></div>
      <div className="report-stat"><span>INCONCLUSIVE</span><b>{inconclusive}</b><small>Require additional context</small></div>
      <div className="report-stat"><span>REVIEW FINDINGS</span><b>{findings}</b><small>Technical observations flagged</small></div>
    </div>
    <div className="report-summary">
      <div className="summary-icon"><ShieldAlert size={20} /></div>
      <div><span className="section-kicker">INVESTIGATION SUMMARY</span><p>{report.summary}</p>
        <small>Case ID: {report.report_id} · Model oversight: {report.human_review_required ? "Human review required" : "Not specified"} · Investigator review: {getInvestigatorReviewProgress(report).reviewed}/{getInvestigatorReviewProgress(report).total} findings ({getInvestigatorReviewProgress(report).complete ? "Complete" : "Pending"})</small>
      </div>
    </div>
    <div className="evidence-heading"><div><span className="section-kicker">EVIDENCE REGISTER</span><h3>Asset-by-asset findings</h3></div><span>{report.assets.length} evidence items</span></div>
    <div className="asset-grid">{report.assets.map((asset) => <EvidenceCard key={asset.asset_id} asset={asset} />)}</div>
    <InvestigatorReview report={report} user={user} onChange={onReviewsChange} />
    {report.cross_modal_review?.length > 0 && <div className="cross-review">
      <div className="cross-header"><div><span className="section-kicker">RELATIONSHIP REVIEW</span><h3>Cross-modal comparison</h3></div><span className="model-pill pending">No fused authenticity score</span></div>
      {report.cross_modal_review.map((item, index) => <div className="cross-item" key={index}>
        <span className="model-pill pending">{String(item.status).replaceAll("_", " ")}</span>
        <p>{item.detail}</p>
      </div>)}
    </div>}
    {report.evidence_events?.length > 0 && <div className="cross-review">
      <div className="cross-header"><div><span className="section-kicker">EVIDENCE TRACE</span><h3>Hash-linked event history</h3></div>
        <span className="model-pill pending">Local ledger</span>
      </div>
      {report.evidence_events.map((event, index) => <div className="cross-item" key={event.event_hash || index}>
        <b>{String(event.event || "event").replaceAll("_", " ")}</b>
        <p>{event.detail}</p><small>{event.timestamp} · Event hash: {event.event_hash || "not available"}</small>
      </div>)}
      <p className="model-note">The local hash chain can detect changes within this ledger. It is not externally anchored or protected against full database replacement.</p>
    </div>}
    <div className="limitations"><b><Shield size={15} /> Limitations & analyst notes</b>
      {(report.limitations || []).map((item) => <p key={item}>• {item}</p>)}
    </div>
  </section>;
}

function CaseRow({ report, onOpen }) {
  const assessed = report.assets.filter((asset) => asset.model_assessment?.status === "model_assessed").length;
  return <button className="case-row" onClick={onOpen}>
    <span className="case-icon"><FolderOpen size={18} /></span>
    <span className="case-main"><b>{report.case_title || "Forensic investigation"}</b><small>CASE #{report.report_id.slice(0, 8).toUpperCase()} · {report.assets.length} files</small></span>
    <span className="case-date">{formatDate(report.created_at)}</span>
    <span className="case-indicator">{assessed} assessed</span>
    <ChevronRight size={17} />
  </button>;
}

function IntegrityPanel({ events, loading, error }) {
  const [copiedId, setCopiedId] = useState("");
  const assets = events.flatMap((event) => (event.evidence || []).map((asset) => ({
    ...asset, caseTitle: event.case_title
  }))).slice(0, 6);
  async function copyHash(asset) {
    try {
      await navigator.clipboard.writeText(asset.sha256 || "");
      setCopiedId(asset.asset_id);
      window.setTimeout(() => setCopiedId(""), 1800);
    } catch { setCopiedId("unavailable"); }
  }
  const matched = assets.filter((asset) => asset.integrity_verification?.status === "match").length;
  return <section className="content-panel integrity-panel">
    <div className="insight-panel-title"><div><span className="section-kicker">SERVER-RECORDED INTEGRITY</span><h2>SHA-256 fingerprints</h2></div><span className="insight-icon"><Fingerprint size={19} /></span></div>
    <div className="integrity-summary"><b>{matched} / {assets.length}</b><span>of the latest six evidence files independently matched the browser's pre-upload digest</span></div>
    {assets.length ? <div className="integrity-list">{assets.map((asset) => {
      const state = asset.integrity_verification?.status || "not_compared";
      const label = state === "match" ? "Verified match" : state === "mismatch" ? "Mismatch" : state === "invalid_client_digest" ? "Invalid digest" : "Not compared";
      return <div className="integrity-item" key={asset.asset_id}>
        <div className="integrity-item-top"><span>{typeIcon(asset.media_type, 14)}<b title={asset.filename}>{asset.filename}</b></span><span className={"integrity-status " + state}>{label}</span></div>
        <div className="integrity-digest"><code title={asset.sha256}>{asset.sha256 || "SHA-256 unavailable"}</code><button onClick={() => copyHash(asset)} disabled={!asset.sha256} title="Copy server SHA-256"><Copy size={13} />{copiedId === asset.asset_id ? "Copied" : copiedId === "unavailable" ? "Blocked" : "Copy"}</button></div>
        <small className="integrity-case">{asset.caseTitle} · {asset.integrity_verification?.checked_at ? "Checked " + formatDate(asset.integrity_verification.checked_at) : "Server digest recorded"}</small>
      </div>;
    })}</div> : <div className="insight-empty"><Fingerprint size={24} /><b>{loading ? "Loading server evidence…" : error ? "Integrity service unavailable" : "No evidence fingerprints yet"}</b><span>{error || "Run a real investigation to create server-verified SHA-256 fingerprints."}</span></div>}
    <p className="insight-footnote">The API computes each digest from the received file and compares it with the browser's pre-upload digest. A match verifies transfer integrity, not who created the file or whether its contents are authentic.</p>
  </section>;
}

function ActivityTimeline({ events, onOpen, loading, error }) {
  return <section className="content-panel activity-panel">
    <div className="insight-panel-title"><div><span className="section-kicker">LIVE SERVER ACTIVITY</span><h2>Recent activity</h2></div><span className="insight-icon"><Clock3 size={18} /></span></div>
    {events.length ? <div className="activity-timeline">{events.slice(0, 5).map((event) =>
      <button className="activity-event" key={event.event_id} onClick={() => onOpen(event.report_id)}>
        <span className="activity-dot"><CheckCircle2 size={13} /></span><span className="activity-event-copy"><b>Investigation completed</b>
          <small>{event.case_title}</small>
          <small>{event.evidence_count} evidence item{event.evidence_count === 1 ? "" : "s"} · {event.verified_count} hash match{event.verified_count === 1 ? "" : "es"}</small></span>
        <time>{formatDate(event.created_at)}</time>
      </button>
    )}</div> : <div className="insight-empty"><Clock3 size={24} /><b>{loading ? "Connecting to activity service…" : error ? "Activity service unavailable" : "No investigations recorded"}</b><span>{error || "Completed investigations recorded by the backend will appear here."}</span></div>}
    <p className="insight-footnote">Authenticated server activity for the current Firebase account. Automatically refreshed every 8 seconds.</p>
  </section>;
}

function HeatmapPreview({ events, user, onOpen }) {
  const entry = events.flatMap((event) => (event.evidence || []).map((asset) => ({ event, asset })))
    .find(({ asset }) => (asset.heatmap_faces || []).some((face) => face.heatmap_available));
  const face = entry?.asset.heatmap_faces.find((item) => item.heatmap_available);
  const [imageUrl, setImageUrl] = useState("");
  const [heatmapStatus, setHeatmapStatus] = useState("idle");
  const reportId = entry?.event.report_id;
  const assetId = entry?.asset.asset_id;
  const faceIndex = face?.face_index ?? 0;

  useEffect(() => {
    let cancelled = false;
    let objectUrl = "";
    setImageUrl("");
    if (!reportId || !assetId || !user) {
      setHeatmapStatus("idle");
      return () => {};
    }
    setHeatmapStatus("loading");
    (async () => {
      try {
        const token = await user.getIdToken();
        const response = await axios.get(
          API + "/api/reports/" + encodeURIComponent(reportId) + "/heatmap",
          { params: { asset_id: assetId, face_index: faceIndex }, responseType: "blob", headers: { Authorization: "Bearer " + token }, timeout: 20000 }
        );
        objectUrl = URL.createObjectURL(response.data);
        if (!cancelled) {
          setImageUrl(objectUrl);
          setHeatmapStatus("ready");
        }
      } catch {
        if (!cancelled) setHeatmapStatus("error");
      }
    })();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [reportId, assetId, faceIndex, user]);

  return <section className="content-panel heatmap-panel">
    <div className="insight-panel-title"><div><span className="section-kicker">PERSISTED MODEL ATTRIBUTION</span><h2>Heatmap preview</h2></div><span className="insight-icon"><Activity size={18} /></span></div>
    {entry ? <div>
      <button className="heatmap-preview-button" onClick={() => onOpen(reportId)} title="Open the saved analysis report">
        <div className="heatmap-image-wrap">
          {imageUrl ? <img src={imageUrl} alt="Persisted Grad-CAM-style model attribution overlay" />
            : <div className="heatmap-loading">{heatmapStatus === "loading" ? <LoaderCircle className="spin" size={23} /> : <AlertTriangle size={23} />}<span>{heatmapStatus === "loading" ? "Loading saved model heatmap…" : "Could not load saved heatmap"}</span></div>}
        </div>
        <div className="heatmap-caption"><span><b>{entry.asset.filename}</b><small>{entry.event.case_title}</small></span><span className="heatmap-open">Open report <ArrowRight size={13} /></span></div>
        <span className="heatmap-model-note">Actual stored model output · Face crop attribution</span>
      </button>
    </div> : <div className="heatmap-empty"><div className="heatmap-placeholder"><Fingerprint size={39} /><span>MODEL ATTRIBUTION</span></div>
      <b>No heatmap artifact returned</b><p>Heatmaps appear only when the image model detects a face and successfully generates a patch-attribution overlay.</p>
      <span className="heatmap-limit">No synthetic or placeholder heatmaps are displayed. Video attribution is not generated by the current pipeline.</span></div>}
    <p className="insight-footnote">The overlay is stored with the case on the backend and fetched using your signed-in identity. It shows model-influential patches, not proof of manipulation.</p>
  </section>;
}

function AuthScreen({ mode, setMode, email, setEmail, password, setPassword, displayName, setDisplayName, busy, error, onSubmit, onGoogle }) {
  const creating = mode === "signup";
  return <div className="auth-screen">
    <div className="auth-backdrop-glow" />
    <div className="auth-scene auth-scene-left">
      <div className="scene-panel-heading"><b>Evidence Files</b><span>03 ITEMS</span></div>
      <div className="scene-evidence-grid">
        <div className="scene-evidence scene-image"><ImageIcon size={25} /><small>image.jpg</small><i>✓</i></div>
        <div className="scene-evidence scene-video"><Video size={25} /><small>video.mp4</small><i>✓</i></div>
        <div className="scene-evidence scene-audio"><AudioLines size={25} /><small>audio.wav</small><i>✓</i></div>
      </div>
      <div className="scene-panel-foot"><Fingerprint size={15} /> Original media · Evidence preserved</div>
    </div>
    <div className="auth-scene auth-scene-right">
      <div className="scene-scan-card">
        <div className="scan-frame"><Fingerprint size={82} /><span className="scan-corner c1" /><span className="scan-corner c2" /><span className="scan-corner c3" /><span className="scan-corner c4" /></div>
        <b>Evidence analysis</b><div className="scan-progress"><i /></div><small>Multimodal review workspace</small>
      </div>
      <div className="scene-meta-card"><b>Evidence Metadata</b><span>File type <strong>Image / Video / Audio</strong></span><span>Integrity <strong>SHA-256</strong></span><span>Review <strong>Human-led</strong></span></div>
    </div>
    <div className="auth-card">
      <div className="auth-brand"><span className="brand-mark"><Fingerprint size={27} /></span>
        <div><b>Truth<span>Lens</span></b><small>DIGITAL FORENSICS & EVIDENCE REVIEW</small></div></div>
      <div className="auth-heading">
        <h1>{creating ? "Create your account" : "Digital Forensics & Evidence Review"}</h1>
        <p>{creating ? "Create a secure workspace to investigate and review digital evidence." : "Analyze, verify, and report digital evidence with confidence."}</p>
      </div>
      <form onSubmit={onSubmit} className="auth-form">
        {creating && <label><div className="auth-input"><UserRound size={18} /><input autoComplete="name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="Full name" required /></div></label>}
        <label><div className="auth-input"><Mail size={19} /><input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Email address" required /></div></label>
        <label><div className="auth-input"><LockKeyhole size={19} /><input type="password" autoComplete={creating ? "new-password" : "current-password"} value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Password" minLength={6} required /></div></label>
        {error && <div className="auth-error"><AlertTriangle size={15} />{error}</div>}
        <button className="button button-primary auth-submit" type="submit" disabled={busy}>
          {busy ? <><LoaderCircle className="spin" size={16} /> Please wait…</> : <>{creating ? "Create account" : "Sign in"} <ArrowRight size={16} /></>}
        </button>
      </form>
      <div className="auth-divider"><span /> <small>OR</small> <span /></div>
      <button className="button button-quiet auth-google" onClick={onGoogle} disabled={busy}><span className="google-g">G</span> Continue with Google</button>
      <p className="auth-switch">{creating ? "Already have an account?" : "New to TruthLens?"}
        <button onClick={() => setMode(creating ? "signin" : "signup")}>{creating ? "Sign in" : "Create an account"}</button>
      </p>
      <div className="auth-privacy"><ShieldCheck size={15} /> Secure Firebase authentication · Private analyst workspace</div>
    </div>
    <div className="auth-footer">TRUTHLENS <span>·</span> INVESTIGATE. ANALYZE. VERIFY.</div>
  </div>;
}

export default function App() {
  const [user, setUser] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const [authMode, setAuthMode] = useState("signin");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authName, setAuthName] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");
  const [page, setPage] = useState("dashboard");
  const [files, setFiles] = useState([]);
  const [context, setContext] = useState("");
  const [caseTitle, setCaseTitle] = useState("");
  const [pairMedia, setPairMedia] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [activeReport, setActiveReport] = useState(null);
  const [search, setSearch] = useState("");
  const [apiStatus, setApiStatus] = useState("unknown");
  const [apiMessage, setApiMessage] = useState("");
  const [workspaceName, setWorkspaceName] = useState(() => localStorage.getItem("truthlens.workspace-name") || "Forensic workspace");
  const [savedReports, setSavedReports] = useState([]);
  const [activityEvents, setActivityEvents] = useState([]);
  const [activitySummary, setActivitySummary] = useState({ case_count: 0, evidence_count: 0, verified_count: 0, manipulation_signals: 0 });
  const [activityLoading, setActivityLoading] = useState(true);
  const [activityError, setActivityError] = useState("");
  const [optionalIntegrations, setOptionalIntegrations] = useState({});
  const [optionalIntegrationsLoading, setOptionalIntegrationsLoading] = useState(false);
  const [optionalIntegrationsError, setOptionalIntegrationsError] = useState("");
  const [optionalIntegrationsRefresh, setOptionalIntegrationsRefresh] = useState(0);

  useEffect(() => {
    if (page !== "settings" || !user) return undefined;
    let cancelled = false;
    setOptionalIntegrationsLoading(true);
    setOptionalIntegrationsError("");
    (async () => {
      try {
        const token = await user.getIdToken();
        const response = await axios.get(API + "/api/capabilities", {
          timeout: 15000,
          headers: { Authorization: "Bearer " + token }
        });
        if (!cancelled) setOptionalIntegrations(response.data.optional_integration_status || {});
      } catch (err) {
        if (!cancelled) setOptionalIntegrationsError(err.response?.data?.detail || "Could not read optional integration status from the backend.");
      } finally {
        if (!cancelled) setOptionalIntegrationsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [page, user, optionalIntegrationsRefresh]);

  useEffect(() => observeAuth((nextUser) => {
    setUser(nextUser);
    setSavedReports(getReportsForUser(nextUser?.uid));
    setActiveReport(null);
    setPage("dashboard");
    setAuthReady(true);
    setAuthError("");
  }), []);

  useEffect(() => {
    if (!user) return;
    const storageKey = STORAGE_KEY + "." + user.uid;
    try {
      localStorage.setItem(storageKey, JSON.stringify(savedReports));
    } catch {
      const compact = (items) => items.map((report) => ({
        ...report,
        assets: report.assets.map((asset) => ({
          ...asset,
          model_assessment: {
            ...(asset.model_assessment || {}),
            faces: (asset.model_assessment?.faces || []).map((face) => ({
              ...face,
              gradcam_overlay_base64: null
            }))
          }
        }))
      }));
      try {
        localStorage.setItem(storageKey, JSON.stringify(compact(savedReports.slice(0, 10))));
        setNotice("Case history was saved in compact mode to fit browser storage. JSON/PDF exports remain available.");
      } catch {
        try {
          localStorage.setItem(storageKey, JSON.stringify(compact(savedReports.slice(0, 3))));
          setNotice("Browser storage is limited. Only the three most recent compact case reports were retained locally.");
        } catch {
          setNotice("Browser storage is full. Export important cases as JSON or PDF before closing this session.");
        }
      }
    }
  }, [savedReports, user]);
  useEffect(() => {
    localStorage.setItem("truthlens.workspace-name", workspaceName);
  }, [workspaceName]);

  const refreshActivity = useCallback(async () => {
    if (!user) {
      setActivityEvents([]);
      setActivitySummary({ case_count: 0, evidence_count: 0, verified_count: 0, manipulation_signals: 0 });
      setActivityLoading(false);
      return;
    }
    try {
      const token = await user.getIdToken();
      const response = await axios.get(API + "/api/activity", {
        params: { limit: 50 },
        timeout: 12000,
        headers: { Authorization: "Bearer " + token }
      });
      setActivityEvents(response.data.events || []);
      setActivitySummary(response.data.summary || { case_count: 0, evidence_count: 0, verified_count: 0, manipulation_signals: 0 });
      setActivityError("");
      setApiStatus("online");
    } catch (err) {
      if (err.response) {
        setApiStatus("online");
        const detail = err.response.data?.detail || err.message || "The API returned an error.";
        setActivityError(`API responded with HTTP ${err.response.status}: ${detail}`);
        setApiMessage("The backend is reachable; check Firebase Admin authentication or the endpoint response.");
      } else {
        setApiStatus("offline");
        setActivityError(`No response from ${API}. Start the FastAPI backend and verify VITE_API_URL and the browser CORS origin.`);
        setApiMessage("No HTTP response was received from the TruthLens API.");
      }
    } finally {
      setActivityLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (!user) return undefined;
    setActivityLoading(true);
    refreshActivity();
    const refreshTimer = window.setInterval(refreshActivity, 8000);
    return () => window.clearInterval(refreshTimer);
  }, [user, refreshActivity]);

  const totalSize = useMemo(() => files.reduce((sum, item) => sum + item.file.size, 0), [files]);
  const latestReports = useMemo(() => activityEvents.map((event) => ({
    report_id: event.report_id,
    case_title: event.case_title,
    created_at: event.created_at,
    overall_status: event.overall_status,
    assets: (event.evidence || []).map((asset) => ({
      ...asset,
      model_assessment: { status: asset.analysis_status, prediction: asset.prediction },
      video_analysis: { prediction: asset.prediction }
    }))
  })), [activityEvents]);
  const filteredReports = latestReports.filter((item) =>
    (item.case_title || "").toLowerCase().includes(search.toLowerCase()) ||
    item.report_id.toLowerCase().includes(search.toLowerCase()) ||
    item.assets.some((asset) => asset.filename.toLowerCase().includes(search.toLowerCase()))
  );
  const reviewedAssets = activitySummary.evidence_count;
  const verifiedAssetsCount = activitySummary.verified_count;
  const integrityPercent = reviewedAssets ? Math.round((verifiedAssetsCount / reviewedAssets) * 100) : 0;

  function addFiles(selected) {
    const next = [...files];
    const rejected = [];
    for (const file of selected) {
      const type = fileType(file);
      if (type === "unsupported") { rejected.push(file.name + ": unsupported file type"); continue; }
      if (file.size > MAX_BYTES) { rejected.push(file.name + ": exceeds 40 MB"); continue; }
      if (next.some((item) => item.file.name === file.name && item.file.size === file.size)) continue;
      if (next.length >= MAX_FILES) { rejected.push("Maximum of 8 files per investigation"); break; }
      next.push({ id: crypto.randomUUID(), file, type, previewUrl: URL.createObjectURL(file) });
    }
    setFiles(next);
    setActiveReport(null);
    setError(rejected.join(" · "));
    setNotice("");
  }
  function removeFile(id) {
    const item = files.find((file) => file.id === id);
    if (item) URL.revokeObjectURL(item.previewUrl);
    setFiles(files.filter((file) => file.id !== id));
  }
  function clearFiles() {
    files.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    setFiles([]);
    setContext("");
    setCaseTitle("");
    setPairMedia(false);
    setError("");
  }
  async function analyze() {
    if (!files.length || loading) return;
    setLoading(true);
    setUploadProgress(0);
    setError("");
    setNotice("");
    setActiveReport(null);
    const data = new FormData();
    files.forEach((item) => data.append("files", item.file));
    data.append("context_text", context);
    data.append("pair_media", pairMedia ? "true" : "false");
    data.append("case_title", caseTitle.trim());
    try {
      const preUploadDigests = await Promise.all(files.map((item) => sha256Hex(item.file)));
      data.append("sha256_digests", JSON.stringify(preUploadDigests));
      const token = await user.getIdToken();
      const response = await axios.post(API + "/api/analyze", data, {
        headers: { Authorization: "Bearer " + token },
        timeout: 600000,
        onUploadProgress: (event) => {
          if (event.total) setUploadProgress(Math.round((event.loaded / event.total) * 100));
        }
      });
      const report = {
        ...response.data,
        case_title: response.data.case_title || caseTitle.trim() || (files.length === 1 ? "Review · " + files[0].file.name : "Investigation · " + files.length + " evidence files"),
        case_context: context
      };
      setActiveReport(report);
      setSavedReports((current) => [report, ...current.filter((item) => item.report_id !== report.report_id)].slice(0, 50));
      setNotice("Investigation and evidence integrity results saved to the TruthLens backend.");
      await refreshActivity();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }
  async function openReport(report) {
    const reportId = typeof report === "string" ? report : report.report_id;
    setPage("reports");
    setNotice("");
    try {
      const token = await user.getIdToken();
      const response = await axios.get(API + "/api/reports/" + encodeURIComponent(reportId), {
        timeout: 30000,
        headers: { Authorization: "Bearer " + token }
      });
      setActiveReport(response.data);
      setError("");
    } catch (err) {
      const cached = savedReports.find((item) => item.report_id === reportId);
      if (cached) setActiveReport(cached);
      setError(err.response?.data?.detail || "Could not load the saved server report.");
    }
  }
  async function removeSavedReport(id) {
    try {
      const token = await user.getIdToken();
      await axios.delete(API + "/api/reports/" + encodeURIComponent(id), {
        headers: { Authorization: "Bearer " + token }
      });
      setSavedReports((current) => current.filter((item) => item.report_id !== id));
      if (activeReport?.report_id === id) setActiveReport(null);
      await refreshActivity();
    } catch (err) {
      setError(err.response?.data?.detail || "Could not delete the saved server report.");
    }
  }
  async function checkApi() {
    setApiStatus("checking");
    setApiMessage("Checking backend health…");
    try {
      const health = await axios.get(API + "/health", { timeout: 8000 });
      if (health.data?.status !== "healthy") {
        throw new Error("The API responded, but its health endpoint did not report healthy.");
      }
    } catch (err) {
      setApiStatus("offline");
      setApiMessage(!err.response
        ? `No response from ${API}. Start the backend in its own terminal and check VITE_API_URL.`
        : `Backend health check failed (${err.response.status}): ${err.response.data?.detail || err.message}`);
      return;
    }
    try {
      const token = await user.getIdToken();
      await axios.get(API + "/api/me", {
        timeout: 8000,
        headers: { Authorization: "Bearer " + token }
      });
      setApiStatus("online");
      setApiMessage("Backend is healthy and Firebase identity was verified.");
    } catch (err) {
      if (err.response) {
        setApiStatus("online");
        setApiMessage(`Backend is reachable, but authentication/API configuration returned HTTP ${err.response.status}: ${err.response.data?.detail || err.message}`);
      } else {
        setApiStatus("offline");
        setApiMessage(`Backend health responded, but the authenticated request received no response. Check CORS for ${API} and refresh your Firebase sign-in.`);
      }
    }
  }
  function newInvestigation() {
    setPage("new");
    setActiveReport(null);
    setNotice("");
  }

  async function submitAuth(event) {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      if (authMode === "signup") {
        if (authPassword.length < 6) throw new Error("Password must contain at least 6 characters.");
        await createEmailAccount(authName, authEmail, authPassword);
      } else {
        await signInEmail(authEmail, authPassword);
      }
      setAuthPassword("");
    } catch (error) {
      setAuthError(authErrorMessage(error));
    } finally {
      setAuthBusy(false);
    }
  }
  async function continueWithGoogle() {
    setAuthBusy(true);
    setAuthError("");
    try { await signInGoogle(); }
    catch (error) { setAuthError(authErrorMessage(error)); }
    finally { setAuthBusy(false); }
  }
  async function logout() {
    try { await signOutUser(); }
    catch (error) { setNotice(authErrorMessage(error)); }
  }

  const nav = [
    ["dashboard", "Home", <LayoutDashboard size={18} />],
    ["new", "New Investigation", <Plus size={18} />],
    ["history", "Case History", <FolderOpen size={18} />],
    ["analysis", "Media Analysis", <Activity size={18} />],
    ["reports", "Reports", <FileBarChart2 size={18} />],
    ["help", "Help & Support", <Globe size={18} />],
    ["settings", "Settings", <Settings size={18} />]
  ];

  if (!authReady) return <div className="auth-loading"><LoaderCircle className="spin" size={24} /><span>Securing your TruthLens workspace…</span></div>;
  if (!user) return <AuthScreen
    mode={authMode} setMode={(value) => { setAuthMode(value); setAuthError(""); }}
    email={authEmail} setEmail={setAuthEmail}
    password={authPassword} setPassword={setAuthPassword}
    displayName={authName} setDisplayName={setAuthName}
    busy={authBusy} error={!firebaseConfigured ? "Firebase is not configured. Add the client values to frontend/.env.local, then restart the frontend." : authError}
    onSubmit={submitAuth} onGoogle={continueWithGoogle}
  />;

  return <div className="app-shell">
    <aside className="sidebar">
      <button className="brand" onClick={() => setPage("dashboard")}>
        <span className="brand-mark"><BrandGlyph size={35} /></span>
        <span><strong>TRUTHLENS</strong><small>FORENSIC INTELLIGENCE</small></span>
      </button>
      <div className="workspace-switch"><span className="workspace-avatar">TL</span><span><small>WORKSPACE</small><b>{workspaceName}</b></span><ChevronRight size={15} /></div>
      <div className="side-label">INVESTIGATE</div>
      <nav className="nav-list">{nav.map(([id, label, icon]) =>
        <button key={id} className={"nav-item " + (page === id ? "active" : "")} onClick={() => { setPage(id); setNotice(""); }}>
          {icon}<span>{label}</span>
          {id === "history" && activitySummary.case_count > 0 && <em>{activitySummary.case_count}</em>}
        </button>)}</nav>
      <div className="sidebar-bottom">
        <div className="profile-avatar">{(user.displayName || user.email || "FA").slice(0, 2).toUpperCase()}</div><div className="profile-copy"><b>{user.displayName || "Forensic analyst"}</b><small>{user.email || "Signed in"}</small></div>
        <button className="profile-settings" onClick={() => setPage("settings")} aria-label="Profile settings"><Settings size={15} /></button>
        <button className="profile-settings" onClick={logout} aria-label="Sign out" title="Sign out"><LogOut size={15} /></button>
      </div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div className="breadcrumb"><span>Workspace</span><ChevronRight size={13} /><b>{nav.find((item) => item[0] === page)?.[1] || "Home"}</b></div>
        <div className="topbar-actions">
          <label className="global-search"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") setPage("history"); }} placeholder="Search investigations, reports…" /></label>
          <span className={"api-indicator " + apiStatus}><i /> {apiStatus === "online" ? "API ONLINE" : apiStatus === "offline" ? "API OFFLINE" : "WORKSPACE"}</span>
          <button className="top-icon" onClick={checkApi} title="Check backend connection"><Activity size={17} /></button>
          <button className="top-icon" onClick={() => setPage("settings")} title="Settings"><Bell size={17} /></button>
          <button className="top-user" onClick={() => setPage("settings")}><span className="top-avatar">{(user.displayName || user.email || "FA").slice(0, 2).toUpperCase()}</span><span>{user.displayName || "Investigator"}</span><ChevronRight size={14} /></button>
        </div>
      </header>

      {page === "dashboard" && <section className="dashboard-page">
        <div className="dashboard-hero">
          <div className="hero-copy"><span className="section-kicker">DIGITAL FORENSICS & DEEPFAKE DETECTION</span>
            <h1>Truth<span>Lens</span></h1>
            <h2>Investigate. Analyze. <em>Verify.</em></h2>
            <p>A multimodal AI-powered forensics platform to detect deepfakes, analyze digital evidence, and generate detailed review reports.</p>
            <div className="hero-actions">
              <button className="button button-primary" onClick={newInvestigation}><Plus size={18} /> Start New Investigation</button>
              <button className="button button-quiet" onClick={() => setPage("history")}><FolderOpen size={17} /> View Case History</button>
            </div>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="hero-monitor">
              <div className="monitor-top"><i /><i /><i /><span>TRUTHLENS / LIVE ANALYSIS</span></div>
              <div className="monitor-content"><div className="hero-face"><div className="face-scan-line" /><div className="face-outline" /></div>
                <div className="monitor-metrics"><b>MEDIA ANALYSIS</b><span className="signal-pill">● REVIEW SIGNALS</span><div className="waveform">{Array.from({length:16},(_,i)=><i key={i} />)}</div><small>IMAGE · VIDEO · AUDIO</small></div>
              </div>
            </div>
            <div className="hero-fingerprint"><Fingerprint size={38} /></div>
          </div>
        </div>

        <label className="quick-upload">
          <input type="file" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.mp4,.mov,.avi,.mkv,.webm,.wav,.flac,.ogg"
            onChange={(event) => { addFiles([...event.target.files]); event.target.value = ""; setPage("new"); }} />
          <span className="quick-upload-main"><UploadCloud size={30} /><b>Upload Evidence</b><small>Drag, drop or click to select files</small></span>
          <span className="quick-upload-types">
            <span><i className="type-image"><ImageIcon size={23} /></i><b>Image</b></span>
            <span><i className="type-video"><Video size={23} /></i><b>Video</b></span>
            <span><i className="type-audio"><AudioLines size={23} /></i><b>Audio</b></span>
            <span><i className="type-document"><FileBarChart2 size={23} /></i><b>Reports</b></span>
          </span>
        </label>

        <div className="stats-grid dashboard-stats">
          <div className="stat-card"><span><FolderOpen size={17} /> TOTAL CASES</span><b>{activitySummary.case_count}</b><small>Server records · live refresh</small></div>
          <div className="stat-card"><span><ShieldAlert size={17} /> MANIPULATION SIGNALS</span><b>{activitySummary.manipulation_signals}</b><small>Signals requiring analyst review</small></div>
          <div className="stat-card"><span><FileBarChart2 size={17} /> REPORTS GENERATED</span><b>{activitySummary.case_count}</b><small>Reports persisted by backend</small></div>
          <div className="stat-card"><span><FileImage size={17} /> MEDIA ANALYZED</span><b>{reviewedAssets}</b><small>Evidence items across cases</small></div>
        </div>

        <div className="dashboard-insights-grid">
          <IntegrityPanel events={activityEvents} loading={activityLoading} error={activityError} />
          <ActivityTimeline events={activityEvents} onOpen={openReport} loading={activityLoading} error={activityError} />
          <HeatmapPreview events={activityEvents} user={user} onOpen={openReport} />
        </div>
        <div className="dashboard-lower">
          <div className="content-panel recent-panel">
            <div className="panel-title"><div><span className="section-kicker">SERVER CASE REGISTER</span><h2>Recent investigations</h2></div>
              <button className="text-button" onClick={() => setPage("history")}>View all <ArrowRight size={14} /></button></div>
            {latestReports.length ? <div className="recent-case-table">
              <div className="recent-case-head"><span>Case ID / Title</span><span>Evidence</span><span>Updated</span><span>Status</span></div>
              {latestReports.slice(0, 4).map((report) => <CaseRow key={report.report_id} report={report} onOpen={() => openReport(report)} />)}
            </div> : <div className="empty-state"><div className="empty-icon"><FolderOpen size={22} /></div><b>No investigations yet</b><p>Your completed cases will appear here after the first analysis.</p><button className="button button-quiet" onClick={newInvestigation}><Plus size={15} /> Create first case</button></div>}
          </div>
          <div className="content-panel analysis-summary">
            <div className="panel-title"><div><span className="section-kicker">AI ANALYSIS SUMMARY</span><h2>Multimodal review</h2></div><ShieldCheck size={20} /></div>
            <div className="analysis-summary-body"><div className="analysis-ring" style={{ background: "conic-gradient(#0edbd0 0 " + integrityPercent + "%, #123c48 " + integrityPercent + "% 100%)" }}><div><b>{integrityPercent}%</b><small>Hash match rate</small></div></div>
              <div className="analysis-checks">
                <span><CheckCircle2 size={16} /> Face-region analysis</span>
                <span><CheckCircle2 size={16} /> Video frame sampling</span>
                <span><CheckCircle2 size={16} /> Audio speech analysis</span>
                <span><CheckCircle2 size={16} /> Metadata & SHA-256</span>
              </div>
            </div>
            <p className="summary-disclaimer">Model outputs are investigative signals, not proof of authenticity. Human review remains essential.</p>
          </div>
        </div>
      </section>}

      {(page === "new" || page === "analysis") && <section className="investigation-page">
        <div className="page-heading"><div><span className="section-kicker">CASE WORKSPACE / INGEST</span><h1>{page === "analysis" ? "Media analysis" : "New investigation"}</h1><p>Capture live photo, video or audio, or upload original evidence files for multimodal review.</p></div>
          <div className="secure-badge"><ShieldCheck size={15} /> Evidence-first workflow</div></div>
        <Pipeline loading={loading} uploadProgress={uploadProgress} report={activeReport} />
        {loading && <div className="progress-panel"><div><b>{uploadProgress >= 100 ? "Evidence received · model analysis running" : "Uploading evidence"}</b><span>{uploadProgress}% transferred</span></div><div className="progress-track"><i style={{ width: uploadProgress + "%" }} /></div><small>Upload transfer is measured by the browser. Model-stage telemetry is not currently streamed by the API.</small></div>}
        <div className="ingest-layout">
          <div className="content-panel ingest-panel">
            <div className="panel-title"><div><span className="section-kicker">01 / EVIDENCE INTAKE</span><h2>Evidence files</h2></div>
              <span className="file-count">{files.length} / {MAX_FILES} files</span></div>
            <LiveCapture onCapture={addFiles} />
            <div className="intake-divider"><span>OR UPLOAD EXISTING EVIDENCE</span></div>
            <label className="dropzone" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); addFiles([...event.dataTransfer.files]); }}>
              <input type="file" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.mp4,.mov,.avi,.mkv,.webm,.wav,.flac,.ogg"
                onChange={(event) => { addFiles([...event.target.files]); event.target.value = ""; }} />
              <span className="upload-symbol"><UploadCloud size={27} /></span><b>Drag evidence files into this area</b>
              <span>or <u>browse from device</u></span><small>Images · Videos · Speech audio · Up to 40 MB per file</small>
            </label>
            {files.length > 0 && <div className="evidence-upload-list">
              <div className="upload-list-heading"><b>Selected evidence</b><div><span>{files.length} files · {formatBytes(totalSize)}</span><button className="text-button danger-text" onClick={clearFiles}><Trash2 size={13} /> Clear all</button></div></div>
              <div className="preview-grid">{files.map((item) => <div className="preview-card" key={item.id}>
                <div className="preview-media">
                  {item.type === "image" ? <img src={item.previewUrl} alt={item.file.name} />
                    : item.type === "video" ? <video src={item.previewUrl} controls preload="metadata" />
                    : <div className="audio-preview"><span><Headphones size={25} /></span><b>Audio evidence</b><small>{item.file.name}</small><audio src={item.previewUrl} controls /></div>}
                  <button className="remove-preview" onClick={() => removeFile(item.id)} aria-label={"Remove " + item.file.name}><X size={14} /></button>
                </div>
                <div className="preview-meta"><b title={item.file.name}>{item.file.name}</b><span>{item.type.toUpperCase()} · {formatBytes(item.file.size)}</span></div>
              </div>)}</div>
              <div className="upload-total"><span>Total evidence payload</span><b>{formatBytes(totalSize)}</b></div>
            </div>}
            <div className="field-row"><label htmlFor="caseTitle">CASE NAME <span>OPTIONAL</span></label><input id="caseTitle" value={caseTitle} onChange={(event) => setCaseTitle(event.target.value)} placeholder="e.g. Social media clip review – Case 014" /></div>
            <div className="field-row"><label htmlFor="context">INVESTIGATION NOTES <span>OPTIONAL</span></label>
              <textarea id="context" value={context} onChange={(event) => setContext(event.target.value)} placeholder="Source details, claimed context, known timeline, transcript or analyst notes…" rows={4} /></div>
            <label className="pair-toggle"><input type="checkbox" checked={pairMedia} onChange={(event) => setPairMedia(event.target.checked)} />
              <span><b>These separate audio and video files belong to the same recording</b><small>Enables comparison of their independent model signals. It does not verify synchronization or provenance.</small></span></label>
            <div className="ingest-bottom"><div className="validation-copy"><Shield size={15} /> Supported formats only · 8 files maximum · 40 MB each</div>
              <button className="button button-primary analyze-button" disabled={!files.length || loading} onClick={analyze}>
                {loading ? <><LoaderCircle className="spin" size={17} /> {uploadProgress >= 100 ? "Models analysing evidence…" : "Uploading evidence…"}</> : <><Activity size={17} /> Initiate investigation <ArrowRight size={16} /></>}
              </button></div>
            {error && <div className="error-box"><AlertTriangle size={17} />{String(error)}</div>}
            {notice && <div className="success-box"><CheckCircle2 size={16} />{notice}</div>}
          </div>
          <aside className="content-panel intake-aside"><span className="section-kicker">ANALYSIS SCOPE</span><h2>What gets examined</h2>
            <div className="scope-item"><span className="scope-icon cyan"><ImageIcon size={18} /></span><div><b>Images</b><small>Face crops, class scores and optional patch attribution</small></div></div>
            <div className="scope-item"><span className="scope-icon purple"><Video size={18} /></span><div><b>Video</b><small>Up to eight sampled frames with timestamps</small></div></div>
            <div className="scope-item"><span className="scope-icon green"><Mic size={18} /></span><div><b>Speech audio</b><small>Speech windows and independent model scores</small></div></div>
            <div className="scope-item"><span className="scope-icon amber"><Fingerprint size={18} /></span><div><b>Evidence integrity</b><small>SHA-256 digest and media metadata</small></div></div>
            <div className="notice"><AlertTriangle size={16} /><span>Some files may be inconclusive when there is no detectable face, insufficient speech, decoding errors or unsupported model scope.</span></div>
          </aside>
        </div>
        {activeReport && <ReportView report={activeReport} user={user} onReviewsChange={(reviews) => setActiveReport((current) => current?.report_id === activeReport.report_id ? { ...current, investigator_reviews: reviews } : current)} onJson={() => downloadJson(activeReport)} onPdf={() => {
          if (!exportPdf(activeReport)) setError("Allow pop-ups to open the printable PDF report.");
        }} />}
      </section>}

      {(page === "history" || page === "reports") && <section className="records-page">
        <div className="page-heading"><div><span className="section-kicker">{page === "history" ? "WORKSPACE / CASE REGISTER" : "WORKSPACE / REPORT LIBRARY"}</span>
          <h1>{page === "history" ? "Case history" : "Reports"}</h1><p>{page === "history" ? "Reopen saved investigations and continue reviewing the evidence." : "Open completed case reports or export an evidence summary."}</p></div>
          <button className="button button-primary" onClick={newInvestigation}><Plus size={16} /> New investigation</button></div>
        {activeReport && page === "reports" && <ReportView report={activeReport} user={user} onReviewsChange={(reviews) => setActiveReport((current) => current?.report_id === activeReport.report_id ? { ...current, investigator_reviews: reviews } : current)} onJson={() => downloadJson(activeReport)} onPdf={() => {
          if (!exportPdf(activeReport)) setError("Allow pop-ups to open the printable PDF report.");
        }} />}
        <div className="content-panel record-list-panel">
          <div className="record-toolbar"><div><span className="section-kicker">{page === "history" ? "SAVED INVESTIGATIONS" : "AVAILABLE REPORTS"}</span><h2>{filteredReports.length} records</h2></div>
            <div className="search-box"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search case, ID or filename…" /></div></div>
          {filteredReports.length ? filteredReports.map((report) => <div className="record-line" key={report.report_id}>
            <CaseRow report={report} onOpen={() => openReport(report)} />
            <button className="delete-record" onClick={() => removeSavedReport(report.report_id)} title="Delete server report"><Trash2 size={15} /></button>
          </div>) : <div className="empty-state"><div className="empty-icon"><FileBarChart2 size={22} /></div><b>{activityLoading ? "Loading server cases…" : activityError ? "Server case history unavailable" : "No saved reports found"}</b><p>{activityError || "Run an investigation to create a case record. Saved reports are stored by the TruthLens backend and synced to your account."}</p></div>}
        </div>
      </section>}

      {page === "help" && <section className="settings-page help-page">
        <div className="page-heading"><div><span className="section-kicker">TRUTHLENS / SUPPORT</span><h1>Help & Support</h1><p>Quick guidance for running an evidence review.</p></div>
          <button className="button button-primary" onClick={newInvestigation}><Plus size={16} /> Start an investigation</button></div>
        <div className="settings-grid">
          <div className="content-panel settings-card"><span className="section-kicker">01 / GETTING STARTED</span><h2>Upload your evidence</h2><p>Add supported image, video, or speech-audio files. You can attach up to eight files, with a 40 MB limit per file.</p></div>
          <div className="content-panel settings-card"><span className="section-kicker">02 / REVIEW FINDINGS</span><h2>Interpret model signals</h2><p>Review modality outputs, metadata, SHA-256 fingerprints and available attribution or sampled-frame details. A model score is not a calibrated probability.</p></div>
          <div className="content-panel settings-card"><span className="section-kicker">03 / SAVE YOUR WORK</span><h2>Case history & reports</h2><p>Completed reports are stored by the backend under your Firebase account. Export a JSON or print-ready PDF copy.</p></div>
          <div className="content-panel settings-card"><span className="section-kicker">04 / CONNECTION</span><h2>Backend status</h2><p>Use the connection control in the top bar or Settings to test the TruthLens API. The FastAPI backend must be running for analysis requests.</p><button className="button button-quiet" onClick={checkApi}><Activity size={15} /> Test API connection</button></div>
        </div>
      </section>}

      {page === "settings" && <section className="settings-page">
        <div className="page-heading"><div><span className="section-kicker">WORKSPACE / CONFIGURATION</span><h1>Settings</h1><p>Manage this local workspace and verify its API connection.</p></div></div>
        <div className="settings-grid">
          <div className="content-panel settings-card"><span className="section-kicker">WORKSPACE PROFILE</span><h2>Workspace identity</h2><label className="field-row"><span>Workspace name</span><input value={workspaceName} onChange={(event) => setWorkspaceName(event.target.value)} /></label>
            <div className="profile-summary"><span className="profile-avatar large">{(user.displayName || user.email || "FA").slice(0, 2).toUpperCase()}</span><div><b>{user.displayName || "Forensic analyst"}</b><small>{user.email || "Signed-in account"}</small></div></div>
            <button className="button button-quiet" onClick={logout}><LogOut size={15} /> Sign out</button></div>
          <div className="content-panel settings-card"><span className="section-kicker">BACKEND CONNECTION</span><h2>TruthLens API</h2><div className="api-url">{API}</div>
            <button className="button button-primary" onClick={checkApi}><Activity size={16} /> Test API connection</button>
            <div className={"connection-state " + apiStatus}><i /> {apiStatus === "online" ? "Backend reachable" : apiStatus === "offline" ? "Backend unreachable" : apiStatus === "checking" ? "Checking connection…" : "Connection not checked"}</div>
            {apiMessage && <p className="settings-help">{apiMessage}</p>}</div>
          <div className="content-panel settings-card optional-integrations-card">
            <div className="optional-integrations-heading"><div><span className="section-kicker">ADVANCED MODULES</span><h2>Optional integration status</h2></div>
              <button className="button button-quiet" onClick={() => setOptionalIntegrationsRefresh((value) => value + 1)} disabled={optionalIntegrationsLoading}>
                <Activity size={14} /> {optionalIntegrationsLoading ? "Checking…" : "Refresh"}
              </button></div>
            <p className="settings-help">These checks run on the backend machine. A ready status confirms detected prerequisites only; it does not guarantee model inference or validate model accuracy.</p>
            {optionalIntegrationsLoading && <p className="settings-help">Reading backend configuration…</p>}
            {optionalIntegrationsError && <p className="settings-help">{optionalIntegrationsError}</p>}
            {Object.entries(optionalIntegrations).length > 0 && <div className="integration-status-list">
              {Object.entries(optionalIntegrations).map(([key, item]) => <div className="integration-status-item" key={key}>
                <div className="integration-status-top"><b>{key.replaceAll("_", " ")}</b><span className="integration-status-tag">{String(item.status || "unknown").replaceAll("_", " ")}</span></div>
                <small>{item.note}</small>
                {item.missing?.length > 0 && <small className="integration-missing">Needs: {item.missing.join(", ")}</small>}
              </div>)}
            </div>}
            {!optionalIntegrationsLoading && !optionalIntegrationsError && !Object.keys(optionalIntegrations).length && <p className="settings-help">No integration status received yet. Check that the updated backend is running.</p>}
            <p className="settings-help">Configure optional modules in backend/.env, restart FastAPI, then refresh this panel. See docs/ADVANCED_FORENSICS.md.</p>
          </div>
          <div className="content-panel settings-card"><span className="section-kicker">DATA & PRIVACY</span><h2>Server-side case storage</h2>
            <p>Completed reports and real model-attribution PNGs are saved by the TruthLens backend and scoped to your verified Firebase account. The dashboard refreshes its activity feed every 8 seconds. Original uploaded media is processed from temporary files and is not retained as a case attachment.</p>
            <div className="settings-metric"><b>{activityEvents.length}</b><span>Reports stored by the API for this account</span></div>
            <p className="settings-help">Backend storage: backend/data/truthlens.sqlite3 and backend/data/heatmaps/. Protect and back up this directory in deployment.</p></div>
          <div className="content-panel settings-card"><span className="section-kicker">MODEL TRANSPARENCY</span><h2>Review before concluding</h2>
            <p>Model scores are not calibrated probabilities. Video uses sampled frame classification, audio is scoped to the selected speech model and cross-modal review compares separate model signals only.</p>
            <div className="safety-note"><Shield size={15} /> TruthLens supports investigation; it does not certify authenticity.</div></div>
        </div>
      </section>}

      <footer className="app-footer"><span>TRUTHLENS</span><i /> Evidence context matters. Automated model signals are not final conclusions.</footer>
    </main>
  </div>;
}
