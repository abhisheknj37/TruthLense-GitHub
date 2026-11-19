import { useEffect, useMemo, useState } from "react";
import axios from "axios";
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

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
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
function exportPdf(report) {
  const rows = report.assets.map((asset) => {
    const model = asset.model_assessment || {};
    const meta = Object.entries(asset.metadata || {}).filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
      .map(([key, value]) => "<tr><th>" + escapeHtml(key.replaceAll("_", " ")) + "</th><td>" + escapeHtml(value) + "</td></tr>").join("");
    return "<section class='asset'><h2>" + escapeHtml(asset.filename) + "</h2><p>" +
      escapeHtml(String(asset.media_type).toUpperCase()) + " · " + escapeHtml(formatBytes(asset.size_bytes)) +
      "</p><p><b>Model signal:</b> " + escapeHtml(predictionLabel(model.prediction)) +
      "</p><p><b>Model:</b> " + escapeHtml(model.model_name || "Not available") +
      "</p><p><b>SHA-256:</b><br><code>" + escapeHtml(asset.sha256) +
      "</code></p><table>" + meta + "</table></section>";
  }).join("");
  const limitations = (report.limitations || []).map((item) => "<li>" + escapeHtml(item) + "</li>").join("");
  const html = "<!doctype html><html><head><title>TruthLens case report</title><style>" +
    "body{font:14px Arial;color:#172033;margin:36px}h1{margin-bottom:4px}h2{font-size:17px}p{line-height:1.5}" +
    ".muted{color:#62718a}.asset{border:1px solid #cbd5e1;border-radius:8px;padding:16px;margin:18px 0;break-inside:avoid}" +
    "code{font-size:10px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px;border-top:1px solid #e2e8f0}" +
    "th{text-transform:capitalize;width:35%}li{margin:7px 0}@media print{button{display:none}}</style></head><body>" +
    "<h1>TruthLens · Forensic case report</h1><p class='muted'>Case " + escapeHtml(report.report_id) +
    " · " + escapeHtml(formatDate(report.created_at)) + "</p><h2>Case summary</h2><p>" +
    escapeHtml(report.summary) + "</p><p><b>Status:</b> " + escapeHtml(report.overall_status) +
    " · <b>Human review:</b> " + (report.human_review_required ? "Required" : "Not required") +
    "</p><h2>Evidence inventory and findings</h2>" + rows +
    "<h2>Cross-modal review</h2>" + (report.cross_modal_review || []).map((item) =>
      "<p><b>" + escapeHtml(item.type || "Review") + " · " + escapeHtml(item.status) +
      ":</b> " + escapeHtml(item.detail) + "</p>").join("") +
    "<h2>Limitations and review notes</h2><ul>" + limitations + "</ul>" +
    "<p class='muted'>Model scores are not calibrated probabilities or proof of authenticity. Human review and original-source context are required.</p>" +
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

function Pipeline({ loading, uploadProgress, report }) {
  const steps = [
    ["Media inspection", "File structure, metadata and SHA-256"],
    ["Modality analysis", "Image, sampled video frames or speech windows"],
    ["Evidence correlation", "Compare paired media model signals"],
    ["Human review & reporting", "Review findings, limitations and export"]
  ];
  const stateFor = (index) => {
    if (report) return index < 3 ? "complete" : "review";
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
        <span className="state-label">{state === "complete" ? "Completed" : state === "active" ? "In progress" : state === "review" ? "Awaiting analyst" : "Queued"}</span>
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
      {Object.entries(details.scores).filter(([key, value]) => typeof value === "number" && (key.includes("fake") || key.includes("real")))
        .map(([key, value]) => <ScoreBar key={key} label={key} value={value} />)}
    </div>}
    {details.note && <p className="model-note">{details.note}</p>}
    {asset.media_type === "image" && model.faces?.length > 0 && <div className="face-results">
      <b>Face-region analysis</b>
      {model.faces.map((face) => <div className="face-result" key={face.face_index}>
        <div className="face-meta"><span>Face {face.face_index + 1} · {predictionLabel(face.prediction)}</span>
          <small>Region: {face.bounding_box_xywh.join(", ")}</small></div>
        {face.gradcam_overlay_base64
          ? <img className="gradcam-image" src={"data:image/png;base64," + face.gradcam_overlay_base64} alt="Transformer patch attribution overlay" />
          : <p className="model-note">Patch attribution was not available for this face.</p>}
        <p className="cam-caption">Patch attribution highlights regions that influenced this model. It is not a manipulation mask or proof.</p>
      </div>)}
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
    <ModelAssessment asset={asset} />
    {(asset.observations || []).map((item) => <div className="observation" key={item.code}>
      <CheckCircle2 size={16} /><div><b>{item.title}</b><p>{item.detail}</p></div>
    </div>)}
  </article>;
}

function ReportView({ report, onJson, onPdf }) {
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
    <div className="report-stats">
      <div className="report-stat"><span>FILES ANALYSED</span><b>{report.assets.length}</b><small>Evidence items received</small></div>
      <div className="report-stat"><span>MODEL ASSESSMENTS</span><b>{assessed}</b><small>Modality signals available</small></div>
      <div className="report-stat"><span>INCONCLUSIVE</span><b>{inconclusive}</b><small>Require additional context</small></div>
      <div className="report-stat"><span>REVIEW FINDINGS</span><b>{findings}</b><small>Technical observations flagged</small></div>
    </div>
    <div className="report-summary">
      <div className="summary-icon"><ShieldAlert size={20} /></div>
      <div><span className="section-kicker">INVESTIGATION SUMMARY</span><p>{report.summary}</p>
        <small>Case ID: {report.report_id} · {report.human_review_required ? "Human review required" : "Review status not specified"}</small>
      </div>
    </div>
    <div className="evidence-heading"><div><span className="section-kicker">EVIDENCE REGISTER</span><h3>Asset-by-asset findings</h3></div><span>{report.assets.length} evidence items</span></div>
    <div className="asset-grid">{report.assets.map((asset) => <EvidenceCard key={asset.asset_id} asset={asset} />)}</div>
    {report.cross_modal_review?.length > 0 && <div className="cross-review">
      <div className="cross-header"><div><span className="section-kicker">RELATIONSHIP REVIEW</span><h3>Cross-modal comparison</h3></div><span className="model-pill pending">No fused authenticity score</span></div>
      {report.cross_modal_review.map((item, index) => <div className="cross-item" key={index}>
        <span className="model-pill pending">{String(item.status).replaceAll("_", " ")}</span>
        <p>{item.detail}</p>
      </div>)}
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

function IntegrityPanel({ reports }) {
  const [copiedId, setCopiedId] = useState("");
  const assets = reports.flatMap((report) => report.assets.map((asset) => ({
    ...asset, caseTitle: report.case_title || "Forensic investigation"
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
    <div className="insight-panel-title"><div><span className="section-kicker">EVIDENCE INTEGRITY</span><h2>SHA-256 fingerprints</h2></div><span className="insight-icon"><Fingerprint size={19} /></span></div>
    <div className="integrity-summary"><b>{matched} / {assets.length}</b><span>browser-to-server hash matches in recent evidence</span></div>
    {assets.length ? <div className="integrity-list">{assets.map((asset) => {
      const state = asset.integrity_verification?.status;
      const label = state === "match" ? "Verified match" : state === "mismatch" ? "Mismatch" : state === "unavailable" ? "Not rechecked" : "Fingerprint recorded";
      return <div className="integrity-item" key={asset.asset_id}>
        <div className="integrity-item-top"><span>{typeIcon(asset.media_type, 14)}<b title={asset.filename}>{asset.filename}</b></span><span className={"integrity-status " + (state || "recorded")}>{label}</span></div>
        <div className="integrity-digest"><code title={asset.sha256}>{asset.sha256 || "SHA-256 unavailable"}</code><button onClick={() => copyHash(asset)} disabled={!asset.sha256} title="Copy SHA-256 digest"><Copy size={13} />{copiedId === asset.asset_id ? "Copied" : copiedId === "unavailable" ? "Blocked" : "Copy"}</button></div>
        <small className="integrity-case">{asset.caseTitle} · {asset.integrity_verification?.checked_at ? "Checked " + formatDate(asset.integrity_verification.checked_at) : "Server fingerprint saved"}</small>
      </div>;
    })}</div> : <div className="insight-empty"><Fingerprint size={24} /><b>No evidence fingerprints yet</b><span>Run an investigation to create SHA-256 evidence fingerprints.</span></div>}
    <p className="insight-footnote">A match confirms that the uploaded bytes produced the same digest in the browser and API. It does not prove who created the file or that its contents are authentic.</p>
  </section>;
}

function ActivityTimeline({ reports, onOpen }) {
  return <section className="content-panel activity-panel">
    <div className="insight-panel-title"><div><span className="section-kicker">WORKSPACE LOG</span><h2>Recent activity</h2></div><span className="insight-icon"><Clock3 size={18} /></span></div>
    {reports.length ? <div className="activity-timeline">{reports.slice(0, 5).map((report) =>
      <button className="activity-event" key={report.report_id} onClick={() => onOpen(report)}>
        <span className="activity-dot"><CheckCircle2 size={13} /></span><span className="activity-event-copy"><b>Investigation saved</b>
          <small>{report.case_title || "Forensic investigation"}</small>
          <small>{report.assets.length} evidence item{report.assets.length === 1 ? "" : "s"} · {report.assets.filter((asset) => asset.sha256).length} fingerprints recorded</small></span>
        <time>{formatDate(report.created_at)}</time>
      </button>
    )}</div> : <div className="insight-empty"><Clock3 size={24} /><b>No recent activity</b><span>Completed investigations will appear here.</span></div>}
    <p className="insight-footnote">Events are based on saved investigation reports in this browser. This is not a server-side audit log.</p>
  </section>;
}

function HeatmapPreview({ reports, onOpen }) {
  const entry = reports.flatMap((report) => report.assets.map((asset) => ({ report, asset })))
    .find(({ asset }) => asset.media_type === "image" && asset.model_assessment?.faces?.some((face) => face.gradcam_overlay_base64));
  const face = entry?.asset.model_assessment.faces.find((item) => item.gradcam_overlay_base64);
  return <section className="content-panel heatmap-panel">
    <div className="insight-panel-title"><div><span className="section-kicker">MODEL EXPLAINABILITY</span><h2>Heatmap preview</h2></div><span className="insight-icon"><Activity size={18} /></span></div>
    {entry && face ? <button className="heatmap-preview-button" onClick={() => onOpen(entry.report)} title="Open the full case report">
      <div className="heatmap-image-wrap"><img src={"data:image/png;base64," + face.gradcam_overlay_base64} alt="Grad-CAM-style transformer patch attribution overlay for analyzed image evidence" /></div>
      <div className="heatmap-caption"><span><b>{entry.asset.filename}</b><small>{entry.report.case_title || "Forensic investigation"}</small></span><span className="heatmap-open">Open report <ArrowRight size={13} /></span></div>
      <span className="heatmap-model-note">Patch attribution · Face crop model</span>
    </button> : <div className="heatmap-empty"><div className="heatmap-placeholder"><Fingerprint size={39} /><span>ATTRIBUTION PREVIEW</span></div>
      <b>No heatmap available yet</b><p>Upload an image with a detectable face. The preview appears when the image model successfully produces patch attribution.</p>
      <span className="heatmap-limit">Not every image returns a heatmap. Video frame heatmaps are not currently generated.</span></div>}
    <p className="insight-footnote">The overlay highlights image patches that influenced the model output. It is not a manipulation mask or proof of tampering.</p>
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

  const totalSize = useMemo(() => files.reduce((sum, item) => sum + item.file.size, 0), [files]);
  const latestReports = useMemo(() => [...savedReports].sort((a, b) => new Date(b.created_at) - new Date(a.created_at)), [savedReports]);
  const filteredReports = latestReports.filter((item) =>
    (item.case_title || "").toLowerCase().includes(search.toLowerCase()) ||
    item.report_id.toLowerCase().includes(search.toLowerCase()) ||
    item.assets.some((asset) => asset.filename.toLowerCase().includes(search.toLowerCase()))
  );
  const reviewedAssets = savedReports.reduce((sum, item) => sum + item.assets.length, 0);

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
    try {
      const token = await user.getIdToken();
      const response = await axios.post(API + "/api/analyze", data, {
        headers: { Authorization: "Bearer " + token },
        timeout: 600000,
        onUploadProgress: (event) => {
          if (event.total) setUploadProgress(Math.round((event.loaded / event.total) * 100));
        }
      });
      const verifiedAssets = [];
      for (const asset of response.data.assets || []) {
        const selected = files.find((item) => item.file.name === asset.filename && item.file.size === asset.size_bytes);
        if (!selected || !asset.sha256) {
          verifiedAssets.push({ ...asset, integrity_verification: { status: "unavailable" } });
          continue;
        }
        try {
          const browserDigest = await sha256Hex(selected.file);
          verifiedAssets.push({
            ...asset,
            integrity_verification: {
              status: browserDigest.toLowerCase() === asset.sha256.toLowerCase() ? "match" : "mismatch",
              browser_sha256: browserDigest,
              checked_at: new Date().toISOString()
            }
          });
        } catch {
          verifiedAssets.push({ ...asset, integrity_verification: { status: "unavailable" } });
        }
      }
      const report = {
        ...response.data,
        assets: verifiedAssets,
        case_title: caseTitle.trim() || (files.length === 1 ? "Review · " + files[0].file.name : "Investigation · " + files.length + " evidence files"),
        case_context: context
      };
      setActiveReport(report);
      setSavedReports((current) => [report, ...current.filter((item) => item.report_id !== report.report_id)].slice(0, 50));
      setNotice("Investigation completed and saved in this browser's case history.");
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }
  function openReport(report) {
    setActiveReport(report);
    setPage("reports");
    setNotice("");
  }
  function removeSavedReport(id) {
    setSavedReports((current) => current.filter((item) => item.report_id !== id));
    if (activeReport?.report_id === id) setActiveReport(null);
  }
  async function checkApi() {
    setApiStatus("checking");
    setApiMessage("");
    try {
      const token = await user.getIdToken();
      const response = await axios.get(API + "/api/me", { timeout: 8000, headers: { Authorization: "Bearer " + token } });
      setApiStatus("online");
      setApiMessage("Firebase identity verified by the TruthLens API.");
    } catch {
      setApiStatus("offline");
      setApiMessage("Could not reach " + API + ". Start the FastAPI backend and check VITE_API_URL.");
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
          {id === "history" && savedReports.length > 0 && <em>{savedReports.length}</em>}
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
          <div className="stat-card"><span><FolderOpen size={17} /> TOTAL CASES</span><b>{savedReports.length}</b><small>Saved investigations</small></div>
          <div className="stat-card"><span><ShieldAlert size={17} /> MANIPULATION SIGNALS</span><b>{savedReports.reduce((sum, item) => sum + item.assets.filter((asset) => (asset.video_analysis?.prediction || asset.model_assessment?.prediction) === "fake_signal").length, 0)}</b><small>Signals requiring analyst review</small></div>
          <div className="stat-card"><span><FileBarChart2 size={17} /> REPORTS GENERATED</span><b>{savedReports.length}</b><small>Reports saved in this browser</small></div>
          <div className="stat-card"><span><FileImage size={17} /> MEDIA ANALYZED</span><b>{reviewedAssets}</b><small>Evidence items across cases</small></div>
        </div>

        <div className="dashboard-insights-grid">
          <IntegrityPanel reports={latestReports} />
          <ActivityTimeline reports={latestReports} onOpen={openReport} />
          <HeatmapPreview reports={latestReports} onOpen={openReport} />
        </div>
        <div className="dashboard-lower">
          <div className="content-panel recent-panel">
            <div className="panel-title"><div><span className="section-kicker">CASE REGISTER</span><h2>Recent investigations</h2></div>
              <button className="text-button" onClick={() => setPage("history")}>View all <ArrowRight size={14} /></button></div>
            {latestReports.length ? <div className="recent-case-table">
              <div className="recent-case-head"><span>Case ID / Title</span><span>Evidence</span><span>Updated</span><span>Status</span></div>
              {latestReports.slice(0, 4).map((report) => <CaseRow key={report.report_id} report={report} onOpen={() => openReport(report)} />)}
            </div> : <div className="empty-state"><div className="empty-icon"><FolderOpen size={22} /></div><b>No investigations yet</b><p>Your completed cases will appear here after the first analysis.</p><button className="button button-quiet" onClick={newInvestigation}><Plus size={15} /> Create first case</button></div>}
          </div>
          <div className="content-panel analysis-summary">
            <div className="panel-title"><div><span className="section-kicker">AI ANALYSIS SUMMARY</span><h2>Multimodal review</h2></div><ShieldCheck size={20} /></div>
            <div className="analysis-summary-body"><div className="analysis-ring"><div><b>4</b><small>Review stages</small></div></div>
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
        <div className="page-heading"><div><span className="section-kicker">CASE WORKSPACE / INGEST</span><h1>{page === "analysis" ? "Media analysis" : "New investigation"}</h1><p>Upload original evidence files to begin a multimodal review.</p></div>
          <div className="secure-badge"><ShieldCheck size={15} /> Evidence-first workflow</div></div>
        <Pipeline loading={loading} uploadProgress={uploadProgress} report={activeReport} />
        {loading && <div className="progress-panel"><div><b>{uploadProgress >= 100 ? "Evidence received · model analysis running" : "Uploading evidence"}</b><span>{uploadProgress}% transferred</span></div><div className="progress-track"><i style={{ width: uploadProgress + "%" }} /></div><small>Upload transfer is measured by the browser. Model-stage telemetry is not currently streamed by the API.</small></div>}
        <div className="ingest-layout">
          <div className="content-panel ingest-panel">
            <div className="panel-title"><div><span className="section-kicker">01 / EVIDENCE INTAKE</span><h2>Evidence files</h2></div>
              <span className="file-count">{files.length} / {MAX_FILES} files</span></div>
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
        {activeReport && <ReportView report={activeReport} onJson={() => downloadJson(activeReport)} onPdf={() => {
          if (!exportPdf(activeReport)) setError("Allow pop-ups to open the printable PDF report.");
        }} />}
      </section>}

      {(page === "history" || page === "reports") && <section className="records-page">
        <div className="page-heading"><div><span className="section-kicker">{page === "history" ? "WORKSPACE / CASE REGISTER" : "WORKSPACE / REPORT LIBRARY"}</span>
          <h1>{page === "history" ? "Case history" : "Reports"}</h1><p>{page === "history" ? "Reopen saved investigations and continue reviewing the evidence." : "Open completed case reports or export an evidence summary."}</p></div>
          <button className="button button-primary" onClick={newInvestigation}><Plus size={16} /> New investigation</button></div>
        {activeReport && page === "reports" && <ReportView report={activeReport} onJson={() => downloadJson(activeReport)} onPdf={() => {
          if (!exportPdf(activeReport)) setError("Allow pop-ups to open the printable PDF report.");
        }} />}
        <div className="content-panel record-list-panel">
          <div className="record-toolbar"><div><span className="section-kicker">{page === "history" ? "SAVED INVESTIGATIONS" : "AVAILABLE REPORTS"}</span><h2>{filteredReports.length} records</h2></div>
            <div className="search-box"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search case, ID or filename…" /></div></div>
          {filteredReports.length ? filteredReports.map((report) => <div className="record-line" key={report.report_id}>
            <CaseRow report={report} onOpen={() => openReport(report)} />
            <button className="delete-record" onClick={() => removeSavedReport(report.report_id)} title="Delete saved report"><Trash2 size={15} /></button>
          </div>) : <div className="empty-state"><div className="empty-icon"><FileBarChart2 size={22} /></div><b>No saved reports found</b><p>Run an investigation to create a case record. Reports are stored locally in this browser.</p></div>}
        </div>
      </section>}

      {page === "help" && <section className="settings-page help-page">
        <div className="page-heading"><div><span className="section-kicker">TRUTHLENS / SUPPORT</span><h1>Help & Support</h1><p>Quick guidance for running an evidence review.</p></div>
          <button className="button button-primary" onClick={newInvestigation}><Plus size={16} /> Start an investigation</button></div>
        <div className="settings-grid">
          <div className="content-panel settings-card"><span className="section-kicker">01 / GETTING STARTED</span><h2>Upload your evidence</h2><p>Add supported image, video, or speech-audio files. You can attach up to eight files, with a 40 MB limit per file.</p></div>
          <div className="content-panel settings-card"><span className="section-kicker">02 / REVIEW FINDINGS</span><h2>Interpret model signals</h2><p>Review modality outputs, metadata, SHA-256 fingerprints and available attribution or sampled-frame details. A model score is not a calibrated probability.</p></div>
          <div className="content-panel settings-card"><span className="section-kicker">03 / SAVE YOUR WORK</span><h2>Case history & reports</h2><p>Completed case reports are saved in this browser and can be reopened from Case History. Export a JSON or print-ready PDF copy.</p></div>
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
          <div className="content-panel settings-card"><span className="section-kicker">DATA & PRIVACY</span><h2>Local report storage</h2>
            <p>Saved investigation reports are kept in this browser's local storage. Uploaded media previews are temporary and are not added to case history. The backend processes uploaded files and removes its temporary working copies.</p>
            <div className="settings-metric"><b>{savedReports.length}</b><span>Saved reports in this browser</span></div></div>
          <div className="content-panel settings-card"><span className="section-kicker">MODEL TRANSPARENCY</span><h2>Review before concluding</h2>
            <p>Model scores are not calibrated probabilities. Video uses sampled frame classification, audio is scoped to the selected speech model and cross-modal review compares separate model signals only.</p>
            <div className="safety-note"><Shield size={15} /> TruthLens supports investigation; it does not certify authenticity.</div></div>
        </div>
      </section>}

      <footer className="app-footer"><span>TRUTHLENS</span><i /> Evidence context matters. Automated model signals are not final conclusions.</footer>
    </main>
  </div>;
}
