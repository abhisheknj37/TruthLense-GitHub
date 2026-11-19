import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import LiveCapture from "./LiveCapture";
import {
  Activity, AlertTriangle, ArrowRight, AudioLines, Bell, CalendarDays, FileText,
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
  audio: ["wav", "flac", "ogg"],
  text: ["txt", "md"]
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
  if (ALLOWED.text.includes(ext) || file.type === "text/plain" || file.type === "text/markdown") return "text";
  return "unsupported";
}
function typeIcon(type, size = 17) {
  if (type === "video") return <Video size={size} />;
  if (type === "audio") return <AudioLines size={size} />;
  if (type === "text") return <FileText size={size} />;
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
  const esc = escapeHtml;
  const humanDate = (value) => value ? esc(formatDate(value)) : "Not recorded";
  const titleCase = (value) => String(value || "Not available").replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
  const safeValue = (key, value) => {
    if (value === null || value === undefined || value === "") return "Not available";
    if (typeof value === "number" && (!Number.isFinite(value) || value < 0)) return "Unavailable (invalid source metadata)";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    return String(value);
  };
  const tableRows = (items) => Object.entries(items || {}).filter(([, value]) =>
    ["string", "number", "boolean"].includes(typeof value) || value === null
  ).map(([key, value]) => "<tr><th>" + esc(titleCase(key)) + "</th><td>" + esc(safeValue(key, value)) + "</td></tr>").join("");
  const renderOcr = (data, mediaType) => {
    if (!data || typeof data !== "object") return "";
    const frames = Array.isArray(data.frames) ? data.frames : [];
    const previewText = (value) => {
      const text = String(value || "").replace(/\\s+/g, " ").trim();
      return text.length > 180 ? text.slice(0, 180) + "… [full text in technical appendix]" : text || "No text recognized";
    };
    const rowFor = (frame) => {
      const confidence = frame.mean_word_confidence ?? frame.confidence;
      return "<tr>" + (mediaType === "video" ? "<td>" + esc(safeValue("timestamp", frame.timestamp_seconds)) + "</td><td>" +
        esc(safeValue("frame", frame.frame_index)) + "</td>" : "") +
        "<td class='ocr-text'>" + esc(previewText(frame.text)) +
        "</td><td>" + esc(frame.engine || data.engine || "Not recorded") + "</td><td>" +
        esc(confidence == null ? "Not reported" : String(confidence) + (frame.confidence_level ? " · " + frame.confidence_level : "")) +
        "</td><td>" + (frame.analysis_eligible ? "Eligible" : "Excluded · review") + "</td></tr>";
    };
    const eligibleRows = frames.filter((frame) => frame.analysis_eligible).map(rowFor).join("");
    const reviewRows = frames.filter((frame) => !frame.analysis_eligible || frame.confidence_level === "low").map(rowFor).join("");
    const rows = !frames.length && data.text ? rowFor({ ...data, timestamp_seconds: null, frame_index: null }) : "";
    const comparison = data.engine_comparison && Object.keys(data.engine_comparison).length
      ? "<h4>Independent OCR engine comparison</h4>" + objectTable(data.engine_comparison) : "";
    const correction = data.investigator_correction?.text
      ? "<div class='human-correction'><h4>INVESTIGATOR-CORRECTED TEXT · HUMAN-EDITED</h4><p class='muted'>Separate from machine OCR; not independently verified.</p><pre>" +
        esc(data.investigator_correction.text) + "</pre><p class='muted'>Reviewer: " +
        esc(data.investigator_correction.reviewer_uid || "Not recorded") + " · Saved: " +
        humanDate(data.investigator_correction.updated_at) + "</p></div>" : "";
    const summary = "<p class='muted'>Machine OCR · " + esc(data.engine || "Engine not recorded") +
      " · " + esc(data.languages || "Language not recorded") + " · " +
      (mediaType === "video" ? frames.length + " sampled frame(s)." : "Image text extraction.") +
      " Confidence is a recognition estimate, not an authenticity probability.</p>";
    const tableHead = "<table class='data-table compact ocr-table'><thead><tr>" +
      (mediaType === "video" ? "<th>Time (s)</th><th>Frame</th>" : "") +
      "<th>Extracted text preview</th><th>Engine</th><th>Confidence</th><th>Analysis</th></tr></thead><tbody>";
    const eligibleTable = eligibleRows ? "<h5>Eligible machine OCR · " + frames.filter((frame) => frame.analysis_eligible).length + "</h5>" + tableHead + eligibleRows + "</tbody></table>" : "";
    const reviewTable = reviewRows ? "<h5>Excluded / low-confidence OCR · human review only</h5>" + tableHead + reviewRows + "</tbody></table>" : "";
    const table = eligibleTable + reviewTable || (rows ? tableHead + rows + "</tbody></table>" : "<p class='empty'>No machine-readable OCR or transcription text was recorded.</p>");
    const low = frames.filter((frame) => !frame.analysis_eligible || frame.confidence_level === "low").length;
    const quality = low ? "<p class='ocr-warning'>" + low + (mediaType === "video" ? " sampled frame(s)" : " OCR result(s)") +
      " are excluded from claim analysis or marked low confidence. Verify text against the original evidence.</p>" : "";
    return "<h4>OCR / transcription review</h4>" + summary + quality + table + comparison + correction;
  };
  const renderTemporal = (data) => {
    if (!data || typeof data !== "object") return "";
    const samples = Array.isArray(data.samples) ? data.samples : [];
    const model = data.trained_model_assessment || data.model_assessment || {};
    const summary = {
      decoded_frame_count: data.decoded_frame_count ?? "Not recorded",
      sampled_frames: data.sampled_frames ?? data.sampled_frame_count ?? "Not recorded",
      sampled_transitions: data.sampled_transitions ?? samples.length,
      scene_change_count: Array.isArray(data.scene_change_timestamps_seconds) ? data.scene_change_timestamps_seconds.length : "Not recorded",
      motion_discontinuity_count: Array.isArray(data.motion_discontinuity_timestamps_seconds) ? data.motion_discontinuity_timestamps_seconds.length : "Not recorded",
      temporal_model_status: model.status || "model_not_configured",
    };
    return "<h4>Temporal video review · heuristic summary</h4><p class='muted'>Optical flow and histogram changes are descriptive review cues, not a trained deepfake verdict. Detailed transition measurements are in the technical appendix.</p>" +
      objectTable(summary) + "<p class='muted'>Validated temporal classifier: " +
      esc(model.status === "model_configured" ? "Configured; validation status must be assessed separately" : "Not configured / not demonstrated") + ".</p>";
  };
  const renderNested = (value) => {
    if (Array.isArray(value)) {
      if (!value.length) return "<span class='empty'>None recorded</span>";
      return "<ul class='nested-list'>" + value.map((item) => "<li>" + renderNested(item) + "</li>").join("") + "</ul>";
    }
    if (value && typeof value === "object") return objectTable(value);
    return esc(safeValue("", value));
  };
  const compactTemporalSamples = (samples) => {
    if (!Array.isArray(samples) || !samples.length) return "<p class='empty'>No transition samples recorded.</p>";
    const rows = samples.map((sample) => "<tr><td>" + esc(safeValue("timestamp", sample.timestamp_seconds)) +
      "</td><td>" + esc(safeValue("mean flow", sample.mean_flow)) +
      "</td><td>" + esc(safeValue("flow p95", sample.flow_p95)) +
      "</td><td>" + esc(safeValue("histogram distance", sample.histogram_distance)) + "</td></tr>").join("");
    return "<p class='muted'>Compact transition measurements · " + samples.length + " samples</p><table class='data-table compact'><thead><tr><th>Time (s)</th><th>Mean flow</th><th>Flow P95</th><th>Histogram distance</th></tr></thead><tbody>" + rows + "</tbody></table>";
  };
  const objectTable = (object) => {
    if (!object || typeof object !== "object" || Array.isArray(object)) return "<p class='empty'>No structured details available.</p>";
    const rows = Object.entries(object).map(([key, value]) =>
      "<tr><th>" + esc(titleCase(key)) + "</th><td>" + (key === "samples" && Array.isArray(value) ? compactTemporalSamples(value) : renderNested(value)) + "</td></tr>"
    ).join("");
    return "<table class='data-table'>" + rows + "</table>";
  };
  const observationDetail = (asset, item) => {
    const detail = String(item.detail || item.note || item.interpretation || "No detail supplied.");
    if (asset.media_type === "video" && /from\s+-\d+\s+reported frames/i.test(detail)) {
      return "The stored report contains an invalid negative container frame count. This older case was not re-decoded; re-upload the original video to obtain a sequentially decoded frame count.";
    }
    return detail;
  };
  const observationList = (asset) => (asset.observations || []).length
    ? "<div class='finding-list'>" + asset.observations.map((item) =>
      "<div class='finding'><div class='finding-head'><b>" + esc(item.title || item.code || "Observation") +
      "</b><span class='tag " + esc(item.severity || "info") + "'>" + esc(titleCase(item.severity || "info")) +
      "</span></div><p>" + esc(observationDetail(asset, item)) + "</p></div>"
    ).join("") + "</div>"
    : "<p class='empty'>No structured observations were returned for this evidence item.</p>";
  const assets = (report.assets || []).map((asset, index) => {
    const model = asset.model_assessment || {};
    const metadata = tableRows(asset.metadata || {});
    const provenance = asset.provenance || {};
    const extras = [
      asset.forensic_parameters && Object.keys(asset.forensic_parameters).length ? "<h4>Image forensic measurements</h4>" + objectTable(asset.forensic_parameters) : "",
      Object.keys(provenance).length ? "<h4>Evidence provenance</h4>" + objectTable(provenance) : "",
      asset.text_extraction ? renderOcr(asset.text_extraction, asset.media_type) : "",
      asset.model_assessment?.voice_synthesis_review ? "<h4>Synthetic / cloned-voice review indicator · research model</h4><p><b>Indicator:</b> " + esc(titleCase(asset.model_assessment.voice_synthesis_review.indicator)) + "</p><p><b>Model:</b> " + esc(asset.model_assessment.voice_synthesis_review.model || model.model_name || "Not recorded") + " · <b>Windows:</b> " + esc(safeValue("windows", asset.model_assessment.voice_synthesis_review.windows_assessed)) + " · <b>Median fake-direction score:</b> " + esc(safeValue("score", asset.model_assessment.voice_synthesis_review.median_fake_model_score)) + "</p><p class='muted'>" + esc(asset.model_assessment.voice_synthesis_review.note || "Not recorded") + "</p>" : "",
      asset.model_assessment?.audio_review_timeline ? "<h4>Audio review timeline</h4>" + objectTable(asset.model_assessment.audio_review_timeline) : "",
      asset.video_analysis?.temporal_forensics ? renderTemporal(asset.video_analysis.temporal_forensics) : "",
      asset.uncertainty_assessment ? "<h4>Confidence and uncertainty assessment</h4><table class='data-table'><tr><th>Uncertainty level</th><td>" + esc(titleCase(asset.uncertainty_assessment.uncertainty_level)) + "</td></tr><tr><th>Signal separation</th><td>" + esc(titleCase(asset.uncertainty_assessment.signal_separation)) + "</td></tr><tr><th>Raw model score</th><td>" + esc(safeValue("score", asset.uncertainty_assessment.raw_model_score)) + " · Uncalibrated model output, not a confidence percentage</td></tr><tr><th>Evidence support</th><td>" + esc(asset.uncertainty_assessment.sample_support_label || "Not available") + " · " + esc(safeValue("samples", asset.uncertainty_assessment.sample_count)) + " sampled unit(s)</td></tr><tr><th>Within-sample score range</th><td>" + esc(safeValue("range", asset.uncertainty_assessment.sample_score_range?.range)) + "</td></tr><tr><th>Calibration</th><td>Not calibrated</td></tr></table><ul>" + (asset.uncertainty_assessment.reasons || []).map(reason => "<li>" + esc(reason) + "</li>").join("") + "</ul><p class='muted'>" + esc(asset.uncertainty_assessment.interpretation || "") + "</p>" : "",
      asset.video_analysis?.face_identity_consistency ? "<h4>Face consistency review</h4>" + objectTable(asset.video_analysis.face_identity_consistency) : ""
    ].join("");
    return "<section class='asset page-section'><div class='asset-title'><span class='eyebrow'>EVIDENCE ITEM " + String(index + 1).padStart(2, "0") +
      "</span><h3>" + esc(asset.filename || "Unnamed evidence") + "</h3><span class='tag neutral'>" +
      esc(String(asset.media_type || "unknown").toUpperCase()) + " · " + esc(formatBytes(asset.size_bytes)) + "</span></div>" +
      "<div class='metric-grid'><div class='metric'><span>MODEL SIGNAL</span><b>" + esc(predictionLabel(model.prediction)) +
      "</b></div><div class='metric'><span>MODEL</span><b>" + esc(model.model_name || "Not available") +
      "</b></div><div class='metric'><span>ASSESSMENT</span><b>" + esc(titleCase(model.status || "inconclusive")) + "</b></div></div>" +
      "<div class='hash-block'><span>SHA-256 · RECEIVED FILE BYTES</span><code>" + esc(asset.sha256 || "Not recorded") +
      "</code><span class='eyebrow'>EVIDENCE REFERENCE</span><code>" + esc(asset.asset_id || "Not assigned") + "</code></div>" +
      "<h4>Observed findings</h4>" + observationList(asset) +
      (metadata ? "<h4>Technical metadata</h4><table class='data-table'>" + metadata + "</table>" : "") + extras + "</section>";
  }).join("");
  const cross = (report.cross_modal_review || []).length
    ? "<div class='review-list'>" + report.cross_modal_review.map((item) =>
      "<div class='review-row'><div><b>" + esc(titleCase(item.type || "Review")) + "</b><span class='tag neutral'>" +
      esc(titleCase(item.status || "Not specified")) + "</span></div><p>" +
      esc(item.detail || item.note || item.interpretation || "No explanatory detail supplied.") + "</p></div>"
    ).join("") + "</div>"
    : "<p class='empty'>No cross-modal comparisons were available for this case.</p>";
  const findingTitles = new Map((report.assets || []).flatMap((asset) =>
    (asset.observations || []).map((finding, index) => [
      asset.asset_id + ":" + String(finding.code || index),
      (finding.title || finding.code || "Technical finding") + " · " + (asset.filename || "Evidence")
    ])
  ));
  const reviews = (report.investigator_reviews || []).length
    ? "<p class='section-intro'>Each decision applies only to the named technical finding. A Confirmed decision does not confirm media authenticity or validate a model. Reviewer identity is included only when supplied by the review API.</p><table class='data-table review-table'><thead><tr><th>Technical finding</th><th>Finding decision</th><th>Analyst rationale</th><th>Reviewer (if supplied)</th><th>Updated</th></tr></thead><tbody>" +
      report.investigator_reviews.map((item) => "<tr><td>" + esc(findingTitles.get(item.finding_id) || item.finding_id || "Unmatched finding") + "<br><code>" + esc(item.finding_id || "") + "</code></td><td><span class='tag " +
      esc(item.status) + "'>" + esc(titleCase(item.status)) + "</span></td><td>" + esc(item.comment || "No analyst rationale recorded") +
      "</td><td>" + esc(item.reviewer_name || item.reviewer_email || item.reviewer_uid || "Not supplied by API") + "</td><td>" + humanDate(item.updated_at) + "</td></tr>").join("") + "</tbody></table>"
    : "<p class='empty'>No saved investigator decisions were available at export time.</p>";
  const events = (report.evidence_events || []).length
    ? "<table class='data-table'><thead><tr><th>Event</th><th>Details</th><th>Timestamp</th><th>Event hash</th></tr></thead><tbody>" +
      report.evidence_events.map((item) => "<tr><td>" + esc(titleCase(item.event)) + "</td><td>" +
      esc(item.detail || item.filename || item.asset_id || "—") + "</td><td>" + humanDate(item.timestamp) +
      "</td><td><code>" + esc(item.event_hash || "Not recorded") + "</code></td></tr>").join("") +
      "</tbody></table>"
    : "<p class='empty'>No evidence events were recorded.</p>";
  const limitations = (report.limitations || []).map((item) => "<li>" + esc(item) + "</li>").join("");
  const technicalAppendix = (report.assets || []).map((asset) => {
    const ocr = asset.text_extraction || {};
    const frames = Array.isArray(ocr.frames) ? ocr.frames : [];
    const temporal = asset.video_analysis?.temporal_forensics || {};
    const samples = Array.isArray(temporal.samples) ? temporal.samples : [];
    const ocrDetails = frames.length ? "<h4>Full machine OCR · " + esc(asset.filename || "Evidence") + "</h4><table class='data-table compact'><thead><tr><th>Time (s)</th><th>Frame</th><th>Full extracted text</th><th>Engine</th><th>Gate</th></tr></thead><tbody>" +
      frames.map((frame) => "<tr><td>" + esc(safeValue("time", frame.timestamp_seconds)) + "</td><td>" + esc(safeValue("frame", frame.frame_index)) +
      "</td><td class='ocr-full'>" + esc(frame.text || "No text recognized") + "</td><td>" + esc(frame.engine || ocr.engine || "Not recorded") +
      "</td><td>" + (frame.analysis_eligible ? "Eligible" : "Excluded") + "</td></tr>").join("") + "</tbody></table>" : "";
    const temporalDetails = samples.length ? "<h4>Full temporal transition measurements · " + esc(asset.filename || "Evidence") + "</h4>" +
      compactTemporalSamples(samples) : "";
    return ocrDetails + temporalDetails;
  }).filter(Boolean).join("");

  const summaryStatus = getAssessmentStatusLabel(report.overall_status);
  const allowedConclusions = ["authenticity_supported", "manipulation_indicators_identified", "inconclusive", "insufficient_evidence"];
  const caseConclusion = allowedConclusions.includes(report.case_conclusion)
    ? titleCase(report.case_conclusion) : "Inconclusive";
  const conclusionNote = report.case_conclusion
    ? "Case conclusion supplied separately from technical finding reviews."
    : "No explicit investigator case conclusion was recorded. Model outputs and confirmed technical findings do not establish authenticity.";
  const html = "<!doctype html><html><head><meta charset='utf-8'><title>TruthLens Forensic Case Report</title><style>" +
    "@page{size:A4;margin:17mm 15mm}*{box-sizing:border-box}body{font:10pt/1.55 Arial,sans-serif;color:#17243a;margin:0;background:#fff}" +
    ".cover{border-bottom:4px solid #173b66;padding:0 0 18px;margin-bottom:22px}.brand{font-size:10pt;font-weight:700;letter-spacing:2px;color:#245b91}" +
    "h1{font-size:25pt;line-height:1.15;margin:10px 0 6px;color:#10243d}h2{font-size:15pt;color:#153c65;margin:25px 0 10px;padding-bottom:6px;border-bottom:1px solid #cbd6e2}" +
    "h3{font-size:12pt;margin:5px 0 8px;color:#142e4b;overflow-wrap:anywhere}h4{font-size:10pt;margin:18px 0 7px;color:#24527c}" +
    "p{margin:5px 0 10px}.muted,.eyebrow{color:#61748b}.eyebrow{font-size:8pt;font-weight:700;letter-spacing:1.2px}" +
    ".case-meta{color:#61748b;font-size:9pt}.summary{background:#f2f6fa;border-left:4px solid #2d6596;padding:13px 15px;border-radius:4px}" +
    ".metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0}.metric{background:#f4f7fa;border:1px solid #dbe3ec;border-radius:5px;padding:9px;min-width:0}" +
    ".metric span,.hash-block span{display:block;font-size:7.5pt;letter-spacing:.8px;color:#61748b;font-weight:700}.metric b{display:block;margin-top:4px;font-size:9pt;overflow-wrap:anywhere}" +
    ".status-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0}.status{border:1px solid #dbe3ec;padding:10px;border-radius:5px}.status b{display:block;margin-top:4px}" +
    ".asset{border:1px solid #cbd6e2;border-radius:7px;padding:14px;margin:12px 0 18px;break-inside:auto}.asset-title{border-bottom:1px solid #e1e7ee;padding-bottom:10px}" +
    ".tag{display:inline-block;border-radius:12px;padding:2px 8px;font-size:8pt;font-weight:700;background:#e9eff5;color:#294968}.tag.warning{background:#fff2d9;color:#85520a}.tag.info{background:#e8f1fb;color:#24527c}.tag.confirmed{background:#e4f4e9;color:#21633a}.tag.rejected{background:#fbe8e8;color:#8a2c2c}.tag.inconclusive{background:#fff2d9;color:#85520a}" +
    ".finding{border:1px solid #e0e6ed;border-radius:5px;padding:9px 11px;margin:7px 0;break-inside:avoid}.finding-head{display:flex;justify-content:space-between;gap:10px}.finding p,.review-row p{margin:5px 0 0}" +
    ".hash-block{background:#f6f8fa;padding:10px;border-radius:5px;margin:10px 0}.hash-block code{display:block;margin:4px 0 9px}" +
    ".data-table{width:100%;border-collapse:collapse;margin:7px 0 12px;table-layout:fixed}.data-table th,.data-table td{text-align:left;vertical-align:top;padding:7px 8px;border-bottom:1px solid #e2e8ef;overflow-wrap:anywhere}.data-table th{width:27%;color:#53677e;font-weight:700;background:#f6f8fa}.data-table td{width:auto}.data-table .data-table{table-layout:auto;width:100%;margin:3px 0 5px}.data-table .data-table th{width:38%;white-space:normal}.data-table .data-table td{width:62%}.review-table{table-layout:auto}.review-table th{width:auto}.review-table td:nth-child(2){font-weight:700}.review-table .tag.confirmed{background:#e8eef5;color:#294968}.data-table.compact th,.data-table.compact td{padding:4px 6px;font-size:8pt}.data-table.compact th{width:auto}.nested-list{margin:3px 0;padding-left:18px}.nested-list li{margin:3px 0}" +
    "code{font:8pt Consolas,monospace;overflow-wrap:anywhere;word-break:break-word}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:8pt/1.4 Consolas,monospace;background:#f6f8fa;padding:8px;border-radius:4px;margin:0}" +
    ".review-row{padding:9px 0;border-bottom:1px solid #e1e7ee;break-inside:avoid}.review-row>div{display:flex;justify-content:space-between;gap:10px}" +
    ".empty{color:#61748b;font-style:italic}.section-intro{color:#61748b;margin-top:-4px}.human-correction{border:1px solid #9aaec3;background:#f4f7fa;border-radius:6px;padding:12px;margin:12px 0;break-inside:avoid}.human-correction h4{margin-top:0}.human-correction pre{white-space:pre-wrap;overflow-wrap:anywhere}.footer{border-top:1px solid #cbd6e2;margin-top:25px;padding-top:10px;color:#61748b;font-size:8pt}" +
    ".page-section{break-inside:auto}thead{display:table-header-group}tr,.finding,.review-row,.human-correction{break-inside:avoid}.ocr-table{font-size:7.5pt}.ocr-table th{width:auto}.ocr-table td{font-size:7.5pt}.ocr-text{white-space:pre-wrap;max-width:180px}.ocr-full{white-space:pre-wrap;overflow-wrap:anywhere}.ocr-warning{background:#fff4df;border-left:3px solid #b7791f;padding:8px 10px;color:#704b12}.asset{padding:12px;margin:8px 0 12px}h2{margin:18px 0 8px}h4{margin:12px 0 5px}.data-table th,.data-table td{padding:5px 6px}@media print{a{color:inherit;text-decoration:none}.asset{break-inside:auto}}" +
    "</style></head><body>" +
    "<header class='cover'><div class='brand'>TRUTHLENS / DIGITAL FORENSICS</div><h1>Forensic Case Report</h1>" +
    "<div class='case-meta'>Case reference: " + esc(report.report_id) + "<br>Generated: " + humanDate(new Date().toISOString()) +
    " · Investigation date: " + humanDate(report.created_at) + "</div></header>" +
    "<h2>01 · Executive case summary</h2><div class='summary'><b>Case assessment</b><p>" + esc(report.summary || "No case summary supplied.") +
    "</p><b>Overall case conclusion · " + esc(caseConclusion) + "</b><p>" + esc(conclusionNote) + "</p></div><div class='status-grid'><div class='status'><span class='eyebrow'>AUTOMATED ASSESSMENT</span><b>" + esc(summaryStatus) +
    "</b></div><div class='status'><span class='eyebrow'>HUMAN OVERSIGHT</span><b>" +
    (report.human_review_required ? "Required" : "Not specified") + "</b></div><div class='status'><span class='eyebrow'>FINDINGS REVIEWED</span><b>" +
    reviewProgress.reviewed + " / " + reviewProgress.total + " · " + (reviewProgress.complete ? "All findings reviewed" : "Review pending") + "</b></div></div>" +
    "<h2>02 · Evidence inventory and technical findings</h2><p class='section-intro'>Each evidence item is identified by its case reference and received-file SHA-256 digest.</p>" +
    (assets || "<p class='empty'>No evidence items were recorded.</p>") +
    "<h2>03 · Cross-modal and contextual review</h2>" + cross +
    (report.text_analysis ? "<h2>04 · Text and claim-context review</h2>" + objectTable(report.text_analysis) : "") +
    "<h2>05 · Investigator decisions</h2><p class='section-intro'>Human decisions are recorded separately from automated model outputs.</p>" + reviews +
    "<h2>06 · Evidence event history</h2>" + events +
    "<h2>07 · Technical appendix · full OCR and temporal measurements</h2><p class='section-intro'>Full machine OCR and per-transition measurements are retained here for traceability. OCR text excluded by the quality gate is not used for claim-context analysis.</p>" + (technicalAppendix || "<p class='empty'>No additional OCR or temporal measurements were recorded for this case.</p>") +
    "<h2>08 · Limitations, caveats and required follow-up</h2><ul>" + (limitations || "<li>No limitations were supplied by the analysis service.</li>") + "</ul>" +
    "<div class='summary'><b>Interpretation notice</b><p>Model scores are not calibrated probabilities or proof of authenticity. A matching SHA-256 digest establishes byte-level consistency for the received file, not creator identity or content truth. Findings require qualified human review and original-source context.</p></div>" +
    "<div class='footer'>TRUTHLENS · Confidential investigative working document · Preserve original source media and maintain chain-of-custody records separately.</div>" +
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

function UncertaintyAssessment({ assessment }) {
  if (!assessment) return null;
  const level = assessment.uncertainty_level || "high";
  const labels = { lower: "Lower review uncertainty", moderate: "Moderate review uncertainty", high: "High review uncertainty" };
  const reasons = assessment.reasons || [];
  return <section className={"uncertainty-card uncertainty-" + level}>
    <div className="uncertainty-heading">
      <div><span className="section-kicker">CONFIDENCE & UNCERTAINTY</span><h4>Score separation and evidence support</h4></div>
      <span className={"uncertainty-level " + level}>{labels[level] || "Uncertainty not available"}</span>
    </div>
    <div className="uncertainty-metrics">
      <div><small>Signal separation</small><b>{String(assessment.signal_separation || "unavailable").replaceAll("_", " ")}</b></div>
      <div><small>Evidence support</small><b>{assessment.sample_support_label || "Not available"}</b></div>
      <div><small>Samples assessed</small><b>{assessment.sample_count ?? 0}</b></div>
    </div>
    {assessment.raw_model_score != null && <div className="uncertainty-score-line">
      <span>Raw model score</span><b>{Number(assessment.raw_model_score).toFixed(3)} / 1.000</b>
      <small>This is an uncalibrated model output, not a confidence percentage.</small>
    </div>}
    {reasons.length > 0 && <div className="uncertainty-reasons"><b>Why this uncertainty level was assigned</b>
      {reasons.map((reason, index) => <p key={index}><span>•</span>{reason}</p>)}
    </div>}
    {assessment.interpretation && <p className="uncertainty-explanation">{assessment.interpretation}</p>}
    <div className="uncertainty-calibration"><span>CALIBRATION STATUS</span><b>Not calibrated</b><small>Held-out dataset calibration has not been performed for this model.</small></div>
  </section>;
}

function ModelAssessment({ asset }) {
  const model = asset.model_assessment || {};
  const video = asset.video_analysis || {};
  const details = asset.media_type === "video" ? video : model;
  const assessed = details.status === "model_assessed";
  const frames = video.frames || [];
  const segments = model.segments || [];
  return <div className="model-result">
    <div className="section-kicker">{asset.media_type === "audio" ? "AUDIO AUTHENTICITY INDICATORS · RESEARCH MODEL" : "MODEL ASSESSMENT"}</div>
    <div className="model-result-head">
      <b>{predictionLabel(details.prediction)}</b>
      <span className={"model-pill " + (assessed ? "available" : "pending")}>{assessed ? "Signal available" : "Inconclusive"}</span>
    </div>
    {details.model_name && <small className="model-name">{details.model_name}</small>}{asset.media_type === "audio" && <p className="model-note">Synthetic / spoof-speech screening only. This model does not identify a cloned speaker, prove voice cloning, or authenticate speaker identity.</p>}
    {asset.media_type === "audio" && model.voice_synthesis_review && <div className="provenance-note">
      <b>SYNTHETIC / CLONED-VOICE REVIEW INDICATOR</b>
      <span>{String(model.voice_synthesis_review.indicator || "inconclusive").replaceAll("_", " ")}</span>
      <small>{model.voice_synthesis_review.note}</small>
      <small>{model.voice_synthesis_review.model || model.model_name} · {model.voice_synthesis_review.windows_assessed || 0} speech window(s) assessed · median fake-direction score {scorePercent(model.voice_synthesis_review.median_fake_model_score)}</small>
    </div>}
    <UncertaintyAssessment assessment={asset.uncertainty_assessment || details.uncertainty_assessment || model.uncertainty_assessment} />
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
      <p className="model-note">{model.audio_review_timeline.interpretation}</p><small>Acoustic review cues are separate from synthetic-speech classification and are not clone detection.</small>
      {model.audio_review_timeline.acoustic_review_points?.map((point, index) => <p key={index}>{point.timestamp_seconds}s · unusual energy review cue · RMS {point.rms}</p>)}
    </div>}
    {asset.media_type === "audio" && segments.length > 0 && <div className="audio-segments">
      <b>Synthetic / spoof-speech model indicators by segment</b>
      {segments.map((segment, index) => <div className="audio-segment" key={index}>
        <div><span>{segment.start_seconds}s – {segment.end_seconds}s</span><b>{scorePercent(segment.fake_model_score)}</b></div>
        <div className="score-track"><i style={{ width: Math.max(0, Math.min(100, (segment.fake_model_score || 0) * 100)) + "%" }} /></div>
        <small>{segment.fake_model_score >= 0.65 ? "Higher model signal" : segment.fake_model_score <= 0.35 ? "Lower model signal" : "Ambiguous band"}</small>
      </div>)}
    </div>}
  </div>;
}

function EvidenceCard({ asset, reportId, user }) {
  const extraction = asset.text_extraction || {};
  const [correctedText, setCorrectedText] = useState(extraction.investigator_correction?.text ?? extraction.text ?? "");
  const [correctionStatus, setCorrectionStatus] = useState("");
  const [savingCorrection, setSavingCorrection] = useState(false);
  async function saveCorrection() {
    if (!user || !reportId || !asset.asset_id) return;
    setSavingCorrection(true);
    setCorrectionStatus("");
    try {
      const token = await user.getIdToken();
      await axios.put(
        API + "/api/reports/" + encodeURIComponent(reportId) + "/assets/" + encodeURIComponent(asset.asset_id) + "/ocr-correction",
        { corrected_text: correctedText },
        { headers: { Authorization: "Bearer " + token }, timeout: 15000 }
      );
      setCorrectionStatus("Correction saved with reviewer and timestamp. Raw OCR remains unchanged.");
    } catch (error) {
      setCorrectionStatus(error.response?.data?.detail || "Could not save correction.");
    } finally {
      setSavingCorrection(false);
    }
  }
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
      {asset.text_extraction.analysis_eligible === false && <small>Excluded from automated claim analysis by the OCR quality gate.</small>}
      {asset.text_extraction.text && <div className="ocr-correction">
        <label htmlFor={"ocr-correction-" + asset.asset_id}>Investigator-corrected text</label>
        <textarea id={"ocr-correction-" + asset.asset_id} rows={5} maxLength={12000}
          value={correctedText} onChange={(event) => setCorrectedText(event.target.value)}
          placeholder="Correct the OCR text after checking the original evidence." />
        <button className="button button-quiet" onClick={saveCorrection} disabled={savingCorrection || !user}>
          {savingCorrection ? "Saving…" : "Save human correction"}
        </button>
        {correctionStatus && <small role="status">{correctionStatus}</small>}
        {asset.text_extraction.investigator_correction?.updated_at && <small>Last saved: {formatDate(asset.text_extraction.investigator_correction.updated_at)}</small>}
      </div>}
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
    {result.semantic_comparison?.status && <div className="semantic-review-block">
      <b>SEMANTIC EVIDENCE REVIEW</b>
      <small>{result.semantic_comparison.status === "completed" ? result.semantic_comparison.model : "Optional NLI model not configured or unavailable"}</small>
      {result.semantic_comparison.comparisons?.slice(0,8).map((item,index)=><article key={index}>
        <span className={"semantic-candidate "+item.review_label}>{item.review_label.replaceAll("_"," ")}</span>
        <b>{item.source_a} ↔ {item.source_b}</b>
        <p>A: {item.sentence_a}</p><p>B: {item.sentence_b}</p>
        <small>Contradiction candidate scores: {Math.max(item.a_to_b.contradiction,item.b_to_a.contradiction).toFixed(3)} · Entailment: {Math.max(item.a_to_b.entailment,item.b_to_a.entailment).toFixed(3)} · uncalibrated NLI outputs</small>
      </article>)}
      <p>{result.semantic_comparison.limitation || result.semantic_comparison.note}</p>
    </div>}
    <p className="model-note">Rule-based triage and scholarly-source discovery do not establish facts. NLI output is a candidate sentence relationship, not a verified true/false verdict. Check original media and primary sources.</p>
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
    <div className="asset-grid">{report.assets.map((asset) => <EvidenceCard key={asset.asset_id} asset={asset} reportId={report.report_id} user={user} />)}</div>
    <InvestigatorReview report={report} user={user} onChange={onReviewsChange} />
    {report.cross_modal_review?.length > 0 && <div className="cross-review">
      <div className="cross-header"><div><span className="section-kicker">RELATIONSHIP REVIEW</span><h3>Cross-modal comparison</h3></div><span className="model-pill pending">No fused authenticity score</span></div>
      {report.cross_modal_review.map((item, index) => <div className="cross-item" key={index}>
        <span className="model-pill pending">{String(item.status).replaceAll("_", " ")}</span>
        {(item.source_a || item.source_b) && <p><b>{item.source_a_kind || "Source"}:</b> {item.source_a} ↔ <b>{item.source_b_kind || "Source"}:</b> {item.source_b}</p>}
        {item.shared_terms?.length > 0 && <p><b>Shared terms:</b> {item.shared_terms.join(", ")} · {item.shared_term_count ?? item.shared_terms.length} matched terms</p>}
        {(item.source_a_preview || item.source_b_preview) && <details><summary>Compare extracted text / claim excerpts</summary>{item.source_a_preview && <p><b>Source A:</b> {item.source_a_preview}</p>}{item.source_b_preview && <p><b>Source B:</b> {item.source_b_preview}</p>}</details>}
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

function LandingPage({ onStart }) {
  return <div className="landing-page">
    <header className="landing-nav">
      <a className="landing-brand" href="#"><span className="landing-brand-mark"><BrandGlyph size={30} /></span><b>Truth<span>Lens</span></b></a>
      <nav><a href="#landing-home">Home</a><a href="#landing-features">Features</a><a href="#landing-about">About</a><a href="#landing-contact">Contact</a></nav>
      <div className="landing-nav-actions">
        <button className="landing-signin" onClick={() => onStart("signin")}>Sign in</button>
        <button className="landing-get-started" onClick={() => onStart("signup")}>Get Started <ArrowRight size={15}/></button>
      </div>
    </header>
    <main>
      <section className="landing-hero" id="landing-home">
        <div className="landing-hero-copy">
          <span className="landing-eyebrow"><i/> VERIFY · ANALYZE · STAY INFORMED</span>
          <h1>See the <span>Truth.</span><br/>Beyond the Noise.</h1>
          <p>Investigate suspicious images, videos and speech with multimodal evidence analysis, source comparison and human-led review.</p>
          <div className="landing-hero-actions">
            <button className="landing-primary" onClick={() => onStart("signup")}>Explore TruthLens <ArrowRight size={16}/></button>
            <a className="landing-secondary" href="#landing-features">Explore Features</a>
          </div>
          <div className="landing-trustline"><ShieldCheck size={16}/> Evidence-first analysis <span/> Human review required</div>
        </div>
        <div className="landing-hero-art" aria-label="TruthLens evidence analysis illustration">
          <div className="landing-orbit orbit-one"/><div className="landing-orbit orbit-two"/>
          <div className="landing-globe"><div className="globe-longitude"/><div className="globe-latitude"/></div>
          <div className="landing-shield"><ShieldCheck size={112} strokeWidth={1.5}/></div>
          <div className="landing-float-card float-left"><CheckCircle2 size={18}/><span><b>Evidence review</b><small>Image · Video · Audio</small></span></div>
          <div className="landing-float-card float-right"><Fingerprint size={18}/><span><b>SHA-256</b><small>Transfer integrity</small></span></div>
          <div className="landing-float-card float-bottom"><Activity size={18}/><span><b>Cross-modal analysis</b><small>Separate model indicators</small></span></div>
        </div>
      </section>
      <section className="landing-features" id="landing-features">
        <div className="landing-section-heading"><span>THE TRUTHLENS WORKFLOW</span><h2>Tools for a more careful review</h2><p>One workspace for examining digital evidence and documenting what the models can — and cannot — tell you.</p></div>
        <div className="landing-feature-grid">
          <article><span className="feature-symbol blue"><ImageIcon size={22}/></span><h3>Multimodal Analysis</h3><p>Review image, video-frame and speech-audio model indicators with their evidence and limitations.</p></article>
          <article><span className="feature-symbol purple"><Fingerprint size={22}/></span><h3>Evidence Integrity</h3><p>Compare browser and server SHA-256 digests and preserve a case-level evidence history.</p></article>
          <article><span className="feature-symbol amber"><Activity size={22}/></span><h3>Cross-modal Review</h3><p>Compare related audio and video signals, plus OCR, captions, transcripts and supplied claims.</p></article>
          <article><span className="feature-symbol green"><FileBarChart2 size={22}/></span><h3>Human Review & Reports</h3><p>Record investigator decisions, rationale and reviewed findings, then export a case report.</p></article>
        </div>
      </section>
      <section className="landing-about" id="landing-about">
        <div className="landing-about-copy"><span>ABOUT TRUTHLENS</span><h2>Built for evidence.<br/><em>Designed for clarity.</em></h2><p>TruthLens brings independent research-model indicators, extracted text and investigator decisions into one reviewable case. It helps analysts organize evidence without presenting automated scores as final truth.</p><button className="landing-primary" onClick={() => onStart("signup")}>Create your workspace <ArrowRight size={16}/></button></div>
        <div className="landing-about-visual"><div className="mini-app-window"><div className="mini-app-top"><span/><span/><span/><b>TRUTHLENS / CASE REVIEW</b></div><div className="mini-app-body"><div className="mini-app-sidebar"><i/><i/><i/><i/></div><div className="mini-app-content"><b>Evidence overview</b><div className="mini-app-stats"><i/><i/><i/></div><div className="mini-app-chart"><span/><span/><span/><span/><span/><span/><span/></div><div className="mini-app-footer"><i/><i/><i/></div></div></div></div>
          <div className="landing-fact"><b>8</b><span>files per investigation</span><b>40 MB</b><span>maximum per file</span><b>3</b><span>supported media families</span></div>
        </div>
      </section>
      <section className="landing-cta" id="landing-contact"><div><span>START WITH THE SOURCE</span><h2>Make every review more transparent.</h2><p>Keep the original evidence in view, understand model limits and document your assessment.</p></div><button className="landing-primary" onClick={() => onStart("signup")}>Get Started Free <ArrowRight size={16}/></button></section>
    </main>
    <footer className="landing-footer"><a className="landing-brand" href="#"><span className="landing-brand-mark"><BrandGlyph size={25}/></span><b>Truth<span>Lens</span></b></a><span>Investigate. Analyze. Verify.</span><small>Automated signals are not proof of authenticity.</small></footer>
  </div>;
}

function AuthScreen({ mode, setMode, email, setEmail, password, setPassword, displayName, setDisplayName, busy, error, onSubmit, onGoogle, onBack }) {
  const creating = mode === "signup";
  return <div className="auth-screen">
    <div className="auth-shell">
      <section className="auth-welcome-panel">
        <button className="auth-visual-brand" type="button" onClick={onBack}>
          <span><BrandGlyph size={31}/></span><b>Truth<span>Lens</span></b>
        </button>
        <div className="auth-welcome-copy">
          <span className="auth-eyebrow"><i/> DIGITAL FORENSICS WORKSPACE</span>
          <h1>{creating ? <>Build a clearer view of <em>the evidence.</em></> : <>See the evidence.<br/><em>Understand the signal.</em></>}</h1>
          <p>Bring image, video and audio analysis together with evidence integrity checks and investigator-led review.</p>
          <div className="auth-feature-list">
            <div><span><ImageIcon size={17}/></span><div><b>Multimodal evidence</b><small>Image · Video · Speech audio</small></div></div>
            <div><span><Fingerprint size={17}/></span><div><b>Evidence integrity</b><small>SHA-256 fingerprints and case history</small></div></div>
            <div><span><FileBarChart2 size={17}/></span><div><b>Human-led reporting</b><small>Review decisions and exportable reports</small></div></div>
          </div>
        </div>
        <div className="auth-visual-art" aria-hidden="true">
          <div className="auth-visual-orbit auth-orbit-one"/>
          <div className="auth-visual-orbit auth-orbit-two"/>
          <div className="auth-shield"><ShieldCheck size={88} strokeWidth={1.45}/></div>
          <div className="auth-floating-chip auth-chip-top"><CheckCircle2 size={15}/><span>Evidence review</span></div>
          <div className="auth-floating-chip auth-chip-bottom"><Fingerprint size={15}/><span>Integrity checked</span></div>
        </div>
        <div className="auth-panel-foot"><ShieldCheck size={14}/> Private analyst workspace <i/> Evidence-first. Human-reviewed.</div>
      </section>

      <section className="auth-card">
        <button type="button" className="auth-back" onClick={onBack}><ArrowRight size={13}/> Back to home</button>
        <div className="auth-mobile-brand"><span className="brand-mark"><BrandGlyph size={27}/></span><b>Truth<span>Lens</span></b></div>
        <div className="auth-heading">
          <span className="auth-form-kicker">{creating ? "CREATE YOUR WORKSPACE" : "WELCOME TO TRUTHLENS"}</span>
          <h2>{creating ? "Create your account" : "Welcome back"}</h2>
          <p>{creating ? "Set up your secure analyst workspace to get started." : "Sign in to continue to your evidence workspace."}</p>
        </div>
        <form onSubmit={onSubmit} className="auth-form">
          {creating && <label className="auth-field"><span>Full name</span><div className="auth-input"><UserRound size={17}/><input autoComplete="name" value={displayName} onChange={(event)=>setDisplayName(event.target.value)} placeholder="Enter your name" required/></div></label>}
          <label className="auth-field"><span>Email address</span><div className="auth-input"><Mail size={17}/><input type="email" autoComplete="email" value={email} onChange={(event)=>setEmail(event.target.value)} placeholder="you@example.com" required/></div></label>
          <label className="auth-field"><span>Password</span><div className="auth-input"><LockKeyhole size={17}/><input type="password" autoComplete={creating?"new-password":"current-password"} value={password} onChange={(event)=>setPassword(event.target.value)} placeholder="Enter your password" minLength={6} required/></div></label>
          {error && <div className="auth-error"><AlertTriangle size={15}/><span>{error}</span></div>}
          <button className="button button-primary auth-submit" type="submit" disabled={busy}>
            {busy ? <><LoaderCircle className="spin" size={16}/> Please wait…</> : <>{creating ? "Create account" : "Sign in to TruthLens"} <ArrowRight size={16}/></>}
          </button>
        </form>
        <div className="auth-divider"><span/><small>OR CONTINUE WITH</small><span/></div>
        <button className="button button-quiet auth-google" onClick={onGoogle} disabled={busy}><span className="google-g">G</span> Continue with Google</button>
        <p className="auth-switch">{creating ? "Already have an account?" : "New to TruthLens?"}
          <button onClick={()=>setMode(creating?"signin":"signup")}>{creating?"Sign in":"Create an account"}</button>
        </p>
        <div className="auth-privacy"><LockKeyhole size={13}/> Secure Firebase authentication <i/> Your case data stays private to your account</div>
      </section>
    </div>
    <div className="auth-footer">TRUTHLENS <span>·</span> INVESTIGATE. ANALYZE. VERIFY.</div>
  </div>;
}

export default function App() {
  const [user, setUser] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const [authMode, setAuthMode] = useState("signin");
  const [showAuth, setShowAuth] = useState(false);
  const [quickMode, setQuickMode] = useState("text");
  const [quickClaim, setQuickClaim] = useState("");
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
  const [audioTestMode, setAudioTestMode] = useState(false);
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
    if (!nextUser) setShowAuth(false);
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
  const evidenceMix = activityEvents.reduce((counts, event) => {
    (event.evidence || []).forEach((item) => {
      if (item.media_type in counts) counts[item.media_type] += 1;
    });
    return counts;
  }, { image: 0, video: 0, audio: 0 });
  const recentMaxEvidence = Math.max(1, ...activityEvents.slice(0, 6).map((event) => event.evidence_count || 0));

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
    setAudioTestMode(false);
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
  if (!user && !showAuth) return <LandingPage onStart={(mode) => { setAuthMode(mode); setAuthError(""); setShowAuth(true); }} />;
  if (!user) return <AuthScreen
    mode={authMode} setMode={(value) => { setAuthMode(value); setAuthError(""); }}
    email={authEmail} setEmail={setAuthEmail}
    password={authPassword} setPassword={setAuthPassword}
    displayName={authName} setDisplayName={setAuthName}
    busy={authBusy} error={!firebaseConfigured ? "Firebase is not configured. Add the client values to frontend/.env.local, then restart the frontend." : authError}
    onSubmit={submitAuth} onGoogle={continueWithGoogle} onBack={() => setShowAuth(false)}
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
        <div className="dashboard-welcome-row">
          <div><span className="dashboard-eyebrow">WORKSPACE / DASHBOARD</span><h1>Welcome back, {user.displayName || user.email?.split("@")[0] || "Analyst"}!</h1><p>Your evidence workspace at a glance. Start a new review or continue with a saved investigation.</p></div>
          <button className="dashboard-start-button" onClick={newInvestigation}><Plus size={17}/> Start Verifying</button>
        </div>

        <div className="dashboard-hero">
          <div className="dashboard-hero-copy">
            <span className="dashboard-hero-kicker"><i/> MULTIMODAL DIGITAL FORENSICS</span>
            <h2>Truth<span>Lens</span></h2>
            <h3>Verify. Analyze. Know the Truth.</h3>
            <p>Examine images, videos and speech with research-model indicators, evidence integrity checks and human-led review.</p>
            <button onClick={newInvestigation}>Start a new investigation <ArrowRight size={16}/></button>
          </div>
          <div className="dashboard-hero-art" aria-hidden="true">
            <div className="dashboard-world-grid"/><div className="dashboard-shield-orb"><ShieldCheck size={94}/></div>
            <div className="dashboard-doc-card"><span/><span/><span/><span/><b>CASE EVIDENCE</b></div>
            <div className="dashboard-hero-label"><Fingerprint size={18}/><span><b>SHA-256</b><small>Evidence fingerprint</small></span></div>
          </div>
        </div>

        <div className="dashboard-kpis">
          <article className="dashboard-kpi kpi-blue"><span className="kpi-icon"><ShieldCheck size={23}/></span><div><small>TOTAL CASES</small><b>{activitySummary.case_count}</b><em>Saved investigations</em></div><span className="kpi-trend">LIVE</span></article>
          <article className="dashboard-kpi kpi-green"><span className="kpi-icon"><CheckCircle2 size={23}/></span><div><small>SHA-256 MATCHES</small><b>{verifiedAssetsCount}</b><em>{integrityPercent}% of compared files</em></div><span className="kpi-trend">REAL</span></article>
          <article className="dashboard-kpi kpi-purple"><span className="kpi-icon"><Activity size={23}/></span><div><small>MODEL SIGNALS</small><b>{activitySummary.manipulation_signals}</b><em>Require analyst review</em></div><span className="kpi-trend">REVIEW</span></article>
          <article className="dashboard-kpi kpi-orange"><span className="kpi-icon"><FileImage size={23}/></span><div><small>MEDIA EVIDENCE</small><b>{reviewedAssets}</b><em>Across saved cases</em></div><span className="kpi-trend">FILES</span></article>
        </div>

        <div className="dashboard-main-grid">
          <section className="dashboard-panel quick-verify-panel">
            <div className="dashboard-panel-heading"><div><span>QUICK ACTION</span><h2>Verify Content</h2><p>Start a case using a claim or supporting media.</p></div><span className="dashboard-panel-icon"><Activity size={20}/></span></div>
            <div className="quick-mode-tabs">
              {["text","image","video","audio"].map((mode) => <button key={mode} className={quickMode===mode?"selected":""} onClick={()=>setQuickMode(mode)}>{mode[0].toUpperCase()+mode.slice(1)}</button>)}
            </div>
            {quickMode === "text" ? <textarea value={quickClaim} onChange={(event)=>setQuickClaim(event.target.value)} placeholder='Paste a claim, caption or context to compare with original evidence…' rows={3}/> :
              <label className="quick-evidence-picker"><input type="file" accept={quickMode==="image"?"image/*":quickMode==="video"?"video/*":".wav,.flac,.ogg,audio/*"} onChange={(event)=>{if(event.target.files?.length){addFiles([...event.target.files]);setPage("new");event.target.value="";}}}/><UploadCloud size={22}/><b>Add {quickMode} evidence</b><small>Choose a source file from your device</small></label>}
            <button className="quick-check-button" onClick={()=>{if(quickMode==="text"&&quickClaim.trim()){setContext(quickClaim.trim());setPage("new");}else if(quickMode==="text"){setPage("new");}}}>{quickMode==="text"?"Continue with this claim":"Open evidence intake"} <ArrowRight size={16}/></button>
            <small className="quick-verify-note">Text is saved as case context. Upload at least one original media file before running an analysis; TruthLens does not issue an automatic true/false verdict.</small>
          </section>

          <section className="dashboard-panel verification-results-panel">
            <div className="dashboard-panel-heading"><div><span>CASE SNAPSHOT</span><h2>Latest Verification Results</h2><p>Most recent saved investigation</p></div><button className="dashboard-link" onClick={()=>setPage("history")}>View cases <ArrowRight size={14}/></button></div>
            {latestReports[0] ? <div className="latest-result-content">
              <div className="latest-result-status"><span><CheckCircle2 size={26}/></span><div><b>{latestReports[0].case_title || "Investigation completed"}</b><small>{latestReports[0].assets.length} evidence item(s) · {formatDate(latestReports[0].created_at)}</small></div></div>
              <div className="latest-evidence-list">{latestReports[0].assets.slice(0,3).map((asset,index)=><div key={asset.asset_id || index}><span>{typeIcon(asset.media_type,15)} {asset.filename}</span><b className={asset.model_assessment?.prediction==="fake_signal"?"review-signal":""}>{predictionLabel(asset.model_assessment?.prediction)}</b></div>)}</div>
              <button className="latest-open-button" onClick={()=>openReport(latestReports[0])}>Open full report <ArrowRight size={14}/></button>
            </div> : <div className="dashboard-empty-result"><ShieldCheck size={34}/><b>No results yet</b><p>Run your first investigation to see its model indicators, source comparisons and report summary here.</p><button onClick={newInvestigation}>Create investigation <ArrowRight size={14}/></button></div>}
            <small className="dashboard-disclaimer">Model indicators are not calibrated probabilities or proof of authenticity. Review original evidence before drawing conclusions.</small>
          </section>
        </div>

        <div className="dashboard-bottom-grid">
          <section className="dashboard-panel analytics-panel">
            <div className="dashboard-panel-heading"><div><span>WORKSPACE ACTIVITY</span><h2>Analytics Overview</h2></div><small>Latest {Math.min(6,activityEvents.length)} cases</small></div>
            {activityEvents.length ? <div className="case-activity-bars">{activityEvents.slice(0,6).reverse().map((event,index)=><button key={event.event_id} onClick={()=>openReport(event.report_id)} title={event.case_title}><span className="activity-bar-label">{(event.case_title||"Case").slice(0,17)}</span><span className="activity-bar-track"><i style={{width:Math.max(8,((event.evidence_count||0)/recentMaxEvidence)*100)+"%"}}/></span><b>{event.evidence_count||0}</b></button>)}</div> : <div className="dashboard-mini-empty">Your saved case activity will appear here after your first analysis.</div>}
            <div className="analytics-legend"><span><i className="legend-blue"/> Evidence items per case</span><span><i className="legend-green"/> Live server activity</span></div>
          </section>

          <section className="dashboard-panel evidence-mix-panel">
            <div className="dashboard-panel-heading"><div><span>SUPPORTED INPUTS</span><h2>Evidence Mix</h2></div><span className="dashboard-panel-icon"><FileImage size={19}/></span></div>
            <div className="evidence-mix-total"><b>{reviewedAssets}</b><span>total media items</span></div>
            {[["image","Images",ImageIcon,"mix-blue"],["video","Videos",Video,"mix-purple"],["audio","Audio",AudioLines,"mix-green"]].map(([key,label,Icon,className])=><div className="evidence-mix-row" key={key}><span className={className}><Icon size={16}/></span><b>{label}</b><div><i style={{width:(reviewedAssets?evidenceMix[key]/reviewedAssets*100:0)+"%"}}/></div><strong>{evidenceMix[key]}</strong></div>)}
            <p className="dashboard-disclaimer">Counts are based on evidence recorded in the current account's activity feed.</p>
          </section>

          <section className="dashboard-panel recent-activity-panel">
            <div className="dashboard-panel-heading"><div><span>SERVER ACTIVITY</span><h2>Recent Activity</h2></div><button className="dashboard-link" onClick={()=>setPage("history")}>View all <ArrowRight size={14}/></button></div>
            {activityEvents.length ? <div className="dashboard-recent-list">{activityEvents.slice(0,4).map((event,index)=><button key={event.event_id} onClick={()=>openReport(event.report_id)}><span className={"recent-activity-icon recent-"+index}><CheckCircle2 size={15}/></span><span><b>{event.case_title || "Investigation completed"}</b><small>{event.evidence_count} evidence item(s) · {event.verified_count} digest match(es)</small></span><time>{formatDate(event.created_at)}</time></button>)}</div> : <div className="dashboard-mini-empty">{activityLoading?"Loading account activity…":activityError||"No recent cases have been recorded yet."}</div>}
          </section>
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
              <input type="file" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.mp4,.mov,.avi,.mkv,.webm,.wav,.flac,.ogg,.txt,.md"
                onChange={(event) => { addFiles([...event.target.files]); event.target.value = ""; }} />
              <span className="upload-symbol"><UploadCloud size={27} /></span><b>Drag evidence files into this area</b>
              <span>or <u>browse from device</u></span><small>Images · Videos · Speech audio · TXT / MD · Up to 40 MB per file</small>
            </label>
            {files.length > 0 && <div className="evidence-upload-list">
              <div className="upload-list-heading"><b>Selected evidence</b><div><span>{files.length} files · {formatBytes(totalSize)}</span><button className="text-button danger-text" onClick={clearFiles}><Trash2 size={13} /> Clear all</button></div></div>
              <div className="preview-grid">{files.map((item) => <div className="preview-card" key={item.id}>
                <div className="preview-media">
                  {item.type === "image" ? <img src={item.previewUrl} alt={item.file.name} />
                    : item.type === "video" ? <video src={item.previewUrl} controls preload="metadata" />
                    : item.type === "text" ? <div className="text-file-preview"><FileText size={34}/><b>Text evidence</b><small>UTF-8 · imported as a source</small></div>
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
            <div className="audio-test-panel">
              <div className="section-kicker">OPTIONAL / AUDIO MODEL VALIDATION</div>
              <h3>Compare an ElevenLabs sample with a human recording</h3>
              <p>Generate a short sample with ElevenLabs Text to Speech. Record a consenting speaker reading the exact same sentence. Upload both files in this case to compare their model indicators.</p>
              <ol>
                <li>Use the same 10–20 second sentence for both recordings.</li>
                <li>Name the ElevenLabs export ELEVENLABS_SYNTHETIC.wav.</li>
                <li>Name the human recording HUMAN_RECORDED.wav.</li>
                <li>Keep the language, sentence and recording conditions as similar as possible.</li>
                <li>Compare the transcript, model scores and segment outputs. Detection is not guaranteed.</li>
              </ol>
              <button type="button" className="button button-quiet" onClick={() => {
                setAudioTestMode(true);
                setCaseTitle("TruthLens · Synthetic vs Human Audio Validation");
                setContext("AUDIO VALIDATION TEST — Sample A: ElevenLabs-generated synthetic speech. Sample B: consenting human speaker reading the same script. Script: [paste exact sentence]. Purpose: compare research model indicators only; this does not establish speaker identity or prove voice cloning.");
              }}>Use audio validation case template</button>
              {audioTestMode && <p className="model-note" role="status">Template added. Replace the script placeholder and upload both labelled audio files.</p>}
              <p className="model-note">Use only voices you own or have permission to synthesize or record. ElevenLabs generates the test sample; TruthLens performs the analysis.</p>
            </div>
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
        {activeReport && <ReportView report={activeReport} user={user} onReviewsChange={(reviews) => setActiveReport((current) => current?.report_id === activeReport.report_id ? { ...current, investigator_reviews: reviews } : current)} onJson={() => downloadJson(activeReport)} onPdf={async () => {
          try {
            const token = await user.getIdToken();
            const response = await axios.get(API + "/api/reports/" + encodeURIComponent(activeReport.report_id) + "/reviews", {
              headers: { Authorization: "Bearer " + token }, timeout: 15000
            });
            const reportForExport = { ...activeReport, investigator_reviews: response.data.reviews || [] };
            if (!exportPdf(reportForExport)) setError("Allow pop-ups to open the printable PDF report.");
          } catch (err) {
            setError(err.response?.data?.detail || "Could not load saved investigator decisions. Please retry PDF export.");
          }
        }} />}
      </section>}

      {(page === "history" || page === "reports") && <section className="records-page">
        <div className="page-heading"><div><span className="section-kicker">{page === "history" ? "WORKSPACE / CASE REGISTER" : "WORKSPACE / REPORT LIBRARY"}</span>
          <h1>{page === "history" ? "Case history" : "Reports"}</h1><p>{page === "history" ? "Reopen saved investigations and continue reviewing the evidence." : "Open completed case reports or export an evidence summary."}</p></div>
          <button className="button button-primary" onClick={newInvestigation}><Plus size={16} /> New investigation</button></div>
        {activeReport && page === "reports" && <ReportView report={activeReport} user={user} onReviewsChange={(reviews) => setActiveReport((current) => current?.report_id === activeReport.report_id ? { ...current, investigator_reviews: reviews } : current)} onJson={() => downloadJson(activeReport)} onPdf={async () => {
          try {
            const token = await user.getIdToken();
            const response = await axios.get(API + "/api/reports/" + encodeURIComponent(activeReport.report_id) + "/reviews", {
              headers: { Authorization: "Bearer " + token }, timeout: 15000
            });
            const reportForExport = { ...activeReport, investigator_reviews: response.data.reviews || [] };
            if (!exportPdf(reportForExport)) setError("Allow pop-ups to open the printable PDF report.");
          } catch (err) {
            setError(err.response?.data?.detail || "Could not load saved investigator decisions. Please retry PDF export.");
          }
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
