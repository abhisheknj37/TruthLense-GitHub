import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Activity, AlertTriangle, ArrowRight, AudioLines, Bell, CalendarDays,
  Check, CheckCircle2, ChevronRight, Clock3, Download, FileBarChart2,
  FileImage, FileJson, Fingerprint, FolderOpen, Headphones, Image as ImageIcon,
  LayoutDashboard, LoaderCircle, Mic, Plus, Printer, Search, Settings,
  Shield, ShieldAlert, ShieldCheck, Trash2, UploadCloud, Video, X
} from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const STORAGE_KEY = "truthlens.saved-investigations.v1";
const MAX_FILES = 8;
const MAX_BYTES = 40 * 1024 * 1024;
const ALLOWED = {
  image: ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
  video: ["mp4", "mov", "avi", "mkv", "webm"],
  audio: ["wav", "flac", "ogg"]
};

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
    <div className="hash-line"><small><Fingerprint size={12} /> SHA-256 EVIDENCE FINGERPRINT</small><code>{asset.sha256}</code></div>
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

export default function App() {
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
  const [savedReports, setSavedReports] = useState(() => {
    try { return cleanReports(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]")); }
    catch { return []; }
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(savedReports));
    } catch {
      // Grad-CAM overlays can be large. Keep the latest reports and omit
      // embedded overlays only in the storage fallback; explicit exports retain them.
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
        localStorage.setItem(STORAGE_KEY, JSON.stringify(compact(savedReports.slice(0, 10))));
        setNotice("Case history was saved in compact mode to fit browser storage. JSON/PDF exports remain available.");
      } catch {
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(compact(savedReports.slice(0, 3))));
          setNotice("Browser storage is limited. Only the three most recent compact case reports were retained locally.");
        } catch {
          setNotice("Browser storage is full. Export important cases as JSON or PDF before closing this session.");
        }
      }
    }
  }, [savedReports]);
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
      const response = await axios.post(API + "/api/analyze", data, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 600000,
        onUploadProgress: (event) => {
          if (event.total) setUploadProgress(Math.round((event.loaded / event.total) * 100));
        }
      });
      const report = {
        ...response.data,
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
      const response = await axios.get(API + "/health", { timeout: 8000 });
      setApiStatus("online");
      setApiMessage(response.data.note || "Backend is responding.");
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

  const nav = [
    ["dashboard", "Dashboard", <LayoutDashboard size={17} />],
    ["new", "New investigation", <Plus size={17} />],
    ["history", "Case history", <FolderOpen size={17} />],
    ["reports", "Reports", <FileBarChart2 size={17} />],
    ["settings", "Settings", <Settings size={17} />]
  ];

  return <div className="app-shell">
    <aside className="sidebar">
      <button className="brand" onClick={() => setPage("dashboard")}>
        <span className="brand-mark"><Fingerprint size={22} /></span>
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
        <div className="profile-avatar">FA</div><div className="profile-copy"><b>Forensic analyst</b><small>Workspace member</small></div>
        <button className="profile-settings" onClick={() => setPage("settings")} aria-label="Profile settings"><Settings size={15} /></button>
      </div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div className="breadcrumb"><span>Workspace</span><ChevronRight size={13} /><b>{nav.find((item) => item[0] === page)?.[1] || "Dashboard"}</b></div>
        <div className="topbar-actions"><span className={"api-indicator " + apiStatus}><i /> {apiStatus === "online" ? "API ONLINE" : apiStatus === "offline" ? "API OFFLINE" : "LOCAL WORKSPACE"}</span>
          <button className="top-icon" onClick={checkApi} title="Check backend connection"><Activity size={17} /></button>
          <button className="top-icon" onClick={() => setPage("settings")} title="Settings"><Bell size={17} /></button>
          <span className="top-avatar">FA</span>
        </div>
      </header>

      {page === "dashboard" && <section className="dashboard-page">
        <div className="welcome-row"><div><span className="section-kicker">EVIDENCE INTELLIGENCE WORKSPACE</span>
          <h1>Investigate with clarity.<br /><em>Stay anchored to evidence.</em></h1>
          <p>Bring media into one review workspace. Inspect model signals, preserve evidence fingerprints and keep the analyst in control.</p>
          </div><div className="welcome-art"><div className="art-orbit orbit-one" /><div className="art-orbit orbit-two" /><Fingerprint size={78} /></div>
        </div>
        <div className="dashboard-actions"><button className="button button-primary" onClick={newInvestigation}><Plus size={17} /> Start new investigation <ArrowRight size={16} /></button>
          <button className="button button-quiet" onClick={() => setPage("history")}><FolderOpen size={16} /> Browse case history</button></div>
        <div className="stats-grid">
          <div className="stat-card"><span><FolderOpen size={16} /> SAVED CASES</span><b>{savedReports.length}</b><small>Stored in this browser</small></div>
          <div className="stat-card"><span><FileImage size={16} /> EVIDENCE ITEMS</span><b>{reviewedAssets}</b><small>Across saved investigations</small></div>
          <div className="stat-card"><span><Activity size={16} /> MODEL ASSESSMENTS</span><b>{savedReports.reduce((sum, item) => sum + item.assets.filter((asset) => asset.model_assessment?.status === "model_assessed").length, 0)}</b><small>Signals returned by models</small></div>
          <div className="stat-card"><span><ShieldCheck size={16} /> REVIEW STATUS</span><b>Human-led</b><small>Automated outputs need review</small></div>
        </div>
        <div className="dashboard-lower">
          <div className="content-panel recent-panel">
            <div className="panel-title"><div><span className="section-kicker">RECENT ACTIVITY</span><h2>Recent investigations</h2></div>
              <button className="text-button" onClick={() => setPage("history")}>View all <ArrowRight size={14} /></button></div>
            {latestReports.length ? latestReports.slice(0, 4).map((report) => <CaseRow key={report.report_id} report={report} onOpen={() => openReport(report)} />)
              : <div className="empty-state"><div className="empty-icon"><FolderOpen size={22} /></div><b>No investigations yet</b><p>Your completed cases will appear here after the first analysis.</p><button className="button button-quiet" onClick={newInvestigation}><Plus size={15} /> Create first case</button></div>}
          </div>
          <div className="content-panel pipeline-overview">
            <span className="section-kicker">HOW TRUTHLENS WORKS</span><h2>Four stages. One evidence trail.</h2>
            <div className="overview-step"><span>01</span><div><b>Media inspection</b><small>File type, metadata and fingerprint</small></div></div>
            <div className="overview-step"><span>02</span><div><b>Modality analysis</b><small>Image, video frame and speech signals</small></div></div>
            <div className="overview-step"><span>03</span><div><b>Evidence correlation</b><small>Compare only user-confirmed paired media</small></div></div>
            <div className="overview-step"><span>04</span><div><b>Human review & reporting</b><small>Interpret findings and export evidence</small></div></div>
            <div className="safety-note"><Shield size={15} /> Model scores are not proof of authenticity.</div>
          </div>
        </div>
      </section>}

      {page === "new" && <section className="investigation-page">
        <div className="page-heading"><div><span className="section-kicker">CASE WORKSPACE / INGEST</span><h1>New investigation</h1><p>Upload original evidence files to begin a multimodal review.</p></div>
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

      {page === "settings" && <section className="settings-page">
        <div className="page-heading"><div><span className="section-kicker">WORKSPACE / CONFIGURATION</span><h1>Settings</h1><p>Manage this local workspace and verify its API connection.</p></div></div>
        <div className="settings-grid">
          <div className="content-panel settings-card"><span className="section-kicker">WORKSPACE PROFILE</span><h2>Workspace identity</h2><label className="field-row"><span>Workspace name</span><input value={workspaceName} onChange={(event) => setWorkspaceName(event.target.value)} /></label>
            <div className="profile-summary"><span className="profile-avatar large">FA</span><div><b>Forensic analyst</b><small>Local workspace profile</small></div></div></div>
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
