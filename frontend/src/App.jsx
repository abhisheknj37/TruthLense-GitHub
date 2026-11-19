import { useMemo, useState } from "react";
import axios from "axios";
import {
  Activity, AudioLines, CheckCircle2, ChevronRight, Download, FileText,
  Fingerprint, Image as ImageIcon, LoaderCircle, ShieldAlert,
  ShieldCheck, UploadCloud, Video, X
} from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function formatBytes(bytes = 0) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function kindIcon(type) {
  if (type === "video") return <Video size={18} />;
  if (type === "audio") return <AudioLines size={18} />;
  return <ImageIcon size={18} />;
}

function predictionLabel(value = "inconclusive") {
  return ({
    fake_signal: "Manipulation-related model signal",
    real_signal: "Authenticity-related model signal",
    inconclusive: "Inconclusive",
  })[value] || value.replaceAll("_", " ");
}

function ScoreBar({ label, value }) {
  if (typeof value !== "number") return null;
  return (
    <div className="score-row">
      <div><span>{label}</span><b>{(value * 100).toFixed(1)}%</b></div>
      <div className="score-track"><i style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div>
    </div>
  );
}

function ModelAssessment({ asset }) {
  const model = asset.model_assessment || {};
  const video = asset.video_analysis || {};
  const details = asset.media_type === "video" ? video : model;
  const assessed = details.status === "model_assessed";
  return (
    <div className="model-result">
      <div className="model-result-head">
        <b>AI model assessment</b>
        <span className={assessed ? "model-pill available" : "model-pill pending"}>
          {assessed ? "MODEL SIGNAL AVAILABLE" : "INCONCLUSIVE"}
        </span>
      </div>
      <h4>{predictionLabel(details.prediction)}</h4>
      {details.model_name && <small className="model-name">{details.model_name}</small>}
      {details.scores && (
        <div className="score-list">
          {Object.entries(details.scores).filter(([key, value]) =>
            typeof value === "number" && (key.includes("fake") || key.includes("real"))
          ).map(([key, value]) => <ScoreBar key={key} label={key.replaceAll("_", " ")} value={value} />)}
        </div>
      )}
      {details.uncertainty_band && <p className="uncertainty">Score separation: {details.uncertainty_band.replaceAll("_", " ")} · not calibrated uncertainty</p>}
      {details.note && <p className="model-note">{details.note}</p>}

      {asset.media_type === "image" && model.faces?.length > 0 && (
        <div className="face-results">
          <b>Detected face regions</b>
          {model.faces.map((face) => (
            <div className="face-result" key={face.face_index}>
              <div className="face-meta">
                <span>Face {face.face_index + 1} · {predictionLabel(face.prediction)}</span>
                <small>Box: {face.bounding_box_xywh.join(", ")}</small>
              </div>
              {face.gradcam_overlay_base64
                ? <img className="gradcam-image" src={`data:image/png;base64,${face.gradcam_overlay_base64}`} alt="Grad-CAM style patch attribution overlay" />
                : <p className="model-note">Patch attribution was not produced for this face.</p>}
              <p className="cam-caption">Transformer patch attribution overlay. Highlighted regions show model sensitivity, not proof of manipulation.</p>
            </div>
          ))}
        </div>
      )}

      {asset.media_type === "video" && video.frames?.length > 0 && (
        <div className="frame-results">
          <b>Frame-level analysis</b>
          {video.frames.map((frame) => (
            <div className="frame-result" key={frame.frame_index}>
              <span>Frame {frame.frame_index}</span>
              <span>{frame.timestamp_seconds ?? "—"}s</span>
              <span>{predictionLabel(frame.prediction)}</span>
            </div>
          ))}
        </div>
      )}

      {asset.media_type === "audio" && model.segments?.length > 0 && (
        <div className="frame-results">
          <b>Speech-window analysis</b>
          {model.segments.map((segment, index) => (
            <div className="frame-result" key={index}>
              <span>{segment.start_seconds}s–{segment.end_seconds}s</span>
              <span>AI-voice score {(segment.fake_model_score * 100).toFixed(1)}%</span>
              <span>{segment.fake_model_score >= 0.65 ? "Signal" : segment.fake_model_score <= 0.35 ? "Low signal" : "Ambiguous"}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [context, setContext] = useState("");
  const [pairMedia, setPairMedia] = useState(false);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const totalSize = useMemo(() => files.reduce((sum, file) => sum + file.size, 0), [files]);

  function addFiles(selected) {
    setFiles((current) => {
      const next = [...current];
      for (const file of selected) {
        if (!next.some((item) => item.name === file.name && item.size === file.size)) next.push(file);
      }
      return next.slice(0, 8);
    });
    setReport(null);
    setError("");
  }

  async function analyze() {
    if (!files.length) return;
    setLoading(true);
    setError("");
    setReport(null);
    const data = new FormData();
    files.forEach((file) => data.append("files", file));
    data.append("context_text", context);
    data.append("pair_media", pairMedia ? "true" : "false");
    try {
      const response = await axios.post(`${API}/api/analyze`, data, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 600000
      });
      setReport(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  function downloadReport() {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `truthlens-report-${report.report_id.slice(0, 8)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Fingerprint size={22} /></div>
          <div><strong>TRUTHLENS</strong><span>Evidence intelligence</span></div>
        </div>
        <div className="side-label">WORKSPACE</div>
        <div className="nav-item active"><Activity size={17} /> New investigation</div>
        <div className="nav-item muted"><FileText size={17} /> Reports <span className="soon">MVP</span></div>
        <div className="sidebar-bottom">
          <div className="secure-icon"><ShieldCheck size={17} /></div>
          <div><strong>Evidence-first analysis</strong><p>Original files are not modified.</p></div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">Workspace <ChevronRight size={14} /> <b>New investigation</b></div>
          <div className="top-status"><span className="status-dot" /> LOCAL WORKSPACE</div>
        </header>

        <section className="intro">
          <div className="eyebrow"><span /> MULTIMODAL DIGITAL FORENSICS</div>
          <h1>Look closer.<br /><em>Verify the evidence.</em></h1>
          <p className="intro-copy">
            Analyze image, video and speech for manipulation-related signals.
            Every model output includes its scope and limitations for human review.
          </p>
        </section>

        <section className="workspace-grid">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div><span className="step">01 / INGEST</span><h2>Evidence files</h2></div>
              <span className="count">{files.length} / 8 files</span>
            </div>
            <label className="dropzone"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => { event.preventDefault(); addFiles([...event.dataTransfer.files]); }}>
              <input type="file" multiple accept="image/*,video/*,.wav,.flac,.ogg"
                onChange={(event) => addFiles([...event.target.files])} />
              <div className="upload-icon"><UploadCloud size={25} /></div>
              <strong>Drop evidence here</strong>
              <span>or <u>browse files</u></span>
              <small>Images · Video · WAV / FLAC / OGG speech · 40 MB each</small>
            </label>

            {files.length > 0 && <div className="file-list">
              {files.map((file, index) => (
                <div className="file-row" key={file.name + file.size}>
                  <div className="file-icon">{kindIcon(file.type.startsWith("video") ? "video" : file.type.startsWith("audio") ? "audio" : "image")}</div>
                  <div className="file-info"><strong>{file.name}</strong><span>{formatBytes(file.size)}</span></div>
                  <button className="icon-button" onClick={() => setFiles(files.filter((_, i) => i !== index))} aria-label="Remove file"><X size={16} /></button>
                </div>
              ))}
              <div className="total-row"><span>Total upload</span><b>{formatBytes(totalSize)}</b></div>
            </div>}

            <label className="context-label" htmlFor="context">CASE CONTEXT <span>OPTIONAL</span></label>
            <textarea id="context" value={context} onChange={(event) => setContext(event.target.value)}
              placeholder="Add source details, claimed context, transcript or notes for the reviewer…" rows={3} />

            <label className="pair-toggle">
              <input type="checkbox" checked={pairMedia} onChange={(event) => setPairMedia(event.target.checked)} />
              <span><b>These audio and video files belong to the same recording</b><small>Enables comparison of their separate model signals. It does not prove synchronization.</small></span>
            </label>

            <button className="analyze-button" disabled={!files.length || loading} onClick={analyze}>
              {loading ? <><LoaderCircle className="spin" size={18} /> Loading models / analyzing…</> : <><ShieldCheck size={18} /> Analyze evidence <ChevronRight size={17} /></>}
            </button>
            {loading && <p className="loading-note">First analysis may take longer while pretrained models download and load.</p>}
            {error && <div className="error-box"><ShieldAlert size={17} />{String(error)}</div>}
          </div>

          <div className="right-column">
            <div className="panel workflow-panel">
              <span className="step">02 / ANALYSIS PIPELINE</span>
              <h2>One case. Multiple signals.</h2>
              <div className="pipeline">
                {[
                  ["01", "Media inspection", "Hash, structure and metadata", <Fingerprint size={17} />],
                  ["02", "Modality analysis", "DeiT image · Wav2Vec2 audio", <Activity size={17} />],
                  ["03", "Frame-level analysis", "Sampled video frames + timestamps", <Video size={17} />],
                  ["04", "Evidence correlation", "Compare paired media signals", <AudioLines size={17} />]
                ].map(([n, title, sub, icon]) => (
                  <div className="pipeline-row" key={n}>
                    <span className="pipeline-number">{n}</span><span className="pipeline-icon">{icon}</span>
                    <div><strong>{title}</strong><small>{sub}</small></div><span className="pipeline-arrow"><ChevronRight size={16} /></span>
                  </div>
                ))}
              </div>
              <div className="notice"><ShieldAlert size={17} /><span><b>Research model notice:</b> Scores are uncalibrated indicators and may fail on unseen generators, languages, compression or media outside the model's training scope.</span></div>
            </div>
            <div className="mini-grid">
              <div className="mini-card"><ImageIcon size={19} /><strong>Image</strong><span>Face crop + patch map</span></div>
              <div className="mini-card"><Video size={19} /><strong>Video</strong><span>Sampled face frames</span></div>
              <div className="mini-card"><AudioLines size={19} /><strong>Audio</strong><span>Speech windows</span></div>
            </div>
          </div>
        </section>

        {report && <section className="report-section">
          <div className="report-title">
            <div><span className="step">03 / CASE REPORT · {report.report_id.slice(0, 8)}</span><h2>Forensic assessment</h2></div>
            <button className="download-button" onClick={downloadReport}><Download size={15} /> Export JSON</button>
          </div>
          <div className="report-summary">
            <strong>{report.summary}</strong>
            <p>Case status: {report.overall_status.replaceAll("_", " ")} · {report.human_review_required ? "Human review required" : "Review complete"}</p>
          </div>
          <div className="asset-grid">
            {report.assets.map((asset) => (
              <article className="asset-card" key={asset.asset_id}>
                <div className="asset-head"><span>{kindIcon(asset.media_type)} {asset.media_type.toUpperCase()}</span><small>{formatBytes(asset.size_bytes)}</small></div>
                <h3>{asset.filename}</h3>
                <div className="metadata-grid">
                  {Object.entries(asset.metadata).filter(([key, value]) => typeof value !== "object" && value !== null).slice(0, 6).map(([key, value]) => (
                    <div key={key}><small>{key.replaceAll("_", " ")}</small><b>{String(value)}</b></div>
                  ))}
                </div>
                <div className="hash-line"><small>SHA-256 EVIDENCE FINGERPRINT</small><code>{asset.sha256}</code></div>
                <ModelAssessment asset={asset} />
                {asset.observations.map((item) => (
                  <div className="observation" key={item.code}><CheckCircle2 size={16} /><div><b>{item.title}</b><p>{item.detail}</p></div></div>
                ))}
              </article>
            ))}
          </div>
          {report.cross_modal_review?.length > 0 && <div className="cross-review">
            <b>Cross-modal evidence review</b>
            {report.cross_modal_review.map((item, index) => (
              <div className="cross-item" key={index}><span className="model-pill pending">{item.status.replaceAll("_", " ")}</span><p>{item.detail}</p></div>
            ))}
          </div>}
          <div className="limitations"><b>Limitations & review notes</b>{report.limitations.map((item) => <p key={item}>— {item}</p>)}</div>
        </section>}
        <footer>TRUTHLENS <span>·</span> Evidence context matters. Automated indicators are not final conclusions.</footer>
      </main>
    </div>
  );
}
