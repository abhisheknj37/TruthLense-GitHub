import { useMemo, useState } from "react";
import axios from "axios";
import {
  Activity, AudioLines, CheckCircle2, ChevronRight, FileText,
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

export default function App() {
  const [files, setFiles] = useState([]);
  const [context, setContext] = useState("");
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const totalSize = useMemo(
    () => files.reduce((sum, file) => sum + file.size, 0),
    [files]
  );

  function addFiles(selected) {
    setFiles((current) => {
      const next = [...current];
      for (const file of selected) {
        if (!next.some((item) => item.name === file.name && item.size === file.size)) {
          next.push(file);
        }
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
    try {
      const response = await axios.post(`${API}/api/analyze`, data, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000
      });
      setReport(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
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
            Inspect image, video and audio files through a traceable analysis workflow.
            Findings are observations for human review—not automatic proof of authenticity.
          </p>
        </section>

        <section className="workspace-grid">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div><span className="step">01 / INGEST</span><h2>Evidence files</h2></div>
              <span className="count">{files.length} / 8 files</span>
            </div>
            <label
              className="dropzone"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                addFiles([...event.dataTransfer.files]);
              }}
            >
              <input
                type="file"
                multiple
                accept="image/*,video/*,.wav,.flac,.ogg"
                onChange={(event) => addFiles([...event.target.files])}
              />
              <div className="upload-icon"><UploadCloud size={25} /></div>
              <strong>Drop evidence here</strong>
              <span>or <u>browse files</u></span>
              <small>Images · Video · WAV / FLAC / OGG audio · 40 MB each</small>
            </label>

            {files.length > 0 && (
              <div className="file-list">
                {files.map((file, index) => (
                  <div className="file-row" key={file.name + file.size}>
                    <div className="file-icon">{kindIcon(file.type.startsWith("video") ? "video" : file.type.startsWith("audio") ? "audio" : "image")}</div>
                    <div className="file-info"><strong>{file.name}</strong><span>{formatBytes(file.size)}</span></div>
                    <button className="icon-button" onClick={() => setFiles(files.filter((_, i) => i !== index))} aria-label="Remove file"><X size={16} /></button>
                  </div>
                ))}
                <div className="total-row"><span>Total upload</span><b>{formatBytes(totalSize)}</b></div>
              </div>
            )}

            <label className="context-label" htmlFor="context">CASE CONTEXT <span>OPTIONAL</span></label>
            <textarea
              id="context"
              value={context}
              onChange={(event) => setContext(event.target.value)}
              placeholder="Add source details, claimed context, transcript or notes for the reviewer…"
              rows={3}
            />
            <button className="analyze-button" disabled={!files.length || loading} onClick={analyze}>
              {loading ? <><LoaderCircle className="spin" size={18} /> Inspecting evidence…</> : <><ShieldCheck size={18} /> Analyze evidence <ChevronRight size={17} /></>}
            </button>
            {error && <div className="error-box"><ShieldAlert size={17} />{String(error)}</div>}
          </div>

          <div className="right-column">
            <div className="panel workflow-panel">
              <span className="step">02 / ANALYSIS PIPELINE</span>
              <h2>One case. Multiple signals.</h2>
              <div className="pipeline">
                {[
                  ["01", "Media inspection", "File structure and metadata", <Fingerprint size={17} />],
                  ["02", "Modality analysis", "Image · video · audio", <Activity size={17} />],
                  ["03", "Evidence correlation", "Compare related inputs", <AudioLines size={17} />],
                  ["04", "Review report", "Traceable observations", <FileText size={17} />]
                ].map(([n, title, sub, icon]) => (
                  <div className="pipeline-row" key={n}>
                    <span className="pipeline-number">{n}</span><span className="pipeline-icon">{icon}</span>
                    <div><strong>{title}</strong><small>{sub}</small></div><span className="pipeline-arrow"><ChevronRight size={16} /></span>
                  </div>
                ))}
              </div>
              <div className="notice"><ShieldAlert size={17} /><span><b>Model status:</b> Authenticity models are not connected yet. This MVP reports technical observations only.</span></div>
            </div>
            <div className="mini-grid">
              <div className="mini-card"><ImageIcon size={19} /><strong>Image</strong><span>Metadata inspection</span></div>
              <div className="mini-card"><Video size={19} /><strong>Video</strong><span>Frame sampling</span></div>
              <div className="mini-card"><AudioLines size={19} /><strong>Audio</strong><span>Signal inspection</span></div>
            </div>
          </div>
        </section>

        {report && <section className="report-section">
          <div className="report-title">
            <div><span className="step">03 / CASE REPORT · {report.report_id.slice(0, 8)}</span><h2>Forensic observations</h2></div>
            <span className="inconclusive"><ShieldAlert size={15} /> INCONCLUSIVE</span>
          </div>
          <div className="report-summary">
            <strong>{report.summary}</strong>
            <p>{report.human_review_required ? "Human review required" : "Review complete"}</p>
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
                <div className="hash-line"><small>SHA-256 EVIDENCE FINGERPRINT</small><code>{asset.sha256}</code></div><div className="model-state"><span className="state-dot" /> MODEL NOT CONFIGURED</div>
                {asset.observations.map((item) => <div className="observation" key={item.code}><CheckCircle2 size={16} /><div><b>{item.title}</b><p>{item.detail}</p></div></div>)}
                {asset.sampled_frames?.length > 0 && (
                  <div className="frames"><b>Sampled frames</b><div>{asset.sampled_frames.map((frame) => <span key={frame.frame_index}>#{frame.frame_index} · {frame.timestamp_seconds ?? "—"}s</span>)}</div></div>
                )}
              </article>
            ))}
          </div>
          {report.cross_modal_review?.length > 0 && <div className="cross-review"><b>Cross-modal review notes</b>{report.cross_modal_review.map((item, i) => <p key={i}>{item.detail}</p>)}</div>}
          <div className="limitations"><b>Limitations & review notes</b>{report.limitations.map((item) => <p key={item}>— {item}</p>)}</div>
        </section>}
        <footer>TRUTHLENS <span>·</span> Evidence context matters. Automated indicators are not final conclusions.</footer>
      </main>
    </div>
  );
}
