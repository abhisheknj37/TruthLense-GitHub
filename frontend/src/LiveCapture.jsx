import { useEffect, useRef, useState } from "react";
import { AlertTriangle, AudioLines, Camera, CheckCircle2, Mic, Radio, Shield, StopCircle, Video } from "lucide-react";

function audioBufferToWav(buffer) {
  const channels = buffer.numberOfChannels;
  const dataLength = buffer.length * channels * 2;
  const output = new ArrayBuffer(44 + dataLength);
  const view = new DataView(output);
  const write = (offset, value) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, dataLength, true);
  let offset = 44;
  for (let i = 0; i < buffer.length; i++) {
    for (let channel = 0; channel < channels; channel++) {
      const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return output;
}

function errorText(error, device) {
  if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
    return `${device} permission was denied. Allow it in the browser's site settings and try again.`;
  }
  if (error?.name === "NotFoundError" || error?.name === "DevicesNotFoundError") {
    return `No ${device.toLowerCase()} device was found. Connect one and try again.`;
  }
  if (error?.name === "NotReadableError" || error?.name === "TrackStartError") {
    return `${device} is busy or unavailable. Close other apps using it and try again.`;
  }
  if (!window.isSecureContext) {
    return "Live capture requires HTTPS or localhost. Open this app through http://localhost:5173 or its HTTPS deployment.";
  }
  return error?.message || `Unable to access the ${device.toLowerCase()}.`;
}

export default function LiveCapture({ onCapture }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [mode, setMode] = useState("photo");
  const [streaming, setStreaming] = useState(false);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  function stopStream() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStreaming(false);
  }

  useEffect(() => () => {
    if (recorderRef.current?.state && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  async function startCamera() {
    setError("");
    setMessage("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not provide camera access. Use a current browser over HTTPS or localhost.");
      return;
    }
    try {
      stopStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: mode === "video",
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStreaming(true);
    } catch (err) {
      setError(errorText(err, "Camera"));
    }
  }

  async function startMicrophone() {
    setError("");
    setMessage("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not provide microphone access. Use a current browser over HTTPS or localhost.");
      return;
    }
    try {
      stopStream();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setStreaming(true);
      setMessage("Microphone connected. Start recording when ready.");
    } catch (err) {
      setError(errorText(err, "Microphone"));
    }
  }

  function capturePhoto() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setError("Camera preview is not ready yet. Wait for the video preview and try again.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      setError("Could not prepare the captured image.");
      return;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setError("Could not capture the photo. Please try again.");
        return;
      }
      const file = new File([blob], `live-capture-photo-${Date.now()}.jpg`, {
        type: "image/jpeg", lastModified: Date.now()
      });
      onCapture([file]);
      setMessage("Photo captured and added to the evidence queue.");
      setError("");
    }, "image/jpeg", 0.95);
  }

  function startRecording(type) {
    const stream = streamRef.current;
    if (!stream) {
      setError("Enable the camera or microphone before starting a recording.");
      return;
    }
    if (!window.MediaRecorder) {
      setError("This browser does not support MediaRecorder. Try the latest Chrome or Edge.");
      return;
    }
    const choices = type === "video"
      ? ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"]
      : ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"];
    const mimeType = choices.find((type) => MediaRecorder.isTypeSupported(type));
    if (!mimeType) {
      setError(`This browser does not support a compatible ${type} recording format.`);
      return;
    }
    try {
      chunksRef.current = [];
      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data?.size) chunksRef.current.push(event.data);
      };
      recorder.onerror = () => setError("The browser reported a recording error. Try again.");
      recorder.onstop = async () => {
        setBusy(true);
        try {
          const blob = new Blob(chunksRef.current, { type: mimeType });
          if (!blob.size) throw new Error("The recording is empty. Try recording again.");
          let file;
          if (type === "audio") {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) throw new Error("Audio conversion is not supported by this browser.");
            const audioContext = new AudioContextClass();
            try {
              const decoded = await audioContext.decodeAudioData(await blob.arrayBuffer());
              const wav = audioBufferToWav(decoded);
              file = new File([wav], `live-capture-audio-${Date.now()}.wav`, {
                type: "audio/wav", lastModified: Date.now()
              });
            } finally {
              await audioContext.close();
            }
          } else {
            file = new File([blob], `live-capture-video-${Date.now()}.webm`, {
              type: "video/webm", lastModified: Date.now()
            });
          }
          onCapture([file]);
          setMessage(`${type === "audio" ? "Audio" : "Video"} recording added to the evidence queue.`);
          setError("");
        } catch (err) {
          setError(err.message || "Could not prepare the recording.");
        } finally {
          setBusy(false);
          stopStream();
          setRecording(false);
          chunksRef.current = [];
        }
      };
      recorder.start(1000);
      setRecording(true);
      setMessage("Recording in progress. Stop it when you have captured the evidence.");
      setError("");
    } catch (err) {
      setError(err.message || "Could not start the recording.");
    }
  }

  function stopRecording() {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }

  function changeMode(next) {
    if (recording || busy) return;
    stopStream();
    setMode(next);
    setError("");
    setMessage("");
  }

  return <section className="live-capture-panel">
    <div className="live-capture-heading">
      <div><span className="section-kicker">02 / LIVE EVIDENCE ACQUISITION</span><h2>Capture from device</h2>
        <p>Capture a photo or record a video/audio file directly from your camera or microphone.</p></div>
      <span className="live-badge"><Radio size={13} /> LIVE CAPTURE</span>
    </div>
    <div className="capture-tabs">
      <button type="button" className={mode === "photo" ? "active" : ""} disabled={recording || busy} onClick={() => changeMode("photo")}><Camera size={16} /> Photo</button>
      <button type="button" className={mode === "video" ? "active" : ""} disabled={recording || busy} onClick={() => changeMode("video")}><Video size={16} /> Video</button>
      <button type="button" className={mode === "audio" ? "active" : ""} disabled={recording || busy} onClick={() => changeMode("audio")}><Mic size={16} /> Audio</button>
    </div>
    <div className="capture-workspace">
      <div className="capture-preview">
        {mode !== "audio"
          ? <video ref={videoRef} muted playsInline autoPlay />
          : <div className="audio-capture-placeholder"><AudioLines size={38} /><b>Microphone recording</b><small>Audio is captured only after you allow microphone access.</small></div>}
        {!streaming && mode !== "audio" && <div className="capture-placeholder"><Camera size={30} /><b>Camera preview</b><small>Start the camera to see a live preview.</small></div>}
        {recording && <span className="recording-indicator"><i /> RECORDING</span>}
        {busy && <span className="recording-indicator"><i /> PREPARING FILE</span>}
      </div>
      <div className="capture-controls">
        {!streaming && <button type="button" className="button button-quiet" onClick={mode === "audio" ? startMicrophone : startCamera} disabled={busy}>
          {mode === "audio" ? <Mic size={16} /> : <Camera size={16} />} {mode === "audio" ? "Enable microphone" : "Enable camera"}
        </button>}
        {streaming && mode === "photo" && <button type="button" className="button button-primary" onClick={capturePhoto}><Camera size={16} /> Capture photo</button>}
        {streaming && mode === "video" && !recording && <button type="button" className="button button-primary" onClick={() => startRecording("video")}><Video size={16} /> Start recording</button>}
        {streaming && mode === "audio" && !recording && <button type="button" className="button button-primary" onClick={() => startRecording("audio")}><Mic size={16} /> Start recording</button>}
        {recording && <button type="button" className="button button-danger" onClick={stopRecording}><StopCircle size={16} /> Stop recording</button>}
        {streaming && !recording && <button type="button" className="button button-quiet" onClick={stopStream}>Stop device</button>}
      </div>
    </div>
    {error && <div className="error-box"><AlertTriangle size={15} />{error}</div>}
    {message && <div className="success-box"><CheckCircle2 size={15} />{message}</div>}
    <p className="capture-privacy"><Shield size={13} /> Camera and microphone start only after you click Enable. Captured files join the evidence queue; review them before analysis.</p>
  </section>;
}
