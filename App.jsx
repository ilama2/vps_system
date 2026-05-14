import { useState, useEffect, useRef } from "react";

const THEME = {
  bg: "#070A12",
  bgSoft: "#0B1020",
  panel: "rgba(18,24,42,0.92)",
  border: "rgba(125,92,255,0.22)",

  primary: "#7C5CFF",
  secondary: "#38BDF8",
  accent: "#F472B6",

  text: "#F8FAFC",
  textSoft: "#CBD5E1",
  textMuted: "#64748B",

  success: "#22C55E",
  warning: "#F59E0B",
  error: "#EF4444",

  grid: "rgba(124,92,255,0.06)",
};

function PointCloud() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let animId;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    resize();

    const pts = Array.from({ length: 800 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      z: Math.random(),
      vx: (Math.random() - 0.5) * 0.18,
      vy: (Math.random() - 0.5) * 0.18,
      color: Math.floor(Math.random() * 3),
    }));

    const colors = [
      [124, 92, 255],
      [56, 189, 248],
      [244, 114, 182],
    ];

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const p of pts) {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        const [r, g, b] = colors[p.color];

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.z * 1.7 + 0.4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${0.18 + p.z * 0.4})`;
        ctx.fill();
      }

      animId = requestAnimationFrame(draw);
    }

    draw();

    window.addEventListener("resize", resize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return <canvas className="point-cloud" ref={canvasRef} />;
}

function Spinner() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" className="spinner">
      <circle
        cx="9"
        cy="9"
        r="7"
        fill="none"
        stroke={THEME.primary}
        strokeWidth="2"
        strokeDasharray="30 14"
      />
    </svg>
  );
}

function MetricRow({ label, value, accent }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong style={{ color: accent || THEME.primary }}>{value}</strong>
    </div>
  );
}

function interpretOrientation(qw, qx, qy, qz) {
  const m02 = 2 * (qx * qz + qy * qw);
  const m10 = 2 * (qx * qy + qz * qw);
  const m11 = 1 - 2 * (qx * qx + qz * qz);
  const m12 = 2 * (qy * qz - qx * qw);
  const m22 = 1 - 2 * (qx * qx + qy * qy);

  const yaw = (Math.atan2(m02, m22) * 180) / Math.PI;
  const pitch =
    (Math.atan2(-m12, Math.sqrt(m02 * m02 + m22 * m22)) * 180) / Math.PI;
  const roll = (Math.atan2(m10, m11) * 180) / Math.PI;

  return {
    yaw,
    pitch,
    roll,
    horizontal:
      yaw > 10
        ? `Facing right ${yaw.toFixed(1)}°`
        : yaw < -10
        ? `Facing left ${Math.abs(yaw).toFixed(1)}°`
        : "Facing forward",
    vertical:
      pitch > 10
        ? `Tilting up ${pitch.toFixed(1)}°`
        : pitch < -10
        ? `Tilting down ${Math.abs(pitch).toFixed(1)}°`
        : "Level vertically",
    rollText:
      roll > 10
        ? `Rolling right ${roll.toFixed(1)}°`
        : roll < -10
        ? `Rolling left ${Math.abs(roll).toFixed(1)}°`
        : "No noticeable roll",
  };
}

function interpretPosition(tx, ty, tz) {
  return {
    x:
      tx > 0.5
        ? `Right ${Math.abs(tx).toFixed(2)}m`
        : tx < -0.5
        ? `Left ${Math.abs(tx).toFixed(2)}m`
        : "Centered",
    y:
      ty > 0.5
        ? `Up ${Math.abs(ty).toFixed(2)}m`
        : ty < -0.5
        ? `Down ${Math.abs(ty).toFixed(2)}m`
        : "Level",
    z:
      tz > 0.5
        ? `Forward ${Math.abs(tz).toFixed(2)}m`
        : tz < -0.5
        ? `Backward ${Math.abs(tz).toFixed(2)}m`
        : "Near origin",
  };
}

function confidenceLevel(percent) {
  if (percent === undefined || percent === null) {
    return { label: "No confidence", color: THEME.textMuted };
  }

  if (percent >= 75) return { label: "Strong Match", color: THEME.success };
  if (percent >= 50) return { label: "Moderate Match", color: THEME.secondary };
  if (percent >= 30) return { label: "Weak Match", color: THEME.warning };

  return { label: "Poor Match", color: THEME.error };
}

function sceneStatus(percent) {
  if (percent >= 75) return "Matched to mapped environment";
  if (percent >= 50) return "Likely inside mapped area";
  if (percent >= 30) return "Uncertain scene match";
  return "Possibly outside mapped area";
}


export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [aiExplanation, setAiExplanation] = useState("");
  const [aiDiagnostics, setAiDiagnostics] = useState([]);

  const pose = result?.pose;
  const percent = pose?.confidence_percent ?? 0;
  const quality = confidenceLevel(percent);

  const orientation = pose?.quaternion
    ? interpretOrientation(
        pose.quaternion.qw,
        pose.quaternion.qx,
        pose.quaternion.qy,
        pose.quaternion.qz
      )
    : null;

  const position = pose?.translation
    ? interpretPosition(
        pose.translation.tx,
        pose.translation.ty,
        pose.translation.tz
      )
    : null;

  async function generateExplanation(
    poseData,
    positionData,
    orientationData,
    confidencePercent
    ) {
    if (!poseData || !positionData || !orientationData) return;
    
    setAiExplanation("Generating AI explanation...");
    setAiDiagnostics([]);
    
    try {
      const response = await fetch("http://127.0.0.1:8000/explain_pose", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
            confidence_percent: confidencePercent,
            confidence_raw: poseData.confidence_raw,
            translation: poseData.translation,
            quaternion: poseData.quaternion,
            orientation: orientationData,
            position: positionData,
            scene_status: sceneStatus(confidencePercent),
        }),
    });
    
        const data = await response.json();
    
        setAiExplanation(
          data.explanation || "No AI explanation returned."
        );
    
        setAiDiagnostics(data.diagnostics || []);
    
      } catch (error) {
        setAiExplanation(
          "AI explanation could not be generated."
        );
    
        setAiDiagnostics([
          "AI diagnostics unavailable."
        ]);
      }
  }

  async function handleUpload() {
    if (!image) return;

    setLoading(true);
    setResult(null);
    setAiExplanation("");

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://127.0.0.1:8000/localize", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);

      if (data.pose?.quaternion && data.pose?.translation) {
        const tempOrientation = interpretOrientation(
          data.pose.quaternion.qw,
          data.pose.quaternion.qx,
          data.pose.quaternion.qy,
          data.pose.quaternion.qz
        );

        const tempPosition = interpretPosition(
          data.pose.translation.tx,
          data.pose.translation.ty,
          data.pose.translation.tz
        );

        await generateExplanation(
          data.pose,
          tempPosition,
          tempOrientation,
          data.pose.confidence_percent ?? 0
        );
      }
    } catch (error) {
      setResult({
        success: false,
        inference_time: 0,
        pose: null,
        stderr: String(error),
      });

      setAiExplanation("");
    }

    setLoading(false);
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(14px); }
          to { opacity: 1; transform: translateY(0); }
        }

        * {
          box-sizing: border-box;
        }

        html,
        body,
        #root {
          margin: 0;
          padding: 0;
          width: 100%;
          overflow-x: hidden;
        }

        body {
          background: ${THEME.bg};
          color: ${THEME.text};
          font-family: 'Space Grotesk', sans-serif;
        }

        canvas {
          display: block;
          max-width: 100%;
        }

        .page {
          min-height: 100vh;
          padding: 36px 40px;
          position: relative;
          overflow-x: hidden;
          background:
            radial-gradient(circle at top left, rgba(124,92,255,0.18), transparent 32%),
            radial-gradient(circle at bottom right, rgba(244,114,182,0.13), transparent 30%),
            ${THEME.bg};
        }

        .point-cloud {
          position: fixed;
          inset: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
          z-index: 0;
          overflow: hidden;
        }

        .grid-bg {
          position: fixed;
          inset: 0;
          background-image:
            linear-gradient(${THEME.grid} 1px, transparent 1px),
            linear-gradient(90deg, ${THEME.grid} 1px, transparent 1px);
          background-size: 40px 40px;
          pointer-events: none;
          z-index: 0;
        }

        .dashboard {
          max-width: 1180px;
          margin: 0 auto;
          position: relative;
          z-index: 1;
        }

        .header {
          text-align: center;
          padding: 30px;
          background: linear-gradient(
            145deg,
            rgba(22,28,48,0.96),
            rgba(10,12,26,0.94)
          );
          border: 1px solid rgba(244,114,182,0.22);
          border-radius: 20px;
          backdrop-filter: blur(14px);
          margin-bottom: 22px;
          animation: fadeUp 0.5s ease;
          box-shadow: 0 0 28px rgba(244,114,182,0.08);
        }

        .tag {
          margin: 0 0 8px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          letter-spacing: 0.18em;
          color: ${THEME.primary};
          text-transform: uppercase;
        }

        h1 {
          margin: 0;
          font-size: 32px;
          letter-spacing: -0.03em;
          color: ${THEME.text};
        }

        .subtitle {
          margin-top: 8px;
          color: ${THEME.textSoft};
          font-size: 14px;
        }

        .main-grid {
          display: grid;
          grid-template-columns: 2fr 1fr;
          gap: 20px;
        }

        .card,
        .results {
          background: linear-gradient(
            145deg,
            rgba(18,24,42,0.96),
            rgba(9,13,28,0.92)
          );
          border: 1px solid rgba(124,92,255,0.24);
          border-radius: 20px;
          padding: 24px;
          backdrop-filter: blur(14px);
          box-shadow:
            0 0 26px rgba(124,92,255,0.10),
            inset 0 0 22px rgba(255,255,255,0.018);
          animation: fadeUp 0.5s ease;
        }

        .card h2,
        .results h2 {
          margin: 0 0 16px;
          font-size: 13px;
          font-family: 'JetBrains Mono', monospace;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: ${THEME.textMuted};
        }

        .drop-zone {
          min-height: 92px;
          border: 1px dashed rgba(56,189,248,0.42);
          border-radius: 14px;
          background: rgba(56,189,248,0.055);
          display: flex;
          align-items: center;
          justify-content: center;
          color: ${THEME.textMuted};
          cursor: pointer;
          font-family: 'JetBrains Mono', monospace;
          font-size: 13px;
          transition: 0.2s ease;
        }

        .drop-zone:hover {
          color: ${THEME.secondary};
          border-color: rgba(56,189,248,0.75);
          background: rgba(56,189,248,0.09);
        }

        .drop-zone input {
          display: none;
        }

        .preview {
          width: 100%;
          max-height: 440px;
          object-fit: contain;
          margin-top: 16px;
          border-radius: 12px;
          border: 1px solid rgba(124,92,255,0.22);
          background: rgba(0,0,0,0.3);
        }

        .run-btn {
          width: 100%;
          margin-top: 16px;
          background: rgba(124,92,255,0.14);
          border: 1px solid rgba(124,92,255,0.42);
          color: ${THEME.primary};
          font-size: 14px;
          font-weight: 600;
          font-family: 'JetBrains Mono', monospace;
          padding: 14px;
          border-radius: 12px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          transition: 0.2s ease;
        }

        .run-btn:hover:not(:disabled) {
          background: rgba(124,92,255,0.22);
          color: ${THEME.text};
          border-color: rgba(124,92,255,0.72);
        }

        .run-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .spinner {
          animation: spin 0.9s linear infinite;
        }

        .metric-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 11px 0;
          border-bottom: 1px solid rgba(124,92,255,0.18);
          font-family: 'JetBrains Mono', monospace;
        }

        .metric-row span {
          color: ${THEME.textMuted};
          font-size: 12px;
          text-transform: uppercase;
        }

        .metric-row strong {
          font-size: 13px;
          text-align: right;
        }

        .results {
          grid-column: 1 / -1;
        }

        .pose-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
          gap: 16px;
        }

        .pose-card {
          border-radius: 16px;
          padding: 18px;
          background: linear-gradient(
            145deg,
            rgba(20,25,44,0.94),
            rgba(11,15,31,0.86)
          );
          border: 1px solid rgba(56,189,248,0.18);
          min-height: 150px;
          box-shadow: inset 0 0 18px rgba(255,255,255,0.014);
        }

        .pose-card h3 {
          margin: 0 0 12px;
          font-size: 12px;
          font-family: 'JetBrains Mono', monospace;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: ${THEME.secondary};
        }

        .pose-card p,
        .pose-card li {
          margin: 6px 0;
          font-size: 13px;
          color: ${THEME.textSoft};
          line-height: 1.6;
        }

        .big-score {
          font-size: 34px !important;
          font-weight: 700;
          color: ${THEME.accent} !important;
          letter-spacing: -0.04em;
        }

        .status-box {
          margin-top: 16px;
          padding: 14px;
          border: 1px solid rgba(244,114,182,0.20);
          border-radius: 14px;
          background: rgba(244,114,182,0.055);
          color: ${THEME.textSoft};
          line-height: 1.7;
          white-space: pre-line;
          font-size: 13px;
        }

        ul {
          padding-left: 18px;
          margin: 0;
        }

        pre {
          margin-top: 16px;
          background: rgba(0,0,0,0.45);
          border: 1px solid rgba(239,68,68,0.25);
          color: ${THEME.error};
          padding: 16px;
          border-radius: 12px;
          overflow-x: auto;
          white-space: pre-wrap;
          font-size: 12px;
          font-family: 'JetBrains Mono', monospace;
        }

        @media (max-width: 900px) {
          .main-grid {
            grid-template-columns: 1fr;
          }

          .page {
            padding: 20px;
          }
        }
      `}</style>

      <div className="page">
        <div className="grid-bg" />
        <PointCloud />

        <div className="dashboard">
          <div className="header">
            <p className="tag">// ACE-G Visual Relocalization</p>
            <h1>ACE-G Visual Relocalization Dashboard</h1>
            <p className="subtitle">
              Upload a query image to estimate camera pose, localization confidence,
              and AI spatial explanation.
            </p>
          </div>

          <div className="main-grid">
            <div className="card">
              <h2>// Query Frame</h2>

              <label className="drop-zone">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const f = e.target.files[0];
                    setImage(f);
                    setResult(null);
                    setAiExplanation("");
                    if (f) setPreview(URL.createObjectURL(f));
                  }}
                />

                {image ? image.name : "[ click to upload query image ]"}
              </label>

              {preview && <img className="preview" src={preview} alt="query" />}

              <button
                className="run-btn"
                onClick={handleUpload}
                disabled={loading || !image}
              >
                {loading ? (
                  <>
                    <Spinner /> running ace-g...
                  </>
                ) : (
                  "→ run_visual_relocalization()"
                )}
              </button>
            </div>

            <div className="card">
              <h2>// Run Summary</h2>

              <MetricRow
                label="status"
                value={!result ? "awaiting" : result.success ? "SUCCESS" : "FAILED"}
                accent={
                  !result
                    ? THEME.textMuted
                    : result.success
                    ? THEME.success
                    : THEME.error
                }
              />

              <MetricRow
                label="inference"
                value={result ? `${result.inference_time}s` : "--"}
              />

              <MetricRow
                label="scene match"
                value={result?.pose ? sceneStatus(percent) : "--"}
                accent={quality.color}
              />

              <MetricRow
                label="pose file"
                value={result?.pose_file ? "created ✓" : "--"}
              />

              <MetricRow
                label="calibration"
                value={result?.calibration_path ? "ok" : "--"}
              />
            </div>

            {result && (
              <div className="results">
                <h2>// ACE-G Output</h2>

                <div className="pose-grid">
                  <div className="pose-card">
                    <h3>Localization Quality</h3>
                    <p className="big-score">
                      {pose?.confidence_percent !== undefined
                        ? `${Number(pose.confidence_percent).toFixed(1)}%`
                        : "—"}
                    </p>
                    <p style={{ color: quality.color }}>{quality.label}</p>
                    <p>Raw confidence: {pose?.confidence_raw ?? "--"}</p>
                  </div>

                  <div className="pose-card">
                    <h3>Scene Match Status</h3>
                    <p>{pose ? sceneStatus(percent) : "No pose detected"}</p>
                    <p>
                      {percent >= 50
                        ? "The query image has useful overlap with the trained map."
                        : "The query image may not overlap well with the trained map."}
                    </p>
                  </div>

                  <div className="pose-card">
                    <h3>Camera Orientation</h3>
                    <p>• {orientation?.horizontal ?? "—"}</p>
                    <p>• {orientation?.vertical ?? "—"}</p>
                    <p>• {orientation?.rollText ?? "—"}</p>
                  </div>

                  <div className="pose-card">
                    <h3>Camera Position</h3>
                    <p>• {position?.x ?? "—"}</p>
                    <p>• {position?.y ?? "—"}</p>
                    <p>• {position?.z ?? "—"}</p>
                  </div>

                  <div className="pose-card">
                    <h3>Pose Values</h3>
                    <p>X: {pose?.translation?.tx ?? "--"}</p>
                    <p>Y: {pose?.translation?.ty ?? "--"}</p>
                    <p>Z: {pose?.translation?.tz ?? "--"}</p>
                  </div>

                  <div className="pose-card">
                    <h3>AI Diagnostics</h3>

                    <ul>
                      {aiDiagnostics.length > 0 ? (
                        aiDiagnostics.map((item, idx) => (
                          <li key={idx}>{item}</li>
                        ))
                      ) : (
                        <li>No diagnostics available.</li>
                      )}
                    </ul>
                  </div>
                </div>

                <div className="status-box">
                  <strong>AI Explanation</strong>
                  <br />
                  {aiExplanation || "No AI explanation available."}
                </div>

                {result.stderr && (
                  <>
                    <h2 style={{ marginTop: 24 }}>// stderr / warnings</h2>
                    <pre>{result.stderr}</pre>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}