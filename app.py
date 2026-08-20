"""
Local web console for the deepfake voice detector — styled as a security
scanning tool (VoxShield).

Usage:
    python3 app.py --model outputs/best_model.pt --port 5100

Then open http://127.0.0.1:5100 and submit a clip for analysis.
"""

import argparse
import hashlib
import os
import tempfile
import time
from datetime import datetime, timezone

from flask import Flask, request, render_template_string
import torch

from infer import load_model, predict

app = Flask(__name__)
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = None

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>VoxShield — Synthetic Voice Intrusion Detection</title>
<style>
  :root {
    --bg: #0a0e13;
    --panel: #0f151c;
    --border: #1e2b38;
    --border-soft: #182430;
    --text: #cddce6;
    --muted: #64798c;
    --accent: #35e6a0;
    --accent-dim: #1c8f66;
    --danger: #ff4d6a;
    --danger-dim: #7a2333;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", "Courier New", monospace;
  }
  * { box-sizing: border-box; }
  body {
    font-family: var(--mono);
    background: var(--bg);
    color: var(--text);
    max-width: 720px;
    margin: 48px auto;
    padding: 0 20px 60px;
    background-image:
      repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 3px);
  }
  header.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 28px;
  }
  .brand { display: flex; align-items: baseline; gap: 10px; }
  .brand .glyph { font-size: 1.3rem; }
  .brand h1 {
    font-size: 1.15rem;
    letter-spacing: 0.08em;
    margin: 0;
    color: #eef5f9;
  }
  .brand .sub {
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    margin-top: 2px;
  }
  .status {
    font-size: 0.72rem;
    color: var(--accent);
    display: flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.04em;
  }
  .status .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 1.8s infinite ease-in-out;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

  .panel {
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 4px;
    padding: 22px 24px;
  }
  .panel + .panel { margin-top: 18px; }
  .panel-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 14px;
    text-transform: uppercase;
  }

  .dropzone {
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 22px;
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
  }
  input[type=file] {
    font-family: var(--mono);
    color: var(--text);
    font-size: 0.8rem;
    margin-top: 10px;
  }
  input[type=file]::file-selector-button {
    font-family: var(--mono);
    background: #16202a;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 6px 12px;
    cursor: pointer;
  }
  button[type=submit] {
    font-family: var(--mono);
    background: var(--accent);
    color: #06231a;
    border: none;
    border-radius: 3px;
    padding: 10px 20px;
    margin-top: 16px;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    font-weight: 700;
    cursor: pointer;
  }
  button[type=submit]:hover { background: #4cf0b3; }

  .verdict {
    border-radius: 4px;
    padding: 18px 20px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
  }
  .verdict.threat { border: 1px solid var(--danger-dim); background: rgba(255,77,106,0.06); }
  .verdict.clear  { border: 1px solid var(--accent-dim); background: rgba(53,230,160,0.06); }
  .verdict .icon { font-size: 1.4rem; line-height: 1; }
  .verdict .title { font-size: 0.95rem; font-weight: 700; letter-spacing: 0.03em; }
  .verdict.threat .title { color: var(--danger); }
  .verdict.clear .title { color: var(--accent); }
  .verdict .sub { color: var(--muted); font-size: 0.78rem; margin-top: 4px; }

  table.report { width: 100%; border-collapse: collapse; margin-top: 18px; font-size: 0.8rem; }
  table.report td { padding: 5px 0; vertical-align: top; }
  table.report td.k { color: var(--muted); width: 220px; }
  table.report td.v { color: var(--text); }

  .gauge-row { margin-top: 6px; }
  .gauge-label { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; }
  .gauge-track { background: #131c25; border: 1px solid var(--border-soft); border-radius: 3px; height: 8px; overflow: hidden; }
  .gauge-fill { height: 100%; }
  .gauge-fill.real { background: var(--accent); }
  .gauge-fill.fake { background: var(--danger); }

  .error-box { border: 1px solid var(--danger-dim); background: rgba(255,77,106,0.06); color: var(--danger); padding: 14px 18px; border-radius: 4px; font-size: 0.85rem; }

  footer {
    margin-top: 30px;
    color: var(--muted);
    font-size: 0.72rem;
    line-height: 1.6;
    border-top: 1px solid var(--border);
    padding-top: 14px;
  }
  footer .path { color: #8fa3b3; }
</style>
</head>
<body>

  <header class="topbar">
    <div class="brand">
      <span class="glyph">🛡️</span>
      <div>
        <h1>VOXSHIELD</h1>
        <div class="sub">SYNTHETIC VOICE INTRUSION DETECTION SYSTEM</div>
      </div>
    </div>
    <div class="status"><span class="dot"></span>ENGINE ONLINE</div>
  </header>

  <div class="panel">
    <div class="panel-label">// Submit Sample</div>
    <form method="post" enctype="multipart/form-data">
      <div class="dropzone">
        Upload an audio sample (wav / mp3 / flac / ogg / m4a) for spoofed-voice forensic analysis.
        <br>
        <input type="file" name="audio" accept="audio/*" required>
      </div>
      <button type="submit">▶ RUN SCAN</button>
    </form>
  </div>

  {% if result %}
  <div class="panel">
    <div class="panel-label">// Scan Report</div>

    <div class="verdict {{ 'threat' if result.label == 'FAKE' else 'clear' }}">
      <div class="icon">{{ '⚠' if result.label == 'FAKE' else '✔' }}</div>
      <div>
        <div class="title">
          {{ 'THREAT DETECTED — SYNTHETIC / CLONED VOICE' if result.label == 'FAKE' else 'VERIFIED — AUTHENTIC HUMAN VOICE' }}
        </div>
        <div class="sub">Classifier confidence: {{ '%.1f' % (result.confidence * 100) }}%</div>
      </div>
    </div>

    <div class="gauge-row">
      <div class="gauge-label"><span>P(authentic)</span><span>{{ '%.1f' % (result.p_real * 100) }}%</span></div>
      <div class="gauge-track"><div class="gauge-fill real" style="width:{{ result.p_real * 100 }}%"></div></div>
    </div>
    <div class="gauge-row">
      <div class="gauge-label"><span>P(synthetic)</span><span>{{ '%.1f' % (result.p_fake * 100) }}%</span></div>
      <div class="gauge-track"><div class="gauge-fill fake" style="width:{{ result.p_fake * 100 }}%"></div></div>
    </div>

    <table class="report">
      <tr><td class="k">Sample</td><td class="v">{{ result.filename }}</td></tr>
      <tr><td class="k">SHA-256</td><td class="v">{{ result.sha256 }}</td></tr>
      <tr><td class="k">Detection engine</td><td class="v">CNN + MFCC(Δ, ΔΔ) v1.0</td></tr>
      <tr><td class="k">Scan duration</td><td class="v">{{ result.elapsed_ms }} ms</td></tr>
      <tr><td class="k">Timestamp (UTC)</td><td class="v">{{ result.timestamp }}</td></tr>
    </table>
  </div>
  {% elif error %}
  <div class="panel">
    <div class="panel-label">// Scan Report</div>
    <div class="error-box">SCAN FAILED — {{ error }}</div>
  </div>
  {% endif %}

  <footer>
    Model checkpoint: <span class="path">{{ model_path }}</span><br>
    VoxShield is a research demo, not a certified forensic tool. Accuracy depends entirely on training
    data coverage of real vs. synthetic/cloned speech.
  </footer>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        f = request.files.get("audio")
        if not f or f.filename == "":
            error = "No file selected."
        else:
            suffix = os.path.splitext(f.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                f.save(tmp.name)
                tmp_path = tmp.name
            try:
                raw = open(tmp_path, "rb").read()
                sha256 = hashlib.sha256(raw).hexdigest()

                start = time.perf_counter()
                label, confidence, probs = predict(tmp_path, MODEL, DEVICE)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

                result = {
                    "filename": f.filename,
                    "label": label,
                    "confidence": confidence,
                    "p_real": probs[0],
                    "p_fake": probs[1],
                    "sha256": sha256,
                    "elapsed_ms": elapsed_ms,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
            except Exception as e:
                error = str(e)
            finally:
                os.unlink(tmp_path)

    return render_template_string(PAGE, result=result, error=error, model_path=MODEL_PATH)


def main():
    global MODEL, MODEL_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/best_model.pt")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"No model checkpoint at {args.model}. Train one first with train.py."
        )

    MODEL_PATH = args.model
    MODEL = load_model(args.model, DEVICE)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
