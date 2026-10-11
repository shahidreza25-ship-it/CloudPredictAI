
from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import os
import csv
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cloud_model.joblib"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "cloud_model.joblib not found. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)
# Prediction history storage
HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prediction_history.csv"
)

HISTORY_FIELDS = [
    "timestamp",
    "cpu_usage",
    "memory_usage",
    "predicted_cpu_usage",
    "resource_status"
]

if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CloudPredict AI | Analytics</title>
<style>
:root {
    --bg: #0a1020;
    --panel: #121c30;
    --panel2: #18253b;
    --border: #293851;
    --text: #edf3ff;
    --muted: #93a4bf;
    --cyan: #4de1d0;
    --blue: #7c9cff;
    --orange: #ffbc72;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input { font: inherit; }
.layout {
    display: grid;
    grid-template-columns: 230px minmax(0, 1fr);
    min-height: 100vh;
}
.sidebar {
    background: #0d1526;
    border-right: 1px solid var(--border);
    padding: 28px 18px;
}
.brand { font-size: 23px; font-weight: 800; margin-bottom: 42px; }
.brand span { color: var(--cyan); }
.nav-label {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 2px;
    margin: 25px 10px 12px;
}
.nav-item {
    display: block;
    padding: 13px 12px;
    margin: 5px 0;
    border-radius: 9px;
    color: #b8c5da;
    text-decoration: none;
    font-size: 14px;
}
.nav-item.active {
    color: var(--cyan);
    background: #17313e;
}
.side-note {
    margin-top: 60px;
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 12px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.7;
}
main { min-width: 0; padding: 28px 34px; }
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    margin-bottom: 34px;
}
.eyebrow {
    color: var(--cyan);
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
}
h1 { font-size: clamp(25px, 3vw, 34px); margin: 8px 0; }
.subtitle { color: var(--muted); line-height: 1.6; }
.status {
    border: 1px solid #246d62;
    background: #12332f;
    color: var(--cyan);
    border-radius: 30px;
    padding: 9px 13px;
    white-space: nowrap;
    font-size: 12px;
}
.grid { display: grid; gap: 18px; }
.metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 21px;
    min-width: 0;
}
.metric-top {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    align-items: center;
}
.metric-label { color: var(--muted); font-size: 13px; }
.metric-icon {
    display: grid;
    place-items: center;
    width: 37px;
    height: 37px;
    border-radius: 10px;
    background: var(--panel2);
    color: var(--cyan);
}
.metric-value { font-size: 32px; font-weight: 750; margin: 18px 0 8px; }
.metric-foot { color: var(--muted); font-size: 12px; line-height: 1.5; }
.middle {
    grid-template-columns: minmax(0, 1.5fr) minmax(280px, 1fr);
    margin-top: 20px;
}
.panel-heading {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    align-items: center;
    margin-bottom: 18px;
}
h2 { font-size: 17px; margin: 0; }
.small { font-size: 12px; color: var(--muted); }
canvas { width: 100%; height: 220px; display: block; }
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 13px; }
label { display: block; color: #c6d2e5; font-size: 12px; margin-bottom: 9px; }
input {
    width: 100%;
    padding: 12px;
    background: #0b1425;
    color: white;
    border: 1px solid #344560;
    border-radius: 9px;
    outline-color: var(--cyan);
}
.primary {
    width: 100%;
    padding: 13px;
    margin-top: 17px;
    background: var(--cyan);
    color: #071723;
    font-weight: 800;
    border: none;
    border-radius: 9px;
    cursor: pointer;
}
.primary:disabled { opacity: .65; cursor: wait; }
.result {
    margin-top: 18px;
    padding: 17px;
    border-radius: 12px;
    background: #0b1425;
    border: 1px solid var(--border);
}
.result-number { font-size: 35px; font-weight: 800; color: var(--cyan); margin: 9px 0; }
.bar {
    height: 7px;
    background: #28364b;
    border-radius: 10px;
    overflow: hidden;
    margin: 12px 0;
}
.bar-fill { height: 100%; width: 0%; background: var(--cyan); transition: width .3s; }
.advice {
    margin-top: 13px;
    border-radius: 10px;
    padding: 13px;
    background: #18253b;
    color: #d2dced;
    font-size: 12px;
    line-height: 1.7;
}
.history { margin-top: 20px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td {
    padding: 13px 9px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    white-space: nowrap;
}
th { color: var(--muted); font-weight: 500; }
.empty { color: var(--muted); text-align: center; padding: 25px; }
footer { color: var(--muted); font-size: 11px; margin: 24px 0 5px; line-height: 1.6; }
@media(max-width: 950px) {
    .layout { grid-template-columns: 1fr; }
    .sidebar { display: none; }
    main { padding: 22px 18px; }
    .middle { grid-template-columns: 1fr; }
}
@media(max-width: 580px) {
    .metrics { grid-template-columns: 1fr; }
    .form-grid { grid-template-columns: 1fr; }
    .topbar { align-items: flex-start; flex-direction: column; }
    .metric-value { font-size: 29px; }
}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
    <div class="brand">Cloud<span>Predict</span> AI</div>
    <div class="nav-label">WORKSPACE</div>
    <a class="nav-item active" href="#overview">▦ &nbsp; Overview</a>
    <a class="nav-item" href="#predictor">⌁ &nbsp; ML Predictor</a>
    <a class="nav-item" href="#history">◷ &nbsp; Prediction History</a>
    <div class="nav-label">PROJECT</div>
    <a class="nav-item" href="#resources">◇ &nbsp; Resources</a>
    <div class="side-note">
        <div style="color:var(--cyan);font-weight:700">MODEL STATUS</div>
        <div>● Trained model loaded</div>
        <div>Random Forest Regressor</div>
        <div>Local Flask application</div>
    </div>
</aside>

<main id="overview">
    <header class="topbar">
        <div>
            <div class="eyebrow">Cloud intelligence platform</div>
            <h1>Monitoring Dashboard</h1>
            <div class="subtitle">Analyze resource usage and predict future CPU demand.</div>
        </div>
        <div class="status">● MODEL READY</div>
    </header>

    <section class="grid metrics">
        <div class="panel">
            <div class="metric-top">
                <span class="metric-label">Current CPU Input</span>
                <span class="metric-icon">⌁</span>
            </div>
            <div class="metric-value" id="cpuCard">75%</div>
            <div class="metric-foot">Value entered for the latest prediction</div>
        </div>
        <div class="panel">
            <div class="metric-top">
                <span class="metric-label">Memory Input</span>
                <span class="metric-icon">▤</span>
            </div>
            <div class="metric-value" id="memoryCard">80%</div>
            <div class="metric-foot">Value entered for the latest prediction</div>
        </div>
        <div class="panel">
            <div class="metric-top">
                <span class="metric-label">Predicted CPU</span>
                <span class="metric-icon">✳</span>
            </div>
            <div class="metric-value" id="forecastCard">—</div>
            <div class="metric-foot" id="forecastFoot">Run a prediction to calculate</div>
        </div>
    </section>

    <section class="grid middle" id="resources">
        <div class="panel">
            <div class="panel-heading">
                <h2>CPU Prediction Trend</h2>
                <span class="small">Your prediction sessions</span>
            </div>
            <canvas id="chart" aria-label="CPU input and predicted CPU trend"></canvas>
            <div class="small">The graph updates when you make predictions. It is not a live server monitor.</div>
        </div>

        <div class="panel" id="predictor">
            <div class="panel-heading">
                <h2>ML Predictor</h2>
                <span class="small">Random Forest</span>
            </div>
            <form id="predictionForm">
                <div class="form-grid">
                    <div>
                        <label for="cpu">CPU usage (%)</label>
                        <input id="cpu" type="number" min="0" max="100"
                               step="any" value="75" required>
                    </div>
                    <div>
                        <label for="memory">Memory usage (%)</label>
                        <input id="memory" type="number" min="0" max="100"
                               step="any" value="80" required>
                    </div>
                </div>
                <button class="primary" id="predictButton" type="submit">
                    Run Prediction →
                </button>
            </form>
            <div class="result" aria-live="polite">
                <div class="small">PREDICTED NEXT CPU USAGE</div>
                <div class="result-number" id="resultValue">—</div>
                <div class="bar"><div class="bar-fill" id="resultBar"></div></div>
                <div class="small" id="resultMessage">Enter values and run your model.</div>
            </div>
            <div class="advice" id="advice">
                <strong>Optimization insight</strong><br>
                Make a prediction to receive a usage-based suggestion.
            </div>
        </div>
    </section>

    <section class="panel history" id="history">
        <div class="panel-heading">
            <h2>Prediction History</h2>
            <span class="small" id="historyCount">0 predictions</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Session</th>
                        <th>CPU Input</th>
                        <th>Memory Input</th>
                        <th>Predicted CPU</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="historyBody">
                    <tr><td colspan="5" class="empty">No predictions yet. Run your first prediction.</td></tr>
                </tbody>
            </table>
        </div>
    </section>

    <footer>
        CloudPredict AI · Machine Learning for Cloud Workload Prediction.<br>
        Predictions are model estimates, not live infrastructure measurements.
        Optimization suggestions are rule-based guidance.
    </footer>
</main>
</div>

<script>
const form = document.getElementById("predictionForm");
const button = document.getElementById("predictButton");
const historyRows = [];
const chart = document.getElementById("chart");
const ctx = chart.getContext("2d");

function drawChart() {
    const rect = chart.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    chart.width = rect.width * dpr;
    chart.height = 220 * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = 220;
    const left = 32, right = 12, top = 15, bottom = 30;
    const plotH = h - top - bottom;
    const plotW = Math.max(10, w - left - right);

    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px sans-serif";
    ctx.fillStyle = "#93a4bf";
    ctx.strokeStyle = "#293851";
    ctx.lineWidth = 1;

    [0, 25, 50, 75, 100].forEach(v => {
        const y = top + plotH * (1 - v / 100);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(w - right, y);
        ctx.stroke();
        ctx.fillText(v + "%", 0, y + 4);
    });

    if (!historyRows.length) {
        ctx.fillStyle = "#93a4bf";
        ctx.fillText("Your prediction trend will appear here", left + 8, h / 2);
        return;
    }

    const rows = historyRows.slice().reverse().slice(-8);
    const x = i => left + (rows.length === 1 ? plotW / 2 : i * plotW / (rows.length - 1));
    const y = v => top + plotH * (1 - v / 100);

    function line(key, color) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        rows.forEach((r, i) => {
            if (i === 0) ctx.moveTo(x(i), y(r[key]));
            else ctx.lineTo(x(i), y(r[key]));
        });
        ctx.stroke();
        rows.forEach((r, i) => {
            ctx.beginPath();
            ctx.fillStyle = color;
            ctx.arc(x(i), y(r[key]), 3, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    line("cpu", "#7c9cff");
    line("predicted", "#4de1d0");

    ctx.fillStyle = "#7c9cff";
    ctx.fillRect(left, h - 12, 9, 3);
    ctx.fillStyle = "#c4cee0";
    ctx.fillText("CPU input", left + 14, h - 8);
    ctx.fillStyle = "#4de1d0";
    ctx.fillRect(left + 100, h - 12, 9, 3);
    ctx.fillStyle = "#c4cee0";
    ctx.fillText("Predicted CPU", left + 114, h - 8);
}

function updateAdvice(value) {
    const advice = document.getElementById("advice");
    if (value >= 80) {
        advice.innerHTML = "<strong>High predicted CPU demand</strong><br>Review workload distribution and consider scaling resources if actual monitoring confirms sustained high usage.";
    } else if (value >= 60) {
        advice.innerHTML = "<strong>Moderate-to-high predicted demand</strong><br>Keep monitoring CPU trends and check available capacity.";
    } else {
        advice.innerHTML = "<strong>Lower predicted CPU demand</strong><br>Review resource allocation and avoid reducing capacity without checking actual workload and service requirements.";
    }
}

form.addEventListener("submit", async event => {
    event.preventDefault();

    const cpu = Number(document.getElementById("cpu").value);
    const memory = Number(document.getElementById("memory").value);

    if (!Number.isFinite(cpu) || !Number.isFinite(memory) ||
        cpu < 0 || cpu > 100 || memory < 0 || memory > 100) {
        alert("CPU and memory must be between 0 and 100.");
        return;
    }

    button.disabled = true;
    button.textContent = "Analyzing...";
    document.getElementById("resultMessage").textContent = "Running the trained model...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({cpu_usage: cpu, memory_usage: memory})
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed.");

        const predicted = data.predicted_cpu_usage;
        document.getElementById("cpuCard").textContent = cpu + "%";
        document.getElementById("memoryCard").textContent = memory + "%";
        document.getElementById("forecastCard").textContent = predicted + "%";
        document.getElementById("forecastFoot").textContent = "Latest model estimate";
        document.getElementById("resultValue").textContent = predicted + "%";
        document.getElementById("resultBar").style.width = predicted + "%";
        document.getElementById("resultMessage").textContent =
            "Prediction returned successfully by the trained model.";

        updateAdvice(predicted);

        historyRows.unshift({
            cpu, memory, predicted,
            time: new Date().toLocaleTimeString()
        });

        document.getElementById("historyCount").textContent =
            historyRows.length + (historyRows.length === 1 ? " prediction" : " predictions");

        document.getElementById("historyBody").innerHTML = historyRows.map((r, i) => `
            <tr>
                <td>${i + 1} · ${r.time}</td>
                <td>${r.cpu}%</td>
                <td>${r.memory}%</td>
                <td style="color:#4de1d0">${r.predicted}%</td>
                <td>${r.predicted >= 80 ? "Review capacity" : "Estimate ready"}</td>
            </tr>
        `).join("");

        drawChart();
    } catch (error) {
        document.getElementById("resultMessage").textContent = error.message;
    } finally {
        button.disabled = false;
        button.textContent = "Run Prediction →";
    }
});

window.addEventListener("resize", drawChart);
drawChart();
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        cpu = float(data["cpu_usage"])
        memory = float(data["memory_usage"])

        if not (0 <= cpu <= 100 and 0 <= memory <= 100):
            return jsonify({
                "error": "CPU and memory must be between 0 and 100."
            }), 400

        sample = pd.DataFrame([{
            "cpu_usage": cpu,
            "memory_usage": memory
        }])

        prediction = float(model.predict(sample)[0])
        prediction = max(0.0, min(100.0, prediction))
        history_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_usage": cpu,
            "memory_usage": memory,
            "predicted_cpu_usage": round(prediction, 2),
            "resource_status": (
                "Critical" if prediction >= 80
                else "Warning" if prediction >= 60
                else "Normal"
            )
        }

        with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
            writer.writerow(history_record)
        return jsonify({
            "predicted_cpu_usage": round(prediction, 2)
        })

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Please enter valid CPU and memory values."
        }), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
