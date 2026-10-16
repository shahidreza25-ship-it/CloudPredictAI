from flask import Flask, render_template, jsonify, request
import joblib
import os
import psutil

app = Flask(__name__)
MODEL_PATH = "cloud_model.joblib"

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/api/system-usage")
def system_usage():
    cpu_usage = psutil.cpu_percent(interval=1)

    memory = psutil.virtual_memory()
    memory_percent = memory.percent

    # Highest resource usage
    resource_usage = max(cpu_usage, memory_percent)

    if resource_usage < 70:
        status = "Normal"
    elif resource_usage <= 80:
        status = "Medium"
    else:
        status = "High Alert"

    # AI prediction using real system usage
    prediction = model.predict([[cpu_usage, memory_percent]])[0]

    return jsonify({
        "cpu": cpu_usage,
        "memory": memory_percent,
        "available_memory": memory.available / (1024 ** 3),
        "prediction": float(prediction),
        "status": status
    })
@app.route("/ml-prediction", methods=["GET", "POST"])
def ml_prediction():

    prediction = None

    if request.method == "POST":
        cpu_usage = float(request.form["cpu_usage"])
        memory_usage = float(request.form["memory_usage"])

        prediction = model.predict([[cpu_usage, memory_usage]])[0]

    return render_template(
        "ml_prediction.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)
