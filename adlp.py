# ============================================================
# CLOUDPREDICT AI - COMPLETE FLASK APPLICATION
# ============================================================

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import csv
import psutil
from datetime import datetime


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "cloud_model.joblib"
)


# ============================================================
# HISTORY PATH
# ============================================================

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "prediction_history.csv"
)


# ============================================================
# HISTORY CSV FIELDS
# ============================================================

HISTORY_FIELDS = [
    "timestamp",
    "cpu_usage",
    "memory_usage",
    "predicted_cpu_usage",
    "resource_status"
]


# ============================================================
# LOAD ML MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "cloud_model.joblib not found. "
        "Please run train_model.py first."
    )


model = joblib.load(MODEL_PATH)


# ============================================================
# CREATE HISTORY FILE IF NOT EXISTS
# ============================================================

if not os.path.exists(HISTORY_PATH):

    with open(
        HISTORY_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_FIELDS
        )

        writer.writeheader()


# ============================================================
# RESOURCE STATUS
# ============================================================

def get_resource_status(cpu, memory):

    if cpu >= 80 or memory >= 80:

        return "Critical"

    elif cpu >= 60 or memory >= 60:

        return "Warning"

    else:

        return "Normal"


# ============================================================
# GET REAL-TIME SYSTEM USAGE
# ============================================================

def get_system_usage():

    cpu = psutil.cpu_percent(
        interval=0.5
    )

    memory = psutil.virtual_memory()


    return {

        "cpu_usage": round(
            cpu,
            2
        ),

        "memory_usage": round(
            memory.percent,
            2
        ),

        "memory_available": round(
            memory.available / (1024 ** 3),
            2
        ),

        "memory_total": round(
            memory.total / (1024 ** 3),
            2
        )

    }


# ============================================================
# HOME PAGE
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# REAL-TIME SYSTEM USAGE API
# ============================================================

@app.route(
    "/api/system-usage",
    methods=["GET"]
)
def system_usage_api():

    try:

        usage = get_system_usage()


        cpu = usage["cpu_usage"]

        memory = usage["memory_usage"]


        usage["resource_status"] = get_resource_status(
            cpu,
            memory
        )


        usage["timestamp"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        return jsonify({

            "success": True,

            **usage

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Get JSON data
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        ) or {}


        # ----------------------------------------------------
        # Check required values
        # ----------------------------------------------------

        if (
            "cpu_usage" not in data
            or
            "memory_usage" not in data
        ):

            return jsonify({

                "success": False,

                "error":
                    "CPU usage and memory usage are required."

            }), 400


        # ----------------------------------------------------
        # Convert values to float
        # ----------------------------------------------------

        cpu = float(
            data["cpu_usage"]
        )


        memory = float(
            data["memory_usage"]
        )


        # ----------------------------------------------------
        # Validate values
        # ----------------------------------------------------

        if not (
            0 <= cpu <= 100
            and
            0 <= memory <= 100
        ):

            return jsonify({

                "success": False,

                "error":
                    "CPU and memory must be between 0 and 100."

            }), 400


        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        sample = pd.DataFrame([{

            "cpu_usage": cpu,

            "memory_usage": memory

        }])


        # ----------------------------------------------------
        # ML Prediction
        # ----------------------------------------------------

        prediction = float(
            model.predict(sample)[0]
        )


        # ----------------------------------------------------
        # Keep prediction between 0 and 100
        # ----------------------------------------------------

        prediction = max(
            0.0,
            min(
                100.0,
                prediction
            )
        )


        # ----------------------------------------------------
        # Resource Status
        # ----------------------------------------------------

        resource_status = get_resource_status(
            cpu,
            memory
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        # ----------------------------------------------------
        # History Record
        # ----------------------------------------------------

        history_record = {

            "timestamp":
                timestamp,

            "cpu_usage":
                round(cpu, 2),

            "memory_usage":
                round(memory, 2),

            "predicted_cpu_usage":
                round(
                    prediction,
                    2
                ),

            "resource_status":
                resource_status

        }


        # ----------------------------------------------------
        # Save Prediction History
        # ----------------------------------------------------

        with open(
            HISTORY_PATH,
            "a",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=HISTORY_FIELDS
            )


            writer.writerow(
                history_record
            )


        # ----------------------------------------------------
        # Return Prediction
        # ----------------------------------------------------

        return jsonify({

            "success":
                True,

            "timestamp":
                timestamp,

            "cpu_usage":
                round(
                    cpu,
                    2
                ),

            "memory_usage":
                round(
                    memory,
                    2
                ),

            "predicted_cpu_usage":
                round(
                    prediction,
                    2
                ),

            "resource_status":
                resource_status

        })


    # --------------------------------------------------------
    # Invalid Input
    # --------------------------------------------------------

    except (
        KeyError,
        TypeError,
        ValueError
    ):

        return jsonify({

            "success":
                False,

            "error":
                "Please enter valid CPU and memory values."

        }), 400


    # --------------------------------------------------------
    # Other Error
    # --------------------------------------------------------

    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# PREDICTION HISTORY API
# ============================================================

@app.route(
    "/api/history",
    methods=["GET"]
)
def prediction_history():

    try:

        history = []


        if not os.path.exists(
            HISTORY_PATH
        ):

            return jsonify({

                "success":
                    True,

                "history":
                    []

            })


        with open(
            HISTORY_PATH,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(
                file
            )


            for row in reader:

                history.append(row)


        # Latest records first

        history.reverse()


        return jsonify({

            "success":
                True,

            "history":
                history

        })


    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# LATEST PREDICTION API
# ============================================================

@app.route(
    "/api/history/latest",
    methods=["GET"]
)
def latest_prediction():

    try:

        if not os.path.exists(
            HISTORY_PATH
        ):

            return jsonify({

                "success":
                    True,

                "data":
                    None

            })


        with open(
            HISTORY_PATH,
            "r",
            newline=""
        ) as file:

            reader = list(
                csv.DictReader(file)
            )


        if not reader:

            return jsonify({

                "success":
                    True,

                "data":
                    None

            })


        return jsonify({

            "success":
                True,

            "data":
                reader[-1]

        })


    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# HEALTH CHECK API
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health_check():

    return jsonify({

        "success":
            True,

        "status":
            "CloudPredict AI is running",

        "model_loaded":
            model is not None,

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
    debug=True,
    host="0.0.0.0",
    port=5001,
    use_reloader=False
)