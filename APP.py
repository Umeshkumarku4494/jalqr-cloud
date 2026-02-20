import cv2
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# ----------------------------
# LOAD RULES (same as yours)
# ----------------------------
conditions_file = "Conditions.txt"

rules = []
with open(conditions_file, "r") as f:
    lines = f.readlines()

for line in lines[1:]:
    parts = line.strip().split("\t")
    if len(parts) >= 6:
        rules.append({
            "Chlorine": parts[1],
            "Nitrate": parts[2],
            "Iron": parts[3],
            "Phosphate": parts[4],
            "Output": parts[5]
        })

# ----------------------------
# Helper Functions (same logic)
# ----------------------------

def detect_color(zone, lower, upper):
    if zone.size == 0:
        return 0
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) / (zone.shape[0] * zone.shape[1])

def chlorine_state(r):
    if r < 0.05: return "Colorless"
    elif r < 0.15: return "Light Pink"
    elif r < 0.30: return "Pink"
    else: return "Dark Magenta"

def nitrate_state(r):
    if r < 0.05: return "White"
    elif r < 0.20: return "Pink"
    else: return "Bright Pink"

def iron_state(r):
    if r < 0.05: return "Clear"
    elif r < 0.20: return "Orange"
    else: return "Dark Orange"

def phosphate_state(r):
    if r < 0.05: return "Clear"
    else: return "Blue"

def match_rule(chlorine, nitrate, iron, phosphate):
    for rule in rules:
        if (rule["Chlorine"] in [chlorine, "Any"] and
            rule["Nitrate"] in [nitrate, "Any"] and
            rule["Iron"] in [iron, "Any"] and
            rule["Phosphate"] in [phosphate, "Any"]):
            return rule["Output"]
    return "CHECK"

# ----------------------------
# IMAGE PROCESSING
# ----------------------------

def process_image(frame):

    chlorine = "NA"
    nitrate = "NA"
    iron = "NA"
    phosphate = "NA"
    status = "CHECK"

    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(frame)

    if bbox is not None and data:
        pts = bbox[0].astype(int)
        qr_x, qr_y, qr_w, qr_h = cv2.boundingRect(pts)

        top = frame[qr_y-50:qr_y, qr_x:qr_x+qr_w]
        bottom = frame[qr_y+qr_h:qr_y+qr_h+50, qr_x:qr_x+qr_w]
        left = frame[qr_y:qr_y+qr_h, qr_x-50:qr_x]
        right = frame[qr_y:qr_y+qr_h, qr_x+qr_w:qr_x+qr_w+50]

        iron = iron_state(detect_color(top,
                         np.array([5,80,80]),
                         np.array([20,255,255])))

        chlorine = chlorine_state(detect_color(bottom,
                         np.array([140,80,80]),
                         np.array([170,255,255])))

        phosphate = phosphate_state(detect_color(left,
                         np.array([100,80,80]),
                         np.array([130,255,255])))

        nitrate = nitrate_state(detect_color(right,
                         np.array([130,50,80]),
                         np.array([160,255,255])))

        status = match_rule(chlorine, nitrate, iron, phosphate)

    return {
        "status": status,
        "chlorine": chlorine,
        "nitrate": nitrate,
        "iron": iron,
        "phosphate": phosphate
    }

# ----------------------------
# API ENDPOINT
# ----------------------------

@app.route("/scan", methods=["POST"])
def scan():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filepath = "temp.jpg"
    file.save(filepath)

    frame = cv2.imread(filepath)
    result = process_image(frame)

    os.remove(filepath)

    return jsonify(result)

# ----------------------------
# ROOT CHECK
# ----------------------------

@app.route("/")
def home():
    return "JALQR Cloud Server Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)