import cv2
import numpy as np
import base64
import os
import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ----------------------------
# CONFIG & PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# We use /tmp/ for Railway because the main directory is often read-only
CONDITION_FILE = os.path.join(BASE_DIR, "Conditions.txt")

# Initialize rules list
rules = []

# Load rules from Conditions.txt if it exists
if os.path.exists(CONDITION_FILE):
    try:
        with open(CONDITION_FILE, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    rules.append({
                        "Chlorine": parts[1],
                        "Nitrate": parts[2],
                        "Iron": parts[3],
                        "Phosphate": parts[4],
                        "Output": parts[5]
                    })
        print(f"Loaded {len(rules)} rules.")
    except Exception as e:
        print(f"Error loading rules: {e}")
else:
    print("Warning: Conditions.txt not found. Using default 'CHECK' status.")

# ----------------------------
# PROCESSING LOGIC (Your Engine)
# ----------------------------

def detect_color(zone, lower, upper):
    if zone is None or zone.size == 0:
        return 0
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) / (zone.shape[0] * zone.shape[1])

def sharpen(image):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(image, -1, kernel)

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

def process_frame(frame):
    frame = sharpen(frame)
    status, chlorine, nitrate, iron, phosphate = "CHECK", "NA", "NA", "NA", "NA"

    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(frame)

    if bbox is not None and data:
        pts = bbox[0].astype(int)
        qr_x, qr_y, qr_w, qr_h = cv2.boundingRect(pts)
        frame_h, frame_w, _ = frame.shape

        # Calculate safe borders for color zones
        desired_border = int(0.35 * qr_w)
        border = max(5, min(desired_border, qr_y, frame_h - (qr_y + qr_h), qr_x, frame_w - (qr_x + qr_w)) - 5)

        if border > 10:
            top = frame[qr_y - border:qr_y, qr_x:qr_x + qr_w]
            bottom = frame[qr_y + qr_h:qr_y + qr_h + border, qr_x:qr_x + qr_w]
            left = frame[qr_y:qr_y + qr_h, qr_x - border:qr_x]
            right = frame[qr_y:qr_y + qr_h, qr_x + qr_w:qr_x + qr_w + border]

            iron = iron_state(detect_color(top, np.array([5,80,80]), np.array([20,255,255])))
            chlorine = chlorine_state(detect_color(bottom, np.array([140,80,80]), np.array([170,255,255])))
            phosphate = phosphate_state(detect_color(left, np.array([100,80,80]), np.array([130,255,255])))
            nitrate = nitrate_state(detect_color(right, np.array([130,50,80]), np.array([160,255,255])))

            status = match_rule(chlorine, nitrate, iron, phosphate)

    return {
        "status": status,
        "chlorine": chlorine,
        "nitrate": nitrate,
        "iron": iron,
        "phosphate": phosphate
    }

# ----------------------------
# FLASK ROUTES
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get base64 image from the browser request
        data = request.json['image']
        img_bytes = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process the frame
        result = process_frame(img)

        # Optional: Log the result to a text file (Railway filesystem is temporary)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CONDITION_FILE, "a") as f:
            f.write(f"{timestamp} | {result['status']} | Cl:{result['chlorine']} | NO3:{result['nitrate']}\n")

        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Railway sets the PORT environment variable automatically
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
