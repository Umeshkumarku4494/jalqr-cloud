import cv2
import numpy as np
from flask import Flask, Response, render_template_string

# ----------------------------
# CONFIG
# ----------------------------
conditions_file = r"C:\Users\Lenovo\Desktop\JalQR\Conditions.txt"

print("JALQR Classification Engine")
print("Loading rules...")

# ----------------------------
# LOAD RULES
# ----------------------------
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

print(f"{len(rules)} rules loaded.")

# ----------------------------
# QUICK MESSAGE MAPPING
# ----------------------------
quick_messages = {
    "SAFE": "Drink normally",
    "CAUTION": "Safe but filter recommended",
    "RISKY": "Boil before drinking",
    "UNSAFE": "Do not drink",
    "CHECK": "Adjust strip position"
}

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def detect_color(zone, lower, upper):
    if zone.size == 0:
        return 0
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) / (zone.shape[0] * zone.shape[1])

def sharpen(image):
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
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

# ----------------------------
# CORE PROCESSING FUNCTION
# ----------------------------
def process_frame(frame):

    frame = sharpen(frame)
    status = "CHECK"

    chlorine = "NA"
    nitrate = "NA"
    iron = "NA"
    phosphate = "NA"

    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(frame)

    if bbox is not None and data:

        pts = bbox[0].astype(int)
        qr_x, qr_y, qr_w, qr_h = cv2.boundingRect(pts)
        frame_h, frame_w, _ = frame.shape

        space_top = qr_y
        space_bottom = frame_h - (qr_y + qr_h)
        space_left = qr_x
        space_right = frame_w - (qr_x + qr_w)

        max_border = min(space_top, space_bottom, space_left, space_right)
        desired_border = int(0.35 * qr_w)
        border = min(desired_border, max_border - 5)

        if border > 15:

            top = frame[qr_y - border:qr_y, qr_x:qr_x + qr_w]
            bottom = frame[qr_y + qr_h:qr_y + qr_h + border, qr_x:qr_x + qr_w]
            left = frame[qr_y:qr_y + qr_h, qr_x - border:qr_x]
            right = frame[qr_y:qr_y + qr_h, qr_x + qr_w:qr_x + qr_w + border]

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
# WEB MODE (Advanced UI)
# ----------------------------
from flask import Flask, Response, render_template_string, jsonify
import threading

app = Flask(__name__)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

latest_result = {
    "status": "CHECK",
    "chlorine": "NA",
    "nitrate": "NA",
    "iron": "NA",
    "phosphate": "NA"
}

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>JALQR – Smart Water Scanner</title>
<style>
body {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    font-family: Arial;
    text-align: center;
    color: white;
}

.container {
    margin-top: 20px;
}

.video-box {
    border: 5px solid white;
    border-radius: 15px;
    overflow: hidden;
    width: 700px;
    margin: auto;
}

.status {
    font-size: 28px;
    margin-top: 15px;
    padding: 10px;
    border-radius: 10px;
    display: inline-block;
}

.chem-box {
    margin-top: 20px;
    background: rgba(255,255,255,0.2);
    padding: 15px;
    border-radius: 10px;
    display: inline-block;
    font-size: 18px;
}
</style>
</head>
<body>

<div class="container">
<h1>💧 JALQR – AI Water Scanner</h1>

<div class="video-box">
<img src="/video_feed">
</div>

<div id="statusBox" class="status">STATUS: CHECK</div>

<div class="chem-box">
<h3>Detected Chemicals</h3>
<div id="chlorine"></div>
<div id="nitrate"></div>
<div id="iron"></div>
<div id="phosphate"></div>
</div>
</div>

<script>
function updateData() {
    fetch('/data')
    .then(response => response.json())
    .then(data => {

        document.getElementById("chlorine").innerHTML = "Chlorine: " + data.chlorine;
        document.getElementById("nitrate").innerHTML = "Nitrate: " + data.nitrate;
        document.getElementById("iron").innerHTML = "Iron: " + data.iron;
        document.getElementById("phosphate").innerHTML = "Phosphate: " + data.phosphate;

        let statusBox = document.getElementById("statusBox");
        statusBox.innerHTML = "STATUS: " + data.status;

        if (data.status == "SAFE")
            statusBox.style.backgroundColor = "green";
        else if (data.status == "UNSAFE")
            statusBox.style.backgroundColor = "red";
        else if (data.status == "CAUTION")
            statusBox.style.backgroundColor = "orange";
        else
            statusBox.style.backgroundColor = "gray";
    });
}

setInterval(updateData, 1000);
</script>

</body>
</html>
"""

def generate_frames():
    global latest_result

    while True:
        success, frame = camera.read()
        if not success:
            break

        result = process_frame(frame)
        latest_result = result

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    return jsonify(latest_result)


# ----------------------------
# MAIN ENTRY
# ----------------------------
if __name__ == "__main__":
    print("Starting JALQR Web Interface...")
    app.run(host="0.0.0.0", port=5000, debug=False)
