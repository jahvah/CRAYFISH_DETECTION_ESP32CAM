from flask import Flask, request, render_template_string, jsonify
import torch
import cv2
import numpy as np
import base64
import json

app = Flask(__name__)

# Load CUSTOM YOLOv5 model for crayfish
model = torch.hub.load(
    'C:/Users/Elijah/YoloRepo/yolov5',
    'custom',
    path='C:/Users/Elijah/YoloRepo/yolov5/runs/train/mixed_crayfish/mixed_crayfishv12/weights/best.pt',
    source='local'
)
model.conf = 0.20

crayfish_class_index = 0
latest_image = None
latest_detection_result = {
    'status': 'no_detection',
    'message': 'No detection available yet',
    'size_category': None
}

# Load CUSTOM YOLOv5 model for eggs
egg_model = torch.hub.load(
    'C:/Users/Elijah/YoloRepo/yolov5',
    'custom',
    path='C:/Users/Elijah/YoloRepo/yolov5/runs/train/eggs_crayfish/eggscrayfishv1/weights/best.pt',
    source='local'
)
egg_model.conf = 0.20
egg_class_index = 0  # adjust if your model has multiple classes

latest_egg_detection_result = {
    'status': 'no_detection',
    'message': 'No egg detection available yet',
    'eggs_detected': 0
}

# Latest pH status from ESP32
latest_ph_status = {
    'ph': None,
    'raw': None,
    'water_state': 'unknown',   # 'acidic', 'basic', 'normal'
    'pump_state': 'off'        # 'on' or 'off'
}

# PIXEL SCALE
PIXELS_PER_CM = 49

# SIZE CATEGORY RANGES (in cm, by length = max(width, height))
# SMALL: length < 10 cm
# BIG:   length >= 10 cm
SMALL_CM_MAX = 10.0  # below this = SMALL, else BIG


def classify_size(width_cm, height_cm):
    """Classify by length (longer side) in cm. Returns (category, range_string)."""
    length_cm = max(width_cm, height_cm)
    if length_cm < SMALL_CM_MAX:
        return "SMALL", f"< {SMALL_CM_MAX} cm"
    return "BIG", f">= {SMALL_CM_MAX} cm"


@app.route('/upload', methods=['POST'])
def upload():
    global latest_image
    global latest_detection_result
    global latest_egg_detection_result

    img_bytes = request.data
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return "invalid_image", 400

    # YOLOv5 expects RGB; OpenCV gives BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    detections = results.xyxy[0].cpu().numpy()

    print("\n----- YOLO DEBUG (Real-time Camera) -----")
    print("Detections found:", len(detections))

    crayfish_found = False
    detection_list = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == crayfish_class_index:
            crayfish_found = True

            width_px = int(x2 - x1)
            height_px = int(y2 - y1)

            # Convert to cm
            width_cm = width_px / PIXELS_PER_CM
            height_cm = height_px / PIXELS_PER_CM

            # Determine category (by length = max dimension, in cm)
            size_category, size_range = classify_size(width_cm, height_cm)

            print("Crayfish detected")
            print("Width:", round(width_cm, 2), "cm")
            print("Height:", round(height_cm, 2), "cm")
            print("Size category:", size_category, "(" + size_range + ")")
            print("Confidence:", round(float(conf), 3))
            print("----------------------")

            detection_list.append({
                'width_cm': round(width_cm, 2),
                'height_cm': round(height_cm, 2),
                'size_category': size_category,
                'size_range': size_range,
                'confidence': round(float(conf), 3)
            })

            # Draw bounding box
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            label = f"{size_category} ({size_range}) {width_cm:.1f}-{height_cm:.1f}cm {conf*100:.0f}%"

            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # --- EGG DETECTION ---
    egg_results = egg_model(img_rgb)
    egg_detections = egg_results.xyxy[0].cpu().numpy()
    eggs_found = 0

    for det in egg_detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == egg_class_index:
            eggs_found += 1
            # Draw bounding box for eggs (red)
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2
            )
            label = f"EGG {conf*100:.0f}%"
            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    latest_egg_detection_result = {
        'status': 'detected' if eggs_found > 0 else 'no_detection',
        'eggs_detected': eggs_found
    }

    latest_image = img

    # Update latest detection result for Arduino to retrieve
    if crayfish_found and detection_list:
        latest_detection_result = {
            'status': 'detected',
            'detections_found': len(detections),
            'crayfish_found': len(detection_list),
            'details': detection_list,
            'size_category': detection_list[0]['size_category']
        }
        return "detected", 200
    else:
        print("No crayfish detected")
        latest_detection_result = {
            'status': 'no_detection',
            'message': 'No crayfish detected in latest frame'
        }
        return "not_detected", 200


@app.route('/')
def index():
    global latest_image

    if latest_image is None:
        return "Waiting for image..."

    _, buffer = cv2.imencode('.jpg', latest_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return f"""
    <h2>YOLOv5 Crayfish Detection</h2>
    <img src="data:image/jpeg;base64,{img_base64}" width="700"/>
    <p>Refresh page to see latest image</p>
    """


@app.route('/test', methods=['GET', 'POST'])
def test_upload():
    """Real-time camera feed from ESP32-S3"""
    global latest_image
    global latest_detection_result
    global latest_egg_detection_result
    global latest_ph_status

    if latest_image is None:
        return '''
        <html>
        <head>
            <title>ESP32 Crayfish Detection - Real-time Camera</title>
            <meta http-equiv="refresh" content="2">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
                .container { max-width: 900px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h2 { color: #333; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .status.waiting { background-color: #fff3cd; color: #856404; border: 1px solid #ffc107; }
                img { max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 5px; margin-top: 10px; }
                .info { margin-top: 15px; font-size: 14px; color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>🎥 ESP32-S3 Real-time Crayfish Detection</h2>
                <div class="status waiting">
                    ⏳ Waiting for first camera frame from ESP32...
                </div>
                <div class="info">
                    This page auto-refreshes every 2 seconds<br>
                    Make sure your ESP32-S3 is connected and posting frames to /upload endpoint
                </div>
            </div>
        </body>
        </html>
        ''', 200

    # Display latest image with detection results
    _, buffer = cv2.imencode('.jpg', latest_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Build detection info if available
    detection_info = ""
    if latest_detection_result:
        if latest_detection_result.get('status') == 'detected':
            crayfish_count = latest_detection_result.get('crayfish_found', 0)
            size_cat = latest_detection_result.get('size_category', 'Unknown')
            detection_info = f"""
            <div class="status detected">
                ✓ <b>Crayfish Detected!</b><br>
                Count: {crayfish_count}<br>
                Size: <b>{size_cat}</b><br>
                (Check console for detailed measurements)
            </div>
            """
        else:
            detection_info = """
            <div class="status no-detection">
                • No crayfish detected in current frame
            </div>
            """

    # Build egg info if available
    egg_info = ""
    if latest_egg_detection_result:
        if latest_egg_detection_result.get('status') == 'detected':
            egg_count = latest_egg_detection_result.get('eggs_detected', 0)
            egg_info = f"""
            <div class="status detected" style="background-color:#f8d7da; color:#721c24; border-left-color:#dc3545;">
                🥚 Eggs Detected! Count: {egg_count}
            </div>
            """
        else:
            egg_info = """
            <div class="status no-detection">
                • No eggs detected in current frame
            </div>
            """

    # Build pH info
    ph_info = ""
    if latest_ph_status:
        ph_val = latest_ph_status.get('ph')
        water_state = latest_ph_status.get('water_state', 'unknown')
        pump_state = latest_ph_status.get('pump_state', 'off')
        if ph_val is not None:
            ph_info = f"""
            <div class="status ph-status" style="background-color:#d1ecf1; color:#0c5460; border-left-color:#17a2b8;">
                💧 <b>Water pH:</b> {ph_val:.2f}<br>
                State: <b>{water_state}</b><br>
                Pump: <b>{pump_state.upper()}</b>
            </div>
            """

    return f"""
    <html>
    <head>
        <title>ESP32 Crayfish Detection - Real-time Camera</title>
        <meta http-equiv="refresh" content="1">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
            .container {{ max-width: 900px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h2 {{ color: #333; }}
            .status {{ padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid; }}
            .status.detected {{ background-color: #d4edda; color: #155724; border-left-color: #28a745; }}
            .status.no-detection {{ background-color: #e2e3e5; color: #383d41; border-left-color: #6c757d; }}
            .camera-feed {{ margin-top: 15px; text-align: center; }}
            .camera-feed img {{ max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 5px; }}
            .info {{ margin-top: 15px; font-size: 13px; color: #666; }}
            .api-info {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 12px; }}
            .refresh-time {{ color: #999; font-size: 11px; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎥 ESP32-S3 Real-time Crayfish Detection</h2>
            {detection_info}
            {egg_info}
            {ph_info}
            <div class="camera-feed">
                <img src="data:image/jpeg;base64,{img_base64}" alt="Camera Feed"/>
            </div>
            <div class="info">
                📡 <b>Camera Status:</b> Streaming<br>
                🔄 Page auto-refreshes every 1 second<br>
                {f'🐉 <b>Latest Detection:</b> {latest_detection_result.get("size_category", "N/A")}' if latest_detection_result and latest_detection_result.get('status') == 'detected' else '🐉 <b>Latest Detection:</b> Waiting...'}
            </div>
            <div class="api-info">
                <b>Arduino Integration:</b><br>
                Arduino polls <code>/api/detect</code> endpoint to get size_category<br>
                Arduino polls <code>/api/detect_eggs</code> endpoint to get egg count<br>
                Servo motor duration depends on detected size
            </div>
            <div class="refresh-time">
                Last update: just now | Auto-refresh enabled
            </div>
        </div>
    </body>
    </html>
    """


@app.route('/api/detect', methods=['GET'])
def api_detect():
    """API endpoint for Arduino to get latest detection results"""
    global latest_detection_result
    return jsonify(latest_detection_result), 200


@app.route('/api/detect_eggs', methods=['GET'])
def api_detect_eggs():
    """API endpoint for Arduino to get latest egg detection results"""
    global latest_egg_detection_result
    return jsonify(latest_egg_detection_result), 200


@app.route('/api/ph', methods=['POST'])
def api_ph_update():
    """ESP32 posts latest pH reading and pump state here"""
    global latest_ph_status

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'status': 'error', 'message': 'invalid json'}), 400

    ph = data.get('ph')
    raw = data.get('raw')
    water_state = data.get('water_state', 'unknown')
    pump_state = data.get('pump_state', 'off')

    latest_ph_status = {
        'ph': ph,
        'raw': raw,
        'water_state': water_state,
        'pump_state': pump_state
    }

    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
