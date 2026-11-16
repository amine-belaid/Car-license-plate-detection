import cv2
import easyocr
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialisation bilingue pour détecter "تونس" et les chiffres
print("--- Initialisation VisiteTech IA (Arabe + Anglais) ---")
reader = easyocr.Reader(['ar', 'en'], gpu=False)

# --- CONFIGURATION ---
video_url = "http://172.20.10.5:8080/video" 
processing_now = False
detection_complete = False

def analyze_frame(frame):
    global processing_now, detection_complete
    try:
        h, w = frame.shape[:2]
        # Découpage du rectangle vert (ROI)
        roi = frame[int(h*0.35):int(h*0.65), int(w*0.20):int(w*0.80)]

        # Lecture OCR
        results = reader.readtext(roi)
        
        if results:
            # On récupère tous les morceaux de texte détectés
            parts = [res[1].strip() for res in results if res[2] > 0.20]
            conf = int(max([res[2] for res in results]) * 100)
            
            # On fusionne les morceaux avec un espace propre
            # Cela permet d'avoir "187 تونس 3985" sur une seule ligne
            full_text = " ".join(parts)

            if conf > 15:
                socketio.emit('log_update', {'msg': f"Lecture: {full_text} ({conf}%)"})

            # Condition de validation (Confiance > 20% pour le test papier)
            if conf >= 20 and len(full_text) >= 3:
                detection_complete = True 
                socketio.emit('detected_plate', {
                    'plate': full_text, 
                    'confidence': conf,
                    'status': 'FINAL'
                })
                print(f"VALIDE : {full_text}")
    except Exception as e:
        print(f"Erreur OCR: {e}")
    finally:
        processing_now = False

def generate_frames():
    global processing_now, detection_complete
    cap = cv2.VideoCapture(video_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        success, frame = cap.read()
        if not success: break

        if not detection_complete and not processing_now:
            processing_now = True
            thread = threading.Thread(target=analyze_frame, args=(frame.copy(),))
            thread.daemon = True
            thread.start()

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@socketio.on('reset_scan')
def handle_reset():
    global detection_complete
    detection_complete = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, debug=False, port=5000)