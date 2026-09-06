from flask import Flask, render_template, request, jsonify
import os, uuid, subprocess, cv2, speech_recognition as sr, librosa

app = Flask(__name__, template_folder="templates")

# IMPORTANT: allow large uploads
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_audio(video_path, audio_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def process_video(video_path):
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    upper = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

    cap = cv2.VideoCapture(video_path)
    eye_t = post_t = frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, 1.3, 5)
        eyes = eye.detectMultiScale(gray)
        uppers = upper.detectMultiScale(gray)

        if len(eyes) >= 2:
            eye_t += 1
        if len(faces) > 0 and len(uppers) > 0:
            post_t += 1

        frames += 1

    cap.release()

    eye_score = (eye_t / frames) * 100 if frames else 0
    posture_score = (post_t / frames) * 100 if frames else 0
    visual = (eye_score + posture_score) / 2

    return round(eye_score, 2), round(posture_score, 2), round(visual, 2)

def process_audio(audio_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as src:
            audio = r.record(src)
        text = r.recognize_google(audio)
        wpm = len(text.split()) / librosa.get_duration(filename=audio_path) * 60
        return round(min(wpm / 150, 1) * 100, 2)
    except:
        return 0

@app.route("/")
def index():
    return render_template("video.html")

@app.route("/result")
def result():
    return render_template("results.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(success=False, error="File not received")

    file = request.files["file"]
    uid = str(uuid.uuid4())

    raw = f"{UPLOAD_FOLDER}/{uid}.webm"
    mp4 = f"{UPLOAD_FOLDER}/{uid}.mp4"
    wav = f"{UPLOAD_FOLDER}/{uid}.wav"

    file.save(raw)

    subprocess.run(["ffmpeg", "-y", "-i", raw, mp4],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    extract_audio(mp4, wav)

    eye, posture, visual = process_video(mp4)
    speech = process_audio(wav)
    final = round((visual + speech) / 2, 2)

    return jsonify(success=True, result={
        "eye_score": eye,
        "posture_score": posture,
        "visual_score": visual,
        "speech_score": speech,
        "final_score": final,
        "feedback": {
            "eye": "Good eye contact" if eye >= 50 else "Improve eye contact",
            "posture": "Good posture" if posture >= 50 else "Sit straight",
            "speech": "Confident speech" if speech >= 50 else "Speak clearly"
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)

