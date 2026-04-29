import cv2
import numpy as np
import dlib
import os
import threading
import pyttsx3
import pyaudio
from vosk import Model, KaldiRecognizer
from PIL import Image, ImageTk
import tkinter as tk
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time

# تهيئة مثيل pyttsx3 عالمي
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
speech_lock = threading.Lock()  # Lock to prevent concurrent speech

# تعريف دالة speak مع قفل
def speak(text):
    with speech_lock:
        try:
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            pass

# تحميل نموذج YOLO الأصلي
try:
    yolo_model = YOLO(r"c:\Users\HP\Desktop\مشروع 1\My First Project.v1i.yolov8\yolov8n.pt")
    yolo_model.to('cpu')
except Exception as e:
    print(f"خطأ في تحميل نموذج YOLO الأصلي: {e}")
    exit(1)

# تحميل النموذج المدرب الخاص بك
try:
    my_yolo_model = YOLO(r"c:\Users\HP\Desktop\مشروع 1\My First Project.v1i.yolov8\yolov8n.pt")
    my_yolo_model.to('cpu')
except Exception as e:
    print(f"خطأ في تحميل النموذج المدرب: {e}")
    exit(1)

# تحميل نماذج Dlib
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"c:\Users\HP\Downloads\Compressed\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1(r"c:\Users\HP\Downloads\Compressed\dlib_face_recognition_resnet_model_v1.dat\dlib_face_recognition_resnet_model_v1.dat")
except Exception as e:
    print(f"خطأ في تحميل نماذج Dlib: {e}")
    exit(1)

# تحميل الوجوه المعروفة
KNOWN_FACES_DIR = r"c:\Users\HP\Desktop\persons"
known_faces_encodings = []
known_faces_names = []

def load_known_faces(directory):
    for person_name in os.listdir(directory):
        person_folder = os.path.join(directory, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(person_folder, filename)
                    image = cv2.imread(path)
                    if image is None:
                        continue
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)
                    if faces:
                        shape = predictor(gray, faces[0])
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        encoding = np.array(face_rec_model.compute_face_descriptor(rgb, shape))
                        known_faces_encodings.append(encoding)
                        known_faces_names.append(person_name)

load_known_faces(KNOWN_FACES_DIR)
print(f"Loaded {len(known_faces_names)} known faces.")

# إعداد الكاميرا
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("فشل فتح الكاميرا")
except Exception as e:
    print(f"خطأ في فتح الكاميرا: {e}")
    exit(1)

frame_skip = 15
frame_count = 0
current_faces = []
ocr_results = []
frozen_frame = None  # إطار مجمد لعرضه أثناء قراءة النصوص
is_reading_text = False  # حالة القراءة

# تهيئة PaddleOCR
try:
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
except Exception as e:
    print(f"خطأ في تهيئة PaddleOCR: {e}")
    exit(1)

# متغيرات تتبع النطق
last_spoken_face = ""
last_face_time = 0
last_seen_faces = set()  # لتتبع الوجوه الموجودة حاليًا
active_texts = set()  # النصوص المرئية حاليًا
spoken_texts = set()  # النصوص التي تم نطقها
paused_texts = set()  # النصوص الموقوفة مؤقتًا
last_seen_objects = set()  # لتتبع الأغراض الموجودة حاليًا

# متغيرات التحكم في الوظائف
enable_face_recognition = False
enable_ocr = False
enable_object_detection = False

# متغيرات الأزرار
face_button = None
ocr_button = None
object_button = None

def preprocess_frame(frame):
    # تحويل الصورة إلى تدرج الرمادي فقط
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # إعادة تحويلها إلى BGR لتتوافق مع PaddleOCR
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame

def recognize_faces(frame):
    global current_faces, last_spoken_face, last_face_time, last_seen_faces
    if not enable_face_recognition:
        return
    try:
        small = cv2.resize(frame, (800, 600))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        detected_faces = []
        current_face_names = set()
        for face in faces:
            scale_x = frame.shape[1] / 800
            scale_y = frame.shape[0] / 600
            x1, y1 = int(face.left() * scale_x), int(face.top() * scale_y)
            x2, y2 = int(face.right() * scale_x), int(face.bottom() * scale_y)
            face_rect = dlib.rectangle(x1, y1, x2, y2)
            try:
                shape = predictor(frame, face_rect)
                encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))
            except:
                continue
            best_match = "Unknown"
            if known_faces_encodings:
                distances = np.linalg.norm(known_faces_encodings - encoding, axis=1)
                idx = np.argmin(distances)
                if distances[idx] < 0.6:
                    best_match = known_faces_names[idx]
            detected_faces.append((x1, y1, x2, y2, best_match))
            current_face_names.add(best_match)
            if best_match != "Unknown" and best_match not in last_seen_faces:
                speak(f"{best_match} detected")
                last_spoken_face = best_match
                last_face_time = time.time()
        current_faces = detected_faces
        last_seen_faces = current_face_names
        if not current_face_names:
            last_seen_faces.clear()
    except Exception as e:
        print(f"خطأ في التعرف على الوجوه: {e}")

def run_ocr(frame):
    global ocr_results, active_texts, spoken_texts, paused_texts, frozen_frame, is_reading_text
    if not enable_ocr:
        return
    try:
        # Preprocess frame
        frame_processed = preprocess_frame(frame.copy())
        # Resize frame for faster processing
        frame_processed = cv2.resize(frame_processed, (1280, 720))
        result = ocr_model.ocr(frame_processed, cls=True)
        ocr_results = []
        current_frame_texts = []
        if result and result[0]:  # Check if result is not None and not empty
            # Collect texts with their bounding boxes
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                score = line[1][1]
                # Filter out low-confidence or short texts
                if score > 0.4 and len(text.strip()) >= 3:
                    pts = np.array(box).astype(np.int32)
                    # Scale bounding box back to original frame size
                    scale_x = frame.shape[1] / 1280
                    scale_y = frame.shape[0] / 720
                    pts = (pts * np.array([scale_x, scale_y])).astype(np.int32)
                    # Calculate bounding box area to filter small texts
                    area = cv2.contourArea(pts)
                    if area > 100:  # Ignore very small text boxes
                        current_frame_texts.append((text, pts, np.mean(pts[:, 1])))  # Store y-coordinate for sorting
            # Sort texts by y-coordinate (top to bottom)
            current_frame_texts.sort(key=lambda x: x[2])
            # Process sorted texts
            active_texts.clear()
            if current_frame_texts:  # If texts are detected
                with speech_lock:
                    frozen_frame = frame.copy()  # Freeze the current frame
                    is_reading_text = True
                for text, pts, _ in current_frame_texts:
                    ocr_results.append((pts, text))
                    active_texts.add(text)
                    # النطق مرة واحدة فقط عند الظهور الأول
                    if text not in spoken_texts and text not in paused_texts:
                        speak(text)
                        spoken_texts.add(text)
                with speech_lock:
                    is_reading_text = False  # Resume live feed after reading
                    frozen_frame = None
    except Exception as e:
        print(f"خطأ في تشغيل OCR: {e}")

def run_object_detection(frame):
    global enable_object_detection, last_seen_objects
    if not enable_object_detection:
        return
    detected_objects = []
    current_objects = set()
    try:
        # استخدام النموذج الأصلي
        results = yolo_model(frame, conf=0.4)
        if results:
            for result in results:
                if hasattr(result, 'boxes'):
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        label = yolo_model.names[cls_id]
                        detected_objects.append(label)
                        current_objects.add(label)
                        if label not in last_seen_objects:
                            speak(f"{label} detected (original model)")
                            last_seen_objects.add(label)
        # استخدام النموذج المدرب
        my_results = my_yolo_model(frame, conf=0.4)
        if my_results:
            for result in my_results:
                if hasattr(result, 'boxes'):
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        label = my_yolo_model.names[cls_id]
                        detected_objects.append(label)
                        current_objects.add(label)
                        if label not in last_seen_objects:
                            speak(f"{label} detected (my model)")
                            last_seen_objects.add(label)
        last_seen_objects = current_objects
        if not current_objects:
            last_seen_objects.clear()
        return detected_objects
    except Exception as e:
        print(f"خطأ في كشف الأغراض: {e}")
        return []

def check_disappeared_items():
    global active_texts, spoken_texts, paused_texts, last_seen_faces, last_seen_objects
    disappearance_lock = threading.Lock()
    while True:
        time.sleep(5)
        with disappearance_lock:
            spoken_texts -= (spoken_texts - active_texts)
            paused_texts &= active_texts
            if not last_seen_faces:
                last_seen_faces.clear()
            if not last_seen_objects:
                last_seen_objects.clear()
            if not active_texts:
                active_texts.clear()

threading.Thread(target=check_disappeared_items, daemon=True).start()

# إعداد Vosk للتعرف الصوتي
try:
    model_path = r"c:\Users\HP\Downloads\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"النموذج غير موجود في {model_path}. تحميل النموذج من https://alphacephei.com/vosk/models")
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)
except Exception as e:
    print(f"خطأ في تهيئة Vosk: {e}")
    exit(0)

def listen_for_commands():
    global enable_face_recognition, enable_ocr, enable_object_detection, current_faces, ocr_results, paused_texts
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        stream.start_stream()
        speak("جاهز لاستقبال الأوامر الصوتية. قل one للوجوه، two للأغراض بالنموذج الأصلي، both للأغراض بالنموذجين، three للنصوص، pause لإيقاف النصوص مؤقتًا، start لاستئناف النصوص، off لإيقاف التابع، أو exit لإنهاء البرنامج.")
        
        while True:
            if stream.is_active():
                data = stream.read(4000, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    command = eval(result)['text'].lower().strip()
                    print(f"الأمر المستلم: {command}")
                    
                    if command == "one":
                        if enable_face_recognition:
                            enable_face_recognition = False
                            current_faces = []
                            last_seen_faces.clear()
                            face_button.config(text="تفعيل التعرف على الوجوه")
                            speak("تم إيقاف التعرف على الوجوه")
                        else:
                            enable_face_recognition = True
                            enable_ocr = False
                            enable_object_detection = False
                            paused_texts.clear()
                            ocr_results = []
                            active_texts.clear()
                            spoken_texts.clear()
                            last_seen_objects.clear()
                            face_button.config(text="إيقاف التعرف على الوجوه")
                            ocr_button.config(text="تفعيل التعرف على النصوص")
                            object_button.config(text="تفعيل التعرف على الأغراض")
                            speak("تم تفعيل التعرف على الوجوه")
                        status_label.config(text="الوضع: التعرف على الوجوه" if enable_face_recognition else "الوضع: معطل")
                        print("التعرف على الوجوه " + ("مفعل" if enable_face_recognition else "معطل"))
                    elif command == "two":
                        if enable_object_detection and not enable_ocr:
                            enable_object_detection = False
                            last_seen_objects.clear()
                            object_button.config(text="تفعيل التعرف على الأغراض")
                            speak("تم إيقاف التعرف على الأغراض")
                        else:
                            enable_object_detection = True
                            enable_face_recognition = False
                            enable_ocr = False
                            current_faces = []
                            paused_texts.clear()
                            ocr_results = []
                            active_texts.clear()
                            spoken_texts.clear()
                            last_seen_faces.clear()
                            face_button.config(text="تفعيل التعرف على الوجوه")
                            ocr_button.config(text="تفعيل التعرف على النصوص")
                            object_button.config(text="إيقاف التعرف على الأغراض")
                            speak("تم تفعيل التعرف على الأغراض بالنموذج الأصلي")
                        status_label.config(text="الوضع: التعرف على الأغراض (الأصلي)" if enable_object_detection else "الوضع: معطل")
                        print("التعرف على الأغراض " + ("مفعل بالنموذج الأصلي" if enable_object_detection else "معطل"))
                    elif command == "both":
                        if enable_object_detection and not enable_ocr:
                            enable_object_detection = False
                            last_seen_objects.clear()
                            object_button.config(text="تفعيل التعرف على الأغراض")
                            speak("تم إيقاف التعرف على الأغراض")
                        else:
                            enable_object_detection = True
                            enable_face_recognition = False
                            enable_ocr = False
                            current_faces = []
                            paused_texts.clear()
                            ocr_results = []
                            active_texts.clear()
                            spoken_texts.clear()
                            last_seen_faces.clear()
                            face_button.config(text="تفعيل التعرف على الوجوه")
                            ocr_button.config(text="تفعيل التعرف على النصوص")
                            object_button.config(text="إيقاف التعرف على الأغراض")
                            speak("تم تفعيل التعرف على الأغراض باستخدام النموذجين")
                        status_label.config(text="الوضع: التعرف على الأغراض (النموذجين)" if enable_object_detection else "الوضع: معطل")
                        print("التعرف على الأغراض " + ("مفعل باستخدام النموذجين" if enable_object_detection else "معطل"))
                    elif command == "three":
                        if enable_ocr:
                            enable_ocr = False
                            ocr_results = []
                            active_texts.clear()
                            spoken_texts.clear()
                            paused_texts.clear()
                            frozen_frame = None
                            is_reading_text = False
                            ocr_button.config(text="تفعيل التعرف على النصوص")
                            speak("تم إيقاف التعرف على النصوص")
                        else:
                            enable_ocr = True
                            enable_face_recognition = False
                            enable_object_detection = False
                            current_faces = []
                            last_seen_faces.clear()
                            last_seen_objects.clear()
                            face_button.config(text="تفعيل التعرف على الوجوه")
                            ocr_button.config(text="إيقاف التعرف على النصوص")
                            object_button.config(text="تفعيل التعرف على الأغراض")
                            speak("تم تفعيل التعرف على النصوص")
                        status_label.config(text="الوضع: التعرف على النصوص" if enable_ocr else "الوضع: معطل")
                        print("التعرف على النصوص " + ("مفعل" if enable_ocr else "معطل"))
                    elif command == "pause" and enable_ocr:
                        paused_texts.update(active_texts)
                        speak("تم إيقاف قراءة النصوص مؤقتًا")
                        status_label.config(text="الوضع: التعرف على النصوص (موقوف مؤقتًا)")
                        print("قراءة النصوص موقوفة مؤقتًا")
                    elif command == "start" and enable_ocr:
                        paused_texts.clear()
                        spoken_texts.clear()
                        speak("تم استئناف قراءة النصوص")
                        status_label.config(text="الوضع: التعرف على النصوص")
                        print("قراءة النصوص مستأنفة")
                    elif command in ["off", "stop"]:
                        enable_face_recognition = False
                        enable_ocr = False
                        enable_object_detection = False
                        current_faces = []
                        last_seen_faces.clear()
                        ocr_results = []
                        active_texts.clear()
                        spoken_texts.clear()
                        paused_texts.clear()
                        last_seen_objects.clear()
                        frozen_frame = None
                        is_reading_text = False
                        face_button.config(text="تفعيل التعرف على الوجوه")
                        ocr_button.config(text="تفعيل التعرف على النصوص")
                        object_button.config(text="تفعيل التعرف على الأغراض")
                        speak("تم إيقاف جميع الوظائف")
                        status_label.config(text="الوضع: معطل")
                        print("جميع الوظائف معطلة")
                    elif command == "exit":
                        enable_face_recognition = False
                        enable_ocr = False
                        enable_object_detection = False
                        current_faces = []
                        ocr_results = []
                        last_seen_faces.clear()
                        active_texts.clear()
                        spoken_texts.clear()
                        paused_texts.clear()
                        last_seen_objects.clear()
                        frozen_frame = None
                        is_reading_text = False
                        face_button.config(text="تفعيل التعرف على الوجوه")
                        ocr_button.config(text="تفعيل التعرف على النصوص")
                        object_button.config(text="تفعيل التعرف على الأغراض")
                        speak("تم إنهاء البرنامج")
                        status_label.config(text="الوضع: معطل")
                        print("إنهاء البرنامج")
                        stream.stop_stream()
                        stream.close()
                        p.terminate()
                        cap.release()
                        cv2.destroyAllWindows()
                        engine.stop()
                        root.quit()
                        break
                else:
                    partial_result = recognizer.PartialResult()
                    print(f"النتيجة الجزئية: {eval(partial_result)['partial']}")
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        print(f"خطأ في التعرف الصوتي: {e}")

# إنشاء واجهة Tkinter
root = tk.Tk()
root.title("Smart Vision")
root.geometry("1000x700")

# إنشاء إطار لعرض الكاميرا
video_label = tk.Label(root)
video_label.pack(pady=10)

# إنشاء إطار للأزرار
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# متغيرات التحكم في الوظائف
def toggle_face_recognition():
    global enable_face_recognition, enable_ocr, enable_object_detection, current_faces, last_seen_faces, paused_texts
    if enable_face_recognition:
        enable_face_recognition = False
        current_faces = []
        last_seen_faces.clear()
        face_button.config(text="تفعيل التعرف على الوجوه")
        speak("تم إيقاف التعرف على الوجوه")
    else:
        enable_face_recognition = True
        enable_ocr = False
        enable_object_detection = False
        paused_texts.clear()
        ocr_results = []
        active_texts.clear()
        spoken_texts.clear()
        last_seen_objects.clear()
        face_button.config(text="إيقاف التعرف على الوجوه")
        ocr_button.config(text="تفعيل التعرف على النصوص")
        object_button.config(text="تفعيل التعرف على الأغراض")
        speak("تم تفعيل التعرف على الوجوه")
    status_label.config(text="الوضع: التعرف على الوجوه" if enable_face_recognition else "الوضع: معطل")
    print("التعرف على الوجوه " + ("مفعل" if enable_face_recognition else "معطل"))

def toggle_ocr():
    global enable_ocr, enable_face_recognition, enable_object_detection, current_faces, active_texts, spoken_texts, paused_texts
    if enable_ocr:
        enable_ocr = False
        ocr_results = []
        active_texts.clear()
        spoken_texts.clear()
        paused_texts.clear()
        frozen_frame = None
        is_reading_text = False
        ocr_button.config(text="تفعيل التعرف على النصوص")
        speak("تم إيقاف التعرف على النصوص")
    else:
        enable_ocr = True
        enable_face_recognition = False
        enable_object_detection = False
        current_faces = []
        last_seen_faces.clear()
        last_seen_objects.clear()
        face_button.config(text="تفعيل التعرف على الوجوه")
        ocr_button.config(text="إيقاف التعرف على النصوص")
        object_button.config(text="تفعيل التعرف على الأغراض")
        speak("تم تفعيل التعرف على النصوص")
    status_label.config(text="الوضع: التعرف على النصوص" if enable_ocr else "الوضع: معطل")
    print("التعرف على النصوص " + ("مفعل" if enable_ocr else "معطل"))

def toggle_object_detection():
    global enable_object_detection, enable_face_recognition, enable_ocr, current_faces, last_seen_objects, paused_texts
    if enable_object_detection:
        enable_object_detection = False
        last_seen_objects.clear()
        object_button.config(text="تفعيل التعرف على الأغراض")
        speak("تم إيقاف التعرف على الأغراض")
    else:
        enable_object_detection = True
        enable_face_recognition = False
        enable_ocr = False
        current_faces = []
        paused_texts.clear()
        ocr_results = []
        active_texts.clear()
        spoken_texts.clear()
        last_seen_faces.clear()
        face_button.config(text="تفعيل التعرف على الوجوه")
        ocr_button.config(text="تفعيل التعرف على النصوص")
        object_button.config(text="إيقاف التعرف على الأغراض")
        speak("تم تفعيل التعرف على الأغراض")
    status_label.config(text="الوضع: التعرف على الأغراض" if enable_object_detection else "الوضع: معطل")
    print("التعرف على الأغراض " + ("مفعل" if enable_object_detection else "معطل"))

def stop_program():
    global enable_face_recognition, enable_ocr, enable_object_detection, current_faces, last_seen_faces, active_texts, spoken_texts, paused_texts, last_seen_objects, frozen_frame, is_reading_text
    enable_face_recognition = False
    enable_ocr = False
    enable_object_detection = False
    current_faces = []
    last_seen_faces.clear()
    active_texts.clear()
    spoken_texts.clear()
    paused_texts.clear()
    last_seen_objects.clear()
    frozen_frame = None
    is_reading_text = False
    face_button.config(text="تفعيل التعرف على الوجوه")
    ocr_button.config(text="تفعيل التعرف على النصوص")
    object_button.config(text="تفعيل التعرف على الأغراض")
    status_label.config(text="الوضع: معطل")
    speak("تم إيقاف جميع الوظائف")
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

# إنشاء الأزرار
face_button = tk.Button(button_frame, text="تفعيل التعرف على الوجوه", command=toggle_face_recognition)
face_button.pack(side=tk.LEFT, padx=5)
ocr_button = tk.Button(button_frame, text="تفعيل التعرف على النصوص", command=toggle_ocr)
ocr_button.pack(side=tk.LEFT, padx=5)
object_button = tk.Button(button_frame, text="تفعيل التعرف على الأغراض", command=toggle_object_detection)
object_button.pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="إيقاف الكل", command=stop_program).pack(side=tk.LEFT, padx=5)

# ملصق لحالة الوظيفة
status_label = tk.Label(root, text="الوضع: معطل", font=("Arial", 12))
status_label.pack(pady=10)

# دالة لتحديث إطار الكاميرا
def update_frame():
    global frame_count, frozen_frame, is_reading_text
    try:
        ret, frame = cap.read()
        if not ret:
            stop_program()
            return

        frame_count += 1
        if frame_count % frame_skip == 0:
            if enable_face_recognition:
                threading.Thread(target=recognize_faces, args=(frame.copy(),), daemon=True).start()
            if enable_ocr:
                threading.Thread(target=run_ocr, args=(frame.copy(),), daemon=True).start()
            if enable_object_detection:
                threading.Thread(target=run_object_detection, args=(frame.copy(),), daemon=True).start()

        # Display frozen frame if reading text, otherwise show live frame
        display_frame = frozen_frame if is_reading_text and frozen_frame is not None else frame

        # رسم الوجوه
        for (x1, y1, x2, y2, name) in current_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # رسم النصوص
        for pts, text in ocr_results:
            cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            x, y = np.mean(pts, axis=0).astype(int)
            cv2.putText(display_frame, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # عرض نتائج YOLO
        if enable_object_detection:
            results = yolo_model(display_frame, conf=0.4)
            if results:
                for result in results:
                    if hasattr(result, 'plot'):
                        display_frame = result.plot()

        # تحويل الإطار إلى صورة Tkinter
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_frame = cv2.resize(display_frame, (640, 480))
        img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # تحديث الإطار كل 30 مللي ثانية
        root.after(30, update_frame)
    except Exception as e:
        print(f"خطأ في تحديث الإطار: {e}")
        stop_program()

# بدء خيط التعرف الصوتي
command_thread = threading.Thread(target=listen_for_commands, daemon=True)
command_thread.start()

# بدء تحديث الكاميرا
update_frame()

# تشغيل التطبيق
try:
    root.mainloop()
except Exception as e:
    print(f"خطأ في تشغيل التطبيق: {e}")

# تنظيف الموارد
cap.release()
cv2.destroyAllWindows()
engine.stop()
