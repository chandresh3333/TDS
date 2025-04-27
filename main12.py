import cv2
import numpy as np
import pyttsx3
import threading

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Function to play alert message
def play_alert(message):
    engine.say(message)
    engine.runAndWait()

# Load OpenCV's Haar cascade models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera")
    exit()

# Counters
sleep = 0
drowsy = 0
active = 0
status = "No Face Detected"
color = (0, 255, 255)  # Yellow
alert_triggered = False  # To avoid multiple voice triggers

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        status = "No Face Detected"
        color = (0, 255, 255)  # Yellow
        alert_triggered = False  # Reset alert when no face
    else:
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Detect eyes within face ROI
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

            if len(eyes) == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)  # Blue
                    if not alert_triggered:
                        threading.Thread(target=play_alert, args=("Warning! Sleeping detected. Stay Alert!",)).start()
                        alert_triggered = True

            elif len(eyes) == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 0, 255)  # Red
                    if not alert_triggered:
                        threading.Thread(target=play_alert, args=("Drowsiness detected! Please be careful.",)).start()
                        alert_triggered = True

            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)  # Green
                    alert_triggered = False  # Reset alert when active

    # Display status
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Drowsiness Detection", frame)

    # Exit when ESC key pressed
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
