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

            # Detect eyes within face RO
