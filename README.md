# Driver Drowsiness and Yawning Detection

This project is a real-time driver drowsiness detection system built using Python and a webcam.
It detects eye closure and yawning using facial landmark analysis and alerts the user when drowsiness is detected.

---

## What this project does
- Detects eye closure using Eye Aspect Ratio (EAR)
- Detects yawning using Mouth Aspect Ratio (MAR)
- Shows live EAR and MAR values on screen
- Triggers an alarm when drowsiness or yawning is detected
- Works in real time using a laptop webcam

---

## Technologies Used
- Python 3.11
- OpenCV
- MediaPipe Face Mesh
- NumPy
- playsound3

---

## Project Structure
drowsy_project/
├── main.py
├── alarm.mp3
├── README.md
└── drowsy-env/

---

## How to Run the Project

### 1. Create virtual environment (Python 3.11)
py -3.11 -m venv drowsy-env

### 2. Activate the environment
drowsy-env\Scripts\activate

### 3. Install required libraries
pip install numpy opencv-python mediapipe playsound3

### 4. Run the program
python main.py

---

## Controls
- Press `q` to quit the application

---

## Output
- Live webcam feed with facial landmark tracking
- EAR and MAR values displayed in real time
- On-screen alert when drowsiness or yawning is detected
- Alarm sound when thresholds are exceeded

---

## Known Limitations
- Accuracy depends on lighting conditions
- Thresholds may vary between users
- Webcam quality affects detection

---

## To Do
- Improve yawning threshold calibration
- Add head pose detection
- Log drowsiness events to a file
- Improve performance on low-end systems

---

## Project Link
https://github.com/marionsebastian2023-dot/drowsiness-detection-system/tree/main

