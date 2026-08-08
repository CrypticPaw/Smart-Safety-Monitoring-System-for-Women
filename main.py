import cv2
import numpy as np
from datetime import datetime
import mediapipe as mp 


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)  

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

def is_night_time(frame):
    current_hour = datetime.now().hour
    is_night_by_time = current_hour >= 19 or current_hour <= 6 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    is_night_by_brightness = brightness < 60  
    
    print(f"Current Hour: {current_hour}, Brightness: {brightness}")
    
    return is_night_by_time or is_night_by_brightness

def detect_safety_scenarios(gender_counts, frame, lone_woman_detected, woman_surrounded_detected):
    total_women = gender_counts.get('Female', 0)
    total_men = gender_counts.get('Male', 0)

    print(f"Total Women: {total_women}, Total Men: {total_men}")

    if total_women == 1 and total_men == 0 and lone_woman_detected:
        cv2.putText(frame, "ALERT: Lone Woman Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print("Lone woman alert triggered")

    if total_women > 0 and total_men > total_women and woman_surrounded_detected:
        cv2.putText(frame, "Alert: Woman Surrounded by Men", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print("Woman surrounded alert triggered")

def is_call_me_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y
    pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y

    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y

    is_thumb_extended = thumb_tip < thumb_base
    is_pinky_extended = pinky_tip < pinky_base
    is_index_folded = index_tip > index_base
    is_middle_folded = middle_tip > middle_base
    is_ring_folded = ring_tip > ring_base

    return (is_thumb_extended and is_pinky_extended and 
            is_index_folded and is_middle_folded and is_ring_folded)

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)
padding = 20

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame, bboxs = faceBox(faceNet, frame)
    
    gender_counts = {'Male': 0, 'Female': 0}
    detected_women_bboxs = []

    for bbox in bboxs:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1), max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        gender_counts[gender] += 1

        if gender == 'Female':
            detected_women_bboxs.append(bbox)
            label = "{}".format(gender)
            cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    night_time = is_night_time(frame)
    detect_safety_scenarios(gender_counts, frame, lone_woman_detected=(len(detected_women_bboxs) == 1), woman_surrounded_detected=True)

    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if is_call_me_gesture(hand_landmarks):
                if len(detected_women_bboxs) > 0:
                    cv2.putText(frame, 'SOS', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  

    cv2.imshow("Smart Safety Monitoring", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()  