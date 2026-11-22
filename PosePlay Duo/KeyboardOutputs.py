import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui
import time

# Load trained model and labels
model = tf.keras.models.load_model("action_model.h5")
label_classes = ['block', 'crouch', 'idle', 'jump', 'jump_left', 'jump_right', 'kick', 'left', 'punch', 'right']

held_keys = set()
key_map = {
    "left": "a",
    "right": "d",
    "jump": "w",
    "block": "g"
}

# Setup MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.5)

# Rolling buffer for 5 frames
buffer = []

# Frame drawing toggle
draw_interval = 5
frame_count = 0

# FPS cap
fps_cap_enabled = True
fps_limit = 15
prev_frame_time = 0

# OpenCV webcam (index 1, set resolution)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

while cap.isOpened():
    if fps_cap_enabled:
        now = time.time()
        if now - prev_frame_time < 1.0 / fps_limit:
            continue
        prev_frame_time = now

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        hip_x = (landmarks[23].x + landmarks[24].x) / 2
        hip_y = (landmarks[23].y + landmarks[24].y) / 2

        keypoints = []
        for lm in landmarks:
            rel_x = lm.x - hip_x
            rel_y = lm.y - hip_y
            keypoints.extend([rel_x, rel_y])
        keypoints = np.array(keypoints)

        # Update rolling buffer
        if len(buffer) < 5:
            buffer.append(keypoints)
        else:
            buffer.pop(0)
            buffer.append(keypoints)

        # Predict when buffer full
        if len(buffer) == 5:
            input_data = np.array(buffer).reshape(1, 5, 66)
            preds = model.predict(input_data, verbose=0)[0]
            top_2_indices = np.argsort(preds)[-2:]
            top_1_idx = top_2_indices[1]
            top_1_name = label_classes[top_1_idx]

            # Keys that should be held for this frame
            current_frame_keys = set()

            if top_1_name == "left":
                current_frame_keys.add(key_map["left"])
            elif top_1_name == "right":
                current_frame_keys.add(key_map["right"])
            elif top_1_name == "jump":
                current_frame_keys.add(key_map["jump"])
            elif top_1_name == "jump_left":
                current_frame_keys.update([key_map["jump"], key_map["left"]])
            elif top_1_name == "jump_right":
                current_frame_keys.update([key_map["jump"], key_map["right"]])
            elif top_1_name == "block":
                current_frame_keys.add(key_map["block"])

            # Press new keys
            for key in current_frame_keys - held_keys:
                pyautogui.keyDown(key)

            # Release old keys
            for key in held_keys - current_frame_keys:
                pyautogui.keyUp(key)

            # Update the held_keys set
            held_keys = current_frame_keys

            # Show prediction label
            cv2.putText(frame, f"Top-1: {top_1_name}", (10, 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)

    else:
        buffer = []

    frame_count += 1
    cv2.imshow("Live Action Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
