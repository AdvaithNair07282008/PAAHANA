import cv2

import numpy as np

import tensorflow as tf

import time

import mediapipe as mp

import pyautogui

from mediapipe.tasks import python

from mediapipe.tasks.python import vision



# Load model and label classes

model = tf.keras.models.load_model("action_model.h5")

label_classes = ["block", "crouch", "idle", "jump", "jump_left", "jump_right", "kick", "left", "punch", "right"]



# PoseLandmarker setup
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
   base_options=base_options,
   output_segmentation_masks=False,
   num_poses=2  # Detect up to 2 poses
)
detector = vision.PoseLandmarker.create_from_options(options)



cap = cv2.VideoCapture(0)



# One buffer for each person (based on index in pose_landmarks)

buffers = [[], []]



# Initialize held_keys tracking outside the loop, only once

if 'held_keys' not in globals():

   held_keys = [set(), set()]



# Define key maps per player

player_keymaps = [

   {"left": "a", "right": "d", "jump": "w"},  # Player 1

   {"left": "left", "right": "right", "jump": "up"}  # Player 2

]



while cap.isOpened():

   success, frame = cap.read()

   if not success:

       break



   frame = cv2.flip(frame, 1)

   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

   detection_result = detector.detect(mp_image)



   if not detection_result.pose_landmarks:

       cv2.imshow("Multi-Person Action Recognition", frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):

           break

       continue



   for i, pose_landmarks in enumerate(detection_result.pose_landmarks):

       if len(pose_landmarks) != 33:

           continue



       hip_x = (pose_landmarks[23].x + pose_landmarks[24].x) / 2

       hip_y = (pose_landmarks[23].y + pose_landmarks[24].y) / 2



       keypoints = []

       for lm in pose_landmarks:

           rel_x = lm.x - hip_x

           rel_y = lm.y - hip_y

           keypoints.extend([rel_x, rel_y])

       keypoints = np.array(keypoints)



       if len(buffers[i]) < 2:

           buffers[i].append(keypoints)

       else:

           buffers[i].pop(0)

           buffers[i].append(keypoints)



       if len(buffers[i]) == 2:

           input_data = np.array(buffers[i]).reshape(1, 2, 66)

           input_data = (input_data - np.mean(input_data)) / (np.std(input_data) + 1e-8)

           preds = model.predict(input_data, verbose=0)



           max_confidence = np.max(preds[0])

           if max_confidence >= 0.6:

               top1_id = np.argmax(preds[0])

               top1_name = label_classes[top1_id] if top1_id < len(label_classes) else "Unknown"



               # Keyboard handling starts here

               current_frame_keys = set()

               keymap = player_keymaps[i]  # i = 0 for Player 1, 1 for Player 2



               if top1_name == "left":

                   current_frame_keys.add(keymap["left"])

               elif top1_name == "right":

                   current_frame_keys.add(keymap["right"])

               elif top1_name == "jump":

                   current_frame_keys.add(keymap["jump"])

               elif top1_name == "jump_left":

                   current_frame_keys.update([keymap["left"], keymap["jump"], keymap["left"]])

               elif top1_name == "jump_right":

                   current_frame_keys.update([keymap["right"], keymap["jump"]])



               for key in current_frame_keys - held_keys[i]:

                   pyautogui.keyDown(key)

               for key in held_keys[i] - current_frame_keys:

                   pyautogui.keyUp(key)

               held_keys[i] = current_frame_keys

               # Keyboard handling ends here



               # Draw label near the nose

               landmark = pose_landmarks[0]  # Nose

               h, w, _ = frame.shape

               cx, cy = int(landmark.x * w), int(landmark.y * h)

               cv2.putText(frame, f"Person {i+1}: {top1_name}", (cx - 60, cy - 10),

                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)



   cv2.imshow("Multi-Person Action Recognition", frame)

   if cv2.waitKey(1) & 0xFF == ord('q'):

       break



cap.release()

cv2.destroyAllWindows()

