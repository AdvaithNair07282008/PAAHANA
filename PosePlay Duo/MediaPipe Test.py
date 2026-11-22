import cv2
import mediapipe as mp
import csv

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Output CSV file for all actions
OUTPUT_FILE = "presentation_pictures.csv"

# Define initial label and sequence tracking
current_label = "kick"  # You can change this live with number keys
sequence_id = 0
recording = False

# Setup webcam
cap = cv2.VideoCapture(0)

# Create the CSV file with headers
with open(OUTPUT_FILE, mode='w', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(['label', 'sequence_id'] + [f'{coord}{i}' for i in range(33) for coord in ['x', 'y']])

  print("Instructions:")
  print("  - Press [r] to toggle recording")
  print("  - Press [n] to start a new sequence")
  print("  - Press [1-8] to change the label (e.g., 1 = jump, 2 = punch, etc.)")
  print("  - Press [q] to quit")

  # Map number keys to gesture labels
  label_map = {
      '1': "kick",
      '2': "jump_left",
  }

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break

      # Flip for mirror view and convert to RGB
      frame = cv2.flip(frame, 1)
      image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      result = pose.process(image_rgb)

      # Draw pose landmarks and record if active
      if result.pose_landmarks:
           mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
           if recording:
               keypoints = result.pose_landmarks.landmark
               row = [current_label, sequence_id]


       # Compute waist center (hip midpoint)
               hip_x = (keypoints[23].x + keypoints[24].x) / 2
               hip_y = (keypoints[23].y + keypoints[24].y) / 2


       # Convert keypoints to waist-relative
               for lm in keypoints:
                   rel_x = lm.x - hip_x
                   rel_y = lm.y - hip_y
                   row.extend([rel_x, rel_y])


               writer.writerow(row)

      # Display current status on screen
      cv2.putText(frame, f'Label: {current_label}', (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      cv2.putText(frame, f'Sequence ID: {sequence_id}', (10, 70),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
      cv2.putText(frame, f'Recording: {"Yes" if recording else "No"}', (10, 110),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255 if not recording else 0), 2)

      # Show webcam feed
      cv2.imshow("Pose Action Recorder", frame)

      # Handle key presses
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
          break
      elif key == ord('r'):
          recording = not recording
          print(f"Recording {'started' if recording else 'paused'}")
      elif key == ord('n'):
          sequence_id += 1
          print(f"New sequence: {sequence_id}")
      elif chr(key) in label_map:
          current_label = label_map[chr(key)]
          print(f"Changed label to '{current_label}'")

# Clean up
cap.release()
cv2.destroyAllWindows()
