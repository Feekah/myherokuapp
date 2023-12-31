import mediapipe as mp
import pandas as pd
import pickle
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import pandas as pd
import pickle
import numpy as np
import csv
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

#Intialize the Mediapipe libraries
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

def export_landmark_to_csv(csv_doc: str, buffered_data) -> None:
    try:
        IMPORTANT_LMS = [
        "NOSE",
        "LEFT_SHOULDER",
        "LEFT_HIP",
        "LEFT_KNEE",
        "LEFT_ANKLE",
        "LEFT_FOOT_INDEX"
            ]

    # Generate all columns of the data frame

        landmarks = ["Position", "Hip_Knee-Ankle_angle", "Shoulder_hip_knee_angle", "Knee_ankle_foot_angle"] # Label column

        for lm in IMPORTANT_LMS:
          landmarks += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
        # Write all the columns to a file
        with open(csv_doc, mode="w", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)
            for data_point in buffered_data:
                results, position, angle, angle2, angle3 = data_point
                landmarks = results.pose_landmarks.landmark
                keypoints = []

                # Extract coordinate of important landmarks
                for lm in IMPORTANT_LMS:
                    keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
                    keypoints.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])

                keypoints = list(np.array(keypoints).flatten())

                # Insert action as the label (first column)
                keypoints.insert(0, position)
                keypoints.insert(1, angle)
                keypoints.insert(2, angle2)
                keypoints.insert(3, angle3)

                # Write the row to the CSV file
                csv_writer.writerow(keypoints)

    except Exception as e:
        print(e)

def preprocess_data(csv_doc):
    # Load the data from the CSV file
    data = pd.read_csv(csv_doc)

    # Label encode the labels
    label_encoder = LabelEncoder()
    data['Position'] = label_encoder.fit_transform(data['Position'])
    print("Processing this rep")
    return data

def predict_rep(data):
    # Replace this with your actual machine learning model prediction code
    predicted_labels = loaded_model.predict(data)
    print("Predicting this rep")
    print(predicted_labels)
    return predicted_labels

def identify_most_common_label(predictions):
    # Count the occurrences of each label
    label_counts = pd.Series(predictions).value_counts()

    # Get the most common label
    most_common_label = label_counts.idxmax()

    test = {0: 'Correct Pose',
    1: 'Low Back',
    2: 'Knee too forward',
    3: 'Early Stopping'}
    print(test[most_common_label])
    return test[most_common_label]

loaded_model = joblib.load('prediction_model.pkl')
csv_doc = "your_dataset.csv"
#init_csv(csv_doc)

squat_counter = 0
squat_stage = 'start'
feed = None
buffered_data = []
movement = 0
dataHasbeenProcessedOnce = False

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Recolor image to RGB
        image.flags.writeable = False
        results = pose.process(image) # Make detection
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Recolor back to BGR
        image.flags.writeable = True
      # Extract landmarks

        try:
          lms = results.pose_landmarks.landmark
          hip = [lms[mp_pose.PoseLandmark.LEFT_HIP.value].x, lms[mp_pose.PoseLandmark.LEFT_HIP.value].y]
          knee = [lms[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lms[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
          ankle = [lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
          shoulder = [lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
          foot_index = [lms[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, lms[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

          # Calculate angle
          angle = calculate_angle(hip, knee, ankle)
          angle2 = calculate_angle(shoulder, hip, knee)
          angle3 = calculate_angle(knee, ankle, foot_index)

          # Visualize angle
          cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(hip, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

          if angle < 150:
            movement = 1
            squat_stage = 'Down'
            buffered_data.append((results, squat_stage, angle, angle2, angle3))
            dataHasbeenProcessedOnce = False
          elif angle > 150 and angle <= 172 and squat_stage != 'start':
              movement  = movement+1 if movement == 1 and squat_stage != 'Up' else movement
              squat_counter = squat_counter + 1 if movement == 2 and squat_stage != 'Up' else squat_counter
              squat_stage = 'Up'
              buffered_data.append((results, squat_stage, angle, angle2, angle3))

              # Clear the buffer to store coordinates for the new repetition
          elif angle > 172 and squat_stage == 'Up' and dataHasbeenProcessedOnce == False and movement == 2:
              export_landmark_to_csv(csv_doc, buffered_data)
              # Preprocess the CSV data and feed it into the ML model
              processed_data = preprocess_data(csv_doc)
              predictions = predict_rep(processed_data)
              feed = identify_most_common_label(predictions)
              print("Results for this repetition:", feed)
              buffered_data.clear()
              #os.remove(csv_doc)

              dataHasbeenProcessedOnce = True

        except:
          pass

      # Render curl counter
              # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data for count
        cv2.putText(image, 'COUNT', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(squat_counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

        # Rep data for Form
        cv2.putText(image, 'FORM', (115, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(feed),
                    (115, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

        # Rep data for Stage
        cv2.putText(image, 'STAGE', (300, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(squat_stage),
                    (200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Shoulder Press Checker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    