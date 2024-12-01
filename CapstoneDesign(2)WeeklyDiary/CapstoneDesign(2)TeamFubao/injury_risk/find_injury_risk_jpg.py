import cv2
import mediapipe as mp
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Mediapipe 포즈 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 모델 로드
loaded_model_wrist = joblib.load('injury_risk_model_wrist.pkl')  # 손목-어깨 위험 모델
loaded_model_shoulder = joblib.load('injury_risk_model_shoulder.pkl')  # 어깨 높이 위험 모델

# 입력 이미지 폴더 지정
image_folder_path = '/Users/kwonknock/Downloads/SRR_image/babel_01/Day04_200924_F/5/C/377-2-1-18-Z4_C'

# 좌표 차 계산 함수
def calculate_axis(left_side, right_side):
    return (left_side - right_side) * 100

baseline_left_shoulder_y = None
baseline_right_shoulder_y = None

# 이미지 파일 리스트
image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith('.jpg')])

# 데이터 저장 리스트
data = []

for index, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder_path, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_file}.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        if index == 0:
            baseline_left_shoulder_y = left_shoulder[1]
            baseline_right_shoulder_y = right_shoulder[1]

        shoulder_wrist_diff_left = calculate_axis(left_shoulder[1], left_wrist[1])
        shoulder_wrist_diff_right = calculate_axis(right_shoulder[1], right_wrist[1])
        left_shoulder_diff = baseline_left_shoulder_y - left_shoulder[1]
        right_shoulder_diff = baseline_right_shoulder_y - right_shoulder[1]

        data.append([image_file, shoulder_wrist_diff_left, shoulder_wrist_diff_right, left_shoulder_diff, right_shoulder_diff])

# Pandas DataFrame으로 변환
test_data = pd.DataFrame(data, columns=['Image', 'S-W Left y_diff', 'S-W Right y_diff', 'Left Shoulder y_diff', 'Right Shoulder y_diff'])

# 예측 수행
X_test = test_data[['S-W Left y_diff', 'S-W Right y_diff', 'Left Shoulder y_diff', 'Right Shoulder y_diff']]
predictions_wrist = loaded_model_wrist.predict(X_test)
predictions_shoulder = loaded_model_shoulder.predict(X_test)

# 결과 저장
test_data['Predicted Wrist Injury Risk Score'] = predictions_wrist
test_data['Predicted Shoulder Elevation Risk Score'] = predictions_shoulder

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['Predicted Wrist Injury Risk Score'], marker='o', linestyle='-', color='b', linewidth=2, label='Arm')
plt.plot(test_data.index, test_data['Predicted Shoulder Elevation Risk Score'], marker='o', linestyle='-', color='r', linewidth=2, label='Shoulder')

# 제목 설정
plt.title('Injury Risk Data')
plt.xlabel('Exercise Process')
plt.ylabel('Injury Risk Score')

for i, score in enumerate(predictions_wrist):
    if score != 0:
        plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')

for i, score in enumerate(predictions_shoulder):
    if score != 0:
        plt.text(i, score, f'{score:.2f}', ha='center', va='top')

plt.xticks(ticks=test_data.index, labels=['' for _ in range(len(test_data))])
plt.xticks(ticks=[0, len(test_data) - 1], labels=['start', 'end'])

plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.ylim(0, 3)
plt.grid(False)
plt.legend()
plt.show()

# 예측 결과 출력
print(test_data[['Image', 'Predicted Wrist Injury Risk Score', 'Predicted Shoulder Elevation Risk Score']])
