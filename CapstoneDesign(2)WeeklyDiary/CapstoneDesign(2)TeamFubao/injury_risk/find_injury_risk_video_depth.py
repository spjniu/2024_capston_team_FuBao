import cv2
import mediapipe as mp
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Mediapipe 포즈 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 손목-어깨 및 어깨 높이 위험 회귀 모델 로드
loaded_model_wrist = joblib.load('injury_risk_model_wrist_depth.pkl')  # 손목-어깨 위험 회귀 모델
loaded_model_shoulder = joblib.load('injury_risk_model_shoulder_depth.pkl')  # 어깨 높이 위험 회귀 모델

# 테스트할 동영상 경로 설정
video_path = '/Users/kwonknock/Fubao/depth_data/output_video5.mp4'  # 동영상 파일 경로
csv_path = '/Users/kwonknock/Fubao/depth_data/output_video5.csv'  # depth 값을 포함하는 CSV 경로

# 동영상 읽기
cap = cv2.VideoCapture(video_path)

previous_depth_value = None  # 이전 깊이 값을 저장할 변수

# depth 값 로드 (CSV 파일에서)
depth_df = pd.read_csv(csv_path)
depth_values = depth_df['Depth Values'].values

baseline_left_shoulder_y = None
baseline_right_shoulder_y = None
frame_index = 0

# 위험 점수 기록을 위한 리스트
wrist_risk_scores = []
shoulder_risk_scores = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_index >= 500:  # 500번째 프레임에서 중단
        break

    # 현재 frame_index가 depth_values의 길이를 초과하지 않도록 확인
    if frame_index >= len(depth_values):
        break  # 인덱스가 배열 크기를 초과하면 종료

    # 이미지 처리
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 현재 깊이 값 추출
    current_depth_value = depth_values[frame_index]

    # 운동 종료 조건 확인
    if previous_depth_value is not None:
        depth_difference = abs(current_depth_value - previous_depth_value)

        if depth_difference > 500:
            break  # 운동 종료 시 루프를 중단

    # 이전 깊이 값을 현재 값으로 업데이트
    previous_depth_value = current_depth_value

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Baseline 설정 (두 번째 프레임일 때)
        if frame_index == 1:
            baseline_left_shoulder_y = left_shoulder[1]
            baseline_right_shoulder_y = right_shoulder[1]

        # 손목-어깨 차이 계산 (현재값 - 기준값)
        left_shoulder_y_diff = (baseline_left_shoulder_y - left_shoulder[1]) if baseline_left_shoulder_y else 0
        right_shoulder_y_diff = (baseline_right_shoulder_y - right_shoulder[1]) if baseline_right_shoulder_y else 0
        wrist_shoulder_left_diff = (left_shoulder[1] - left_wrist[1])  
        wrist_shoulder_right_diff = (right_shoulder[1] - right_wrist[1])  

        # 위험 수준 예측
        wrist_risk = loaded_model_wrist.predict([[wrist_shoulder_left_diff, wrist_shoulder_right_diff, left_shoulder_y_diff, right_shoulder_y_diff, depth_values[frame_index]]])
        shoulder_risk = loaded_model_shoulder.predict([[wrist_shoulder_left_diff, wrist_shoulder_right_diff, left_shoulder_y_diff, right_shoulder_y_diff, depth_values[frame_index]]])

        # 위험 점수 리스트에 추가
        wrist_risk_scores.append(wrist_risk[0])
        shoulder_risk_scores.append(shoulder_risk[0])

    frame_index += 1

# 동영상 종료 후
cap.release()
cv2.destroyAllWindows()

# 시각화
plt.figure(figsize=(12, 6))

# 손목-어깨 위험 점수 그래프
plt.plot(range(len(wrist_risk_scores)), wrist_risk_scores, linestyle='-', color='b', linewidth=2, label='Arm Elevation Risk')

# 어깨 높이 위험 점수 그래프
plt.plot(range(len(shoulder_risk_scores)), shoulder_risk_scores, linestyle='-', color='r', linewidth=2, label='Shoulder Elevation Risk')

plt.title('Injury Risk Data')
plt.xlabel('Frame Index')
plt.ylabel('Injury Risk Score')

# 세로축 최대값 설정
plt.ylim(0, 3)

plt.legend(loc='upper left')
plt.grid(True)
plt.show()
