import cv2
import mediapipe as mp
import math
import os

# Mediapipe 포즈 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 각도 계산 함수
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    if angle < 0:
        angle += 360
    return angle

# 왼쪽 및 오른쪽 좌표 차 계산 함수
def calculate_axis(left_side, right_side):
    return abs(left_side - right_side) * 100

# 외부 동영상 파일 경로 지정
video_path = 'output.mp4'  # 파일명 맞게 설정
output_dir = 'static/captured_videos'
output_file = os.path.join(output_dir, 'output_video.mp4')

# 출력 디렉토리가 존재하는지 확인하고, 없다면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# 비디오 속성 읽기
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter 객체 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Reached the end of the video or failed to read frame.")
        break

    # 성능 향상을 위해 이미지를 RGB로 변환
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # 이미지 처리를 통해 포즈를 인식
    results = pose.process(image)

    # 결과를 BGR로 변환 (cv2.imshow는 BGR 이미지를 기대함)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # 좌표 추출
        landmarks = results.pose_landmarks.landmark
        
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        center_hip = [(left_hip[0] + right_hip[0]) * 0.5, (left_hip[1] + right_hip[1]) * 0.5]
        center_shoulder = [(left_shoulder[0] + right_shoulder[0]) * 0.5 , (left_shoulder[1] + right_shoulder[1]) * 0.5]

        # 화면 좌표로 변환
        height, width, _ = image.shape

        left_shoulder_x = int(left_shoulder[0] * width)
        left_shoulder_y = int(left_shoulder[1] * height)
        right_shoulder_x = int(right_shoulder[0] * width)
        right_shoulder_y = int(right_shoulder[1] * height)
        nose_x = int(nose[0] * width)
        nose_y = int(nose[1] * height)

        center_shoulder_x = int(center_shoulder[0] * width)
        center_shoulder_y = int(center_shoulder[1] * height)
        center_hip_x = int(center_hip[0] * width)
        center_hip_y = int(center_hip[1] * height)

        shoulder_diff_y = calculate_axis(left_shoulder[1], right_shoulder[1])
        elbow_diff_y = calculate_axis(left_elbow[1], right_elbow[1])
        wrist_diff_y = calculate_axis(left_wrist[1], right_wrist[1])
        hip_diff_y = calculate_axis(left_hip[1], right_hip[1])

        # 중심점을 기준으로 좌우 각도 계산
        left_hip_angle = 360 - calculate_angle(left_hip, center_hip, nose) 
        right_hip_angle = calculate_angle(right_hip, center_hip, nose)
        left_shouder_angle = 360 - calculate_angle(left_shoulder, center_shoulder, nose)
        right_shouder_angle = calculate_angle(right_shoulder, center_shoulder, nose)
    
        # 기준선 그리기
        # 세로선
        cv2.line(image, (nose_x, 0), (nose_x, height), (0, 0, 255), 1)  # 중앙선
        cv2.line(image, (left_shoulder_x, 0), (left_shoulder_x, height), (0, 255, 0), 1)
        cv2.line(image, (right_shoulder_x, 0), (right_shoulder_x, height), (0, 255, 0), 1)
        # 가로선
        cv2.line(image, (0, center_shoulder_y), (width, center_shoulder_y), (0, 0, 255), 1)
        cv2.line(image, (0, center_hip_y), (width, center_hip_y), (0, 0, 255), 1)

        # 좌우 y좌표의 차이를 텍스트로 화면에 출력
        cv2.putText(image, f'Shoulder diff: {shoulder_diff_y:.2f}', (5, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Elbow diff: {elbow_diff_y:.2f}', (5, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Wrist diff: {wrist_diff_y:.2f}', (5, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Waist diff: {hip_diff_y:.2f}', (5, image.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 화면에 각도 출력
        cv2.putText(image, f'Left Waist Angle: {left_hip_angle:.2f}', (5, image.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Right Waist Angle: {right_hip_angle:.2f}', (5, image.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(image, f'Left Shoulder Angle: {left_shouder_angle:.2f}', (5, image.shape[0] - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Right Shoulder Angle: {right_shouder_angle:.2f}', (5, image.shape[0] - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 결과 프레임을 파일로 저장
    out.write(image)
    
    # 결과를 화면에 표시 (test)
    cv2.imshow('Mediapipe Exercise Calculator', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()
