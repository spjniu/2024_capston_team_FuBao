import cv2
import mediapipe as mp

# Mediapipe 포즈 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 좌표 차 계산 함수
def calculate_axis(left_side, right_side):
    return (left_side - right_side) * 100

def findError(diff_axis):
    if diff_axis > 0:
        return "Warning"
    else:
        return "Good"

# 웹캠 캡처 객체 생성
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

photo_taken = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 프레임을 보여주기 (실시간 스트리밍)
    cv2.imshow('Webcam - Press Space to Capture', frame)

    # 'space' 키를 눌러 사진 촬영
    if cv2.waitKey(1) & 0xFF == ord(' '):
        photo = frame.copy()  # 현재 프레임을 복사해서 저장
        photo_taken = True
        print("Photo captured!")
        break

# 웹캠 릴리즈 (스트리밍 종료)
cap.release()
cv2.destroyAllWindows()

# 촬영된 사진을 처리
if photo_taken:
    # 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    
    # 이미지 처리를 통해 포즈를 인식
    results = pose.process(image_rgb)
    
    # 결과를 BGR로 변환 (cv2.imshow는 BGR 이미지를 기대함)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        #  사람 인식하는 실루엣을 표시
        mp_drawing.draw_landmarks(
            image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 좌표 추출
        landmarks = results.pose_landmarks.landmark
        
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # 화면 좌표로 변환
        height, width, _ = image_bgr.shape

        shoulder_elbow_diff_left = calculate_axis(left_shoulder[1], left_elbow[1])
        shoulder_elbow_diff_right = calculate_axis(right_shoulder[1], right_elbow[1])
        elbow_wrist_diff_left = calculate_axis(left_elbow[1], left_wrist[1])
        elbow_wrist_diff_right = calculate_axis(right_elbow[1], right_wrist[1])

        SE_condition_left = findError(shoulder_elbow_diff_left)
        SE_condition_right = findError(shoulder_elbow_diff_right)
        EW_condition_left = findError(elbow_wrist_diff_left)
        EW_condition_right = findError(elbow_wrist_diff_right)
        
        # 배경이 되는 사각형
        cv2.rectangle(image_bgr, (0, height - 250), (330, height - 10), (0, 0, 0), -1)

        # 좌우 y좌표의 차이를 텍스트로 화면에 출력
        cv2.putText(image_bgr, f'S-E Left diff: {shoulder_elbow_diff_left:.2f}', (5, image_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, f'S-E Right diff: {shoulder_elbow_diff_right:.2f}', (5, image_bgr.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, f'E-W Left diff: {elbow_wrist_diff_left:.2f}', (5, image_bgr.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, f'E-W Right diff: {elbow_wrist_diff_right:.2f}', (5, image_bgr.shape[0] - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image_bgr, f'S-E Left condition: {SE_condition_left}', (5, image_bgr.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, f'S-E Right condition: {SE_condition_right}', (5, image_bgr.shape[0] - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, f'E-W Left condition: {EW_condition_left}', (5, image_bgr.shape[0] - 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, f'E-W Right condition: {EW_condition_right}', (5, image_bgr.shape[0] - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # 결과를 화면에 표시
    cv2.imshow('Mediapipe Pose Estimation', image_bgr)
    cv2.waitKey(0)  # 키보드 입력이 있을 때까지 대기
    cv2.destroyAllWindows()
