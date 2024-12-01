import cv2
import mediapipe as mp
import os
import csv

# Mediapipe 포즈 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 좌표 차 계산 함수
def calculate_axis(left_side, right_side):
    return (left_side - right_side) * 100

baseline_left_shoulder_y = None
baseline_right_shoulder_y = None

# 자동으로 사용할 경로들 설정
image_folders = [
"/sample_folder",
]

output_dir = '/sample_folder'

# 디렉터리가 존재하지 않으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for image_folder_path in image_folders:
    if not os.path.exists(image_folder_path):
        print(f"오류: 폴더 경로 '{image_folder_path}'가 존재하지 않습니다.")
        continue

    # 이미지 파일 리스트
    image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith('.jpg')])

    if not image_files:
        print(f"폴더에 이미지가 없습니다: {image_folder_path}")
        continue

    # output file name
    folder_name = os.path.basename(os.path.normpath(image_folder_path))
    output_file_path = f'{output_dir}/pose_diff_{folder_name}.csv'

    # CSV 파일 쓰기 준비
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # CSV 헤더 작성
        writer.writerow(['Image', 'S-W Left y_diff', 'S-W Right y_diff', 'Left Shoulder y_diff', 'Right Shoulder y_diff', 
                         'Left Shoulder (x, y)', 'Right Shoulder (x, y)', 'Left Elbow (x, y)', 'Right Elbow (x, y)',
                         'Left Wrist (x, y)', 'Right Wrist (x, y)'])

        for index, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder_path, image_file)

            # 이미지 읽기
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Could not load image {image_file}.")
                continue

            # 이미지를 RGB로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 이미지 처리를 통해 포즈를 인식
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # 좌표 추출
                landmarks = results.pose_landmarks.landmark

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Baseline 설정 (첫 번째 이미지일 때)
                if index == 0:
                    baseline_left_shoulder_y = left_shoulder[1]
                    baseline_right_shoulder_y = right_shoulder[1]
                    
                # 차이 계산
                shoulder_wrist_diff_left = calculate_axis(left_shoulder[1], left_wrist[1])
                shoulder_wrist_diff_right = calculate_axis(right_shoulder[1], right_wrist[1])
                left_shoulder_diff = baseline_left_shoulder_y - left_shoulder[1]
                right_shoulder_diff = baseline_right_shoulder_y - right_shoulder[1]
                wrist_y_diff = left_wrist[1] - right_wrist[1]

                # CSV 파일에 데이터 추가
                writer.writerow([image_file, shoulder_wrist_diff_left, shoulder_wrist_diff_right, left_shoulder_diff, 
                                 right_shoulder_diff, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist])

            else:
                print(f"Warning: No pose landmarks found in {image_file}.")

    # 모든 이미지 처리가 끝난 후
    print(f"Results saved to {output_file_path}.")

cv2.destroyAllWindows()
