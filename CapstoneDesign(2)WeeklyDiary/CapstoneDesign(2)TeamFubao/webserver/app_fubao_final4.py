from flask import Flask, request, render_template, redirect, url_for
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import cv2
import mediapipe as mp
import matplotlib
import shutil

matplotlib.use('Agg')  # GUI 백엔드 대신 Agg로 설정

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 모델 로드
loaded_model_wrist = joblib.load('injury_risk_model_wrist_depth.pkl')
loaded_model_shoulder = joblib.load('injury_risk_model_shoulder_depth.pkl')

# Mediapipe 포즈 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET'])
def upload():
    return render_template('upload.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return redirect(request.url)

    file = request.files['video_file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # CSV 경로 설정
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(file.filename)[0]}.csv")

        # find_injury_risk 함수 호출, video_path를 넘겨주도록 수정
        find_injury_risk(video_path=file_path, csv_path=csv_path)

        # 분석이 완료되면 결과 페이지로 리디렉션
        return redirect(url_for('injury_risk'))
    
    
@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    if 'folder' not in request.files:
        return 'No file part'
    
    folder = request.files.getlist('folder')

    # 업로드된 파일을 각 사용자별 폴더에 저장
    user_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_data')
    if not os.path.exists(user_data_path):
        os.makedirs(user_data_path)

    # 기존 폴더가 있으면 삭제하고 새로 만듬
    if os.path.exists(user_data_path):
        shutil.rmtree(user_data_path)
    os.makedirs(user_data_path)

    # 업로드한 파일 저장
    for file in folder:
        if file.filename != '':
            file_path = os.path.join(user_data_path, file.filename)
            file.save(file_path)

    # 자세 분석 수행
    analyze_pose(user_data_path)

    return redirect(url_for('accuracy'))

def analyze_pose(user_data_path):
    user1_keypoints_dict = {"LEFT_ELBOW": [], "RIGHT_ELBOW": [], "LEFT_WRIST": [], "RIGHT_WRIST": []}

    # 이미지 분석
    for img_file in sorted(os.listdir(user_data_path)):
        img_path = os.path.join(user_data_path, img_file)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # 팔꿈치, 손목 좌표 추출 및 저장
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            user1_keypoints_dict["LEFT_ELBOW"].append(left_elbow.y)
            user1_keypoints_dict["RIGHT_ELBOW"].append(right_elbow.y)
            user1_keypoints_dict["LEFT_WRIST"].append(left_wrist.y)
            user1_keypoints_dict["RIGHT_WRIST"].append(right_wrist.y)

    # 이동 평균 CSV 파일 로드
    data = pd.read_csv("average_keypoints_V3.csv")

    # 이동 패턴 그래프 그리기 (user1 데이터와 이동 평균 데이터 비교)
    plt.figure(figsize=(18, 8))  # 그래프 크기를 설정

    num_keypoints = len(user1_keypoints_dict.keys())
    rows = (num_keypoints + 1) // 2  # 2개씩 출력하므로 행 수 계산

    correlations = []  # 상관 계수를 저장할 리스트
    feedbacks = []  # 피드백 메시지를 저장할 리스트

    # 그래프 그리기 전에 피드백을 준비
    for i, keypoint in enumerate(user1_keypoints_dict.keys()):
        min_length = min(len(user1_keypoints_dict[keypoint]), len(data[keypoint].dropna()))
        user1_values = user1_keypoints_dict[keypoint][:min_length]
        moving_average_values = data[keypoint].dropna().values[:min_length]
        
        # 상관 계수 계산
        correlation = np.corrcoef(user1_values, moving_average_values)[0, 1]*100
        correlations.append((keypoint, correlation))

        # 피드백 메시지 생성 (상관 계수 기반)
        if correlation > 80:
            feedback = "The posture is close to perfect!"
        elif correlation > 70:
            feedback = "Average performance, but there is room for improvement."
        else:
            feedback = "Try to maintain a more consistent and higher posture."
        
        feedbacks.append(f'{keypoint}: {feedback}')

    # 피드백을 그래프 위에 먼저 표시하기
    plt.text(0.5, 1.05, '\n'.join(feedbacks), ha='center', va='bottom', fontsize=12, wrap=True)

    # 그래프 그리기
    for i, keypoint in enumerate(user1_keypoints_dict.keys()):
        plt.subplot(rows, 2, i + 1)  # 2개의 열과 계산된 행 수로 subplot 설정
        min_length = min(len(user1_keypoints_dict[keypoint]), len(data[keypoint].dropna()))
        user1_values = user1_keypoints_dict[keypoint][:min_length]
        moving_average_values = data[keypoint].dropna().values[:min_length]
        
        # user1 데이터 그래프
        plt.plot(range(min_length), user1_values, color='red', linestyle='-', label=f'{keypoint} User Y Position')
        # 이동 평균 데이터 그래프
        plt.plot(range(min_length), moving_average_values, color='blue', linestyle='-', label=f'{keypoint} General Y Position')

        # 상관 계수 계산 및 저장
        correlation = np.corrcoef(user1_values, moving_average_values)[0, 1]*100
        correlations.append((keypoint, correlation))

        plt.xlabel('Frame Index')
        plt.ylabel('Y Coordinate')
        plt.title(f'{keypoint} (Grade: {correlation:.0f})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pose_analysis_graph.png')
    plt.savefig(graph_path, bbox_inches='tight', dpi=90, format='png')  # 그래프를 파일로 저장
    plt.close()


@app.route('/accuracy')
def accuracy():
    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pose_analysis_graph.png')
    return render_template('accuracy.html', graph_path=graph_path)
    

@app.route('/injury_risk')
def injury_risk():
    # 생성된 그래프를 injury_risk.html에서 보여주기 위해 파일 경로 전달
    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'injury_risk_graph.png')
    return render_template('injury_risk.html', graph_path=graph_path)

def find_injury_risk(video_path, csv_path):
    # Mediapipe 포즈 솔루션 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # 위험 점수 기록을 위한 리스트
    wrist_risk_scores = []
    shoulder_risk_scores = []

    # 동영상 읽기
    cap = cv2.VideoCapture(video_path)

    previous_depth_value = None  # 이전 깊이 값을 저장할 변수

    # depth 값 로드 (CSV 파일에서)
    depth_df = pd.read_csv(csv_path)
    depth_values = depth_df['Depth Values'].values

    baseline_left_shoulder_y = None
    baseline_right_shoulder_y = None
    frame_index = 0

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

    # 최대 위험 점수 계산 (소수점 첫째자리까지)
    max_wrist_risk = round(max(wrist_risk_scores, default=0), 1)
    max_shoulder_risk = round(max(shoulder_risk_scores, default=0), 1)

    # 피드백 생성
    arm_elevation_feedback = f"Maximum Arm Elevation Risk is {max_wrist_risk}."
    shoulder_elevation_feedback = f"Maximum Shoulder Elevation Risk is {max_shoulder_risk}."

    # 시각화
    plt.figure(figsize=(10, 4))

    # 손목-어깨 위험 점수 그래프
    plt.plot(range(len(wrist_risk_scores)), wrist_risk_scores, linestyle='-', color='b', linewidth=2, label='Arm Elevation Risk')

    # 어깨 높이 위험 점수 그래프
    plt.plot(range(len(shoulder_risk_scores)), shoulder_risk_scores, linestyle='-', color='r', linewidth=2, label='Shoulder Elevation Risk')

    # 제목에 피드백 추가
    plt.title(f"{arm_elevation_feedback}\n{shoulder_elevation_feedback}", fontsize=10, fontweight='light', loc='left')
    plt.xlabel('Frame Index')
    plt.ylabel('Injury Risk Score')

    # 세로축 최대값 설정
    plt.ylim(0, 3)

    plt.legend(loc='upper left')
    plt.grid(True)

    # 그래프 저장 (이미지 크기 줄이기)
    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'injury_risk_graph.png')
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
