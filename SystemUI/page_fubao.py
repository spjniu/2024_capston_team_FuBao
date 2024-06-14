from flask import Flask, Response, render_template, request, redirect, url_for, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import serial
import time
import subprocess
import os
import cv2
import mediapipe as mp
import math

# 시리얼 통신 버전
ser = serial.Serial('/dev/ttyACM0', 19200) 

app = Flask(__name__)
CORS(app)

UPLOAD_VIDEO_FOLDER = 'static/uploaded_videos'
app.config['UPLOAD_VIDEO_FOLDER'] = UPLOAD_VIDEO_FOLDER

# 전역 변수
signal = None
camera_process = None
left_front = 0
left_back = 0
right_front = 0
right_back = 0
both_left = 0
both_right = 0
signal = 0
cnt = 0
cnt_quit = 0
# 0608 sungbin add
start_flag=None
weight_sum=0
result_condition=""

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

# 시리얼 통신 함수 모음
def send_weight_to_arduino(weight):
    ser.write(bytes(str(weight), 'utf-8'))  
    print("Weight sent to Arduino:", weight)

def send_start_command():
    ser.write(b'start\n')  
    print("Sent 'start' command to Arduino")

def send_end_command():
    ser.write(b'end\n')  
    print("Sent 'end' command to Arduino")

def send_continue_command():
    ser.write(b'continue\n')  
    print("Sent 'continue' command to Arduino")

def send_quit_command():
    ser.write(b'quit\n')  
    print("Sent 'quit' command to Arduino")
    
def send_back_command(): ## 0608 sungbin add
    ser.write(b'back\n')  
    print("Sent 'back' command to Arduino")


def read_con_data():
    global left_back, left_front, right_back, right_front, both_left, both_right,result_condition,weight_sum

    # 변수 초기화
    left_front = None
    left_back = None
    right_front = None
    right_back = None
    both_left = None
    both_right = None
    result_condition=" "

    while True:
        if ser.in_waiting > 0:
            try:

                line = ser.readline().decode('utf-8').strip()
                print("Received data from con_Arduino:", line)
                if line.startswith("Weight_sum:"):
                    weight_sum = int(line.split(": ")[1])                
                elif line.startswith("rate_left_front:"):
                    left_front = int(line.split(": ")[1])
                elif line.startswith("rate_left_back:"):
                    left_back = int(line.split(": ")[1])
                elif line.startswith("rate_right_front:"):
                    right_front = int(line.split(": ")[1])
                elif line.startswith("rate_right_back:"):
                    right_back = int(line.split(": ")[1])
                elif line.startswith("rate_both_left:"):
                    both_left = int(line.split(": ")[1])
                elif line.startswith("rate_both_right:"):
                    both_right = int(line.split(": ")[1])
                elif line in ("Error: There is no enough weight"): # 0608 sungbin add
                    result_condition = line      

                # 모든 값이 설정되었는지 체크
                if all([v is not None for v in [left_front, left_back, right_front, right_back, both_left, both_right]]):
                    break

            except ValueError:
                print("Error in data format.")
            except Exception as e:
                print("Unexpected error:", e)

def read_quit_data():
    global left_back, left_front, right_back, right_front, both_left, both_right,result_condition,weight_sum

    # 변수 초기화
    left_front = None
    left_back = None
    right_front = None
    right_back = None
    both_left = None
    both_right = None
    result_condition=" "

    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                print("Received data from quit_Arduino:", line)

                if line.startswith("Weight_sum:"):
                    weight_sum = int(line.split(": ")[1])   
                elif line.startswith("rate_left_front:"):
                    left_front = int(line.split(": ")[1])
                elif line.startswith("rate_left_back:"):
                    left_back = int(line.split(": ")[1])
                elif line.startswith("rate_right_front:"):
                    right_front = int(line.split(": ")[1])
                elif line.startswith("rate_right_back:"):
                    right_back = int(line.split(": ")[1])
                elif line.startswith("rate_both_left:"):
                    both_left = int(line.split(": ")[1])
                elif line.startswith("rate_both_right:"):
                    both_right = int(line.split(": ")[1])
                elif line in ("Try again", "Good","Error: There is no enough weight"): # 0608 sungbin add
                    result_condition = line 
                # 모든 값이 설정되었는지 체크
                if all([v is not None for v in [left_front, left_back, right_front, right_back, both_left, both_right]]):
                    break

            except ValueError:
                print("Error in data format.")
            except Exception as e:
                print("Unexpected error:", e)
def read_start_data():
    global left_back, left_front, right_back, right_front, both_left, both_right,cnt,result_condition,weight_sum

    # 변수 초기화
    left_front = None
    left_back = None
    right_front = None
    right_back = None
    both_left = None
    both_right = None
    result_condition=" "

    while (cnt!=1):
        if ser.in_waiting > 0:
            try:

                line = ser.readline().decode('utf-8').strip()
                print("Received data from start_Arduino:", line)

                if line.startswith("Weight_sum:"):
                    weight_sum = int(line.split(": ")[1])   
                elif line.startswith("rate_left_front:"):
                    left_front = int(line.split(": ")[1])
                elif line.startswith("rate_left_back:"):
                    left_back = int(line.split(": ")[1])
                elif line.startswith("rate_right_front:"):
                    right_front = int(line.split(": ")[1])
                elif line.startswith("rate_right_back:"):
                    right_back = int(line.split(": ")[1])
                elif line.startswith("rate_both_left:"):
                    both_left = int(line.split(": ")[1])
                elif line.startswith("rate_both_right:"):
                    both_right = int(line.split(": ")[1])
                elif line in ("Try again", "Good","Error: There is no enough weight"): # 0608 sungbin add
                    result_condition = line    
                    print(result_condition)   
                elif line.startswith("back"):
                    print("Break is done")
                    break      
                # 모든 값이 설정되었는지 체크
                if all([v is not None for v in [left_front, left_back, right_front, right_back, both_left, both_right]]):
                    cnt=1
                    break

            except ValueError:
                print("Error in data format.")
            except Exception as e:
                print("Unexpected error:", e)


def read_end_data():
    global left_back, left_front, right_back, right_front, both_left, both_right,cnt,weight_sum

    # 변수 초기화
    left_front = None
    left_back = None
    right_front = None
    right_back = None
    both_left = None
    both_right = None
    result_condition=" "

    while (cnt!=1):
        if ser.in_waiting > 0:
            try:

                line = ser.readline().decode('utf-8').strip()
                print("Received data from end_Arduino:", line)

                if line.startswith("Weight_sum:"):
                    weight_sum = int(line.split(": ")[1])   
                elif line.startswith("rate_left_front:"):
                    left_front = int(line.split(": ")[1])
                elif line.startswith("rate_left_back:"):
                    left_back = int(line.split(": ")[1])
                elif line.startswith("rate_right_front:"):
                    right_front = int(line.split(": ")[1])
                elif line.startswith("rate_right_back:"):
                    right_back = int(line.split(": ")[1])
                elif line.startswith("rate_both_left:"):
                    both_left = int(line.split(": ")[1])
                elif line.startswith("rate_both_right:"):
                    both_right = int(line.split(": ")[1])
                
                # 모든 값이 설정되었는지 체크
                if all([v is not None for v in [left_front, left_back, right_front, right_back, both_left, both_right]]):
                    cnt=1
                    break

            except ValueError:
                print("Error in data format.")
            except Exception as e:
                print("Unexpected error:", e)


@app.route('/')
def start():
    return render_template('start.html')

@app.route('/inputWeight')
def inputWeight():
    return render_template('inputWeight.html')

@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

# 0608 sungbin add
@app.route('/result')
def result():
    global result_condition
    time.sleep(5)
    return render_template('result.html',result_condition=result_condition)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')
# 0608 sungbin add
@app.route('/select')
def select():
    return render_template('select.html')

    
@app.route('/start_capture')
def start_capture():
    try:
        subprocess.Popen(['python3', 'camera_capture.py'])
        return jsonify({"message": "Capture started successfully"})
    except Exception as e:
        return jsonify({"message": str(e)})
    
@app.route('/start_capture_video')
def start_capture_video():
    try:
        subprocess.Popen(['python3', 'video_capture.py'])
        return jsonify({"message": "Capture started successfully"})
    except Exception as e:
        return jsonify({"message": str(e)})

@app.route('/save_weight', methods=['POST'])
def save_weight():
    global stored_weight
    data = request.get_json()
    weight = data.get('weight')
    print(f"Received weight: {weight}")
    stored_weight = float(weight)  # 저장된 weight 값을 전역 변수에 저장
    send_weight_to_arduino(weight)  # 아두이노로 몸무게 보내는 함수 실행 
    time.sleep(1.0) 
    return jsonify({"status": "success", "weight": weight})

@app.route('/send_signal', methods=['POST'])
def send_signal():
    global cnt
    global signal
    data = request.get_json()
    signal = data.get('signal')
    print(f"Received signal: {signal}")
    time.sleep(1)
    if(signal=='start'): # 0608 sungbin add
        send_start_command()
        start_flag=1
        time.sleep(1)
        read_start_data()
    elif(signal=='continue'):
        ser.flushInput()
        send_continue_command()
        read_con_data()
    elif(signal=='end'):
        send_end_command()
        time.sleep(5) # 
        ser.flushInput()
    elif(signal=='quit'):
        ser.flushInput()
        send_quit_command()
        read_quit_data()
        # 0608 sungbin add
    elif(signal=='back'): ## 0607 added code
        send_back_command()
        start_flag=0
        
        
    elif(signal=='squat'):
        ser.write(b'squat\n')  
        print("Sent 'squat' command to Arduino")
    elif(signal=='sidelateralraise'):
        ser.write(b'sidelateralraise\n')  
        print("Sent 'sidelateralraise' command to Arduino")
    elif(signal=='armcurl'):
        ser.write(b'armcurl\n')  
        print("Sent 'armcurl' command to Arduino")
    elif(signal=='deadlift'):
        ser.write(b'deadlift\n')  
        print("Sent 'deadlift' command to Arduino")
    elif(signal=='babelrow'):
        ser.write(b'babelrow\n')  
        print("Sent 'babelrow' command to Arduino")
    elif(signal=='shoulderpress'):
        ser.write(b'shoulderpress\n')  
        print("Sent 'shoulderpress' command to Arduino")

    # 시그널 신호 반환
    return jsonify({"status": "success", "signal": signal})

    
@app.route('/get_values', methods=['GET'])
def get_values():
    values = {
        "value1": left_front,
        "value2": left_back,
        "value3": right_front,
        "value4": right_back,
        "value5": both_left,
        "value6": both_right
    }
    return jsonify(values)

@app.route('/update_values', methods=['POST'])
def update_values():
    global left_front, left_back, right_front, right_back, both_left, both_right
    data = request.get_json()
    left_front = data.get('left_front')
    left_back = data.get('left_back')
    right_front = data.get('right_front')
    right_back = data.get('right_back')
    both_left = data.get('both_left')
    both_right = data.get('both_right')
    return jsonify({"status": "success"})

@app.route('/upload', methods=['POST'])
def upload_file():
    global file

    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    file.save(os.path.join(app.config['UPLOAD_VIDEO_FOLDER'], file.filename))
    return 'File uploaded successfully'

@app.route('/process_image', methods=['POST'])
def process_image():
    global uploaded_video_filename
    
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
   
    uploaded_video_filename = file.filename
    print(f"{uploaded_video_filename}")

    image_path = os.path.join(app.config['UPLOAD_VIDEO_FOLDER'], uploaded_video_filename)

    output_file = f"static/captured_videos/output_{uploaded_video_filename}"

    # 이미지 파일 읽기
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        exit()

    # 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지 처리를 통해 포즈 인식
    results = pose.process(image_rgb)


    if results.pose_landmarks:

        results = pose.process(image_rgb)

        # 좌표 추출
        landmarks = results.pose_landmarks.landmark

        # 이미지의 너비와 높이
        height, width, _ = image.shape

        # 랜드마크를 이미지에 그리기
        for landmark in landmarks:
            # 랜드마크 좌표를 이미지의 실제 좌표로 변환
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            # 초록색 원으로 랜드마크 그리기
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        #  nose의 x좌표를 적용
        center_hip = [nose[0], (left_hip[1] + right_hip[1]) * 0.5]
        center_shoulder = [nose[0] , (left_shoulder[1] + right_shoulder[1]) * 0.5]

        # 어깨 좌표
        left_shoulder_x = int(left_shoulder[0] * width)
        left_shoulder_y = int(left_shoulder[1] * height)
        right_shoulder_x = int(right_shoulder[0] * width)
        right_shoulder_y = int(right_shoulder[1] * height)

        # 코 좌표
        nose_x = int(nose[0] * width)
        nose_y = int(nose[1] * height)

        # 중심점 좌표
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
        left_shoulder_angle = 360 - calculate_angle(left_shoulder, center_shoulder, nose)
        right_shoulder_angle = calculate_angle(right_shoulder, center_shoulder, nose)

        if left_shoulder_angle > 180:
            left_shoulder_angle = calculate_angle(left_shoulder, center_shoulder, nose)
        if right_shoulder_angle > 180:
            right_shoulder_angle = 360 - calculate_angle(right_shoulder, center_shoulder, nose)


        # 기준선 그리기
            # 세로선
        cv2.line(image, (nose_x, 0), (nose_x, height), (0, 0, 255), 1)  # 중앙선
        cv2.line(image, (left_shoulder_x, 0), (left_shoulder_x, height), (0, 255, 0), 1)
        cv2.line(image, (right_shoulder_x, 0), (right_shoulder_x, height), (0, 255, 0), 1)
            #가로선
        cv2.line(image, (0, center_shoulder_y), (width, center_shoulder_y), (0, 0, 255), 1)
        cv2.line(image, (0, center_hip_y), (width, center_hip_y), (0, 0, 255), 1)

        # 좌우 y좌표의 차이를 텍스트로 화면에 출력
        cv2.putText(image, f'Shoulder diff: {shoulder_diff_y:.2f}', (5, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Elbow diff: {elbow_diff_y:.2f}', (5, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Wrist diff: {wrist_diff_y:.2f}', (5, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'hip diff: {hip_diff_y:.2f}', (5, image.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 화면에 각도 출력
        cv2.putText(image, f'Left hip Angle: {left_hip_angle:.2f}', (5, image.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Right hip Angle: {right_hip_angle:.2f}', (5, image.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(image, f'Left Shoulder Angle: {left_shoulder_angle:.2f}', (5, image.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Right Shoulder Angle: {right_shoulder_angle:.2f}', (5, image.shape[0] - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # output_file로 이미지를 저장
        cv2.imwrite(output_file, image)

        # 이미지 창 닫기
        cv2.destroyAllWindows()

        return render_template('image.html', uploaded_video_filename=uploaded_video_filename)
    
@app.route('/process_video', methods=['POST'])
def process_video():    
    global uploaded_video_filename

    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    uploaded_video_filename = file.filename
    print(f"{uploaded_video_filename}")

    # 입력 파일 경로 설정
    video_path = os.path.join(app.config['UPLOAD_VIDEO_FOLDER'], uploaded_video_filename)
    output_file = f"static/captured_videos/output_{uploaded_video_filename}"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Error: Could not open video file."

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 비디오 코덱 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    previous_landmarks = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 이미지 전처리 및 포즈 추출
            # result값을 추출하기 위해 RGB로 변환
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = pose.process(image)
            # OpenCV 사용을 위해 BGR로 변환
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            current_landmarks = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]

            # 이전 프레임과 현재 프레임의 랜드마크를 사용한 중간 지점 계산
            if previous_landmarks:
                interpolated_landmarks = []
                for prev, curr in zip(previous_landmarks, current_landmarks):
                    interpolated_landmarks.append(
                        [(prev[0] + curr[0]) / 2, (prev[1] + curr[1]) / 2]
                    )

                # 랜드마크를 이미지에 그리기
                for i, landmark in enumerate(interpolated_landmarks):
                    x, y = int(landmark[0] * width), int(landmark[1] * height)
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

                if previous_landmarks:
                    nose = [interpolated_landmarks[mp_pose.PoseLandmark.NOSE.value][0], 
                    interpolated_landmarks[mp_pose.PoseLandmark.NOSE.value][1]]

                    left_shoulder = [interpolated_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0],
                            interpolated_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]]
                    left_elbow = [interpolated_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0], 
                            interpolated_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]]
                    left_wrist = [interpolated_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0], 
                            interpolated_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]]
                    left_hip = [interpolated_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0],
                            interpolated_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]]

                    right_shoulder = [interpolated_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0],
                            interpolated_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]]
                    right_elbow = [interpolated_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][0],
                            interpolated_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][1]]
                    right_wrist = [interpolated_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0],
                            interpolated_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]]
                    right_hip = [interpolated_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0],
                            interpolated_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]]
                    
                    center_hip = [nose[0], (left_hip[1] + right_hip[1]) * 0.5]
                    center_shoulder = [nose[0] , (left_shoulder[1] + right_shoulder[1]) * 0.5]

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

                    # 좌우 좌표 계산
                    shoulder_diff_y = calculate_axis(left_shoulder[1], right_shoulder[1])
                    elbow_diff_y = calculate_axis(left_elbow[1], right_elbow[1])
                    wrist_diff_y = calculate_axis(left_wrist[1], right_wrist[1])
                    hip_diff_y = calculate_axis(left_hip[1], right_hip[1])

                    # 중심점을 기준으로 좌우 각도 계산
                    left_hip_angle = 360 - calculate_angle(left_hip, center_hip, nose) 
                    right_hip_angle = calculate_angle(right_hip, center_hip, nose)
                    left_shoulder_angle = 360 - calculate_angle(left_shoulder, center_shoulder, nose)
                    right_shoulder_angle = calculate_angle(right_shoulder, center_shoulder, nose)

                    if left_shoulder_angle > 180:
                        left_shoulder_angle = calculate_angle(left_shoulder, center_shoulder, nose)
                    if right_shoulder_angle > 180:
                        right_shoulder_angle = 360 - calculate_angle(right_shoulder, center_shoulder, nose)

                    # 중간값 지시선 그리기
                    thick = 2   # 선의 굵기

                        # 세로선
                    cv2.line(image, (nose_x, 0), (nose_x, height), (0, 0, 255), 1)  # 중앙선
                    cv2.line(image, (left_shoulder_x, 0), (left_shoulder_x, height), (0, 255, 0), thick)
                    cv2.line(image, (right_shoulder_x, 0), (right_shoulder_x, height), (0, 255, 0), thick)
                        # 가로선
                    cv2.line(image, (0, center_shoulder_y), (width, center_shoulder_y), (0, 0, 255), thick)
                    cv2.line(image, (0, center_hip_y), (width, center_hip_y), (0, 0, 255), thick)

                    # 좌우 y좌표의 차이를 텍스트로 화면에 출력
                    cv2.putText(image, f'Shoulder diff: {shoulder_diff_y:.2f}', (5, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, f'Elbow diff: {elbow_diff_y:.2f}', (5, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, f'Wrist diff: {wrist_diff_y:.2f}', (5, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, f'Hip diff: {hip_diff_y:.2f}', (5, image.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    # 화면에 각도 출력
                    cv2.putText(image, f'Left Hip Angle: {left_hip_angle:.2f}', (5, image.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, f'Right Hip Angle: {right_hip_angle:.2f}', (5, image.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    cv2.putText(image, f'Left Shoulder Angle: {left_shoulder_angle:.2f}', (5, image.shape[0] - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, f'Right Shoulder Angle: {right_shoulder_angle:.2f}', (5, image.shape[0] - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            previous_landmarks = current_landmarks

        out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return render_template('video.html', uploaded_video_filename=uploaded_video_filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
