import cv2
import numpy as np
import subprocess
import os
from datetime import datetime

video_dir = 'static/uploaded_videos'
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

def record_video_from_camera(camera_index, window_name, output_file):
    # libcamera-vid 명령어 설정 (카메라로 직접 녹화)
    command = [
        'libcamera-vid',
        '--camera', str(camera_index),  # 카메라 인덱스 설정
        '--codec', 'h264',              # 코덱 설정
        '--width', '640',               # 프레임 너비 설정
        '--height', '480',              # 프레임 높이 설정
        '--framerate', '20',
        '-t', '0',                      # 무제한 시간 설정
        '-o', output_file,              # 출력 파일 설정
        '--exposure', 'normal',         # 노출 설정
        '--awb', 'auto'                 # 자동 화이트 밸런스 설정
    ]
    
    subprocess.run(command)
    
    # OpenCV로 화면에 비디오 출력 및 처리
    cap = cv2.VideoCapture(output_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) != -1:  # 아무 키나 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

# 0번 카메라로 녹화 시작
current_time = datetime.now().strftime("%m%d_%H%M")
output_file = os.path.join(video_dir, f'video_{current_time}.mp4')

record_video_from_camera(0, 'Camera 0', output_file)
