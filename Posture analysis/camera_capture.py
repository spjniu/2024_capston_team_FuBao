import cv2
import numpy as np
import subprocess
import os

# 캡처한 이미지를 저장할 디렉토리
capture_dir = 'calibration_images'
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

def capture_images_from_camera(camera_index, window_name):
    # libcamera-vid 명령어 설정
    command = [
        'libcamera-vid',
        '--camera', str(camera_index),  # 카메라 인덱스 설정
        '--codec', 'mjpeg',             # 코덱 설정
        '--width', '640',               # 프레임 너비 설정
        '--height', '480',              # 프레임 높이 설정
        '-t', '0',                      # 무제한 시간 설정
        '-o', '-',                      # 표준 출력으로 스트림 설정
        '--exposure', 'normal',         # 노출 설정
        '--awb', 'auto'                 # 자동 화이트 밸런스 설정
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

    bytes_buffer = bytes()
    image_counter = 0
    try:
        while True:
            bytes_buffer += process.stdout.read(1024)  # 스트림에서 데이터를 읽어 버퍼에 추가
            a = bytes_buffer.find(b'\xff\xd8')  # JPEG 시작 지점 찾기
            b = bytes_buffer.find(b'\xff\xd9')  # JPEG 종료 지점 찾기
            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]  # JPEG 이미지 추출
                bytes_buffer = bytes_buffer[b+2:]  # 버퍼 업데이트
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)  # 이미지를 디코드하여 프레임으로 변환

                if frame is not None:
                    cv2.imshow(window_name, frame)  # 프레임을 창에 표시

                    if cv2.waitKey(1) & 0xFF == ord('c'):  # 'c' 키가 눌리면
                        image_counter += 1
                        image_path = os.path.join(capture_dir, f"calibration_image_{image_counter}.jpg")
                        cv2.imwrite(image_path, frame)  # 프레임을 이미지로 저장
                        print(f"Captured {image_path}")

                    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키가 눌리면 루프 종료
                        break
    finally:
        process.stdout.close()  # 프로세스의 표준 출력을 닫음
        process.terminate()     # 프로세스 종료
        process.wait()          # 프로세스 종료 대기
        cv2.destroyWindow(window_name)  # 창 닫기

# 0번 카메라에서 이미지 캡처
capture_images_from_camera(0, 'Camera 0')
# 1번 카메라에서 이미지 캡처 (주석 처리됨)
# capture_images_from_camera(1, 'Camera 1')
