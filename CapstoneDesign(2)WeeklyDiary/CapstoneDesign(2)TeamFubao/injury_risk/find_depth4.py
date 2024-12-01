import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd  # pandas 라이브러리 추가

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# 클리핑 거리 설정
clipping_distance_in_meters = 2.3  # 2.3 미터
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)

# Haar Cascade 분류기 로드
face_cascade = cv2.CascadeClassifier("C:/haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error loading cascade classifier. Check the file path.")

# 얼굴 인식 위치를 저장하기 위한 리스트
face_locations = []
num_frames = 5  # 평균화할 프레임 수

# 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 비디오 코덱 설정
video_out = cv2.VideoWriter('output_video3.mp4', fourcc, 30.0, (640, 480))  # 비디오 파일 생성
depth_values = []  # 깊이 값을 저장할 리스트

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 백그라운드 제거
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        bg_removed = np.where((depth_image_3d > clipping_distance) 
                              | (depth_image_3d <= 0), grey_color, color_image)

        # 얼굴 인식
        gray_image = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 얼굴 위치가 있을 경우에만 추가
        if len(faces) > 0:
            face_locations.append(faces)

        # 평균화할 프레임 수에 따라 리스트의 길이 조정
        if len(face_locations) > num_frames:
            face_locations.pop(0)  # 가장 오래된 프레임 삭제

        # 최근 프레임의 얼굴 위치가 있을 경우 평균화
        if face_locations:
            avg_faces = np.zeros((len(face_locations), 4))  # 얼굴의 위치는 (x, y, w, h) 형식
            for i, face in enumerate(face_locations):
                avg_faces[i, :] = face[0]  # 첫 번째 얼굴만 사용 (여러 얼굴이 감지된 경우)

            # 평균 좌표 계산
            avg_faces = np.mean(avg_faces, axis=0)
            x, y, w, h = avg_faces.astype(int)
            cv2.rectangle(bg_removed, (x, y), (x + w, y + h), (255, 0, 0), 2)
            depth_value = depth_image[y + h // 2, x + w // 2]
            depth_values.append(depth_value)  # 깊이 값을 리스트에 추가
            cv2.putText(bg_removed, f'Depth: {depth_value}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 이미지 렌더링
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 두 이미지를 수평으로 연결
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)

        # 비디오 프레임 저장
        video_out.write(bg_removed)  # 저장할 영상에 현재 프레임 추가

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # 모든 작업 완료 후 자원 정리
    video_out.release()  # 비디오 파일 저장
    
    # 깊이 값을 CSV 파일에 저장
    depth_df = pd.DataFrame(depth_values, columns=['Depth Values'])  # 데이터프레임 생성
    depth_df.to_csv('output_video3.csv', index=False)  # CSV 파일로 저장

    pipeline.stop()
