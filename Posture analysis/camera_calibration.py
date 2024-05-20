import cv2
import numpy as np
import glob
import os

capture_dir = 'calibration_images'  # 캡처된 이미지가 저장된 디렉토리

CHECKERBOARD = (8, 6)  # 체커보드 패턴의 크기 (가로, 세로)

# 체커보드의 3D 좌표 준비
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)

objpoints = []  # 3D 공간에서의 점들
imgpoints = []  # 이미지 평면에서의 점들

# 캡처 디렉토리에서 모든 jpg 이미지 파일 로드
images = glob.glob(os.path.join(capture_dir, '*.jpg'))

if not images:
    print(f"No images found in {capture_dir}.")
    exit()  # 이미지가 없으면 종료

for fname in images:
    print(f"Processing {fname}...")
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load {fname}.")
        continue  # 이미지를 로드할 수 없으면 다음 파일로 넘어감

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH 
                                             + cv2.CALIB_CB_FAST_CHECK 
                                             + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        objpoints.append(objp)  # 3D 점 추가
        # 더 정확한 코너 위치 찾기
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)  # 2D 점 추가

        # 이미지에 체커보드 코너 그리기
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)  # 이미지를 창에 표시
        cv2.waitKey(500)  # 0.5초 대기
    else:
        print(f"Checkerboard corners not found in {fname}.")

cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

if not objpoints or not imgpoints:
    print("Not enough points for calibration.")
    exit()  # 충분한 점이 없으면 종료

# 카메라 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과 출력
print("Camera matrix: \n", mtx)
print("Distortion coefficients: \n", dist)

# 캘리브레이션 데이터를 파일에 저장
# np.savez('camera0_calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
np.savez('camera1_calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
