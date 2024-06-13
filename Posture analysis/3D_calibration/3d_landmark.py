import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import mediapipe as mp

# JSON ���� �ε�
with open('calibration_data.json', 'r') as f:
    calibration_data = json.load(f)

cameraMatrix1 = np.array(calibration_data['cameraMatrix1'])
distCoeffs1 = np.array(calibration_data['distCoeffs1'])
cameraMatrix2 = np.array(calibration_data['cameraMatrix2'])
distCoeffs2 = np.array(calibration_data['distCoeffs2'])
R = np.array(calibration_data['R'])
T = np.array(calibration_data['T'])

# �̹��� ũ��
frame_size = (640, 480)

# ���׷��� ��Ƽ�����̼�
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
                                                   cameraMatrix2, distCoeffs2,
                                                   frame_size, R, T)

# ��Ƽ�����̼� �� ���
left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, frame_size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, frame_size, cv2.CV_16SC2)

# �̹��� �б�
img_left = cv2.imread('/home/Fubao/Camera0_001.jpg')
img_right = cv2.imread('/home/Fubao/Camera1_001.jpg')  # ���÷� ���� �̹��� ���, �����δ� �ٸ� �̹��� ���

# �̹��� ����
rectified_left = cv2.remap(img_left, left_map1, left_map2, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, right_map1, right_map2, cv2.INTER_LINEAR)

# �̵�������� �ʱ�ȭ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# ���� �̹������� ���� ����
results_left = pose.process(cv2.cvtColor(rectified_left, cv2.COLOR_BGR2RGB))
results_right = pose.process(cv2.cvtColor(rectified_right, cv2.COLOR_BGR2RGB))

# ���帶ũ ����
landmarks_left = results_left.pose_landmarks.landmark if results_left.pose_landmarks else []
landmarks_right = results_right.pose_landmarks.landmark if results_right.pose_landmarks else []

# ���� ������ ����Ͽ� 3D ��ǥ ���
points_3d = []
if landmarks_left and landmarks_right:
    for lm_left, lm_right in zip(landmarks_left, landmarks_right):
        # 2D ��ǥ ����
        xL, yL = int(lm_left.x * frame_size[0]), int(lm_left.y * frame_size[1])
        xR, yR = int(lm_right.x * frame_size[0]), int(lm_right.y * frame_size[1])

        # 3D ��ǥ ���
        disparity = xL - xR
        if disparity != 0:
            depth = Q[2, 3] / disparity
            X = (xL + Q[0, 3]) * depth
            Y = (yL + Q[1, 3]) * depth
            Z = Q[2, 3] * depth
            points_3d.append((X, Y, Z))

# 3D ���帶ũ �ð�ȭ
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for point in points_3d:
    ax.scatter(point[0], point[1], point[2], c='r', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
