import cv2 as cv
import numpy as np
import glob

def calibrate_camera(image_dir, camera_id):
    chessboard_size = (8, 6)
    square_size = 2.8
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = glob.glob(f'{image_dir}/*.jpg')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            img = cv.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
        else:
            print(f'Chessboard not found in image: {fname}')

    cv.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("Error: No valid images for calibration")
        return

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    np.savetxt(f'/home/Fubao/vscode/camera_parameters/c{camera_id}.dat', mtx, header='Intrinsic Matrix')
    np.savetxt(f'/home/Fubao/vscode/camera_parameters/dist_c{camera_id}.dat', dist, header='Distortion Coefficients')
    
    with open(f'/home/Fubao/vscode/camera_parameters/rot_trans_c{camera_id}.dat', 'w') as f:
        for rvec, tvec in zip(rvecs, tvecs):
            rvec_str = ' '.join(map(str, rvec.flatten()))
            tvec_str = ' '.join(map(str, tvec.flatten()))
            f.write(f'Rotation:\n{rvec_str}\nTranslation:\n{tvec_str}\n')

def read_camera_parameters(camera_id):
    cmtx = np.loadtxt(f'/home/Fubao/vscode/camera_parameters/c{camera_id}.dat')
    dist = np.loadtxt(f'/home/Fubao/vscode/camera_parameters/dist_c{camera_id}.dat')
    return cmtx, dist

def read_rotation_translation(camera_id, savefolder = '/home/Fubao/vscode/camera_parameters/'):
    inf = open(savefolder + 'rot_trans_c'+ str(camera_id) + '.dat', 'r')
    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def get_projection_matrix(camera_id):
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    # Calculate projection matrix
    P = cmtx @ np.hstack((rvec, tvec))
    return P

if __name__ == '__main__':
    calibrate_camera('/home/Fubao/vscode/left_calibration', 0)
    calibrate_camera('/home/Fubao/vscode/right_calibration', 1)
