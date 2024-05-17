import cv2
import numpy as np
import subprocess
import threading

def camera_stream(camera_index, window_name):
    command = [
    'libcamera-vid',
    '--camera', str(camera_index),
    '--codec', 'mjpeg',
    '--width', '640',
    '--height', '480',
    '-t', '0',
    '-o', '-',
    '--exposure', 'normal',  
    '--awb', 'auto'  
]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

    bytes_buffer = bytes()
    try:
        while True:
            bytes_buffer += process.stdout.read(1024)
            a = bytes_buffer.find(b'\xff\xd8')  
            b = bytes_buffer.find(b'\xff\xd9')  
            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    cv2.imshow(window_name, frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()
        cv2.destroyWindow(window_name)

threads = []
for index in range(2):  
    window_name = f"Camera {index+1}"
    thread = threading.Thread(target=camera_stream, args=(index, window_name))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
