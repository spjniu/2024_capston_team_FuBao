import os
import pandas as pd

# 분석할 폴더 경로
folder_path = '/Users/kwonknock/Fubao/csv_379'

# 제외할 파일명
exclude_file = None

# 초기 변수 설정
left_max_values = []
right_max_values = []
left_max_file = ""
right_max_file = ""

# 폴더 내 CSV 파일 반복
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # 제외할 파일 건너뛰기
    if file_path == exclude_file:
        continue
    
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # 좌, 우 어깨 최대값 계산
    left_max = df['Left Shoulder y_diff'].max()
    right_max = df['Right Shoulder y_diff'].max()
    
    # 최대값 리스트에 추가
    left_max_values.append((left_max, file_path))
    right_max_values.append((right_max, file_path))

# 전체 최대값들의 평균, 최소값, 최대값 계산
left_max_average = sum(val[0] for val in left_max_values) / len(left_max_values)
left_max_min = min(val[0] for val in left_max_values)
left_max_max = max(left_max_values, key=lambda x: x[0])

right_max_average = sum(val[0] for val in right_max_values) / len(right_max_values)
right_max_min = min(val[0] for val in right_max_values)
right_max_max = max(right_max_values, key=lambda x: x[0])

# 결과 출력
print(f"Left Shoulder y_diff - 전체 최대값들의 평균: {left_max_average}")
print(f"Left Shoulder y_diff - 전체 최대값 중 최소값: {left_max_min}")
print(f"Left Shoulder y_diff - 전체 최대값 중 최대값: {left_max_max[0]}")
print(f"Left Shoulder y_diff - 전체 최대값이 발생한 파일: {left_max_max[1]}")

print(f"Right Shoulder y_diff - 전체 최대값들의 평균: {right_max_average}")
print(f"Right Shoulder y_diff - 전체 최대값 중 최소값: {right_max_min}")
print(f"Right Shoulder y_diff - 전체 최대값 중 최대값: {right_max_max[0]}")
print(f"Right Shoulder y_diff - 전체 최대값이 발생한 파일: {right_max_max[1]}")
