import os
import pandas as pd

# 분석할 폴더 경로
folder_path = '/Users/kwonknock/Fubao/csv_377'

# 제외할 파일명 (없을 경우 None으로 설정)
exclude_files = [
    '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z58_C.csv',
    '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z7_C.csv',
    '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z105_C.csv',
    '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z2_C.csv',

    '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z129_C.csv',
    '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z6_C.csv',

    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z19_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z41_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z24_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z114_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z84_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z3_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z17_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z39_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z8_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z112_C.csv',
    # '/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z20_C.csv',

]

# 초기 변수 설정
left_max_values = []
right_max_values = []

# 폴더 내 CSV 파일 반복
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # 제외할 파일이 설정되어 있을 때 해당 파일 건너뛰기
    if file_path in exclude_files:
        continue
    
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # S-W 좌, 우 y_diff 최대값 계산
    left_max = df['S-W Left y_diff'].max()
    right_max = df['S-W Right y_diff'].max()
    
    # 최대값 리스트에 추가
    left_max_values.append((left_max, file_path))
    right_max_values.append((right_max, file_path))

# 전체 최대값들의 평균, 최소값, 최대값 계산
left_max_average = sum(val[0] for val in left_max_values) / len(left_max_values)
left_max_min = min(left_max_values, key=lambda x: x[0])
left_max_max = max(left_max_values, key=lambda x: x[0])

right_max_average = sum(val[0] for val in right_max_values) / len(right_max_values)
right_max_min = min(right_max_values, key=lambda x: x[0])
right_max_max = max(right_max_values, key=lambda x: x[0])

# 결과 출력
print(f"S-W Left y_diff - 전체 최대값들의 평균: {left_max_average}")
print(f"S-W Left y_diff - 최소값: {left_max_min[0]}")
print(f"S-W Left y_diff - 최소값 파일: {left_max_min[1]}")
print(f"S-W Left y_diff - 최대값: {left_max_max[0]}")
print(f"S-W Left y_diff - 최대값 파일: {left_max_max[1]}")

print(f"S-W Right y_diff - 전체 최대값들의 평균: {right_max_average}")
print(f"S-W Right y_diff - 최소값: {right_max_min[0]}")
print(f"S-W Right y_diff - 최소값 파일: {right_max_min[1]}")
print(f"S-W Right y_diff - 최대값: {right_max_max[0]}")
print(f"S-W Right y_diff - 최대값 파일: {right_max_max[1]}")
