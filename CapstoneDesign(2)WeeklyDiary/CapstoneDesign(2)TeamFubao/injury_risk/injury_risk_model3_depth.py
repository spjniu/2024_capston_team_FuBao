import pandas as pd
import glob
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# CSV 파일 경로
normal_data_path = '/sample.csv'
injured_data_path = '/sample.csv'

# 데이터 로드 및 통합 함수
def load_and_combine_csv_files(data_path, label):
    all_files = glob.glob(data_path)
    combined_data = []
    
    for file in all_files:
        df = pd.read_csv(file)
        df['Label'] = label  # 정상 자세와 부상 위험이 있는 자세에 레이블 추가
        df['Depth'] = 4000
        combined_data.append(df)
    
    return pd.concat(combined_data, ignore_index=True)

# 데이터 로드
normal_df = load_and_combine_csv_files(normal_data_path, label=0)  # 정상 자세 레이블
injured_df = load_and_combine_csv_files(injured_data_path, label=1)  # 부상 위험 자세 레이블

# 데이터 통합
df = pd.concat([normal_df, injured_df], ignore_index=True)

# 부상 위험 단계 함수 정의 (depth scaling을 고려한 계산)
def calculate_wrist_shoulder_risk(sw_left_diff, sw_right_diff, depth_value):
    # depth_value를 고려한 스케일링 (2D 좌표에 영향을 미침)
    scaled_diff_left = sw_left_diff * depth_value
    scaled_diff_right = sw_right_diff * depth_value
    max_diff = max(scaled_diff_left, scaled_diff_right)
    
    # 위험도 평가
    if max_diff < -1.5210330486297607:
        return 0.0  # 안전
    elif max_diff > 2.621201597727262:
        return 3.0  # 심각한 위험
    else:
        risk_score = (max_diff + 1.5210330486297607) / (2.621201597727262 + 1.5210330486297607) * 3
        return round(risk_score, 3)  # 소수 셋째 자리까지 반올림하여 반환

def calculate_shoulder_elevation_risk(left_shoulder_diff, right_shoulder_diff, depth_value):
    # depth_value를 고려한 어깨 높이 차이 스케일링
    scaled_left_diff = left_shoulder_diff
    scaled_right_diff = right_shoulder_diff
    max_diff = max(scaled_left_diff, scaled_right_diff)
    
    # 위험도 평가
    if max_diff < 0.022588066922293724:
        return 0.0  # 안전
    elif 0.022588066922293724 <= max_diff < 0.0281768143177032:
        # 0~1 사이로 정규화
        risk_score = (max_diff - 0.022588066922293724) / (0.0281768143177032 - 0.022588066922293724)
        return round(risk_score, 3)
    elif 0.0281768143177032 <= max_diff < 0.0420467257499694:
        # 1~3 사이로 정규화
        risk_score = 1 + (max_diff - 0.0281768143177032) / (0.0420467257499694 - 0.0281768143177032) * 2
        return round(risk_score, 3)
    else:
        return 3.0  # 심각한 위험

# 위험 계산 적용
df['Wrist Injury Risk Level'] = df.apply(lambda row: calculate_wrist_shoulder_risk(row['S-W Left y_diff'], row['S-W Right y_diff'], 4000), axis=1)
df['Shoulder Elevation Risk Level'] = df.apply(lambda row: calculate_shoulder_elevation_risk(row['Left Shoulder y_diff'], row['Right Shoulder y_diff'], 4000), axis=1)

# 특성과 타겟 변수 선택
X = df[['S-W Left y_diff', 'S-W Right y_diff', 'Left Shoulder y_diff', 'Right Shoulder y_diff', 'Depth']]
y_wrist = df['Wrist Injury Risk Level']  # 손목 위험 점수
y_shoulder = df['Shoulder Elevation Risk Level']  # 어깨 높이 위험 점수

# 데이터셋 분할
X_train_wrist, X_test_wrist, y_train_wrist, y_test_wrist = train_test_split(X, y_wrist, test_size=0.2, random_state=42)
X_train_shoulder, X_test_shoulder, y_train_shoulder, y_test_shoulder = train_test_split(X, y_shoulder, test_size=0.2, random_state=42)

# 랜덤 포레스트 회귀기 학습 (손목 모델)
model_wrist = RandomForestRegressor(n_estimators=100, random_state=42)
model_wrist.fit(X_train_wrist, y_train_wrist)

# 랜덤 포레스트 회귀기 학습 (어깨 모델)
model_shoulder = RandomForestRegressor(n_estimators=100, random_state=42)
model_shoulder.fit(X_train_shoulder, y_train_shoulder)

# 예측 (손목)
y_pred_wrist = model_wrist.predict(X_test_wrist)

# 예측 (어깨)
y_pred_shoulder = model_shoulder.predict(X_test_shoulder)

# 모델 평가
mse_wrist = mean_squared_error(y_test_wrist, y_pred_wrist)
r2_wrist = r2_score(y_test_wrist, y_pred_wrist)
print(f'Wrist Model - Mean Squared Error: {mse_wrist:.4f}, R² Score: {r2_wrist:.4f}')

mse_shoulder = mean_squared_error(y_test_shoulder, y_pred_shoulder)
r2_shoulder = r2_score(y_test_shoulder, y_pred_shoulder)
print(f'Shoulder Model - Mean Squared Error: {mse_shoulder:.4f}, R² Score: {r2_shoulder:.4f}')

# 모델 저장
joblib.dump(model_wrist, 'injury_risk_model_wrist_depth.pkl')  # 손목 모델 저장
joblib.dump(model_shoulder, 'injury_risk_model_shoulder_depth.pkl')  # 어깨 모델 저장
