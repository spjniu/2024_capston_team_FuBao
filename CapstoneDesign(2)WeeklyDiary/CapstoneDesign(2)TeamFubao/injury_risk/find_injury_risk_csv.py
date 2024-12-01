import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 저장된 모델 불러오기
loaded_model_wrist = joblib.load('injury_risk_model_wrist.pkl')  # 손목-어깨 위험 회귀 모델
loaded_model_shoulder = joblib.load('injury_risk_model_shoulder.pkl')  # 어깨 높이 위험 회귀 모델

# 테스트 데이터 로드 (예: CSV 파일)
test_data = pd.read_csv('/Users/kwonknock/Fubao/csv_377/pose_diff_377-2-1-18-Z4_C.csv')

# 특성 선택
X_test = test_data[['S-W Left y_diff', 'S-W Right y_diff', 'Left Shoulder y_diff', 'Right Shoulder y_diff']]

# 손목-어깨 위험 예측 수행
predictions_wrist = loaded_model_wrist.predict(X_test)

# 어깨 높이 위험 예측 수행
predictions_shoulder = loaded_model_shoulder.predict(X_test)

# 예측 결과 추가
test_data['Predicted Wrist Injury Risk Score'] = predictions_wrist  # 손목-어깨 위험 점수
test_data['Predicted Shoulder Elevation Risk Score'] = predictions_shoulder  # 어깨 높이 위험 점수

# 시각화
plt.figure(figsize=(12, 6))

# 손목-어깨 위험 점수 그래프
plt.plot(test_data.index, test_data['Predicted Wrist Injury Risk Score'], marker='o', linestyle='-', color='b', linewidth=2, label='Arm')

# 어깨 높이 위험 점수 그래프
plt.plot(test_data.index, test_data['Predicted Shoulder Elevation Risk Score'], marker='o', linestyle='-', color='r', linewidth=2, label='Shoulder')

plt.title('Injury Risk Data')
plt.xlabel('Exercise Process')
plt.ylabel('Injury Risk Score')

# 수치 표시
for i, score in enumerate(predictions_wrist):
    if score != 0:  # 0인 경우에는 표시하지 않음
        plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')  # 소수점 둘째 자리까지 표시

for i, score in enumerate(predictions_shoulder):
    if score != 0:  # 0인 경우에는 표시하지 않음
        plt.text(i, score, f'{score:.2f}', ha='center', va='top')  # 소수점 둘째 자리까지 표시

# 가로축 아래의 이름 숨기기
plt.xticks(ticks=test_data.index, labels=['' for _ in range(len(test_data))])  # 모든 이름 숨김
plt.xticks(ticks=[0, len(test_data) - 1], labels=['start', 'end'])  # 첫 번째와 마지막 이름 설정

plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')

# 세로축 최대값 설정
plt.ylim(0, 3)

# 격자표 숨기기
plt.grid(False)

# 범례 추가
plt.legend()

plt.show()

# 예측 결과 출력
print(test_data[['Image', 'Predicted Wrist Injury Risk Score', 'Predicted Shoulder Elevation Risk Score']])  # 이미지 이름과 예측 점수만 출력
