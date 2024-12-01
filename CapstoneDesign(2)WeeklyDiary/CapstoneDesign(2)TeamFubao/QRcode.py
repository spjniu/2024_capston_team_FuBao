import qrcode

# QR 코드에 담을 영상 정보 (URL이나 파일 경로)
video_info = "https://sample"  # 여기에 영상의 URL 또는 경로를 입력하세요

# QR 코드 생성
qr = qrcode.QRCode(
    version=1,  # QR 코드의 크기 (1은 가장 작음)
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,  # 하나의 박스 크기
    border=4,  # 여백 (기본값은 4)
)

qr.add_data(video_info)
qr.make(fit=True)

# QR 코드 이미지를 저장
img = qr.make_image(fill_color="black", back_color="white")
img.save("video_qr_code.png")

print("QR 코드가 video_qr_code.png 파일로 저장되었습니다.")
