import ffmpeg
import os

# 입력 파일 및 출력 파일 경로
input_file_path = "output.mp4"  # 파일명 맞게 지정
output_file_path = "output_after.mp4"

# 절대 경로 설정
input_file_path = os.path.abspath(input_file_path)
output_file_path = os.path.abspath(output_file_path)

# 입력 파일 스트림 설정
input_stream = ffmpeg.input(input_file_path)

# 출력 파일 설정 (비디오 속도 2배로 설정하여 길이를 반으로 줄임)
output_stream = ffmpeg.output(
    input_stream,
    output_file_path,
    format='mp4',                # 출력 파일 포맷
    vf="setpts=2*PTS",           # 현재 비디오 속도 0.5배로 설정 (길이를 반으로 줄임)
    af="atempo=2.0",             # 현재 오디오 속도 0.5배로 설정 (길이를 반으로 줄임)
    vcodec='libx264',            # 비디오 코덱 설정
    acodec='aac',                # 오디오 코덱 설정
    r=15                         # 비디오 프레임 레이트 설정
)

# 출력 파일 덮어쓰기 설정
output_stream = ffmpeg.overwrite_output(output_stream)

# FFmpeg 실행
ffmpeg.run(output_stream)

print(f"Processed video saved to {output_file_path}")
