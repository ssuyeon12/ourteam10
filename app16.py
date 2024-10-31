import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import time
import os

# 페이지 레이아웃 설정
st.set_page_config(layout="wide")

# 제목
st.title("비디오 사물 검출 앱")

# 모델 파일 업로드
model_file = st.file_uploader("모델 파일을 업로드하세요", type=["pt"])
if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
        temp_model_file.write(model_file.read())
        model_path = temp_model_file.name
    model = YOLO(model_path)
    st.success("모델이 성공적으로 로드되었습니다.")

# 비디오 파일 업로드
uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])

# 레이아웃 설정
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.header("원본 영상")
        if uploaded_file is not None:
            st.video(uploaded_file)
        else:
            st.write("원본 영상을 표시하려면 비디오 파일을 업로드하세요.")

    with col2:
        st.header("사물 검출 결과 영상")
        result_placeholder = st.empty()
        if "processed_video" in st.session_state and st.session_state["processed_video"] is not None:
            result_placeholder.video(st.session_state["processed_video"])
        else:
            result_placeholder.markdown(
                """
                <div style='width:100%; height:620px; background-color:#d3d3d3; display:flex; align-items:center; justify-content:center; border-radius:5px;'>
                    <p style='color:#888;'>여기에 사물 검출 결과가 표시됩니다.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 사물 검출 버튼 클릭 이벤트 처리
if st.button("사물 검출 실행") and uploaded_file and model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_output:
        output_path = temp_output.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    cap = cv2.VideoCapture(temp_input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 예측 수행
        results = model(frame)
        detections = results[0].boxes if len(results) > 0 else []

        if len(detections) > 0:
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                label = f"{class_name} {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # 저장 후 대기 시간
    time.sleep(1)

    # 결과 비디오를 세션 상태에 저장
    st.session_state["processed_video"] = output_path
    result_placeholder.video(output_path)
    st.success("사물 검출이 완료되어 오른쪽에 표시됩니다.")

    # 다운로드 링크 제공
    with open(output_path, "rb") as file:
        st.download_button(
            label="결과 영상 다운로드",
            data=file,
            file_name="detected_video.avi",
            mime="video/avi"
        )
