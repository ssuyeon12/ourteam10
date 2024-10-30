import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "포트폴리오 주제가 뭔가요?",
    "프로잭트를 진행하며 어떤 부분이 가장 어려웠나요?",
    "조원은 누구인가요?",
    "조장은 누구인가요?",
    "유재현님은 어떤 장점을 가졌나요?",
    "황지영님은 어떤 장점을 가졌나요?",
    "한수연님은 어떤 장점을 가졌나요?",
    "프로잭트 기간은 어떻게 되나요?",
    "영업시간이 어떻게 되나요?",
    "가격이 어떻게 되나요?",
    "주차가 가능한가요?",
    "사람들이 좋아하는 메뉴는 무엇인가요?",
    "목요일 12시에 예약 가능한가요?",
    "메뉴가 무엇이 있나요?",
    "위치가 어디인가요?"
]

answers = [
    "포트폴리오 주제는 딥보이스 기술과 상담챗봇을 이용해 심리상담 챗봇을 만드는 것 입니다.",
    "조장은 유재현 입니다.",
    "조원은 한수연, 황지영 입니다.",
    "유재현 오빠는 섬세하고 다정해요! 정리, 수렴을 굉장히 잘합니다^^",
    "황지영 언니는 츤데레 같지만 따뜻합니다^^ 그리고 숨.고 입니다",
    "한수연은 유머 & 아이디어 뱅크~^^ ",
    "총 3주 입니다. 기획과 구현, 발표준비를 했습니다.",
    "연결성 있도록 상담챗봇을 만드는 것과 딥보이스 TTS 기술 구현이 어려웠어요.",
    "평일 영업시간은 10:00 - 21:00, 주말은 10:00 - 22:00입니다.",
    "메뉴 가격은 비빔밥 9000원, 김치찌개 8000원, 된장찌개 8000원, 갈비탕 12000원입니다.",
    "네, 주차 가능합니다.",
    "사람들이 가장 좋아하는 메뉴는 비빔밥과 갈비탕입니다.",
    "목요일 12시에 예약 가능합니다.",
    "메뉴에는 비빔밥, 김치찌개, 된장찌개, 갈비탕 등이 있습니다.",
    "맛있는 한식당은 강남 국기원 사거리 삼원빌딩 1층에 있습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("식당 챗봇")
st.write("한식당에 관한 질문을 입력해보세요. 예: 영업시간이 어떻게 되나요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")