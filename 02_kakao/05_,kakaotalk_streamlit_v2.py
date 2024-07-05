# streamlit share 배포용
# Last Updated : 2024-07-05 17:00

import streamlit as st
from kakaotalk_loader import KakaotalkLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import re
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
from io import BytesIO
from langchain_core.documents import Document

# Sidebar 추가
st.sidebar.title("네트워크 인사이트 정보")
st.sidebar.markdown(
    """
이 애플리케이션은 카카오톡 대화를 분석하여 다음과 같은 정보를 제공합니다:
- 대화 요약
- 성격 분석
- 주요 키워드 워드클라우드
- 대화 타임라인

사용 방법:
1. 카카오톡 대화 내보내기 기능을 사용하여 txt 파일을 생성합니다.
2. 생성된 txt 파일을 업로드합니다.
3. 분석하고자 하는 닉네임을 입력합니다.
4. 분석 결과를 확인합니다.

※ 주의 : gpt-4o 모델이 사용됩니다!!
"""
)

# OpenAI API Key 입력
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API Key가 설정되었습니다!")
else:
    st.sidebar.warning("API Key를 입력해주세요.")

# model
model_name = "gpt-4o"  # 실제 사용 가능한 모델명으로 변경


# 문서 추출 함수
def extract_documents_by_nickname(docs, nickname):
    context = []
    nickname_pattern = re.compile(re.escape(nickname), re.IGNORECASE)

    for doc in docs:
        doc_nickname = doc.metadata.get("nickName", "")
        if nickname_pattern.search(doc_nickname):
            # context.append({"page_content": doc.page_content, "metadata": doc.metadata})
            context.append(
                Document(page_content=doc.page_content, metadata=doc.metadata)
            )

    return context


# 워드 클라우드 생성 함수
def generate_wordcloud(text):
    font_path = "GmarketSansTTFLight.ttf"
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", font_path=font_path
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


# 대화 타임라인 데이터 생성 함수
def generate_timeline_data(context):
    date_counter = Counter([item.metadata["createDate"] for item in context])
    dates = list(date_counter.keys())
    counts = list(date_counter.values())
    return pd.DataFrame({"날짜": dates, "메시지 수": counts})


# 검색한 문서 결과를 하나의 문단으로 합친다.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 메인 앱 부분
st.title("네트워크 인사이트")
st.write("카카오톡 대화를 통한 스마트 인맥 분석기")

# API Key 체크
if not api_key:
    st.warning("사이드바에서 OpenAI API Key를 입력해주세요.")
    st.stop()

# 파일 업로드
uploaded_file = st.file_uploader("카카오톡 대화 txt 파일을 업로드하세요", type="txt")

if uploaded_file is not None:
    # 임시 파일로 저장
    with open("temp_chat.txt", "wb") as f:
        f.write(uploaded_file.getvalue())

    # KakaotalkLoader 사용
    loader = KakaotalkLoader("temp_chat.txt")
    docs = loader.load()

    # 닉네임 입력
    nickname = st.text_input("분석할 닉네임을 입력하세요:")

    if nickname:
        # 해당 nickname의 대화 내용 추출
        context = extract_documents_by_nickname(docs, nickname)

        if context:
            message_count = len(context)
            st.write(f"'{nickname}'을 포함하는 닉네임의 메시지 수: {message_count}")

            # LLM 설정
            template = """다음의 <context>를 활용하여 {nickname}의 대화를 요약 정리해 주세요.
            총 메시지 수는 {message_count}개입니다. 이 수치를 기준으로 분석해 주세요.

            <context>
            {context}
            </context>

            그리고, 대화를 분석하여 성향을 3개의 블릿 리스트 스타일로 정리해 주세요.
            Answer 형식:
            [대화 개수] : {message_count}
            [대화 요약] : 전체적인 대화 요약을 100자 정도로 정리.
            [성격 분석] :
            • (첫 번째 성격 특성)
            • (두 번째 성격 특성)
            • (세 번째 성격 특성)
            [예상 MBTI] : 대화 내용으로 추정되는 MBIT와 해설을 50자 정도로 정리
            """
            prompt = ChatPromptTemplate.from_template(template)

            llm = ChatOpenAI(model_name=model_name, temperature=0)

            chain = prompt | llm | StrOutputParser()

            format_context = format_docs(context)[:5000]

            with st.spinner("분석 중..."):
                result = chain.invoke(
                    {
                        "context": format_context,
                        "nickname": nickname,
                        "message_count": message_count,
                    }
                )

            st.subheader("분석 결과")

            # 결과를 줄 단위로 분리
            lines = result.split("\n")

            # 각 섹션별로 처리
            for line in lines:
                if line.startswith("[대화 개수]"):
                    st.metric(label="대화 개수", value=line.split(":")[1].strip())
                elif line.startswith("[대화 요약]"):
                    # st.subheader("대화 요약")
                    st.text_area(
                        "대화 요약",
                        value=line.split(":")[1].strip(),
                        height=100,
                        disabled=False,
                    )
                elif line.startswith("[성격 분석]"):
                    st.subheader("성격 분석")
                    traits = [
                        trait.strip()
                        for trait in lines[lines.index(line) + 1 :]
                        if trait.strip().startswith("•")
                    ]
                    for trait in traits:
                        st.markdown(trait)

                elif line.startswith("[예상 MBTI]"):
                    # st.subheader("예상 MBTI")
                    st.text_area(
                        "예상 MBTI",
                        value=line.split(":")[1].strip(),
                        height=50,
                        disabled=False,
                    )

            # 워드 클라우드 생성
            st.subheader("주요 키워드")
            all_text = " ".join([item.page_content for item in context])
            try:
                fig = generate_wordcloud(all_text)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"워드 클라우드 생성 중 오류 발생: {str(e)}")
                st.write("대신 주요 키워드를 텍스트로 표시합니다.")
                words = all_text.split()
                word_freq = Counter(words).most_common(20)
                st.write(", ".join([f"{word} ({count})" for word, count in word_freq]))

            # 대화 타임라인 시각화
            st.subheader("대화 타임라인")
            timeline_data = generate_timeline_data(context)
            st.line_chart(timeline_data.set_index("날짜"))

        else:
            st.write(f"'{nickname}'을 포함하는 닉네임의 메시지를 찾을 수 없습니다.")

    # 임시 파일 삭제
    os.remove("temp_chat.txt")
