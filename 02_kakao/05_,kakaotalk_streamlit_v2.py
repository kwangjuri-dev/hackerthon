# 사람 간의 유사도 방식이 사전에 계산된 파일을 사용하는 것임.
# streamlit share에 배포가 된 파일 버전임.

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
import base64
from datetime import datetime
import json

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
5. 분석 결과를 Obsidian 형식의 Markdown으로 다운로드 받습니다.

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


# Markdown 생성 함수
def generate_markdown(nickname, analysis_result, matching_nicknames):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    markdown = f"""---
time_created: {current_time}
tags:
  - Networking
  - KakaoTalk
  - {nickname}
MOC: [[Networking]]
---

# {nickname} 분석 결과

{analysis_result}

## 네트워크 유사성 스펙트럼
{matching_nicknames}

"""
    return markdown


# 파일 다운로드 링크 생성 함수
def get_binary_file_downloader_html(bin_file, file_label="File"):
    bin_str = base64.b64encode(bin_file).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">다운로드 {file_label}</a>'
    return href


def score_to_stars(score):
    if score >= 0.5:
        return "⭐⭐⭐⭐⭐"
    elif score >= 0.4:
        return "⭐⭐⭐⭐"
    elif score >= 0.3:
        return "⭐⭐⭐"
    elif score >= 0.2:
        return "⭐⭐"
    else:
        return "⭐"


# # JSON 파일 로드
with open("similarity_nickName_top5.json", "r", encoding="utf-8") as f:
    similarity_data = json.load(f)

# 메인 앱 부분
st.title("네트워크 인사이트")
st.write("카카오톡 대화를 통한 스마트 인맥 분석기")
st.write("현재는 랭체인 단톡방만 이용할 수 있습니다.")

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

    # 분석 버튼 추가
    if st.button("분석하기"):
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
                [대화 요약] : 전체적인 대화 요약을 150자 정도로 정리.
                [성격 분석] :
                • (첫 번째 성격 특성) 50자 정도로 정리
                • (두 번째 성격 특성) 50자 정도로 정리
                • (세 번째 성격 특성) 50자 정도로 정리
                [MBTI 분석] : 예상 MBTI와 50자 정도의 해설

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
                    elif line.startswith("[MBTI 분석]"):
                        st.text_area(
                            "MBTI 분석",
                            value=line.split(":")[1].strip(),
                            height=50,
                            disabled=False,
                        )

                # 유사도 데이터 표시
                st.subheader("네트워크 유사성 스펙트럼")

                # 정규표현식 패턴 생성
                nickname_pattern = re.compile(re.escape(nickname), re.IGNORECASE)

                # 매칭되는 모든 닉네임 찾기
                matching_nicknames = [
                    nick
                    for nick in similarity_data.keys()
                    if nickname_pattern.search(nick)
                ]

                if matching_nicknames:
                    st.write(
                        f"'{nickname}'과 유사한 {len(matching_nicknames)}개의 닉네임을 찾았습니다."
                    )

                    all_similar_nicknames = []
                    for matched_nick in matching_nicknames:
                        similar_nicknames = similarity_data[matched_nick]
                        all_similar_nicknames.extend(similar_nicknames)

                    df = pd.DataFrame(all_similar_nicknames)
                    df.columns = ["닉네임", "네트워크 근접도"]
                    df["유사성 지수"] = df["네트워크 근접도"].apply(score_to_stars)
                    df = df.sort_values(
                        "네트워크 근접도", ascending=False
                    ).drop_duplicates("닉네임")

                    # 테이블 스타일 적용
                    st.markdown(
                        """
                    <style>
                        .dataframe {
                            font-size: 16px;
                            width: 100%;
                        }
                        .dataframe th {
                            text-align: left;
                            font-weight: bold;
                            padding: 10px;
                        }
                        .dataframe td {
                            text-align: left;
                            padding: 10px;
                        }
                        .dataframe tr:nth-child(even) {
                            background-color: #f5f5f5;
                        }
                    </style>
                    """,
                        unsafe_allow_html=True,
                    )

                    # 테이블 표시 (네트워크 근접도 열 제외)
                    st.table(df[["닉네임", "유사성 지수"]])

                    st.info(f"총 {len(df)}개의 유사한 닉네임을 표시하고 있습니다.")
                else:
                    st.warning(
                        f"'{nickname}'과 유사한 닉네임에 대한 네트워크 유사성 데이터를 찾을 수 없습니다."
                    )

                # 워드 클라우드 생성
                st.subheader("주요 키워드")
                all_text = " ".join([item.page_content for item in context])
                try:
                    fig = generate_wordcloud(all_text)
                    wordcloud_path = f"{nickname}_wordcloud.png"
                    plt.savefig(wordcloud_path)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"워드 클라우드 생성 중 오류 발생: {str(e)}")
                    st.write("대신 주요 키워드를 텍스트로 표시합니다.")
                    words = all_text.split()
                    word_freq = Counter(words).most_common(20)
                    st.write(
                        ", ".join([f"{word} ({count})" for word, count in word_freq])
                    )
                    wordcloud_path = None

                # 대화 타임라인 시각화
                st.subheader("대화 타임라인")
                timeline_data = generate_timeline_data(context)
                st.line_chart(timeline_data.set_index("날짜"))

                # Markdown 생성
                markdown_result = generate_markdown(
                    nickname, result, matching_nicknames
                )

                # Markdown 파일 다운로드 버튼
                st.markdown("## 분석 결과 다운로드")
                markdown_file = markdown_result.encode()
                st.markdown(
                    get_binary_file_downloader_html(
                        markdown_file, f"{nickname}_분석결과.md"
                    ),
                    unsafe_allow_html=True,
                )

            else:
                st.write(f"'{nickname}'을 포함하는 닉네임의 메시지를 찾을 수 없습니다.")
        else:
            st.warning("닉네임을 입력해주세요.")

    # 임시 파일 삭제
    os.remove("temp_chat.txt")
    if os.path.exists(f"{nickname}_wordcloud.png"):
        os.remove(f"{nickname}_wordcloud.png")
