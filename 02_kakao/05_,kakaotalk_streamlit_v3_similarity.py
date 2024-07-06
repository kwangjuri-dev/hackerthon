# 사람별 유사도 계산을 직접 수행하는 방식을 적용한 것임. 
# 그런데, 시간도 오래 걸리고, 로직이 좀 이상함. 

import streamlit as st
from kakaotalk_loader import KakaotalkLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

from kakaotalk_loader import KakaotalkLoader
from langchain_community.utils.math import cosine_similarity
import numpy as np
from typing import List, Dict

# from langchain.schema import Document


def truncate_text(text, max_length=100):
    """텍스트를 주어진 최대 길이로 자르는 함수"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def extract_unique_nicknames(docs: List[Document]) -> List[str]:
    unique_nicknames = set()
    for doc in docs:
        nickname = doc.metadata.get("nickName")
        if nickname:
            unique_nicknames.add(nickname)
    return list(unique_nicknames)


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


def combine_conversations(conversations: List[Dict]) -> str:
    return "\n\n".join(conv.page_content for conv in conversations)


def extract_and_combine_all_conversations(docs: List[Document]) -> Dict[str, str]:
    all_nicknames = extract_unique_nicknames(docs)
    combined_conversations = {}

    for nickname in all_nicknames:
        conversations = extract_documents_by_nickname(docs, nickname)
        combined_conversations[nickname] = combine_conversations(conversations)

    return combined_conversations


# -------------------------------------------------------
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

※ 주의 : gpt-4 모델이 사용됩니다!!
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
def generate_markdown(nickname, analysis_result, wordcloud_path):
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

## 주요 키워드
![워드클라우드]({wordcloud_path})

"""
    return markdown


# 파일 다운로드 링크 생성 함수
def get_binary_file_downloader_html(bin_file, file_label="File"):
    bin_str = base64.b64encode(bin_file).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">다운로드 {file_label}</a>'
    return href


def score_to_stars(score):
    if score >= 0.7:
        return "⭐⭐⭐⭐⭐"
    elif score >= 0.5:
        return "⭐⭐⭐⭐"
    elif score >= 0.3:
        return "⭐⭐⭐"
    elif score >= 0.2:
        return "⭐⭐"
    else:
        return "⭐"


# # JSON 파일 로드
# with open("similarity_nickName_top5.json", "r", encoding="utf-8") as f:
#     similarity_data = json.load(f)

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

    # 유사도 검사 로직
    # 임시 파일로 저장
    with open("temp_chat.txt", "wb") as f:
        f.write(uploaded_file.getvalue())

    # step 1 : KakaotalkLoader 사용
    loader = KakaotalkLoader("temp_chat.txt")
    docs = loader.load()

    # print(docs)

    all_combined_conversations = extract_and_combine_all_conversations(docs)

    # 결과 출력 (예시)
    # print(len(all_combined_conversations))

    def conversations_to_dataframe(all_combined_conversations, max_length=200):
        # 딕셔너리를 리스트로 변환하면서 텍스트 길이 제한
        data = [
            {
                "nickName": nickname,
                "conversation_all": truncate_text(conversation, max_length),
            }
            for nickname, conversation in all_combined_conversations.items()
        ]

        # 데이터프레임 생성
        df = pd.DataFrame(data)

        return df

    # 사용 예시
    all_combined_conversations = extract_and_combine_all_conversations(docs)
    df = conversations_to_dataframe(all_combined_conversations)

    total_list_datas = df["conversation_all"].tolist()

    # 닉네임과 인덱스를 매핑하는 딕셔너리 생성
    nickname_to_index = {
        nickname: index for index, nickname in enumerate(df["nickName"])
    }
    index_to_nickname = {
        index: nickname for nickname, index in nickname_to_index.items()
    }

    # embedding 선언
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    with st.spinner("유사도 계산 중... 잠시만 기다려 주세요."):
        # st.write("유사도 계산 중... 잠시만 기다려 주세요.")
        embed_result = []
        for idx, input_data in enumerate(total_list_datas):
            embed_result.append(embeddings.embed_query(input_data))

        similarity_matrix = cosine_similarity(embed_result, embed_result)

        # 각 사람별로 유사도가 높은 상위 5명 찾기
        similarity_results = {}
        for nickname, i in nickname_to_index.items():
            # 자기 자신을 제외하고 정렬
            similar_indices = np.argsort(similarity_matrix[i])[::-1][1:6]
            similar_scores = similarity_matrix[i][similar_indices]

            # 결과를 딕셔너리에 저장 (닉네임 사용)
            similarity_results[nickname] = [
                {"person": index_to_nickname[int(idx)], "score": float(score)}
                for idx, score in zip(similar_indices, similar_scores)
            ]

    # st.write("유사도 계산이 완료 되었습니다!")

    # -----------------------------
    # # KakaotalkLoader 사용
    # loader = KakaotalkLoader("temp_chat.txt")
    # docs = loader.load()

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
                [대화 요약] : 전체적인 대화 요약을 200자 정도로 정리.
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
                if nickname in similarity_results:
                    similar_nicknames = similarity_results[nickname]
                    df = pd.DataFrame(similar_nicknames)
                    df.columns = ["닉네임", "네트워크 근접도"]
                    df["유사성 지수"] = df["네트워크 근접도"].apply(score_to_stars)

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
                else:
                    st.write(
                        f"'{nickname}'에 대한 네트워크 유사성 데이터를 찾을 수 없습니다."
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
                markdown_result = generate_markdown(nickname, result, wordcloud_path)

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
