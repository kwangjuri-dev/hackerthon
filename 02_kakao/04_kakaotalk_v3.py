# 2024-07-06
# 단톡방에 참여 한 모든 사람들에 대한 유사도 검사!!!!

from kakaotalk_loader import KakaotalkLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
import numpy as np
import json
import re
from typing import List, Dict
from langchain.schema import Document
import pandas as pd

def truncate_text(text, max_length=100):
    """텍스트를 주어진 최대 길이로 자르는 함수"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def extract_unique_nicknames(docs: List[Document]) -> List[str]:
    unique_nicknames = set()
    for doc in docs:
        nickname = doc.metadata.get('nickName')
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


# step 1 : KakaotalkLoader 사용
loader = KakaotalkLoader(r"C:\hackerthon\02_kakao\KakaoTalk_20240706_0845_03_230_group.txt")
docs = loader.load()

# print(docs)

all_combined_conversations = extract_and_combine_all_conversations(docs)

# 결과 출력 (예시)
print(len(all_combined_conversations))

def conversations_to_dataframe(all_combined_conversations, max_length=200):
    # 딕셔너리를 리스트로 변환하면서 텍스트 길이 제한
    data = [
        {
            "nickName": nickname, 
            "conversation_all": truncate_text(conversation, max_length)
        } 
        for nickname, conversation in all_combined_conversations.items()
    ]
    
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    return df

# 사용 예시
all_combined_conversations = extract_and_combine_all_conversations(docs)
df = conversations_to_dataframe(all_combined_conversations)

# 결과 출력
print(df.head())
print(df.info())

total_list_datas = df["conversation_all"].tolist()

# 닉네임과 인덱스를 매핑하는 딕셔너리 생성
nickname_to_index = {nickname: index for index, nickname in enumerate(df['nickName'])}
index_to_nickname = {index: nickname for nickname, index in nickname_to_index.items()}

# embedding 선언
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("유사도 계산 중...")
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

# JSON 파일 경로
json_file_path = r".\similarity_nickName_top5.json"

# JSON 파일 작성
with open(json_file_path, "w", encoding="utf-8") as jsonfile:
    json.dump(similarity_results, jsonfile, ensure_ascii=False, indent=2)

print(f"JSON 파일이 성공적으로 생성되었습니다: {json_file_path}")

# JSON 파일 내용 출력
print("\nJSON 파일 내용:")
with open(json_file_path, "r", encoding="utf-8") as jsonfile:
    print(json.dumps(json.load(jsonfile), ensure_ascii=False, indent=2))