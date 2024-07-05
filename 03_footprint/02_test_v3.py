# output 형식 : json
# 최종 버전 : 2024-07-06

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
import numpy as np
import json

# CSV 파일 경로
file_path = r".\survey_data_kr.csv"

# CSV 파일을 DataFrame으로 읽어오기
df = pd.read_csv(file_path, encoding="utf-8")

# 지정된 컬럼들의 텍스트 데이터를 하나로 묶기
columns_to_combine = [
    "job_field",
    "interest",
    "career_level",
    "networking_purpose",
    "professional_skill",
]
combined_text = df[columns_to_combine].apply(
    lambda row: " ".join(row.values.astype(str)), axis=1
)

# 새로운 Series 생성
combined_series = pd.Series(combined_text, name="combined_text")

# 원본 DataFrame에 새로운 컬럼으로 추가
df["combined_text"] = combined_series

# embedding 선언
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

total_list_datas = df["combined_text"].tolist()

print("유사도 계산 중...")
embed_result = []
for idx, input_data in enumerate(total_list_datas):
    embed_result.append(embeddings.embed_query(input_data))

similarity_matrix = cosine_similarity(embed_result, embed_result)

# 각 사람별로 유사도가 높은 상위 5명 찾기
similarity_results = {}
for i in range(similarity_matrix.shape[0]):
    # 자기 자신을 제외하고 정렬
    similar_indices = np.argsort(similarity_matrix[i])[::-1][1:6]
    similar_scores = similarity_matrix[i][similar_indices]

    # 결과를 딕셔너리에 저장
    similarity_results[i + 1] = [
        {"person": int(idx + 1), "score": float(score)}
        for idx, score in zip(similar_indices, similar_scores)
    ]

# JSON 파일 경로
json_file_path = r".\similarity_top5.json"

# JSON 파일 작성
with open(json_file_path, "w", encoding="utf-8") as jsonfile:
    json.dump(similarity_results, jsonfile, ensure_ascii=False, indent=2)

print(f"JSON 파일이 성공적으로 생성되었습니다: {json_file_path}")

# JSON 파일 내용 출력
print("\nJSON 파일 내용:")
with open(json_file_path, "r", encoding="utf-8") as jsonfile:
    print(json.dumps(json.load(jsonfile), ensure_ascii=False, indent=2))
