# output 형식 : json
# 시각화 테스트

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
import numpy as np
import json
import networkx as nx
import plotly.graph_objects as go

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

# # JSON 파일 내용 출력
# print("\nJSON 파일 내용:")
# with open(json_file_path, "r", encoding="utf-8") as jsonfile:
#     print(json.dumps(json.load(jsonfile), ensure_ascii=False, indent=2))


# 네트워크 그래프 생성
G = nx.Graph()

# 노드 추가
for i in range(1, len(similarity_results) + 1):
    G.add_node(i, name=df.loc[i-1, 'name'])  # 'name' 열이 있다고 가정

# 엣지 추가 (각 사람마다 상위 5개의 유사도만 반영)
for person, similar_persons in similarity_results.items():
    for sim in similar_persons:
        G.add_edge(person, sim['person'], weight=sim['score'])

# 그래프 레이아웃 계산
pos = nx.spring_layout(G)

# 엣지 트레이스
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# 노드 트레이스
node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# 노드 색상 및 호버 텍스트 설정
node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append(f'Person: {G.nodes[node+1]["name"]}<br># of connections: {len(adjacencies[1])}')

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

# 그래프 생성
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Similarity Network (Top 5 connections per person)',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: NetworkX & Plotly",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

# HTML 파일로 저장
fig.write_html("similarity_network.html")
print("네트워크 그래프가 'similarity_network.html' 파일로 저장되었습니다.")

# (선택사항) 그래프 표시
fig.show()