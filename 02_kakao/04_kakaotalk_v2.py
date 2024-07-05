from kakaotalk_loader import KakaotalkLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import re

def extract_documents_by_nickname(docs, nickname):
    context = []
    nickname_pattern = re.compile(re.escape(nickname), re.IGNORECASE)
    
    for doc in docs:
        doc_nickname = doc.metadata.get('nickName', '')
        if nickname_pattern.search(doc_nickname):
            context.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
    
    return context

# step 1 : KakaotalkLoader 사용
loader = KakaotalkLoader(r"C:\hackerthon\02_kakao\KakaoTalk_20240705_1300_30_157_group.txt")
docs = loader.load()

# 사용자로부터 nickname 입력 받기
nickname = input("Enter the nickname to extract: ")

# 해당 nickname의 대화 내용 추출
context = extract_documents_by_nickname(docs, nickname)

# # 결과 출력
# if context:
#     print(f"Messages from nicknames containing '{user_nickname}':")
#     for item in context:
#         print(f"Nickname: {item['metadata']['nickName']}")
#         print(f"Message: {item['content']}")
#         print(f"Date: {item['metadata']['createDate']} {item['metadata']['createTime']}")
#         print("-" * 50)
#     print(f"\nTotal messages: {len(context)}")
# else:
#     print(f"No messages found for nicknames containing '{user_nickname}'")


# step 2
template = """다음의 <context>를 활용하여 {nickname}의 대화를 요약 정리해 주세요.:
<context>
{context}
</context>
그리고, 대화를 분석하여 성향을 3개의 블릿 리스트 스타일로 정리해 주세요.
Answer : 
[대화 개수] : 진행된 대화의 개수 표시
[대화 요약] : 전체적인 대화 요약을 100자 정도로 정리.
[성격 분석] : 3개의 블릿 리스트 스타일로 30자 정도의 서술형으로 정리. 
"""
prompt = ChatPromptTemplate.from_template(template)

# step 3
llm = ChatOpenAI(model_name="gpt-4o", temperature= 0)

# step 4
chain = (
    prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke({"context": context, "nickname": nickname})

print(result)