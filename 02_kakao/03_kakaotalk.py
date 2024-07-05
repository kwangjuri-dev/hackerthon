from kakaotalk_loader import KakaotalkLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

input_question = input("질문을 해 주세요 : ")

# step 1 : KakaotalkLoader 사용
loader = KakaotalkLoader(r"C:\hackerthon\02_kakao\KakaoTalk_20240705_1033_18_476_group.txt")
docs = loader.load()

print(len(docs))

# 결과 출력
# for doc in docs:
#     print(doc)

# step 2
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap = 0)
split_documents = text_splitter.split_documents(docs)

print(len(split_documents))
# print(split_documents)

# step 3
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# step 4
vectorstore = FAISS.from_documents(documents= docs, embedding= embeddings)

# step 5
retriever = vectorstore.as_retriever(k = 10)

print('Result of Search : ', retriever.invoke(input_question))

# step 6
template = """다음의 <context>를 활용하여 질문에 답을 해 주세요. :
<context>
{context}
</context>\n
실제 대화 내용과 <metadata>에 있는 시간에 대한 정보도 같이 소개해 주세요. \n\n

Question : {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# step 7
llm = ChatOpenAI(model_name="gpt-4o", temperature= 0)

# step 8
input_generator = RunnableParallel({"context":retriever, "question": RunnablePassthrough()})

# step 9

retriever_chain = (
    input_generator
    | prompt
    | llm
    | StrOutputParser()
)

result = retriever_chain.invoke(input_question)

print('-------------------------')
print('Question : ', input_question)
print('Answer   : ', result)