import re
from datetime import datetime
from langchain_core.documents import Document

def process_kakao_chat(text):
    lines = text.split('\n')
    campName = lines[0].split(' 님과')[0]
    
    current_date = None
    conversations = {}
    
    for line in lines[1:]:
        date_match = re.match(r'--------------- (\d{4}년 \d{1,2}월 \d{1,2}일 [월화수목금토일]요일) ---------------', line)
        if date_match:
            current_date = date_match.group(1)
            conversations[current_date] = []
        elif current_date:
            conversations[current_date].append(line)
    
    return campName, conversations

def process_conversations(conversations, campName):
    documents = []
    for date, messages in conversations.items():
        current_message = ""
        skip_next = False
        
        for message in messages:
            if "님이 들어왔습니다." in message or "님이 나갔습니다." in message:
                continue
            if "불법촬영물등 식별 및 게재제한 조치 안내" in message:
                skip_next = True
                continue
            if skip_next:
                skip_next = False
                continue
            
            match = re.match(r'\[(.+?)\] \[(.+?)\] (.+)', message)
            if match:
                if current_message:
                    doc = create_document(current_message, campName, date)
                    if doc:
                        documents.append(doc)
                nickname, time, content = match.groups()
                current_message = f"[{nickname}] [{time}] {content}"
            else:
                current_message += " " + message.strip()
        
        if current_message:
            doc = create_document(current_message, campName, date)
            if doc:
                documents.append(doc)
    
    return documents

def create_document(message, campName, date):
    match = re.match(r'\[(.+?)\] \[(.+?)\] (.+)', message)
    if match:
        nickname, time, content = match.groups()
        return Document(
            page_content=content,
            metadata={
                "campName": campName,
                "createDate": date,
                "createTime": time,
                "nickName": nickname
            }
        )
    return None

# 텍스트 데이터
kakao_chat = """지피터스 11기 | 랭체인으로 개발자되기 님과 카카오톡 대화
저장한 날짜 : 2024-07-05 10:33:21
--------------- 2024년 6월 10일 월요일 ---------------
이른아침에님이 들어왔습니다.타인, 기관 등의 사칭에 유의해 주세요. 금전 또는개인정보를 요구 받을 경우 신고해 주시기 바랍니다.운영정책을 위반한 메시지로 신고 접수 시 카카오톡 이용에 제한이 있을 수 있습니다. 
불법촬영물등 식별 및 게재제한 조치 안내
그룹 오픈채팅방에서 동영상・압축파일 전송 시 전기통신사업법에 따라 방송통신심의위원회에서 불법촬영물등으로 심의・의결한 정보에 해당하는지를 비교・식별 후 전송을 제한하는 조치가 적용됩니다. 불법촬영물등을 전송할 경우 관련 법령에 따라 처벌받을 수 있사오니 서비스 이용 시 유의하여 주시기 바랍니다.
스마트한닥터(청강)님이 들어왔습니다.
안계준(청강)님이 들어왔습니다.
손명진님이 들어왔습니다.
머리 빗는 네오님이 들어왔습니다.
김이언 | LA | 교육님이 들어왔습니다.
[김이언 | LA | 교육] [오후 12:16] 안녕하세요~ 반갑습니다. 잘 부탁드려요!
김도연(Stephi)/통계연구님이 들어왔습니다.
유니 | 노코드 | 코파일럿프로님이 들어왔습니다.
[유니 | 노코드 | 코파일럿프로] [오후 12:29] 안녕하세요 노코드백서, 코파일럿 파트너 유니입니다! 잘부탁드립니다💕😍
[유니 | 노코드 | 코파일럿프로] [오후 12:29] 이모티콘
김지홍 JASON님이 들어왔습니다.
박시우님이 들어왔습니다.
김채정님이 들어왔습니다.
먹보 네오님이 들어왔습니다.
☆슬로앤스테디님이 들어왔습니다.
[☆슬로앤스테디] [오후 12:37] 삭제된 메시지입니다.
[☆슬로앤스테디] [오후 12:38] 안녕허세요!
[곽은철 | 파트너] [오후 12:42] 안녕하세요! 랭체인방은 입장하실분들이 충분히 입장한 이후에 한번에 안내드릴게요!
[댕댕이멍멍 | 파트너] [오후 12:42] 안녕하세요. 반갑습니다.
[반문섭 /공공 /기획&설계(청강)] [오후 12:48] @곽은철 | 파트너
문의.. 저번 프롬프트 강의하실 때 시트 나눠준다고 한 사용한 도구가 뭔가요
프롬프팅 하면서 chtgpt의 저장만이 아니라 필요한 것들을 저장하고 활용하는 방식에 응용하면 어떨까 싶어서....
김도남님이 들어왔습니다.
waters님이 들어왔습니다.
Tony (청강)님이 들어왔습니다.
조대연 | 교육 | 대학님이 들어왔습니다.
[조대연 | 교육 | 대학] [오후 1:18] 안녕하세요? 반갑습니다. 10기에 처음 참여했고 이번 두번째 참여하고 있습니다. 열심히 하겠습니다.
박정기님이 들어왔습니다.
[댕댕이멍멍 | 파트너] [오후 2:08] 신청해 주셔서 감사합니다.
GPTer (청강)님이 들어왔습니다.
Summer(청강)님이 들어왔습니다.
하이젠버그님이 들어왔습니다.
카일(청강)님이 들어왔습니다.
허세임 AI | SNS자동화 파트너님이 들어왔습니다.
[허세임 AI | SNS자동화 파트너] [오후 5:33] 안녕하세요! SNS자동화 파트너 허세임입니다. 잘부탁드립니다 😆 
일하자 (청강)님이 들어왔습니다.
일하자 (청강)님이 나갔습니다.
일하자 (청강)님이 들어왔습니다.
Sole(청강)님이 들어왔습니다.
양경호(MEC:D) | 랭체인 서포터님이 들어왔습니다.
자니(청강)님이 들어왔습니다.
류재혁님이 들어왔습니다.
--------------- 2024년 6월 11일 화요일 ---------------
로긴(김정민)님이 들어왔습니다.
[김도연(Stephi)/통계연구] [오전 10:41] 문자로 보내주신 설문링크 다시보내주실 수 있나요? 연결이 안되서요.
[Spark] [오전 10:50] 삭제된 메시지입니다.
[곽은철 | 파트너] [오전 10:50] 오 이런... 금방 다시 보내드리겠습니다
[Spark] [오전 10:50] [Web발신]
안녕하세요 랭체인 & LLM으로 대체불가 개발자 되기에 곽은철 파트너 입니다. 본격적인 캠프가 시작되기 전에 간단한 설문조사를 준비했습니다. 이후 진행과정에서 꼭 필요한 정보이오니 반드시 설문조사에 참여해주시면 감사하겠습니다.
설문 링크 : https://forms.gle/vZGsNevYUjaTeQT28
가온王님이 들어왔습니다.
타래 | 청강님이 들어왔습니다.
[타래 | 청강] [오후 2:41] 안녕하세요. 노코드 서포터 타래라고 합니다. 잘 부탁드려요!
[댕댕이멍멍 | 파트너] [오후 2:42] 반갑습니다. 잘 부탁드립니다.
[유니 | 노코드 | 코파일럿프로] [오후 2:42] 어거오세요~ 저희 서포터님🥰❤️
[양경호(MEC:D) | 랭체인 서포터] [오후 2:53] 안녕하세요 반갑습니다~~
[유니 | 노코드 | 코파일럿프로] [오후 3:03] @양경호(MEC:D) | 랭체인 서포터님 역시 지난기수에 이어서 서포터를 하시는군요!! 너무 기대됩니다🤩
[☆슬로앤스테디] [오후 3:03] 오늘 청강하시는 분들, 혹시 어떤거 들으시나요?
한영균(청강)님이 들어왔습니다.
[Fri-mer/연구원/생물] [오후 3:47] 오늘은 우수사례 발표져?
[곽은철 | 파트너] [오후 3:49] 이번주는 캠프 OT와 파트너 사례발표, 강의가 있습니다
[양경호(MEC:D) | 랭체인 서포터] [오후 4:05] 유니님 반가워요ㅋㅋ 안 짤리고 다시 돌아왔어요! 열심히 준비해서 재미있는거 많이 보여드릴게요
끌매지식(청강)님이 들어왔습니다.
윤주식_청강님이 들어왔습니다.
박종우(청강)님이 들어왔습니다.
tony(청강)님이 들어왔습니다.
아이앰데이타 | 개발님이 들어왔습니다.
[Fri-mer/연구원/생물] [오후 10:26] 혹시 오늘 캠프 끝났나요? 끝난것 같기도 하네요.
[Spark] [오후 10:27] 화요일 캠프 오티는 끝났고, 각 캠프마다 소모임으로 들어가서 상세 오티 진행 중입니다.
[Fri-mer/연구원/생물] [오후 10:35] 제가 개인적 일 때문에 지금 현재 나와있는데.. 지금 들어가겠습니다
[먹보 네오] [오후 10:37] 저희 캠프 OT는 내일 아니였나요?!
[Fri-mer/연구원/생물] [오후 10:38] 엇 캠프 zoom 들어가려고 했는데, 아무도 없네요;;
휘파람 프로도님이 들어왔습니다.
[Fri-mer/연구원/생물] [오후 10:39] 있으시지만, 말씀이 없으셔요..
[Spark] [오후 10:39] 수욜 캠프 ot는 내일이고, 오늘은 화욜 캠프 오티만 진행했습니다~
[이른아침에] [오후 10:39] 소회의실로 다 들어가 계신 듯요.. 
[Spark] [오후 10:39] 오른쪽 아래 소회의 목록 중에 골라 들어가시면 되요
[Fri-mer/연구원/생물] [오후 10:40] 앗 감사합니다
[이른아침에] [오후 10:40] 사진
"""

campName, conversations = process_kakao_chat(kakao_chat)
documents = process_conversations(conversations, campName)

print(f"Camp Name: {campName}")
print(f"\nTotal number of documents: {len(documents)}")
print("\nSample documents:")
for i, doc in enumerate(documents[:5]):  # Print first 5 documents as a sample
    print(f"\nDocument {i+1}:")
    print(f"Content: {doc.page_content}")
    print("Metadata:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")