import re
from langchain_core.documents import Document

class KakaoDocument(Document):
    def __repr__(self):
        return f"Document(page_content={self.page_content!r}, metadata={self.metadata!r})"

class KakaotalkLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        campName, conversations = self._process_kakao_chat(text)
        return self._process_conversations(conversations, campName)

    def _process_kakao_chat(self, text):
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

    def _process_conversations(self, conversations, campName):
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
                        doc = self._create_document(current_message, campName, date)
                        if doc:
                            documents.append(doc)
                    nickname, time, content = match.groups()
                    current_message = f"[{nickname}] [{time}] {content}"
                else:
                    current_message += " " + message.strip()
            
            if current_message:
                doc = self._create_document(current_message, campName, date)
                if doc:
                    documents.append(doc)
        
        return documents

    def _create_document(self, message, campName, date):
        match = re.match(r'\[(.+?)\] \[(.+?)\] (.+)', message)
        if match:
            nickname, time, content = match.groups()
            return KakaoDocument(
                page_content=content,
                metadata={
                    "campName": campName,
                    "createDate": date,
                    "createTime": time,
                    "nickName": nickname
                }
            )
        return None