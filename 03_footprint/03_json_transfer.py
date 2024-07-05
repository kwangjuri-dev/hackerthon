# csv data file을 json 파일로 변환

import csv
import json

# CSV 파일 경로
csv_file_path = r".\survey_data_kr.csv"

# JSON 파일 경로
json_file_path = r".\survey_data_kr.json"

# 한글 키를 영문으로 변환하는 딕셔너리
key_mapping = {
    "연번": "id",
    "이름": "name",
    "이메일": "email",
    "전화번호": "phone",
    "직업분야": "job_field",
    "관심사": "interest",
    "경력수준": "career_level",
    "네트워킹목적": "networking_purpose",
    "전문기술": "professional_skill",
    "거주지": "residence",
}

# CSV 파일을 읽고 JSON으로 변환
data = {}
with open(csv_file_path, "r", encoding="utf-8") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        id = row["연번"]
        english_row = {key_mapping[k]: v for k, v in row.items() if k != "연번"}
        if id not in data:
            data[id] = []
        data[id].append(english_row)

# JSON 파일로 저장
with open(json_file_path, "w", encoding="utf-8") as jsonfile:
    json.dump(data, jsonfile, ensure_ascii=False, indent=2)

print(f"JSON 파일이 성공적으로 생성되었습니다: {json_file_path}")

# JSON 파일의 내용 확인 (처음 2개 항목만)
print("\nJSON 파일의 처음 2개 항목:")
with open(json_file_path, "r", encoding="utf-8") as jsonfile:
    json_data = json.load(jsonfile)
    print(json.dumps(dict(list(json_data.items())[:2]), ensure_ascii=False, indent=2))
