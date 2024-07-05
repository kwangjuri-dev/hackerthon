from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Selenium 웹드라이버 설정 (Chrome 사용 예시)
driver = webdriver.Chrome()

url = "https://www.gpters.org/c/llm/langchain-streamilt"
driver.get(url)

try:
    # 페이지가 로드될 때까지 대기
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/div/div/div[2]/div/div[2]/div[1]/main/div[1]/div/div/div/div/div/div[1]/div/div[1]/div/div[1]/div[2]/button/div/div[1]/a"))
    )
    print(f"작성자: {element.text}")
except Exception as e:
    print(f"요소를 찾을 수 없습니다: {e}")
 
driver.quit()