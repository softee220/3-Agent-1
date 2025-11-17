#!/usr/bin/env python
# coding: utf-8

# # 🧠 Agent 수업 1주차 – Agent Full-Loop & 설계 심화

# ## 📘 개요
# 
# 이 노트북은 프롬프트 엔지니어링, RAG, API 사용 경험이 있는 수강생을 위한 1주차 심화 실습 자료입니다.
# 
# ### 학습 목표
# 1.  **Agent의 필요성**을 일반 LLM과의 비교를 통해 체감합니다.
# 2.  Function Calling을 이용해 <b>Agent의 전체 동작 사이클(Full-Loop)</b>을 수동으로 구현합니다.
# 3.  여러 도구가 있을 때 LLM이 어떻게 판단하는지 실험합니다.
# 4.  **Agent가 잘못된 판단(Hallucination)을 내리는 경우**를 직접 확인하고, 그에 대한 대응 설계의 필요성을 이해합니다.
# 5.  도구 실행 시 발생할 수 있는 에러를 어떻게 처리하는지 학습합니다.
# 6.  수동 Agent 구현의 한계를 통해 LangChain과 같은 프레임워크의 필요성을 체감합니다.

# In[1]:


import openai
import os
import json
import requests
import wikipedia
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()
MODEL = "gpt-4o-mini"


# ---

# ## 🔧 2. Tool 정의: 모든 함수와 스키마 통합
# 실습에 사용할 모든 함수(Tool)와 그 함수의 명세(JSON Schema)를 이 섹션에서 한 번에 정의합니다.

# In[11]:


# === Tool 1: 오늘의 날씨 함수 및 스키마 ===
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_today_weather(city: str) -> str:
    """지정된 도시의 현재 날씨 정보를 가져옵니다.
    
    Args:
        city: 도시 이름 (예: "Seoul", "Busan", "New York")
    
    Returns:
        JSON 문자열 형태의 날씨 정보 (날씨, 온도, 습도, 체감온도 등)
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # OpenWeatherMap API 응답에서 필요한 정보 추출
        weather_main = data.get("weather", [{}])[0].get("main", "Unknown")
        weather_description = data.get("weather", [{}])[0].get("description", "Unknown")
        temp = data.get("main", {}).get("temp")
        feels_like = data.get("main", {}).get("feels_like")
        humidity = data.get("main", {}).get("humidity")
        city_name = data.get("name", city)
        
        # JSON 형태로 반환
        result = {
            "city": city_name,
            "weather": weather_main,
            "description": weather_description,
            "temp": round(temp, 1) if temp else None,
            "feels_like": round(feels_like, 1) if feels_like else None,
            "humidity": humidity
        }
        return json.dumps(result, ensure_ascii=False)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return json.dumps({"error": f"'{city}' 도시를 찾을 수 없습니다. 도시 이름을 확인해주세요."}, ensure_ascii=False)
        else:
            return json.dumps({"error": f"API 요청 중 에러 발생: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"API 요청 중 에러 발생: {e}"}, ensure_ascii=False)

weather_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_today_weather",
        "description": "지정된 도시의 현재 날씨 정보(날씨 상태, 온도, 체감온도, 습도 등)를 실시간으로 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "날씨를 조회할 도시 이름 (예: Seoul, Busan, New York, Tokyo)"
                }
            },
            "required": ["city"]
        }
    }
}

# === 사용 가능한 함수를 매핑 ===
available_functions = {
    "get_today_weather": get_today_weather
}


# ---

# ## 🔁 3. Agent Full-Loop 수동 구현 (⭐핵심⭐)
# Agent가 실제로 동작하는 전 과정을 단계별로 나누어 직접 코드를 실행해봅니다.

# In[12]:


messages = [{"role": "user", "content": "오늘 서울 날씨 어때?"}]
tools = [weather_tool_schema]

print("💬 1단계: LLM에 Tool 호출 요청")
response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, tool_choice="auto")
response_message = response.choices[0].message
messages.append(response_message)

tool_calls = response_message.tool_calls
if tool_calls:
    print(" LLM이 Tool 사용을 결정했습니다.")
    print("\n 2단계: 결정된 Tool을 실제로 실행")
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        print(f"- 실행 함수: {function_name}({', '.join([f'{k}={v}' for k, v in function_args.items()])})")

        function_response = function_to_call(**function_args)
        print(f"- 실행 결과: {function_response}")

        messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})

    print("\n 3단계: Tool 실행 결과를 바탕으로 최종 답변 생성")
    final_response = client.chat.completions.create(model=MODEL, messages=messages)
    print("\n===== Agent의 최종 답변 =====")
    agent_answer = final_response.choices[0].message.content
    print(agent_answer)
else:
    print(" LLM이 Tool 사용을 결정하지 않았습니다.")
    final_response = client.chat.completions.create(model=MODEL, messages=messages)
    print("\n===== Agent의 최종 답변 =====")
    print(final_response.choices[0].message.content)


# ---

# ### 추가 테스트: 다른 도시 날씨 조회
# 
# 다양한 도시에 대해 날씨 Agent가 잘 작동하는지 테스트
# 

# In[18]:


# 다른 도시들에 대한 날씨 조회 테스트
test_cities = ["Busan", "Tokyo", "New York"]

for city in test_cities:
    print(f"\n{'='*100}")
    print(f"🌍 {city} 날씨 조회")
    print('='*100)
    
    messages = [{"role": "user", "content": f"오늘 {city} 날씨 어때?"}]
    tools = [weather_tool_schema]
    
    # 1단계: Tool 호출 요청
    response = client.chat.completions.create(
        model=MODEL, 
        messages=messages, 
        tools=tools, 
        tool_choice="auto"
    )
    response_message = response.choices[0].message
    messages.append(response_message)
    
    # 2단계: Tool 실행
    tool_calls = response_message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            
            messages.append({
                "tool_call_id": tool_call.id, 
                "role": "tool", 
                "name": function_name, 
                "content": function_response
            })
        
        # 3단계: 최종 답변 생성
        final_response = client.chat.completions.create(model=MODEL, messages=messages)
        print(final_response.choices[0].message.content)
    print()


# 
