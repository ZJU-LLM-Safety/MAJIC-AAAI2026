from openai import OpenAI
attacktype = "gpt-3.5-turbo"
victim_messages = [
    {"role": "user", "content": "hello can i get a loan"},
]

API_SECRET_KEY= "sk-wvdsTSCg4Edw5do22c734f65145349A99b48Ed06202cD59a" # 填写我们给您的apikey
BASE_URL = "https://api.ai-gaochao.cn/v1"
print("attacking.....")
gpt_client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
victim_response = gpt_client.chat.completions.create(model=attacktype, messages=victim_messages, max_tokens=512).choices[0].message.content
print("attacking over.....")
print("answer is ",victim_response)