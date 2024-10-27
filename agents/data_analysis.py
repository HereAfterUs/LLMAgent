import os
from zhipuai import ZhipuAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json

# 初始化ZhipuAI客户端
ZHIPU_API_KEY = "22a6753b6eb44bb187a00a9bd6ef6a19.lbzGv63p3Glpgzqk"  # 请替换为您的实际API密钥
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 会话存储
store = {}


def get_session(session_id: str):
    if session_id not in store:
        store[session_id] = {
            'history': InMemoryChatMessageHistory(),
            'file_content': None
        }
    return store[session_id]


def get_session_history(session_id: str):
    session = get_session(session_id)
    return session['history']


# 提示模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一名优秀的助理，能够帮助用户解决各项问题。特别是在“数据分析”方面。"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 创建链式调用
chain = prompt_template
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)


# 上传文件并提取内容
def upload_and_extract_file(file_path):
    from pathlib import Path

    # 上传文件
    file_object = client.files.create(file=Path(file_path), purpose="file-extract")

    # 提取文件内容
    file_content_response = client.files.content(file_id=file_object.id)
    file_content = json.loads(file_content_response.content)["content"]

    # 删除文件，暂时忽略
    # client.files.delete(file_id=file_object.id)

    return file_content


# 获取响应
def get_response(session_id, user_input, file_path):
    session = get_session(session_id)
    history = session['history']

    # 如果有新上传的文件，提取内容并存储在会话中
    if file_path and os.path.exists(file_path):
        file_content = upload_and_extract_file(file_path)
        session['file_content'] = file_content  # 将文件内容存储在会话中
    else:
        # 如果没有新的文件，尝试从会话中获取文件内容
        file_content = session.get('file_content', None)
        if not file_content:
            return "请先上传需要分析的文件。"

    # 组合用户输入和文件内容
    combined_input = f"以下是用户提供的文件内容：\n{file_content}\n\n用户的问题是：{user_input}"

    # 更新消息历史
    history.add_user_message(user_input)

    # 生成响应
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=[
            {"role": "user", "content": combined_input}
        ],
    )

    assistant_reply = response.choices[0].message.content

    # 将模型回复添加到历史中
    history.add_ai_message(assistant_reply)

    return assistant_reply
