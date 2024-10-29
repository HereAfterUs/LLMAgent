# agents/data_analysis.py

import os
from zhipuai import ZhipuAI
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 初始化ZhiPuAI客户端
ZHIPU_API_KEY = "22a6753b6eb44bb187a00a9bd6ef6a19.lbzGv63p3Glpgzqk"  # 使用环境变量存储密钥
client = ZhipuAI(api_key=ZHIPU_API_KEY)

import json

# 会话存储
store = {}


def get_session(session_id: str):
    if session_id not in store:
        store[session_id] = {
            'history': InMemoryChatMessageHistory(),
            'file_contents': None
        }
    return store[session_id]


def get_session_history(session_id: str):
    session = get_session(session_id)
    return session['history']


# 提示模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一名专业的助理，具备较强的数据分析能力，并以专业的角度回答用户的问题。"),
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


# 上传并提取多个文件的内容
def upload_and_extract_files(file_paths):
    file_contents = []
    for idx, file_path in enumerate(file_paths):
        # 上传文件
        file_object = client.files.create(file=Path(file_path), purpose="file-extract")
        # 提取文件内容
        file_content_response = client.files.content(file_id=file_object.id)
        file_content = json.loads(file_content_response.content)["content"]
        file_contents.append(file_content)
        # 删除文件，防止超过文件数量限制（可选）
        # client.files.delete(file_id=file_object.id)
    return file_contents


# 获取响应
def get_response(session_id, user_input, file_paths):
    session = get_session(session_id)
    history = session['history']

    # 如果有新上传的文件，提取内容并存储在会话中
    if file_paths:
        file_contents = upload_and_extract_files(file_paths)
        session['file_contents'] = file_contents  # 将文件内容存储在会话中
    else:
        # 尝试从会话中获取文件内容
        file_contents = session.get('file_contents', None)
        if not file_contents:
            return "请先上传需要分析的文件。"

    # 构建消息内容，将不同文件的内容放入消息中
    message_content = "请对以下文件进行分析，并回答我的问题：\n\n"
    for idx, content in enumerate(file_contents, 1):
        message_content += f"第{idx}个文件内容如下：\n{content}\n\n"

    message_content += f"用户的问题是：{user_input}"

    # 更新消息历史
    history.add_user_message(user_input)

    # 生成响应
    response = client.chat.completions.create(
        model="glm-4-long",
        messages=[
            {"role": "user", "content": message_content}
        ],
    )

    assistant_reply = response.choices[0].message.content

    # 将模型回复添加到历史中
    history.add_ai_message(assistant_reply)

    return assistant_reply
