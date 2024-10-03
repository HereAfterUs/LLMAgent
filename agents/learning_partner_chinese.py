import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ZHIPU_API_KEY = "22a6753b6eb44bb187a00a9bd6ef6a19.lbzGv63p3Glpgzqk"
model = ChatZhipuAI(
    temperature=0.5,
    api_key=ZHIPU_API_KEY,
    model="glm-4-air"
)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名中小学语文教师，你的任务是帮助学生解决一系列学习难题，为学生提供个性化指导。"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt_template | model
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)


def get_response(session_id, user_input):
    config = {"configurable": {"session_id": session_id}}
    responses = []
    for r in with_message_history.stream(
            {
                "messages": [HumanMessage(content=user_input)]
            },
            config=config,
    ):
        responses.append(r.content)
    return "".join(responses)
