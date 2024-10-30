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


def load_teaching_plan():
    example_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'teaching_plan.txt')
    with open(example_path, 'r', encoding='utf-8') as file:
        return file.read()


teaching_plan = load_teaching_plan()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名中小学语文教学助理，你的任务是帮助教师完成一系列教学任务。"
                   "当教师需要生成教案时，你需要根据参考示例的格式和写作风格来撰写符合教师要求的教案。"
                   f"以下是一个教案的示例结构：\n{teaching_plan}"),
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
