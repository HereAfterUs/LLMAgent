import logging

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from paddleocr import PaddleOCR

# 设置日志格式和级别
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 获取 API 密钥
ZHIPU_API_KEY = "22a6753b6eb44bb187a00a9bd6ef6a19.lbzGv63p3Glpgzqk"
model = ChatZhipuAI(
    temperature=0.5,
    api_key=ZHIPU_API_KEY,
    model="glm-4-air"
)

# 存储会话历史
store = {}


# 获取会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 改进后的提示模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名资深的语文老师，擅长写作和辅导学生的作文。"
                   "学生给出了作文题目和正文，你需要根据作文题目和正文给出详细的评价。"
                   "同时对学生的作文进行必要的润色。"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 构建链式调用
chain = prompt_template | model
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)


# 生成大模型的评分与建议
def get_response(session_id, title, body):
    config = {"configurable": {"session_id": session_id}}
    input_content = [
        HumanMessage(content=f"题目: {title}\n正文: {body}")
    ]
    result = with_message_history.invoke(
        {"messages": input_content},
        config=config,
    )
    return result.content


# 提取OCR识别的文本
def extract_text_from_ocr_result(ocr_result):
    extracted_text = ""
    for line in ocr_result:
        for word_info in line:
            extracted_text += word_info[1][0] + " "
    return extracted_text.strip()


# 将 OCR 文本分为题目和正文
def split_title_and_body(ocr_text):
    # 找到第一个空格
    first_space_index = ocr_text.find(" ")

    # 分离题目和正文
    title = ocr_text[:first_space_index].strip()
    body = ocr_text[first_space_index:].replace(" ", "").strip()

    return title, body


# 用于处理上传作文图片并调用评分与建议生成
def process_image_and_evaluate(image_path):
    try:
        # 使用 PaddleOCR 进行文本识别
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        result = ocr.ocr(image_path)
        # 提取识别出的文本
        ocr_text = extract_text_from_ocr_result(result)
        # 将 OCR 文本分为题目和正文
        title, body = split_title_and_body(ocr_text)
        # 将提取的题目和正文传给大模型进行作文评价
        evaluation_result = get_response("ocr_session_id", title, body)

        return ocr_text, evaluation_result
    except Exception as e:
        logging.error(f"Error during OCR or evaluation: {e}")
        return None, None
