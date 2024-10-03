import os
import logging
import cv2
import numpy as np
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from paddleocr import PaddleOCR

# 设置日志格式和级别
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # 控制日志级别为 INFO，只输出重要信息
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
                   "同时对学生的作文进行必要的润色"),
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
    # 构建消息内容，并使用 HumanMessage 将其包装为消息
    input_content = [
        HumanMessage(content=f"题目: {title}\n正文: {body}")
    ]
    responses = []
    # 传递 messages 变量给 with_message_history
    for r in with_message_history.stream(
            {"messages": input_content},  # 确保传递 messages 变量
            config=config,
    ):
        responses.append(r.content)
    return "".join(responses)


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
    title = ocr_text[:first_space_index].strip()  # 题目是第一个空格前的内容
    body = ocr_text[first_space_index:].replace(" ", "").strip()  # 正文去除所有空格

    return title, body


# 分析图像亮度
def analyze_image_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness


# CLAHE 自适应直方图均衡化，动态调整参数
def adaptive_histogram_equalization(image_path):
    image = cv2.imread(image_path)
    mean_brightness = analyze_image_brightness(image)

    # 根据亮度动态调整参数
    if mean_brightness < 100:  # 图像较暗
        clip_limit = 3.0
        tile_grid_size = (8, 8)
    else:  # 图像较亮
        clip_limit = 2.0
        tile_grid_size = (8, 8)

    # 应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adjusted = clahe.apply(gray)

    return adjusted


# 用于处理上传作文图片并调用评分与建议生成
def process_image_and_evaluate(image_path):
    try:
        # CLAHE 处理
        processed_image = adaptive_histogram_equalization(image_path)
        temp_image_path = "clahe_processed_image.jpg"
        cv2.imwrite(temp_image_path, processed_image)

        # 使用 PaddleOCR 进行文本识别
        logging.info("开始OCR处理...")
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        result = ocr.ocr(temp_image_path)  # 使用 CLAHE 处理后的图片

        # 提取识别出的文本
        ocr_text = extract_text_from_ocr_result(result)

        # 将 OCR 文本分为题目和正文
        title, body = split_title_and_body(ocr_text)
        print(f"题目: {title}\n")
        print(f"正文: {body}\n")

        # 将提取的题目和正文传给大模型进行作文评价
        logging.info("开始生成作文评价...")
        evaluation_result = get_response("ocr_session_id", title, body)
        logging.info(f"作文评价结果: {evaluation_result}")

        return ocr_text, evaluation_result
    except Exception as e:
        logging.error(f"Error during OCR or evaluation: {e}")
        return None, None
