import os
import streamlit as st
from agents.learning_partner_chinese import get_response as student_response
from agents.teaching_assistant import get_response as teacher_response
from agents.data_analysis import get_response as data_analysis_response
from agents.essay_review import process_image_and_evaluate  # 确保导入正确的函数

# 确保必要的文件夹存在
if not os.path.exists("image"):
    os.makedirs("image")
if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")

# 设置页面标题和布局
st.set_page_config(page_title="AI Agent", layout="wide")

# 侧边栏导航
st.sidebar.title("导航栏")
page = st.sidebar.selectbox(
    "",
    ["虚拟教师", "教学助理", "作文评分（OCR）", "数据分析"],
    label_visibility="hidden"
)

# 检查是否需要清空聊天记录
if "last_page" not in st.session_state:
    st.session_state["last_page"] = None

if st.session_state["last_page"] != page:
    st.session_state["messages"] = [{"role": "ai", "content": "很高兴为您服务🤖"}]  # 初始化消息记录
    st.session_state["last_page"] = page  # 更新当前页面状态
    st.session_state["uploaded_file_paths"] = None  # 重置上传的文件路径


# 聊天界面
def chat_interface(agent_response_function, session_id, input_key, allow_file_upload=False):
    # 显示历史聊天记录
    for message in st.session_state["messages"]:
        st.chat_message(message["role"]).write(message["content"])

    # 文件上传组件（可选）
    if allow_file_upload:
        uploaded_files = st.file_uploader(
            "请上传相关的文件（支持 PDF、Word 等）",
            type=["pdf", "docx", "doc", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            file_paths = []
            for uploaded_file in uploaded_files:
                # 保存上传的文件
                file_path = os.path.join("uploaded_files", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            st.session_state["uploaded_file_paths"] = file_paths
            st.success("文件上传成功")
    else:
        file_paths = None  # 如果不允许文件上传，file_paths 设为 None

    # 用户输入框，指定唯一的 key
    user_input = st.chat_input(placeholder="请输入...", key=input_key)
    if user_input:
        # 更新消息记录
        st.session_state["messages"].append({"role": "human", "content": user_input})
        st.chat_message("human").write(user_input)
        # 获取上传的文件路径列表
        if allow_file_upload:
            file_paths = st.session_state.get("uploaded_file_paths", None)
            # 生成响应，传递 file_paths
            with st.spinner("正在思考中🙃"):
                response = agent_response_function(session_id, user_input, file_paths)
        else:
            # 生成响应，不传递 file_paths
            with st.spinner("正在思考中🙃"):
                response = agent_response_function(session_id, user_input)
        st.session_state["messages"].append({"role": "ai", "content": response})
        st.chat_message("ai").write(response)


# 作文评分（OCR）界面
def ocr_interface():
    st.header("作文评分（OCR）")

    # 将内容放在一个可折叠的区域中
    with st.expander("", expanded=True):
        # 上传图片
        uploaded_file = st.file_uploader("在此上传图片", type=["png", "jpg", "jpeg"])

        # 确认按钮
        if uploaded_file is not None:
            if st.button("确认上传"):
                # 保存图片到 image 文件夹
                image_path = os.path.join("image", uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 调用OCR
                with st.spinner("正在思考中🙃"):
                    ocr_text, evaluation_result = process_image_and_evaluate(image_path)
                    st.write(evaluation_result)


# 数据分析界面
def data_analysis_interface():
    st.header("数据分析")

    # 显示历史聊天记录
    for message in st.session_state["messages"]:
        st.chat_message(message["role"]).write(message["content"])

    # 文件上传组件
    uploaded_files = st.file_uploader(
        "请上传需要分析的文件（支持 PDF、Excel、Word 等）",
        type=["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt", "md", "ppt", "pptx", "png", "jpg", "jpeg", "bmp",
              "gif"],
        accept_multiple_files=True  # 允许多文件上传
    )

    if uploaded_files:
        file_paths = []
        for uploaded_file in uploaded_files:
            # 保存上传的文件
            file_path = os.path.join("uploaded_files", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        st.session_state["uploaded_file_paths"] = file_paths
        st.success("文件上传成功")

    # 聊天输入框
    user_input = st.chat_input(placeholder="请输入...", key="data_analysis_input")
    if user_input:
        # 更新消息记录
        st.session_state["messages"].append({"role": "human", "content": user_input})
        st.chat_message("human").write(user_input)
        # 获取上传的文件路径列表
        file_paths = st.session_state.get("uploaded_file_paths")
        # 生成响应
        with st.spinner("正在思考中🙃"):
            response = data_analysis_response(
                "data_analysis_session",
                user_input,
                file_paths
            )
        st.session_state["messages"].append({"role": "ai", "content": response})
        st.chat_message("ai").write(response)


# 根据选择的页面展示不同内容
if page == "虚拟教师":
    st.header("虚拟教师")
    chat_interface(student_response, session_id="student_session", input_key="student_input")
elif page == "教学助理":
    st.header("教学助理")
    chat_interface(teacher_response, session_id="teaching_assistant_session", input_key="teacher_input",
                   allow_file_upload=True)
elif page == "作文评分（OCR）":
    ocr_interface()
elif page == "数据分析":
    data_analysis_interface()
