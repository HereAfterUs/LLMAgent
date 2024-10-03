import os
import streamlit as st
from agents.learning_partner_chinese import get_response as student_response
from agents.teaching_assistant import get_response as teacher_response
from agents.demo import process_image_and_evaluate  # 修改后的接口函数

# 确保 image 文件夹存在
if not os.path.exists("image"):
    os.makedirs("image")

# 设置页面标题和布局
st.set_page_config(page_title="AI Agent", layout="wide")

# 侧边栏导航
st.sidebar.title("导航栏")
page = st.sidebar.selectbox("", ["虚拟教师", "教学助理", "作文评分（OCR）"], label_visibility="hidden")

# 检查是否需要清空聊天记录
if "last_page" not in st.session_state:
    st.session_state["last_page"] = None

if st.session_state["last_page"] != page:
    st.session_state["messages"] = [{"role": "ai", "content": f"很高兴为您服务🤖"}]  # 初始化消息记录
    st.session_state["last_page"] = page  # 更新当前页面状态


# 聊天界面
def chat_interface(agent_response_function, input_key):
    # 显示历史聊天记录
    for message in st.session_state["messages"]:
        st.chat_message(message["role"]).write(message["content"])

    # 用户输入框，指定唯一的 key
    user_input = st.chat_input(placeholder="请输入...", key=input_key)
    if user_input:
        # 更新消息记录
        st.session_state["messages"].append({"role": "human", "content": user_input})
        st.chat_message("human").write(user_input)
        # 生成响应
        with st.spinner(f"正在思考中🙃"):
            response = agent_response_function("88888888", user_input)
        st.session_state["messages"].append({"role": "ai", "content": response})
        st.chat_message("ai").write(response)


# 作文评分（OCR）界面，使用 expander 和 container 限制大小
def ocr_interface():
    st.header("")

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

                # 显示上传的图片
                # st.image(uploaded_file, caption="已上传的作文图片", use_column_width=True)
                # 调用OCR
                with st.spinner("正在思考中🙃"):
                    ocr_text, evaluation_result = process_image_and_evaluate(image_path)
                    st.write(evaluation_result)


# 根据选择的页面展示不同内容
if page == "虚拟教师":
    chat_interface(student_response, input_key="student_input")
elif page == "教学助理":
    chat_interface(teacher_response, input_key="teacher_input")
elif page == "作文评分（OCR）":
    ocr_interface()
