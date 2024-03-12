from openai import OpenAI
import os
import streamlit as st
import json
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

DEFAULT_MODEL = "gpt-4-1106-preview"

MODELS = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]
CHAT_HISTORY_DIR = "./chat_history"


st.set_page_config(layout="wide")


st.title("chatting")


def init():
    st.session_state.messages = []
    st.session_state.loaded_file = ""


# Initialize chat history
if "messages" not in st.session_state:
    init()
st.session_state["openai_model"] = st.sidebar.selectbox("Choose model:", MODELS)


def generate_log_name():
    prompt = "Please summarize our conversation so that it fits within a short filename. For example, if we talked about turtle migration, use this format: 'turtle_migration'"
    if st.session_state.messages:
        messages = st.session_state.messages.copy()
    else:
        return "empty"
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        stream=False,
    )

    return response.choices[0].message.content


with st.sidebar.popover("Load from file"):
    st.markdown("load from")
    filenames = [""] + os.listdir(CHAT_HISTORY_DIR)
    filename = st.selectbox("Select a file", filenames)
    if filename:
        filepath = os.path.join(CHAT_HISTORY_DIR, filename)
        st.session_state.loaded_file = filename
        if st.button("Load"):
            with open(filepath, "r") as file:
                loaded_dict = json.load(file)
                st.session_state.messages = loaded_dict


with st.sidebar.popover("Save chat to file"):
    st.markdown("name")
    name = st.text_input("(optional) filename", value=st.session_state.loaded_file)

    ymd = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    if st.button(f"Save"):
        if not name:
            name = generate_log_name()
        if not ".json" in filename:
            filename = f"{ymd}_{name}.json"
        st.session_state.loaded_file = filename
        filepath = os.path.join(CHAT_HISTORY_DIR, filename)
        with open(filepath, "w") as file:
            json.dump(st.session_state.messages, file)
        st.write(f"saved to: {filepath}")

if st.sidebar.button("New Chat"):
    init()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
