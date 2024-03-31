from openai import OpenAI
import os
import streamlit as st
import json
from datetime import datetime

from dotenv import load_dotenv

from chat_client import AnthropicClient, OpenAIClient

load_dotenv()


clients = {
    "Open AI": OpenAIClient,
    "Claude": AnthropicClient,
}

CHAT_HISTORY_DIR = "./chat_history"

if "messages" not in st.session_state:
    st.session_state.messages = []
st.set_page_config(layout="wide")


st.title("chatting")


def reset():
    st.session_state.messages = []
    st.session_state.loaded_file = ""


# # Initialize chat history
if "messages" not in st.session_state:
    reset()
# st.session_state["openai_model"] = st.sidebar.selectbox("Choose model:", MODELS)


# def generate_log_name():
#     prompt = "Please summarize our conversation so that it fits within a short filename. For example, if we talked about turtle migration, use this format: 'turtle_migration'"
#     if st.session_state.messages:
#         messages = st.session_state.messages.copy()
#     else:
#         return "empty"
#     messages.append({"role": "user", "content": prompt})
#     response = client.chat.completions.create(
#         model=st.session_state["openai_model"],
#         messages=[{"role": m["role"], "content": m["content"]} for m in messages],
#         stream=False,
#     )

#     return response.choices[0].message.content


def get_client():
    client = clients[st.session_state.client_name]
    return client(messages=st.session_state.messages)


def chat_sidebar():
    if st.sidebar.button("New Chat"):
        reset()

    st.sidebar.selectbox("Choose client", clients.keys(), key="client_name")

    client = get_client()

    client.model = st.sidebar.selectbox("Choose model:", client.models, key="model")
    # with st.sidebar.popover("Load from file"):
    #     st.markdown("load from")
    #     filenames = [""] + os.listdir(CHAT_HISTORY_DIR)
    #     filename = st.selectbox("Select a file", filenames)
    #     if filename:
    #         filepath = os.path.join(CHAT_HISTORY_DIR, filename)
    #         st.session_state.loaded_file = filename
    #         if st.button("Load"):
    #             client.load_from_file(filepath)
    #             # with open(filepath, "r") as file:
    #             #     loaded_dict = json.load(file)
    #             #     st.session_state.messages = loaded_dict

    # with st.sidebar.popover("Save chat to file"):
    #     st.markdown("name")
    #     name = st.text_input("(optional) filename", value=st.session_state.loaded_file)

    #     ymd = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    #     if st.button(f"Save"):
    #         if not name:
    #             name = generate_log_name()
    #         if not ".json" in filename:
    #             filename = f"{ymd}_{name}.json"
    #         st.session_state.loaded_file = filename
    #         filepath = os.path.join(CHAT_HISTORY_DIR, filename)
    #         with open(filepath, "w") as file:
    #             json.dump(st.session_state.messages, file)
    #         st.write(f"saved to: {filepath}")

    return client


def chat_app():
    client = chat_sidebar()

    container_A = st.container()
    container_B = st.container()

    # Display chat messages from history on app rerun
    with container_A:
        for message in client.messages:
            chat_msg = st.chat_message(message["role"])
            chat_msg.markdown(message["content"])

    # Accept user input
    # with
    with container_B:
        prompt = st.chat_input("What is up?")
    if prompt:
        client.prompt(prompt)

        # Write p
        with container_A:
            chat_msg = st.chat_message("user")
            chat_msg.markdown(prompt)

        with container_A:
            stream = client.stream_generator()
            chat_msg = st.chat_message("assistant")
            response = chat_msg.write_stream(stream)

        client.add_message(role="assistant", message=response)

    # Chat finished, save messages
    st.session_state.messages = client.messages


tab1, tab2 = st.tabs(["chat", "dalle"])

with tab1:
    chat_app()

with tab2:
    prompt = st.text_input("prompt")
    client = OpenAIClient()
    if st.button("generate"):
        with st.spinner("Generating..."):
            response = client.client_api.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

        image_url = response.data[0].url
        st.image(image_url)
