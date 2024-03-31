from openai import OpenAI
import os
import streamlit as st
import json
from datetime import datetime
import tiktoken

from dotenv import load_dotenv

from chat_client import AnthropicClient, OpenAIClient

load_dotenv()


clients = {
    "Open AI": OpenAIClient,
    "Claude": AnthropicClient,
}

CHAT_HISTORY_DIR = "./chat_history"


def reset():
    st.session_state.messages = []
    st.session_state.loaded_file = ""


if "messages" not in st.session_state:
    reset()
    st.session_state.max_tokens = 1024
st.set_page_config(layout="wide")


st.title("chatting")


def num_tokens_from_messages(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 4
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def generate_log_name():
    prompt = "Please summarize our conversation so that it fits within a short filename. For example, if we talked about turtle migration, use this format: 'turtle_migration'"
    if not st.session_state.messages:
        return "empty"

    temp_client = OpenAIClient(messages=st.session_state.messages.copy())
    temp_client.model = "gpt-3.5-turbo"
    temp_client.prompt(prompt)

    return temp_client.get_response()


def get_client():
    client = clients[st.session_state.client_name]
    return client(messages=st.session_state.messages)


def chat_sidebar():
    if st.sidebar.button("New Chat"):
        reset()

    st.sidebar.selectbox("Choose client", clients.keys(), key="client_name")

    client = get_client()

    client.model = st.sidebar.selectbox("Choose model:", client.models, key="model")

    with st.sidebar.expander("Advanced Options"):
        client.max_tokens = st.slider(
            "Max response tokens:",
            min_value=128,
            max_value=1024,
            step=64,
            key="max_tokens",
        )

    with st.sidebar.popover("Load from file"):
        st.markdown("load from")
        filenames = [""] + os.listdir(CHAT_HISTORY_DIR)
        filename = st.selectbox("Select a file", filenames)
        if filename:
            filepath = os.path.join(CHAT_HISTORY_DIR, filename)
            st.session_state.loaded_file = filename
            if st.button("Load"):
                client.load_from_file(filepath)

    with st.sidebar.popover("Save chat to file"):
        st.markdown("name")
        if st.button("generate"):
            st.session_state.loaded_file = generate_log_name()

        name = st.text_input("(optional) filename", value=st.session_state.loaded_file)

        ymd = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        if st.button(f"Save"):
            if not ".json" in filename:
                filename = f"{ymd}_{name}.json"
            st.session_state.loaded_file = filename
            filepath = os.path.join(CHAT_HISTORY_DIR, filename)
            client.save_to_file(filepath)
            st.write(f"saved to: {filepath}")

    st.sidebar.write("Current tokens", num_tokens_from_messages(client.messages))

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
    if st.button("generate image"):
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
