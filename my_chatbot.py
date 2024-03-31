import os
import streamlit as st
from datetime import datetime

from chat_client import AnthropicClient, OpenAIClient, GoogleClient

clients = {
    "Open AI": OpenAIClient,
    "Claude": AnthropicClient,
    "Google": GoogleClient,
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


def generate_log_name():
    prompt = "Please summarize our conversation so that it fits within a short filename. For example, if we talked about turtle migration, use this format: 'turtle_migration', with no special characters"
    if not st.session_state.messages:
        return "empty"

    temp_client = GoogleClient(messages=st.session_state.messages.copy())
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

    model_name = st.sidebar.selectbox("Choose model:", client.models, key="model")
    client.set_model(model_name)

    with st.sidebar.expander("Advanced Options"):
        client.max_tokens = st.slider(
            "Max response tokens:",
            min_value=128,
            max_value=1024,
            step=64,
            key="max_tokens",
        )

    with st.sidebar.expander("Load from file"):
        st.markdown("load from")
        filenames = [""] + os.listdir(CHAT_HISTORY_DIR)
        filename = st.selectbox("Select a file", filenames)
        if filename:
            filepath = os.path.join(CHAT_HISTORY_DIR, filename)
            st.session_state.loaded_file = filename
            if st.button("Load"):
                client.load_from_file(filepath)

    with st.sidebar.expander("Save chat to file"):
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

    st.sidebar.write("Current tokens", client.count_tokens())

    with st.sidebar.popover("Usage"):
        st.markdown(
            """
        [Anthropic Usage](https://console.anthropic.com/settings/plans)\n
        [OpenAI Usage](https://platform.openai.com/usage)\n
        [Google is free rn](https://ai.google.dev/pricing)
        """
        )

    return client


def chat_app():
    client = chat_sidebar()

    # Needed this to get consistent ordering of the chat elements
    container_A = st.container()
    container_B = st.container()

    # Display chat messages from history on app rerun
    with container_A:
        for message in client.get_messages():
            chat_msg = st.chat_message(message["role"])
            chat_msg.markdown(message["content"])

    # Accept user input
    with container_B:
        prompt = st.chat_input("Type Here")
    if prompt:
        client.prompt(prompt)

        # Display user input
        with container_A:
            chat_msg = st.chat_message("user")
            chat_msg.markdown(prompt)

        # Get and display client response
        with container_A:
            stream = client.stream_generator()
            chat_msg = st.chat_message("assistant")
            response = chat_msg.write_stream(stream)

        client.add_message(role=client.assistant_role, content=response)

    # Chat finished, save messages
    # note: client should be writing to same message list, but observed odd behavior without this
    st.session_state.messages = client.get_messages()


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
