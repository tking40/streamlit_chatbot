import os
import streamlit as st
from datetime import datetime

from chat_client import LiteLLMClient

CHAT_HISTORY_DIR = "./chat_history"


def reset():
    if "client" in st.session_state:
        st.session_state.client.reset()
    else:
        st.session_state.client = LiteLLMClient()
    st.session_state.loaded_file = ""


if "client" not in st.session_state:
    reset()
st.set_page_config(layout="wide")


st.title("chatting")


def generate_log_name(client):
    prompt = "Please summarize our conversation so that it fits within a short filename. For example, if we talked about turtle migration, use this format: 'turtle_migration', with no special characters"

    temp_client = client.copy()
    temp_client.prompt(prompt)

    return temp_client.get_response()


def chat_sidebar():
    client = st.session_state.client

    if st.sidebar.button("New Chat"):
        client.reset()
        reset()

    popular_options = {
        "sonnet3.7": ("anthropic", "claude-3-7-sonnet-latest"),
        "4o-mini": ("openai", "gpt-4o-mini"),
    }
    options = popular_options.keys()
    selection = st.sidebar.segmented_control(
        "Popular Choices", options, selection_mode="single"
    )

    if selection:
        provider_name, model_name = popular_options[selection]
    else:
        provider_name = st.sidebar.selectbox(
            "Choose provider", client.provider_models.keys(), key="provider"
        )

        model_name = st.sidebar.selectbox(
            "Choose model:", client.provider_models[provider_name], key="model"
        )
    client.set_model(provider_name=provider_name, model_name=model_name)

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

        if st.button("Save"):
            filename = name if name.endswith(".json") else f"{ymd}_{name}.json"
            st.session_state.loaded_file = filename
            filepath = os.path.join(CHAT_HISTORY_DIR, filename)
            client.save_to_file(filepath)
            st.success(f"Chat history saved to: {filepath}")

    st.sidebar.write("Current tokens", client.count_tokens())

    with st.sidebar.popover("Usage"):
        st.markdown(
            """
        [Anthropic Usage](https://console.anthropic.com/settings/plans)\n
        [OpenAI Usage](https://platform.openai.com/usage)\n
        [Google is free rn](https://ai.google.dev/pricing)\n
        [Groq, also has free tier](https://console.groq.com/settings/usage)
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


tab1, tab2 = st.tabs(["chat", "dalle"])

with tab1:
    chat_app()

with tab2:
    prompt = st.text_input("prompt")
    if st.button("generate image"):
        with st.spinner("Generating..."):
            response = st.session_state.client.client_api.image_generation(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

        image_url = response.data[0].url
        st.image(image_url)
