import glob
import json
import os
from functools import lru_cache

import litellm
import streamlit as st


def combine_files_into_string(glob_pattern):
    combined_content = []

    for file_path in glob.glob(glob_pattern):
        with open(file_path, encoding="utf-8") as file:
            combined_content.append("# " + file_path)
            combined_content.append(file.read())

    return "\n".join(combined_content)


def read_catalog_files():
    combined_content = []

    for file_path in glob.glob("src/unitxt/catalog/**/*.json"):
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
        id = (
            file_path.replace("src/unitxt/catalog/", "")
            .replace(".json", "")
            .replace("/", ".")
        )
        item = f'id: {id}, type: {data["__type__"]}'
        if "__description__" in data:
            item += f', desc: {data["__description__"]}'
        combined_content.append(item)

    return "\n".join(combined_content)


def make_context():
    context = "\n# Tutorials: \n"
    context += combine_files_into_string("docs/*.rst")
    context += combine_files_into_string("docs/docs/*.rst")
    context += "\n# Examples: \n"
    context += combine_files_into_string("examples/*.py")
    context += "\n# Catalog: \n"
    context += read_catalog_files()
    return context


@lru_cache
def get_context():
    context_file_path = os.path.join(os.path.dirname(__file__), "context.txt")
    if not os.path.exists(context_file_path):
        context = make_context()
        with open(context_file_path, "w") as f:
            f.write(context)
    else:
        with open(context_file_path) as f:
            context = f.read()
    return context


context = get_context()

st.set_page_config(
    page_title="Unitxt Assistant", page_icon="🦄", initial_sidebar_state="collapsed"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_user_content" not in st.session_state:
    st.session_state.pending_user_content = None


def generate_response(messages, model, max_tokens=500):
    messages = [
        {
            "role": "system",
            "content": "Your job is to assist users with Unitxt Library and Catalog. Refuse to do anything else.\n\n # Answer only based on the following Information:\n\n"
            + context,
        },
        *messages,
    ]
    response = litellm.completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in response:
        yield chunk.choices[0].delta.content or ""


with st.sidebar:
    st.title("Assistant")
    model = st.selectbox("Model", ["watsonx/meta-llama/llama-3-3-70b-instruct"])
    max_tokens = st.number_input(
        "Max Tokens", min_value=1, max_value=10000, value=500, step=50, format="%d"
    )

chat_container = st.container()
user_content = st.chat_input("Ask anything about Unitxt...")

if user_content:
    st.session_state.pending_user_content = user_content

if st.session_state.pending_user_content is not None:
    st.session_state.messages.append(
        {"role": "user", "content": st.session_state.pending_user_content}
    )

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    with chat_container:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(
                """
                <style>
                .dots {
                    display: inline-block;
                }
                .dots span {
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    margin: 0 2px;
                    background-color: #333;
                    border-radius: 50%;
                    animation: dots 1.4s infinite ease-in-out both;
                }
                .dots span:nth-child(1) {
                    animation-delay: -0.32s;
                }
                .dots span:nth-child(2) {
                    animation-delay: -0.16s;
                }
                .dots span:nth-child(3) {
                }
                @keyframes dots {
                    0%, 80%, 100% {
                        transform: scale(0);
                    }
                    40% {
                        transform: scale(1);
                    }
                }
                </style>
                <div class="dots">
                    <span></span><span></span><span></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    stream = generate_response(
        st.session_state.messages, model=model, max_tokens=max_tokens
    )

    response = placeholder.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.session_state.pending_user_content = None

else:
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
