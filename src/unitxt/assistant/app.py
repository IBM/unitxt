import datetime
import json
import os
import uuid

import litellm
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer


@st.cache_resource
def load_data():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    metadata_df = pd.read_parquet(os.path.join(current_file_dir, "metadata.parquet"))
    embeddings = np.load(os.path.join(current_file_dir, "embeddings.npz"))["embeddings"]
    return metadata_df, embeddings


def search(query, metadata_df, embeddings, max_tokens=5000, min_text_length=50):
    # Generate embedding for the query using litellm
    response = litellm.embedding(
        model="watsonx/intfloat/multilingual-e5-large",
        input=[query],
    )

    query_embedding = torch.tensor(response.data[0]["embedding"], dtype=torch.float32)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Compute cosine similarity
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0), embeddings_tensor
    )

    # Sort indices by similarity
    sorted_indices = torch.argsort(similarities, descending=True).numpy()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    )

    # Collect results until max_tokens is reached
    total_tokens = 0
    results = {}

    for idx in sorted_indices:
        row = metadata_df.iloc[idx]
        path = row["path"]
        if path in results:
            results[path]["count"] += 1
            continue

        text = row["document"]

        if len(text) < min_text_length:
            continue

        token_count = len(tokenizer.tokenize(text))

        if total_tokens + token_count > max_tokens:
            break

        total_tokens += token_count

        results[row["path"]] = {
            "count": 1,
            "text": text,
            "path": row["path"],
            "similarity": similarities[idx].item(),
        }

    return results


def generate_response(messages, metadata_df, embeddings, model, max_tokens=500):
    user_query = messages[-1]["content"]  # Use the latest user message as the query
    search_results = search(user_query, metadata_df, embeddings, max_tokens=5000)

    # Combine top results as context
    context = "\n\n".join(
        [f"Path: {v['path']}\nText: {v['text']}" for v in search_results.values()]
    )

    system_prompt = (
        "Your job is to assist users with Unitxt Library and Catalog. "
        "Refuse to do anything else.\n\n"
        "# Answer only based on the following Information:\n\n" + context
    )

    messages = [
        {"role": "system", "content": system_prompt},
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


def save_messages_to_disk(
    messages, session_id, model, max_tokens, output_dir="history"
):
    data = {
        "messages": messages,
        "session_id": session_id,
        "timestamp": str(datetime.datetime.now()),
        "config": {
            "model": model,
            "max_tokens": max_tokens,
        },
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{session_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_feedback_to_disk(feedback, session_id, output_dir="feedback"):
    data = {
        "feedback": feedback,
        "session_id": session_id,
        "timestamp": str(datetime.datetime.now()),
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{session_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


st.set_page_config(
    page_title="Unitxt Assistant", page_icon="🦄", initial_sidebar_state="collapsed"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_user_content" not in st.session_state:
    st.session_state.pending_user_content = None

metadata_df, embeddings = load_data()

with st.sidebar:
    st.title("Assistant")
    model = st.selectbox("Model", ["watsonx/meta-llama/llama-3-3-70b-instruct"])
    max_tokens = st.number_input(
        "Max Tokens", min_value=1, max_value=10000, value=500, step=50, format="%d"
    )

    with st.popover("What do you think?"):
        st.markdown("We value your feedback!")
        with st.form("feedback_form"):
            feedback = st.text_area(
                "Your Feedback", placeholder="Write your feedback here..."
            )

            submitted = st.form_submit_button("Submit Feedback")

            if submitted:
                if feedback.strip() == "":
                    st.error("Feedback cannot be empty. Please provide your feedback.")
                else:
                    save_feedback_to_disk(
                        feedback=feedback,
                        session_id=st.session_state.session_id,
                        output_dir="feedback",
                    )
                    st.success("Thank you for your feedback!")

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
        st.session_state.messages,
        metadata_df,
        embeddings,
        model=model,
        max_tokens=max_tokens,
    )

    response = placeholder.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

    save_messages_to_disk(
        st.session_state.messages,
        session_id=st.session_state.session_id,
        model=model,
        max_tokens=max_tokens,
        output_dir="history",
    )

    st.session_state.pending_user_content = None
else:
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
