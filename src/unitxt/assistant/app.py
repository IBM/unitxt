import datetime
import json
import logging
import os
import uuid

import litellm
import numpy as np
import pandas as pd
import streamlit as st
import torch
from litellm.llms.watsonx.common_utils import IBMWatsonXMixin
from transformers import AutoTokenizer

logger = logging.getLogger("unitxt-assistance")
original_validate_environment = IBMWatsonXMixin.validate_environment


def wrapped_validate_environment(self, *args, **kwargs):
    kwargs = {**kwargs, "headers": {}}
    return original_validate_environment(self, *args, **kwargs)


IBMWatsonXMixin.validate_environment = wrapped_validate_environment


class Assistant:
    def __init__(
        self,
        assistant_dir=None,
        model_name="watsonx/meta-llama/llama-3-3-70b-instruct",
        tokenizer_name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        embedding_model_name="watsonx/intfloat/multilingual-e5-large",
    ):
        if assistant_dir is None:
            assistant_dir = os.path.dirname(os.path.abspath(__file__))

        self.metadata_df = pd.read_parquet(
            os.path.join(assistant_dir, "metadata.parquet")
        )
        self.embeddings = torch.tensor(
            np.load(os.path.join(assistant_dir, "embeddings.npz"))["embeddings"],
            dtype=torch.float32,
        )

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.embedding_model_name = embedding_model_name

    def search(self, query, max_tokens=5000, min_text_length=50):
        response = litellm.embedding(
            model=self.embedding_model_name,
            input=[query],
        )

        query_embedding = torch.tensor(
            response.data[0]["embedding"], dtype=torch.float32
        )

        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), self.embeddings
        )

        sorted_indices = torch.argsort(similarities, descending=True).numpy()
        total_tokens = 0
        results = {}

        for idx in sorted_indices:
            row = self.metadata_df.iloc[idx]
            path = row["path"]
            if path in results:
                results[path]["count"] += 1
                continue

            text = row["document"]

            if len(text) < min_text_length:
                continue

            token_count = len(self.tokenizer.tokenize(text))

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

    def generate_response(self, messages, max_tokens=5000):
        user_query = messages[-1]["content"]
        search_results = self.search(user_query, max_tokens=max_tokens)

        context = "\n\n".join(
            [f"Path: {v['path']}\nText: {v['text']}" for v in search_results.values()]
        )

        system_prompt = (
            "Your job is to assist users with Unitxt Library and Catalog. "
            "Refuse to do anything else. "
            "Based your answers on the information below add link to its origin Path with this format: https://www.unitxt.ai/en/latest/{path}.html"
            "\n\n"
            "# Answer only based on the following information:\n\n" + context
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in response:
            yield chunk.choices[0].delta.content or ""

    def save_to_disk(self, data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


class App:
    def __init__(self, assistant, history_dir="history", feedback_dir="feedback"):
        self.assistant = assistant
        self.history_dir = history_dir
        self.feedback_dir = feedback_dir

    def set_assistant(self, assistant):
        self.set_assistant = assistant

    def render_sidebar(self):
        with st.sidebar:
            st.title("Assistant Settings")
            self.model = st.selectbox(
                "Model", ["watsonx/meta-llama/llama-3-3-70b-instruct"]
            )
            self.max_tokens = st.number_input(
                "Context Tokens",
                min_value=5000,
                max_value=100000,
                value=10000,
                step=5000,
            )
            st.markdown("---")
            with st.popover("What do you think?"):
                st.markdown("**Feedback**")
                feedback = st.text_area(
                    "Your Feedback", placeholder="Write feedback here..."
                )
                if st.button("Submit Feedback"):
                    if feedback.strip():
                        self.save_feedback_to_disk(feedback)
                        st.success("Feedback submitted. Thank you!")
                    else:
                        st.error("Feedback cannot be empty.")

    def render_chat(self):
        chat_container = st.container()
        user_content = st.chat_input("Ask anything about Unitxt...")
        if user_content:
            st.session_state.pending_user_content = user_content

        if st.session_state.pending_user_content is not None:
            st.session_state.messages.append(
                {"role": "user", "content": st.session_state.pending_user_content}
            )
            self.display_messages(chat_container)
            self.generate_assistant_response(chat_container)
            self.save_messages_to_disk()
            st.session_state.pending_user_content = None
        else:
            self.display_messages(chat_container)

    def set_feedback(self, key, val, content):
        st.session_state[key] = val
        self.save_feedback_to_disk({"type": val, "message": content})

    def render_message_feedback(self, i, content):
        feedback_key = f"feedback-{i}"
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None
        if st.session_state[feedback_key] is None:
            cols = st.columns(12)
            with cols[-2]:
                st.button(
                    "",
                    key=f"like-{i}",
                    icon=":material/thumb_up:",
                    on_click=self.set_feedback,
                    args=(feedback_key, "like", content),
                )
            with cols[-1]:
                st.button(
                    "",
                    key=f"dislike-{i}",
                    icon=":material/thumb_down:",
                    on_click=self.set_feedback,
                    args=(feedback_key, "dislike", content),
                )
        else:
            cols = st.columns(3)
            cols[-1].markdown(
                f"\n**You {st.session_state[feedback_key]}d this message.**"
            )

    def display_messages(self, chat_container):
        with chat_container:
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg["role"] == "assistant":
                        self.render_message_feedback(i, msg["content"])

    def generate_assistant_response(self, chat_container):
        with chat_container:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                # Add CSS spinner while generating response
                placeholder.markdown(
                    """
                    <style>
                    @keyframes fade {
                        0% { opacity: 0; }
                        50% { opacity: 1; }
                        100% { opacity: 0; }
                    }
                    .searching span {
                        font-weight: bold;
                        font-size: 16px;
                        animation: fade 1.5s infinite;
                    }
                    .searching span:nth-child(1) {
                        animation-delay: 0s;
                    }
                    .searching span:nth-child(2) {
                        animation-delay: 0.5s;
                    }
                    .searching span:nth-child(3) {
                        animation-delay: 1s;
                    }
                    </style>
                    <div class="searching">
                        Searching<span>.</span><span>.</span><span>.</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            stream = self.assistant.generate_response(
                st.session_state.messages, max_tokens=self.max_tokens
            )
            response = placeholder.write_stream(stream)
            self.render_message_feedback(len(st.session_state.messages), response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    def save_messages_to_disk(self):
        data = {
            "messages": st.session_state.messages,
            "session_id": st.session_state.session_id,
            "timestamp": str(datetime.datetime.now()),
            "config": {"model": self.model, "max_tokens": self.max_tokens},
        }
        os.makedirs(self.history_dir, exist_ok=True)
        file_path = os.path.join(
            self.history_dir, f"{st.session_state.session_id}.json"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def save_feedback_to_disk(self, feedback):
        data = {
            "feedback": feedback,
            "session_id": st.session_state.session_id,
            "timestamp": str(datetime.datetime.now()),
        }
        os.makedirs(self.feedback_dir, exist_ok=True)
        file_path = os.path.join(
            self.feedback_dir, f"{st.session_state.session_id}.json"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def setup_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pending_user_content" not in st.session_state:
            st.session_state.pending_user_content = None
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                key = f"feedback-{i}"
                if key not in st.session_state:
                    st.session_state[key] = None

    def run(self):
        self.setup_session_state()
        self.render_sidebar()
        self.render_chat()


st.set_page_config(
    page_title="Unitxt Assistant",
    page_icon="🦄",
    initial_sidebar_state="collapsed",
)


@st.cache_resource()
def get_assistant():
    return Assistant()


if __name__ == "__main__":
    app = App(get_assistant())
    app.run()
