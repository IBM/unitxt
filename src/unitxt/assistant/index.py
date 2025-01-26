import os
import pickle
import re

import numpy as np
import pandas as pd
from docutils import nodes
from litellm import embedding
from tqdm import tqdm
from transformers import AutoTokenizer


# Function to recursively extract text content from nodes
def extract_text(node):
    if isinstance(node, nodes.Text):
        return node.astext()
    if isinstance(node, nodes.Element):
        return "".join(extract_text(child) for child in node.children)
    return ""


# Function to extract text content from a single doctree file
def process_doctree_file(file_path):
    with open(file_path, "rb") as file:
        doctree = pickle.load(file)
    return extract_text(doctree)


# Function to remove HTML tags and their content
def clean_html(text):
    return re.sub(r"<[^>]*?>.*?</[^>]*?>", "", text, flags=re.DOTALL)


# Tokenizer for chunking content
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")


# Function to split text into chunks
def split_into_chunks(text, max_tokens=500, stride=250):
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)

        # Move the start by the stride
        if end == len(tokens):
            break
        start += stride

    return chunks


# Directory where Sphinx `.doctree` files are located
doctree_dir = "docs/_build/doctrees"

# Path to save the chunked JSON file
output_json_path = "extracted_website_content_chunks.json"

# Dictionary to hold extracted and chunked content
chunked_content = {}

# Total chunk counter
all_chunks = []

# Process all `.doctree` files recursively
for root, _, files in os.walk(doctree_dir):  # Recursively traverse all subdirectories
    for filename in files:
        if filename.endswith(".doctree"):
            file_path = os.path.join(root, filename)
            content = process_doctree_file(file_path)

            # Clean HTML tags and their content from the text
            cleaned_content = (
                clean_html(content)
                .split("Explanation about")[0]
                .replace("Read more about catalog usage here.", "")
                .replace("To use this tutorial, you need to install unitxt.", "")
            )

            # Create the relative path for the JSON key
            relative_path = os.path.relpath(file_path, doctree_dir).replace(
                ".doctree", ""
            )

            # Split content into chunks and add to the dictionary
            chunked_content[relative_path] = []
            for i, chunk in enumerate(split_into_chunks(cleaned_content)):
                all_chunks.append(
                    {
                        "document": cleaned_content,
                        "text": chunk,
                        "path": relative_path,
                        "chunk_index": i,
                    }
                )


# Assuming `all_chunks` is a list of chunks containing text entries
batch_size = 50

total_batches = len(all_chunks) // batch_size + (
    1 if len(all_chunks) % batch_size else 0
)

for i in tqdm(
    range(0, len(all_chunks), batch_size),
    desc="Processing batches",
    total=total_batches,
):
    batch = all_chunks[i : i + batch_size]

    # Prepare inputs for the batch
    inputs = [chunk["text"] for chunk in batch]

    # Call the embedding model for the batch
    response = embedding(
        model="watsonx/intfloat/multilingual-e5-large",
        input=inputs,
    )

    # Update each chunk in the batch with its corresponding embedding
    for chunk, embedding_data in zip(batch, response.data):
        chunk.update(
            {
                "embedding": embedding_data["embedding"],
                "embedding_model": "watsonx/intfloat/multilingual-e5-large",
            }
        )

# Separate metadata and embeddings
metadata = []
embeddings = []

for chunk in all_chunks:
    metadata.append({key: value for key, value in chunk.items() if key != "embedding"})
    embeddings.append(chunk.get("embedding", []))

current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Save metadata as a Parquet file
metadata_df = pd.DataFrame(metadata)
metadata_df.to_parquet(
    os.path.join(current_file_dir, "metadata.parquet"), compression="gzip"
)

# Save embeddings as a compressed .npz file
embeddings_array = np.array(embeddings, dtype=np.float32)
np.savez_compressed(
    os.path.join(current_file_dir, "embeddings.npz"), embeddings=embeddings_array
)
