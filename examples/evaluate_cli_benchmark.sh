# Define the inference provider from watsonx, azure, together-ai, openai, and more:
export UNITXT_DEFAULT_PROVIDER="watsonx"

# Set the relevant environments variables for the inference provider:
export WX_API_KEY="..."
export WX_PROJECT_ID="..."
export WX_URL="https://us-south.ml.cloud.ibm.com"

unitxt-evaluate \
    --tasks "benchmarks.bluebench" \
    --model cross_provider \
    --model_args "model_name=llama-3-1-8b-instruct,max_tokens=256" \
    --split validation \
    --limit 10 \
    --output_path ./results/evaluate_cli \
    --log_samples \
    --verbosity INFO \
    --trust_remote_code \
    --apply_chat_template \
    --batch_size 8