recipe="card=cards.wnli,template=templates.classification.multi_class.relation.default"
hf_model="microsoft/phi-1_5"

helm-run \
    --run-entries "unitxt:$recipe,model=$hf_model" \
    --enable-huggingface-models $hf_model \
    --max-eval-instances 10 --suite v1

if [ ! -f "benchmark_output/runs/v1/unitxt:card=cards.wnli,template=templates.classification.multi_class.relation.default,model=microsoft_phi-1_5/scenario_state.json" ]; then
  echo "Error: File does not exist." >&2
  exit 1
fi