recipe="card=cards.sst2,template=templates.classification.multi_class.default,num_demos=1,demos_pool_size=10,loader_limit=100"
model="microsoft/phi-1_5"

helm-run \
    --run-entries "unitxt:$recipe,model=$model" \
    --enable-huggingface-models $model \
    --max-eval-instances 1 --suite v1

model=$(echo "$model" | sed 's/\//_/g')
results_file="benchmark_output/runs/v1/unitxt:$recipe,model=$model/stats.json"
if [ ! -f "$results_file" ]; then
  echo "Error: Helm run did not produce the expected rsults file: $results_file." >&2
  exit 1
fi


result=$(python -c "import json; print(next(item['mean'] for item in json.load(open('$results_file')) if item['name']['name'] == 'accuracy'))")

if [[ "$result" == "1.0" ]]; then
    echo "Success: The results are consistent, mean accuracy is $result as expected."
else
    echo "Error: The results are inconsistent, mean accuracy should be 1.0, but got $result."
    exit 1
fi