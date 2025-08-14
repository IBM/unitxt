echo "blis==0" > constraints.txt
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --upgrade --system torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --system -c constraints.txt -e ".[tests]"
pip install --only-binary :all: spacy
pip install coverage[toml]
