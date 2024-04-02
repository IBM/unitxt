.PHONY: docs format

# Absolute path to this make file
THIS_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))

# Directory of this file
DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

docs-files:
	cd $(DIR)/docs && sphinx-apidoc -e -f -o . ../src/unitxt

docs-html: docs-files
	@$(MAKE) -C $(DIR)/docs html

clear-docs:
	rm $(DIR)/docs/modules.rst
	rm $(DIR)/docs/unitxt.rst
	rm $(DIR)/docs/unitxt.*.rst
	rm $(DIR)/docs/catalog.*.rst
	rm -r $(DIR)/docs/_build/

docs: docs-html

test-docs: docs clear-docs

format:
	ruff check . --fix
	ruff format .
	codespell --toml pyproject.toml 

pre-commit:
	pre-commit install
	pre-commit run --all-files

# command: make version={new_version} new-version
new-version:
	bash $(DIR)/utils/update_version $(version)

version-tag:
	bash $(DIR)/utils/create_tag_for_new_version

build-docs-server:
	cd $(DIR)/docs/_build/html && python3 -m http.server 8478

docs-server: docs
	trap 'make clear-docs' EXIT; \
	make build-docs-server

profile:
	bash profile/profile.sh

pypi:
	python setup.py sdist bdist_wheel
	twine upload dist/*

dataset:
	bash $(DIR)/utils/hf/prepare_dataset_imports.sh
	python $(DIR)/utils/hf/prepare_dataset.py

metric:
	bash $(DIR)/utils/hf/prepare_metric_imports.sh
	python $(DIR)/utils/hf/prepare_metric.py

build:
	format
	pypi

# command: make tag_name=${TAG_NAME} metric-service-build
# example: make tag_name=unitxt-service-metric:b1v0.1 metric-service-build
# Use the unitxt dir as the build context for docker, so the entire codebase
# can be copied into the image. This way the latest code changes are integrated into
# the image, without requiring a formal unitxt release.
metric-service-build:
	cd $(DIR) && docker build --tag $(tag_name) --file $(DIR)/src/unitxt/service/metrics/Dockerfile .

# command: make tag_name=${TAG_NAME} metric-service-run-bash
# example: make tag_name=unitxt-service-metric:b1v0.1 metric-service-run-bash
metric-service-run-bash:
	docker run -it $(tag_name) /bin/bash

# command: make tag_name=${TAG_NAME} metric-service-run
# example: make tag_name=unitxt-service-metric:b1v0.1 metric-service-run
metric-service-run:
	docker run -p 8000:8000 --memory=20g $(tag_name)
