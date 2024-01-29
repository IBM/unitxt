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
	bash $(DIR)/utils/format

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
