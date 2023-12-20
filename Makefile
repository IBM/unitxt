.PHONY: docs format

# Absolute path to this make file
THIS_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))

# Directory of this file
DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

docs-files:
	cd $(DIR)/docs && sphinx-apidoc -f -o . ../src/unitxt

docs-html: docs-files
	@$(MAKE) -C $(DIR)/docs html

docs: docs-html
	echo "Docs built successfully!"

test-docs: docs
	rm $(DIR)/docs/unitxt.rst
	rm $(DIR)/docs/unitxt.*.rst

format:
	bash $(DIR)/utils/format

# command: make version={new_version} new-version
new-version:
	bash $(DIR)/utils/update_version $(version)

version-tag:
	bash $(DIR)/utils/create_tag_for_new_version

docs-server: docs
	cd $(DIR)/docs/_build/html && python3 -m http.server 8478

profile:
	bash profile/profile.sh

pypi:
	. $(DIR)/SECRETS.SH
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
