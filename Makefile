.PHONY: docs format

# Absolute path to this make file
THIS_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))

# Directory of this file
DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

docs:
	# generate .rst files for sphinx
	cd $(DIR)/docs && sphinx-apidoc -f -o . ../src/unitxt
	# create the html documentation
	@$(MAKE) -C $(DIR)/docs html

format:
	black $(DIR)/src/unitxt

docs-server:
	cd $(DIR)/docs/_build/html && python3 -m http.server 8478

profile:
	bash profile/profile.sh

publish:
	python setup.py sdist bdist_wheel
	twine upload dist/*
