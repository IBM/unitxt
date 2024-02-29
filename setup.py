import os
import warnings

import setuptools

from version import version

with open("README.md") as fh:
    long_description = fh.read()

requirements_dir = "requirements"
extras_require = {}

# Collecting all requirements from the .rqr files
for file in os.listdir(requirements_dir):
    if file.endswith(".rqr"):
        key = file.rsplit(".")[0]  # Remove .rqr extension to get the key
        with open(os.path.join(requirements_dir, file)) as f:
            requirements_list = []
            for requirement in f.read().splitlines():
                requirement = requirement.strip()
                if requirement.startswith("git+"):
                    warnings.warn(
                        f"****IMPORTANT****\nThe requirement {requirement} of setup '{key}' can be installed directly with 'pip install {requirement}' or 'pip install -r {file}'",
                        stacklevel=2,
                    )
                else:
                    requirements_list.append(requirement)
            extras_require[key] = requirements_list

# Adding the 'all' key
all_requirements = []
for req_list in extras_require.values():
    all_requirements.extend(req_list)
extras_require["all"] = list(set(all_requirements))  # Removing duplicates

setuptools.setup(
    name="unitxt",
    version=version,
    author="IBM Research",
    author_email="elron.bandel@ibm.com",
    description="Load any mixture of text to text data in one line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ibm/unitxt",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"unitxt": ["catalog/*.json", "ui/banner.png"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=extras_require["base"],
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "unitxt-explore=unitxt.ui:launch",
            "unitxt-metrics-service=unitxt.service.metrics.main:start_metrics_http_service",
        ],
    },
)
