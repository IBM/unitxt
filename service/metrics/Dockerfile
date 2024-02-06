FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" \
    apt install -y git tini\
    openjdk-11-jdk \
    python3 \
    python3-pip \
    python3.8-venv \
    ca-certificates && \
    rm -rf /var/lib/apt/lists

# Install Java, python and other depedencies
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN pip freeze | tee pip_freeze.log

# Setup and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
## This automatically enables virtualenv for both, Docker RUN and CMD commands
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip

# Install Torch [GPU]
#RUN pip install 'torch~=1.11.0' --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install torch transformers nltk

RUN python -m nltk.downloader punkt

WORKDIR /app

# set the python path
ENV PYTHONPATH=/app

## install requirements
COPY requirements.txt service/metrics/requirements.txt
RUN pip install --no-cache-dir -r service/metrics/requirements.txt
RUN pip freeze | tee pip_freeze_after_metrics_requirements.log

## copy the code bases
COPY . service/metrics/.

# clean ups and patches for security reasons
# RUN pip install -U setuptools
# RUN python -m pip uninstall -y pip
# RUN apt-get -y remove git
# RUN dpkg --remove libcurl3-gnutls

# prepare for run
WORKDIR /app/service/metrics
ENV HF_HOME=/app/hf/misc
ENV HF_DATASETS_CACHE=/app/hf/datasets
ENV TRANSFORMERS_CACHE=/app/hf/models
EXPOSE 8000

CMD python3 main.py
