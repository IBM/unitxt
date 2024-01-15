FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt update
RUN apt upgrade -y

# Ubuntu 22.04 needs the additional ppa for finding python 3.11 that is not RC.
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa

RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" \
    apt install -y git tini\
    openjdk-11-jdk \
    python3.11 \
    python3-pip \
    python3-venv \
    ca-certificates && \
    rm -rf /var/lib/apt/lists

# Install Java, python and other depedencies
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Setup and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
## This automatically enables virtualenv for both, Docker RUN and CMD commands
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# add a non-root user to be used in run time
ENV NONROOT_USER=user \
    NONROOT_UID=1000 \
    NONROOT_GID=1000
RUN groupadd -g ${NONROOT_GID} ${NONROOT_USER}
RUN useradd -u ${NONROOT_UID} -g ${NONROOT_GID} -G users -m -c "" -e "" -l -s /bin/bash ${NONROOT_USER}

ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# set the python path
ENV PYTHONPATH=/app

## install requirements
COPY service/metrics/requirements.txt service/metrics/requirements.txt
RUN pip install --no-cache-dir -r service/metrics/requirements.txt

RUN python -m nltk.downloader punkt

## copy the code bases
COPY service/metrics/. service/metrics/.

# clean ups and patches for security reasons
RUN pip install -U setuptools
RUN python -m pip uninstall -y pip
RUN apt-get -y remove git
# RUN dpkg --remove libcurl3-gnutls

# make sure we have access as non-root user
RUN chown -R ${NONROOT_USER} .
RUN chgrp -R ${NONROOT_USER} .

# prepare for run
WORKDIR /app/service/metrics
ENV HF_HOME=\my_drive\hf\misc
ENV HF_DATASETS_CACHE=\my_drive\hf\datasets
ENV TRANSFORMERS_CACHE=\my_drive\hf\models
USER ${NONROOT_USER}
EXPOSE 8000

# If running behind a proxy like Nginx or Traefik add --proxy-headers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
