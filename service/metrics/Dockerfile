FROM registry.access.redhat.com/ubi8/ubi:latest

# Disable Red Hat Subscripstion
RUN sed -i 's/1/0/g' /etc/yum/pluginconf.d/subscription-manager.conf

# Set up conda env var
ENV CONDA_HOME=/opt/conda \
    PATH=/opt/conda/bin:/usr/local/nvidia/bin:$PATH \
    CONFIG_DIR=/root/config_dir \
    BUILD_DIR=/tmp/wml_report \
    #IBM_POWERAI_LICENSE_ACCEPT=yes \
    USE_TF=0 \
    USE_TORCH=1

USER root

RUN yum -y update --allowerasing --nobest && yum -y upgrade --allowerasing --nobest && \
    yum -y install --allowerasing --nobest \
        wget \
        bzip2 \
        git \
        unzip && \
    yum clean all

# Install Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    chmod 755 ./Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    /bin/bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -p ${CONDA_HOME} && \
    rm -rf ./Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    echo >>"${CONDA_HOME}/conda-meta/pinned" "conda=4.8" && \
    conda config --system --set always_yes True && \
    conda config --system --set auto_update_conda False && \
    conda config --system --set notify_outdated_conda False && \
    conda install python~=3.8.0 && \
    conda clean -a

# Install pkgs from OPEN-CE and conda-forge
RUN conda config --prepend channels https://ftp.osuosl.org/pub/open-ce/current/
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64/:/opt/conda/lib/
# Fix versions for deployment on early access cluster
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Non-root user
ENV NONROOT_USER=gpuuser \
    NONROOT_UID=1000 \
    NONROOT_GID=1000 \
    NONROOT_HOME=/home/gpuuser \
    PATH="/home/gpuuser/.local/bin:${PATH}"

RUN groupadd -g ${NONROOT_GID} ${NONROOT_USER} && \
    useradd -u ${NONROOT_UID} -g ${NONROOT_GID} -G users -m -c "" -e "" -l -s /bin/bash ${NONROOT_USER} && \
    mkdir -p /var/run/sshd && mkdir -p ${NONROOT_HOME}/.ssh && \
    chown ${NONROOT_USER} ${NONROOT_HOME}/.ssh && \
    chmod -R 700 ${NONROOT_HOME}

# Update pip
RUN pip install --upgrade pip

ARG IMAGE_TAG=undefined
RUN mkdir -p /usr/local/bin/

USER ${NONROOT_USER}

# Copy unitxt into the image
COPY --chown=${NONROOT_USER}:${NONROOT_GID} . /app/unitxt/.

# Install the metrics service requirements
RUN cat /app/unitxt/service/metrics/requirements.txt
RUN pip3 install -r /app/unitxt/service/metrics/requirements.txt
#RUN pip3 freeze | tee ./pip_freeze_after_requirements.log

# install problematic packages
RUN yes | pip3 install -U pyyaml joblib

# pip install unitxt
WORKDIR /app/unitxt
RUN pip install -e ".[all]"

WORKDIR /app/unitxt/service/metrics
ENV HF_HOME=/app/hf/misc
ENV HF_DATASETS_CACHE=/app/hf/datasets
ENV TRANSFORMERS_CACHE=/app/hf/models
EXPOSE 8000

ENV PYTHONPATH=/app/unitxt
ENV PYTHONHASHSEED 0

# code obfuscation
# RUN python3 -m compileall .
# RUN echo $(python3 ./prepare_pyc_files_to_replace_py_files.py)
# RUN find . -type f -not -name "*.pyc" -not -path "./models/*" -delete
# RUN ls -R

RUN pip3 install cffi --upgrade

RUN env
RUN pip3 list
RUN conda list
CMD python3 main.py