FROM nvidia/cuda:11.3.0-devel-ubuntu20.04  as base

ENV NV_CUDA_LIB_VERSION "11.3.0"

FROM base as base-amd6

LABEL maintainer "Yichen Sheng <sheng30@purdue.edu>"

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37
ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    htop \ 
    wget \
	bzip2 \
	xorg-dev \
	cmake \
	libwayland-dev \ 
	libxkbcommon-dev \
	libglu1-mesa \
	libglu1-mesa-dev \
	libgl1-mesa-glx \
	libgl1-mesa-dev \
	wayland-protocols \
	extra-cmake-modules \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    sudo \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN apt update
RUN apt install -y software-properties-common
RUN rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   && mkdir /root/.conda \
   && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
   && rm -f Miniconda3-latest-Linux-x86_64.sh

# setup nerfstudio
ENV PATH /opt/miniconda3/bin:$PATH

WORKDIR /home/root/
RUN conda init 

# setup pixht 
ENV conda_env=pixht
RUN conda create --name $conda_env -y python=3.8 

RUN /bin/bash -c "source activate $conda_env && pip install --upgrade pip pybind11"
RUN /bin/bash -c "source activate $conda_env && python -m pip install --upgrade pip && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113"
RUN /bin/bash -c "source activate $conda_env && pip install omegaconf"

# install data generation code
COPY DBG_GUI /home/root/DBG_GUI

RUN /bin/bash -c "source activate $conda_env && cd /home/root/DBG_GUI/data && python setup.py install"

# default conda activation
RUN echo "source activate $conda_env" >> ~/.bashrc

ENTRYPOINT [  "/bin/bash" ]
