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
    cmake \
    curl \
    ffmpeg \
    git \
    htop \ 
    wget \
    colmap \ 
    tmux \ 
    xorg-dev \
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
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python3.8 \
    python3-pip \
    python3.8-distutils \ 
    python3.8-dev \
    python-is-python3 \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

# RUN apt-get update && \
#     apt-get install -y software-properties-common

# RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
# 	apt-get update && \ 
# 	apt-get install libstdc++6 -y


RUN apt update
RUN apt install -y software-properties-common
RUN rm -rf /var/lib/apt/lists/*

# colmap seems to have a bug in the docker image, a workaround is this: 
# https://stackoverflow.com/questions/63627955/cant-load-shared-library-libqt5core-so-5 
RUN strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5
RUN ldconfig

WORKDIR /home/root/

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository universe \
    && add-apt-repository multiverse
RUN apt update
RUN rm -rf /var/lib/apt/lists/*

# Set up symlinks for python and pip
RUN /usr/bin/python --version 
RUN ls /usr/bin/python*
RUN rm /usr/bin/python && \
    rm /usr/bin/pip && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN python3.8 -m pip install --upgrade pip pybind11

# install data generation code
COPY DBG_GUI /home/root/DBG_GUI
RUN /bin/bash -c "cd /home/root/DBG_GUI/data && python3.8 setup.py install"

RUN python3.8 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3.8 -m pip install pandas matplotlib tqdm
RUN python3.8 -m pip install opencv-python

RUN python3.8 -m pip install h5py pyyaml dominate
RUN python3.8 -m pip install parse

RUN python3.8 -m pip uninstall Pillow -y && python3.8 -m pip install Pillow==9.5.0
RUN python3.8 -m pip install tensorboard
RUN python3.8 -m pip install scipy 

# imagen
RUN git clone https://github.com/holoviz-topics/imagen.git /home/root/imagen
RUN python3.8 -m pip install imagen
RUN cd /home/root/imagen && python3.8 setup.py install --force

ENTRYPOINT [  "/bin/bash" ]
