#Deriving the tensorflow with gpu
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
ARG DOCKER_USER=claramingyue

# Updating the CUDA Linux GPG Repository Key
# RUN apt-key del 7fa2af80
# RUN apt-get install wget
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*
# RUN sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*

#Install required software and libraries
RUN apt-get update -y
RUN apt install -y software-properties-common

#This is always needed from opencv
RUN apt-get install libgl1 -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y

# Define working direcrtory 
RUN addgroup --system $DOCKER_USER && adduser --system $DOCKER_USER --ingroup $DOCKER_USER
WORKDIR /home/claramingyue

# Virtual environment
ENV VIRTUAL_ENV=/opt/venv
# RUN apt-get install -y python3.10 
# RUN apt-get install -y python3.10-distutils
# RUN apt-get install -y python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install virtualenv
RUN virtualenv --python python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt /home/claramingyue/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

USER claramingyue