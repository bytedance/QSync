# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ARG PROJECT_DIR=/qsync_niti_based

ADD . $PROJECT_DIR
WORKDIR $PROJECT_DIR
ENV HOME $PROJECT_DIR
# change data path and number of trainers here
ENV PYTHONPATH $PROJECT_DIR
# UPDATE for NVIDIA
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    wget numactl libnuma-dev build-essential libc6-dev iputils-ping
RUN pip3 install --upgrade pip
RUN pip3 install bokeh tensorboard pandas jupyterlab scikit-learn pulp ninja cython pycocotools matplotlib pynvml seaborn \
    mxnet tensorflow==2.9.0 tensorflow-probability onnx-tf byteps tensorboardX accelerate datasets networkx ujson xlsxwriter scapy intervaltree \
    cvxpy transformers
# RUN apt-get update &&
#   apt-get -y upgrade
# # use for download
# RUN apt install wget

CMD echo "===========END========="
CMD /bin/bash
