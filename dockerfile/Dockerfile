# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ENV HOME /root
ENV PROJECT_DIR /qsync
ENV PYTHONPATH $PROJECT_DIR
WORKDIR $PROJECT_DIR

# UPDATE for NVIDIA
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
# Necessary packages
RUN apt-get update && apt-get install -y \
    wget numactl libnuma-dev build-essential libc6-dev iputils-ping nano tmux libcudnn8 libcudnn8-dev libnccl2 libnccl-dev
# update pip
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install pandas
RUN python3 -m pip install torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# need to reinstalled to modification
RUN python3 -m pip install transformers==4.22.0
RUN python3 -m pip install tqdm pandas scikit-learn pulp ninja cython pynvml accelerate datasets tensorboard tensorboardX 
RUN python3 -m pip install regex tokenizers==0.12.0 accelerate
# to support tensorboard (may not needed?)
RUN python3 -m pip install tensorflow

# install ploting
# RUN python3 -m pip install bokeh tensorboard pandas jupyterlab scikit-learn
# RUN pip3 install bokeh tensorboard pandas jupyterlab scikit-learn pulp ninja cython pycocotools matplotlib pynvml seaborn \
#     mxnet tensorflow==2.9.0 tensorflow-probability onnx-tf tensorboardX accelerate datasets networkx ujson xlsxwriter scapy intervaltree \
#     cvxpy transformers
# RUN apt-get update &&
#   apt-get -y upgrade
# # use for download
# RUN apt install wget
WORKDIR $PROJECT_DIR




CMD echo "===========END========="
CMD /bin/bash
