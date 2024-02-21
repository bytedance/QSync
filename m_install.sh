DIR_PATH=<abspath_to_root>
export PYTHONPATH=$DIR_PATH
cd $PYTHONPATH
apt-get update && apt-get install -y wget numactl libnuma-dev build-essential libc6-dev iputils-ping nano tmux libcudnn8 libcudnn8-dev libnccl2 libnccl-dev
pip3 install --upgrade pip -i https://pypi.doubanio.com/simple
pip3 install pandas -i https://pypi.doubanio.com/simple
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install tqdm pandas scikit-learn pulp ninja cython pynvml accelerate datasets tensorboard tensorboardX -i https://pypi.doubanio.com/simple
pip3 install regex tokenizers==0.12.1 accelerate -i https://pypi.doubanio.com/simple
pip3 install tensorflow  -i https://pypi.doubanio.com/simple
echo "end"
make