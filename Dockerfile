FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN apt update
RUN apt install -y git less vim unzip

# install python packages
RUN pip install kornia
RUN conda install -y opencv
RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
RUN pip install neovim

# Download Pointnet2_PyTorch
WORKDIR /root
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
RUN pip install Pointnet2_PyTorch/pointnet2_ops_lib/.
WORKDIR /workspace
COPY ./ /workspace/

# neovim
RUN apt -y install neovim

# environment encoding variables
# https://stackoverflow.com/questions/55646024/writing-accented-characters-from-user-input-to-a-text-file-python-3-7
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8