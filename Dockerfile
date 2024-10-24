ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

# Pull the docker image
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


# Set environment variables and compilation options
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# (Optional, use Mirror to speed up downloads)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//https:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# groupadd -r algorithm
# useradd -m --no-log-init -r -g algorithm algorithm
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN cat /etc/group


# RUN mkdir
# chown algorithm:algorithm
RUN mkdir -p /opt/algorithm /inputs /outputs
RUN chown algorithm:algorithm /opt/algorithm /inputs /outputs


# switch to algorithm user, and set /opt/algorithm as work-dir
USER algorithm
WORKDIR /opt/algorithm


# append environment PATH /home/algorithm/.local/bin
ENV PATH="/home/algorithm/.local/bin:${PATH}"


# RUN python -m pip install --user -U pip
# python -m pip install --user pip-tools
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools



# TODO: install packages
# 用于将requirements.txt文件复制到镜像中，并以algorithm用户的身份安装其中的依赖
COPY --chown=algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt



# TODO: copy your algorithm files
# 用于将算法的模型文件、脚本文件等复制到镜像中，并设置它们的所有权为algorithm用户和组。注释部分提示用户可以复制模型权重文件
COPY --chown=algorithm:algorithm model /opt/algorithm/model
COPY --chown=algorithm:algorithm predict.sh /opt/algorithm/

RUN echo "Clearing cache for the next step"
COPY --chown=algorithm:algorithm run_inference /opt/algorithm/
COPY --chown=algorithm:algorithm dataset.py /opt/algorithm/
COPY --chown=algorithm:algorithm teeth_idx_dict.py /opt/algorithm/
COPY --chown=algorithm:algorithm reserve_largest_region_np.py /opt/algorithm/
COPY --chown=algorithm:algorithm inputs /opt/algorithm/
COPY --chown=algorithm:algorithm checkpoint /opt/algorithm/
COPY --chown=algorithm:algorithm outputs /opt/algorithm/