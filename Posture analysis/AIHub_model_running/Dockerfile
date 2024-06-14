FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel                                                                                                                                                                                               
RUN apt-get update && apt-get install -y \
    ca-certificates python3-dev git wget sudo

RUN pip install tensorboard cython
RUN pip install scipy tqdm
RUN pip install opencv-python

WORKDIR /workspace
COPY . .
ENV PYTHONIOENCODING UTF-8
