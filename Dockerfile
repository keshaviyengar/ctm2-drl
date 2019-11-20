FROM tensorflow/tensorflow:1.12.0-py3

ENV CODE_DIR /home/code
ENV VENV /root/venv

WORKDIR $CODE_DIR

COPY ctm2-envs/ $CODE_DIR/ctm2-envs/

RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev python3-tk libopenmpi-dev python-pip zlib1g-dev cmake libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1

RUN git clone https://github.com/keshaviyengar/baselines.git

RUN	pip3 install -e $CODE_DIR/ctm2-envs/ && \
	pip3 install PyYAML && \
	pip3 install mpi4py && \
	pip3 install -e $CODE_DIR/baselines/

CMD /bin/bash
