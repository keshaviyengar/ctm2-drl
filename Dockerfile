FROM araffin/stable-baselines:latest

ENV CODE_DIR /root/code
ENV VENV /root/venv

COPY ctm2-envs/ $CODE_DIR/ctm2-envs/
COPY stable-baselines/ $CODE_DIR/stable-baselines/
RUN . $VENV/bin/activate && \
	pip install -e $CODE_DIR/ctm2-envs/ && \
	pip install stable-baselines && \
	pip install PyYAML

ENV PATH=$VENV/bin:$PATH
CMD /bin/bash
