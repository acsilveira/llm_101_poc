FROM quay.io/jupyter/scipy-notebook

COPY src/ /home/jovyan/workspace/src
COPY data/ /home/jovyan/workspace/data
COPY output/ /home/jovyan/workspace/output

WORKDIR /home/jovyan/workspace

RUN pip install -r src/requirements.txt