FROM python:3.6

COPY . /PFLD-pytorch
WORKDIR /PFLD-pytorch

RUN apt update
RUN apt install -y python3-tk
RUN pip3 install -r requirements.txt
RUN apt install -y libgl1-mesa-dev