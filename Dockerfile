FROM python:3

COPY requirements.txt ./

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN git clone https://github.com/sharkzeeh/PFLD-pytorch.git && cd ./PFLD-pytorch
RUN pip3 install -r requirements.txt

#https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo

# RUN apt update && \
#     apt install git && \
#     git clone https://github.com/sharkzeeh/yolov3.git && \
#     cd ./yolov3 && \
#     pip3 install -r requirements.txt

# pip install opencv-python
# pip3 uninstall -y thop	