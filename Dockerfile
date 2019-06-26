#use latest pytorch image
FROM pytorch/pytorch

ENV DEBIAN_FRONTEND noninteractive

LABEL maintainer="landon_chambers@dell.com"

#install jdk 1.8 on ubuntu and install MineRL
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get install -y openjdk-8-jdk && \
    pip install minerl && \
    pip install -U matplotlib


#install xorg and xvfb for rendering in headless server, install x11vnc to view rendering.
RUN apt-get install -y xorg openbox && \
    apt-get install -y xvfb && \
    apt-get install -y git x11vnc

#Set Environment Variables
ENV DISPLAY=:20
ENV MINERL_DATA_ROOT="/workspace/data"

#Expose port 1337
EXPOSE 1337