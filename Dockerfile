#use latest pytorch image
FROM pytorch/pytorch

LABEL maintainer="landon_chambers@dell.com"
 
#install jdk 1.8 on ubuntu and install MineRL
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get install -y openjdk-8-jdk && \
    pip install minerl

#install xorg and xvfb for rendering in headless server
RUN apt-get install -y xorg openbox && \
    apt-get install -y xvfb

#Expose port 1337
EXPOSE 1337