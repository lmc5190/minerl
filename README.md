# MineRL Competition
[The MineRL Competition!](http://minerl.io/competition/)

[Competition Rules](https://gitlab.aicrowd.com/minerl/minerl-resources/blob/master/Rules.md)
# Development Environment
Landon has been using a Windows VDI to get into the lab environment and then SSHing into a GPU server where he runs code on a docker container.
 
## Servers 
We have two C4130 servers each with
 - 4x P100 GPUs
 - 72 CPU Cores
 - 515 GB of Memory
 - ~1.2 TB of SSD(soon to be > 5 TB)
 
The servers are named GPU1 and GPU2 located at 172.20.5.18 and 172.20.5.19


## Container
Landon has been using a Pytorch docker container to run and develop code!

Example docker container run

`NV_GPU=0 nvidia-docker run --cpus="6" -m="112g" --rm -ti --ipc=host -p 5920:5920 --mount type=bind,src=/home/landon_chambers@SAAS.LOCAL/minerl,dst=/workspace landonchambers/minerl
`

- NV_GPU=0 isolates GPU 0. To isolate two GPUS, you could use NV_GPU=1,2 for example.
- restricted to 6 cpus and 112GB of memory
- port 800 on the docker container will talk to port 4000 on the server.
- I 'bind' a mounted drive on the server to a drive on the docker container. This is needed so I can SSH into the server with my windows VDI and develop in VS Code!
- It is important to note that use of the `--rm` flag will destroy the container upon exit. Any state that one would like persisted after the container exits should accommodate accordingly. 
- The image will be pulled from dockerhub in the landonchambers/minerl repository. Dockerfile [here](https://github.com/lmc5190/minerl/blob/master/Dockerfile)

On GPU1 server, the command is a bit different since there seems to be a different version of nvidia-docker (THIS NEEDS TO BE FIXED!)

`docker run --runtime=nvidia  -e NVIDIA_VISIBLE_DEVICES=0 --cpus="6" -m="112g" --rm -ti --ipc=host -p 5920:5920 \
--mount type=bind,src=/home/landon_chambers@SAAS.LOCAL/minerl,dst=/workspace landonchambers/minerl`

The container has Ubuntu OS 16.04 so remember to use Debian commands!

## Develop from Windows VDI
Landon is doing everything in VS Code. It's great! I have my python code, server terminal, and container terminal all on one screen :)

Here are some high-level instructions for using VS Code with an SSH Tunnel to a server
Follow instructions [here](https://code.visualstudio.com/docs/remote/ssh) and see Landon's notes below.
- Installed Git for Windows for OpenSSH compatible SSH client
- Installed the Remote Development extension pack using the extension installer in VS Code
- Enabled the "show terminal" setting for the Remote Development extension and used ssh password to login to server. 


## View Rendering with VNC
Oh the joys of virtual rendering within a docker container on a headless server.

VNC rendering requires the configuration of three systems. 
- client
- server
- docker container

The client is a machine that renders a desktop for you already. The docker container runs the code that renders your images and is hosted on a server. How to you pass the rendered video from the docker container to your client? Let's get into it!

### Client
This is, for example, a Windows VDI or some VDI outside of the headless server. The strategy is to view the rednered image with a VNC viewer. The steps below will enable this capability.
- Download a VNC Viewer. [Here is the one I used](https://www.realvnc.com/en/connect/download/viewer/)
- Download PuTTY
Peform the following steps *on the client*:
- Start a PuTTY session and input server IP address
- Under Connections > SSH > X11 : Enable X11 forwarding and enter localhost:5920 as the X Display Location

### Server
The server does not allow X11 forwarding natively. You must enable X11 forwarding in the system's ssh config files *on the server*. (Landon already did this on gpu2)
- Sudo open ssh_config. Set Forwardagent and ForwardXll to "yes"
- Sudo open sshd_config. Set AllowAgentForward, AllowTcpForwarding, and X11 Forwarding to "yes"
- If any of the above settings changed, you must reboot the server to apply these settings!

Now create a docker container. The command for "docker-run" will require an important port mapping "5920:5920". This maps 5920 on the server to 5920 on the docker container, allowing for communication through these ports. We set the display environment variable *on the server* before starting the container.

`export DISPLAY=localhost:5920`

```
NV_GPU=0 nvidia-docker run --cpus="6" -m="112g" --rm -ti --ipc=host -p 5920:5920 --mount type=bind,src=/home/landon_chambers@SAAS.LOCAL/minerl,dst=/workspace landonchambers/minerl
```
### Container
The startegy here is to kickoff a virtual monitor on :20 (turns out, :20 is mapped to port 5920 when we are talking about display ports) and point the VNC viewer the server's display on ServerIPAddress:5920.

Do the next few steps *within the container*, which requires dependencies built within the Docker image. See [Dockerfile](https://github.com/lmc5190/minerl/blob/master/Dockerfile).
- Set the display to port 5920. `export DISPLAY=:20`
- Run the following code to kickoff the virtural monitor
```
Xvfb :20 -screen 0 1280x1024x24 &
x11vnc -passwd TestVNC -display :20 -N -forever &
```
- *On the client* run your VNC Client and connect to ServerIPAddress:5920
- Type in the password you set (above we used TestVNC as the pw)
- *In the container* Run your script normally `python yourscript.py`.
You should shortly see Minecraft rendering in your VNC Viewer :)
