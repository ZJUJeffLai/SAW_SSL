# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04
ENV REPONAME=${REPONAME}

#########################################
# SECTION 1: Essentials                 #
#########################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim git curl wget yasm cmake unzip pkg-config \
    checkinstall build-essential ca-certificates \
    software-properties-common apt-utils bash-completion \
  && apt-get upgrade -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

#########################################
# SECTION 2: Setup Image Libraries      #
#########################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev libjpeg-dev libpng-dev xvfb ffmpeg xorg-dev \
    xorg-dev libboost-all-dev libsdl2-dev swig \
    libblas-dev liblapack-dev \
    libopenblas-base libatlas-base-dev graphviz \
    libvips \
  && apt-get upgrade -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

#########################################
# SECTION 3: Install Python Libraries   #
#########################################
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python libraries
RUN apt-get update && apt-get install -y \
    --no-install-recommends libvips libvips-dev
RUN apt-get -y upgrade

RUN pip3 install pip --upgrade
RUN pip3 install -U numpy cython ipython \
    jupyter matplotlib pandas scikit-learn \
    scikit-image scipy tqdm pyvips

# This uses CUDA '9.0.176' in PyTorch's view (incompatible w/ 2080 and above)
# RUN pip3 install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html

# This uses CUDA '10.0.130' in PyTorch's view (compatible w/ 2080 and above)
# For more info, in Python3 run "torch.version.cuda" :D
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install progress

# Install randAugment
RUN pip3 install git+https://github.com/ildoonet/pytorch-randaugment

RUN pip3 install -U opencv-python==3.4.5.20

# Install jupyter notebook extensions (code folding)
RUN pip3 install jupyter_contrib_nbextensions \
  && jupyter contrib nbextension install \
  && pip3 install jupyter_nbextensions_configurator \
  && jupyter nbextensions_configurator enable

# Install argcomplete for auto tab completing argparse in Python
RUN pip3 install argcomplete \
  && activate-global-python-argcomplete

######################################
# SECTION 4: Additional Utilities    #
######################################
RUN echo "export PATH=$PATH:~/.local/bin/" >> /root/.bashrc
RUN apt update && apt -y upgrade

WORKDIR /root/${REPONAME}

######################################
# SECTION 5: Add running instruction #
######################################
RUN echo "echo -e \"\n\"" >> /root/.bashrc
RUN echo "echo -e \"################################################################################\n\"" >> /root/.bashrc
RUN echo "echo -e \"\tCommand to enter repository:\n\t\tcd /root/${REPONAME} && jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=3366 &\n\"" >> /root/.bashrc
RUN echo "echo -e \"################################################################################\n\"" >> /root/.bashrc