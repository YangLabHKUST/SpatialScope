# Pull Base Image
FROM continuumio/anaconda3

# Maintainer
MAINTAINER Jiashun Xiao (xiaojiashun@sribd.cn)

WORKDIR /home/

RUN conda update -y conda

RUN apt-get update && apt-get install -y git rsync

# Clone the repository from GitHub
RUN git clone https://github.com/YangLabHKUST/SpatialScope.git

RUN cd SpatialScope

WORKDIR /home/SpatialScope

# Create and activate the Conda environment
RUN conda env create -f environment.yml

RUN echo "conda activate SpatialScope" >> ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# fix bug of squidpy, locate the lib with `which python`
RUN rsync ./src/_feature_mixin.py /opt/conda/envs/SpatialScope/lib/python3.9/site-packages/squidpy/im/_feature_mixin.py


# Add entrypoint script
ADD ./run.sh /home/
RUN chmod +x /home/run.sh

CMD /home/run.sh


#docker run  -it --gpus all --ipc=host  -v /home/jxiaoae/spatial/SpatialScope:/home/jxiaoae/spatial/SpatialScope spatialscope /bin/bash