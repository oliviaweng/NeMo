FROM pytorch/pytorch

#Install dependencies
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg
RUN apt-get update && apt-get install -y git 
RUN apt-get update && \
    apt-get install -y \
    libsndfile1 sox \
    libfreetype6 \
    python-setuptools swig \
    python-dev ffmpeg
RUN apt-get update && apt-get install vim tmux htop
RUN apt-get update && apt-get install -y gcc g++
# RUN pip3 install nemo_toolkit['all']
RUN pip3 install Cython==0.29.27
RUN pip3 install tqdm
RUN pip3 install sox

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app/
USER appuser

# Install editable NeMo
RUN mkdir /app/data
RUN mkdir /app/NeMo
RUN git clone --branch skip-connections https://github.com/oliviaweng/NeMo.git /app/NeMo
RUN python -m pip install -e /app/NeMo
# RUN python -m pip install git+https://github.com/oliviaweng/NeMo.git@main#egg=nemo_toolkit[all]

CMD ["sleep", "infinity"]