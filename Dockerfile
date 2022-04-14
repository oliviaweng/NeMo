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
RUN apt-get update && \
    apt-get install -y \
    vim tmux
RUN apt-get update && apt-get install -y gcc g++
# RUN pip3 install nemo_toolkit['all']
RUN pwd; ls
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN python -m pip install -r /app/requirements.txt

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 1000 --disabled-password --gecos "" appuser && chown -R appuser /app/
USER appuser

# Install editable NeMo

CMD [ "bash", "-c", "cd /app/dev-nemo/NeMo; python -m pip install -e .; cd /app; sleep infinity" ]