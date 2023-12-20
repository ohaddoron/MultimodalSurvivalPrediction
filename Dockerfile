# Use an official Python runtime as a parent image
FROM python:3.9-slim


WORKDIR /opt/MultimodalSurvivalPrediction



# Copy the current directory contents into the container at /usr/src/app
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev \
    musl-dev \
    postgresql-client \
    openssh-server \
    curl \
    wget \
    vim \
    less \
    git \
    zsh \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y \
    && pip install --upgrade pip

# Create SSH directory and authorized_keys file
RUN mkdir -p /root/.ssh && touch /root/.ssh/authorized_keys

# Copy the public key from your host to the Docker image
COPY id_rsa.pub /root/.ssh/id_rsa.pub

# Add the public key to the authorized_keys
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

# Make port available to the world outside this container
EXPOSE 22

# Start ssh service
ENTRYPOINT service ssh restart && bash
