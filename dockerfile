# Use an official Python runtime as a parent image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHON_VERSION=3.12.0
ENV PYTHON_ROOT=/usr/local/python-$PYTHON_VERSION
ENV PATH=$PYTHON_ROOT/bin:$PATH

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.12
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz \
    && tar xzf Python-$PYTHON_VERSION.tgz \
    && cd Python-$PYTHON_VERSION \
    && ./configure --enable-optimizations --prefix=$PYTHON_ROOT \
    && make altinstall \
    && cd .. \
    && rm -rf Python-$PYTHON_VERSION Python-$PYTHON_VERSION.tgz

# Set Python 3.12 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 $PYTHON_ROOT/bin/python3.12 1 \
    && update-alternatives --set python3 $PYTHON_ROOT/bin/python3.12 \
    && update-alternatives --install /usr/bin/pip3 pip3 $PYTHON_ROOT/bin/pip3.12 1 \
    && update-alternatives --set pip3 $PYTHON_ROOT/bin/pip3.12



# Verify Python installation
RUN python3 --version && pip3 --version

# Install PyTorch first
RUN pip3 install --upgrade pip
RUN pip3 install torch

# Install project dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy project files
COPY . .

# Expose the port that the app will run on
EXPOSE 5000

# Run the application
CMD ["python3", "app.py"]