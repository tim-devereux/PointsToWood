FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    vim \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH=$CONDA_DIR/bin:$PATH

# Create Conda environment with specific Python version
RUN conda create -n myenv python=3.10 mamba -c conda-forge -y

# Set up shell to use the conda environment by default
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate myenv" >> ~/.bashrc

# Install PyTorch and related packages
RUN conda run -n myenv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    conda run -n myenv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html && \
    conda run -n myenv pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html && \
    conda run -n myenv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html && \
    conda run -n myenv pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html && \
    conda run -n myenv pip install torch-geometric && \
    conda run -n myenv pip install pandas pykdtree numba jupyter matplotlib scikit-learn

# Set the working directory
WORKDIR /workspace

# Set up entry script that activates conda environment
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate myenv\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]