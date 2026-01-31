# base
FROM python:3.11-slim

WORKDIR /app

ARG REQUIREMENTS=requirements.txt
ARG INSTALL_TORCH=false

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV REQUIREMENTS=${REQUIREMENTS}
ENV INSTALL_TORCH=${INSTALL_TORCH}

# system deps that help build common Python packages used by ML stacks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libsndfile1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# copy requirement files first so Docker layer can cache pip installs when requirements don't change
COPY requirements*.txt ./

# upgrade pip and install torch (conditionally) then the rest of requirements
# Note: we explicitly install a CPU wheel for torch from PyTorch index when requested.
RUN pip install --upgrade pip setuptools wheel && \
    if [ "${INSTALL_TORCH}" = "true" ]; then \
      pip install --no-cache-dir torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html ; \
    fi && \
    pip install --no-cache-dir -r ${REQUIREMENTS}

# Copy the project
COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["bash"]

