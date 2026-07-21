FROM python:3.12-slim

# Install system dependencies
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Install the package normally
RUN pip install --no-cache-dir .

# Register CUDA libraries installed by tensorflow[and-cuda].
RUN set -eux; \
    SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    find "$SITE_PACKAGES/nvidia" -type d -name lib -print \
        | sort > /etc/ld.so.conf.d/python-nvidia.conf; \
    test -s /etc/ld.so.conf.d/python-nvidia.conf; \
    ldconfig

