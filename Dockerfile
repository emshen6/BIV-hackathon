FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y \
    git \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

COPY . /copium

WORKDIR /copium

CMD ["python", "run.py"]
