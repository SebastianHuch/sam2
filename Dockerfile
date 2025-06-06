FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev git libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy repo contents to /workspace
WORKDIR /workspace
COPY . .

# Install Python requirements
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install SAM2
RUN pip install -e .

CMD ["python", "sam2_video_auto_track.py", "--help"]
