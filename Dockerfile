FROM --platform=linux/arm64/v8 nvcr.io/nvidia/deepstream-l4t:5.0.1-20.09-base

RUN apt update && apt install -y python3-pip && \
    apt install -y gcc libssl1.0.0 \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstrtspserver-1.0-0 \
    libjansson4 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python-gi-dev \
    libpython3.6-dev 
    
RUN pip3 install --upgrade pip

RUN useradd -ms /bin/bash diycam && echo "diycam:password" | chpasswd && usermod -a -G video diycam

USER diycam

WORKDIR /home/diycam/deepstream_rdx_sample

COPY --chown=diycam:diycam requirements.txt requirements.txt

RUN pip3 install --user -r requirements.txt

COPY --chown=diycam:diycam . .

ENV PYTHONPATH=/opt/nvidia/deepstream/deepstream/lib

ENV OPENBLAS_CORETYPE=ARMV8

CMD [ "python3", "deepstream_rdx_sample.py" ]
