FROM dustynv/l4t-pytorch:r36.2.0


RUN apt-get update && apt-get upgrade -y && \
 apt-get install -y --no-install-recommends ffmpeg kmod && rm -rf /var/lib/apt/lists/*

COPY ../FaceXHuBERT /root/

RUN pip3 install --upgrade pip && \
 pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "executable" ]
CMD python3 /root/FaceXHuBERT/server.py --subject F1 --condition F3 --emotion 1