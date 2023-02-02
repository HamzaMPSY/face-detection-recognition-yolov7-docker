FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y python3-opencv\
    libsm6 \
    libxext6 \
    libxrender-dev

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "detect.py", "--source", "http://192.168.11.101:8080/video" ]