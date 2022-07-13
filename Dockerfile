FROM python:3.8-slim

WORKDIR /app

ADD . /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "train.py"]
