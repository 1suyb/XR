From pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
WORKDIR /app
COPY . /app

RUN apt-get update
RUN apt-get install git -y

RUN pip install -r requirements.txt

CMD ["python", "app.py"]