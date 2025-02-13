FROM python:3.11

USER root

WORKDIR /app

COPY new_uploader /app

RUN pip install -r requirements.txt

RUN apt-get update -y && apt-get install -y libsndfile1 ffmpeg

# Expose the port your Flask/FastAPI app will listen on
EXPOSE 8080

# Start your Flask/FastAPI app
CMD ["python", "app.py"]