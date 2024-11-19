FROM python:3.11
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY requirements.txt requirements.txt
COPY chat /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN cd chat

RUN mkdir -p /nltk_data && chmod -R 777 /nltk_data

EXPOSE 8000
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]