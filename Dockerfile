FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /tmp/streamlit

ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV TMPDIR=/tmp

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]