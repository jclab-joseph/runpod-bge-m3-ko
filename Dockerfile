FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python3 download_model.py

COPY server.py .
COPY handler.py .

CMD ["python3", "server.py"]

EXPOSE 8000
