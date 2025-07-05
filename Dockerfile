# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY "01-embeddings-tf-idf.py" ./

# Set the default command to run the script
CMD ["python", "01-embeddings-tf-idf.py"]
