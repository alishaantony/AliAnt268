# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir Flask==2.0.1 Werkzeug==2.0.2 && \
    pip install --no-cache-dir scikit-learn==0.24.2 && \
    pip install --no-cache-dir pandas==1.3.1 && \
    pip install --no-cache-dir joblib==1.0.1

# Copy all files into the container at /app
COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python","app.py"]