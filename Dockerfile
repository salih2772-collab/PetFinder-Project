# 1. Use Python 3.10 slim image
FROM python:3.10-slim

# 2. Install system dependencies (Java for H2O, GLib for OpenCV)
RUN apt-get update && apt-get install -y \
    default-jdk \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all app files and models
COPY . .

# 6. Expose Streamlit port
EXPOSE 8501

# 7. Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]