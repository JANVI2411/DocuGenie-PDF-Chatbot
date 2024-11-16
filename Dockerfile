# Use the official Python image, which is a lightweight Linux-based image with Python pre-installed
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Make the startup script executable
RUN chmod +x /app/start.sh

# Install dependencies directly since Python and pip are already available in the image
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Expose the required ports for FastAPI and Streamlit
EXPOSE 5000
EXPOSE 9000

# Start both FastAPI and Streamlit using the startup script
CMD ["sh","/app/start.sh"]
