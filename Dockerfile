# Start with a RHEL Universal Base Image
FROM ubuntu:22.04

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Make the startup script executable
RUN chmod +x /app/start.sh

# Update and install Python, pip, and other dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install -r requirements.txt && \
    apt-get clean all && rm -rf /var/cache/yum /var/cache/dnf /root/.cache/pip

# Expose the required ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Start both FastAPI and Streamlit using the startup script
CMD ["sh","/app/start.sh"]

# # Start with a RHEL Universal Base Image
# FROM registry.access.redhat.com/ubi9/ubi-minimal

# # Set the working directory
# WORKDIR /app

# # Copy all files into the container
# COPY . /app

# # Make the startup script executable
# RUN chmod +x /app/start.sh

# # Update and install Python, pip, and other dependencies
# RUN microdnf install -y python3 python3-pip && \
#     python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
#     python3 -m pip install -r requirements.txt && \
#     microdnf clean all && rm -rf /var/cache/yum /var/cache/dnf /root/.cache/pip

# # Expose the required ports for FastAPI and Streamlit
# EXPOSE 8000
# EXPOSE 8501

# # Start both FastAPI and Streamlit using the startup script
# CMD ["/app/start.sh"]

# # Start with an Ubuntu base image
# FROM ubuntu:22.04

# # Set the working directory
# WORKDIR /app

# # Copy all files into the container
# COPY . /app

# # Update and install Python, pip, and other dependencies
# RUN apt-get update && apt-get install -y python3 python3-pip supervisor
# RUN python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# RUN python3 -m pip install -r requirements.txt && \
#     apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# # Expose the required ports for FastAPI and Streamlit
# EXPOSE 8000
# EXPOSE 8501

# # Copy the supervisor configuration file
# COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# # Start Supervisor to manage FastAPI and Streamlit
# CMD ["/usr/bin/supervisord"]