# Use an official PyTorch CUDA-enabled base image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime



# Install Python 3.10.13
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Python dependencies
# Note: You might need to add other dependencies depending on your project
COPY requirements.txt ./
RUN python3.8 -m pip install --no-cache-dir -r requirements.txt

# Copy the training script and other necessary files into the container
COPY train_col.py .
COPY pairs.csv .
COPY long_string.csv .

# Run the training script
# Note: This command can be overridden when running the container
CMD ["python3.10", "train_col.py"]
