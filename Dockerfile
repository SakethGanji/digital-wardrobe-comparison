# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Create a group with the same ID as your host user group
RUN groupadd -g 1000 hostgroup

# Create a user with the same ID as your host user and add it to the group
RUN useradd -u 1000 -g 1000 -ms /bin/bash hostuser

# Set the working directory in the container to /app
WORKDIR /workspace/digital-wardrobe-recommendation

