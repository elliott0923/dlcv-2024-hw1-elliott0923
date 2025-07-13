#!/bin/bash

# TODO - run your inference Python3 code
echo "Starting BYOL pretraining..."
python3 train_ssl.py
echo "Done. Model saved to ./improved-net.pt"