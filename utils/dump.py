#!/usr/bin/env python3
"""
dumping script
TODO:
    generate inputs and params
"""

import os
import subprocess

SYNC_CMD = ["rsync", "-av"]
DATA_DIR = os.path.join("..", "data", "mnist")
INPUT_SRC_DIR = "okinawa:/home/work/takau/2.mlearn/mnist_data/"
PARAM_SRC_DIR = "okinawa:/home/work/takau/1.hw/bhewtek/data/mnist/lenet"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

subprocess.run(SYNC_CMD + [INPUT_SRC_DIR, DATA_DIR])
subprocess.run(SYNC_CMD + [PARAM_SRC_DIR, DATA_DIR])

