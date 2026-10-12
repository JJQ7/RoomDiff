"""Unit tests are CPU-only: hide all GPUs so nothing can initialise CUDA by accident."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
