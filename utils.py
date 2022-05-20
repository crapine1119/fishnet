import numpy as np
import pandas as pd
import cv2 as cv
import pickle
import torch
from torch.utils.data import Dataset


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class custom(Dataset):
    def __init__(self):
        super().__init__()