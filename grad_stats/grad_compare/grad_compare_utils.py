import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchdump.utils import get_logger

logger = get_logger()

CSV_BLACK_LIST = r'^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]'

def check_path_exists(path):
    if not os.path.exists(path):
        raise Exception(f"The file path {path} does not exist.")

def read_csv(filepath):
    check_path_exists(filepath)
    try:
        csv_data = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Read csv file {filepath} failed.") from e
    return csv_data

def create_directory(output_dir):
    output_dir = os.path.realpath(output_dir)
    if os.path.exists(output_dir):
        logger.warning("Output directory: {} has already exists and will be overwritten.".format(output_dir))
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def plt_savefig(fig_save_path):
    try:
        plt.savefig(fig_save_path)
    except Exception as e:
        raise RuntimeError(f"save plt figure {fig_save_path} failed") from e

def load_npy(filepath):
    check_path_exists(filepath)
    try:
        npy = np.load(filepath, allow_pickle=False)
    except Exception as e:
        raise RuntimeError(f"Load numpy file {filepath} failed.") from e
    return npy
