import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from brisque import BRISQUE
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

from water_common import data_num


obj = BRISQUE(url=False)
funcs = {
    'ssim': lambda img1, img2: ssim(img1, img2, multichannel=True, channel_axis=2),
    'psnr': psnr,
    'brisque': lambda img1, _: obj.score(img1),
    'brisque_w': lambda _, img2: obj.score(img2),
    'rmse': lambda img1, img2: np.sqrt(np.mean((np.array(img1).astype(np.int8) - np.array(img2).astype(np.int8)) ** 2)),
    'mae': lambda img1, img2: np.mean(np.abs(np.array(img1).astype(np.int8) - np.array(img2).astype(np.int8)))
}


def worker(file_pair: tuple, metrics=None):

    base_file, stega_file = file_pair
    # print(base_file, stega_file)
    # for tp in file_pairs:
        # print('tp', tp)
    base_img = cv2.imread(os.path.join(base_dir, base_file))
    stega_img = cv2.imread(os.path.join(stega_dir, stega_file))
    
    if metrics is None:
        metrics = funcs.keys()
    
    return {m: funcs[m](base_img, stega_img) for m in metrics}


def worker_wrapper(args):
    # print(args)
    return worker(args[0], metrics=args[1])
    

if __name__ == '__main__':
    import sys
    
    database = sys.argv[1]
    # argv_metrics = None if len(sys.argv) < 3 else 
    metric_lst = list(funcs.keys()) if len(sys.argv) < 3 else sys.argv[2].split(',')
    
    # YOU SHOULD SUBSTITUTE ... WITH YOUR OWN LOCAL FOLDER
    # base_dir = f'/root/autodl-tmp/data/{database}/raw'
    # stega_dir = f'/root/autodl-tmp/data/{database}/riva/pure'
    base_dir = ...
    stega_dir = ...
    
    common = [f'{i}.png' for i in range(data_num[database])]
    base_files = common
    stega_files = common
    
    file_zip = list(zip(base_files, stega_files))
    
    args = [(tp, metric_lst) for tp in file_zip]

    pool = Pool(processes=30)
    print('begin multi-processing')
    results = pool.map(worker_wrapper, args)
    print('end multi-processing')
    
    print('results: ', end="")
    for m in metric_lst:
        scores = [r[m] for r in results]
        print(f'{m} score: {np.mean(scores)}', end=" ")
    print('')
    