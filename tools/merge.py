import os
import numpy as np
from tqdm import tqdm
from pathlib import Path


ROOT = "/data/home/liujing/projects/pointcept/data/forinstance_grid/test"
ROOT = Path(ROOT)

ORIGIN = "/data/home/liujing/projects/pointcept/data/forinstance/test"
ORIGIN = Path(ORIGIN)

result = Path("/data/home/liujing/projects/pointcept/work_dirs/ptv3-forinstance-500/grid_best/origin_result/")


for origin in list(ORIGIN.glob("*")):
    origin_name = origin.parts[-1]
    coord = np.load(ORIGIN / (origin_name + '/coord.npy'))
    new_pred = np.ones((coord.shape[0], 4)) * -1

    # print(coord.shape)
    # print(new_pred.shape)
    print(origin_name)


    all_index = []
    all_files = list(ROOT.glob(f"{origin_name}*"))
    print(f"processing {origin_name} : {len(all_files)}")
    for dir_name in tqdm(all_files):
        name = dir_name.parts[-1]
        # coord = np.load(ROOT / (dir_name / 'coord.npy'))

        pi = np.load(ROOT / (dir_name / 'point_index.npy'))

        all_index.extend(pi)
        data = np.load(result / f'{name}_pred.npy')
        new_pred[pi] = new_pred[pi] + data


    # pred = new_pred.argmax(axis=1)
    pred = new_pred

    np.save(f"work_dirs/ptv3-forinstance-500/grid_best/result/{origin_name}_pred.npy", pred)
