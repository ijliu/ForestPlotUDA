import argparse
from pathlib import Path
import numpy as np

def p():
    pass

def main():
    root = Path("work_dirs/ptv3-boreal3d-500/forinstance/result")
    files = list(root.glob("*.npy"))

    for name in files:
        data = np.load(name)
        print(data.shape)
        exit()

    pass

if __name__ == '__main__':
    main()