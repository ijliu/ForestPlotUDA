# ForestPlotUDA

This project is built upon the **Pointcept** codebase, which is a powerful and flexible framework for point cloud perception research. You can find more information about Pointcept [here](https://github.com/Pointcept/Pointcept).

## Table of Contents
1. [Dataset Preparation](#dataset-preparation)
   - Boreal3D
   - Forinstance
2. [Training](#training)
3. [Testing](#testing)

---

## Dataset Preparation

Before starting, please ensure that you have installed all required dependencies by following the instructions in the [Pointcept repository](https://github.com/Pointcept/Pointcept).

### 1. Boreal3D
To process the **Boreal3D** dataset, follow these steps:

```bash
$ python pointcept/datasets/preprocessing/boreal3d/preprocess_boreal3d.py \
    --dataset_root ${RAW_BOREAL3D_DIR} \
    --output_root ${PROCESSED_BOREAL3D_DIR}
```

### 2. Forinstance

```bash
$ python pointcept/datasets/preprocessing/forinstance/preprocess_forinstance.py \
    --dataset_root ${RAW_FORINSTANCE_DIR} \
    --output_root ${PROCESSED_FORINSTANCE_DIR}
```

## Training
```bash
$ export PYTHONPATH=./
$ python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```
## Testing
```bash
$ export PYTHONPATH=./
$ python tools/test.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```
