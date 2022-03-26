<br />
<p align="center">
  <h1 align="center">Sal.Co </h3>  

  <h3 align="center">PyTorch-Based Co-Saliency testing framework</h3>

  <p align="center">
    Automatically evaluate your saliency algorithm in co-saliency detection task.
    Extend your saliency algorithm with co-localization funcionalities.
  </p>
</p>

## Description

TODO

## Usage
### Requirements
- Python >= 3.5,
- Numpy,
- Pandas,
- OpenCV,
- PyTorch,
- DenseCRF.

### Datasets
Dataset has to have data separated into classes (categories).

Example testing datasets: 
- CoSAL2015,
- CoSOD3k
- CoCA.

Usually saliency algorithms doesn't produce results separated into classes. We provide utility scripts to flatten directory structure and bring it back to original form. Those scripts are located in `src/utils` directory.

### How to run?
```
    python3 src/main.py \
        --train_dir {PATH_TO_YOUR_DATA_DIR}/CoCA/image \
        --sal_dir {PATH_TO_YOUR_DATA_DIR}/LDF_group/CoCA \
        --save_dir {GIT_ROOT}/cosal_sal_testing/data/results/CoCA
```

You could use GPU if you have one.
```
    CUDA_VISIBLE_DEVICES=0 python3 src/main.py \
        --train_dir {PATH_TO_YOUR_DATA_DIR}/CoCA/image \
        --sal_dir {PATH_TO_YOUR_DATA_DIR}/LDF_group/CoCA \
        --save_dir {GIT_ROOT}/cosal_sal_testing/data/results/CoCA
```

## Results

We used our framework to evaluate two state-of-the-art saliency methods in co-saliency detection task. Specifically, we used [LDF](https://github.com/weijun88/LDF) and [U<sup>2</sup>-Net](https://github.com/xuebinqin/U-2-Net).

We used the [Eval Co-SOD](https://github.com/zzhanghub/eval-co-sod) tool for evaluation.

TODO

insert results for coca

insert table with results for coca


<!-- ### Contact -->


