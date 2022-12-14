# UvGRAF: UV-mapped Generative Radience Field

## Installation

To install and activate the environment, run the following command:
```
conda env create -f environment.yml -p env
conda activate ./env
```
This repo is built on top of [StyleGAN3](https://github.com/NVlabs/stylegan3), so make sure that it runs on your system.

## My notes

* train my model
```
CUDA_VISIBLE_DEVICES=0 python src/infra/launch.py hydra.run.dir=. hydra.job.chdir=True dataset=compcars dataset.resolution=64 num_gpus=1 training.batch_size=4 model=eg3d exp_suffix=dev_1
```


* process srn chair
```
python scripts/data_scripts/process_srn.py --target_dir data/chairs_64 --size 64
```

* carla_64: https://drive.google.com/file/d/1sIJY-MJ-ph9vdFpj4y6hbH65riRTJMkq/view?usp=sharing

* srnchairs_64: https://drive.google.com/file/d/1BJLTarlb0MShWqkqlZ3Dx2BatBy0ZI6p/view?usp=sharing