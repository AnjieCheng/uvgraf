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

* train my model with no_patch & on Atlantis
```
CUDA_VISIBLE_DEVICES=0 python src/infra/launch.py hydra.run.dir=. hydra.job.chdir=True dataset=compcars dataset.resolution=64 num_gpus=1 training=no_patch training.batch_size=4 model=canograf dataset.path=/home/anjie/Downloads/CADTextures/CompCars exp_suffix=XXX
```

* process srn chair
```
python scripts/data_scripts/process_srn.py --target_dir data/chairs_64 --size 64
```

* carla_64: https://drive.google.com/file/d/1sIJY-MJ-ph9vdFpj4y6hbH65riRTJMkq/view?usp=sharing

* srnchairs_64: https://drive.google.com/file/d/1BJLTarlb0MShWqkqlZ3Dx2BatBy0ZI6p/view?usp=sharing

* Currently does not support no_patch 32x32 discriminator (b4 block wrong resolution)

* Currently does not support snapshot vis grid with odd number... because torch.gradient not supporting batch_size=1
