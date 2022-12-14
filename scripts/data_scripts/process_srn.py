import os
import re
import shutil
import argparse
import csv

from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed

from scripts.utils import tqdm_joblib, resize_and_save_image, file_ext

#----------------------------------------------------------------------------

def resize_dataset(target_dir: str=None, size: int=None, format: str='.jpg', num_jobs: int=8, ignore_regex: str=None, ignore_ext: str=None, images_only: bool=False, fname_prefix: str=''):
    assert not size is None

    Image.init() # to load the extensions
    # target_dir = f'{source_dir}_{size}' if target_dir is None else target_dir
    # file_names = {os.path.relpath(os.path.join(root, fname), start=source_dir) for root, _dirs, files in os.walk(source_dir) for fname in files}

    # if not ignore_ext is None:
    #     file_names = {f for f in file_names if not f.endswith(ignore_ext)}

    # if not ignore_regex is None:
    #     file_names = {f for f in file_names if not re.fullmatch(ignore_regex, f)}

    with open('/data/anjie/data/srn_chairs/srn_chairs_train_filted.csv', newline='') as f:
        reader = csv.reader(f)
        file_names = list(reader)

    file_names = [os.path.join('/data/anjie/data/srn_chairs/chairs_train/chairs_2.0_train', x[0]) for x in file_names]

    jobs = []
    dirs_to_create = set()

    for src_path in tqdm(file_names, desc=f'Collecting jobs'):
        if file_ext(src_path) in Image.EXTENSION:
            object_name = src_path.split('/')[-3]
            basename = os.path.splitext(os.path.basename(src_path))[0]
            trg_path = os.path.join(target_dir, fname_prefix + object_name + '_' + basename + format)
            print(trg_path)
            jobs.append(delayed(resize_and_save_image)(
                src_path=src_path,
                trg_path=trg_path,
                size=size,
            ))
        else:
            trg_path = None

        if not trg_path is None:
            dirs_to_create.add(os.path.dirname(trg_path))

    for d in tqdm(dirs_to_create, desc='Creating necessary directories'):
        if d != '':
            os.makedirs(d, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Executing jobs", total=len(jobs))) as progress_bar:
        Parallel(n_jobs=num_jobs)(jobs)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_dir', required=False, type=str, default=None, help='Target directory (default: `{source_dir}_{size}`)')
    parser.add_argument('-s', '--size', required=True, type=int, help='Target size.')
    parser.add_argument('-f', '--format', type=str, default='.jpg', help='In which format should we save?')
    parser.add_argument('-j', '--num_jobs', type=int, default=8, help='Number of jobs for parallel execution')
    parser.add_argument('--ignore_ext', type=str, default='.DS_Store', help='File extension to ignore.')
    parser.add_argument('--fname_prefix', type=str, default='', help='Add this prefix to each file name.')
    parser.add_argument('--images_only', action='store_true', help='Process images only?')
    args = parser.parse_args()

    resize_dataset(
        target_dir=args.target_dir,
        size=args.size,
        format=args.format,
        num_jobs=args.num_jobs,
        ignore_ext=args.ignore_ext,
        fname_prefix=args.fname_prefix,
        images_only=args.images_only,
    )

#----------------------------------------------------------------------------
