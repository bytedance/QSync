import os 
from os import listdir
from os.path import isfile, join, isdir, abspath, exists
import shutil
import argparse
parser = argparse.ArgumentParser(description=' mv profiled result scripts.')
parser.add_argument('--model-name', type=str, default='bert',
                    help='mode_name') 
args = parser.parse_args()

ana_exp_folder = os.path.join(__file__, os.pardir, os.pardir, os.pardir, "analytical_predictor/traces/exp")
name = args.model_name
if not exists(abspath(join(ana_exp_folder, name))): os.makedirs(abspath(join(ana_exp_folder, name)))
dst_folder = ''

log_folder = "log"
only_dirs = [f for f in listdir(log_folder) if isdir(join(log_folder, f))]
for name_dir in only_dirs:
    if "@" not in name_dir: continue
    name_parts = name_dir.split("@")
    model_name, file_name = name_parts
    one_profiler_folder = join(log_folder, name_dir)
    onlyfiles = [f for f in listdir(one_profiler_folder) if isfile(join(one_profiler_folder, f))]
    # copy one under the 
    one_file = onlyfiles[0]
    one_file_path = join(one_profiler_folder,one_file)
    dst_file_path = abspath(join(ana_exp_folder, model_name.lower(), f"{file_name}.json"))
    print(one_file_path, dst_file_path)
    shutil.copyfile(one_file_path, dst_file_path)