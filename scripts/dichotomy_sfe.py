from argparse import Namespace, ArgumentParser
import time
import os
import sys
import clip
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import subprocess
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

CODE_DIR = 'StyleFeatureEditor'
os.chdir(f'/home/ayavasileva/{CODE_DIR}')
sys.path.append(".")
sys.path.append("..")

from arguments import inference_arguments
from runners.inference_runners import inference_runner_registry
from utils.common_utils import printer, setup_seed

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--inverted_images_dir', type=str, required=True,
                       help='Path to directory with inverted images')
    parser.add_argument('--output_root_dir', type=str, default='/home/ayavasileva/data',
                       help='Root directory for output folders')
    parser.add_argument('--annotation_path', type=str, default="/home/ayavasileva/3000.txt",
                       help='Path to annotations file')
    parser.add_argument('--attribute_name', type=str, required=True,
                       help='Name of attribute to edit (e.g., smile)')
    parser.add_argument('--attribute_column', type=str, required=True,
                       help='Column name in annotation file for the attribute')
    parser.add_argument('--fid_script_path', type=str, default='/home/ayavasileva/StyleFeatureEditor/scripts/fid_calculation.py',
                       help='Path to FID calculation script')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name')
    parser.add_argument('--method_name', type=str, required=True)
    return parser.parse_args()

def make_folder_with_attribute(attr_name: str, col_name: str, annotation, img_dir, output_root, model_name, create_imgs=True):
    if col_name.endswith("_r"):
        col_name = col_name[:-2]

    if not col_name in annotation.columns:
        print("No such column")
        return

    files_with_attribute = annotation[annotation[col_name] == 1]["path"]
    files_without_attribute = annotation[annotation[col_name] != 1]["path"]
    print(f"Files with attribute: {len(files_with_attribute)}")
    print(f"Files without attribute: {len(files_without_attribute)}")
    without_attribute_idx = annotation[annotation[col_name] != 1].index
    with_attribute_idx = annotation[annotation[col_name] == 1].index
    
    new_dir = os.path.join(output_root, f'files_with_{attr_name}_{model_name}')
    new_without_dir = os.path.join(output_root, f'files_without_{attr_name}_{model_name}')

    if create_imgs:
        os.makedirs(new_dir, exist_ok=True)
        os.makedirs(new_without_dir, exist_ok=True)

        for im_path in files_with_attribute:
            source_file_path = os.path.join(img_dir, im_path)
            target_file_path = os.path.join(new_dir, im_path)
            if os.path.exists(source_file_path):
                shutil.copyfile(source_file_path, target_file_path)

        for im_path in files_without_attribute:
            source_file_path = os.path.join(img_dir, im_path)
            target_file_path = os.path.join(new_without_dir, im_path)
            if os.path.exists(source_file_path):
                shutil.copyfile(source_file_path, target_file_path)
        if os.path.exists(new_dir + "/.ipynb_checkpoints"):
            subprocess.run(["rm", "-r", new_dir + "/.ipynb_checkpoints"], check=True)
        if os.path.exists(new_without_dir + "/.ipynb_checkpoints"):
            subprocess.run(["rm", "-r", new_without_dir + "/.ipynb_checkpoints"], check=True)

    return without_attribute_idx

def binary_search_for_s(runner, strengths, inds_without, output_root, 
                       method_name, direction_name, celeba_attr, annotation_path, fid_script_path, model_name):
    known_fid_scores = {}
    left = 0
    right = len(strengths) - 1
    mid = left + (right - left) // 2

    cur_edit_path = os.path.join(output_root, f"{model_name}")
    runner.config.inference.editings_data = {method_name: [strengths[mid]]}
    
    runner.run_editing()

    orig_path = os.path.join(output_root, f"files_with_{direction_name}_{model_name}")

    result = subprocess.run(['python', fid_script_path, 
                           f'--orig_path={orig_path}',
                           f'--synt_path={cur_edit_path +f"/{method_name}/edit_power_{strengths[mid]:.4f}"}',
                           f'--attr_name={celeba_attr}',  
                           f'--celeba_attr_table_pth={annotation_path}'],
                          capture_output=True, text=True)
    print(result)
    fid_score_mid = float(result.stdout.split('\n')[-2].split()[-1])
    known_fid_scores[strengths[mid]] = fid_score_mid
    
    while left < right:
        if right - left == 1:
            if known_fid_scores[strengths[left]] < known_fid_scores[strengths[right]]:
                best_score = known_fid_scores[strengths[left]]
                best_strength = strengths[left]
            else:
                best_score = known_fid_scores[strengths[right]]
                best_strength = strengths[right]
            print("best score is: ", best_score, best_strength)
            return best_score, best_strength
        
        y = left + (mid - left) // 2
        print(f"Left: {left}, Mid: {mid}, Right: {right}")
        print(f"Strengths: {strengths[left]}, {strengths[mid]}, {strengths[right]}")

        # Check strength y
        if strengths[y] not in known_fid_scores:
            runner.config.inference.editings_data = {method_name: [strengths[y]]}
            runner.run_editing()

          
            result = subprocess.run(['python', fid_script_path, 
                                   f'--orig_path={orig_path}',
                                   f'--synt_path={cur_edit_path +f"/{method_name}/edit_power_{strengths[y]:.4f}"}',
                                   f'--attr_name={celeba_attr}',  
                                   f'--celeba_attr_table_pth={annotation_path}'],
                                  capture_output=True, text=True)
            fid_score_y = float(result.stdout.split('\n')[-2].split()[-1])
            known_fid_scores[strengths[y]] = fid_score_y
        else:
            fid_score_y = known_fid_scores[strengths[y]]

        print(f"fid_score={fid_score_y} for strength={strengths[y]}")
        print(f"fid_score={fid_score_mid} for strength={strengths[mid]}")    

        if known_fid_scores[strengths[y]] <= fid_score_mid:
            right = mid
            mid = y
            fid_score_mid = known_fid_scores[strengths[mid]]
        else:
            z = mid + (right - mid) // 2
            if strengths[z] not in known_fid_scores:
                runner.config.inference.editings_data = {method_name: [strengths[z]]}
                runner.run_editing()
                
                result = subprocess.run(['python', fid_script_path, 
                                       f'--orig_path={orig_path}',
                                       f'--synt_path={cur_edit_path +f"/{method_name}/edit_power_{strengths[z]:.4f}"}',
                                       f'--attr_name={celeba_attr}',  
                                       f'--celeba_attr_table_pth={annotation_path}'],
                                      capture_output=True, text=True)
                # print(result)
                fid_score_z = float(result.stdout.split('\n')[-2].split()[-1])
                known_fid_scores[strengths[z]] = fid_score_z
                
            if fid_score_mid <= known_fid_scores[strengths[z]]:
                left = y
                right = z
            else:
                left = mid
                mid = z
                fid_score_mid = known_fid_scores[strengths[mid]]
    print(f"best_score is: {fid_score_mid} and best strength is: {strengths[mid]}")
    return fid_score_mid, strengths[mid]

def main():
    args = parse_args()
    config = inference_arguments.load_config()
    setup_seed(config.exp.seed)
    printer(config)
    

    inference_runner = inference_runner_registry[config.inference.inference_runner](
        config
    )
    inference_runner.setup()

    # # Run inversion
    inference_runner.run_inversion()
    
    # Load annotation data
    annotation = pd.read_csv(args.annotation_path, delimiter=" ", skiprows=[0])
    annotation.reset_index(inplace=True)
    annotation.drop(columns=["level_1"], inplace=True)
    annotation.rename(columns={'level_0': 'path'}, inplace=True)

    # Prepare data folders and get indices without attribute
    inds_without = make_folder_with_attribute(
        attr_name=args.attribute_name,
        col_name=args.attribute_column,
        annotation=annotation,
        img_dir=args.inverted_images_dir,
        output_root=args.output_root_dir,
        model_name=args.model_name,
        create_imgs=True
    )

    # Perform binary search for best strength
    strengths = [float(x) for x in np.linspace(0, 30, 20)]
    best_score, best_strength = binary_search_for_s(
        runner=inference_runner,
        strengths=strengths,
        inds_without=inds_without,
        output_root=args.output_root_dir,
        method_name=args.method_name,
        direction_name=args.attribute_name,
        celeba_attr=args.attribute_column,
        annotation_path=args.annotation_path,
        fid_script_path=args.fid_script_path,
        model_name=args.model_name
    )

#     print(f"Optimization complete. Best FID score: {best_score} with strength: {best_strength}")

if __name__ == "__main__":
    main()