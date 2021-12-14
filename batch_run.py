import os
import csv
import json
import torch
import re

home = os.environ['HOME']
contest_round = 'trojai/round3/round3-train-dataset'
folder_root = os.path.join(home, 'share/' + contest_round)
gt_path = os.path.join(folder_root, 'METADATA.csv')
row_filter = {
    'poisoned': ['True'],
    # 'trigger_option': ['both_trigger'],
    'trigger_type': None,
    # 'model_architecture':['google/electra-small-discriminator'],
    # 'model_architecture':['deepset/roberta-base-squad2'], # 'model_architecture': ['roberta-base'],
    'model_architecture': None,
    # 'source_dataset': ['squad_v2'],
    'source_dataset': None,
}


def read_gt(filepath):
    rst = list()
    with open(filepath, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            rst.append(row)
    return rst


def get_available_values(gt_csv):
    all_keys = dict()
    for row in gt_csv:
        for key in row:
            datastr = row[key]
            dataobj = None
            if datastr == 'None':
                dataobj = None
            elif datastr[0].isdigit():
                dataobj = json.loads(datastr)
            elif datastr.startswith('id'):
                continue
            elif datastr.startswith('{'):
                continue
            elif datastr.startswith('['):
                continue
            else:
                dataobj = datastr
            if isinstance(dataobj, str):
                if key not in all_keys:
                    all_keys[key] = list()
                all_keys[key].append(dataobj)

    for key in all_keys:
        all_keys[key] = set(all_keys[key])
    return all_keys


def filter_data(data_dict):
    new_dict = dict()
    for mdid in data_dict:
        good = True
        for item in row_filter:
            if row_filter[item] is None:
                continue
            feasible_list = row_filter[item]
            if item not in data_dict[mdid]:
                raise 'no item error'
            if data_dict[mdid][item] not in feasible_list:
                good = False
                break
        if good:
            new_dict[mdid] = data_dict[mdid]
    return new_dict


if __name__ == '__main__':

    gt_csv = read_gt(gt_path)
    all_keys = get_available_values(gt_csv)

    data_dict = dict()
    for row in gt_csv:
        md_name = row['model_name']
        data_dict[md_name] = row

    data_dict = filter_data(data_dict)

    dirs = sorted(data_dict.keys())
    for k, md_name in enumerate(dirs):
        name_num = int(md_name.split('-')[1])

        folder_path = os.path.join(folder_root, md_name)
        if not os.path.exists(folder_path):
            print(folder_path + ' dose not exist')
            continue
        if not os.path.isdir(folder_path):
            print(folder_path + ' is not a directory')
            continue

        model_filepath = os.path.join(folder_path, 'model.pt')
        examples_dirpath = os.path.join(folder_path, 'clean_example_data')

        md_archi = data_dict[md_name]['model_architecture']

        poisoned = data_dict[md_name]['poisoned']
        print('folder ', k + 1)
        print(md_name)
        print('poisoned:', poisoned)
        print('model_architecture:', md_archi)

        # run_script='singularity run --nv ./example_trojan_detector.simg'
        run_script = 'CUDA_VISIBLE_DEVICES=1 python3 trojan_example.py'
        cmmd = run_script + ' --model_filepath=' + model_filepath + ' --examples_dirpath=' + examples_dirpath

        print(cmmd)
        os.system(cmmd)

        #break
