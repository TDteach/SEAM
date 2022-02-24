import os
import csv
import json
import re

home = os.environ['HOME']
# contest_round = 'round5/round5-train-dataset'
# folder_root = os.path.join(home, 'data/' + contest_round)
# model_factories_file = 'model_factories_r5.py'
contest_round = 'round6/round6-leftover-dataset'
folder_root = os.path.join(home, contest_round)
model_factories_file = 'model_factories_r6.py'
gt_path = os.path.join(folder_root, 'METADATA.csv')
row_filter = {
    'poisoned': ['True'],
    'trigger_type': None,
    'model_architecture': None,
    'source_dataset': None,
}


def get_extended_name(embedding, embedding_flavor):
    a = re.split('-|/', embedding_flavor)
    b = [embedding]
    b.extend(a)
    return '-'.join(b)


def check_cls_token(embedding, cls_token_is_first):
    if len(cls_token_is_first) == 0:
        if embedding == 'BERT':
            cls_token_is_first = 'True'
        elif embedding == 'DistilBERT':
            cls_token_is_first = 'True'
        elif embedding == 'GPT-2':
            cls_token_is_first = 'False'
    return cls_token_is_first


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
            # print(key, row[key])
        # exit(0)

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

    z = 0
    w = 0
    wt = 0
    data_dict = dict()
    for row in gt_csv:
        md_name = row['model_name']
        z += float(row['final_clean_data_test_acc'])
        if row['poisoned'] == 'True':
            w += float(row['final_triggered_data_test_acc'])
            wt += 1
        data_dict[md_name] = row
    print(z/len(gt_csv))
    print(w/wt)
    exit(0)

    data_dict = filter_data(data_dict)

    dirs = sorted(data_dict.keys())
    for k, md_name in enumerate(dirs):
        name_num = int(md_name.split('-')[1])

        # if name_num < 900:
        #     continue


        embedding = data_dict[md_name]['embedding']  # BERT
        cls_token_is_first = check_cls_token(embedding, '')
        embedding_flavor = data_dict[md_name]['embedding_flavor']  # bert-base-uncased
        ext_embedding = get_extended_name(embedding, embedding_flavor)

        tokenizer_filepath = os.path.join(folder_root, 'tokenizers', ext_embedding + '.pt')
        embedding_filepath = os.path.join(folder_root, 'embeddings', ext_embedding + '.pt')

        folder_path = os.path.join(folder_root, 'models/'+md_name)
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
        run_script = 'CUDA_VISIBLE_DEVICES=0 python3 trojan_example_r5.py'
        cmmd = run_script + ' --model_filepath=' + model_filepath + ' --examples_dirpath=' + examples_dirpath + \
               ' --tokenizer_filepath=' + tokenizer_filepath + ' --embedding_filepath=' + embedding_filepath
        if cls_token_is_first == 'True':
            cmmd = cmmd + ' --cls_token_is_first'

        cp_cmmd = 'cp '+model_factories_file+' model_factories.py'
        os.system(cp_cmmd)

        print(cmmd)
        os.system(cmmd)

        # break
