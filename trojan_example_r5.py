# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import torch
import json
import pickle
import time

from torch.utils.data import TensorDataset, DataLoader

from trojan_example_r34 import test_model, train

import warnings

warnings.filterwarnings("ignore")


def load_models(model_filepath, tokenizer_filepath, embedding_filepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    cls_model = torch.load(model_filepath, map_location=torch.device(device))

    tok_model = torch.load(tokenizer_filepath)
    if 'input_ids' not in tok_model.model_input_names:
        tok_model.model_input_names = ['input_ids', 'attention_mask']

    # set the padding token if its undefined
    if not hasattr(tok_model, 'pad_token') or tok_model.pad_token is None:
        tok_model.pad_token = tok_model.eos_token
    # load the specified embedding
    emb_model = torch.load(embedding_filepath, map_location=torch.device(device))

    # identify the max sequence length for the given embedding
    max_input_length = tok_model.max_model_input_sizes[tok_model.name_or_path]

    return cls_model, tok_model, emb_model, max_input_length


def load_data(folder, tok_model, emb_model, max_input_length, cls_token_is_first):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fns = os.listdir(folder)
    fns.sort()

    all_lbs = list()
    all_emb = list()
    for fn in fns:
        if not fn.endswith('.txt'):
            continue
        items = fn.split('_')
        lbs = list()
        for k, it in enumerate(items):
            if it == 'class':
                lbs.append(int(items[k + 1]))
        lb = lbs[-1]
        all_lbs.append(lb)

        with open(os.path.join(folder, fn), 'r') as fh:
            text = fh.read()

        # tokenize the text
        results = tok_model(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
        # extract the input token ids and the attention mask
        input_ids = results.data['input_ids']
        attention_mask = results.data['attention_mask']

        # convert to embedding
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
            embedding_vector = emb_model(input_ids, attention_mask=attention_mask)[0]

            # ignore all but the first embedding since this is sentiment classification
            if cls_token_is_first:
                # for BERT-like models use the first token as the text summary
                embedding_vector = embedding_vector[:, 0, :]
                embedding_vector = embedding_vector.cpu().detach().numpy()
            else:
                # for GPT-2 use the last token as the text summary
                # embedding_vector = embedding_vector[:, -1, :]  # if all sequences are the same length
                # embedding_vector = embedding_vector.cpu().detach().numpy()
                embedding_vector = embedding_vector.cpu().detach().numpy()
                attn_mask = attention_mask.detach().cpu().detach().numpy()
                emb_list = list()
                for i in range(attn_mask.shape[0]):
                    idx = int(np.argwhere(attn_mask[i, :] == 1)[-1])
                    emb_list.append(embedding_vector[i, idx, :])
                embedding_vector = np.stack(emb_list, axis=0)

            # reshape embedding vector to create batch size of 1
            embedding_vector = np.expand_dims(embedding_vector, axis=0)
            # embedding_vector is [1, 1, <embedding length>]

            all_emb.append(embedding_vector)

    all_lbs = np.asarray(all_lbs)
    all_emb = np.concatenate(all_emb, axis=0)

    return all_emb, all_lbs


def inference_examples(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, examples_dirpath):
    print('model_filepath = {}'.format(model_filepath))
    print('cls_token_is_first = {}'.format(cls_token_is_first))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('embedding_filepath = {}'.format(embedding_filepath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    batch_size = 32
    max_epochs = 300

    md_folder, tail = os.path.split(examples_dirpath)
    rt_folder, mdid = os.path.split(md_folder)
    poisoned_dirpath = os.path.join(md_folder, 'poisoned_example_data')
    config_dirpath = os.path.join(md_folder, 'config.json')
    with open(config_dirpath, 'r') as jsonf:
        config = json.load(jsonf)
    trigger_config = config['triggers']

    cls_model, tok_model, emb_model, max_input_length = load_models(model_filepath, tokenizer_filepath,
                                                                    embedding_filepath)
    X, Y = load_data(examples_dirpath, tok_model, emb_model, max_input_length, cls_token_is_first)
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    dataset = TensorDataset(X_tensor, Y_tensor)
    n_classes = max(Y)+1

    X, Y = load_data(poisoned_dirpath, tok_model, emb_model, max_input_length, cls_token_is_first)
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    poisoned_dataset = TensorDataset(X_tensor, Y_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    poisoned_dataloader = DataLoader(poisoned_dataset, batch_size=batch_size, pin_memory=True)

    print(n_classes)

    model = cls_model


    acc = test_model(model, dataloader)
    asr = test_model(model, poisoned_dataloader)
    print('ACC:', acc, 'ASR:', asr)

    print('===========forget=================')

    model, for_records, for_time = train(model, dataloader, n_classes, epochs=max_epochs, random_label=True,
                                         poisoned_dataloader=poisoned_dataloader)
    print('time usage:', for_time)

    print('===========recover=================')

    model, rec_records, rec_time = train(model, dataloader, n_classes, epochs=max_epochs, random_label=False,
                                         poisoned_dataloader=poisoned_dataloader)
    print('time usage:', rec_time)

    out_records = {'forget': for_records, 'recover': rec_records, 'for_time': for_time, 'rec_time': rec_time}

    out_path = os.path.join('records', mdid+'.pkl')

    with open(out_path, 'wb') as f:
        pickle.dump(out_records, f)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.',
                        required=True)
    parser.add_argument('--cls_token_is_first',
                        help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.',
                        action='store_true', default=False)
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model file to be evaluated.',
                        required=True)
    parser.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model file to be evaluated.',
                        required=True)
    parser.add_argument('--examples_dirpath', type=str,
                        help='File path to the folder of examples which might be useful for determining whether a model is poisoned.',
                        required=True)

    args = parser.parse_args()

    inference_examples(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, args.embedding_filepath,
                       args.examples_dirpath)
