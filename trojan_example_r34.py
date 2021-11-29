# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import skimage.io
import torch
import advertorch.attacks
import advertorch.context
import json
import pickle
import time

from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings("ignore")

DEVICE = 'cuda'


def image_transform(img):
    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    # normalize the image matching pytorch.transforms.ToTensor()
    img = img / 255.0
    return img


def load_img(fn):
    img = skimage.io.imread(fn)
    img = img.astype(dtype=np.float32)

    # perform center crop to what the CNN is expecting 224x224
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy + 224, dx:dx + 224, :]

    img = image_transform(img)
    return img


def load_img_folder(folder):
    # Inference the example images in data
    fns = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith(example_img_format)]
    fns.sort()  # ensure file ordering

    imgs = list()
    for fn in fns:
        img = load_img(fn)
        imgs.append(img)

    imgs_numpy = np.asarray(imgs)
    return imgs_numpy


class ImageFolderDataset(Dataset):
    def __init__(self, root):
        self.cls_dict = self.get_classes_from_dir(root)
        self.imgs, self.labs = self.load()
        self.n_classes = len(self.cls_dict.keys())

    def load(self):
        imgs = list()
        labs = list()
        for cls_id in self.cls_dict:
            for fn in self.cls_dict[cls_id]:
                img = load_img(fn)
                imgs.append(img)
                labs.append(cls_id)

        imgs = np.asarray(imgs, dtype=np.float32)
        labs = np.asarray(labs, dtype=np.int64)

        return imgs, labs

    def get_classes_from_dir(self, directory):
        fnames = sorted(entry.name for entry in os.scandir(directory) if not entry.is_dir())
        cls_dict = dict()
        for fn in fnames:
            if fn.startswith('class'):
                cls_id = int(fn.split('_')[1])
                if cls_id not in cls_dict:
                    cls_dict[cls_id] = list()
                cls_dict[cls_id].append(os.path.join(directory, fn))
        return cls_dict

    def __getitem__(self, idx):
        return self.imgs[idx], self.labs[idx]

    def __len__(self):
        return len(self.imgs)


class PoisonedFolderDataset(ImageFolderDataset):
    def __init__(self, root, trigger_config):
        self.tgr_cfg = trigger_config
        self.cls_dict = self.get_classes_from_dir(root)
        self.imgs, self.labs = self.load()
        self.n_classes = len(self.cls_dict.keys())

    def get_classes_from_dir(self, directory):
        fnames = sorted(entry.name for entry in os.scandir(directory) if not entry.is_dir())
        cls_dict = dict()
        for fn in fnames:
            if fn.startswith('class'):
                tgr_id = int(fn.split('_')[3])
                tgt_cls = self.tgr_cfg[tgr_id]['target_class']
                if tgt_cls not in cls_dict:
                    cls_dict[tgt_cls] = list()
                cls_dict[tgt_cls].append(os.path.join(directory, fn))
        return cls_dict


def test_model(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    crt, tot = 0, 0
    for img, lab in dataloader:
        img = img.to(device)
        lab = lab.to(device)
        logits = model(img)
        pred = torch.argmax(logits, axis=1)
        crt += torch.sum(torch.eq(pred, lab)).detach().cpu().numpy()
        tot += len(lab)

    return crt / tot


def train(model, dataloader, n_classes, epochs=10, random_label=False, poisoned_dataloader=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_sum = 0
    records = list()
    if random_label is True:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.5), lr=5e-4, weight_decay=5e-5)
        #optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    patience = 10
    max_acc = 0
    for epoch in range(epochs):
        st_time = time.time()
        model.train()
        for img, lab in dataloader:
            if random_label:
                lab = torch.randint(low=0, high=n_classes, size=lab.shape)

            optimizer.zero_grad()
            img = img.to(device)
            lab = lab.to(device)
            logits = model(img)
            loss = loss_fn(logits, lab)
            pred = torch.argmax(logits, axis=1)
            correct = torch.sum(torch.eq(pred, lab)).detach().cpu().numpy()
            loss.backward()
            optimizer.step()

        ed_time = time.time()
        time_sum += ed_time-st_time

        acc = test_model(model, dataloader)
        print('epoch %d:' % epoch, 'test acc:', acc, 'time elapse:', ed_time-st_time)
        if poisoned_dataloader is not None:
            asr = test_model(model, poisoned_dataloader)
            print('ASR:', asr)
        records.append({'epoch': epoch, 'acc': acc, 'asr': asr})

        max_acc = max(max_acc, acc)

        if random_label:
            if acc < 2/n_classes:
                patience -= 1
        else:
            if acc > 0.99:
                patience -= 1
            elif epoch > 50 and acc < max_acc-1e-3:
                patience -= 1
        if patience <= 0:
            break

    return model, records, time_sum


def fake_trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):
    print('model_filepath = {}'.format(model_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
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

    # load the model and move it to the GPU
    model = torch.load(model_filepath, map_location=torch.device(DEVICE))

    dataset = ImageFolderDataset(examples_dirpath)
    poisoned_dataset = PoisonedFolderDataset(poisoned_dirpath, trigger_config)
    n_classes = dataset.n_classes
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    poisoned_dataloader = DataLoader(poisoned_dataset, batch_size=batch_size, pin_memory=True)

    print('===========forget=================')

    asr = test_model(model, poisoned_dataloader)
    print('ASR:', asr)

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
                        default='./model.pt')
    parser.add_argument('--result_filepath', type=str,
                        help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.',
                        default='./output')
    parser.add_argument('--scratch_dirpath', type=str,
                        help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.',
                        default='./scratch')
    parser.add_argument('--examples_dirpath', type=str,
                        help='File path to the folder of examples which might be useful for determining whether a model is poisoned.',
                        default='./example')

    args = parser.parse_args()
    fake_trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)
