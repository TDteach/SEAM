import os
import torch
import pickle
import time
import copy

from test import load_trigger, get_pattern_trigger_func, get_box_trigger_func, add_trigger
from test import make_SP_test_dataset
from train_cifar10 import prepare_dataset, build_model, load_model, train_epoch
from train_cifar10 import test as test_func
from utils import split_dataset, get_device

from functools import partial
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def test_model(model, dataloader, device):
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


def train(model, dataloader, n_classes, config, device, max_epochs=10, random_label=False, poisoned_dataloader=None,
          test_dataloader=None):
    time_sum = 0
    model.train()
    records = list()
    if random_label is True:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['forget_lr'], weight_decay=config['forget_wd'])
        # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)
        patience = 1
    else:
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.5, 0.9), lr=config['recover_lr'], weight_decay=config['recover_wd'])
        # optimizer = torch.optim.Adam(model.parameters(), betas=(0.5, 0.9), lr=5e-4, weight_decay=5e-5)
        # optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, weight_decay=5e-5) #for fine-tuning
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2) #for fine-tuning

        patience = 10
    # print(lr, weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    max_acc = 0
    for epoch in range(max_epochs):
        st_time = time.time()
        model.train()
        for img, lab in dataloader:
            if random_label:
                rd_lab = torch.randint(low=0, high=n_classes - 1, size=lab.shape)
                rd_lab[rd_lab >= lab] += 1
                lab = rd_lab
                # lab = (lab+1)%10
                # lab = torch.randint(low=0, high=n_classes, size=lab.shape)

            optimizer.zero_grad()
            img = img.to(device)
            lab = lab.to(device)
            logits = model(img)
            loss = loss_fn(logits, lab)
            # pred = torch.argmax(logits, axis=1)
            # correct = torch.sum(torch.eq(pred, lab)).detach().cpu().numpy()
            loss.backward()
            optimizer.step()

        ed_time = time.time()
        time_sum += ed_time - st_time

        if test_dataloader is not None:
            acc = test_model(model, test_dataloader, device=device)
        else:
            acc = test_model(model, dataloader, device=device)
        print('epoch %d:' % epoch, 'test acc:', acc, 'time elapse:', ed_time - st_time)
        if poisoned_dataloader is not None:
            asr = test_model(model, poisoned_dataloader, device=device)
            print('ASR:', asr)
            records.append({'epoch': epoch, 'acc': acc, 'asr': asr})
        else:
            records.append({'epoch': epoch, 'acc': acc})

        max_acc = max(max_acc, acc)

        if random_label:
            if acc < 2 / n_classes:
                patience -= 1
        else:
            if acc > 0.92:
                patience -= 1
            elif epoch > 50 and acc < max_acc - 1e-3:
                patience -= 1
        if patience <= 0:
            break

    return model, records, time_sum


def seam(config, model, poisoned_dataloader, test_dataloader, forget_dataloader, recover_dataloader, device, n_classes=10,
         savename=None):
    max_epochs = 300

    print('===========forget=================')

    ori_asr = test_model(model, poisoned_dataloader, device=device)
    ori_acc = test_model(model, test_dataloader, device=device)
    print('original ASR:', ori_asr, 'original ACC:', ori_acc)

    for_time = 0
    for_records = None
    model, for_records, for_time = train(model, forget_dataloader, n_classes, config=config, device=device, max_epochs=max_epochs, random_label=True,
                                         poisoned_dataloader=poisoned_dataloader, test_dataloader=test_dataloader)
    print('forget time usage:', for_time)

    print('===========recover=================')

    model, rec_records, rec_time = train(model, recover_dataloader, n_classes, config=config, device=device, max_epochs=max_epochs,
                                         random_label=False,
                                         poisoned_dataloader=poisoned_dataloader, test_dataloader=test_dataloader)
    print('recover time usage:', rec_time)
    print('total time comsuption: %.3f s' % (for_time + rec_time))

    best_fid, seam_acc, seam_asr = -1, -1, -1
    for rd in rec_records:
        if rd['acc'] - rd['asr'] > best_fid:
            seam_acc = rd['acc']
            seam_asr = rd['asr']
            best_fid = seam_acc - seam_asr
    print('Seam ASR:', seam_asr, 'Seam ACC:', seam_acc)
    print('Fidelity: %.2f%%' % (100 * best_fid / ori_acc))

    out_records = {'forget': for_records, 'recover': rec_records, 'for_time': for_time, 'rec_time': rec_time}

    if savename is None: savename = 'test'
    out_path = os.path.join('records', savename + '.pkl')

    with open(out_path, 'wb') as f:
        pickle.dump(out_records, f)

    return best_fid/ori_acc


def main(config=None):
    if config is None:
        config = dict()
    def_config = make_config()
    for key in def_config:
        if key not in config:
            config[key] = def_config[key]

    print(config)

    batch_size = 32

    home_path = '/home/tdteach/workspace/SEAM'
    trainset, testset = prepare_dataset(home_path)

    forget_dataset, _ = split_dataset(trainset, ratio=config["forget_ratio"])
    recover_dataset, _ = split_dataset(trainset, ratio=config["recover_ratio"])

    # model_path = home_path+'/checkpoint/benign_cifar10_resnet18.pth'
    model_path = home_path+'/checkpoint/box_4x4_resnet18.pth'
    # model_path = './checkpoint/trojan_0.8.pth'
    trigger_path = './checkpoint/trigger_first_try_0.8.pth'

    net = build_model()
    model, best_acc, start_epoch, _ = load_model(net, model_path)
    device = get_device()
    model.to(device)

    src_lb = 3
    tgt_lb = 5
    # mask_tanh_tensor, pattern_tanh_tensor, src_lb, tgt_lb = load_trigger(trigger_path)
    # trigger_func = get_pattern_trigger_func(mask_tanh_tensor, pattern_tanh_tensor)
    trigger_func = get_box_trigger_func()

    testsetBD = copy.deepcopy(testset)
    testsetBD = make_SP_test_dataset(testsetBD, src_lb, tgt_lb, trigger_func)

    recover_dataset, _ = add_trigger(recover_dataset, src_lb, tgt_lb, trigger_func, injection=config["injection"])

    forgetloader = torch.utils.data.DataLoader(
        forget_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    recoverloader = torch.utils.data.DataLoader(
        recover_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    testBDloader = torch.utils.data.DataLoader(
        testsetBD, batch_size=100, shuffle=False, num_workers=2)

    fid = seam(config, model, testBDloader, testloader, forgetloader, recoverloader, device, n_classes=10, savename=None)
    return fid


def tune_operator(config):
    fid = main(config)
    # fid = np.random.rand()
    tune.report(accuracy=fid)


def main_tune(num_samples=10, max_num_epochs=10, gpus_per_trial=0):
    config = {
        "forget_lr": tune.loguniform(1e-6, 1e-3),
        "forget_wd": tune.loguniform(1e-6, 1e-3),
        "recover_lr": tune.loguniform(1e-6, 1e-3),
        "recover_wd": tune.loguniform(1e-6, 1e-3),
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["accuracy"])
    result = tune.run(
        partial(tune_operator),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("accuracy","max","last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final fidelity: {}".format(
        best_trial.last_result["accuracy"]))

    best_fid = main(best_trial.config)
    print("Best trial test fidelity: {}".format(best_fid))


def make_config():
    config = {
        "forget_lr": 3e-5,
        "forget_wd": 5e-5,
        "recover_lr": 5e-4,
        "recover_wd": 5e-5,
        "forget_ratio": 0.001,
        "recover_ratio": 0.1,
        "injection": 2,
    }
    return config


if __name__ == '__main__':
    # record = dict()
    # for inj in range(2,200+1, 2):
    #     fid = main({'injection':inj+0.1})
    #     record[inj] = fid

    outpath = 'one_out_of_k.pkl'
    # with open(outpath,'wb') as f:
    #     pickle.dump(record, f)
    with open(outpath, 'rb') as f:
        record = pickle.load(f)
    print(record)
    with open('one_out_of_k.txt','w') as f:
        keys = sorted(record.keys())
        for k in keys:
            f.write('%d %.4f\n'%(k,record[k]))

    # main()
    # main_tune(num_samples=2, max_num_epochs=1, gpus_per_trial=0)
