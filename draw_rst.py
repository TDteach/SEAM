import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def read_records(folder):
    rst_dict = dict()

    fns = os.listdir(folder)
    for fn in fns:
        if not fn.endswith('.pkl'):
            continue
        fpath = os.path.join(folder, fn)
        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        tot_time = data['for_time'] + data['rec_time']
        z = data['recover']

        z.sort(key=lambda w: w['acc'] - w['asr'], reverse=True)
        acc, asr = z[0]['acc'], z[0]['asr']

        rst_dict[fn] = {'time': tot_time, 'acc': acc, 'asr': asr, 'fidelity': acc - asr}

    return rst_dict


if __name__ == '__main__':

    # folders = ['records_round4_0','records_round4_1']
    folders = ['records_round6_0']
    # folders = ['records']
    rsts = list()
    for fo in folders:
        rsts.append(read_records(fo))

    fns = list(rsts[0].keys())
    fns.sort()

    fide_list = list()
    time_list = list()
    asr_list = list()
    acc_list = list()
    for fn in fns:
        accs = [rst[fn]['acc'] for rst in rsts]
        asrs = [rst[fn]['asr'] for rst in rsts]
        fids = [rst[fn]['fidelity'] for rst in rsts]
        tims = [rst[fn]['time'] for rst in rsts]
        acc = max(accs)
        asr = min(asrs)
        fid = max(fids)
        tim = min(tims)

        fide_list.append(fid)
        time_list.append(tim)
        asr_list.append(asr)
        acc_list.append(acc)

    order = list(range(len(fns)))
    order.sort(key=lambda k: asr_list[k], reverse=True)
    for o in order:
        print(fns[o], asr_list[o], acc_list[o], fide_list[o])

    print('mean fidelity:', np.mean(fide_list))
    print('mean time:', np.mean(time_list))
    print('mean asr:', np.mean(asr_list))
    print('mean acc:', np.mean(acc_list))

    print(len(fide_list))
    print(len(acc_list))

    hist, bin_edges = np.histogram(np.asarray(fide_list), bins=50, density=False)
    hist = hist.astype(np.float32) / sum(hist)
    print(sum(hist))

    z = 0
    for h,b in zip(hist,bin_edges[1:]):
        z += h
        print(z,b)

    # for i in range(1, len(hist)):
    #     hist[i] += hist[i - 1]

    plt.plot(bin_edges[1:], hist)
    plt.show()
