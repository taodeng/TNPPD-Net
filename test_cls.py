import os
import argparse
import time
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, log_loss, brier_score_loss
from scipy.stats import binom_test
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

from MyDataSet import MyDataset_all
from utils import show_flops_params, test_cls

from model.TNPPD import TNPPD


def bootstrap_ci(true_labels, pred_labels, all_probs=None, n_bootstrap=1000, alpha=0.05):
    metrics = {'acc': [], 'f1': [], 'kappa': [], 'logloss': [], 'brier': []}
    y_true_np = np.array(true_labels)
    y_pred_np = np.array(pred_labels)
    y_probs_np = np.array(all_probs) if all_probs is not None else None

    for _ in range(n_bootstrap):
        idxs = np.random.choice(len(y_true_np), len(y_true_np), replace=True)
        yt = y_true_np[idxs]
        yp = y_pred_np[idxs]

        metrics['acc'].append(accuracy_score(yt, yp))
        metrics['f1'].append(f1_score(yt, yp))
        metrics['kappa'].append(cohen_kappa_score(yt, yp))

        if y_probs_np is not None:
            yp_probs = y_probs_np[idxs]
            metrics['logloss'].append(log_loss(yt, yp_probs))
            metrics['brier'].append(brier_score_loss(yt, yp_probs))
        else:
            metrics['logloss'].append(np.nan)
            metrics['brier'].append(np.nan)

    ci = {}
    for key in metrics:
        m = np.nanmean(metrics[key])
        lower = np.nanpercentile(metrics[key], 100*alpha/2)
        upper = np.nanpercentile(metrics[key], 100*(1-alpha/2))
        ci[key] = (m, lower, upper)
    return ci

def test_model(model, data_loader, device, compare=False):
    val_loss, result, pred, true, all_probs = test_cls(
        model,
        data_loader,
        device,
        epoch=0,
        epochs=0,
        return_preds=True
    )
    ci = bootstrap_ci(true, pred, all_probs)

    return ci, pred, true, all_probs


def main(args):
    device = args.device
    model = TNPPD().to(device)
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers per process')

    if args.show_flops_params: show_flops_params(model, device)

    model_test = args.model_test_path
    assert os.path.exists(model_test), f"weights file '{model_test}' not exist."
    weights_dict = torch.load(model_test, map_location=device, weights_only=True)
    model.load_state_dict(weights_dict, strict=False)
    print(f"Load test model success")

    data_path0 = os.path.join(args.data_path_c, 'test')
    data_path1 = os.path.join(args.data_path_c, 'external_test1')
    data_path2 = os.path.join(args.data_path_c, 'external_test2')
    data_path3 = os.path.join(args.data_path_c, 'external_test3')
    
    _class = [cls for cls in os.listdir(data_path0) if os.path.isdir(os.path.join(data_path0, cls))]
    _class.sort()
    class_indices = {k: v for v, k in enumerate(_class)}
    with open('class_indices.json', 'w') as f:
        json.dump({v: k for k, v in class_indices.items()}, f, indent=4)

    test_dataset0 = MyDataset_all(data_path0, image_type1="c", image_type2="test")
    test_loader0 = torch.utils.data.DataLoader(test_dataset0, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    test_dataset1 = MyDataset_all(data_path1, image_type1="c", image_type2="test")
    test_loader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)
    
    test_dataset2 = MyDataset_all(data_path2, image_type1="c", image_type2="test")
    test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)
    
    test_dataset3 = MyDataset_all(data_path3, image_type1="c", image_type2="test")
    test_loader3 = torch.utils.data.DataLoader(test_dataset3, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    ci0, pred0, true0, pred0_prob = test_model(model, test_loader0, device)
    ci1, pred1, true1, pred1_prob = test_model(model, test_loader1, device)
    ci2, pred2, true2, pred2_prob = test_model(model, test_loader2, device)
    ci3, pred3, true3, pred3_prob = test_model(model, test_loader3, device)


    print("=" * 130)
    print(f"{'Phase':<10}{'Acc(%)':<20}{'F1(%)':<20}{'Kappa(%)':<20}{'LogLoss':<23}{'Brier':<23}")
    print("-" * 130)
    print(f"{'test':<10}"
        f"{ci0['acc'][0]*100:.2f} ± {(ci0['acc'][0]-ci0['acc'][1])*100:.2f}"
        f"{'':<8}{ci0['f1'][0]*100:.2f} ± {(ci0['f1'][0]-ci0['f1'][1])*100:.2f}"
        f"{'':<8}{ci0['kappa'][0]*100:.2f} ± {(ci0['kappa'][0]-ci0['kappa'][1])*100:.2f}"
        f"{'':<8}{ci0['logloss'][0]:.4f} ± {(ci0['logloss'][0]-ci0['logloss'][1]):.4f}"
        f"{'':<8}{ci0['brier'][0]:.4f} ± {(ci0['brier'][0]-ci0['brier'][1]):.4f}"
        )

    print(f"{'test1':<10}"
        f"{ci1['acc'][0]*100:.2f} ± {(ci1['acc'][0]-ci1['acc'][1])*100:.2f}"
        f"{'':<8}{ci1['f1'][0]*100:.2f} ± {(ci1['f1'][0]-ci1['f1'][1])*100:.2f}"
        f"{'':<8}{ci1['kappa'][0]*100:.2f} ± {(ci1['kappa'][0]-ci1['kappa'][1])*100:.2f}"
        f"{'':<8}{ci1['logloss'][0]:.4f} ± {(ci1['logloss'][0]-ci1['logloss'][1]):.4f}"
        f"{'':<8}{ci1['brier'][0]:.4f} ± {(ci1['brier'][0]-ci1['brier'][1]):.4f}"
        )
    
    print(f"{'test2':<10}"
        f"{ci2['acc'][0]*100:.2f} ± {(ci2['acc'][0]-ci2['acc'][1])*100:.2f}"
        f"{'':<8}{ci2['f1'][0]*100:.2f} ± {(ci2['f1'][0]-ci2['f1'][1])*100:.2f}"
        f"{'':<8}{ci2['kappa'][0]*100:.2f} ± {(ci2['kappa'][0]-ci2['kappa'][1])*100:.2f}"
        f"{'':<8}{ci2['logloss'][0]:.4f} ± {(ci2['logloss'][0]-ci2['logloss'][1]):.4f}"
        f"{'':<8}{ci2['brier'][0]:.4f} ± {(ci2['brier'][0]-ci2['brier'][1]):.4f}"
        )
    
    print(f"{'test3':<10}"
        f"{ci3['acc'][0]*100:.2f} ± {(ci3['acc'][0]-ci3['acc'][1])*100:.2f}"
        f"{'':<8}{ci3['f1'][0]*100:.2f} ± {(ci3['f1'][0]-ci3['f1'][1])*100:.2f}"
        f"{'':<8}{ci3['kappa'][0]*100:.2f} ± {(ci3['kappa'][0]-ci3['kappa'][1])*100:.2f}"
        f"{'':<8}{ci3['logloss'][0]:.4f} ± {(ci3['logloss'][0]-ci3['logloss'][1]):.4f}"
        f"{'':<8}{ci3['brier'][0]:.4f} ± {(ci3['brier'][0]-ci3['brier'][1]):.4f}"
        )      
    print("=" * 130)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--show_flops_params', type=bool, default=True)
    parser.add_argument('--data_path_c', type=str, default='./TNUS/part1_for_cls')
    parser.add_argument('--model_test_path', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='TNPPD')
    opt = parser.parse_args()
    main(opt)

    ## test cls result
    # python test_cls.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_c '../TNUS/part1_for_cls' --model_test_path './TNPPD_weights/step2_tra_cls.pth'
    # python test_cls.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_c '../TNUS/part1_for_cls' --model_test_path './TNPPD_weights/step3_retra_seg.pth'

