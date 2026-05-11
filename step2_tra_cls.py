import os
import argparse
import pandas as pd
import json

import torch
import torch.optim as optim

from MyDataSet import MyDataset_all
from utils import train_one_epoch_cls, test_cls, show_flops_params

from model.TNPPD import TNPPD

 
def main(args):
#------------------activatemodel----------------------
    device = args.device
    model = TNPPD().to(device)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    if args.show_flops_params: show_flops_params(model, device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    model_resume = args.model_resume_path
    model_preseg = args.model_preseg_path
    if model_resume != "":
        assert os.path.exists(model_resume), "weights file: '{}' not exist.".format(model_resume)
        weights_dict = torch.load(model_resume, map_location=device, weights_only=True)
        print(f"For resume: {model.load_state_dict(weights_dict)}")
    else :
        assert os.path.exists(model_preseg), "weights file: '{}' not exist.".format(model_preseg)
        weights_dict = torch.load(model_preseg, map_location=device, weights_only=True)
        print(f"For pre_seg: {model.load_state_dict(weights_dict, strict=False)}")

    frozen_layers = [model.seg_decoder1, model.seg_decoder2, model.seg_decoder3, model.seg_decoder4, model.seg_decoder5,
                    model.seg_decoder2_0, model.seg_decoder3_0, model.seg_decoder4_0] 

    
#------------------loaddata----------------------   
    data_path_c_train = args.data_path_c+'/train'
    data_path_c_test = args.data_path_c+'/test'

    _class = [cls for cls in os.listdir(data_path_c_test) if os.path.isdir(os.path.join(data_path_c_test, cls))]
    _class.sort()
    class_indices = dict((k, v) for v, k in enumerate(_class))
    json_str = json.dumps(dict((test, key) for key, test in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
        
    train_dataset_c = MyDataset_all(data_path_c_train, image_type1="c", image_type2="train")
    test_dataset_c = MyDataset_all(data_path_c_test, image_type1="c", image_type2="test")
    train_loader_c = torch.utils.data.DataLoader(train_dataset_c, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=nw)
    test_loader_c = torch.utils.data.DataLoader(test_dataset_c, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    df_c = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'best_test_loss_epoch', 'best_test_loss', 'best_test_acc_epoch', 'best_test_acc'])
    df_c.to_csv(args.csv_path_c, index=False)

    best_test_acc_epoch = 0
    best_test_acc = 0.0
    best_test_loss_epoch_c = 0
    best_test_loss_c = 100.0

    weightpath0=args.weights_path + '/cls_lastepoch.pth' 
    weightpath1=args.weights_path + '/cls_bestacc.pth'

#------------------trainmodel---------------------- 
    for epoch in range(args.epochs):
        # train
        train_loss_c, train_result = train_one_epoch_cls(model,optimizer,train_loader_c,device,epoch,args.epochs,frozen_layers)
        scheduler.step()
        # test
        test_loss_c, test_result = test_cls(model,test_loader_c,device,epoch,args.epochs)
        torch.save(model.state_dict(), weightpath0)
        #save
        if test_result[0] > best_test_acc:
            best_test_acc_epoch = epoch + 1
            best_test_acc = test_result[0]
            best_test_acc_result = test_result
            torch.save(model.state_dict(), weightpath1)
        #print

        result_c = [epoch + 1, train_loss_c, train_result[0], test_loss_c, test_result[0], best_test_loss_epoch_c, best_test_loss_c, best_test_acc_epoch, best_test_acc]
        data_c = pd.DataFrame([result_c])
        data_c.to_csv(args.csv_path_c, mode='a', header=False, index=False)

        print("=" * 110)
        print(f"{'Phase':<10}{'Loss':<13}{'acc':<10}{'pre':<10}{'rec':<10}{'f1':<10}{'kappa':<10}{'ap':<10}{'auc':<10}{'logloss':<10}{'brier':<10}")
        print("-" * 110)
        
        print(f"{'train':<10}{test_loss_c:<13.4f}", end='')
        for i in range(7):
            print(f"{test_result[i] * 100:<10.2f}", end='')
        print(f"{train_result[7] :<10.4f}", end='')
        print(f"{train_result[8] :<10.4f}", end='')
        print()

        print(f"{'test':<10}{test_loss_c:<13.4f}", end='')
        for i in range(7):
            print(f"{test_result[i] * 100:<10.2f}", end='')
        print(f"{test_result[7] :<10.4f}", end='')
        print(f"{test_result[8] :<10.4f}", end='')
        print()
        print("-" * 110)

        print(f"{'best_test':<10}{'--':<13}", end='')
        for i in range(7):
            print(f"{best_test_acc_result[i] * 100:<10.2f}", end='')
        print(f"{best_test_acc_result[7] :<10.4f}", end='')
        print(f"{best_test_acc_result[8] :<10.4f}", end='')
        print()
        print("-" * 110)
        print(f"{args.exp_name + '_cls_result'} || "  
              f"{'best ACC: '}{best_test_acc_result[0] * 100:<.2f} @ epoch {best_test_acc_epoch}")
        print("=" * 110)

    print('Step2 finished Training!') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--show_flops_params', type=str, default=True)
    parser.add_argument('--data_path_c', type=str, default='./TNUS/part1_for_cls')

    parser.add_argument('--model_resume_path', type=str, default='')
    parser.add_argument('--model_preseg_path', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='TNPPD')

    opt = parser.parse_args()

    save_path = f"./result/step2_cls/{opt.exp_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    opt.weights_path = save_path
    opt.csv_path_c = f"{save_path}/result_c.csv"

    main(opt)

    ## step2 train cls
    # python step2_tra_cls.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_c './TNUS/part1_for_cls' --model_preseg_path './TNPPD_weights/step1_tra_seg.pth'

    