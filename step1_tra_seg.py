import os
import argparse
import pandas as pd

import torch
import torch.optim as optim

from MyDataSet import MyDataset_all
from utils import train_one_epoch_seg, test_seg, show_flops_params

from model.TNPPD import TNPPD



def main(args):
    #------------------activatemodel----------------------
    device = args.device
    model = TNPPD().to(device)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 24])
    print('Using {} dataloader workers every process'.format(nw))
    if args.show_flops_params: show_flops_params(model, device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    model_resume = args.model_resume_path
    if model_resume != "":
        assert os.path.exists(model_resume), "weights file: '{}' not exist.".format(model_resume)
        weights_dict = torch.load(model_resume, map_location=device, weights_only=True)
        print(f"For resume: {model.load_state_dict(weights_dict, strict=False)}")

    frozen_layers = [model.cls_level2, model.cls_level3, model.cls_blocks, model.fu_conv, model.fu_convfusion, model.fu_qkvfusion, model.cls_bn, model.cls_head]

#------------------loaddata----------------------   
    data_path_s_train = args.data_path_s+'/train'
    data_path_s_test = args.data_path_s+'/test'
        
    train_dataset_s = MyDataset_all(data_path_s_train, image_type1="s", image_type2="train")
    test_dataset_s = MyDataset_all(data_path_s_test, image_type1="s", image_type2="test")
    train_loader_s = torch.utils.data.DataLoader(train_dataset_s, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=nw)
    test_loader_s = torch.utils.data.DataLoader(test_dataset_s, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    df_s = pd.DataFrame(columns=['epoch', 'train_loss', 'train_IOU', 'test_loss', 'test_IOU', 'best_test_IOU_epoch', 'best_test_IOU'])
    df_s.to_csv(args.csv_path_s, index=False)

    best_test_IOU_epoch = 0
    best_test_IOU = 0

    weightpath0=args.weights_path + '/seg_lastepoch.pth' 
    weightpath1=args.weights_path + '/seg_bestiou.pth'

#------------------trainmodel---------------------- 
    for epoch in range(args.epochs):
        # train
        train_loss_s, metrics = train_one_epoch_seg(model,optimizer,train_loader_s,device,epoch,args.epochs,frozen_layers)
        scheduler.step()
        # test
        test_loss_s, metrics_v = test_seg(model,test_loader_s,device,epoch,args.epochs)
        torch.save(model.state_dict(), weightpath0)
        #save
        if metrics_v['iou'] > best_test_IOU:
            best_test_IOU_epoch = epoch + 1
            best_test_IOU = metrics['iou']
            torch.save(model.state_dict(), weightpath1)
        #print
        print("=" * 70)
        print(f"{'Phase':<12}{'loss':<12}{'IoU(%)':<10}{'Dice(%)':<10}{'Pre(%)':<10}{'Rec(%)':<10}{'ASD':<10}")
        print("-" * 70)
        print(f"{'train':<12}{train_loss_s:<12.4f}{metrics['iou']*100:<10.2f}{metrics['dice']*100:<10.2f}{metrics['precision']*100:<10.2f}{metrics['recall']*100:<10.2f}{metrics['asd']:<10.2f}")
        print(f"{'test':<12}{test_loss_s:<12.4f}{metrics_v['iou']*100:<10.2f}{metrics_v['dice']*100:<10.2f}{metrics_v['precision']*100:<10.2f}{metrics_v['recall']*100:<10.2f}{metrics_v['asd']:<10.2f}")
        print("-" * 70)
        print(f"{args.exp_name + '_seg_result'} || "  
              f"{'best IOU: '}{best_test_IOU * 100:<.2f} @ epoch {best_test_IOU_epoch}")
        print("=" * 70)
        result_s = [epoch + 1, train_loss_s, metrics['iou'], test_loss_s, metrics_v['iou'], best_test_IOU_epoch, best_test_IOU]
        data_s = pd.DataFrame([result_s])
        data_s.to_csv(args.csv_path_s, mode='a', header=False, index=False)

    print('Step1 finished Training!') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--show_flops_params', type=str, default=True)
    parser.add_argument('--data_path_s', type=str, default='./TNUS/part2_for_seg')
    parser.add_argument('--model_resume_path', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='TNPPD')

    opt = parser.parse_args()

    save_path = f"./result/step1_seg/{opt.exp_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    opt.weights_path = save_path
    opt.csv_path_s = f"{save_path}/result_s.csv"

    main(opt)


    ## step1 train seg
    # python step1_tra_seg.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_s './TNUS/part2_for_seg'