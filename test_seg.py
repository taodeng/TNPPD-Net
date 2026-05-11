import os
import argparse
import torch
import time
from MyDataSet import MyDataset_all
from utils import test_seg, show_flops_params

from model.TNPPD import TNPPD

 
def main(args):
    device = args.device
    model = TNPPD().to(device)


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    if args.show_flops_params: show_flops_params(model, device)


    model_test = args.model_test_path
    assert os.path.exists(model_test), "weights file: '{}' not exist.".format(model_test)
    weights_dict = torch.load(model_test, map_location=device, weights_only=True)
    print(f"For test model: {model.load_state_dict(weights_dict, strict=False)}")


    data_path_s = args.data_path_s+'/test'
    test_dataset_s = MyDataset_all(data_path_s, image_type1="s", image_type2="test")
    test_loader_s = torch.utils.data.DataLoader(test_dataset_s, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)
  

    test_loss_s, metrics = test_seg(model=model,data_loader_s=test_loader_s,device=device,epoch=0,epochs=0)
    print("=" * 70)
    print(f"{'Phase':<10}{'IoU(%)':<10}{'Dice(%)':<10}{'Pre(%)':<10}{'Rec(%)':<10}{'ASD':<10}")
    print("-" * 70)
    print(f"{'test':<10}{metrics['iou']*100:<10.2f}{metrics['dice']*100:<10.2f}{metrics['precision']*100:<10.2f}{metrics['recall']*100:<10.2f}{metrics['asd']:<10.2f}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--show_flops_params', type=str, default=True)
    parser.add_argument('--data_path_s', type=str, default='./TNUS/part2_for_seg')
    parser.add_argument('--exp_name', type=str, default='my')
    parser.add_argument('--model_test_path', type=str, default='')
    opt = parser.parse_args()
    main(opt)

    ## test seg result
    # python test_seg.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_s './TNUS/part2_for_seg' --model_test_path './TNPPD_weights/step1_tra_seg.pth'
    # python test_seg.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_s './TNUS/part2_for_seg' --model_test_path './TNPPD_weights/step2_tra_cls.pth'
    # python test_seg.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_s './TNUS/part2_for_seg' --model_test_path './TNPPD_weights/step3_retra_seg.pth'