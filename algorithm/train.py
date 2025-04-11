
import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision import models
from utils import read_split_data, plot_data_loader_image, train_one_epoch, evaluate
from dataset import MyDataSet
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR
from model.resnet import resnet50, resnet18, resnet152, resnext101_64x4d
# from model.vit_model import vit_base_patch16_224_in21k as vit_base
from model.efficient import efficientnet_b5 
from model.efficient_v2 import efficientnetv2_s
# from model.vgg import vgg
from model.se_resnext import se_resnext152, se_resnext101

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(args):
    # set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create weights folder
    if os.path.exists("/home/zty/data/arr_code_0326/weights/20250326resnet50") is False:
        os.makedirs("/home/zty/data/arr_code_0326/weights/20250326resnet50")

    # init tensorboard
    tb_writer = SummaryWriter(log_dir=rf"/home/zty/data/arr_code_0326/log/20250326resnet50")

    # split dataset into train and validate
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.annotations_path, args.data_path)

    # data transform
    size_ = 224
    data_transform = {
        "train": transforms.Compose([
                                     transforms.Resize((size_ ,size_ )),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
                                   transforms.Resize((size_ ,size_)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                   }

    # instantiate dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"],
                              label_transform=args.label_transform
                              )
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"],
                            label_transform=args.label_transform
                            )
    
    batch_size = args.batch_size
    nw = 0
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(val_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=False, 
                                                  num_workers=nw)

    # plot_data_loader_image(train_loader)

    train_num = len(train_dataset)
    val_num = len(val_dataset)


    print("using {} images for training, {} images for testing.".format(train_num,
                                                                           val_num))
    

    net = resnet50(num_classes=args.num_classes, include_top=True)

    
    if args.weights != '':
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        pretrained_dict = torch.load(args.weights, map_location='cpu')
        # print(pretrained_dict.keys())

        # print(net.load_state_dict(pretrained_dict, strict=False))


        del pretrained_dict['fc.weight']
        del pretrained_dict['fc.bias']
        

        missing_keys, unexpected_keys = net.load_state_dict(pretrained_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    net.to(device)

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # load checkpoint
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location=device)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("Resuming the training process from epoch {}...".format(args.start_epoch))
    else:
        args.start_epoch = 0



    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_loss = train_one_epoch(net, 
                                     optimizer,
                                     train_loader, 
                                     device, 
                                     epoch)
        scheduler.step()

        val_loss = evaluate(model=net, 
                            data_loader=validate_loader, 
                            device=device,
                            epoch=epoch)

        tags = ["train_loss", "val_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]['lr'], epoch)

        state = {
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
        }
        torch.save(state, '/home/zty/data/arr_code_0326/weights/20250326resnet50/resnet50_{}.pth'.format(epoch))    
        # torch.save(net.state_dict(), './weights/epoch_{}.pth'.format(epoch))

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=961)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--label_transform', type=bool, default=True)

    parser.add_argument('--data-path', type=str,
                        default=r"/home/zty/data/arr_code_0326/train_and_val/figs_train_test")
    parser.add_argument('--annotations-path', type=str, 
                        default=r'/home/zty/data/arr_code_0326/train_and_val/force_ditribution_train_test.csv')     


    parser.add_argument('--weights', type=str, 
                        default= r'/home/zty/data/arr_code_0326/pre_train/resnet50.pth',
                        # default=r'',
                        help='initial weights path')

    
    parser.add_argument('--resume', type=str, default='',
                    help='Path to resume training from a checkpoint')

    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)


