
import os
import pandas as pd
import sys
import random
import json
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def read_split_data(annotations_file, img_dir, train_ratio=0.8):

    data = pd.read_csv(annotations_file, header=0)   

    train_data, val_data = train_test_split(data, test_size=1 - train_ratio, random_state=42)

    train_images_path = [os.path.join(img_dir, f"{int(row[0]):06d}.jpg") for row in train_data.values]
    train_images_label = [row[1:].tolist() for row in train_data.values]
    val_images_path = [os.path.join(img_dir, f"{int(row[0]):06d}.jpg") for row in val_data.values]
    val_images_label = [row[1:].tolist() for row in val_data.values]

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_split_data1(root: str, val_rate: float = 0.2):
    random.seed(0)  
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    flower_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  
    train_images_label = []  
    val_images_path = []  
    val_images_label = []  
    every_class_num = []  
    supported = [".jpg", ".JPG", ".png", ".PNG"]  

    for cla in flower_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
  
        images.sort()

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else: 
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:

        plt.bar(range(len(flower_class)), every_class_num, align='center')

        plt.xticks(range(len(flower_class)), flower_class)

        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('image class')

        plt.ylabel('number of images')

        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label



def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        labels_combined = torch.stack(labels, dim=1)
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = img[:, :, :3]

            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])*255
            
            labels = labels_combined[i].numpy()[:3]  
            label_str = ", ".join([str(int(label)) for label in labels])  
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(label_str)
            plt.xticks([])  
            plt.yticks([])  
            plt.imshow(img.astype('uint8'))
        plt.show()

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  


    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels_list = data
        labels = torch.stack(labels_list, dim=1).to(device)
        labels = labels.float()
        logits = model(images.to(device))

        print("Logits shape:", logits.shape)
        print("Labels shape:", labels.shape)

        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accu_loss += loss.item()
        data_loader.desc = "[train epoch {}] loss: {:.6f}".format(epoch,
                                                                  accu_loss.item() / (step + 1))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
    return accu_loss.item() / (step + 1)



@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.MSELoss()
    model.eval()

    accu_loss = torch.zeros(1).to(device)  

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels_list = data
        labels = torch.stack(labels_list, dim=1).to(device)
        labels = labels.float()

        pred = model(images.to(device))
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.6f}".format(epoch, accu_loss.item() / (step + 1))

    return accu_loss.item() / (step + 1)