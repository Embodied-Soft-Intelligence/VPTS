import os
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from model.resnet import resnet152
from sklearn.metrics import mean_squared_error, mean_absolute_error
from parameter import label_mean, label_std
from dataset import MyDataSet
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kendalltau
from model.efficient import efficientnet_b5 
from model.efficient_v2 import efficientnetv2_s
from model.resnet import resnet50, resnet101, resnet152, resnext101_64x4d

def inverse_transform(lst, mean, std):
    lst = np.array(lst)
    mean, std = np.array(mean), np.array(std)
    return (lst * std + mean).tolist()

def calculate_metrics(true_labels, preds):
    mse = mean_squared_error(true_labels, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(true_labels, preds)
    return mse, rmse, mae

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    weights_dir = "/home/zty/data/arr_code_0326/weights/20250326resnet50/"
    weight_files = sorted([f for f in os.listdir(weights_dir) if f.endswith('.pth')])
    # selected_epochs = [i for i in range(40,41)]
    selected_epochs = [40]
    result_file = "/home/zty/data/arr_code_0326/weights/20250326resnet50/results.txt"
    
    label_path = "/home/zty/data/arr_code_0326/train_and_val/force_ditribution_val.csv"
    df = pd.read_csv(label_path)
    labels = df.iloc[:, 1:].values
    image_dir = "/home/zty/data/arr_code_0326/train_and_val/figs_val"  
                                                            
    image_paths = [os.path.join(image_dir, f"{img:06d}.jpg") for img in df.iloc[:, 0].values]
    dataset = MyDataSet(images_path=image_paths, images_class=labels, transform=data_transform, label_transform=True)
    
    with open(result_file, "w") as f:
        f.write("Epoch\tAxis\tRMSE\tMSE\tMAE\n")

        for weight_file in weight_files:
            epoch = int(weight_file.split('_')[-1].split('.')[0])
            if selected_epochs and epoch not in selected_epochs:
                continue
            
            model = resnet50(num_classes=961).to(device)
            checkpoint = torch.load(os.path.join(weights_dir, weight_file), map_location=device)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            
            all_preds, all_labels = [], []
            

            with torch.no_grad():
                for img, label in dataset:
                    img = img.unsqueeze(0).to(device)
                    output = torch.squeeze(model(img)).cpu().numpy().tolist()
                    output = inverse_transform(output, label_mean, label_std)
                    label = inverse_transform(label, label_mean, label_std)
                    
                    all_preds.append(output)
                    all_labels.append(label)
                    
        

            for axis, (start, end, axis_name) in enumerate([(0, 960, 'X')]):
                preds = [p[start:end] for p in all_preds]  
                true_labels = [l[start:end] for l in all_labels]  
                
                mse, rmse, mae = calculate_metrics(true_labels, preds)  
                
                f.write(f"{epoch}\t{axis_name}\t{rmse:.10f}\t{mse:.10f}\t{mae:.10f}\n")
                print(f"Epoch {epoch} - {axis_name} evaluation completed and saved.")


if __name__ == '__main__':
    main()
