import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from parameter import label_mean, label_std
from model.resnet import resnet50, resnet101, resnet152, resnext101_64x4d
from model.vit_model import vit_base_patch16_224_in21k as vit_base
import pandas as pd
from pathlib import Path 

def inverse_transform(list, mean, std):
    list = np.array(list)
    mean = np.array(mean)
    std = np.array(std)
    list = list * std + mean
    list = list.tolist()
    return list

def process_image(index, img_path, fixed_img_path, model, device):

    img = Image.open(img_path)
    img2 = Image.open(fixed_img_path)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = data_transform(img)
    img2 = data_transform(img2)
    img_concatenated = torch.cat((img, img2), dim=0).unsqueeze(0)


    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_concatenated.to(device))).cpu()
        output = inverse_transform(output, label_mean, label_std)  
        output = torch.tensor(output)

    output_z = output.reshape(31, 31).numpy()



    # clim_low = -80
    # clim_high = 100

    # plt.subplot(1, 2, 2)
    # plt.imshow(output_z * 1000, cmap='jet')
    # plt.title("Predicted Z Heatmap")
    # plt.colorbar()
    # plt.clim(clim_low, clim_high)
    # out_put_path = f'/home/zty/data/arr_code_0326/result/figs_val'
    # Path(out_put_path).mkdir(parents=True, exist_ok=True)
    # plt.savefig(f'{out_put_path}/pre_Real_{index}.png', dpi=600)
    plt.close()
    

    # print(f"Image {index}: Mean Absolute Error = {error:.8f}")
    output_z = output.reshape(1, -1).numpy()
    return output_z

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = resnet50(num_classes=961).to(device)
    weights_path = r"/home/zty/data/arr_code_0326/weights/20250326resnet50/resnet50_40.pth"
    assert os.path.exists(weights_path), f"File: '{weights_path}' does not exist."
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])


    # val_path = "/home/zty/data/zty_vbts/dataset_vbts/fctact/hertz_force_map/force_ditribution_val.csv"
    img_dir = "/home/zty/data/arr_code_0326/val/trigger_0403_val/final_results_20250403_000747"
    
    fixed_img_path = r"/home/zty/data/arr_code_0326/origin/003.jpg"
    

    index_all = [i for i in range(0,30)]
    all_predictions = []
    for index in index_all:
        # img_path = os.path.join(img_dir, f"{index:06d}.jpg")
        img_path = os.path.join(img_dir, f"frame_{index:04d}.jpg")
 
        if not os.path.exists(img_path):
            print(f"Warning: File '{img_path}' does not exist. Skipping index {index}.")
            continue
        

        output_xyz = process_image(index, img_path, fixed_img_path, model, device)
        all_predictions.append(output_xyz[0])
        print(f"Processed image {index} successfully")

    df = pd.DataFrame(all_predictions)
    result_excel_path = rf"/home/zty/data/arr_code_0326/result/trigger_0403_val/final_results_20250403_000747/pre_result.xlsx"
    df.to_excel(result_excel_path, index_label="Image_Index")
    
if __name__ == '__main__':
    main()


