
import cv2
import os

def images_to_video(image_folder, output_video, fps=30):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the specified folder.")
        return
    
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(image_folder, output_video)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Skipping {image}, unable to read.")
            continue
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved as {output_video_path}")


name = "20250402_200043"
image_folder = f"/home/zty/data/arr_code_0326/result/pen/final_results_{name}"  
output_video = f"video_{name}.mp4"     
fps = 10  

images_to_video(image_folder, output_video, fps)


