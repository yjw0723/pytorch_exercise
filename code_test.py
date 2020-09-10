from PIL import Image
import cv2, os
from tqdm import tqdm

root_dir = 'E:/data/multi-label_classification_Xray'
folder_list = os.listdir(root_dir)[4:16]
save_dir = 'E:/data/multi-label_classification_Xray/resized_imgs'
os.makedirs(save_dir, exist_ok=True)
for folder in folder_list:
    folder_path = os.path.join(root_dir, folder, 'images')
    file_list = os.listdir(folder_path)
    for filename in tqdm(file_list):
        path = os.path.join(folder_path, filename)
        save_path = os.path.join(save_dir, filename)
        img = Image.open(path)
        img = img.resize((512,512), Image.ANTIALIAS)
        img.save(save_path)
