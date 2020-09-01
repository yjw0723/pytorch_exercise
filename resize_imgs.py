import os, cv2
from tqdm import tqdm
from skimage import io, transform

dir = 'E:/data/multi-label_classification_FASHION/imgs'
save_dir = 'E:/data/multi-label_classification_FASHION/imgs_resized'
os.makedirs(save_dir, exist_ok=True)
filenames = os.listdir(dir)

for filename in tqdm(filenames):
    path = os.path.join(dir, filename)
    image = io.imread(path)
    image = transform.resize(image, (224,224))
    save_path = os.path.join(save_dir, filename)
    io.imsave(save_path, image)
