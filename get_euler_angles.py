import argparse
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset.datasets import WLFWDatasets
import torchvision.models as models
from PIL import Image
import os
from scipy.spatial.transform import Rotation
import detect
import shutil
from plot_euler_angles import plot_euler_angles

opt = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def load_angle_model(path):
    '''
    Loads a pretrained model
    '''
    model = models.mobilenet_v2(num_classes=3)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# def quat_to_euler(vec):
#     '''
#     Transforms quaternion to euler angle (in degrees)
#     '''
#     rot = Rotation.from_quat(vec.cpu())
#     eulers = rot.as_euler('xyz', degrees=True)
#     return eulers

def run_inference(opt=opt):
    '''
    Runs inference on a batch of images or a single image
    '''
    preprocess = transforms.Compose([transforms.ToTensor()])
    if os.path.isdir(opt.cropped):
        images = [os.path.join(opt.cropped, img) for img in os.listdir(opt.cropped)]
    else:
        images = [opt.cropped]
    model.eval()
    for img in images:
        input_image = Image.open(img)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        out = output[0]
        pitch, yaw, roll = torch.rad2deg(out)
        print(f"Image: {img.split('/')[-1]} has the following angles:\n Pitch: {pitch:.2f}; Yaw: {yaw:.2f}; Roll: {roll:.2f}\n")
        plot_euler_angles(pitch, yaw, roll, img)


if __name__ == '__main__':
    model = load_angle_model('./model.pth')
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/samples', help='source')  # file or directory
    parser.add_argument('--cropped', type=str, default='./data/cropped', help='cropped images')  # file or directory
    opt = parser.parse_args()
    detect.run(source=opt.source)
    run_inference(opt=opt)
    if os.path.exists('./data/cropped'):
        shutil.rmtree('./data/cropped')  # delete output folder
    os.makedirs('./data/cropped')  # make new output folders

    if os.path.exists('./data/web'):  # delete images downloaded from web if there are any
        shutil.rmtree('./data/web')  # delete output folder