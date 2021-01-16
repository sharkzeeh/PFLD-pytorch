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
from pfld.utils import *
from pfld.pfld import PFLDInference, AuxiliaryNet
from plot_euler_angles import plot_euler_angles

opt = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def load_angle_model(path):
    '''
    Loads a pretrained model
    '''
    checkpoint = torch.load(path, map_location=device)
    model = PFLDInference().to(device)
    model.load_state_dict(checkpoint['plfd_backbone'])
    return model


def run_inference(model, opt=opt):
    '''
    Runs inference on a batch of images or a single image
    '''
    preprocess = transforms.Compose([transforms.ToTensor()])
    if os.path.isdir(opt.cropped):
        images = [os.path.join(opt.cropped, img) for img in os.listdir(opt.cropped)]
    else:
        images = [opt.cropped]
    model.eval()
    model.to(device)
    for img_name in images:
        img = Image.open(img_name)
        img = np.asarray(img)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = preprocess(img)
        batch = tensor.unsqueeze(0)
        if torch.cuda.is_available():
            batch = batch.to('cuda')

        with torch.no_grad():
            _, landmarks = model(batch)

        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
        point_dict = {}
        i = 0
        for (x,y) in pre_landmark.astype(np.float32):
            point_dict[f'{i}'] = [x,y]
            i += 1

        # yaw
        point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
        point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
        point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
        crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
        yaw_mean = point_point(point1, point31) / 2
        yaw_right = point_point(point1, crossover51)
        yaw = (yaw_mean - yaw_right) / yaw_mean
        yaw = yaw * 71.58 + 0.7037

        # pitch
        pitch_dis = point_point(point51, crossover51)
        if point51[1] < crossover51[1]:
            pitch_dis = -pitch_dis
        pitch = 1.497 * pitch_dis + 18.97

        # roll
        roll_tan = abs(get_num(point_dict,60,1) - get_num(point_dict,72,1)) / abs(get_num(point_dict,60,0) - get_num(point_dict,72,0))
        roll = math.atan(roll_tan)
        roll = math.degrees(roll)
        if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
            roll = -roll

        print(f"Image: {img_name.split('/')[-1]} has the following angles:\n Pitch: {pitch:.2f}; Yaw: {yaw:.2f}; Roll: {roll:.2f}\n")
        plot_euler_angles(pitch, yaw, roll, img_name)


if __name__ == '__main__':
    model = load_angle_model('./model.pth')
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/samples', help='source')  # file or directory
    parser.add_argument('--cropped', type=str, default='./data/cropped', help='cropped images')  # file or directory
    opt = parser.parse_args()
    detect.run(source=opt.source)
    run_inference(model=model, opt=opt)
    if os.path.exists('./data/cropped'):
        shutil.rmtree('./data/cropped')  # delete output folder
    os.makedirs('./data/cropped')  # make new output folders

    if os.path.exists('./data/web'):  # delete images downloaded from web if there are any
        shutil.rmtree('./data/web')  # delete output folder