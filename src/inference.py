import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
import cv2
import PIL
from matplotlib import pyplot as plt
import seaborn as sns
import albumentations
from albumentations.pytorch import ToTensorV2
import os
import random
import tqdm
import time
import argparse
import pretrainedmodels
import timm


parser=argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=1,help='size of batch for training')
parser.add_argument('--num_cpus',type=int,default=1,help='number of cpu workers')
parser.add_argument('--img_size',type=int,default=160,help='image size') ######
parser.add_argument('--kfd_size',type=int,default=1,help='K-Fold size') ######
parser.add_argument('--tta',type=int,default=3,help='TTA size') ######
input_args=parser.parse_args()

hyp_param={'debug':False,
           'location':'nb', #'Flyai', 'Colab', 'NB', 'kaggle'
           'train':False,
           'pretrained':False,
           'target_size':102, #######
           'target_col':'label',
           'seed':42,
           'model_name':'se_resnext50', #resnet18, resnet50, se_resnext50, effb0 ######
           }

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

csv_path='./test.csv'
img_path='./test_images'
model_path='./results/img160_se_resnext50_pretrained/' ######

################################################################################################################################

### Sub-Functions ###

def fix_random():
    seed_num=hyp_param['seed']
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

################################################################################################################################

### Dataset ###
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.img_ids = df['image_id'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        image = cv2.imread(f'{img_path}/{image_name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

### Transforms ###
def get_transforms(mode):
    if mode == 'train':
        return albumentations.Compose([
            albumentations.Resize(input_args.img_size, input_args.img_size),
            #albumentations.RandomResizedCrop(input_args.img_size, input_args.img_size),
            #albumentations.RandomCrop(input_args.img_size, input_args.img_size),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            #albumentations.HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
            albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],), #mean: [0.43032, 0.49673, 0.31342] std: [0.21909, 0.223943, 0.20059]
            ToTensorV2(),
        ])
    elif mode == 'valid':
        return albumentations.Compose([
            albumentations.Resize(input_args.img_size, input_args.img_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ])

################################################################################################################################

### NN model ###
class nn_model(torch.nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        if pretrained:
            print('\n* Loading pretrained model...')
        if hyp_param['model_name']=='se_resnext50':
            print('Model: < SE_Resnext50_32x4d >')
            self.model=pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet' if pretrained else None)
            self.model.avg_pool=torch.nn.AdaptiveAvgPool2d((1,1))
            self.model.last_linear=torch.nn.Linear(self.model.last_linear.in_features, hyp_param['target_size'])
        elif hyp_param['model_name']=='resnet50':
            print('Model: < Resnet50 >')
            self.model=torchvision.models.resnet50(pretrained=pretrained)
            self.model.fc=torch.nn.Linear(self.model.fc.in_features, hyp_param['target_size'])
        elif hyp_param['model_name']=='resnet18':
            print('Model: < Resnet18 >')
            self.model=torchvision.models.resnet18(pretrained=pretrained)
            self.model.fc=torch.nn.Linear(self.model.fc.in_features, hyp_param['target_size'])
        elif hyp_param['model_name']=='effb0':
            print('Model: < EfficientNet_b0 >')
            self.model=timm.create_model('tf_efficientnet_b0',pretrained=pretrained)
            self.model.classifier=torch.nn.Linear(self.model.classifier.in_features,hyp_param['target_size'],bias=True)
        else:
            print('@@ Wrong model name...')
            exit()
    
    def forward(self, x):
        x = self.model(x)
        return x

################################################################################################################################

def main():
    print('\n*** CLD Inference ***\n')

    ### fix random seeds ###
    print('* Fixing random\n')
    fix_random()

    ### load train.csv ###
    print('* Loading CSV file\n')
    df_test=pd.read_csv(csv_path)
    test_dataset = TestDataset(df_test, transform=get_transforms(mode='valid'))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=input_args.batch_size, shuffle=False, num_workers=input_args.num_cpus, pin_memory=True)

    model = nn_model()
    model.to(device)
    acc_result=np.zeros((len(test_dataset),hyp_param['target_size']))
    df_test['predict']=-1
    for fold in range(input_args.kfd_size):
        print(f'*Loading {fold}_th fold model...')
        ckpt=torch.load(model_path+f'CLD_fold{fold}_best.pth')
        pnames=list(ckpt['model'].keys())
        for name in pnames:
            if name[:7]=='module.':
                new_name=name[7:]
                ckpt['model'][new_name]=ckpt['model'].pop(name)
        model.load_state_dict(ckpt['model'])
        model.eval()
        print(' *inferencing...')
        for tta in range(input_args.tta):
            print('  *TTA: ',tta)
            for idx,image in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
                image = image.to(device)
            
                with torch.no_grad():
                    y_pred = model(image)
                    acc_result[idx]+=y_pred.softmax(1).to('cpu').numpy()[0]
    
    for idx in range(len(test_dataset)):
        df_test.loc[idx,'predict']=acc_result[idx].argmax()
    acc_score = accuracy_score(df_test['label'], df_test['predict'])
    print('Accuracy Score: ',round(acc_score,5))


################################################################################################################################

if __name__ == '__main__':
    main()
