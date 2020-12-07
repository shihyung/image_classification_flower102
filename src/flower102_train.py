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
import yaml
import pretrainedmodels
import timm

parser=argparse.ArgumentParser()
parser.add_argument('--num_epochs',type=int,default=10,help='number of epochs for training')
parser.add_argument('--batch_size',type=int,default=16,help='size of batch for training')
parser.add_argument('--num_cpus',type=int,default=8,help='number of cpu workers')
parser.add_argument('--img_size',type=int,default=320,help='image size')
parser.add_argument('--kfd_size',type=int,default=5,help='K-Fold size')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
input_args=parser.parse_args()

hyp_param={'debug':False,
           'location':'Colab', #'Flyai', 'Colab', 'NB'
           'train':True,
           'pretrained':True,
           'cache_img':True,
           'lr':1e-4,
           'weight_decay':1e-6,
           'gradient_accumulations':1,
           'scheduler':'CosineAnnealingLR', # 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts'
           'factor':0.5, # ReduceLROnPlateau
           'patience':3, # ReduceLROnPlateau
           'eps':1e-6, # ReduceLROnPlateau
           'T_0':10, # CosineAnnealingWarmRestarts
           'min_lr':1e-6,
           'print_freq':100,
           'max_grad_norm':1000,
           'target_size':102,
           'target_col':'label',
           'seed':42,
           'model_name':'effb0', #resnet18, resnet34, resnet50, resnext50, se_resnext50
           }

#device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if hyp_param['pretrained'] and hyp_param['lr']>1e-4:
    print(f"\n\n*** Warning!  LR is {hyp_param['lr']} when loading pretrained model...")

if hyp_param['location'].lower()=='flyai':
    csv_path='/home/dataset/flower102/train.csv'
    img_path='/home/dataset/flower102/train_images'
else:
    csv_path='./train.csv'
    img_path='./train_images'

output_path='./output/'
os.makedirs(output_path,exist_ok=True)

if hyp_param['debug']:
    print('\n\n\n*** Running on Debug Mode ***\n\n\n')

with open('./output/hyp_args.yaml','w') as fout:
    dump_dict=dict()
    dump_dict['num_epochs']=input_args.num_epochs
    dump_dict['batch_size']=input_args.batch_size
    dump_dict['num_cpus']=input_args.num_cpus
    dump_dict['img_size']=input_args.img_size
    dump_dict['kfd_size']=input_args.kfd_size
    dump_dict['device']=input_args.device
    dump_dict.update(hyp_param)
    yaml.dump(dump_dict,fout,sort_keys=False)
fout.close()

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = '\n*Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %(s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

################################################################################################################################

### Dataset ###
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.img_ids = df['image_id'].values
        self.img_labels = df['label'].values
        self.transform = transform
        if hyp_param['cache_img']:
            print('\n*** Loading images into cache...')
            self.img_cache=[]
            t1=time.time()
            for idx in tqdm.tqdm(range(self.df.shape[0])):
                image_name = self.img_ids[idx]
                image = cv2.imread(f'{img_path}/{image_name}')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.img_cache.append(image)
            t2=time.time()
            print(f'*time: {int(t2-t1)}s')
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if hyp_param['cache_img']:
            image=self.img_cache[idx]
        else:
            image_name = self.img_ids[idx]
            image = cv2.imread(f'{img_path}/{image_name}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.img_labels[idx]).long()
        return image, label

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
            albumentations.CoarseDropout (max_holes=30, max_height=5, max_width=5, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
            albumentations.HueSaturationValue (hue_shift_limit=0.2, sat_shift_limit=0.3, val_shift_limit=0.2, p=0.5),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ])

    elif mode == 'valid':
        return albumentations.Compose([
            albumentations.Resize(input_args.img_size, input_args.img_size),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ])

### average ###
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
        elif hyp_param['model_name']=='resnext50':
            print('Model: < Resnext50_32x4d >')
            self.model=torchvision.models.resnext50_32x4d(pretrained=pretrained)
            self.model.fc=torch.nn.Linear(self.model.fc.in_features, hyp_param['target_size'])
        elif hyp_param['model_name']=='resnet50':
            print('Model: < Resnet50 >')
            self.model=torchvision.models.resnet50(pretrained=pretrained)
            self.model.fc=torch.nn.Linear(self.model.fc.in_features, hyp_param['target_size'])
        elif hyp_param['model_name']=='resnet34':
            print('Model: < Resnet34 >')
            self.model=torchvision.models.resnet34(pretrained=pretrained)
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

def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()
    preds = []
    tgt_lbs=[]
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(images)
        bloss = criterion(y_preds, labels)
        # record loss
        losses.update(bloss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').detach().numpy())
        tgt_lbs.append(labels.to('cpu').numpy())
        if hyp_param['gradient_accumulations'] > 1:
            bloss = bloss / hyp_param['gradient_accumulations']
        bloss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_param['max_grad_norm'])
        if (step + 1) % hyp_param['gradient_accumulations'] == 0:
            optimizer.step()
            optimizer.zero_grad()
        if (step+1) % hyp_param['print_freq'] == 0 or step == (len(train_loader)-1):
            print(f'Train - Batch: {step+1}/{len(train_loader)}, CELoss_avg: {round(losses.avg,5)}')
    predictions = np.concatenate(preds)
    train_labels=np.concatenate(tgt_lbs)
    acc_score = accuracy_score(train_labels, predictions.argmax(1))

    '''
    print('\n*Train Confusion Matrix:\n')
    cfm=confusion_matrix(train_labels,predictions.argmax(1))
    for ii in range(hyp_param['target_size']):
        for jj in range(hyp_param['target_size']):
            print('{:>5} '.format(int(cfm[ii][jj])),end='')
        print('\n')
    '''

    return losses.avg, acc_score, predictions

def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    scores = AverageMeter()

    model.eval()
    preds = []
    tgt_lbs=[]
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        bloss = criterion(y_preds, labels)
        losses.update(bloss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').numpy())
        tgt_lbs.append(labels.to('cpu').numpy())
        if hyp_param['gradient_accumulations'] > 1:
            bloss = bloss / hyp_param['gradient_accumulations']
        if (step+1) % hyp_param['print_freq'] == 0 or step == (len(valid_loader)-1):
            print(f'Valid - Batch: {step+1}/{len(valid_loader)}, CELoss_avg: {round(losses.avg,5)}')
    predictions = np.concatenate(preds)
    valid_labels=np.concatenate(tgt_lbs)
    acc_score = accuracy_score(valid_labels, predictions.argmax(1))

    '''
    print('\n*Valid Confusion Matrix:\n')
    cfm=confusion_matrix(valid_labels,predictions.argmax(1))
    for ii in range(hyp_param['target_size']):
        for jj in range(hyp_param['target_size']):
            print('{:>5} '.format(int(cfm[ii][jj])),end='')
        print('\n')
    '''

    return losses.avg, acc_score, predictions

def Fold_model_train_valid(folds_df, fold):
    tb_writer = SummaryWriter(f'{output_path}fold{fold}/')

    # dataset & dataloader
    train_idx = folds_df[folds_df['fold'] != fold].index
    valid_idx = folds_df[folds_df['fold'] == fold].index

    train_folds_df = folds_df.loc[train_idx].reset_index(drop=True)
    valid_folds_df = folds_df.loc[valid_idx].reset_index(drop=True)

    train_dataset = TrainDataset(train_folds_df, transform=get_transforms(mode='train'))
    valid_dataset = TrainDataset(valid_folds_df, transform=get_transforms(mode='valid'))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=input_args.batch_size, shuffle=True, num_workers=input_args.num_cpus, pin_memory=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=input_args.batch_size, shuffle=False, num_workers=input_args.num_cpus, pin_memory=True, drop_last=False)

    # model
    model = nn_model(pretrained=hyp_param['pretrained'])
    model.to(device)

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp_param['lr'], weight_decay=hyp_param['weight_decay'], amsgrad=False)
    #optimizer = torch.optim.SGD(model.parameters(), lr=hyp_param['lr'],  momentum=0.937, nesterov=True)
    if hyp_param['scheduler']=='ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=hyp_param['factor'], patience=hyp_param['patience'], verbose=True, eps=hyp_param['eps'])
    elif hyp_param['scheduler']=='CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=input_args.num_epochs, eta_min=hyp_param['min_lr'], last_epoch=-1)
    elif hyp_param['scheduler']=='CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hyp_param['T_0'], T_mult=1, eta_min=hyp_param['min_lr'], last_epoch=-1)

    #loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    #Begin Training
    best_valid_acc_score = 0.
    best_epoch=-1
    best_loss = np.inf
    for epoch in range(input_args.num_epochs):
        if hyp_param['scheduler']=='ReduceLROnPlateau':
            print('\n{} Epoch: {} / {}, Learning rate: {} {}'.format('*'*16,epoch,input_args.num_epochs-1,get_lr(optimizer),'*'*16))
        else:
            print('\n{} Epoch: {} / {}, Learning rate: {} {}'.format('*'*16,epoch,input_args.num_epochs-1,scheduler.get_last_lr(),'*'*16))
        epoch_time_start=time.time()

        # train
        train_avg_loss, train_acc_score, train_predictions = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        train_labels = train_folds_df[hyp_param['target_col']].values

        # valid
        valid_avg_loss, valid_acc_score, valid_predictions = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds_df[hyp_param['target_col']].values
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_avg_loss)
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()

        if valid_acc_score > best_valid_acc_score:
            print('\n* Better accuracy score! Saving model...')
            best_valid_acc_score = valid_acc_score
            torch.save({'model': model.state_dict(), 'valid_preds': valid_predictions},output_path+f'CLD_fold{fold}_best.pth')
            best_epoch=epoch
        
        #Tensorboard output
        tb_writer.add_scalar('Learning_rate', get_lr(optimizer), epoch)
        tb_writer.add_scalar('Loss_avg/train', train_avg_loss, epoch)
        tb_writer.add_scalar('Loss_avg/valid', valid_avg_loss, epoch)
        tb_writer.add_scalar('accuracy/train', train_acc_score, epoch)
        tb_writer.add_scalar('accuracy/valid', valid_acc_score, epoch)
        print(f'Accuracy Score: train({round(train_acc_score,5)}) / valid({round(valid_acc_score,5)})')

        epoch_time_end=time.time()
        hours, rem = divmod(epoch_time_end-epoch_time_start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time consumption: {:0>2} : {:0>2} : {:05.2f}".format(int(hours),int(minutes),seconds))
        
    print(f'\n*{fold}-fold, {best_epoch}-epoch, best accuracy: {best_valid_acc_score}\n')
    check_point = torch.load(output_path+f'CLD_fold{fold}_best.pth')
    valid_folds_df[[str(c) for c in range(hyp_param['target_size'])]] = check_point['valid_preds']
    valid_folds_df['preds'] = check_point['valid_preds'].argmax(1)

    return valid_folds_df
################################################################################################################################

def main():
    print('\n*** CLD Training ***\n')

    ### fix random seeds ###
    print('* Fixing random\n')
    fix_random()

    ### load train.csv ###
    print('* Loading CSV file\n')
    df_train=pd.read_csv(csv_path)
    if hyp_param['debug']:
        input_args.num_epochs = 3
        df_train = df_train.sample(n=2000, random_state=hyp_param['seed']).reset_index(drop=True)

    ### K-Fold split ###
    print('* K-Fold spliting\n')
    df_folds = df_train.copy()
    skfd=StratifiedKFold(n_splits=input_args.kfd_size, shuffle=True, random_state=hyp_param['seed'])
    for n, (train_index, valid_index) in enumerate(skfd.split(df_folds, df_folds[hyp_param['target_col']])):
        df_folds.loc[valid_index, 'fold'] = int(n)
    df_folds['fold'] = df_folds['fold'].astype(int)
    
    ### k-fold Training Model ###
    print('* K-Fold Training\n')
    if hyp_param['train']:
        df_oof_result = pd.DataFrame()
        for fold_num in range(input_args.kfd_size):
            if fold_num>0:
                continue
            print(f'\n*** Begin {fold_num+1}th-Fold train/valid ***')

            df_fold_result = Fold_model_train_valid(df_folds, fold_num)
            df_oof_result = pd.concat([df_oof_result, df_fold_result])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        df_oof_result.to_csv(output_path+'OOF_result.csv', index=False)

################################################################################################################################

device = select_device(input_args.device, batch_size=input_args.batch_size)
if __name__ == '__main__':
    main()
