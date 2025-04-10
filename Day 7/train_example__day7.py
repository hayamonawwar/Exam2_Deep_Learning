# This is a sample Python script.
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights



'''
LAST UPDATED 11/10/2021, lsdr
'''

## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep


os.chdir(OR_PATH) # Come back to the directory where the code resides , all files will be left on this directory

n_epoch = 15
BATCH_SIZE = 30
LR = 0.001
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait for improvement


## Image processing
CHANNELS = 3
IMAGE_SIZE = 380

NICKNAME = "Themisto"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
#---- Define the model ---- #

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, OUTPUTS_a)
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.act(self.conv2(self.act(x)))
        return self.linear(self.global_avg_pool(x).view(-1, 128))

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data, target_type):
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        if self.type_data == 'train':
            y = xdf_dset.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_test.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")

        if self.target_type == 2:
            labels_ohe = [int(e) for e in y]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)
            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            file = DATA_DIR + xdf_dset.id.get(ID)
        else:
            file = DATA_DIR + xdf_dset_test.id.get(ID)

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform_train(img) if self.type_data == 'train' else transform_test(img)

        return img, y

def compute_class_weights():
    """Compute positive weights for BCEWithLogitsLoss."""
    targets = xdf_dset['target_class'].apply(lambda x: list(map(int, x.split(','))))
    matrix = np.array([list(map(int, str(x).split(','))) for x in xdf_dset['target_class']])

    pos_counts = matrix.sum(axis=0)  # Count of positive labels per class
    neg_counts = matrix.shape[0] - pos_counts  # Count of negative labels per class
    class_weights = torch.tensor(neg_counts / (pos_counts + 1e-5), dtype=torch.float32)  # To prevent division by zero

    return class_weights.to(device)

from sklearn.model_selection import train_test_split

def read_data(target_type):
    global xdf_dset, xdf_dset_val, xdf_dset_test

    list_of_ids = list(xdf_dset.index)
    train_ids, val_ids = train_test_split(list_of_ids, test_size=0.15, random_state=42)

    partition = {
        'train': train_ids,
        'val': val_ids,
        'test': list(xdf_dset_test.index)
    }

    training_set = Dataset(partition['train'], 'train', target_type)
    val_set = Dataset(partition['val'], 'val', target_type)
    test_set = Dataset(partition['test'], 'test', target_type)

    loader_params = {'batch_size': BATCH_SIZE}

    training_generator = data.DataLoader(training_set, shuffle=True, **loader_params)
    val_generator = data.DataLoader(val_set, shuffle=False, **loader_params)
    test_generator = data.DataLoader(test_set, shuffle=False, **loader_params)

    return training_generator, val_generator, test_generator
def save_model(model):
    # Open the file

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

def model_definition(pretrained=True):
    if pretrained:
        weights = EfficientNet_B4_Weights.DEFAULT
        model = efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, OUTPUTS_a)
    else:
        model = CNN()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    class_weights = compute_class_weights()
    criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0)

    save_model(model)

    return model, optimizer, criterion, scheduler

def train_and_test(train_ds, val_ds, list_of_metrics, list_of_agg, save_on, pretrained = True):
    # Use a breakpoint in the code line below to debug your script.

    model, optimizer, criterion, scheduler = model_definition(pretrained)

    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = 0
    no_improve_epochs = 0

    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])
        test_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata,xtarget in train_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()

                output = model(xdata)

                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(train_hist) == 0:
                    train_hist = xtarget.cpu().numpy()
                else:
                    train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        test_loss, steps_test = 0, 0
        met_test = 0

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata,xtarget in test_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    optimizer.zero_grad()

                    output = model(xdata)

                    loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    cont += 1

                    steps_test += 1

                    test_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(test_hist) == 0:
                        tast_hist = xtarget.cpu().numpy()
                    else:
                        test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        #acc_test = accuracy_score(real_labels[1:], pred_labels)
        #hml_test = hamming_loss(real_labels[1:], pred_labels)

        avg_test_loss = test_loss / steps_test

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)


        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        print(xstrres)

        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))

            xdf_dset_results = xdf_dset_test.copy()

            # Convert prediction vectors to comma-separated strings
            xfinal_pred_labels = []
            for i in range(len(pred_labels)):
                joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results['results'] = xfinal_pred_labels
            xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

            print("✅ The model has been saved (F1 improved)!")
            met_test_best = met_test
            no_improve_epochs = 0  # reset patience
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
                print("⛔ Early stopping triggered.")
                break


def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict
def evaluate_model(model, dataloader, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    pred_logits = []
    true_labels = []

    with torch.no_grad():
        for xdata, xtarget in tqdm(dataloader, desc="Evaluating"):
            xdata, xtarget = xdata.to(device), xtarget.to(device)
            output = model(xdata)

            pred_logits.append(output.cpu().numpy())
            true_labels.append(xtarget.cpu().numpy())

    pred_logits = np.vstack(pred_logits)
    true_labels = np.vstack(true_labels)

    # Apply sigmoid then threshold
    probs = torch.sigmoid(torch.tensor(pred_logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    results = {
        "f1_macro": f1_score(true_labels, preds, average="macro"),
        "f1_micro": f1_score(true_labels, preds, average="micro"),
        "f1_weighted": f1_score(true_labels, preds, average="weighted")
    }

    return results

def process_target(target_type):
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''

    dict_target = {}
    xerror = 0

    if target_type == 2:
        ## The target comes as a string  x1, x2, x3,x4
        ## the following code creates a list
        target = np.array(xdf_data['target'].apply( lambda x : x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 1:
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names=(xtarget)
        xdf_data['target_class'] = final_target

    ## We add the column to the main dataset


    return class_names


if __name__ == '__main__':

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    ## Process Classes
    ## Input and output


    ## Processing Train dataset
    ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    class_names = process_target(target_type = 2)

    ## Comment

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    xdf_dset_test= xdf_data[xdf_data["split"] == 'test'].copy()

    ## read_data creates the dataloaders, take target_type = 2

    train_ds, val_ds, test_ds = read_data(target_type=2)  # ✅ CORRECT


    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    train_and_test(train_ds, val_ds, list_of_metrics, list_of_agg, save_on='f1_macro', pretrained=True)
    print("\nEvaluating final model on test set...")
    model, _, _, _ = model_definition(pretrained=True)
    model.load_state_dict(torch.load("model_{}.pt".format(NICKNAME), map_location=device))
    model = model.to(device)

    test_results = evaluate_model(model, test_ds)

    print("\n📊 Final Test Results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
