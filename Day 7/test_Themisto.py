#------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision import models
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#------------------------------------------------------------------------------------------------------------------
'''
LAST UPDATED 11/10/2021, lsdr
02/14/2022 am lsdr check for consistency
02/14/2022 pm lsdr check numpy() to cpu().numpy() in target, 
              model = model.to(device) was commented.
'''
#------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
parser.add_argument("--split", default=False, type=str, required=True)  # validate, test, train

args = parser.parse_args()

PATH = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT = args.split

BATCH_SIZE = 30
LR = 0.001

## Image processing
CHANNELS = 3
IMAGE_SIZE = 380

NICKNAME = "Themisto"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

#------------------------------------------------------------------------------------------------------------------
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


#------------------------------------------------------------------------------------------------------------------
## ------------------ Data Loader definition-----------------------------------------------------------------------
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

        file = DATA_DIR + xdf_dset_test.id.get(ID)

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform_test(img)

        return img, y

#------------------------------------------------------------------------------------------------------------------

def read_data(target_type):

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids_test = list(xdf_dset_test.index)


    # Datasets
    partition = {
        'test' : list_of_ids_test
    }

    # Data Loader

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return test_generator
#------------------------------------------------------------------------------------------------------------------
def compute_class_weights():
    """Compute positive weights for BCEWithLogitsLoss."""
    targets = xdf_data['target_class'].apply(lambda x: list(map(int, x.split(','))))
    matrix = np.array([list(map(int, str(x).split(','))) for x in xdf_data['target_class']])

    pos_counts = matrix.sum(axis=0)  # Count of positive labels per class
    neg_counts = matrix.shape[0] - pos_counts  # Count of negative labels per class
    class_weights = torch.tensor(neg_counts / (pos_counts + 1e-5), dtype=torch.float32)  # To prevent division by zero

    return class_weights.to(device)

def model_definition(pretrained=True):
    if pretrained:
        weights = EfficientNet_B4_Weights.DEFAULT
        model = efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, OUTPUTS_a)
    else:
        model = CNN()

    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
    model = model.to(device)
    # Get class weights for the loss function
    class_weights = compute_class_weights()

    criterion = nn.BCEWithLogitsLoss(weight=class_weights)

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))  # save model summary to disk

    return model, criterion

#------------------------------------------------------------------------------------------------------------------

def test_model(test_ds, list_of_metrics, list_of_agg , pretrained = True):
    # Use a breakpoint in the code line below to debug your script.

    model, criterion  = model_definition(pretrained)
    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))

    cont = 0
    test_loss_item = list([])
    test_hist=list([])

    pred_labels_per_hist = list([])

    test_loss = 0
    steps_test = 0
    pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

    model.eval()

    with torch.no_grad():

        with tqdm(total=len(test_ds), desc="Testing the Model") as pbar:

            for xdata,xtarget in test_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)

                output = model(xdata)

                loss = criterion(output, xtarget)

                test_loss += loss.item()
                cont += 1

                steps_test += 1

                test_loss_item.append([ loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).cpu().numpy()

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

        avg_test_loss = test_loss / steps_test

        xstrres = ''

        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)

        print(xstrres) #Print metrics



        xfinal_pred_labels = []
        for i in range(len(pred_labels)):
            joined_string = ",".join(str(int(e)) for e in pred_labels[i])
            xfinal_pred_labels.append(joined_string)

        xdf_dset_test['results'] = xfinal_pred_labels
        xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

        for i in range(OUTPUTS_a): # F1 score and confusion matrix for each class
            y_true_i = real_labels[1:, i]
            y_pred_i = pred_labels[:, i]
            # cm = confusion_matrix(y_true_i, y_pred_i)
            f1 = f1_score(y_true_i, y_pred_i, zero_division=0)
            print(f"F1 Score for Class {i}: {f1:.4f}")
            # plt.figure(figsize=(4, 3))
            # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            # plt.title(f'Confusion Matrix for Class {i}')
            # plt.xlabel('Predicted')
            # plt.ylabel('Actual')
            # plt.tight_layout()
            # plt.show()


#------------------------------------------------------------------------------------------------------------------


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

    xcont = 0
    xsum = 0
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
#------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Multiclass or Multilabel ( binary  ( 0,1 ) )
    :return:
    '''

    dict_target = {}
    xerror = 0

    if target_type == 1:
        # takes the classes and then
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names=(xtarget)
        xdf_data['target_class'] = final_target


    if target_type == 2:
        target = np.array(xdf_data['target'].apply( lambda x : x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 3:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = mlb.classes_

    ## We add the column to the main dataset


    return class_names
#------------------------------------------------------------------------------------------------------------------

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

    ## Balancing classes , all groups have the same number of observations
    xdf_dset_test= xdf_data[xdf_data["split"] == SPLIT].copy()

    ## read_data creates the dataloaders, take target_type = 2

    test_ds = read_data(target_type = 2)

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    test_model(test_ds, list_of_metrics, list_of_agg, pretrained=True)