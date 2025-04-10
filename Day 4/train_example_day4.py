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


OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep


os.chdir(OR_PATH) # Come back to the directory where the code resides , all files will be left on this directory

n_epoch = 25
BATCH_SIZE = 30
LR = 0.001

CHANNELS = 3
IMAGE_SIZE = 100

NICKNAME = "Themisto"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

# Data augmentation with more transformations
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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


def read_data(target_type):
    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = xdf_dset['target_class']

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)

    partition = {
        'train': list_of_ids,
        'test': list_of_ids_test
    }

    params = {'batch_size': BATCH_SIZE, 'shuffle': True}
    training_set = Dataset(partition['train'], 'train', target_type)
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': BATCH_SIZE, 'shuffle': False}
    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    return training_generator, test_generator


def save_model(model):
    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))


def model_definition(pretrained=True):
    if pretrained == True:
        model = models.resnet50(pretrained=True)  # Use a deeper pretrained model
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)  # Adjust the final layer for your output
    else:
        model = CNN()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Get class weights for the loss function
    class_weights = compute_class_weights()

    criterion = nn.BCEWithLogitsLoss(weight=class_weights)  # Applying weighted BCE loss

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0)

    save_model(model)

    return model, optimizer, criterion, scheduler


def compute_class_weights():
    """Compute positive weights for BCEWithLogitsLoss."""
    targets = xdf_dset['target_class'].apply(lambda x: list(map(int, x.split(','))))
    matrix = np.array([list(map(int, str(x).split(','))) for x in xdf_dset['target_class']])

    pos_counts = matrix.sum(axis=0)  # Count of positive labels per class
    neg_counts = matrix.shape[0] - pos_counts  # Count of negative labels per class
    class_weights = torch.tensor(neg_counts / (pos_counts + 1e-5), dtype=torch.float32)  # To prevent division by zero

    return class_weights.to(device)



def find_optimal_thresholds(model, val_loader, num_classes):
    best_thresholds = []

    # Iterate over each class
    for class_idx in range(num_classes):
        best_f1 = 0
        best_threshold = 0.5  # Default starting point

        # Try different threshold values from 0 to 1
        for threshold in np.linspace(0, 1, 100):
            y_true = []
            y_pred = []

            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)

                    # Convert outputs to probabilities
                    output_probs = torch.sigmoid(output).cpu().numpy()

                    # Get predictions based on the current threshold
                    predicted = (output_probs[:, class_idx] >= threshold).astype(int)

                    y_true.extend(labels[:, class_idx].cpu().numpy())
                    y_pred.extend(predicted)

            # Calculate F1 score for the current threshold
            f1 = f1_score(y_true, y_pred)

            # Track the best threshold based on the highest F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        best_thresholds.append(best_threshold)

    return best_thresholds


def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on, pretrained=True):
    model, optimizer, criterion, scheduler = model_definition(pretrained)

    cont = 0
    train_loss_item = list([])

    model.phase = 0
    met_test_best = 0
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:
            for xdata, xtarget in train_ds:
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

                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        test_loss, steps_test = 0, 0
        met_test = 0

        with torch.no_grad():
            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:
                for xdata, xtarget in test_ds:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    output = model(xdata)

                    loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    cont += 1
                    steps_test += 1

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        # Convert predictions to binary (0 or 1) using the threshold
        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation
        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_test_loss = test_loss / steps_test

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test ' + met + ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        print(xstrres)

        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
            print("The model has been saved!")
            met_test_best = met_test

def metrics_func(metrics, aggregates, y_true, y_pred):
    def f1_score_metric(y_true, y_pred, type):
        res = f1_score(y_true, y_pred, average=type)
        return res

    xcont = 1
    xsum = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_macro':
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        res_dict[xm] = xmet

    return res_dict

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
    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    xdf_data = pd.read_excel(FILE_NAME)

    class_names = process_target(target_type=2)

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()

    train_ds, test_ds = read_data(target_type=2)

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on='f1_macro', pretrained=True)
