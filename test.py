import pandas as pd
import os
from DataLoader import *
from torchvision import transforms
import torch.nn as nn
import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dino import DinoWithFC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from train import *


def test(weights_path, data_folder, num_classes=1, dino=False):
    all_xray_df = pd.read_csv("./data/Data_Entry_2017.csv")  # read datafile
    all_xray_df = all_xray_df[(all_xray_df['Patient Age'] >= 20) & (all_xray_df['Patient Age'] < 70)]  # only keep data for ages 20 - 70
    all_xray_df = all_xray_df.drop_duplicates(subset='Patient ID', keep='first')  # remove images from the same person
    all_image_paths = {os.path.basename(x): x for x in glob(os.path.join(data_folder, '*.png'))}  # read image paths

    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

    # split data:
    train_df, valid_df, test_df = np.split(all_xray_df.sample(frac=1), [int(.6*len(all_xray_df)), int(.8*len(all_xray_df))])
    print('train', train_df.shape[0], 'validation', valid_df.shape[0], 'test', test_df.shape[0])

    # fix leakage between data sets:
    train_df, valid_df = fix_leakage(train_df, valid_df, data1_name="training", data2_name="validation")
    train_df, test_df = fix_leakage(train_df, test_df, data1_name="training", data2_name="test")
    test_df, valid_df = fix_leakage(test_df, valid_df, data1_name="test", data2_name="validation")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dsetTest = CustomDataset(test_df, test_transform, num_classes=num_classes)

    testloader = torch.utils.data.DataLoader(dataset=dsetTest, batch_size=64, shuffle=False, num_workers=8)

    model = models.densenet121(pretrained=False)  # Don't load default pretrained weights
    if dino:
        model = DinoWithFC(num_classes=num_classes)

    checkpoint = torch.load(weights_path, map_location='cuda:0')

    # get state dict from checkpoint:
    state_dict = checkpoint['model_state_dict']
    if num_classes>1: # classification
        # Replace the classifier with a new FC layer for age prediction with multiple classes
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)  # Output probabilities for each of the classes
        criterion = nn.CrossEntropyLoss()
    else:# regression
        # Replace the classifier with a new FC layer for age prediction
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1)  # Output a single value for age
        criterion = nn.L1Loss()

    # load checkpoint and send to device:
    model.load_state_dict(state_dict, strict=True)  # Use strict=False to ignore non-matching keys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is: {device}")
    model.to(device)

    # test phase:
    model.eval()  # Set the model to evaluation mode
    test_score = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        epoch_preds = []
        epoch_ages = []
        for images, ages in testloader:
            if num_classes>1:
                images, ages = images.to(device), ages.to(device)  # For classification
            else:
                images, ages = images.to(device), ages.to(device).float().view(-1, 1) # for regression
	    
            outputs = model(images)
            if num_classes>1: # calculate accuracy for classification
                _, prediction = torch.max(outputs, 1)
                test_accuracy += (prediction == ages).sum().item()
                epoch_preds.extend(prediction.cpu().detach().numpy().tolist())
                epoch_ages.extend(ages.cpu().detach().numpy().tolist())

            loss = criterion(outputs, ages)
            test_score += loss.item() * images.size(0)
        test_epoch_score = test_score / len(testloader.sampler)

        if num_classes>1:
            test_epoch_accuracy = test_accuracy / len(testloader.sampler)
            ## confusion matrix:
            cm = confusion_matrix(epoch_ages, epoch_preds)
            heatmap_plot = sns.heatmap(cm,annot=True,fmt='g')#.get_figure()
            plt.xlabel('Prediction', fontsize=13)
            plt.ylabel('Actual', fontsize=13)
            plt.title(f'Confusion Matrix', fontsize=17)
            # plt.show()

        print(f'test score'
              f': {test_epoch_score:.6f}')
        if num_classes>1:
            print(f'test accuracy'
                  f': {test_epoch_accuracy:.6f}')

test("./runs/best/checkpoint", "./resized_images_224", num_classes=1, dino=False)
