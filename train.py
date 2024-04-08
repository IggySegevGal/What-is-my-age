
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
from sklearn.manifold import TSNE


def fix_leakage(train_df, valid_df, data1_name="training", data2_name="validation"):
    ids_train = train_df['Patient ID'].values
    ids_valid = valid_df['Patient ID'].values
    # Create a "set" datastructure of the training set id's to identify unique id's
    ids_train_set = set(ids_train)
    # print(f'There are {len(ids_train_set)} unique Patient IDs in the {data1_name} set')
    # Create a "set" datastructure of the validation set id's to identify unique id's
    ids_valid_set = set(ids_valid)
    # print(f'There are {len(ids_valid_set)} unique Patient IDs in the {data2_name} set')
    # Identify patient overlap by looking at the intersection between the sets
    patient_overlap = list(ids_train_set.intersection(ids_valid_set))
    n_overlap = len(patient_overlap)
    train_overlap_idxs = []
    valid_overlap_idxs = []
    for idx in range(n_overlap):
        train_overlap_idxs.extend(train_df.index[train_df['Patient ID'] == patient_overlap[idx]].tolist())
        valid_overlap_idxs.extend(valid_df.index[valid_df['Patient ID'] == patient_overlap[idx]].tolist())
    # Drop the overlapping rows from the validation set
    valid_df.drop(valid_overlap_idxs, inplace=True)
    return train_df, valid_df


def training(title, block_to_unfreeze,pretrained, learning_rate, batch_size,  num_epochs, data_folder, num_classes=1, data_frac=1, dino=False):
    run_name = f"{title}_learning_rate{learning_rate}_num_epochs_{num_epochs}_blocks_to_unfreeze_{block_to_unfreeze}_pretrained_{pretrained}_numClasses_{num_classes}_dataFrac_{data_frac}_dino_{dino}"
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"data_folder: {data_folder}")
    print(f"run_name: {run_name}")
    print(f"blocks to unfreeze: {block_to_unfreeze}")
    writer = SummaryWriter(f'runs/{run_name}')
    best_val_score = float('inf')
    # DATA READING
    all_xray_df = pd.read_csv("./data/Data_Entry_2017.csv")  # read datafile
    all_xray_df = all_xray_df[(all_xray_df['Patient Age'] >= 20) & (all_xray_df['Patient Age'] < 70)]  # only keep data for ages 20 - 70
    all_xray_df = all_xray_df.drop_duplicates(subset='Patient ID', keep='first')  # remove images from the same person
    # plt.hist(all_xray_df['Patient Age'], bins=11, range=(20,75), edgecolor='black')
    # plt.show()
    all_image_paths = {os.path.basename(x): x for x in glob(os.path.join(data_folder, '*.png'))}  # read image paths

    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

    # # normalize 0
    # all_xray_df["Patient Age"].max() - all_xray_df["Patient Age"].min())
    # if not classification:
    #     min_age = all_xray_df["Patient Age"].min()
    #     max_age = all_xray_df["Patient Age"].max()
    #     all_xray_df["Patient Age"] = (all_xray_df["Patient Age"] - min_age) / (max_age - min_age)

    # split data to (train, val,test):
    # train_df, test_df, _, _ = train_test_split(all_xray_df, all_xray_df, test_size=0.2, random_state=1)
    #
    # train_df, valid_df, _, _ = train_test_split(train_df, train_df, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
    train_df, valid_df, test_df = np.split(all_xray_df.sample(frac=1), [int(.6*len(all_xray_df)), int(.8*len(all_xray_df))])
    train_df = train_df.sample(frac=data_frac)
    print('train', train_df.shape[0], 'validation', valid_df.shape[0], 'test', test_df.shape[0])

    # fix leakage between data sets:
    train_df, valid_df = fix_leakage(train_df, valid_df, data1_name="training", data2_name="validation")
    train_df, test_df = fix_leakage(train_df, test_df, data1_name="training", data2_name="test")
    test_df, valid_df = fix_leakage(test_df, valid_df, data1_name="test", data2_name="validation")

    train_transform = transforms.Compose([
        # transforms.RandomRotation(10),
        # transforms.RandomResizedCrop(224, scale=(0.8, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        # transforms.Resize(230),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dsetTrain = CustomDataset(train_df, train_transform, num_classes=num_classes)
    dsetVal = CustomDataset(valid_df, test_transform, num_classes=num_classes)
    dsetTest = CustomDataset(test_df, test_transform, num_classes=num_classes)

    trainloader = torch.utils.data.DataLoader(dataset=dsetTrain, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(dataset=dsetVal, batch_size=batch_size, shuffle=False, num_workers=8)
    testloader = torch.utils.data.DataLoader(dataset=dsetTest, batch_size=batch_size, shuffle=False, num_workers=8)

    # Initialize a DenseNet121 model
    model = models.densenet121(pretrained=pretrained)  # Don't load default pretrained weights
    # Load your custom trained weights

    # Load the state dictionary into your model
    if not pretrained:
        weights_path = './m-30012020-104001.pth.tar'  # Update this path
        checkpoint = torch.load(weights_path, map_location='cuda:0')  # Adjust map_location as needed
        # If the checkpoint contains a key named 'state_dict', extract it
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # change classifier size to chesxnet size:
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 14)

        # Modify the model
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("densenet121.", "")
            if name == "classifier.0.weight" or name == "classifier.0.bias":
                name = name.replace(".0", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)  # Use strict=False to ignore non-matching keys

    if num_classes>1:
        # Replace the classifier with a new FC layer for age prediction
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)  # Output probabilities for each of the classes
        criterion = nn.CrossEntropyLoss()
    else:
        # Replace the classifier with a new FC layer for age prediction
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1)  # Output a single value for age
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
    # Freeze all layers except the classifier
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    if block_to_unfreeze >= 1:  # unfreeze one block - block 4
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True
    if block_to_unfreeze >= 2:  # unfreeze 2 blocks - blocks 3,4
        for param in model.features.denseblock3.parameters():
            param.requires_grad = True
        for param in model.features.transition3.parameters():
            param.requires_grad = True
    if block_to_unfreeze >= 3:  # unfreeze 3 blocks - blocks 2,3,4
        for param in model.features.denseblock2.parameters():
            param.requires_grad = True
        for param in model.features.transition2.parameters():
            param.requires_grad = True


    ## if we want to use dino, just load the dino model instead
    if dino:
        model = DinoWithFC(num_classes=num_classes)
        for param in model.transformer.parameters():
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is: {device}")
    model.to(device)

    # optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    #### train loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for images, ages in trainloader:
            if num_classes > 1:
                images, ages = images.to(device), ages.to(device)  # For classification, keep ages as a 1D tensor.
            else:
                images, ages = images.to(device), ages.to(device).float().view(-1, 1)  # Assuming ages is a 1D array

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(trainloader.sampler)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_score = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            correct = 0
            total = 0
            epoch_preds = []
            epoch_ages = []
            for images, ages in valloader:
                if num_classes>1:
                    images, ages = images.to(device), ages.to(device)  # For classification, keep ages as a 1D tensor.
                else:
                    images, ages = images.to(device), ages.to(device).float().view(-1, 1)  # Assuming ages is a 1D array
                outputs = model(images)
                if num_classes>1:
                    if num_classes == 50:
                        _, prediction = torch.topk(outputs,5, 1)
                        val_accuracy += np.sum([ages[i] in prediction[i] for i in range(len(ages))])
                        # take median/max pred for confusion matrix
                        preds, _ = torch.max(prediction, 1)
                        epoch_preds.extend(preds.cpu().detach().numpy().tolist())
                        epoch_ages.extend(ages.cpu().detach().numpy().tolist())
                    else:
                        _, prediction = torch.max(outputs, 1)
                        val_accuracy += (prediction == ages).sum().item()
                        epoch_preds.extend(prediction.cpu().detach().numpy().tolist())
                        epoch_ages.extend(ages.cpu().detach().numpy().tolist())

                loss = criterion(outputs, ages)
                val_score += loss.item() * images.size(0)

            if num_classes>1:
                val_epoch_accuracy = 100 * val_accuracy / len(valloader.sampler)
                writer.add_scalar('accuracy/validation', val_epoch_accuracy, epoch)
                ## confusion matrix:
                cm = confusion_matrix(epoch_ages, epoch_preds)
                heatmap_plot = sns.heatmap(cm)#,annot=True,fmt='g'
                plt.xlabel('Prediction', fontsize=13)
                plt.ylabel('Actual', fontsize=13)
                plt.title(f'Confusion Matrix, epoch: {epoch}', fontsize=17)
                # plt.show()
            val_epoch_score = val_score / len(valloader.sampler)
            writer.add_scalar('loss/validation', val_epoch_score, epoch)
            if val_epoch_score < best_val_score:  # Use '>' for metrics where higher is better
                print(f"Epoch {epoch + 1 }: New best score! Saving checkpoint.")
                best_val_score = val_epoch_score
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_score': best_val_score,
                }
                # Save the checkpoint
                torch.save(checkpoint, f'runs/{run_name}/checkpoint')
                plt.savefig(f'runs/{run_name}/best_confusion_matrix.png')
            plt.show()
                # heatmap_plot.savefig(f'runs/{run_name}/best_confusion_matrix.png')


        scheduler.step(val_epoch_score)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
        print(f'Epoch {epoch + 1}, Validation score'
              f': {val_epoch_score:.6f}')
        if num_classes>1:
            print(f'Epoch {epoch + 1}, Validation accuracy'
                  f': {val_epoch_accuracy:.6f}')


if __name__ == '__main__':
    training("test_num_classes_50_top5_accuracy", 3, False, 0.0003, 128,
             100, "./resized_images_224", 5, dino=False, data_frac=1,)
