import numpy as np
import torch
from ultils.datasets import ImdbDataset
from ultils.model import inception_V3, Densenet, Resnet
from ultils.transfroms import Rotate_Image, RGB_ToTensor, Normalization
from torch.utils.data import DataLoader
from ultils.metrics import Accuracy, MeanAbsoluteError
from torchvision import transforms
from torch import nn, optim
import argparse as argparse


def train_model(model,train_dataloader, test_dataloader, device, criterion1, criterion2, optimizer, n_epochs=25):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    train_loss, train_age_mae, train_age_acc, train_gender_acc = [],[],[],[]
    val_loss, val_age_mae, val_age_acc, val_gender_acc = [], [], [], []
    for epoch in range(1, n_epochs):

        # train the model #
        model.train()
        running_loss = 0.0
        running_age_mae = 0.0
        running_age_acc = 0.0
        running_gender_acc = 0.0
        epoch_loss = 0.0
        epoch_age_mae = 0.0
        epoch_age_acc = 0.0
        epoch_gender_acc = 0.0
        for batch_idx, sample_batched in enumerate(train_dataloader):
            # importing data and moving to GPU
            image, label1, label2= sample_batched['image'].to(device,dtype=torch.float), sample_batched['label_age'].to(device,dtype=torch.long),  \
                                   sample_batched['label_gender'].to(device,dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()
            output = model(image)
            label1_hat = output['label1'].cuda()
            label2_hat = output['label2'].cuda()

            #calculate metrics
            age_MAE = MeanAbsoluteError()(label1_hat, label1.squeeze())
            age_Accuracy = Accuracy()(label1_hat, label1.squeeze())
            gender_Accuracy = Accuracy()(label2_hat, label2.squeeze())

            # calculate loss
            loss1 = criterion1(label1_hat, label1.squeeze())
            loss2 = criterion2(label2_hat, label2.squeeze())

            loss = loss1 + loss2

            # back prop
            loss.backward()

            # grad
            optimizer.step()

            running_loss += loss.item()
            running_age_mae += age_MAE.item()
            running_age_acc += age_Accuracy.item()
            running_gender_acc += gender_Accuracy.item()
            epoch_loss += loss.item()
            epoch_age_acc += age_MAE.item()
            epoch_age_mae += age_Accuracy.item()
            epoch_gender_acc += gender_Accuracy.item()

            if batch_idx % 50 == 0:
                print('Epoch %d, Batch %d loss: %.6f Age_MAE: %.4f Age_Accuracy: %.4f Gender_Accuracy: %.4f' %
                      (epoch, batch_idx + 1, running_loss/50, running_age_mae/50, running_age_acc/50, running_gender_acc/50))
                running_loss = 0.0
                running_age_mae = 0.0
                running_age_acc = 0.0
                running_gender_acc = 0.0

        train_loss.append(epoch_loss/len(train_dataloader))
        train_age_acc.append(epoch_age_acc/len(train_dataloader))
        train_age_mae.append(epoch_age_mae/len(train_dataloader))
        train_gender_acc.append(epoch_gender_acc/len(train_dataloader))

        epoch_loss = 0.0
        epoch_age_mae = 0.0
        epoch_age_acc = 0.0
        epoch_gender_acc = 0.0

        # validate the model #
        model.eval()

        epoch_val_loss = 0.0
        epoch_val_age_mae = 0.0
        epoch_val_age_acc = 0.0
        epoch_val_gender_acc = 0.0

        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_dataloader):
                image, label1, label2 = sample_batched['image'].to(device,dtype=torch.float), \
                                                sample_batched['label_age'].to(device,dtype=torch.long), \
                                                sample_batched['label_gender'].to(device,dtype=torch.long)
                output = model(image)
                label1_hat = output['label1'].cuda()
                label2_hat = output['label2'].cuda()

                # calculate metrics
                age_MAE = MeanAbsoluteError()(label1_hat, label1.squeeze())
                age_Accuracy = Accuracy()(label1_hat, label1.squeeze())
                gender_Accuracy = Accuracy()(label2_hat, label2.squeeze())

                # calculate loss
                loss1 = criterion1(label1_hat, label1.squeeze())
                loss2 = criterion2(label2_hat, label2.squeeze())

                loss = loss1 + loss2

                epoch_val_loss += loss.item()
                epoch_val_age_acc += age_MAE.item()
                epoch_val_age_mae += age_Accuracy.item()
                epoch_val_gender_acc += gender_Accuracy.item()

            val_loss.append(epoch_val_loss / len(test_dataloader))
            val_age_acc.append(epoch_val_age_acc / len(test_dataloader))
            val_age_mae.append(epoch_val_age_mae / len(test_dataloader))
            val_gender_acc.append(epoch_val_gender_acc / len(test_dataloader))

            epoch_val_loss = 0.0
            epoch_val_age_acc = 0.0
            epoch_val_age_mae = 0.0
            epoch_val_gender_acc = 0.0

        # print training/validation statistics
        print('*****\nEpoch: {} \nTraining Loss: {:.6f} \tTrain Age MAE: {:.4f} \tTrain Age Accuracy: {:.4f} '
              '\tTrain Gender Accuracy: {:.4f}'
              '\nValidation Loss: {:.6f} \tValidation Age MAE: {:.4f} \tValidation Age Accuracy: {:.4f} '
              '\tValidation Gender Accuracy: {:.4f}\n*****'.format(
            epoch, train_loss[-1], train_age_mae[-1],train_age_acc[-1], train_gender_acc[-1],
            val_loss[-1], val_age_mae[-1], val_age_acc[-1], val_gender_acc[-1]))

        ## TODO: save the model if validation loss has decreased
        if val_loss[-1] < valid_loss_min:
            torch.save(model, 'model.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                val_loss[-1]))
            valid_loss_min = val_loss[-1]
    # return trained model
    return model


def main(df_train_path, df_test_path,data_root_path,learning_rate, epochs, batch_size):

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
                                    Rotate_Image(),
                                    RGB_ToTensor(),
                                    Normalization(cnn_normalization_mean, cnn_normalization_std)])

    test_transforms = transforms.Compose([
                                    RGB_ToTensor(),
                                    Normalization(cnn_normalization_mean, cnn_normalization_std)])

    train_dataset = ImdbDataset(dataframe_path=df_train_path, data_root_path=data_root_path, transform=train_transforms)
    test_dataset = ImdbDataset(dataframe_path=df_test_path, data_root_path=data_root_path, transform=test_transforms)

    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #Setting model and moving to device
    model = Resnet().to(device)
    #For binary output:gender
    criterion_binary= nn.NLLLoss()
    #For multilabel output: race and age
    criterion_multioutput = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    model_history =train_model(model,
                               train_dataloader,
                               test_dataloader,
                               device,
                               criterion_multioutput,
                               criterion_binary,
                               optimizer,
                               n_epochs=epochs)

    return model_history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dftrain-path', required=True)
    parser.add_argument('--dftest-path', required=True)
    parser.add_argument('--dtroot-path', required=True)
    args = parser.parse_args()

    learning_rate = 0.005
    epochs= 50
    batch_size = 256

    model_history = main(df_train_path=args.dftrain_path,
                         df_test_path=args.dftest_path,
                         data_root_path=args.dtroot_path,
                         learning_rate=learning_rate,
                         epochs=epochs,
                         batch_size= batch_size)
