import numpy as np
import torch
from ultils.datasets import ImdbDataset
from ultils.model import inception_V3, Densenet, Resnet
from ultils.transfroms import Rotate_Image, RGB_ToTensor, Normalization
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import argparse as argparse


def train_model(model,train_dataloader, test_dataloader, device, criterion1, criterion2, optimizer, n_epochs=25):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(train_dataloader):
            # importing data and moving to GPU
            image, label1, label2= sample_batched['image'].to(device,dtype=torch.float), sample_batched['label_age'].to(device,dtype=torch.float),  \
                                   sample_batched['label_gender'].to(device,dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()
            output = model(image)
            label1_hat = output['label1'].cuda()
            label2_hat = output['label2'].cuda()

            # calculate loss
            loss1 = criterion1(label1_hat, label1.squeeze())
            loss2 = criterion2(label2_hat, label2.squeeze())

            loss = loss1 + loss2

            # back prop
            loss.backward()

            # grad
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            if batch_idx % 50 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                      (epoch, batch_idx + 1, train_loss))

        # validate the model #
        model.eval()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_dataloader):
                image, label1, label2 = sample_batched['image'].to(device,dtype=torch.float), \
                                                sample_batched['label_age'].to(device,dtype=torch.float), \
                                                sample_batched['label_gender'].to(device,dtype=torch.float)
                output = model(image)
                label1_hat = output['label1'].cuda()
                label2_hat = output['label2'].cuda()

                # calculate loss
                loss1 = criterion1(label1_hat, label1.squeeze())
                loss2 = criterion2(label2_hat, label2.squeeze())

                loss = loss1 + loss2
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model, 'model.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            valid_loss_min = valid_loss
    # return trained model
    return model


def main(df_train_path, df_test_path,data_root_path,learning_rate, epochs):

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

    train_dataloader = DataLoader(train_dataset,batch_size=50, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #Setting model and moving to device
    model = Resnet().to(device)
    #For binary output:gender
    criterion_binary= nn.BCELoss()
    #For multilabel output: race and age
    criterion_multioutput = nn.CrossEntropyLoss()
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

    learning_rate = 0.001
    epochs= 50

    model_history = main(df_train_path=args.dftrain_path,
                         df_test_path=args.dftest_path,
                         data_root_path=args.dtroot_path,
                         learning_rate=learning_rate,
                         epochs=epochs)
