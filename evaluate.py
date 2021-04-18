import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from torchvision import  transforms
from utils.transfroms import RGB_ToTensor,Normalization
from utils.datasets import ImdbDataset
from torch.utils.data import DataLoader
from utils.metrics import Accuracy, MeanAbsoluteError
from utils.model import ResnetV3, ResnetV2
from torch import nn
import argparse as argparse
import random


def plot_losses_metrics(df_losses_metrics_path, model_save_path):
    df = pd.read_csv(df_losses_metrics_path)
    df['Epoch'] = df['index'].values + 1
    plt.figure(1,figsize=(10,5))
    sns.lineplot(data=df,x="Epoch",y='loss',legend="full", label = "Train loss")
    sns.lineplot(data=df, x="Epoch", y='val_loss', legend="full" , label = "Val loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(model_save_path,"losses.jpg"))
    plt.show()

    plt.figure(2, figsize=(10, 5))
    sns.lineplot(data=df, x="Epoch", y='train_age_mae', legend="full", label = "Train Age MAE")
    sns.lineplot(data=df, x="Epoch", y='val_age_mae', legend="full", label = "Val Age MAE")
    plt.title("Age MAE")
    plt.legend()
    plt.savefig(os.path.join(model_save_path, "age_maes.jpg"))
    plt.show()

    plt.figure(3, figsize=(10, 5))
    sns.lineplot(data=df, x="Epoch", y='train_age_acc', legend="full", label = "Train Age Accuracy")
    sns.lineplot(data=df, x="Epoch", y='val_age_acc', legend="full", label = "Val Age Accuracy")
    plt.title("Age Accuracy")
    plt.legend()
    plt.savefig(os.path.join(model_save_path, "age_accs.jpg"))
    plt.show()

    plt.figure(4, figsize=(10, 5))
    sns.lineplot(data=df, x="Epoch", y='train_gender_acc', legend="full", label = "Train Gender Accuracy")
    sns.lineplot(data=df, x="Epoch", y='val_gender_acc', legend="full", label = "Val Gender Accuracy")
    plt.title("Gender Accuracy")
    plt.legend()
    plt.savefig(os.path.join(model_save_path, "gender_accs.jpg"))
    plt.show()


def evaluate_test_set(model,criterion1,criterion2,test_dataloader,device):

    # validate the model #
    model.eval()

    test_loss = 0.0
    test_age_mae = 0.0
    test_age_acc = 0.0
    test_gender_acc = 0.0

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_dataloader):
            image, label1, label2 = sample_batched['image'].to(device, dtype=torch.float), \
                                    sample_batched['label_age'].to(device, dtype=torch.long), \
                                    sample_batched['label_gender'].to(device, dtype=torch.long)
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

            loss = 3 * loss1 + loss2

            test_loss += loss.item()
            test_age_acc += age_Accuracy.item()
            test_age_mae += age_MAE.item()
            test_gender_acc += gender_Accuracy.item()

        print("*****\nTest loss: {:.6f}"
              "\nTest Age MAE: {:.4f}"
              "\nTest Age Accuracy: {:.4f}"
              "\nTest Gender Accuracy: {:.4f}".format(test_loss/len(test_dataloader),test_age_mae/len(test_dataloader),
                                                      test_age_acc/len(test_dataloader),test_gender_acc/len(test_dataloader)))




def predict(model,model_save_path,test_dataset,device, num_predicts = 20):
    for i in range(num_predicts):
        index = random.randint(0,10000)
        sample = test_dataset[index]
        image, age, gender = sample['image'], sample['label_age'], sample['label_gender']
        image_pre = torch.unsqueeze(image, 0).to(device)
        output = model(image_pre)
        age_hat = output['label1']
        gender_hat = output['label2']
        ax = plt.figure(figsize=(10,10))
        plt.imshow(image.numpy().transpose((1, 2, 0))*255)
        gender = "M" if gender.item() == 1.0 else "FM"
        gender_hat = torch.argmax(gender_hat).item()
        gender_hat = "M" if gender_hat == 1 else "FM"
        age = int(age.mul_(100).item())
        age_hat = int(age_hat.mul_(100).item())
        pre_str = str(age_hat) + "," + gender_hat
        actual_str = str(age) + "," + gender
        textstr = '\n' + 'Predict: ' + pre_str + '\nActual: ' + actual_str
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.axis('off')
        figname = str(age_hat) + "_" + gender_hat + "_" + str(age) + "_" + gender + ".jpg"
        plt.savefig(os.path.join(model_save_path,figname))


def main(model_save_path, df_test_path, dt_root_path, batch_size=256):
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    test_transforms = transforms.Compose([
        RGB_ToTensor(),
        # Normalization(cnn_normalization_mean, cnn_normalization_std)
    ])

    test_dataset = ImdbDataset(dataframe_path=df_test_path, data_root_path=dt_root_path, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    checkpoint = torch.load(os.path.join(model_save_path, "best_checkpoint.tar"), map_location='cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResnetV2().to(device)
    # For binary output:gender
    criterion_binary = nn.NLLLoss()
    # For multilabel output: and age
    criterion_multioutput = nn.NLLLoss()
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Evaluating........")
    evaluate_test_set(model = model,
                      criterion1=criterion_multioutput,
                      criterion2=criterion_binary,
                      test_dataloader = test_dataloader,
                      device=device)
    print("Plotting........")
    df_losses_metrics_path = os.path.join(model_save_path,"losses_metrics.csv")
    plot_losses_metrics(df_losses_metrics_path= df_losses_metrics_path,
                        model_save_path=model_save_path)
    print("Predict some pictures...")
    predict(model = model,
            model_save_path= model_save_path,
            test_dataset= test_dataset,
            device=device,
            num_predicts=10)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dftest-path', required=True)
    parser.add_argument('--dtroot-path', required=True)
    parser.add_argument('--mdsave-path', required=True)
    args = parser.parse_args()
    main(model_save_path=args.mdsave_path,
         df_test_path=args.dftest_path,
         dt_root_path=args.dtroot_path)
