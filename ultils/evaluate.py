import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_losses_metrics(df_losses_metrics_path):
    df = pd.read_csv(df_losses_metrics_path)
    df['Epoch'] = df['index'].values + 1
    plt.figure(1,figsize=(20,10))
    sns.lineplot(data=df,x="Epoch",y='loss',legend="full")
    sns.lineplot(data=df, x="Epoch", y='val_loss', legend="full")
    plt.show()

    plt.figure(2, figsize=(20, 10))
    sns.lineplot(data=df, x="Epoch", y='train_age_mae', legend="full")
    sns.lineplot(data=df, x="Epoch", y='val_age_mae', legend="full")
    plt.show()

    plt.figure(3, figsize=(20, 10))
    sns.lineplot(data=df, x="Epoch", y='train_age_acc', legend="full")
    sns.lineplot(data=df, x="Epoch", y='val_age_acc', legend="full")
    plt.show()

    plt.figure(4, figsize=(20, 10))
    sns.lineplot(data=df, x="Epoch", y='train_gender_acc', legend="full")
    sns.lineplot(data=df, x="Epoch", y='val_gender_acc', legend="full")
    plt.show()

