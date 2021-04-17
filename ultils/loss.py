from torch import nn
import torch

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, age_hat, gender_hat , age, gender):
        mse, crossEntropy = nn.MSELoss(), nn.NLLLoss()

        sages = age*100
        idx1 = (sages < 20) | ((sages > 40) & (sages <= 60))
        idx2 = sages > 60

        loss0 = mse(age_hat, age) + 2 * mse(age_hat[idx1], age[idx1]) + 3 * mse(age_hat[idx2], age[
            idx2])  # trying to account for the imbalance
        loss1 = crossEntropy(gender_hat, gender.squeeze())


        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * loss1 + self.log_vars[1]


        return loss0 + loss1


class Classify_Loss(nn.Module):
    def __init__(self, weight):
        super(Classify_Loss, self).__init__()
        self.weight = weight
        self.log_vars = nn.Parameter(torch.ones((2)))

    def forward(self, age_hat, gender_hat, age, gender):
        # loss_age = nn.CrossEntropyLoss()
        idx1 = (age < 20) | ((age > 40) & (age <= 60))
        idx2 = age > 60
        loss_age = nn.NLLLoss()
        loss_gender = nn.NLLLoss()
        loss1 = loss_age(age_hat, age) + 2 * loss_age(age_hat[idx1], age[idx1]) + 3 * loss_age(age_hat[idx2], age[
            idx2])
        loss1 = loss1*torch.exp(-self.log_vars[0])  + self.log_vars[0]
        loss2 = loss_gender(gender_hat, gender)*torch.exp(-self.log_vars[1]) + self.log_vars[1]

        #loss_total = self.weight[0] * loss1 + self.weight[1] * loss2
        loss_total = loss1+loss2
        return loss_total