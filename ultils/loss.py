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