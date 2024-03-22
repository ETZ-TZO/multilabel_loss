#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
import torch


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l

class MultipleOutputLoss2Sander(MultipleOutputLoss2): #Sander edit

    def __init__(self, loss, weight_factors=None):
        super(MultipleOutputLoss2Sander, self).__init__(loss, weight_factors)

    def forward(self, output, target, file_names):

        assert len(output) == 5

        new_output_etz = []
        new_target_etz = []
        new_output_brats = []
        new_target_brats = []
        for i in range(len(file_names)):
            if file_names[i].split('_')[0] == 'ETZDataPaul' or file_names[i].split('_')[0] == 'CombinedDataEtz' or \
                    file_names[i].split('_')[0] == 'CombedDataEtz':
                print("ETZ data detected")
                for j in range(len(output)):
                    new_output_etz.append(output[j][i][1].unsqueeze(0).unsqueeze(0))
                    new_target_etz.append(target[j][i][1].unsqueeze(0).unsqueeze(0))
            else:
                print("Brats data detected")

                for j in range(len(output)):
                    new_output_brats.append(output[j][i].unsqueeze(0))
                    new_target_brats.append(target[j][i].unsqueeze(0))

        losses = torch.zeros(1).cuda()
        for i in range(int(len(new_output_etz)/5)):
            start = i*5
            end = 5 + i*5
            sub_loss = super(MultipleOutputLoss2Sander, self).forward(new_output_etz[start:end], new_target_etz[start:end])
            print('ETZ', sub_loss.item())
            losses = losses + sub_loss
        for i in range(int(len(new_output_brats)/5)):
            start = i*5
            end = 5 + i*5
            sub_loss = super(MultipleOutputLoss2Sander, self).forward(new_output_brats[start:end], new_target_brats[start:end])
            print('BraTS', sub_loss.item())
            losses = losses + sub_loss

        loss = losses / len(file_names)

        # origloss = self.loss(output, target)

        print("Loss: ",loss.item())

        return loss

