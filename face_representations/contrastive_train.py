import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from celeba_data import AugmentedCelebA
import numpy as np
import time

learning_rate = .1
batch_size = 256
device = torch.device("cuda:1")

class EncodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18()
        self.fc_in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, input):
        output = self.resnet(input)
        return output

class ProjectionHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, input):
        output = self.fc1(input)
        output = F.relu(output)
        output = self.fc2(output)
        return output

def contrastive_loss(z1, z2, temp=1):
    # get unit vectors
    z1_unit = z1 / torch.norm(z1, p=2, dim=1).view(-1,1)
    z2_unit = z2 / torch.norm(z2, p=2, dim=1).view(-1,1)

    # compute z_i * z_j for all i,j in z1
    intra_cos_sims = z1_unit @ z1_unit.T

    # compute z_i * z_k for all i in z1, k in z2
    inter_cos_sims = z1_unit @ z2_unit.T
    cos_sims = torch.cat((intra_cos_sims, inter_cos_sims), dim=1)/temp

    # compute cross-entropy loss term
    # subtract out e to remove the zi * zi term from intra_cos_sims
    exp_numerator = torch.exp(torch.diagonal(inter_cos_sims))
    sum_exp = torch.sum(torch.exp(cos_sims), dim=1) - np.e
    losses = -torch.log(exp_numerator / sum_exp)

    return torch.mean(losses)

def train(encoding_net, proj_head, lr, bs):
    dataset = AugmentedCelebA("../../../CelebA", "../../celeba-segmented")
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=6, shuffle=True)

    optimizer = optim.Adam([{'params' : encoding_net.parameters()},
                            {'params' : proj_head.parameters()}])

    print_iter = 30
    start_time = time.time()
    for epoch in range(10):
        print(f"EPOCH {epoch}")
        avg_loss = 0
        for i, (images1, images2) in enumerate(dataloader):
            images1 = images1.to(device)
            images2 = images2.to(device)

            proj_head.zero_grad()
            encoding_net.zero_grad()
            rep1 = proj_head(encoding_net(images1))
            rep2 = proj_head(encoding_net(images2))

            loss = contrastive_loss(rep1, rep2)
            #print(loss)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()

            if i % print_iter == 0 and i != 0:
                print(f"{i/len(dataloader)*100:6.2f}% - "
                      f"{time.time() - start_time:0.0f} - "
                      f"{avg_loss/print_iter}")
                avg_loss = 0

                torch.save(encoding_net.state_dict(),"contrastive-model/model")
        torch.save(encoding_net.state_dict(), "contrastive-model/model")

if __name__ == "__main__":
    # testing the contrastive loss function:
    #mtx_2 = torch.tensor([[1.,2,3], [2,4,2], [1,4,8]])
    #mtx_1 = torch.tensor([[2.,2,3], [8,4,1], [2,2,2]])
    #print(contrastive_loss(mtx_1, mtx_2))
    #exit()

    encoding_net = EncodingNet().to(device)
    proj_head = ProjectionHead(encoding_net.fc_in_feats).to(device)
    train(encoding_net, proj_head, lr=learning_rate, bs=batch_size)
