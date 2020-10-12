# facial recognition
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler
from facenet_pytorch import MTCNN, InceptionResnetV1
from face_dataset import FaceDataset, FaceDatasetFull, FaceDatasetFull2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = InceptionResnetV1(pretrained=None, classify=True,num_classes = len(dataset.df['id'].value_counts()))
        # self.cnn1 = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(1, 4, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(4),
        #     nn.Dropout2d(p=.2),
        #
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(4, 8, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(8),
        #     nn.Dropout2d(p=.2),
        #
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(8, 8, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(8),
        #     nn.Dropout2d(p=.2),
        # )
        #
        # self.fc1 = nn.Sequential(
        #     nn.Linear(8 * 100 * 100, 500),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(500, 500),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(500, 5)
        # )

    def forward_once(self, x):
        # output = self.cnn1(x)
        # output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class CustomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, first, last):
        self.groups = dataset.df.groupby('id_mapped').groups
        self.first = first
        self.last = last
        self.cnt = 0
        for i in range(first, last):
            self.cnt += len(self.groups[i])

    def __iter__(self):
        li = []
        for i in range(first, last):
            for j in range(0, len(self.groups[i])):
                x = self.groups[i][j]

                # 50%
                if np.random.randint(2) == 0:
                    y = np.random.choice(self.groups[np.random.choice(np.arange(first, last))])
                else:
                    y = np.random.choice(self.groups[i])
                # this isnt guaranteed diff, but probably. i can make it guarantedd by taking it out of arange
                # diff = np.random.choice(self.groups[np.random.choice(np.arange(first, last))])
                # similar = np.random.choice(self.groups[i])

                li.append((x, y))

        return iter(li)

    def __len__(self):
        return self.cnt

# https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79
dataset = FaceDatasetFull2()

first = 0
last = int(len(dataset.df.groupby('id_mapped').groups)*0.8)
train_sampler = CustomSampler(first, last)
train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=8)

first = last
last = len(dataset.df.groupby('id_mapped').groups)
test_sampler = CustomSampler(first, last)
val_loader = DataLoader(dataset, sampler=test_sampler, batch_size=8)

# model = InceptionResnetV1(pretrained=None, classify=True,
#                              num_classes = len(dataset.df['id'].value_counts()))
siamese = SiameseNetwork()
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
# criterion = nn.CrossEntropyLoss()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(siamese.parameters(), lr=1e-4)

num_epochs = 101
print("num epochs to train on:", num_epochs)

losses = []

for epoch in range(0, num_epochs):
    print("epoch" + str(epoch))
    running_loss = 0
    training_hits = 0
    running_loss_val = 0
    val_hits = 0
    siamese.train()
    for index, item in enumerate(train_loader):
        input_batch = item[0] # create a mini-batch as expected by the model
        input_batch2 = item[1]
        output_tensor = item[2]

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            input_batch2 = input_batch2.to('cuda')
            siamese = siamese.to('cuda')
            output_tensor = output_tensor.to('cuda')
        y_pred, y_pred2 = siamese(input_batch, input_batch2)
        optimizer.zero_grad()
        loss = criterion(y_pred, y_pred2, output_tensor)
        # print("dissimilarity: " + str(F.pairwise_distance(y_pred, y_pred2)[0].item()) + ", same or not: " +
        #       str(output_tensor[0]))
        loss.backward()
        optimizer.step()

        # collect some loss data
        running_loss += loss.item()

        # for val in range(0, len(y_pred)): # len y_pred should be batch size
        #     if torch.argmax(y_pred[val]).item() == output_tensor[val]:
        #         training_hits += 1
    print("Training loss: " + str(running_loss))
    # print("Training acc: " + str(training_hits / len(train_sampler)))
    with torch.no_grad():
        siamese.eval()
        for index, item in enumerate(val_loader):
            input_batch = item[0]  # create a mini-batch as expected by the model
            input_batch2 = item[1]
            output_tensor = item[2]

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                input_batch2 = input_batch2.to('cuda')
                siamese = siamese.to('cuda')
                output_tensor = output_tensor.to('cuda')
            y_pred, y_pred2 = siamese(input_batch, input_batch2)
            # loss = criterion(y_pred, output_tensor)
            loss = criterion(y_pred, y_pred2, output_tensor)
            running_loss_val += loss.item()
            # collect some acc data
            # for val in range(0, len(y_pred)):
            #     if torch.argmax(y_pred[val]).item() == output_tensor[val]:
            #         val_hits += 1
    print("Validation loss: " + str(running_loss_val))
    losses.append(running_loss_val)
    # print("Validation acc: " + str(val_hits / len(test_sampler)))

with open('val_losses.txt', 'w') as f:
    for item in losses:
        f.write("%s\n" % item)
# https://github.com/fg91/visualizing-cnn-feature-maps
# https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030