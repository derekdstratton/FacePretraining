# facial recognition
import json

from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler
from facenet_pytorch import MTCNN, InceptionResnetV1
from face_dataset import FaceDataset, FaceDatasetFull, FaceDatasetFull2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from face_representations import contrastive_train
import random
from datetime import datetime

# https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = contrastive_train.EncodingNet()
        self.model.load_state_dict(torch.load('contrastive-model/model'))
        self.model.fc = nn.Identity()
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return self.fc1(torch.abs(output1 - output2))


class CustomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, first, last):
        self.groups = groups_shuffled
        self.first = first
        self.last = last
        self.cnt = 0
        for i in range(first, last):
            self.cnt += len(self.groups[i])

    def __iter__(self):
        li = []
        for i in range(self.first, self.last):
            for j in range(0, len(self.groups[i])):
                x = self.groups[i][j]

                # 50%
                if np.random.randint(2) == 0:
                    y = np.random.choice(self.groups[np.random.choice(np.arange(first, last))])
                else:
                    y = np.random.choice(self.groups[i])
                # this isnt guaranteed diff, but probably. i can make it guarantedd by taking it out of arange

                li.append((x, y))

        return iter(li)

    def __len__(self):
        return self.cnt

# https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79
dataset = FaceDatasetFull2()

# shuffle keys in dict, so the amount of pictures per group is random between train and test
groups_unshuffled = dataset.df.groupby('id_mapped').groups
keys = list(groups_unshuffled.keys())
random.shuffle(keys)
groups_shuffled = {keys[i]: groups_unshuffled[i] for i in range(len(keys))}
# this is questionable, since the groups are sorted by highest amount of samples...
first = 0
last = int(len(groups_shuffled)*0.8)
train_sampler = CustomSampler(first, last)
train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=32)

first = last
last = len(groups_shuffled)
test_sampler = CustomSampler(first, last)
val_loader = DataLoader(dataset, sampler=test_sampler, batch_size=32)

siamese = SiameseNetwork()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(siamese.parameters(), lr=1e-4)

num_epochs = 50
print("num epochs to train contrastive verify on:", num_epochs)

metrics = {
    'training_loss':[],
    'training_acc':[],
    'val_loss':[],
    'val_acc':[]
}
try:
    for epoch in range(0, num_epochs):
        print("epoch" + str(epoch))
        running_loss = 0
        training_hits = 0
        running_loss_val = 0
        val_hits = 0
        t_misses = 0
        v_misses = 0
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
            y_pred = siamese(input_batch, input_batch2)
            optimizer.zero_grad()
            y_pred = y_pred.reshape(y_pred.shape[0]) # reshape to batch size
            loss = criterion(y_pred.double(), output_tensor)
            loss.backward()
            optimizer.step()

            # collect some loss data
            running_loss += loss.item()

            for val in range(0, len(y_pred)): # len y_pred should be batch size
                x = 0. if y_pred[val].item() < 0.5 else 1.
                if x == output_tensor[val]:
                    training_hits += 1
                else:
                    t_misses += 1
        print("Training loss: " + str(running_loss))
        print("Training acc: " + str(training_hits / len(train_sampler)))
        metrics['training_acc'].append(training_hits / len(train_sampler))
        metrics['training_loss'].append(running_loss)
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
                y_pred = siamese(input_batch, input_batch2)
                y_pred = y_pred.reshape(y_pred.shape[0])  # reshape to batch size
                loss = criterion(y_pred.double(), output_tensor)
                running_loss_val += loss.item()
                # collect some acc data
                for val in range(0, len(y_pred)):  # len y_pred should be batch size
                    x = 0. if y_pred[val].item() < 0.5 else 1.
                    if x == output_tensor[val]:
                        val_hits += 1
                    else:
                        v_misses += 1
        print("Validation loss: " + str(running_loss_val))
        metrics['val_acc'].append(val_hits / len(test_sampler))
        metrics['val_loss'].append(running_loss_val)
        print("Validation acc: " + str(val_hits / len(test_sampler)))
except KeyboardInterrupt:
    print("interrupted: still printing metrics to file.")
    now = str(datetime.date(datetime.now()))
    with open('metrics-' + now + '.txt', 'w') as f:
        json.dump(metrics, f)

now = str(datetime.date(datetime.now()))
with open('metrics-' + now + '.txt', 'w') as f:
    json.dump(metrics, f)
# https://github.com/fg91/visualizing-cnn-feature-maps
# https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030