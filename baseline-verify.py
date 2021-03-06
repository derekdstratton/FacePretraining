# facial recognition
import json

from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler
#from facenet_pytorch import MTCNN, InceptionResnetV1
from face_dataset import FaceDataset, FaceDatasetFull, FaceDatasetFull2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from face_representations import contrastive_train
import random
from datetime import datetime
import time

MODEL = "contrastive"

# https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        if MODEL == "baseline":
            self.model = models.resnet18()
        elif MODEL == "contrastive":
            self.model = contrastive_train.EncodingNet()
            self.model.load_state_dict(torch.load('contrastive-model/model'))
        else:
            raise Exception("illegal MODEL string")
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

    def __init__(self, groups):
        self.groups = groups
        # self.first = first
        # self.last = last
        self.cnt = 0
        for i in self.groups:
            self.cnt += len(self.groups[i])

    def __iter__(self):
        li = []
        for i in self.groups:
            for j in range(0, len(self.groups[i])):
                x = self.groups[i][j]

                # 50%
                if torch.rand(1).item() > .5:
                    group_choice = i
                    while group_choice == i:
                        group_choice = torch.randint(0, len(self.groups) - 1, (1,)).item()
                        group_choice = list(self.groups.keys())[group_choice] # convert from index to id
                    group_ims = self.groups[group_choice]
                    im_choice = torch.randint(len(group_ims),(1,)).item()
                    y = self.groups[group_choice][im_choice]
                else:
                    im_choice=torch.randint(len(self.groups[i])-1,(1,)).item()
                    if im_choice >= j:
                        im_choice += 1
                    y = self.groups[i][im_choice]

                li.append((x, y))

        return iter(li)

    def __len__(self):
        # return len(self.groups)
        return self.cnt

# https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79
dataset = FaceDatasetFull2(augmentations=True)
dataset_val = FaceDatasetFull2(augmentations=False)

# shuffle keys in dict, so the amount of pictures per group is random between train and test
all_groups = dataset.df.groupby('id').groups
with open('train_ids.txt') as f:
    train_ids = f.read().splitlines()
    train_ids = [int(x.split('-')[-1]) for x in train_ids]
    train_groups = {k: all_groups[k] for k in train_ids if k in all_groups.keys()}
with open('val_ids.txt') as f:
    val_ids = f.read().splitlines()
    val_ids = [int(x.split('-')[-1]) for x in val_ids]
    val_groups = {k: all_groups[k] for k in val_ids if k in all_groups.keys()}
### TODO: make the train and test groups here, then pass that into the sampler instead of first and last.
# keys = list(groups_unshuffled.keys())
# random.shuffle(keys)
# groups_shuffled = {keys[i]: groups_unshuffled[i] for i in range(len(keys))}
# this is questionable, since the groups are sorted by highest amount of samples...
batch_size = 128
# first = 0
# last = int(len(groups_shuffled)*0.8)
train_sampler = CustomSampler(train_groups)
train_loader = DataLoader(dataset, sampler=train_sampler,
                          num_workers=8, batch_size=batch_size)

# first = last
# last = len(groups_shuffled)
test_sampler = CustomSampler(val_groups)
val_loader = DataLoader(dataset_val, sampler=test_sampler,
                        num_workers=8, batch_size=batch_size)

siamese = SiameseNetwork()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(siamese.parameters())

num_epochs = 50
print("num epochs to train " + MODEL + " on:" + str(num_epochs))

metrics = {
    'training_loss':[],
    'training_acc':[],
    'val_loss':[],
    'val_acc':[]
}
try:
    start_time = time.time()
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

            if index % 250 == 249:
                print(f"{index/len(train_loader)*100:5.2f}% / "
                      f"{time.time() - start_time:5.0f}s | "
                      f"loss: {running_loss/(index + 1):.4f} / "
                      f"acc: {training_hits/(index + 1)*100/batch_size:.2f}")

        print("Training loss: " + str(running_loss / len(train_loader)))
        print("Training acc: " + str(training_hits / len(train_sampler)))
        metrics['training_acc'].append(training_hits / len(train_sampler))
        metrics['training_loss'].append(running_loss / len(train_loader))
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
        print("Validation loss: " + str(running_loss_val / len(val_loader)))
        metrics['val_acc'].append(val_hits / len(test_sampler))
        metrics['val_loss'].append(running_loss_val / len(val_loader))
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
