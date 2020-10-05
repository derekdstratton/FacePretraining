from torch.utils.data import DataLoader, SubsetRandomSampler
from facenet_pytorch import MTCNN, InceptionResnetV1
from face_dataset import FaceDataset, FaceDatasetFull
import torch
import torch.nn as nn
# https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79
dataset = FaceDatasetFull()
# dataloader = DataLoader(dataset,batch_size=8,shuffle=True)

train_len = int(len(dataset)*0.8)
lengths = [train_len, len(dataset) - train_len]
subsetA, subsetB = torch.utils.data.random_split(dataset, lengths)
train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(subsetA.indices), batch_size=8)
val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(subsetB.indices), batch_size=8)

model = InceptionResnetV1(pretrained=None, classify=True,
                             num_classes = len(dataset.df['id'].value_counts()))
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# im using this as a mapping between unique indices and the identity value.
# its probably inefficient this way to keep referencing this
# mapping = dataset.df['id'].value_counts().index

print("num epochs to train on: 100")

for epoch in range(0, 100):
    print("epoch" + str(epoch))
    running_loss = 0
    training_hits = 0
    running_loss_val = 0
    val_hits = 0
    for index, item in enumerate(train_loader):
        input_batch = item[0] # create a mini-batch as expected by the model
        output_tensor = item[1]
        # output_indices = []
        # for identity in output_batch:
        #     output_indices.append(mapping.get_loc(identity.item()))
        # output_tensor = torch.tensor(output_batch)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model = model.to('cuda')
            output_tensor = output_tensor.to('cuda')
        y_pred = model(input_batch)
        optimizer.zero_grad()
        loss = criterion(y_pred, output_tensor)
        loss.backward()
        optimizer.step()

        # collect some loss data
        running_loss += loss.item()

        # collect some acc data
        # if len(y_pred) != 8:
        #     print(len(y_pred))
        #     for val in range(0, len(y_pred)):
        #         print(torch.argmax(y_pred[val]).item())
        #         print(output_tensor[val])

        for val in range(0, len(y_pred)): # len y_pred should be batch size, i think!
            if torch.argmax(y_pred[val]).item() == output_tensor[val]:
                training_hits += 1
    print("Training loss: " + str(running_loss))
    print("Training acc: " + str(training_hits / lengths[0]))
    for index, item in enumerate(val_loader):
        input_batch = item[0]  # create a mini-batch as expected by the model
        output_tensor = item[1]
        # output_indices = []
        # for identity in output_batch:
        #     output_indices.append(mapping.get_loc(identity.item()))
        # output_tensor = torch.tensor(output_batch)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model = model.to('cuda')
            output_tensor = output_tensor.to('cuda')
        y_pred = model(input_batch)
        loss = criterion(y_pred, output_tensor)
        running_loss_val += loss.item()
        # collect some acc data
        for val in range(0, len(y_pred)):
            if torch.argmax(y_pred[val]).item() == output_tensor[val]:
                val_hits += 1
    print("Validation loss: " + str(running_loss_val))
    print("Validation acc: " + str(val_hits / lengths[1]))

