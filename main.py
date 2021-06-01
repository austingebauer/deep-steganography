import torch.nn as nn
from torch import utils
import torch.optim as optim
import tqdm
from model import Model
from tiny_imagenet_dataset import TinyImageNet
import torch.utils.data

num_epochs = 3
batch_size = 32

if __name__ == "__main__":
    # Load tiny imagenet training and test data sets
    train_dataset = TinyImageNet(split='train')
    validation_dataset = TinyImageNet(split='val')

    # Create data loaders using datasets
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    model = Model()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # Train the model
    criterion = nn.CrossEntropyLoss()
    running_loss = []
    for epoch in range(num_epochs):
        learning_rate = 0.01 * 0.8 ** epoch
        learning_rate = max(learning_rate, 1e-6)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        for features, labels in tqdm.tqdm(train_dataloader):
            if use_cuda:
                features, labels = features.cuda(), labels.cuda()

            print(features.size)
            print(labels.size)

            # TODO: forward + backward + optimize
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # TODO: use the model on testing data
