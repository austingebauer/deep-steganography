import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch import utils
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from stegnet import Model
from tiny_image_net import TinyImageNet
import torch.utils.data

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]
num_epochs = 3
batch_size = 2
learning_rate = 0.0001
beta = 1

def customized_loss(S_prime, C_prime, S, C, B):
    ''' Calculates loss specified on the paper.'''

    loss_cover = torch.nn.functional.mse_loss(C_prime, C)
    loss_secret = torch.nn.functional.mse_loss(S_prime, S)
    loss_all = loss_cover + B * loss_secret
    return loss_all, loss_cover, loss_secret


def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image


def imshow(img, idx, learning_rate, beta):
    '''Prints out an image given in tensor format.'''

    img = denormalize(img, std, mean)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('Example '+str(idx)+', lr='+str(learning_rate)+', B='+str(beta))
    plt.show()
    return

if __name__ == "__main__":
    train_dataset = TinyImageNet(split='train', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]))
    test_dataset = TinyImageNet(split='val', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Set the torch seed
    torch.manual_seed(1)

    # Create the model object
    model = Model()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("There are", num_params, "parameters in this model")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    nsamples = 208
    loss_history = []
    for epoch in range(num_epochs):
        model.train()

        train_losses = []
        for idx, data in enumerate(train_dataloader):
            if idx > nsamples:
                break

            features, _ = data

            # Saves secret images and secret covers
            train_covers = features[:len(features)//2]
            train_secrets = features[len(features)//2:]

            # Creates variable from secret and cover images
            train_secrets = Variable(train_secrets, requires_grad=False)
            train_covers = Variable(train_covers, requires_grad=False)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            train_hidden, train_output = model(train_secrets, train_covers)

            # Calculate loss and perform backprop
            train_loss, train_loss_cover, train_loss_secret = customized_loss(train_output, train_hidden, train_secrets, train_covers, beta)
            train_loss.backward()
            optimizer.step()

            # Saves training loss
            train_losses.append(train_loss.item())
            loss_history.append(train_loss.item())

            print('Training: Batch {0}/{1}. Loss of {2:.4f}, cover loss of {3:.4f}, secret loss of {4:.4f}'.format(
                idx+1, len(train_dataloader), train_loss.item(), train_loss_cover.item(), train_loss_secret.item()))

        mean_train_loss = np.mean(train_losses)

        # Prints epoch average loss
        print('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(epoch+1, num_epochs, mean_train_loss))

    plt.plot(loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.show()