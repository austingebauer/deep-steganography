import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim
import model
from tiny_imagenet_dataset import TinyImageNet
import torch.utils.data

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]
num_epochs = 3
batch_size = 2
learning_rate = 0.0001
beta = 1


def customized_loss(S_prime, C_prime, S, C, B):
    loss_cover = torch.nn.functional.mse_loss(C_prime, C)
    loss_secret = torch.nn.functional.mse_loss(S_prime, S)
    loss_all = loss_cover + B * loss_secret
    return loss_all, loss_cover, loss_secret


def denormalize(image, std, mean):
    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image


def imshow(img, idx, learning_rate, beta):
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

    # Create the model object
    model = model.CNN()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("There are", num_params, "parameters in this model")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_history = []

    # Train the model
    for epoch in range(num_epochs):
        train_losses = []
        for idx, data in enumerate(train_dataloader):
            # Underscore is labels since enumerate gives us idx, features, labels
            features, _ = data

            # Reject if not 3 channel tensor
            if len(features.size()) == 4 and features.size()[1] != 3:
                print("skipping tensor that does not have 3 channels (RGB)")
                continue

            if len(features) != batch_size:
                print(len(features))
                print(features.shape)
                print("skipping batch that is not of length 2")
                continue

            # Saves secret images and secret covers
            train_secrets = features[len(features)//2:]
            train_covers = features[:len(features)//2]

            # Creates variable from secret and cover images
            train_secrets = Variable(train_secrets, requires_grad=False)
            train_covers = Variable(train_covers, requires_grad=False)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            train_hidden, train_revealed = model(train_secrets, train_covers)

            # Calculate loss and perform backprop
            train_loss, train_loss_cover, train_loss_secret = customized_loss(train_revealed, train_hidden, train_secrets, train_covers, beta)
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

    # save the model for future reference
    torch.save(model, "model.pth")

    # model = torch.load("model.pth")

    test_losses = []
    model.eval()

    # Test the model
    for idx, data in enumerate(test_dataloader):
        # Saves images
        features, _ = data

        if len(features.size()) == 4 and features.size()[1] != 3:
            print("skipping tensor that does not have 3 channels (RGB)")
            continue

        if len(features) != batch_size:
            print(len(features))
            print(features.shape)
            print("skipping batch that is not of length 2")
            continue

        # Saves secret images and secret covers
        test_secret = features[:len(features)//2]
        test_cover = features[len(features)//2:]

        # Creates variable from secret and cover images
        test_secret = Variable(test_secret, volatile=True)
        test_cover = Variable(test_cover, volatile=True)

        # Compute output
        test_hidden, test_revealed = model(test_secret, test_cover)

        # Calculate loss
        test_loss, loss_cover, loss_secret = customized_loss(test_revealed, test_hidden, test_secret, test_cover, beta)

        print('Total loss: {:.2f} \nLoss on secret: {:.2f} \nLoss on cover: {:.2f}'.format(test_loss.item(), loss_secret.item(), loss_cover.item()))

        # Creates img tensor
        imgs = [test_secret.data, test_cover.data, test_hidden.data, test_revealed.data]
        imgs_tsor = torch.cat(imgs, 0)

        # Prints Images
        imshow(utils.make_grid(imgs_tsor), idx+1, learning_rate=learning_rate, beta=beta)

        test_losses.append(test_loss.item())

    mean_test_loss = np.mean(test_losses)
    print('Average loss on test set: {:.2f}'.format(mean_test_loss))
