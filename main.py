import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch import utils
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from stegnet import Model
from tiny_image_net import TinyImageNet
import torch.utils.data

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]
num_epochs = 1
batch_size = 32
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
    train_dataset = TinyImageNet(split='train')
    test_dataset = TinyImageNet(split='val')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = Model()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(num_epochs):

        train_losses = []
        for features, labels in tqdm.tqdm(train_dataloader):
            if use_cuda:
                features, labels = features.cuda(), labels.cuda()

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
            train_losses.append(train_loss.data[0])
            loss_history.append(train_loss.data[0])

            # Prints mini-batch losses
            # print('Training: Batch {0}/{1}. Loss of {2:.4f}, cover loss of {3:.4f}, secret loss of {4:.4f}'
            # .format(inputs + 1, len(train_dataloader), train_loss.data[0], train_loss_cover.data[0],
            # train_loss_secret.data[0]))

        mean_train_loss = np.mean(train_losses)

        # Prints epoch average loss
        print('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(epoch+1, num_epochs, mean_train_loss))

    plt.plot(loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.show()

    # test_losses = []
    # # Show images
    # for idx, test_batch in enumerate(test_dataloader):
    #     # Saves images
    #     dat, _ = test_batch
    #
    #     # Saves secret images and secret covers
    #     test_secret = dat[:len(dat)//2]
    #     test_cover = dat[len(dat)//2:]
    #
    #     # Creates variable from secret and cover images
    #     test_secret = Variable(test_secret, volatile=True)
    #     test_cover = Variable(test_cover, volatile=True)
    #
    #     # Compute output
    #     test_hidden, test_output = model(test_secret, test_cover)
    #
    #     # Calculate loss
    #     test_loss, loss_cover, loss_secret = customized_loss(test_output, test_hidden, test_secret, test_cover, beta)
    #
    #     if idx in [1, 2, 3, 4]:
    #         print('Total loss: {:.2f} \nLoss on secret: {:.2f} \nLoss on cover: {:.2f}'.format(test_loss.data[0], loss_secret.data[0], loss_cover.data[0]))
    #
    #         # Creates img tensor
    #         imgs = [test_secret.data, test_output.data, test_cover.data, test_hidden.data]
    #         imgs_tsor = torch.cat(imgs, 0)
    #
    #         # Prints Images
    #         imshow(utils.make_grid(imgs_tsor), idx+1, learning_rate=learning_rate, beta=beta)
    #
    #     test_losses.append(test_loss.data[0])
    #
    # mean_test_loss = np.mean(test_losses)
    #
    # print('Average loss on test set: {:.2f}'.format(mean_test_loss))
