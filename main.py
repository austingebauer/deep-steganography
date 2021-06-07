import torch
import sys
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
import model
from tiny_imagenet_dataset import TinyImageNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np


EPOCHS = 2
BATCH_SIZE = 32
BETA = 0.75


def print_net(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(str(net))
    print('Total number of parameters: %d' % num_params)


def train(train_loader, epoch, hide_net, reveal_net, criterion):
    hide_losses = []
    reveal_losses = []
    sum_losses = []

    # switch to train mode
    hide_net.train()
    reveal_net.train()

    for i, data in enumerate(train_loader, 0):
        hide_net.zero_grad()
        reveal_net.zero_grad()

        this_batch_size = int(data.size()[0] / 2)
        cover_img = data[0:this_batch_size, :, :, :]
        secret_img = data[this_batch_size:this_batch_size * 2, :, :, :]

        concat_img = torch.cat([cover_img, secret_img], dim=1)

        concat_imgv = Variable(concat_img)
        cover_imgv = Variable(cover_img)

        container_img = hide_net(concat_imgv)
        err_hide = criterion(container_img, cover_imgv)
        hide_losses.append(err_hide.item())

        rev_secret_img = reveal_net(container_img)
        secret_imgv = Variable(secret_img)
        err_reveal = criterion(rev_secret_img, secret_imgv)
        reveal_losses.append(err_reveal.item())

        beta_err_reveal = BETA * err_reveal
        err_sum = err_hide + beta_err_reveal
        sum_losses.append(err_sum.item())
        err_sum.backward()

        optimizerH.step()
        optimizerR.step()

        print('[%d/%d][%d/%d]\thide_loss: %.4f reveal_loss: %.4f sum_loss: %.4f' % (
            epoch + 1, EPOCHS, i + 1, len(train_loader),
            err_hide.item(), err_reveal.item(), err_sum.item()))

        if i % 10 == 0:
            save_image_results(this_batch_size, cover_img, container_img.data, secret_img,
                               rev_secret_img.data, epoch, i, './training')

    epoch_log = "epoch learning rate: optimizer_hide_lr = %.8f\toptimizer_reveal_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_avg_hide_loss=%.6f\tepoch_avg_reveal_loss=%.6f\tepoch_avg_sum_loss=%.6f" % (
        np.mean(hide_losses), np.mean(reveal_loss), np.mean(sum_loss))
    print(epoch_log)


def validate(val_loader, epoch, hide_net, reveal_net, criterion):
    hide_losses = []
    reveal_losses = []

    # switch to validation mode
    hide_net.eval()
    reveal_net.eval()

    for i, data in enumerate(val_loader, 0):
        hide_net.zero_grad()
        reveal_net.zero_grad()
        all_pics = data
        this_batch_size = int(all_pics.size()[0] / 2)

        cover_img = all_pics[0:this_batch_size, :, :, :]
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        concat_img = torch.cat([cover_img, secret_img], dim=1)
        concat_imgv = Variable(concat_img)
        cover_imgv = Variable(cover_img)

        container_img = hide_net(concat_imgv)
        err_hide = criterion(container_img, cover_imgv)
        hide_losses.append(err_hide.item())

        rev_secret_img = reveal_net(container_img)
        secret_imgv = Variable(secret_img)
        err_reveal = criterion(rev_secret_img, secret_imgv)
        hide_losses.append(err_reveal.item())

        if i % 10 == 0:
            save_image_results(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                            './validation')

    avg_hide_loss = np.mean(hide_losses)
    avg_reveal_loss = np.mean(reveal_losses)
    avg_sum_loss = avg_hide_loss + BETA * avg_reveal_loss

    val_log = "validation[%d] avg_hide_loss = %.6f\t avg_reveal_loss = %.6f\t avg_sum_loss = %.6f" % (
        epoch + 1, avg_hide_loss, avg_reveal_loss, avg_sum_loss)
    print(val_log)

    return avg_hide_loss, avg_reveal_loss, avg_sum_loss


def save_image_results(this_batch_size, cover_image, container_image, secret_image, revealed_image, epoch, i, save_path):
        # TODO: might not need this?
        # with torch.no_grad():
        #     originalFrames = cover_image.resize_(this_batch_size, 3, 256, 256)
        #     containerFrames = container_image.resize_(this_batch_size, 3, 256, 256)
        #     secretFrames = secret_image.resize_(this_batch_size, 3, 256, 256)
        #     revSecFrames = revealed_image.resize_(this_batch_size, 3, 256, 256)

        showContainer = torch.cat([cover_image, container_image], 0)
        showReveal = torch.cat([secret_image, revealed_image], 0)
        resultImg = torch.cat([showContainer, showReveal], 0)
        resultImgName = '%s/result_images_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(resultImg, resultImgName, nrow=this_batch_size, padding=1, normalize=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    train_dataset = TinyImageNet(split='train', transform=transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                   shuffle=True, drop_last=True)

    val_dataset = TinyImageNet(split='val', transform=transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ]))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                                 drop_last=True)
    print("Number of training examples: ", len(train_dataset))
    print("Number of validation examples: ", len(val_dataset))

    hide_net = model.HideNet()
    hide_net.apply(weights_init)
    print_net(hide_net)

    reveal_net = model.RevealNet()
    reveal_net.apply(weights_init)
    print_net(reveal_net)

    criterion = nn.MSELoss()
    optimizerH = optim.Adam(hide_net.parameters(), lr=0.001, betas=(0.5, 0.999))
    schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)
    optimizerR = optim.Adam(reveal_net.parameters(), lr=0.001, betas=(0.5, 0.999))
    schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)

    smallestLoss = sys.maxsize
    for epoch in range(EPOCHS):
        train(train_dataloader, epoch, hide_net=hide_net, reveal_net=reveal_net, criterion=criterion)

        with torch.no_grad():
            hide_loss, reveal_loss, sum_loss = validate(val_dataloader, epoch, hide_net=hide_net, reveal_net=reveal_net, criterion=criterion)

        schedulerH.step(sum_loss)
        schedulerR.step(reveal_loss)

        if sum_loss < smallestLoss:
            smallestLoss = sum_loss
            torch.save(hide_net,
                       './checkpoints/hide_net_epoch_%d,sum_loss=%.6f,hide_loss=%.6f.pth' % (
                        epoch, sum_loss, hide_loss))
            torch.save(reveal_net,
                       './checkpoints/reveal_net_epoch_%d,sum_loss=%.6f,reveal_loss=%.6f.pth' % (
                        epoch, sum_loss, reveal_loss))
