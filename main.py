import torch
import sys
import time
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
import matplotlib.pyplot as plt


EPOCHS = 3
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

        print('[%d/%d][%d/%d] hide_loss: %.4f reveal_loss: %.4f sum_loss: %.4f' % (
            epoch + 1, EPOCHS, i + 1, len(train_loader),
            err_hide.item(), err_reveal.item(), err_sum.item()))

        if i % 7 == 0:
            save_image_results(this_batch_size, cover_img, container_img.data, secret_img,
                               rev_secret_img.data, epoch + 1, i + 1, './training')

    return hide_losses, reveal_losses, sum_losses


def validate(val_loader, epoch, hide_net, reveal_net, criterion):
    hide_losses = []
    reveal_losses = []
    sum_losses = []

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
        reveal_losses.append(err_reveal.item())

        beta_err_reveal = BETA * err_reveal
        err_sum = err_hide + beta_err_reveal
        sum_losses.append(err_sum.item())

        if i % 7 == 0:
            save_image_results(this_batch_size, cover_img, container_img.data, secret_img,
                               rev_secret_img.data, epoch + 1, i + 1, './validation')

    return hide_losses, reveal_losses, sum_losses


def save_image_results(this_batch_size, cover_image, container_image, secret_image, revealed_image, epoch, i, save_path):
    covers_containers = torch.cat([cover_image, container_image], 0)
    secrets_reveals = torch.cat([secret_image, revealed_image], 0)
    image_results = torch.cat([covers_containers, secrets_reveals], 0)
    image_results_file = "%s/result_images_epoch%d_batch%d.png" % (save_path, epoch, i)
    vutils.save_image(image_results, image_results_file, nrow=this_batch_size, padding=1, normalize=True)


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
    ]), images_per_class_train=3)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                                   shuffle=True, drop_last=True)

    val_dataset = TinyImageNet(split='val', transform=transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ]), num_val_images=30)
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

    train_hide_losses = []
    train_reveal_losses = []
    total_elapsed_seconds = 0
    smallestLoss = sys.maxsize
    for epoch in range(EPOCHS):
        start = time.time()

        print("----- Training: START -----")
        hide_losses, reveal_losses, sum_losses = train(train_dataloader, epoch, hide_net=hide_net, reveal_net=reveal_net, criterion=criterion)
        train_hide_losses.append(hide_losses)
        train_reveal_losses.append(reveal_losses)

        avg_hide_loss_t = np.mean(hide_losses)
        avg_reveal_loss_t = np.mean(reveal_losses)
        avg_sum_loss_t = np.mean(sum_losses)

        train_summary = "epoch learning rate: optimizer_hide_lr = %.8f optimizer_reveal_lr = %.8f" % (
            optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
        train_summary = train_summary + "epoch_avg_hide_loss=%.6f epoch_avg_reveal_loss=%.6f epoch_avg_sum_loss=%.6f" % (
            avg_hide_loss_t, avg_reveal_loss_t, avg_sum_loss_t)
        print(train_summary)

        plt.plot(train_hide_losses)
        plt.title('Training Hide Loss')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()

        plt.plot(train_reveal_losses)
        plt.title('Training Reveal Loss')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()
        print("----- Training: END -----")

        print("----- Validation: START -----")
        with torch.no_grad():
            hide_losses, reveal_losses, sum_losses = validate(val_dataloader, epoch, hide_net=hide_net, reveal_net=reveal_net, criterion=criterion)

        avg_hide_loss_v = np.mean(hide_losses)
        avg_reveal_loss_v = np.mean(reveal_losses)
        avg_sum_loss_v = np.mean(sum_losses)

        val_summary = "validation[%d] avg_hide_loss = %.6f avg_reveal_loss = %.6f avg_sum_loss = %.6f" % (
            epoch + 1, avg_hide_loss_v, avg_reveal_loss_v, avg_sum_loss_v)
        print(val_summary)
        print("----- Validation: END -----")

        schedulerH.step(avg_sum_loss_v)
        schedulerR.step(avg_reveal_loss_v)

        elapsed = time.time() - start
        total_elapsed_seconds += elapsed
        print("Epoch %d: elapsed seconds %d" % (epoch + 1, elapsed))

        if avg_sum_loss_v < smallestLoss:
            smallestLoss = avg_sum_loss_v
            torch.save(hide_net,
                       './checkpoints/hide_net_epoch_%d,sum_loss=%.6f,hide_loss=%.6f.pth' % (
                           epoch, avg_sum_loss_v, avg_hide_loss_v))
            torch.save(reveal_net,
                       './checkpoints/reveal_net_epoch_%d,sum_loss=%.6f,reveal_loss=%.6f.pth' % (
                           epoch, avg_sum_loss_v, avg_reveal_loss_v))

    print("Total elapsed seconds %d" % total_elapsed_seconds)
