from __future__ import print_function

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model import Generator, Discriminator
from datagen import ListDataset
from tqdm import tqdm


def train(args):
    img_list = args.image_list
    num_workers = args.num_workers
    batch_size = args.batch_size
    image_size = args.image_size
    nc = args.nc
    nz = args.nz
    ngf = args.ngf
    ndf = args.ndf
    epochs = args.epochs
    lr = args.lr
    beta1 = 0.5
    ngpu = 1

    device = torch.device("cuda:{}".format(args.gpu) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = ListDataset(img_list, transform, (image_size, image_size))
    dataloader = torch.utils.data.DataLoader(trainset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # dataroot = "./data/"
    # dataset = dset.ImageFolder(root=dataroot,
                               # transform=transforms.Compose([
                                   # transforms.Resize(image_size),
                                   # transforms.CenterCrop(image_size),
                                   # transforms.ToTensor(),
                                   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # dataloader = torch.utils.data.DataLoader(dataset,
                                             # batch_size=batch_size,
                                             # shuffle=True,
                                             # num_workers=num_workers)
    print("Dataloader satisfied.")

    # Generator
    netG = Generator(nz, ngf, nc, ngpu).to(device)
    if (device.type == "cuda") and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Discriminator
    netD = Discriminator(nc, ndf).to(device)
    if (device.type == "cuda") and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    print("Models satisfied!")

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    itrs = 0
    model_save_path = 'ckpt/*.pkl'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs('ckpt', exist_ok=True)
    print("start training")
    print("output directory: {}".format(args.output_dir))

    for epoch in range(epochs):
        for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc='Training'):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            itrs += 1

            # print log
            if (i % args.log_every == 0):
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # print(f"Loss D: {errD}, Loss G: {errG} ")

            # save image
            if itrs % args.save_every == 0:
                save_path = os.path.join(args.output_dir, str(epoch) + "_" + str(i) + ".jpg")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                fake = transforms.ToPILImage()(fake[0]).convert('RGB')
                fake.save(save_path)

            # adjust lr
            if itrs % 10000 == 0 and itrs < 300001:
                print("Adjusting learning rate..")
                lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = lr
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = lr

            # save model
            if itrs != 0 and itrs % 3000 == 0:
                print("Save model")
                torch.save(netG, './ckpt/netG_face.pkl')
                torch.save(netD, './ckpt/netD_face.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list", default="./img.list", type=str, help="")
    parser.add_argument("--num_workers", default=4, type=int, help="")
    parser.add_argument("--batch_size", default=32, type=int, help="")
    parser.add_argument("--image_size", default=512, type=int, help="")
    parser.add_argument("--gpu", default='0', type=str, help="")
    parser.add_argument("--nc", default=3, type=int, help="")
    parser.add_argument("--nz", default=100, type=int, help="")
    parser.add_argument("--ngf", default=64, type=int, help="")
    parser.add_argument("--ndf", default=64, type=int, help="")
    parser.add_argument("--epochs", default=50, type=int, help="")
    parser.add_argument("--lr", default=0.0002, type=float, help="")
    parser.add_argument("--output_dir", default="output_yifei", type=str, help="")
    parser.add_argument("--save_every", default=10, type=int, help="")
    parser.add_argument("--log_every", default=10000000000, type=int, help="")
    args = parser.parse_args()

    train(args)

