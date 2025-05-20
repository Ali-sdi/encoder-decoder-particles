import time
from model import UNet
import os
from tkinter import filedialog
from PIL import Image
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import StepLR
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
# ignore warnings, i.e those annoying ones that appear for every run
import warnings
warnings.filterwarnings("ignore")

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, dir):

        self.dir = dir
        self.imgs = list(sorted(os.listdir(os.path.join(dir, 'image'))))
        self.labels = list(sorted(os.listdir(os.path.join(dir, 'label'))))
        self.tf = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                               ])



    def __getitem__(self, idx):

        img = os.path.join(self.dir, 'image', self.imgs[idx])
        label = os.path.join(self.dir, 'label', self.labels[idx])

        img = Image.open(img).convert('RGB')
        label = Image.open(label).convert('RGB')


        img = self.tf(img)
        label = self.tf(label)


        return img, label


    def __len__(self):
        return len(self.imgs)

TrainDataset = MyDataset('D:\\USERS\\SEYDI\\dataset1\\train')
validDataset = MyDataset('D:\\USERS\\SEYDI\\dataset1\\valid')



# -------------------------------------#
#           CREATE THE AE NETWORK       #
# -------------------------------------#

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),nn.ReLU(),
            # nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        recon_x = self.decoder(x)

        return recon_x

def loss_fn(recon_x, x):
    #loss = functional.mse_loss(recon_x, x)

    BCELoss = functional.binary_cross_entropy(recon_x, x)
    #KL_Div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCELoss
# -----------------------------#
#       GET THE DATASET       #
# -----------------------------#


# Create data loaders
train_dataloader = torch.utils.data.DataLoader(TrainDataset, batch_size=32)
valid_dataloader = torch.utils.data.DataLoader(validDataset, batch_size=32)


# ------------------------------------#
#           SHOW THE IMAGES          #
# ------------------------------------#

def showImage(img, img_recon, epoch):
    # unnormalize
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    #img = img/2 + 0.5
    img = img.numpy()
    plt.title(label="Original Epoch: #" + str(epoch))
    plt.imshow(np.transpose(img, (1, 2, 0)))
    fig.add_subplot(1, 2, 2)
    #img_recon = img_recon/2 + 0.5
    img_recon = img_recon.numpy()
    plt.title(label="Reconstruction Epoch: #" + str(epoch))
    plt.imshow(np.transpose(img_recon, (1, 2, 0)))
    plt.show(block=True)


# print(len(train_dataloader))
# print(len(TrainDataset))
# print(len(train_dataloader.dataset))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}\n")


#model = VAE().to(device)
model = UNet().to(device)

# pretty table
#summary(model, (1, 128, 128))

learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 75
scheduler = StepLR(optimizer, step_size=25, gamma=0.1) #second line
valid_loss = 0.0
running_loss = 0.0


losses = []
validation_loss = []

ans=input('if you  want to train the model enter "y" \n'
          'OTHERWISE press any other keys:')
if ans=='y':
    t0 = time.time()
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-----------------------------------")
        model.train()
        for idx, (images, labels) in enumerate(train_dataloader): #idx = batch size

            images = images.to(device)
            labels = labels.to(device)

            recon_x = model(images)


            loss = loss_fn(recon_x , labels)
            #loss = losss(recon_x, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            #running_loss = running_loss /len(TrainDataset)

        print("Epoch {} \t Training loss:{:.6f}\t ".format(t+1, running_loss/len(train_dataloader)))
        print(optimizer.param_groups[0]['lr'])

        losses.append(np.array(running_loss/len(train_dataloader)))
        running_loss = 0.0
        #scheduler.step()


        #VALIDATION
        model.eval()
        for idx, (images, labels) in enumerate(valid_dataloader):

            images = images.to(device)
            labels = labels.to(device)

            recon_valid = model(images)


            val_loss = loss_fn(recon_valid , labels)
            valid_loss += val_loss.item()
        print("\t validation loss:{:.6f}\t ".format(valid_loss / len(valid_dataloader)))

        validation_loss.append(np.array(valid_loss / len(valid_dataloader)))
        valid_loss = 0.0

    t1 = time.time()
    duration = t1 - t0
    print('duration: ',duration)

    epochss = range(1, epochs+1)
    fig, ax = plt.subplots()
    ax.plot(epochss,losses)
    ax.plot(epochss,validation_loss)
    plt.title('Average loss curve')
    plt.xlabel('epochs')
    plt.ylabel('LOSS')
    plt.legend(['Training', 'Validation'])
    plt.show()

    showImage(torchvision.utils.make_grid(images.to("cpu")), torchvision.utils.make_grid(recon_x.to("cpu")),
            t + 1)

    torch.save(model.state_dict(), "MODEL_d25.pt")
    print("Model saved.")
##################################################################### #######################################################  TEST
print("Loading previously saved model..")
device = 'cpu'
#model = VAE().to(device)
model = UNet().to(device)

model.load_state_dict(torch.load("MODEL_d13.pt"))
#model.load_state_dict(torch.load("MNIST_VAE_MODEL.pt"))
#print(model)
model.eval()
print("Model has been loaded.")

transform=transforms.Compose([

    transforms.ToTensor()
])
input_test_image_path= filedialog.askopenfilename(filetypes=[("Image files", '.png')])
input_image = Image.open(input_test_image_path).convert('RGB')
input_image = transform(input_image).unsqueeze(0)


#feed input data to the model for inference
with torch.no_grad():
    output = model(input_image)
# for tensor in output:
#     print(tensor.shape)

output = (output[0])
output = torch.squeeze(output,0)
output = output.permute(1,2,0)
output = np.asarray(output)



input_image = torch.squeeze(input_image,0)
input_image = input_image.permute(1,2,0)
input_image = np.asarray(input_image)

test_image_path= filedialog.askopenfilename(filetypes=[("Image files", '.png')])
test_image = Image.open(test_image_path).convert('RGB')
test_image = transform(test_image).unsqueeze(0)

test_image = torch.squeeze(test_image,0)
test_image = test_image.permute(1,2,0)
test_image = np.asarray(test_image)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

axes[0].imshow(input_image)
axes[0].axis('off')
axes[1].imshow(output)
axes[1].axis('off')
axes[2].imshow(test_image)
axes[2].axis('off')

axes[0].set_title("Porous media")
axes[1].set_title("Reconstructed image")
axes[2].set_title("Ground truth ")


plt.show()
# plt.gca().axes.get_yaxis().set_visible(False)
# plt.gca().axes.get_xaxis().set_visible(False)
from torchvision.utils import save_image

output = transform(output).squeeze(0)
save_image(output, 'img1.png')
