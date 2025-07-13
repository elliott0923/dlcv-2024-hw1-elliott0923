import torch
from byol_pytorch import BYOL
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
#config  這邊定義好一些基本的東東 還有transform
image_size = 128
batch_size = 64
num_epochs = 30
learning_rate = 1e-4
num_classes = 65

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_img = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])

# load dataset 把作業提供的資料load進來
train_dir = './hw1_data/p1_data/office/train'
val_dir = './hw1_data/p1_data/office/val'

dataset = ImageFolder(train_dir,transform_img)
dataloader = DataLoader(
    dataset,    
    batch_size=batch_size,
    shuffle=True
    num_workers=4,
    pin_memory=True
)

# load pretrained resnet
resnet = models.resnet50(pretrained=False)  
resnet.load_state_dict(torch.load('improved-net.pt'))
resnet.fc = nn.Linear(2048, num_classes)
resnet = resnet.to(device)




# training
for _ in range(num_epochs):
    for images, _ in dataloader:
        opt.zero_grad()
        loss = learner(images)
        images = images.to(device)
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        print(f'Loss: {loss.item()}')

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')