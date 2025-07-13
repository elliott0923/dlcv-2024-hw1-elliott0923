import torch
from byol_pytorch import BYOL
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

#config
image_size = 128
batch_size = 128
num_epochs = 100
learning_rate = 3e-4
data_dir = './mini-imagenet/train'

resnet = models.resnet50(pretrained=False)

learner = BYOL(
    resnet,
    image_size = image_size,
    hidden_layer = 'avgpool'
)
opt = torch.optim.Adam(learner.parameters(), lr=3e-4) #BYOL uses Adam optimizer

transform_img = transforms.Compose([
    transforms.RandomResizedCrop(image_size),        # 隨機裁切並resize，模擬不同構圖
    transforms.RandomHorizontalFlip(),               # 隨機左右翻轉，讓模型不只記得方向
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),       # 顏色隨機擾動，模擬光線環境變化
    transforms.RandomGrayscale(p=0.2),               # 20% 機率轉灰階，讓模型不依賴顏色
    transforms.GaussianBlur(3),                      # 模糊化圖片，模擬拍攝模糊
    transforms.ToTensor()                         # 轉成 tensor
])
# load dataset
data_dir = './hw1_data/p1_data/mini/train'
dataset = ImageFolder(data_dir,transform_img)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)
learner = learner.to(device)

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