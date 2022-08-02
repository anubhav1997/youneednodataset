
import torch
from torchvision import datasets, transforms
import time


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    for num_workers in range(0,15,5):  # 遍历worker数
        kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=64, shuffle=True, **kwargs)



        start = time.time()
        for epoch in range(1, 5):
            for batch_idx, (data, target) in enumerate(train_loader): # 不断load
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end-start,num_workers))