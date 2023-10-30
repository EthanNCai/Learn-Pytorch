import math

import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class WineDataset(Dataset):
    def __init__(self):
        # data loading
        dataset = pd.read_csv('../../Datas/wine.csv', skiprows=1)
        self.y = [[row[dataset.columns[0]].tolist()] for _, row in dataset.iterrows()]
        self.x = [row[dataset.columns[1:]].tolist() for _, row in dataset.iterrows()]
        self.y = torch.tensor(self.y)
        self.x = torch.tensor(self.x)
        self.n_samples = dataset.shape[0]

    def __getitem__(self, index):
        # dataset[index]
        return self.x[index], self.y[index]

    def __len__(self):
        # dataset.length
        return self.n_samples


batch_size = 4

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            print(f'epoch{epoch+1}/{num_epochs}, steps {i+1}/{n_iterations}, input: {x.shape}')


