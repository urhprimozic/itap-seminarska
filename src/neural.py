# 
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import transforms
from tqdm import tqdm
import pickle
from utils import get_data, get_reference, reference_forest, reference_buildings, array_4D

CHUNK_SIZE = 10
def reference_target(ref):
    # return reference_forest(ref)
    return reference_buildings(ref)

# two pixle-models:
with open('../img/eval-features/obj_svc', 'rb') as f_svc:
    svc = pickle.load(f_svc)
with open('../img/eval-features/obj_knn', 'rb') as f_knn:
    knn = pickle.load(f_knn)

# data
train = [2, 3, 4,  7, 8, 9,  12, 13, 14,  17, 19]
test = [1, 6, 11, 16]
features = [5, 12, 11,  6, 13,1,2,3]
# namesto celotnega dataseta vzamemo samo featurse
X_train = np.concatenate([get_data(i) for i in train])[..., features]
y_train = np.concatenate(
    [reference_target(get_reference(i)) for i in train])[..., features]
X_test = np.concatenate([get_data(i) for i in test])[..., features]
y_test = np.concatenate([reference_target(get_reference(i)) for i in test])[..., features]
# dodaj rezultate pixel-modelov
f_svc_train = svc.predict(X_train, y_train)
f_knn_train = knn.predict(X_train, y_train)
f_svc_test = svc.predict(X_test, y_test)
f_knn_test = knn.predict(X_test, y_test)
# zlimaš skupej
X_train_extra = np.zeros((X_train.shape[0], X_train.shape[1] + 2))
X_train_extra[:,:-2] = X_train
X_train[:,-2] = f_svc_train
X_train[:,-1] = f_knn_train
X_train = array_4D(X_train_extra)
X_test_extra = np.zeros((X_test.shape[0], X_test.shape[1] + 2))
X_test_extra[:,:-2] = X_test
X_test[:,-2] = f_svc_test
X_test[:,-1] = f_knn_test
X_test = array_4D( X_test_extra)


model = nn.Sequential(
    #
    nn.Conv2d(len(features) + 2, 4, 3, stride=1),
    nn.ReLU(),
    # akumulacijska plast
    # velikost jedra
    nn.MaxPool2d(2),
    # še enva konvolucijka
    nn.Conv2d(4, 2,2, stride=1),
    nn.MaxPool2d(2),

    # 2d v 1d
    nn.Flatten(start_dim=1),
    # fully connected
    # vhodna velikost (more se ujemat) in izhodna velikost
    nn.Linear(8, 2)
)
# počekiramo če maš gpuuuuuuu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


epochs = 10#100
for epoch in tqdm(range(epochs), total=epochs):
    train_loses = []
    pravilni = 0
    for (images, label) in train_dl:
        images = images.to(device=device)
        print(images.shape)
        label=label.to(device=device)
        optimizer.zero_grad()
        #napoved
        output = model(images)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loses.append(loss.item())
        # natančnmost
        napovedi = output.argmax(dim=1)
        pravilni += sum(napovedi == label)
        natancnost = pravilni/len(train_ds)
        print("accuracy: ", natancnost)