# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import Dataset, DataLoader
# import torch
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import os
# import copy
#
# # TensorBoard için bir SummaryWriter oluşturun
# writer = SummaryWriter("log_directory")
#
# loss_recorder = []
#
# with torch.no_grad():
#     for data in dataloader:  # Veri kümesini döngüüsü
#         inputs, labels = data
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#
#         # Yanlış tahmin
#         wrong_mask = predicted != labels
#         for i in range(len(wrong_mask)):
#             if wrong_mask[i]:
#                 writer.add_image(f'Wrong_Prediction_{i}', inputs[i], global_step=i)
#                 writer.add_text(f'True_Label_{i}', str(labels[i]), global_step=i)
#                 writer.add_text(f'Predicted_Label_{i}', str(predicted[i]), global_step=i)
#
#
# loss_recorder = []
# for epoch in range(num_epochs):
#     running_loss = 0.0
#
#     for i, data in enumerate(dataloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#         if i % log_interval == 0:
#             current_iteration = epoch * len(dataloader) + i
#             loss_recorder.append((current_iteration, loss.item()))
#             writer.add_scalar('Training_Loss', loss.item(), current_iteration)
#
#     print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')
#
#
# writer.close()
