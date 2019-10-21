import torch
import torchvision
import numpy as np
import torch.nn as nn
import tensorboard_helper
import pytorch_dataset
import train_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def mat_to_label(gt_matrix, threshold = 0.6):
    #print(gt_matrix)
    labels = np.zeros((0, 5))
    pc = []
    for i in range(gt_matrix.shape[1]):
      for j in range(gt_matrix.shape[2]):
        if gt_matrix[0, i, j] >= threshold:
          class_number = np.argmax([gt_matrix[5, i, j], gt_matrix[6, i, j], gt_matrix[7, i, j]])
          x1 = float(j*16 + gt_matrix[1, i, j] - gt_matrix[3, i, j]*64)
          y1 = float(i*16 + gt_matrix[2, i, j] - gt_matrix[4, i, j]*64)
          x2 = float(x1 + gt_matrix[3, i, j]*128)
          y2 = float(y1 + gt_matrix[4, i, j]*128)
          labels = np.vstack((labels, np.array([class_number, x1, y1, x2, y2])))
          pc.append(torch.sigmoid(gt_matrix[0, i, j]))

    return torch.from_numpy(labels).float(), pc

colors = ['r', 'b', 'g']


def iou_pairs(orig_label, pc):

  rem_label = []
  for i in range(orig_label.shape[0]):
    flag = 0
    for j in range(orig_label.shape[0]):
      if i!=j:
        xA = max(orig_label[i,1], orig_label[j,1])
        yA = max(orig_label[i,2], orig_label[j,2])
        xB = min(orig_label[i,3], orig_label[j,3])
        yB = min(orig_label[i,4], orig_label[j,4])

        inter = max(0, xB - xA) * max(0, yB - yA)

        boxAarea = (orig_label[i,3] - orig_label[i,1]) * (orig_label[i,4] - orig_label[i,2])
        boxBarea = (orig_label[j,3] - orig_label[j,1]) * (orig_label[j,4] - orig_label[j,2])

        ou = inter / float(boxAarea + boxBarea - inter)

        if ou >= 0.5 and pc[i] < pc[j]:
          flag = 1

    if flag == 0:
      rem_label.append(orig_label[i])

  rem_label = torch.stack(rem_label)
  return rem_label

def test():

    train_loader, test_loader = pytorch_dataset.get_loaders()
    model = train_model.YOLOish().to(device).double()
    # Download the model from the link
    # https://drive.google.com/file/d/1Oed91n7DKPdR0PwX_Y_hL0UGJMzYPO-P/view?usp=sharing
    checkpoint = torch.load('yolo_model.pt')
    model.load_state_dict(checkpoint['net'])
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device).double()
            labels = labels.to(device).double()
            outputs = model(images)
            # No post processing
            np_pred_label, l = mat_to_label(outputs[0,:,:,:],0.0)
            # Remove Pr < 0.6
            pred_label, pc = mat_to_label(outputs[0,:,:,:], 0.6)
            orig_label, pc_ = mat_to_label(labels[0,:,:,:], 1.0)
            # NMS
            if pred_label.shape[0] == 0:
              continue
            rem_label = iou_pairs(pred_label, pc)
            fig, ax = plt.subplots(1,4, figsize=(15,15))
            ax[0].imshow(images[0,:,:,:].cpu().permute(1, 2, 0) * 0.5 + 0.5)
            ax[1].imshow(images[0,:,:,:].cpu().permute(1, 2, 0) * 0.5 + 0.5)
            ax[2].imshow(images[0,:,:,:].cpu().permute(1, 2, 0) * 0.5 + 0.5)
            ax[3].imshow(images[0,:,:,:].cpu().permute(1, 2, 0) * 0.5 + 0.5)
            for i in range(orig_label.shape[0]):
              rect = patches.Rectangle((orig_label[i, 1],orig_label[i, 2]), \
                                 orig_label[i, 3]-orig_label[i, 1], \
                                 orig_label[i, 4]-orig_label[i, 2], \
                                 linewidth=2, \
                                 edgecolor=colors[int(orig_label[i, 0])], \
                                 facecolor='none')
              ax[0].add_patch(rect)

            for i in range(np_pred_label.shape[0]):
              rect = patches.Rectangle((np_pred_label[i, 1],np_pred_label[i, 2]), \
                                 np_pred_label[i, 3]-np_pred_label[i, 1], \
                                 np_pred_label[i, 4]-np_pred_label[i, 2], \
                                 linewidth=2, \
                                 edgecolor=colors[int(np_pred_label[i, 0])], \
                                 facecolor='none')
              ax[1].add_patch(rect)

            for i in range(pred_label.shape[0]):
              rect = patches.Rectangle((pred_label[i, 1],pred_label[i, 2]), \
                                 pred_label[i, 3]-pred_label[i, 1], \
                                 pred_label[i, 4]-pred_label[i, 2], \
                                 linewidth=2, \
                                 edgecolor=colors[int(pred_label[i, 0])], \
                                 facecolor='none')
              ax[2].add_patch(rect)

            for i in range(rem_label.shape[0]):
              rect = patches.Rectangle((rem_label[i, 1],rem_label[i, 2]), \
                                 rem_label[i, 3]-rem_label[i, 1], \
                                 rem_label[i, 4]-rem_label[i, 2], \
                                 linewidth=2, \
                                 edgecolor=colors[int(rem_label[i, 0])], \
                                 facecolor='none')
              ax[3].add_patch(rect)

            plt.show()

if __name__ == '__main__':
    test()
