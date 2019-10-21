import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torchvision
import numpy as np
import pytorch_dataset

def gt_to_label(gt_matrix):
    gt_matrix = gt_matrix[0, :, :, :]
    labels = np.zeros((0, 5))
    for i in range(gt_matrix.shape[1]):
      for j in range(gt_matrix.shape[2]):
        if gt_matrix[0, i, j] == 1.0:
          class_number = np.argmax([gt_matrix[5, i, j], gt_matrix[6, i, j], gt_matrix[7, i, j]])
          x1 = float(j*16 + gt_matrix[1, i, j] - gt_matrix[3, i, j]*64)
          y1 = float(i*16 + gt_matrix[2, i, j] - gt_matrix[4, i, j]*64)
          x2 = float(x1 + gt_matrix[3, i, j]*128)
          y2 = float(y1 + gt_matrix[4, i, j]*128)
          labels = np.vstack((labels, np.array([class_number, x1, y1, x2, y2])))
    return torch.from_numpy(labels).float()


def dataset_test():
    colors = ['r', 'b', 'g']
    train_loader, test_loader = pytorch_dataset.get_loaders()
    fig, ax = plt.subplots(1)
    image, output_label = next(iter(train_loader))
    ax.imshow(image[0,:,:,:].permute(1, 2, 0) * 0.5 + 0.5)

    gt_mat = torch.zeros(1, 8, 8, 8)
    gt_mat[0, :, :, :] = output_label[0,:,:,:]
    orig_label = gt_to_label(gt_mat)
    for i in range(orig_label.shape[0]):
        rect = patches.Rectangle((orig_label[i, 1],orig_label[i, 2]), \
                             orig_label[i, 3]-orig_label[i, 1], \
                             orig_label[i, 4]-orig_label[i, 2], \
                             linewidth=2, \
                             edgecolor=colors[int(orig_label[i, 0])], \
                             facecolor='none')
        ax.add_patch(rect)
    plt.show()
    channel_names = ['P(Objectness)','x','y','w','h','P(Pedestrian)','P(Traffic Lights)','P(Car)']
    fig, ax = plt.subplots(1, 8, figsize=(20, 20))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.5)
    for i in range(8):
        val = ax[i].imshow(gt_mat[0,i,:,:], cmap='jet')
        print(gt_mat[0,i,:,:])
        ax[i].set_title(channel_names[i])
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(val, cax=cax)
    plt.show()

if __name__ == '__main__':
    dataset_test()
