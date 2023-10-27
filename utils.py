import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np



            

plt.style.use('ggplot')

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    path = f'./outputs'
    plt.figure(figsize=(10,7))
    plt.plot(
        train_acc, color='red', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{path}/accuracy.png')


    plt.figure(figsize=(10,7))
    plt.plot(
        train_loss, color='green', linestyle='--',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='orange', linestyle='--',
        label='validation loss'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path}/loss.png')




def sphere_plot(embeddings, labels, epoch, figure_path):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    ax.plot_surface(
        x, y, z,  
        rstride=1, 
        cstride=1, 
        color='w', 
        alpha=0.3, 
        linewidth=0
    )

    ax.scatter(
        embeddings[:,0], 
        embeddings[:,1], 
        embeddings[:,2], 
        c = labels, 
        s = 0.1,
        marker = '.'
    )   

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # ax.set_aspect("equal")
    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(f'{figure_path}/{epoch+1}.png')
