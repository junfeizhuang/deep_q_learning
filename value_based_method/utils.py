import numpy as np
import matplotlib.pyplot as plt

def plot(score_list,loss_list,plot_frequency):
    length = len(score_list[::plot_frequency])
    es = list(range(length))
    scores, losses = list(), list()
    for i in range(length):
        scores.append(np.mean(score_list[i:i+plot_frequency]))
        losses.append(np.mean(loss_list[i:i+plot_frequency]))
    plt.figure(figsize=(9, 5))
    plt.title("Result")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    # episode_data_x = data_x_update(loss_data_x, episode_data_x)
    plt.plot(es, scores, color='green', label='Reward')
    plt.legend()
    plt.savefig('reward.jpg')

    plt.figure(figsize=(9, 5))
    plt.title("Result")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    # episode_data_x = data_x_update(loss_data_x, episode_data_x)
    plt.plot(es, losses, color='yellow', label='Loss')
    plt.legend()
    plt.savefig('Loss.jpg')