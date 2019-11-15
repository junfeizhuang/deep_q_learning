import torch
import numpy as np
import matplotlib.pyplot as plt


def clip(t,t_min,t_max):
    t = torch.FloatTensor(t)
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def creat_figure(x_label, y_label):
    plt.figure(figsize=(18, 5))
    plt.title("Result")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

def draw_curve(original_list, plot_frequency, name, color, idx):
    length = len(original_list[::plot_frequency])
    es = list(range(length))
    new_list = list()
    for i in range(length):
        new_list.append(np.mean(original_list[i:i+plot_frequency]))
    plt.subplot(1, 3, idx)
    plt.plot(es, new_list, color=color, label= name)

def plot(name, scores, critic_loss, actor_loss, plot_frequency):
    creat_figure("Episode","Score")
    draw_curve(scores,plot_frequency,'score','red',1)
    plt.legend()
    draw_curve(critic_loss,plot_frequency,'critic_loss','blue',2)
    plt.legend()
    draw_curve(actor_loss,plot_frequency,'actor_loss','green',3)
    plt.legend()
    plt.savefig(name + '.jpg')

    