import numpy as np
import matplotlib.pyplot as plt

def creat_figure(x_label, y_label):
    plt.figure(figsize=(9, 5))
    plt.title("Result")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

def draw_curve(original_list, plot_frequency, name, color):
    length = len(original_list[::plot_frequency])
    es = list(range(length))
    new_list = list()
    for i in range(length):
        new_list.append(np.mean(original_list[i:i+plot_frequency]))
    plt.plot(es, new_list, color=color, label= name)
    

def plot(plot_frequency, **kwargs):
    dqn_score_list = kwargs['dqn_score_list']
    ddqn_score_list = kwargs['ddqn_score_list']
    dueling_score_list = kwargs['dueling_score_list']
    dqn_loss_list = kwargs['dqn_loss_list']
    ddqn_loss_list = kwargs['ddqn_loss_list']
    dueling_loss_list = kwargs['dueling_loss_list']
    
    creat_figure("Episode","Score")
    draw_curve(dqn_score_list,plot_frequency,'dqn_score','red' )
    draw_curve(ddqn_score_list,plot_frequency,'ddqn_score','blue' )
    draw_curve(dueling_score_list,plot_frequency,'dueling_score','green' )
    plt.legend()
    plt.savefig('Reward.jpg')
    
    creat_figure("Episode","Loss")
    draw_curve(dqn_loss_list,plot_frequency,'dqn_loss','red')
    draw_curve(ddqn_loss_list,plot_frequency,'ddqn_loss','blue')
    draw_curve(dueling_loss_list,plot_frequency,'dueling_loss','green' )
    plt.legend()
    plt.savefig('Loss.jpg')