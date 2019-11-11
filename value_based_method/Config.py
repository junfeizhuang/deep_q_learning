import gym

class config:
    n_state = 4
    n_action = 2
    memory_capacity = 1000  
    epsilon_min = 0.01
    epsilon_decay = 0.5
    epsilon = 1.0 
    episodes = 300
    gamma = 0.95
    lr_gamma = 0.5
    step_size = 100
    learning_rate = 0.001
    plot_frequency = 20
    max_play_iters = 300
    batch_size = 32

cfg = config()
        