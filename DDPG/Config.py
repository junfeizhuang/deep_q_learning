class config:
    max_epoches = 1000
    max_play_iters = 500
    map_action_upper = 1.0 # manual setting according tanh output [-1,1]
    map_action_low = -1.0 # manual setting according tanh output [-1,1]
    mid_critic = 64
    mid_actor = 64
    memory_capacity = 5000
    epsilon_min = 0.01
    epsilon_decay = 0.9
    epsilon = 0.8
    gamma = 0.99
    learning_rate = 0.001
    tau_actor = 0.1
    tau_critic = 0.1
    plot_frequency = 20
    batch_size = 64