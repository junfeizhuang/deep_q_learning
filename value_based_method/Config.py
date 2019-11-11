import gym

class config:
    n_state = 4
    n_action = 2
    n_mid_neuron = 50
    memory_capacity = 1000  
    epsilon_min = 0.01
    epsilon_decay = 0.5
    epsilon = 1.0 
    episodes = 2000
    gamma = 0.95
    lr_gamma = 0.5
    step_size = 500
    learning_rate = 0.0001
    plot_frequency = 20
    max_play_iters = 300
    batch_size = 32
    double_net = False
    use_dueling_net = False
    cuda = False

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self,key):
                setattr(self,key,value)
            else:
                raise ValueError('{} attribute is not in config class.'.format(str(key)))