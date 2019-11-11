import numpy as np

class RL_brain(object):
    def __init__(self, n_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.n_actions = n_actions
        self.q_table = np.zeros((1,n_actions))
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        
    def choose_action(self, s_idx):
        self.check_state_existence(s_idx)
        if s_idx == 0:
            return 'right'
        q_action = self.q_table[s_idx-1,:]
        if (np.random.uniform(0,1)> self.epsilon) or (q_action ==0).all():
            action_idx = np.random.choice([0,1])
        else:
            action_idx = np.argmax(np.array(q_action))
        if action_idx == 0:
            return 'left'
        elif action_idx ==1:
            return 'right'
        else:
            raise ValueError('invaild action_idx')

    def learn(self, s_idx, action_idx , s_idx_, reward, done):
        self.check_state_existence(s_idx_)
        if not done:
            q_target = reward + self.gamma*self.q_table[s_idx_-1,:].max()
        else:
            q_target = reward
        q_predict = self.q_table[s_idx-1, action_idx]
        self.q_table[s_idx-1, action_idx] += self.lr*(q_target - q_predict)

    def check_state_existence(self,s_idx):
        #print(s_idx > self.q_table.shape[0] - 1)
        if s_idx > self.q_table.shape[0] - 1:
            self.q_table = np.vstack((self.q_table,\
                np.zeros(self.n_actions))) 


class Environment(object):
    def __init__(self, n_status):
        # treasure format -----T  T: treasure
        self.n_status = n_status
        self.init_s_idx = 0
        self.action_value_dict = {'left':-1, 'right':+1}

    def reset(self):
        env = ['o'] + ['-'] * (self.n_status-2) + ['T']
        print(''.join(env))
        return 0
        

    def step(self,s_idx,action):
        env = ['-'] + ['-'] * (self.n_status-2) + ['T']
        env[s_idx] = 'o'
        print(''.join(env))
        s_idx += self.action_value_dict[action]
        s_idx_ = s_idx
        if s_idx_ == self.n_status-1:
            done, reward = True, +5
        else:
            done, reward = False, 0
        return s_idx_, reward, done

def main():
    env = Environment(10)
    brain = RL_brain(2)
    for epoch in range(20):
        s_idx = env.init_s_idx
        done = False
        step = 0
        while not done:
            action = brain.choose_action(s_idx)
            action_idx = list(env.action_value_dict.keys()).index(action)
            s_idx_, reward, done = env.step(s_idx,action)
            brain.learn(s_idx,action_idx,s_idx_,reward,done)
            s_idx = s_idx_
            step+=1
        print('Exploer success: {} epoch used {} steps.'.format(epoch,step))

if __name__ == '__main__':
    main()