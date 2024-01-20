import new_belief_function as bf
import numpy as np
import torch
# from tqdm import tqdm
# import pdb
import math
from enum import Enum
import copy

RATIONAL_FACTOR = 5  
Print_interval = 20
Regularzation_coeff = 0.01
num_loop = 100
total_prob = 0

class Animal(Enum):
    RABBIT = 0
    SHEEP = 1

class Action(Enum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

def log_prob_loss(score, index):
    prob = torch.softmax(RATIONAL_FACTOR * score, dim = 0)
    return -torch.log(prob[index])

def regularzation_loss(para, type):
    sum = 0
    if type == "L2":
        for num in para:
            sum += torch.sum(torch.pow(num, 2))
    elif type == "L1":
        for num in para:
            sum += torch.sum(torch.abs(num))
    return sum

def dist_loss(k, c, avg_dist, true_time):
    return torch.abs( k * avg_dist + c - true_time)

def learning1(data, belief_data, final_prey, final_time, epoch, exp_num, learning_rate1, para,total_prob, threshold):
    gamma, k, c, ratio = para
    total_loss = 0.0
    if final_prey == Animal.RABBIT or final_prey == Animal.SHEEP:
        for j in range(final_time):
            hunter_layer = data[j, 0, ...]
            belief_rabbit = belief_data[j, 0, ...]
            belief_sheep = belief_data[j, 1, ...]
            hunter_pos = bf.Hunter_find(hunter_layer)
            if final_prey == Animal.RABBIT:
                avg_time = bf.Dist_calc(belief_rabbit, k, c, threshold, hunter_pos[0], hunter_pos[1])
            elif final_prey == Animal.SHEEP:
                avg_time = bf.Dist_calc(belief_sheep, k, c, threshold, hunter_pos[0], hunter_pos[1])
            d_loss = torch.abs(avg_time - (final_time-j)) + Regularzation_coeff * regularzation_loss([k, c],"L1")
            total_loss += d_loss
            # d_loss.backward()
        total_loss.backward()
        total_loss = total_loss.item()
        # print(total_loss)
        with torch.no_grad():
            k -= learning_rate1 * k.grad/final_time
            c -= learning_rate1 * c.grad/final_time
        k.grad.zero_()
        c.grad.zero_()
        #if epoch==1:
            #print(f'k:{bf.Print_format((k**2).item())} c:{bf.Print_format(c.item())}')
    else:
        print("WARNING: hunt nothing in this episode!")
    return gamma, k, c, ratio, total_prob
def learning2(data, belief_data, final_prey, final_time, epoch, exp_num, learning_rate2, para,check_flag,total_prob, threshold):
    gamma, k, c, ratio = para
    total_loss = 0.0
    avg_prob = 0.0
    for j in range(final_time):
        hunter_layer = data[j, 0, ...]
        belief_rabbit = belief_data[j, 0, ...]
        belief_sheep = belief_data[j, 1, ...]
        true_action = Find_true_action(hunter_pos_list[j], hunter_pos_list[j+1])
        hunter_pos = bf.Hunter_find(hunter_layer)
        score_func = bf.Belief_pred(belief_rabbit, belief_sheep,hunter_pos)
        score = score_func(1 / (1 + gamma**2), (k), c, threshold, ratio**2) # 为了控制γ在0~1，k, ratio为正
        loss1 = log_prob_loss(score, true_action.value)
        loss2 = regularzation_loss([gamma, ratio], "L2")
        total_loss += loss1 + Regularzation_coeff * loss2
        avg_prob += torch.exp(-loss1).item()
    total_loss.backward()
    avg_prob /= final_time
    if check_flag :
        total_prob += avg_prob
        #print(f'Epoch [{epoch+1}/{exp_num}], Prob: {total_loss}.')
        k.grad.zero_()
        c.grad.zero_()
        gamma.grad.zero_()
        ratio.grad.zero_()
    else:
        """
        with torch.no_grad():
           k -= learning_rate2 * k.grad/final_time
           c -= learning_rate2 * c.grad/final_time
        k.grad.zero_()
        c.grad.zero_()
        """
        # 更新模型参数
        with torch.no_grad():
            gamma -= learning_rate2 * gamma.grad/final_time
            ratio -= learning_rate2 * ratio.grad/final_time

        # 清零梯度
        gamma.grad.zero_()
        ratio.grad.zero_()
    #if epoch == 1:
        #print(f'gamma:{bf.Print_format((1 / (1 + gamma**2)).item())} ratio:{bf.Print_format((ratio**2).item())} pred_log_gamma^(ratio){math.log((ratio**2).item())/math.log((1 / (1 + gamma**2)).item())}')

    return gamma, k, c, ratio, total_prob    

def Find_final_prey(state):
    hunter_pos = np.where(state[0] == 1)
    if state[2][hunter_pos] == 1:
        return Animal.SHEEP
    if state[1][hunter_pos] == 1:
        return Animal.RABBIT
    return None

def Find_true_action(curr_pos, next_pos):
    diff = next_pos - curr_pos
    if diff[0] == -1:
        return Action.UP
    elif diff[0] == 1:
        return Action.DOWN
    elif diff[1] == -1:
        return Action.LEFT
    elif diff[1] == 1:
        return Action.RIGHT
    else:
        return Action.STAY
    
experiments = [
    ["full", 0.95, "exp1704541232", "stop"],
    ["partial", 0.95, "exp1704541309", "stop"],
    ["full", 0.5, "exp1704541352", "stop"],
    ["partial", 0.5, "exp1704541394", "stop"],
    ["full", 0.95, "exp1704597158", "move"],
    ["partial", 0.95, "exp1704597224", "move"],
    ["full", 0.5, "exp1704597341", "move"],
    ["partial", 0.5, "exp1704598063", "move"]
]

learn_num = 5
check_num = 95
loop_index = list(range(2, 102))*2
record_dict_item = {'k':[], 'c':[], 'gamma': [], 'ratio': [], 'pred': [], 'avg_prob': []}
record_dict = [copy.deepcopy(record_dict_item) for _ in range(8)]
for l in range(10):
    for T in range(8):
        print(f'obs_radius:{experiments[T][0]} gamma:{experiments[T][1]} animal:{experiments[T][3]}')
        Learning_rate1 = 0.01
        Learning_rate2 = 0.1
        data_list = [np.load(f'./record/{experiments[T][2]}/ep{loop_index[i]}.npy') for i in range(learn_num*l, learn_num*(l+1))]
        check_list = [np.load(f'./record/{experiments[T][2]}/ep{loop_index[i]}.npy') for i in range(learn_num*(l+1), learn_num*(l+1) + check_num)]
        gamma = torch.randn(1, requires_grad=True)
        k = torch.randn(4, dtype=torch.float32, requires_grad=True)
        c = torch.randn(1, dtype=torch.float32, requires_grad=True)
        ratio = torch.tensor([1], dtype=torch.float32, requires_grad=True)

        # final_prey = [Find_final_prey(data[-1]) for data in data_list]
        # belief_data = [bf.Belief_init(data, 21) for data in data_list]
        # final_time = []

        for _ in range(num_loop):    
            Learning_rate1*=0.99
            for index,data in enumerate(data_list):
                final_prey = Find_final_prey(data[-1])
                hunter_pos_list = [bf.Hunter_find(data[j, 0]) for j in range(data.shape[0])]
                if final_prey is None:
                    assert data.shape[0] > 100, 'error'
                    continue
                if experiments[T][0] == 'full':
                    threshold = 21
                else:
                    threshold = 2
                belief_data = bf.Belief_init(data, threshold)
                final_time = data.shape[0] - 1
                gamma,k,c,ratio,total_prob = learning1(data, belief_data, final_prey, final_time, index+1, 
                                                    learn_num, Learning_rate1, [gamma, k, c, ratio], 
                                                    total_prob, threshold)
            pass
        for _ in range(num_loop):
            Learning_rate2*=0.999
            for index,data in enumerate(data_list):
                final_prey = Find_final_prey(data[-1])
                hunter_pos_list = [bf.Hunter_find(data[j, 0]) for j in range(data.shape[0])]
                if final_prey is None:
                    assert data.shape[0] > 100, 'error'
                    continue
                if experiments[T][0] == 'full':
                    threshold = 21
                else:
                    threshold = 2
                belief_data = bf.Belief_init(data, threshold)
                final_time = data.shape[0] - 1
                gamma,k,c,ratio,total_prob = learning2(data, belief_data, final_prey, final_time, index+1, 
                                                    learn_num, Learning_rate2, [gamma, k, c, ratio], 0, 
                                                    total_prob, threshold)
                pass
            pass
        print('k:', k)
        print('c:', c)
        print('gamma:', 1 / (1 + gamma**2))
        print('ratio:', ratio**2)
        print(f'true_log_gamma^(ratio){math.log(3)/math.log(experiments[T][1])}')
        print(f'pred_log_gamma^(ratio){math.log((ratio**2).item())/math.log((1 / (1 + gamma**2)).item())}')
        record_dict[T]['k'].append(list(k))
        record_dict[T]['c'].append(c.item())
        record_dict[T]['gamma'].append((1 / (1 + gamma**2)).item())
        record_dict[T]['ratio'].append((ratio**2).item())
        record_dict[T]['pred'].append(math.log((ratio**2).item())/math.log((1 / (1 + gamma**2)).item()))

        total_prob=0.0
        for index,data in enumerate(check_list):
            final_prey = Find_final_prey(data[-1])
            hunter_pos_list = [bf.Hunter_find(data[j, 0]) for j in range(data.shape[0])]
            if final_prey is None:
                assert data.shape[0] > 100, 'error'
                continue
            if experiments[T][0] == 'full':
                threshold = 21
            else:
                threshold = 2
            belief_data = bf.Belief_init(data, threshold)
            final_time = data.shape[0] - 1
            gamma,k,c,ratio,total_prob = learning2(data, belief_data, final_prey, final_time, index+1, 
                                                check_num-learn_num, Learning_rate2, [gamma, k, c, ratio], 1, 
                                                total_prob, threshold)
            pass
        total_prob/=check_num-learn_num
        print(f'avg_prob:{total_prob}')
        record_dict[T]['avg_prob'].append(total_prob)

for T in range(8):
    print(f'obs_radius:{experiments[T][0]} gamma:{experiments[T][1]} animal:{experiments[T][3]}')
    print('pred:',sum(record_dict[T]['pred'])/10)
    print('avg_prob:',sum(record_dict[T]['avg_prob'])/10)