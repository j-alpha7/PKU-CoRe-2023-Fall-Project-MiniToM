import numpy as np
import torch
import concurrent.futures as cf
def Animal_in_view(i,j,human_pos,obs_radius): #判断i,j是否在视野范围里
    return max(human_pos[0]-obs_radius,0)<=i and i<=min(human_pos[0]+obs_radius,11-1) and max(human_pos[1]-obs_radius,0)<=j and j<=min(human_pos[1]+obs_radius,11-1)

def Animal_find(data,human_pos,obs_radius): #判断视野范围里是否有animal
    obs = data[max(human_pos[0]-obs_radius,0):min(human_pos[0]+obs_radius,11-1)+1, max(human_pos[1]-obs_radius,0):min(human_pos[1]+obs_radius,11-1)+1]  
    return np.any(obs == 1)

def Hunter_find(data): #找到Hunter的position
    return np.array(np.where(data == 1)).T[0]

def Print_format(data): #输出为保留6位小数
    return round(data,6)

def Belief_adjust(belief,human_pos,obs_radius): #根据观察调整belief
    sum = 0.000001
    new_belief=np.zeros((11,11))
    for i in range(11):
        for j in range(11):
            direct_num = 5
            if i == 0 or i+1==11 :
                direct_num -= 1 
            if j == 0 or j+1==11 :
                direct_num -= 1 
            cnt=0
            if Animal_in_view(i,j,human_pos,obs_radius) == 0:
                cnt+=1
            if i>0 and Animal_in_view(i-1,j,human_pos,obs_radius) == 0: 
                cnt+=1
            if i+1 < 11 and Animal_in_view(i+1,j,human_pos,obs_radius) == 0:
                cnt+=1
            if j > 0 and Animal_in_view(i,j-1,human_pos,obs_radius) == 0:
                cnt+=1
            if j+1 < 11 and Animal_in_view(i,j+1,human_pos,obs_radius) == 0:
                cnt+=1
            belief[i][j]/=direct_num
            sum+=belief[i][j]*cnt
            if Animal_in_view(i,j,human_pos,obs_radius) == 0:
                new_belief[i][j]+=belief[i][j]
            if i>0 and Animal_in_view(i-1,j,human_pos,obs_radius) == 0: 
                new_belief[i-1][j]+=belief[i][j]
            if i+1 < 11 and Animal_in_view(i+1,j,human_pos,obs_radius) == 0:
                new_belief[i+1][j]+=belief[i][j]
            if j > 0 and Animal_in_view(i,j-1,human_pos,obs_radius) == 0:
                new_belief[i][j-1]+=belief[i][j]
            if j+1 < 11 and Animal_in_view(i,j+1,human_pos,obs_radius) == 0:
                new_belief[i][j+1]+=belief[i][j]
            
    return new_belief/sum

def Generate(old_belief): #belief迭代，所有方向移动概率相同
    #TODO: 并行化，np.pad一圈0，再乘上(1/direct_num)矩阵，然后切片得到5个矩阵加起来得到new_belief
    new_belief = np.zeros((11,11))
    for i in range(0,11):
        for j in range(0,11):
            direct_num = 5
            if i == 0 or i+1==11 :
                direct_num -= 1 
            if j == 0 or j+1==11 :
                direct_num -= 1 
            prob = old_belief[i][j]/direct_num
            new_belief[i][j] += prob
            if i > 0:
                new_belief[i-1][j] += prob
            if i+1 < 11:
                new_belief[i+1][j] += prob
            if j > 0:
                new_belief[i][j-1] += prob
            if j+1 < 11:
                new_belief[i][j+1] += prob
    return new_belief

def Reward_calc(gamma,k,c,belief,x,y): #计算E[γ^(k*dis+c)]
    belief_tensor = torch.tensor(belief)
    i_tensor = torch.arange(0, 11).reshape(11, 1)
    j_tensor = torch.arange(0, 11).reshape(1, 11)
    abs_diff = torch.abs(x - i_tensor) + torch.abs(y - j_tensor)
    gamma_power = gamma ** (k * abs_diff + c)
    weighted_sum = torch.sum(belief_tensor * gamma_power)
    return weighted_sum

def Dist_calc(belief, x, y):
    belief_tensor = torch.tensor(belief)
    i_tensor = torch.arange(0, 11).reshape(11, 1)
    j_tensor = torch.arange(0, 11).reshape(1, 11)
    abs_diff = torch.abs(x - i_tensor) + torch.abs(y - j_tensor)
    return torch.sum(belief_tensor * abs_diff)

def Belief_pred(belief_rabbit,belief_sheep,human_pos):#给定当下的belief和hunter位置，预测接下来走各个方向的收益,gamma表示折扣因子，k*dis+c是和期望步数相关的一次函数
    belief_rabbit=Generate(belief_rabbit)
    belief_sheep=Generate(belief_sheep) 
    i=human_pos[0]
    j=human_pos[1]

    def score_func(gamma, k, c, ratio_reward):
        ans=torch.full((5,), -torch.inf, dtype=torch.float32)
        ans[0] = torch.max(Reward_calc(gamma,k,c,belief_rabbit,i,j), Reward_calc(gamma,k,c, belief_sheep,i,j)*ratio_reward)
        if i > 0:
            ans[1] = torch.max(Reward_calc(gamma,k,c,belief_rabbit,i-1,j), Reward_calc(gamma,k,c,belief_sheep,i-1,j)*ratio_reward)
        if i+1 < 11:
            ans[2] = torch.max(Reward_calc(gamma,k,c,belief_rabbit,i+1,j), Reward_calc(gamma,k,c,belief_sheep,i+1,j)*ratio_reward)
        if j > 0:
            ans[3] = torch.max(Reward_calc(gamma,k,c,belief_rabbit,i,j-1), Reward_calc(gamma,k,c,belief_sheep,i,j-1)*ratio_reward)
        if j+1 < 11:
            ans[4] = torch.max(Reward_calc(gamma,k,c,belief_rabbit,i,j+1), Reward_calc(gamma,k,c,belief_sheep,i,j+1)*ratio_reward)
        # ans[0,1,2,3,4]分别表示不动，向上，向下, 向左，向右
        return ans
    
    return score_func
    
def Belief_init(data,obs_radius):#给出指定视野范围下的belief概率图
    prob = 1.0 / 121
    belief_rabbit = np.full((11,11),prob)
    belief_sheep = np.full((11,11),prob)
    belief_data = np.zeros((data.shape[0],2,11,11))
    for index,trace in enumerate(data):
        human_pos = Hunter_find(trace[0])
        if Animal_find(trace[1],human_pos,obs_radius):
            belief_rabbit = trace[1]
        else :
            belief_rabbit = Belief_adjust(belief_rabbit,human_pos,obs_radius)
        if Animal_find(trace[2],human_pos,obs_radius):
            belief_sheep = trace[2]
        else : 
            belief_sheep = Belief_adjust(belief_sheep,human_pos,obs_radius)
        belief_data[index] = np.stack([belief_rabbit,belief_sheep])
    return belief_data

