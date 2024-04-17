import sys
import torch
from numpy import unravel_index as unravel
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import heapq
import math
import os
from collections import deque
import cv2

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

action_dict = {'a': torch.tensor([0., -1.]), 'd': torch.tensor([0., 1.]), 'w': torch.tensor([-1., 0.]), 's': torch.tensor([1., 0.])}

def do(snake: torch.Tensor, action):
    reward = 0
    positions = snake.flatten().topk(2)[1]
    [pos_cur, pos_prev] = [torch.Tensor(unravel(x, snake.shape)) for x in positions]
    #print('direction', (pos_cur - pos_prev)) # Направление движения
    pos_next = (pos_cur + action) % torch.Tensor([snake.shape]).squeeze(0)

    pos_cur = pos_cur.int()
    pos_next = pos_next.int()

    # Проверка на столкновение
    if (snake[tuple(pos_next)] > 1).any():
        reward = -10
        return reward,(snake[tuple(pos_cur)] - 2).item()  # Возвращаем счёт (длина змейки минус 2)

    # Кушаем яблоко
    if snake[tuple(pos_next)] == -1:
        pos_food = (snake == 0).flatten().to(torch.float).multinomial(1)[0] # Генерируем позицию яблока
        snake[unravel(pos_food, snake.shape)] = -1 # Добавляем яблоко в игру
        reward = 10

    else: # Двигаемся в пустую клетку
        snake[snake > 0] -= 1  # Устанавливаем все значения в теле змеи равными 1

    snake[tuple(pos_next)] = snake[tuple(pos_cur)] + 1 # перемещаем голову
    return reward, (snake[tuple(pos_cur)] - 2).item()

class Neuro_BigBoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1, 64, kernel_size=(3,3), padding = 1)
        self.drop = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(64, 128, kernel_size=(3,3), padding = 1)
        self.conv3=nn.Conv2d(128, 256, kernel_size=(3,3), padding = 1)
        self.conv4=nn.Conv2d(256, 512, kernel_size=(3,3), padding = 1)
        # self.fl = nn.Flatten()
        self.fc1=nn.Linear(128*8*8, 512)
        self.fc2=nn.Linear(512, 256)
        self.fc3=nn.Linear(256,3)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = self.pool(x)
        x= F.relu(self.conv2(x))
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = self.drop(x)
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.drop(x)
        x = self.pool(x)
        x = x.view(-1, 128*8*8)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Neuro_NotSoBigBoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(17, 256)
        self.fc2=nn.Linear(256,3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.model.load_state_dict(torch.load('model/model992.pth'))#, map_location=torch.device('cpu')))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()##.cuda()

    def train_step(self, state, action, reward, next_state, done):
        # print(state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        state = state#.cuda()
        next_state=next_state#.cuda()
        action =action#.cuda()
        reward#.cuda()
        # (n, x)
        # if len(state.shape) == 2:
        #     # (1, x)
        #     state = torch.unsqueeze(state, 0)#.cuda()
        #     next_state = torch.unsqueeze(next_state, 0)#.cuda()
        #     action = torch.unsqueeze(action, 0)#.cuda()
        #     reward = torch.unsqueeze(reward, 0)#.cuda()
        #     done = (done, )

        # 1: predicted Q values with current state
        # print(state.shape)
        pred = self.model(state.unsqueeze(0))

        target = pred.clone()
        
        Q_new = reward
        if not done:
            Q_new = reward + self.gamma * torch.max(self.model(next_state.unsqueeze(0)))
        # print(target.shape)
        target[0][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        # print(loss)

        self.optimizer.step()

class Champion():
    def __init__(self, model):
        super().__init__()
        self.n_games = 0
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = model
        self.eps = 80
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
           self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = (80-self.eps) - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state = torch.tensor(state, dtype=torch.float)
            # print(state.shape)
            with torch.no_grad():
                prediction = self.model(state.unsqueeze(0))
                # print(prediction)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


class Snake:
    def __init__(self):
        self.field = torch.zeros((32, 32), dtype=torch.float)
        self.field[0, :32] = torch.Tensor([1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
        self.field[1,:32] = torch.Tensor([64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33]) # [хвост, голова, яблоко]
        # self.field[0, :4] = torch.Tensor([1, 2, 3,4]) # [хвост, голова, яблоко]
        # self.field[1,5] =  torch.Tensor([-1])
        self.field[2,1] = torch.Tensor([65])
        self.field[2,5] =  torch.Tensor([-1])
        self.a = torch.zeros((32, 32), dtype=torch.float)
        self.dirrection = torch.tensor([0,1])
        self.head_cords = [0,1]
        self.apple_cords = [0,2]
        self.neighbours = [[31,1],[0,0],[0,2],[1,1]]
        self.collision = [False, True, False, False]
        self.availible_passes = [torch.tensor([-1,0]),torch.tensor(self.dirrection),torch.tensor([1,0])]
        self.old_distance = 0
        self.new_distance = 0
        self.state = []

    def set_dirrection(self, dir):
        dir = torch.tensor(dir)
        if not torch.allclose(dir, self.dirrection):
            self.dirrection = dir
            if torch.allclose(dir, torch.tensor([0, 1])):
                self.availible_passes = [torch.tensor([-1, 0]), torch.tensor(self.dirrection), torch.tensor([1, 0])]
            if torch.allclose(dir, torch.tensor([1, 0])):
                self.availible_passes = [torch.tensor([0, 1]), torch.tensor(self.dirrection), torch.tensor([0, -1])]
            if torch.allclose(dir, torch.tensor([0, -1])):
                self.availible_passes = [torch.tensor([1, 0]), torch.tensor(self.dirrection), torch.tensor([-1, 0])]
            if torch.allclose(dir, torch.tensor([-1, 0])):
                self.availible_passes = [torch.tensor([0, -1]), torch.tensor(self.dirrection), torch.tensor([0, 1])]

    def set_head_cords(self,head):
        self.head_cords = head

    def set_apple_cords(self,apple):
        self.apple_cords = apple

    def set_neighbours(self):
        counter = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                # Игнорируем текущий узел
                if not ((dx == -1 and dy == 0) or (dx == 0 and dy == 1) or (dx == 1 and dy == 0) or (dx == 0 and dy == -1)):
                    continue
                x = self.head_cords[0] + dx
                y = self.head_cords[1] + dy

                if x < 0 :
                    x=31
                if x>=32:
                    x=0
                if y<0:
                    y=31
                if y>=32:
                    y=0

                self.neighbours[counter] = [x,y]
                if self.field[x][y] > 0:
                    self.collision[counter] = True
                else:
                    self.collision[counter] = False
                counter+=1

    def get_state(self):
        u=self.dirrection.tolist()==[-1,0]
        l = self.dirrection.tolist()==[0,-1]
        r =self.dirrection.tolist()==[0,1]
        d =self.dirrection.tolist()==[1,0]
        return [u,l,r,d,
                # self.head_cords[0]<self.apple_cords[0],self.head_cords[0]>self.apple_cords[0],
                # self.head_cords[1]<self.apple_cords[1],self.head_cords[1]>self.apple_cords[1],
                (u==self.collision[0]==1 or l==self.collision[1]==1 or r==self.collision[2]==1 or d==self.collision[3]==1),
                (u==self.collision[1]==1 or l==self.collision[3]==1 or r==self.collision[0]==1 or d==self.collision[2]==1),
                (u==self.collision[2]==1 or l==self.collision[0]==1 or r==self.collision[3]==1 or d==self.collision[1]==1)]

    def make_step(self, steps = 0):
        if steps == 0:
            passes = [self.availible_passes[0].clone(),self.availible_passes[1].clone(),self.availible_passes[2].clone()]
            step=self.availible_passes[np.argmax(steps)]
            reward = 0
            help = self.field.clone()
            a = torch.zeros(self.field.shape)
            a[self.field>0]=1
            a[self.field==self.field.max()]=2
            a[self.field<0]=-1
            head = [self.field.topk(1)[0].argmax().numpy(),self.field.topk(1)[1][self.field.topk(1)[0].argmax()].numpy()]
            apple = [(self.field.argmin()/32).int().item(),
                (self.field.argmin()%32).item()]
            head[1]=head[1][0]
            head_cords = [head[0].item(),head[1]]
            self.set_apple_cords(apple)
            # print(self,head_cords)
            self.set_head_cords(head_cords)
            # print(self.head_cords)
            self.set_neighbours()
            self.set_dirrection([1,0])
            self.state = self.get_state()
            
            for foot in self.availible_passes:
                reward, score = do(self.field, foot)
                a = torch.zeros(self.field.shape)
                a[self.field>0]=1
                a[self.field==self.field.max()]=2
                a[self.field<0]=-1
                head = [self.field.topk(1)[0].argmax().numpy(),self.field.topk(1)[1][self.field.topk(1)[0].argmax()].numpy()]
                apple = [(self.field.argmin()/32).int().item(),
                    (self.field.argmin()%32).item()]
                head[1]=head[1][0]
                head_cords = [head[0].item(),head[1]]
                self.set_apple_cords(apple)
                # print(self,head_cords)
                self.set_head_cords(head_cords)
                # print(self.head_cords)
                self.set_neighbours()
                self.set_dirrection(foot)
                self.state.extend(self.get_state()[-3:])


                self.field = help.clone()
                step=passes[np.argmax(steps)]
                reward = 0
                a = torch.zeros(self.field.shape)
                a[self.field>0]=1
                a[self.field==self.field.max()]=2
                a[self.field<0]=-1
                head = [self.field.topk(1)[0].argmax().numpy(),self.field.topk(1)[1][self.field.topk(1)[0].argmax()].numpy()]
                apple = [(self.field.argmin()/32).int().item(),
                    (self.field.argmin()%32).item()]
                head[1]=head[1][0]
                head_cords = [head[0].item(),head[1]]
                self.set_apple_cords(apple)
                # print(self,head_cords)
                self.set_head_cords(head_cords)
                # print(self.head_cords)
                self.set_neighbours()
                self.set_dirrection([1,0])
                
        else:
            passes = [self.availible_passes[0].clone(),self.availible_passes[1].clone(),self.availible_passes[2].clone()]
            step=self.availible_passes[np.argmax(steps)]
            reward = 0
            help = self.field.clone()
            reward, score = do(self.field, step)#, steps_before_eaten_apple)
            
            a = torch.zeros(self.field.shape)
            a[self.field>0]=1
            a[self.field==self.field.max()]=2
            a[self.field<0]=-1
            head = [self.field.topk(1)[0].argmax().numpy(),self.field.topk(1)[1][self.field.topk(1)[0].argmax()].numpy()]
            apple = [(self.field.argmin()/32).int().item(),
                (self.field.argmin()%32).item()]
            head[1]=head[1][0]
            head_cords = [head[0].item(),head[1]]
            self.set_apple_cords(apple)
            # print(self,head_cords)
            self.set_head_cords(head_cords)
            # print(self.head_cords)
            self.set_neighbours()
            self.set_dirrection(step)
            self.state = self.get_state()
            
            for foot in self.availible_passes:
                reward, score = do(self.field, foot)
                a = torch.zeros(self.field.shape)
                a[self.field>0]=1
                a[self.field==self.field.max()]=2
                a[self.field<0]=-1
                head = [self.field.topk(1)[0].argmax().numpy(),self.field.topk(1)[1][self.field.topk(1)[0].argmax()].numpy()]
                apple = [(self.field.argmin()/32).int().item(),
                    (self.field.argmin()%32).item()]
                head[1]=head[1][0]
                head_cords = [head[0].item(),head[1]]
                self.set_apple_cords(apple)
                # print(self,head_cords)
                self.set_head_cords(head_cords)
                # print(self.head_cords)
                self.set_neighbours()
                self.set_dirrection(foot)
                self.state.extend(self.get_state()[-3:])


                self.field = help.clone()
                step=passes[np.argmax(steps)]
                reward = 0
                reward, score = do(self.field, step)#, steps_before_eaten_apple)
                a = torch.zeros(self.field.shape)
                a[self.field>0]=1
                a[self.field==self.field.max()]=2
                a[self.field<0]=-1
                head = [self.field.topk(1)[0].argmax().numpy(),self.field.topk(1)[1][self.field.topk(1)[0].argmax()].numpy()]
                apple = [(self.field.argmin()/32).int().item(),
                    (self.field.argmin()%32).item()]
                head[1]=head[1][0]
                head_cords = [head[0].item(),head[1]]
                self.set_apple_cords(apple)
                # print(self,head_cords)
                self.set_head_cords(head_cords)
                # print(self.head_cords)
                self.set_neighbours()
                self.set_dirrection(step)
    
        
        a_np = a.numpy()
        a[self.field<0]=10
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', a_np)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
        done = False
        if reward == -10:
            done = True
        return reward, done, score


plot_scores = []
plot_mean_scores = []
total_score = 0
record = 0
model = Neuro_NotSoBigBoss()#.cuda()
agent = Champion(model)
game = Snake()
reward_per_long_moves = 0
reward0_in_a_row=0
new_distance = 0
apple_eaten = 0
old_apple_cords = []
new_apple_cords = []
old_dirrection = torch.tensor([])
new_dirrection = torch.tensor([])
bad = 0
while True:
    # get old state
    game.make_step()
    state_old = game.state
    state_old.append(bad)
    old_apple_cords = game.apple_cords
    old_dirrection = game.dirrection
    # get move
    final_move = agent.get_action(state_old)
    # perform move and get new state
    game.old_distance = abs(game.head_cords[0]-game.apple_cords[0])+abs(game.head_cords[1]-game.apple_cords[1])
    reward, done, score = game.make_step(final_move)
    new_apple_cords = game.apple_cords
    game.new_distance = abs(game.head_cords[0]-game.apple_cords[0])+abs(game.head_cords[1]-game.apple_cords[1])
    new_dirrection = game.dirrection
    if new_dirrection.tolist()==old_dirrection.tolist() and reward == 0:
        bad+=1
        if reward == 0 and bad>32:
            reward = -bad/32
    else:
        bad = 0
    if bad>65:
        reward = -10
        bad = 0
        # print(new_dirrection.tolist(), old_dirrection.tolist())
    if not new_apple_cords == old_apple_cords:
        game.new_distance=0
    if new_distance>31:
        new_distance = 62 - new_distance
    if reward == 10:
        apple_eaten+=1
        reward=13
        bad = 0
    #     if reward0_in_a_row<70:
    #         reward+=5
    #     else:
    #         reward-=5*apple_eaten
        reward0_in_a_row=0

    if reward == -bad/32 or reward==0:
        # reward = -0.25
    #     if reward == 0:
        # reward += abs(32-game.new_distance)/8
        # if game.new_distance>game.old_distance:
        #     reward = -reward
    # if reward == -0.25:
        reward0_in_a_row+=1
        # if reward0_in_a_row > 65:
        #     reward -= reward0_in_a_row/100
        if reward0_in_a_row>1400:
            reward = -10
    # print(reward)
    if reward == -10:
        apple_eaten=0
        reward == -20
        if done == True:
            reward = -40 - score*5 
            if score > record:
                reward = -10 
        else:
            reward = -20 - score*5
        done = True
    #     reward = -5
        # if score < record:
        #     reward =-(score*record)*10
        #     if reward < -100:
        #         reward = -100
        reward0_in_a_row=0

    state_new = game.state
    state_new.append(bad>33)

    # train short memory
    # agent.train_short_memory(state_old, final_move, reward, state_new, done)

    # # remember
    # agent.remember(state_old, final_move, reward, state_new, done)

    if done:
        # print(game.collision)
        # train long memory, plot result
        game=Snake()
        agent.n_games += 1
        # agent.train_long_memory()

        if score > record:
            record = score
            # agent.model.save()


        print('Game', agent.n_games, 'Score', score, 'Record:', record)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)