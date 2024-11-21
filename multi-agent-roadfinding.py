import math
##########开始界面
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def confirm_input():
    global input_value
    try:
        value = int(input_entry.get())
        if 1 <= value <= 100000000:
            input_value = value
            window.destroy()
        else:
            tk.messagebox.showerror("Error", "请输入一个正整数！")
    except ValueError:
        tk.messagebox.showerror("Error", "请输入一个正整数！")

def exit_program():
    window.destroy()

window = tk.Tk()
window.title("Input Window")

# 设置窗口居中
window_width = 400
window_height = 150
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# 创建标签和输入框
label = tk.Label(window, text="请输入强化学习的次数（建议为2000-50000次）")
label.pack(pady=10)

input_entry = ttk.Entry(window)
input_entry.pack(pady=10)

# 创建确认和退出按钮
confirm_button = ttk.Button(window, text="确认", command=confirm_input)
confirm_button.pack(side=tk.LEFT, padx=10, pady=10)

exit_button = ttk.Button(window, text="退出", command=exit_program)
exit_button.pack(side=tk.RIGHT, padx=10, pady=10)

input_value = None
window.mainloop()

if input_value is not None:
    num_episodes = input_value
else:
    print("已退出")




# 定义地图大小和障碍物位置
n_rows, n_cols = 10,10

###设置地图池和初始位置
maps = [
    [(1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (2, 7), (2, 8), (3, 1), (3, 5), (3, 7), (3, 8), (4, 1),
    (4, 5), (4, 7), (4, 8),
    (5, 1), (5, 5), (5, 7), (5, 8), (6, 1), (6, 5), (6, 7), (6, 8),(7, 1), (7, 5), (8, 1), (8, 5), (8, 7), (8, 8),
    (9,1) , (9, 7), (9, 8)],
]

MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 1, 0]
]


obstacles= maps[0]



##定义初始位置和目标部分
start1 = (0,0)
goal1=(5,2)

start2=(4,0)
goal2=(9,4)

start3 = (1,5)
goal3=(9,9)

start4=(0,9)
goal4=(9,6)





# 定义动作集合和初始Q值
actions = ['up', 'down', 'left', 'right', 'stay']
Q1 = np.zeros((n_rows, n_cols, len(actions)))
Q2 = np.zeros((n_rows, n_cols, len(actions)))
Q3 = np.zeros((n_rows, n_cols, len(actions)))
Q4 = np.zeros((n_rows, n_cols, len(actions)))

act = actions[4]



# 定义奖励和惩罚
rewards1 = np.full((n_rows, n_cols), -5)
for obstacle in obstacles:
    rewards1[obstacle] = -10000
rewards1[goal1] = 500


rewards2 = np.full((n_rows, n_cols), -10)
for obstacle in obstacles:
    rewards2[obstacle] = -10000
rewards2[goal2] = 1000


rewards3 = np.full((n_rows, n_cols), -10)
for obstacle in obstacles:
    rewards3[obstacle] = -10000
    rewards3[goal3] = 1000


rewards4 = np.full((n_rows, n_cols), -10)
for obstacle in obstacles:
    rewards4[obstacle] = -10000
rewards4[goal4] = 1000

# 定义超参数
gamma = 0.9
epsilon = 0.1
alpha = 0.1

punish = -100





# 定义动作函数
def get_action(state, num):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(actions)
    else:
        if (num == 1):
            return actions[np.argmax(Q1[state])]
        if (num == 2):
            return actions[np.argmax(Q2[state])]
        if (num == 3):
            return actions[np.argmax(Q3[state])]
        if (num == 4):
            return actions[np.argmax(Q4[state])]


# 定义更新Q值函数
def update_Q(state, action, next_state, reward, num):
    if (num == 1):
        best_next_action = np.argmax(Q1[next_state])
        Q1[state][actions.index(action)] += alpha * (reward + gamma * Q1[next_state][best_next_action] - Q1[state][actions.index(action)])
    if (num == 2):
        best_next_action = np.argmax(Q2[next_state])
        Q2[state][actions.index(action)] += alpha * (reward + gamma * Q2[next_state][best_next_action] - Q2[state][actions.index(action)])
    if (num == 3):
        best_next_action = np.argmax(Q3[next_state])
        Q3[state][actions.index(action)] += alpha * (reward + gamma * Q3[next_state][best_next_action] - Q3[state][actions.index(action)])
    if (num == 4):
        best_next_action = np.argmax(Q4[next_state])
        Q4[state][actions.index(action)] += alpha * (reward + gamma * Q4[next_state][best_next_action] - Q4[state][actions.index(action)])



# 训练Q-learning
for _ in range(num_episodes):
    state1 = start1
    state2 = start2
    state3 = start3
    state4 = start4



    total_reward1 = 0
    total_reward2 = 0
    total_reward3 = 0
    total_reward4 = 0



    while state1 != goal1 or state2 != goal2 or state3 !=goal3 or state4 !=goal4:
        s1 = state1
        s2 = state2
        s3 = state3
        s4 = state4

        action1 = get_action(state1, 1)
        action2 = get_action(state2, 2)
        action3 = get_action(state3, 3)
        action4 = get_action(state4, 4)

        if state1 == goal1:
            action1 = 'stay'
            next_state1 = state1
        if action1 == 'up' and state1[0] > 0:
            next_state1 = (state1[0] - 1, state1[1])
        elif action1 == 'down' and state1[0] < n_rows - 1:
            next_state1 = (state1[0] + 1, state1[1])
        elif action1 == 'left' and state1[1] > 0:
            next_state1 = (state1[0], state1[1] - 1)
        elif action1 == 'right' and state1[1] < n_cols - 1:
            next_state1 = (state1[0], state1[1] + 1)
        else:
            next_state1 = state1


        if state2 == goal2:
            action2 = 'stay'
            next_state2 = state2
        elif action2 == 'up' and state2[0] > 0:
            next_state2 = (state2[0] - 1, state2[1])
        elif action2 == 'down' and state2[0] < n_rows - 1:
            next_state2 = (state2[0] + 1, state2[1])
        elif action2 == 'left' and state2[1] > 0:
            next_state2 = (state2[0], state2[1] - 1)
        elif action2 == 'right' and state2[1] < n_cols - 1:
            next_state2 = (state2[0], state2[1] + 1)
        else:
            next_state2 = state2

        if state3 == goal3:
            action3 = 'stay'
            next_state3 = state3
        if action3 == 'up' and state3[0] > 0:
            next_state3 = (state3[0] - 1, state3[1])
        elif action3 == 'down' and state3[0] < n_rows - 1:
            next_state3 = (state3[0] + 1, state3[1])
        elif action3 == 'left' and state3[1] > 0:
            next_state3 = (state3[0], state3[1] - 1)
        elif action3 == 'right' and state3[1] < n_cols - 1:
            next_state3 = (state3[0], state3[1] + 1)
        else:
            next_state3 = state3




        if state4 == goal4:
            action4 = 'stay'
            next_state4 = state4
        if action4 == 'up' and state4[0] > 0:
            next_state4 = (state4[0] - 1, state4[1])
        elif action4 == 'down' and state4[0] < n_rows - 1:
            next_state4 = (state4[0] + 1, state4[1])
        elif action4 == 'left' and state4[1] > 0:
            next_state4 = (state4[0], state4[1] - 1)
        elif action4 == 'right' and state4[1] < n_cols - 1:
            next_state4 = (state4[0], state4[1] + 1)
        else:
            next_state4 = state4


        ###不能碰撞：第一种情况：
        if next_state1==next_state2:

            update_Q(state1, action1, next_state1, punish, 1)
            update_Q(state2, action2, next_state2, punish, 2)



        if next_state3==next_state4:

            update_Q(state3, action3, next_state3, punish, 3)
            update_Q(state4, action4, next_state4, punish, 4)



        ###不能碰撞；第二种情况
        if s1==next_state2 and s2==next_state1:
            update_Q(state1, action1, next_state1, punish, 1)
            update_Q(state2, action2, next_state2, punish, 2)



        if s3 == next_state4 and s4 == next_state3:
            update_Q(state3, action3, next_state3, punish, 3)
            update_Q(state4, action4, next_state4, punish, 4)




####待填############这里添加收益函数部分


        l1 = math.sqrt((goal1[0] - next_state1[0])**2 + (goal1[1] - next_state1[1])**2)
        if l1>3:
            r1 = -30
        else:
            r1 = 30
        l2 = math.sqrt((next_state1[0]-next_state2[0])**2 + (next_state1[1] - next_state2[1])**2)
        if l2>2.5:
            r1=r1+30
        else:
            r1 = r1-30
        reward1 = rewards1[next_state1] + r1
        update_Q(state1, action1, next_state1, reward1, 1)
        state1 = next_state1
        total_reward1 += reward1


        l2 = math.sqrt((goal2[0] - next_state2[0])**2 + (goal2[1] - next_state2[1])**2)
        if l2>3:
            r2 = -30
        else:
            r2 = 30
        reward2 = rewards2[next_state2] + r2
        update_Q(state2, action2, next_state2, reward2, 2)
        state2 = next_state2
        total_reward2 += reward2





        l1 = math.sqrt((goal3[0] - next_state3[0])**2 + (goal3[1] - next_state3[1])**2)
        if l1>3:
            r1 = -30
        else:
            r1 = 30
        l2 = math.sqrt((next_state4[0]-next_state4[0])**2 + (next_state4[1] - next_state4[1])**2)
        if l2>2.5:
            r1=r1+30
        else:
            r1 = r1-30
        reward3 = rewards3[next_state3] + r1
        update_Q(state3, action3, next_state3, reward3, 3)
        state3 = next_state3
        total_reward3 += reward3


        l2 = math.sqrt((goal4[0] - next_state4[0])**2 + (goal4[1] - next_state4[1])**2)
        if l2>3:
            r2 = -30
        else:
            r2 = 30
        reward4 = rewards4[next_state4] + r2
        update_Q(state4, action4, next_state4, reward4, 4)
        state4 = next_state4
        total_reward4 += reward4
    print(f"Episode: {_}")



def on_confirm():
    window.destroy()

window = tk.Tk()
window.title("强化学习结束！")

# 设置窗口居中
window_width = 500
window_height = 300
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# 创建标签和按钮
label = tk.Label(window, text="强化学习完成！点击确认展示结果", font=("Arial", 16))
label.pack(pady=30)

button = tk.Button(window, text="确认", command=on_confirm)
button.pack(pady=10)

window.mainloop()




















####设置记录路径的数组：以下均为可视化代码，可视化总路径

road1=[]
road2=[]
road3=[]
road4=[]





fig, ax = plt.subplots(figsize=(10, 10))

for i in range(n_rows):
    for j in range(n_cols):
        if (i, j) in obstacles:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black', alpha=0.5)  # 创建实心方块
            ax.add_patch(rect)  # 添加到图形中

current_state = start1
plt.text(current_state[1], current_state[0], 'Start1', ha='center', va='center', fontsize=12)
num = 0
while current_state != goal1:
    if(num>50):
        print("本次强化学习1号未能找到路径，请增加学习次数！")
        exit(0)


    road1.append(current_state)
    action = actions[np.argmax(Q1[current_state])]
    if action == 'up' and current_state[0] > 0:
        next_state = (current_state[0] - 1, current_state[1])
    elif action == 'down' and current_state[0] < n_rows - 1:
        next_state = (current_state[0] + 1, current_state[1])
    elif action == 'left' and current_state[1] > 0:
        next_state = (current_state[0], current_state[1] - 1)
    elif action == 'right' and current_state[1] < n_cols - 1:
        next_state = (current_state[0], current_state[1] + 1)
    else:
        next_state = current_state

    plt.arrow(current_state[1], current_state[0], next_state[1] - current_state[1], next_state[0] - current_state[0],
              head_width=0.2, head_length=0.4, fc='blue', ec='blue')
    current_state = next_state
    num +=1
road1.append(current_state)
plt.text(current_state[1], current_state[0], 'Goal1', ha='center', va='center', fontsize=12)

########


current_state = start2
plt.text(current_state[1], current_state[0], 'Start2', ha='center', va='center', fontsize=12)
num = 0
while current_state != goal2:
    if(num>50):
        print("本次强化学习2号未能找到路径，请增加学习次数！")
        exit(0)
    road2.append(current_state)
    action = actions[np.argmax(Q2[current_state])]
    if action == 'up' and current_state[0] > 0:
        next_state = (current_state[0] - 1, current_state[1])
    elif action == 'down' and current_state[0] < n_rows - 1:
        next_state = (current_state[0] + 1, current_state[1])
    elif action == 'left' and current_state[1] > 0:
        next_state = (current_state[0], current_state[1] - 1)
    elif action == 'right' and current_state[1] < n_cols - 1:
        next_state = (current_state[0], current_state[1] + 1)
    else:
        next_state = current_state

    plt.arrow(current_state[1], current_state[0], next_state[1] - current_state[1], next_state[0] - current_state[0],
              head_width=0.2, head_length=0.2, fc='red', ec='red')
    current_state = next_state
    num +=1
road2.append(current_state)
plt.text(current_state[1], current_state[0], 'Goal2', ha='center', va='center', fontsize=12)




current_state = start3
plt.text(current_state[1], current_state[0], 'Start3', ha='center', va='center', fontsize=12)
num = 0
while current_state != goal3:
    if(num>50):
        print("本次强化学习3号未能找到路径，请增加学习次数！")
        exit(0)
    road3.append(current_state)
    action = actions[np.argmax(Q3[current_state])]
    if action == 'up' and current_state[0] > 0:
        next_state = (current_state[0] - 1, current_state[1])
    elif action == 'down' and current_state[0] < n_rows - 1:
        next_state = (current_state[0] + 1, current_state[1])
    elif action == 'left' and current_state[1] > 0:
        next_state = (current_state[0], current_state[1] - 1)
    elif action == 'right' and current_state[1] < n_cols - 1:
        next_state = (current_state[0], current_state[1] + 1)
    else:
        next_state = current_state

    plt.arrow(current_state[1], current_state[0], next_state[1] - current_state[1], next_state[0] - current_state[0],
              head_width=0.2, head_length=0.4, fc='yellow', ec='yellow')
    current_state = next_state
    num +=1
road3.append(current_state)
plt.text(current_state[1], current_state[0], 'Goal3', ha='center', va='center', fontsize=12)


current_state = start4
plt.text(current_state[1], current_state[0], 'Start4', ha='center', va='center', fontsize=12)
num = 0
while current_state != goal4:
    if(num>50):
        print("本次强化学习4号未能找到路径，请增加学习次数！")
        exit(0)
    road4.append(current_state)
    action = actions[np.argmax(Q4[current_state])]
    if action == 'up' and current_state[0] > 0:
        next_state = (current_state[0] - 1, current_state[1])
    elif action == 'down' and current_state[0] < n_rows - 1:
        next_state = (current_state[0] + 1, current_state[1])
    elif action == 'left' and current_state[1] > 0:
        next_state = (current_state[0], current_state[1] - 1)
    elif action == 'right' and current_state[1] < n_cols - 1:
        next_state = (current_state[0], current_state[1] + 1)
    else:
        next_state = current_state

    plt.arrow(current_state[1], current_state[0], next_state[1] - current_state[1], next_state[0] - current_state[0],
              head_width=0.2, head_length=0.2, fc='green', ec='green')
    current_state = next_state
    num +=1
road4.append(current_state)
plt.text(current_state[1], current_state[0], 'Goal4', ha='center', va='center', fontsize=12)


plt.xlim(-0.5, n_cols - 0.5)
plt.ylim(-0.5, n_rows - 0.5)
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()



###加入代码进行路径记录 方便输出动画  对于路径进行处理
steps = len(road1) + len(road2) + len(road3) +len(road4) -4



def crush1(s1,s2):
    minlen =min(len(s1),len(s2))
    cra = []
    i=0
    while i < minlen:
        if (s1[i] == s2[i]):
            cra.append(i)
            j = i + 1
            if(j>=len(s1) or j>= len(s2)):
                return
            while (s1[j] == s2[j]):
                if (j >= len(s1) or j >= len(s2)):
                    return
                j += 1
                if (j >= len(s1) or j >= len(s2)):
                    return
            i = j
        i = i + 1

    for i in cra:
        s1.insert(i, s1[i - 1])


crush1(road1,road2)
crush1(road1,road3)
crush1(road1,road4)
crush1(road2,road3)
crush1(road2,road4)
crush1(road3,road4)


minlenth = min(len(road1),len(road2),len(road3),len(road4))



def crush2(s1, s2):
    for i in s1:
        if(i not in s2):
            continue
        else:
            flag=0
            position = s2.index(i)
            a= s1.index(i)
            if a > position:
                a,position=position,a
                for j in range(a-1,position+1):
                    if j<len(s1) and a+position-j<len(s2):
                        if s1[j]!=s2[a+position-j]:
                            flag=1
            if(flag==0):
                return a,position-a+1
    return -1,-1

a,h = crush2(road1,road2)
if a!= -1 and h != -1:
    for i in range(h):
        road1.insert(a,road1[a-1])

a,h = crush2(road1,road3)
if a!= -1 and h != -1:
    for i in range(h):
        road1.insert(a,road1[a-1])


a,h = crush2(road1,road4)
if a!= -1 and h != -1:
    for i in range(h):
        road1.insert(a,road1[a-1])


a,h = crush2(road2,road3)
if a!= -1 and h != -1:
    for i in range(h):
        road2.insert(a,road2[a-1])

a,h = crush2(road2,road4)
if a!= -1 and h != -1:
    for i in range(h):
        road2.insert(a,road2[a-1])

a,h = crush2(road3,road4)
if a!= -1 and h != -1:
    for i in range(h):
        road3.insert(a,road3[a-1])





##############路线动画演示模块
maxlenth = max(len(road1),len(road2),len(road3),len(road4))
fig, ax = plt.subplots(figsize=(10, 10))
line1,  = ax.plot([], [], marker='o')
line2, =ax.plot([], [], marker='o')
line3, =ax.plot([], [], marker='o')
line4, =ax.plot([], [], marker='o')

for i in range(n_rows):
    for j in range(n_cols):
        if (i, j) in obstacles:
            rect = plt.Rectangle((j - 0.5, 8.5 - i ), 1, 1, color='black', alpha=0.5)  # 创建实心方块
            ax.add_patch(rect)  # 添加到图形中
def update(frame):
    x1 = [p[1] for p in road1[:frame + 1]]
    y1 = [n_rows - p[0] - 1.1 for p in road1[:frame + 1]]

    x2 = [p[1] for p in road2[:frame + 1]]
    y2 = [n_rows - p[0] - 1.05 for p in road2[:frame + 1]]

    x3 = [p[1] for p in road3[:frame + 1]]
    y3 = [n_rows - p[0] - 0.95 for p in road3[:frame + 1]]

    x4 = [p[1] for p in road4[:frame + 1]]
    y4 = [n_rows - p[0] - 1 for p in road4[:frame + 1]]


    line1.set_data(x1, y1)
    line2.set_data(x2, y2)
    line3.set_data(x3, y3)
    line4.set_data(x4, y4)

    return line1,line2,line3,line4

def init():
    ax.set_xlim(-0.5, n_rows - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xticks(range(n_rows))
    ax.set_yticks(range(n_rows))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('Robot\'s Action Path Animation')
    ax.grid(True)

    return line1,

ani = FuncAnimation(fig, update, frames=maxlenth, init_func=init, blit=True,interval=500)

plt.show()


time_steps = maxlenth
steps = str(steps)
time_steps =str(time_steps)




###写入结果部分
file_name = "roads.txt"

with open(file_name, 'a') as file:
    # 遍历二维数组列表的每一行
    for row in road1,road2,road3,road4:
        # 将每一行转换为字符串，并写入文件，使用空格分隔数字
        file.write(' '.join(map(str, row)) + '\n')



##A*算法
import heapq

# 地图大小
MAP_SIZE = 10

class Node:
    def __init__(self, position, g, h):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

# 计算曼哈顿距离作为启发式函数
def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

# A*算法
def astar(start, goal):
    open_list = []
    closed_list = set()

    heapq.heappush(open_list, Node(start, 0, heuristic(start, goal)))

    while open_list:
        curr = heapq.heappop(open_list)

        if curr.position == goal:
            path = [curr.position]
            while hasattr(curr, 'parent'):
                curr = curr.parent
                path.append(curr.position)
            return path[::-1]

        closed_list.add(curr.position)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_position = (curr.position[0] + dx, curr.position[1] + dy)
            if 0 <= new_position[0] < MAP_SIZE and 0 <= new_position[1] < MAP_SIZE and MAP[new_position[0]][new_position[1]] == 0 and new_position not in closed_list:
                new_node = Node(new_position, curr.g + 1, heuristic(new_position, goal))
                new_node.parent = curr
                heapq.heappush(open_list, new_node)

    return None

# 寻找路径
path1 = astar(start1, goal1)
path2 = astar(start2, goal2)
path3 = astar(start3, goal3)
path4 = astar(start4, goal4)

A_steps = len(path1)+ len(path2) + len(path3) + len(path4) - 4
A_timesteps = max (len(path1),len(path2),len(path3),len(path4))
A_steps = str(A_steps)
A_timesteps=str(A_timesteps)




def show_window():
    window = tk.Tk()
    window.title("Process Summary")

    # 设置窗口居中
    window_width = 500
    window_height = 300
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # 创建标签
    label_text ="所有车辆共行驶了"+steps+"步到达目标"+'\n'+"耗用时间为"+time_steps+'个时间间隔'+'\n'+'\n'+"A*算法结果为：共"+A_steps+'步和共'+A_timesteps+"时间间隔"
    label = tk.Label(window, text=label_text, font=("Arial", 18))
    label.pack(pady=20)

    window.mainloop()

# 调用函数显示窗口
show_window()