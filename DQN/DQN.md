# 🔥 DQN（Deep Q-Network）深度强化学习 — 学习笔记

> 一句话总结：Q-learning 用表格记 Q 值，状态一多就爆炸；DQN 改用神经网络逼近 Q 值，再用「经验回放」打断样本相关、「目标网络」稳住移动靶，让深度网络能稳定地学会打游戏。

先想一个场景 👇

你教 AI 玩"打砖块"：画面（一坨像素）是状态，动作是「左移 / 右移 / 不动」。如果用一张表记下"每个画面下每个动作能得几分"——画面的组合是天文数字，表根本建不起来。

Q-learning 的表格思路很对，但撞上了**状态爆炸**。DQN 的思路是：别用表了，用神经网络去"背"这个 Q 值函数。

---

## 0. 先回顾 Q-learning 的 idea（几句话）

Q-learning 想学一个函数 $Q(s_t,a_t)$："在状态 $s_t$ 下做动作 $a_t$，未来能拿多少回报"。

更新方式就一句话（贝尔曼方程 + 时序差分）：

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \Bigl[ r_t + \gamma \max_{a_{t+1}} Q(s_{t+1},a_{t+1}) - Q(s_t,a_t) \Bigr]
$$

**人话**：用「当前奖励 $r_t$ + 下一状态的最大 Q 值」当作一个"更准的估计"，然后往它挪一小步（学习率 $\alpha$）。

表格版 Q-learning 的致命前提：$Q$ 能存成一张**有限离散**的表。一遇到图像这种高维连续状态，表直接废掉。

---

## 1. 从表格到神经网络

自然的想法：用一个神经网络 $Q(s_t,a_t;\theta)$ 去"拟合"这张表，参数是 $\theta$。输入状态 $s_t$，输出每个动作的 Q 值。

这就是 DQN 的起点。但"换表为网络"不是白换的——它会埋下两个雷（见下文创新一）。

---

## 一、核心公式（纯推导）

DQN 要做的，就是让 Q 网络去逼近**贝尔曼最优方程**。

先把目标值（标签）写出来：

$$
y_t = r_t + \gamma \max_{a_{t+1}} Q(s_{t+1}, a_{t+1};\ \theta^-)
$$

其中 $\theta^-$ 是**目标网络**的参数（先记住它和 $\theta$ 不一样，后面细讲）。

损失函数（均方误差）：

$$
L(\theta) = \mathbb{E}\Bigl[ \bigl( y_t - Q(s_t,a_t;\theta) \bigr)^2 \Bigr]
$$

展开：

$$
L(\theta) = \mathbb{E}\Bigl[ \bigl( r_t + \gamma \max_{a_{t+1}} Q(s_{t+1},a_{t+1};\theta^-) - Q(s_t,a_t;\theta) \bigr)^2 \Bigr]
$$

对 $\theta$ 求梯度。注意 $y_t$ 里的 $\theta^-$ 是固定的（不参与求导），梯度只作用在 $Q(s_t,a_t;\theta)$ 这一项上：

$$
\nabla_\theta L(\theta) = \mathbb{E}\Bigl[ 2\bigl( Q(s_t,a_t;\theta) - y_t \bigr) \cdot \nabla_\theta Q(s_t,a_t;\theta) \Bigr]
$$

这就是一个普通的回归梯度——把 $y_t$ 当"常数标签"看。

参数更新：

$$
\theta \leftarrow \theta - \eta\, \nabla_\theta L(\theta)
$$

---

## 二、DQN 的细节：三个创新

### 创新一：用 NN 逼近 Q 值 → 引出两个问题

用神经网络代替表格，直观，但埋了两个雷：

**问题 1：时序相关（样本不独立）**

游戏是一帧一帧连续的，相邻状态几乎一模一样。直接拿连续样本训练，等于用高度相关的数据喂 SGD——而 SGD 的前提是样本 **i.i.d.**（独立同分布）。相关性会让梯度有偏、训练不稳。

**问题 2：移动靶（目标非平稳，Q 发散）**

看标签 $y_t = r_t + \gamma \max_{a_{t+1}} Q(s_{t+1},a_{t+1};\theta)$。这个"标签" $y_t$ 本身是用 Q 网络算出来的，而 Q 网络**正在被更新**。于是：你每更新一步 $\theta$，标签 $y_t$ 也跟着变——像在追一个一直移动的靶子，容易震荡甚至发散。

> 这两个问题，分别由创新二、创新三来治。

### 创新二：经验回放（Experience Replay）

**做法**：把每一步的 $(s_t, a_t, r_t, s_{t+1}, \text{done})$ 存进一个回放缓冲区 $D$（比如存最近 100 万条）。训练时**随机采样**一个 mini-batch 来更新，而不是按时间顺序喂。

**为什么能治问题 1**：

- 随机采样打乱了样本顺序，切断了相邻帧之间的强相关；
- 顺带提高了样本利用率（一条经验能被采很多次）。

### 创新三：目标网络（Target Network）

**做法**：额外维护一个"目标网络" $Q(\cdot;\theta^-)$，专门用来算标签里的 $\max_{a_{t+1}} Q(s_{t+1},a_{t+1};\theta^-)$。$\theta^-$ 不随训练实时更新，而是每隔 $C$ 步把 $\theta$ 整个复制过去（$\theta^- \leftarrow \theta$）。

**为什么能治问题 2**：

- 两次复制之间，$\theta^-$ 固定不动 → 标签 $y_t$ 在一段时间内是"静止的靶"；
- 目标稳定了，Q 网络才能安稳地朝它逼近，而不是自己追自己。

---

## 三、完整流程

**① 初始化**

- Q 网络参数 $\theta$（随机初始化）
- 目标网络：$\theta^- \leftarrow \theta$（先复制一份）
- 回放缓冲区 $D$（容量 $N$，如 100 万，初始为空）
- 超参：折扣因子 $\gamma$、学习率 $\eta$、探索率 $\varepsilon$、目标同步周期 $C$

**② 每个 episode，对每个 step $t$：**

**(a) ε-greedy 选动作**

$$
a_t = \begin{cases} \text{随机动作} & \text{以概率 } \varepsilon \\ \operatorname*{argmax}_{a_t} Q(s_t, a_t; \theta) & \text{以概率 } 1-\varepsilon \end{cases}
$$

**(b) 执行动作、存储经验**

执行 $a_t$，观测奖励 $r_t$ 和下一状态 $s_{t+1}$，把一条经验存入缓冲区：

$$
D \leftarrow D \cup \bigl\{ (s_t, a_t, r_t, s_{t+1}, \text{done}) \bigr\}
$$

**(c) 随机采样、计算目标值**

从 $D$ 随机采样一个 mini-batch $\{(s_j, a_j, r_j, s_{j+1})\}$，对每个样本算目标 $y_j$：

$$
y_j = \begin{cases} r_j & \text{若 } s_{j+1} \text{ 是终止态} \\ r_j + \gamma \max_{a_{j+1}} Q(s_{j+1}, a_{j+1};\ \theta^-) & \text{否则} \end{cases}
$$

**(d) 梯度下降更新 Q 网络**

$$
L(\theta) = \frac{1}{m}\sum_{j=1}^{m} \bigl( y_j - Q(s_j, a_j;\theta) \bigr)^2
$$

$$
\theta \leftarrow \theta - \eta\, \nabla_\theta L(\theta)
$$

**(e) 定期同步目标网络**

每 $C$ 步执行一次：

$$
\theta^- \leftarrow \theta
$$

---

## 四、深入理解：DQN 的几个经典问题

### 问题 1：Q 值高估——max 的锅

目标值里有个 $\max$：

$$
y_t = r_t + \gamma \max_{a_{t+1}} Q(s_{t+1}, a_{t+1};\ \theta^-)
$$

这个 $\max$ 看着无辜，其实是"系统性高估"的元凶。

**直觉**：$Q(s_{t+1},a_{t+1})$ 是估计值，带着噪声。假设某状态下所有动作的真实 Q 值都差不多（比如都是 10），但估计值有的偏上（10.5）有的偏下（9.5）。$\max$ 操作会**专门挑那个偏上的**，于是目标 $y_t$ 就被系统性抬高了。

**公式角度（Jensen 不等式）**：

$$
\mathbb{E}\bigl[ \max_{a_{t+1}} Q(s_{t+1},a_{t+1}) \bigr] \;\ge\; \max_{a_{t+1}} \mathbb{E}\bigl[ Q(s_{t+1},a_{t+1}) \bigr] = \max_{a_{t+1}} Q^*(s_{t+1},a_{t+1})
$$

左边是"期望的 max"，右边是"max 的期望"。DQN 实际算的是左边（对带噪声的估计取 max），右边才是真实最优——所以取 $\max$ 必然高估。

**后果**：高估一层层累积，Q 值虚高，策略跟着学偏。

**解法（Double DQN）**：把"选动作"和"评估动作"拆开——当前网络 $\theta$ 负责选动作，目标网络 $\theta^-$ 负责打分：

$$
y_t = r_t + \gamma\, Q\bigl( s_{t+1},\ \operatorname*{argmax}_{a_{t+1}} Q(s_{t+1}, a_{t+1};\theta);\ \theta^- \bigr)
$$

选动作用 $\theta$（有噪声，但只负责"选哪个"），评估用 $\theta^-$（负责"打多少分"），两边噪声不完全同步，高估就被显著削弱。

---

### 问题 2：为什么要奖励裁剪（reward clipping）

DQN 原文把奖励硬裁到 $[-1, +1]$，即 $r_t \in \{-1, 0, +1\}$。为什么？

**从损失和梯度看**：

$$
L(\theta) = \bigl( y_t - Q(s_t,a_t;\theta) \bigr)^2 ,\qquad \nabla_\theta L = 2\bigl(Q(s_t,a_t;\theta) - y_t\bigr)\, \nabla_\theta Q(s_t,a_t;\theta)
$$

其中 $y_t = r_t + \gamma \max_{a_{t+1}} Q(s_{t+1},a_{t+1};\theta^-)$ 里藏着 $r_t$。如果 $r_t$ 的尺度很大（有的游戏吃个豆 +10，有的 +10000），那么：

- TD 误差 $(Q(s_t,a_t;\theta) - y_t)$ 的尺度跟着 $r_t$ 走，差别巨大；
- 梯度 $\nabla_\theta L$ 的尺度也跟着差巨大。

**问题**：要训练几十个游戏，奖励尺度天差地别，用同一个学习率 $\eta$，就会出现"这个游戏梯度爆炸、那个游戏梯度小得动不了"，根本没法统一训。

**奖励裁剪的解法**：把 $r_t$ 统一裁到 $[-1,1]$，等于抹掉"奖励大小"、只留"好/坏/中性"的符号。于是：

- 所有游戏的 TD 误差尺度被拉齐；
- 梯度幅度稳定，一套超参通吃所有游戏。

**代价**：丢了"这个动作到底多好"的量级信息，但换来稳定和通用——对 DQN 来说是笔划算的买卖。

---

### 问题 3：目标网络"滞后"的权衡

$\theta^-$ 每隔 $C$ 步才同步一次，所以它永远"落后"于 $\theta$。这是故意的（让目标静止），但滞后也有代价：$C$ 太小 → 移动靶又回来了；$C$ 太大 → 目标太旧，拖慢收敛。$C$ 是个经验性的权衡。

### 问题 4：均匀采样的浪费 → Prioritized Replay

经验回放是**均匀**采样，但不同经验的"信息量"不同——TD 误差大的样本（预测和实际差得远）更值得学。（比如训练贪吃蛇游戏，到后期缓冲区里几乎都是长度比较大的样本了）Prioritized Replay 就按 TD 误差大小加权采样，优先学"最让你吃惊"的样本。

---

## 延伸阅读

- **Double DQN**：解决 $\max$ 操作导致的 Q 值高估
- **Dueling DQN**：把 Q 拆成价值 V + 优势 A
- **Prioritized Replay**：按 TD 误差大小加权采样，不再均匀采样
- 再往前，把"打分"换成"直接学策略分布"，就是 [策略梯度](PolicyGradient/fomula.md) 那条线 → Actor-Critic → PPO
- 📄 原文：Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015

---

> 📌 本文为个人学习笔记，如有疏漏欢迎指正。
