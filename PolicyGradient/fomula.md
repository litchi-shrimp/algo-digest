# 🎯 策略梯度定理（Policy Gradient）推导 — 强化学习学习笔记

> **一句话总结**：DQN系列只能给离散动作"打分"，遇到连续动作就无解，所以需要让网络学习动作的策略分布 $\pi_\theta$ ，根据分布随机采样连续动作；但若要学习更新这个策略，则需要对策略分布求梯度；而这条梯度之所以严格成立，靠 log 技巧把算不出的环境项梯度消掉，最后只剩可计算的 $\nabla\log\pi(a\mid s)\times$ $R(\tau)$。

## 先说说：这篇推导在强化学习地图里的位置 🗺️

在策略梯度之前，最主流的是 DQN（价值网络），它走的是"打分"路线：

```
状态 s ──→ Q 网络 ──→ [ Q(s,左), Q(s,右), Q(s,跳) ] ──→ argmax 选最大
```

DQN 给**每一个动作**打一个分，再选分数最高的那个。这条路有个天生局限——**动作必须是有限个、能枚举的**。

因为要对每个动作打分，一旦动作多得枚举不完，这招就废了。举个连续动作的例子 👇

> 开车时方向盘角度是 −30° 到 +30° 之间的任意实数。你怎么给"每个角度"都打一个分？打不完。

**策略梯度走的是另一条路**：不评分，直接学"动作的分布"（概率密度函数）。

```
状态 s ──→ 策略网络 πθ ──→ P(a | s)（一个分布）──→ 从分布里采样
```

对于连续动作，直接根据πθ这个策略的分布（有均值 $\mu$ 和方差 $\sigma$），动作从分布里采样——**这就把强化学习从"离散预测"带进了"连续预测"**。

而这篇要推导的，正是这个策略网络 $\pi_\theta$ 的参数该怎么更新，也就是**策略梯度定理**——它是 Actor-Critic、TRPO、PPO 这一整条线共同的基石。

## 推导总览（先看全景，纯公式）

$$
J(\theta)=\mathbb{E}_{\tau \sim P(\tau \mid \theta)} \big[ R(\tau) \big]=\int R(\tau)\, P(\tau \mid \theta)\, d\tau \\
P(\tau \mid \theta) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t,\, a_t) \\
$$

$$
\begin{aligned}
\nabla J(\theta)
&= \int R(\tau)\, \nabla P(\tau \mid \theta)\, d\tau \\
&= \int R(\tau)\, \nabla P(\tau \mid \theta)\, \dfrac{P(\tau \mid \theta)}{P(\tau \mid \theta)}\, d\tau \quad \text{🔴}\\
&= \int P(\tau \mid \theta)\, R(\tau)\, \dfrac{\nabla P(\tau \mid \theta)}{P(\tau \mid \theta)}\, d\tau \\
&= \int P(\tau \mid \theta)\, R(\tau)\, \nabla \log P(\tau \mid \theta)\, d\tau \quad \text{🔵}\\
&= \mathbb{E}_{\tau \sim P(\tau \mid \theta)} \big[ R(\tau)\, \nabla \log P(\tau \mid \theta) \big] \\
&= \mathbb{E}_{\tau \sim P(\tau \mid \theta)} \left[ R(\tau)\, \nabla \Big( \log \mu(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t,\, a_t) \Big) \right] \\
&= \mathbb{E}_{\tau \sim P(\tau \mid \theta)} \left[ R(\tau) \cdot \sum_{t=0}^{T-1} \nabla \log \pi_\theta(a_t \mid s_t) \right] \quad \text{🟢}
\end{aligned}
$$

第一次看晕了很正常，下面一步步拆,看看具体是如何一步步消除**环境转移项** **$P(s_{t+1} \mid s_t, a_t)$**

***

## 一、核心符号

| 符号                    | 含义                                                                    |
| :-------------------- | :-------------------------------------------------------------------- |
| $\tau$                | **轨迹**：一局游戏从开始到结束的完整过程，即 $s_0, a_0, r_0, s_1, a_1, r_1, \dots$        |
| $R(\tau)$             | **总回报**：一条轨迹 $\tau$ 的总得分（累计奖励），是环境给的分数，**不含参数** **$\theta$**（求导时当常数看） |
| $P(\tau \mid \theta)$ | 在当前策略 $\pi_\theta$ 下，走出这条轨迹 $\tau$ 的**概率**                            |
| $\nabla J(\theta)$    | **梯度**：目标函数 $J(\theta)$ 的优化方向，用**梯度上升**更新参数 $\theta$                  |

各符号的具体形式如下。

轨迹 $\tau$ 的展开（$s_0$ 为初始状态）：

$$
\tau = (s_0,\, a_0,\, r_0,\, s_1,\, a_1,\, r_1,\, \dots,\, s_{T-1},\, a_{T-1},\, r_{T-1},\, s_T)
$$

总回报 $R(\tau)$（轨迹内每一步奖励之和）：

$$
R(\tau) = \sum_{t=0}^{T-1} r_t
$$

轨迹概率 $P(\tau \mid \theta)$ 的分解（$\mu(s_0)$ 为初始状态分布）：

$$
P(\tau \mid \theta) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t,\, a_t)
$$

***

## 二、目标函数与梯度

优化目标是最大化期望总回报：

$$
J(\theta) = \mathbb{E}_{\tau \sim P(\tau \mid \theta)} \big[ R(\tau) \big] = \int R(\tau)\, P(\tau \mid \theta)\, d\tau
$$

为了优化 $J(\theta)$，对其求梯度。注意 $R(\tau)$ 是环境给的回报，**不含** **$\theta$**，求导时是常数：

$$
\begin{aligned}
\nabla J(\theta)
&= \nabla \int R(\tau)\, P(\tau \mid \theta)\, d\tau \\
&= \int R(\tau)\, \nabla P(\tau \mid \theta)\, d\tau
\end{aligned}
$$

但直接计算 $\nabla P(\tau \mid \theta)$ 会卡住：$P(\tau \mid \theta)$ 里含环境转移项 $P(s_{t+1} \mid s_t, a_t)$，它是环境模型给出的概率，我们并不知晓，却会出现在梯度里，造成"死锁"。所以下一步要把它消掉。

***

## 三、🔴 为什么要构造 $\dfrac{P(\tau \mid \theta)}{P(\tau \mid \theta)}$

回顾概率论中期望的定义：若随机变量 $X$ 的概率密度为 $f(x)$，则

$$
E[X] = \int x\, f(x)\, dx
$$

更一般地，对任意函数 $g$：

$$
E\big[ g(X) \big] = \int g(x)\, f(x)\, dx
$$

期望形式的特征是：**被积函数 = 「概率密度」 × 「某个量」**。而当前的

$$
\nabla J(\theta) = \int R(\tau)\, \nabla P(\tau \mid \theta)\, d\tau
$$

并不满足这个形式（被积函数里没有 $p$ 本身）。为了凑成期望，我们乘上一个恒等于 $1$ 的因子 $\dfrac{P(\tau \mid \theta)}{P(\tau \mid \theta)}$——分子分母相同，乘上去等于没乘，却能变出我们想要的结构：

$$
\begin{aligned}
\nabla J(\theta)
&= \int R(\tau)\, \nabla P(\tau \mid \theta)\, d\tau \\
&= \int R(\tau)\, \nabla P(\tau \mid \theta)\, \dfrac{P(\tau \mid \theta)}{P(\tau \mid \theta)}\, d\tau \\
&= \int P(\tau \mid \theta)\, R(\tau)\, \dfrac{\nabla P(\tau \mid \theta)}{P(\tau \mid \theta)}\, d\tau
\end{aligned}
$$

这样被积函数就变成了「概率密度 $P(\tau \mid \theta)$」 × 「$R(\tau)\, \dfrac{\nabla p}{p}$」，恰好是期望形式。

***

## 四、🔵 对数微分技巧（log-derivative trick）

接下来的关键是恒等式：

$$
\dfrac{\nabla f(x)}{f(x)} = \nabla \log f(x)
$$

**原理**：由链式法则，

$$
\nabla \log f(x) = \dfrac{1}{f(x)}\, \nabla f(x) = \dfrac{\nabla f(x)}{f(x)}
$$

**直观理解**：

- $\nabla f(x)$ 是 $f$ 的"绝对"变化率；除以 $f(x)$ 后，$\dfrac{\nabla f(x)}{f(x)}$ 变成"相对"变化率，与 $f$ 的量纲、尺度无关，数值上更稳定。
- 取 $\log$ 相当于把"乘法结构"拉直成"加法结构"：若 $f = g \cdot h$，则 $\log f = \log g + \log h$，求导后乘积被拆成简单的求和。这正是后面能消掉环境项的关键。

代入 $f = P(\tau \mid \theta)$，得：

$$
\dfrac{\nabla P(\tau \mid \theta)}{P(\tau \mid \theta)} = \nabla \log P(\tau \mid \theta)
$$

于是：

$$
\begin{aligned}
\nabla J(\theta)
&= \int P(\tau \mid \theta)\, R(\tau)\, \nabla \log P(\tau \mid \theta)\, d\tau \\
&= \mathbb{E}_{\tau \sim P(\tau \mid \theta)} \big[\, \nabla \log P(\tau \mid \theta)\, R(\tau)\, \big]
\end{aligned}
$$

***

## 五、🟢 消去环境转移项

再来看 $\nabla \log P(\tau \mid \theta)$：

$$
\begin{aligned}
\nabla \log P(\tau \mid \theta)
&= \nabla \Big[ \log \mu(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t,\, a_t) \Big] \\
&= \nabla \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t)
\end{aligned}
$$

注意：$\log \mu(s_0)$ 和 $\log P(s_{t+1} \mid s_t, a_t)$ 都不含参数 $\theta$，对 $\theta$ 求导后为 $0$，被自然消去——环境模型项就这样被"对数微分技巧"消掉了。**这就是当初非要用 log 不可的原因：把连乘变连加，环境项才能以"加法里的一项"身份单独被扔掉。**

***

## 六、最终结果

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim P(\tau \mid \theta)} \left[ R(\tau) \cdot \sum_{t=0}^{T-1} \nabla \log \pi_\theta(a_t \mid s_t) \right]
$$

此时 $\nabla J(\theta)$ 与环境转移项 $P(s_{t+1} \mid s_t, a_t)$ 无关，只与参数 $\theta$ 有关，因此可以用**梯度上升**更新参数：

$$
\theta \leftarrow \theta + \alpha\, \nabla J(\theta)
$$

其中期望 $\mathbb{E}$ 在代码中只需通过大量采样（模拟）来近似（大数定律）。

***

## 七、代码实现

在代码中，只需得到每个 episode 中每一步的 $\log \pi_\theta(a_t \mid s_t)$（即 log\_prob）：

```python
losses_per_episode = - rewards * batch_log_probs   # 负号：用梯度下降来实现梯度上升
final_loss = losses_per_episode.mean()
final_loss.backward()
optimizer.steP()
```

***

## 延伸阅读 & 下一步

- 这里用的是整条轨迹的回报 $R(\tau)$，方差很大，实际会减去一个 baseline（如价值函数 $V$）来降方差 → **REINFORCE with baseline**
- 再用神经网络去估计这个 baseline，就是 **Actor-Critic**，一路演进到 **PPO**
- 📄 经典原文：Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*（REINFORCE）

***

> 📌 本文为个人学习笔记，如有疏漏欢迎指正。

