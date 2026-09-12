# 🎭 PPO（2）：Actor-Critic 与 GAE

> **一句话总结**：上一节的 $G_t$ 必须跑完一整局才能算，这节改用 Bellman 方程，把优势函数写成「即时奖励 $r_t$ + 下一步价值 $\gamma V(s_{t+1})$」减当前价值——这个差值就是 TD 误差 $\delta_t$，让策略网络能单步更新，这正是 Actor-Critic 的核心。再把单步 TD 推广到 $k$ 步、并用 $\lambda$ 把所有 $k$ 步估计指数加权平均，就得到 GAE：方差比 MC 小、偏差比单步 TD 小。

## 本节地图 🗺️

1. **Bellman 方程与 TD 误差**：用 $r_t + \gamma V(s_{t+1})$ 替换 $G_t$，摆脱“必须跑完一整局”；
2. **Actor-Critic 网络**：Critic 拟合价值、Actor 拟合策略，二者配合实现单步更新；
3. **从 TD 误差到 GAE**：单步 TD → $k$ 步 TD → 指数加权平均，方差与偏差的折中。

---

## 一、Bellman 方程与 TD 误差：用单步替换整局 🔁
根据前文，可以先把奖励目标函数的梯度写成一个理论通用形式：
$$
\nabla J(\theta) = \mathbb{E}_\tau \left[
\sum_{t=0}^{T-1}(Q^\pi(s_t,a_t) - V^\pi(s_t)) \nabla \log \pi_\theta(a_t \mid s_t)
\right]
$$
其中优势函数的定义是
$$
A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^{\pi}(s_t)
$$
我们用 $V\phi(s)$ 近似 $V^\pi(s)$；用单次蒙特卡洛采样 $G_t$ 近似 $Q^\pi(s,a)$：
$$
\hat{A}_{t}^{M C}=G_{t}-V_{\phi}\left(s_{t}\right)
$$
但由于 $G_t$ 的局限性，就需要换种方式把 $G_t$ 替代掉。重新回到通用表达式，可以用 Bellman 方程把 $Q^\pi$ 递归表示出来：
$$
Q^\pi(s_t,a_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1})]
$$
此时
$$
A^\pi(s_t,a_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1})] - V^{\pi}(s_t)
$$
当然，这依然是理论上的期望，实际需要用大量采样对其进行近似。这里用单次采样去掉期望，并用 $V_{\phi}$ 替代 $V^\pi$：
$$
\hat{A}_t = [r_t + \gamma V_{\phi}(s_{t+1})] - V_{\phi}(s_t)
$$
这就是 **TD误差**，记作 $\delta_t$：
$$
\delta_t = [r_t + \gamma V_{\phi}(s_{t+1})] - V_{\phi}(s_t)
$$
在更新过程中，如果 $\delta_t > 0$，说明这一步比预期平均好，就调整策略参数。 

此时，我们可以看到
$$
\nabla J(\theta) = \mathbb{E}_{s_t,a_t}\left[
\sum_{t=0}^{T-1} [(r_t + \gamma V_{\phi}(s_{t+1})) - V_{\phi}(s_t)] \nabla \log \pi_\theta(a_t \mid s_t)
\right]
$$
注意：计算策略梯度时，$V_{\phi}$ 被视为常数，不参与对 $\theta$ 的求导。同时 $V_{\phi}$ 自身通过最小化 TD 误差的平方来更新。

这样一来，策略网络每一步都可以通过网络 $V\phi$ 提供梯度来更新，就不用等完整 episode 求出 $G_t$ 才能更新了。


***

## 二、Actor-Critic 网络：两个网络各司其职 🎭
有了前面的基础后，AC 架构就很容易理解了。本质上就是用两个网络分别拟合策略和价值函数。
Critic：拟合价值函数 $V_\phi(s_t)$。
Actor：拟合策略网络 $\pi_\theta(a_t\mid s_t)$，用 Critic 提供的 $\delta_t$ 来调整策略，让好动作的概率变大。

### Critic 的更新：最小化 TD 误差
Critic 的目标就是让网络 $V_\phi(s_t)$ 尽可能准确地预测从 $s_t$ 状态开始的期望回报。根据 Bellman 方程：
$$
V^\pi(s_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1})]
$$
我们用单次采样和当前近似网络构造目标值：
$$
y_t = r_t + \gamma V_\phi(s_{t+1})
$$
这里的 $y_t$ 就相当于“移动靶”，$V_\phi(s_{t+1})$ 就是当前网络的输出，在更新时视为常数（不参与对 $\phi$ 的求导，代码中用 detach），否则目标值会跟网络一起动，不稳定。

损失函数就是最小化平方误差：
$$
\begin{array}{l} 
  \left\{\begin{matrix} 
    L_\phi = \frac{1}{2}(y_t - V_\phi(s_t))^2 = \frac{1}{2}\delta_t^2\\
    \nabla L_\phi = -(y_t - V_\phi(s_t))\nabla V_\phi(s_t) = -\delta_t \nabla V_\phi(s_t)
\end{matrix}\right.    
\end{array} 
$$
之后神经网络梯度下降优化 $\phi$ 即可。简单来说，Critic 看了实际奖励和下一步的估计，发现自己对当前状态的估值偏了，就调整自己，争取下次估得更准。

### Actor 的更新：最大化期望回报
$$
\nabla J(\theta) = \mathbb{E}_{s_t,a_t}[A^\pi(s_t,a_t)\nabla \log \pi_\theta(a_t \mid s_t)]
$$
这里用刚才 Critic 提供的 $\delta_t$ 作为优势函数 $A^\pi(s_t,a_t)$ 的有偏估计。
$$
\nabla J(\theta)\approx \mathbb{E}_{s_t,a_t}[\delta_t\nabla \log \pi_\theta(a_t \mid s_t)]
$$
同样，这里由 Critic 提供的 $\delta_t$ 也需要视为常数，同样需要 detach。目标就是通过不断调整 $\theta$ 最大化回报。所以 Actor 的损失函数（做梯度上升，代码里取负号下降）可以写为：
$$
L_\theta = -\delta_t \log \pi_\theta(a_t \mid s_t)
$$
或者对于一批数据就是：
$$
L_\theta = -\frac{1}{N}\sum_{i}\delta_i \log \pi_\theta(a_i \mid s_i)
$$
其实刚才的 $\mathbb{E}_{s_t,a_t}[*]$ 更严谨的写法应该是 $\mathbb{E}_{s_t\sim d^\pi , a\sim \pi}[*]$，这里的 $s_t \sim d^\pi$ 就是说：状态 $s_t$ 不是均匀随机抽的，而是按它在游戏中出现的频率（折扣后）来抽的（经常出现的状态对梯度贡献大）。$d^\pi(s)$ 是折扣状态访问分布，用来描述“在策略 $\pi$ 下，状态 $s$ 出现的概率”，同时加了个折扣因子：$d^\pi(s) = \sum_{0}^{\infty}\gamma ^t P(s_t = s \mid \pi)$。
### 完整 AC 更新流程
1：在状态 $s_t$，Actor 根据 $\pi_\theta(\cdot \mid s_t)$ 采样一个动作 $a_t$ 
2：执行动作 $a_t$ 后环境返回 $r_t,s_{t+1},done$。（其中 done 是布尔变量）
3：Critic分别计算：
- $v_t = V_\phi(s_t)$
- $v_{t+1} = V_\phi(s_{t+1})$（如果 done 为 True，$v_{t+1} = 0$，同时 $v_{t+1}$ 需要 detach）
- $\delta_t = r_t +\gamma v_{t+1} - v_t$

4：Critic 更新，最小化 $L_\phi$
5：Actor 更新，得到 $\delta_t$ 后，更新，最小化 $L_\theta$（注意：这里 $\delta_t$ 需要 detach）
6：重复以上步骤，直到收敛

当然以上的形式是“单步更新”，训练中通常是批量更新，也就是跑 N 步，收集 $ (s_t,a_t,r_t,s_{t+1},done) $，最后取平均更新。

### 总结
Actor-Critic 就是用一个 Critic 网络实时估计 $V(s)$，用 TD 误差作为优势，同时更新策略和价值网络。它让策略梯度摆脱了“必须跑完一整局”的限制。但可以发现，$ V_\phi(s)$ 是 $V^\pi(s)$ 的**有偏估计**：AC 带来了降低方差以及单步更新的能力，代价是引入了偏差，而这个偏差可以通过多步 TD 或 GAE 来缓解。

***

## 三、从 TD 误差到 GAE：给“步数”加权平均 📐
### 两种极端：MC 与单步 TD
对于优势函数，现在有两种极端：
- MC 优势估计：$\hat{A}_t^{MC} = G_t - V_\phi(s_t)$，无偏，但方差大，且需跑完一整局。
- TD 误差（上节提到的 AC 结构）：$\hat{A}^{TD}_t = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$，方差低，但有偏（因为用了自举）。

所以自然会有一种想法：从单步 TD 到 k 步 TD，例如：
$$
\begin{array}{l} 
\hat{A}^{(1)}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) = \delta_t\\
\hat{A}^{(2)}_t = r_t + \gamma r_{t+1} + \gamma^2 V_\phi(s_{t+2}) - V_\phi(s_{t})\\
\cdots\\
\hat{A}^{(k)}_t = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V_\phi(s_{t+k}) - V_\phi(s_t)
\end{array}
$$
一般地，$k$ 步优势估计可以转化成一个优雅简洁的表达式：
$$
\hat{A}^{(k)}_t = \sum_{i=0}^{k-1}\gamma^i \delta_{t+i} 
$$

也就是说，k 步优势估计就是前 k 个 TD 误差的折扣和。

接下来一起简单推导一下，如何转化成这个优雅简洁的表达式。先以前两步为例：
$$
\delta_t  = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$
$$
\delta_{t+1}  = r_{t+1} + \gamma V_\phi(s_{t+2}) - V_\phi(s_{t+1})
$$
$$
\begin{aligned}
\hat{A}^{(2)}_t &= r_t + \gamma r_{t+1} + \gamma^2 V_\phi(s_{t+2}) - V_\phi(s_{t})\\
&= r_t + \gamma \delta_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \\
&= \delta_t + \gamma \delta_{t+1}
\end{aligned}
$$
类似地递推，可得到：
$$
\hat{A}^{(k)}_t = \sum_{i=0}^{k-1}\gamma^i \delta_{t+i} 
$$

在这个 k 步优势估计中，当 $k \to \infty$ 时，就变成了 MC 优势估计，无偏差，但高方差；当 $k \to 1$ 时，就变成了单步 TD 误差，方差低，但有偏差。所以这个超参数 $k$ 需要根据任务调节。既然不同的 $k$ 各有优劣，那能不能不选一个固定的 $k$，而是把所有 $k$ 的估计按权重取平均？这就是 GAE 的做法。

### GAE 优势估计
GAE 的做法是：对 $k$ 步优势估计做指数加权平均，权重由 $\lambda \in [0,1]$ 控制。
$$
\hat{A}_{t}^{G A E}=(1-\lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_{t}^{(k)}
$$

其中 $(1-\lambda)$ 是为了归一化，确保所有 $k$ 步优势估计的权重和为 $1$。简单证明一下：$(1-\lambda)(1+\lambda+\lambda^2+\cdots) = (1-\lambda)(\frac{1}{1-\lambda}) = 1 (\lambda \in [0,1]时成立)$

其实写到这里，这个 GAE 表达式也能计算，但看起来并不精简。继续化简一下，把 $\hat{A}_{t}^{(k)}$ 代入可得：
$$
\hat{A}_{t}^{G A E} = (1-\lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \sum_{i=0}^{k-1}\gamma^i \delta_{t+i} = (1-\lambda) \sum_{k=1}^{\infty} \sum_{i=0}^{k-1}\lambda^{k-1}\gamma^i \delta_{t+i}
$$
先不考虑 $1-\lambda$ ，对于 $\sum_{k=1}^{\infty} \sum_{i=0}^{k-1}\lambda^{k-1}\gamma^i \delta_{t+i}$ ：
$$
\begin{aligned}
k=1:\ & \lambda^{0} \gamma^{0} \delta_{t} \\
k=2:\ & \lambda^{1} \gamma^{0} \delta_{t}+\lambda^{1} \gamma^{1} \delta_{t+1} \\
k=3:\ & \lambda^{2} \gamma^{0} \delta_{t}+\lambda^{2} \gamma^{1} \delta_{t+1}+\lambda^{2}\gamma^{2}\delta_{t+2}
\end{aligned}
$$
这就像一张倒三角的表格。现在我们要**按列重新看**，而不是按行看。原来的顺序是“先按 $k$ 分组，再按 $i$ 分组”，现在我们要“先按 $i$ 分组，再按 $k$ 分组”。所以交换求和后可以得到：
$$
\sum_{k=1}^{\infty} \sum_{i=0}^{k-1} \lambda^{k-1} \gamma^{i} \delta_{t+i}=\sum_{i=0}^{\infty} \sum_{k=i+1}^{\infty} \lambda^{k-1} \gamma^{i} \delta_{t+i} = \sum_{i=0}^{\infty}\gamma^i \delta_{t+i}\left(  \sum_{k=i+1}^{\infty} \lambda^{k-1}\right)
$$
内层是：
$$
\begin{aligned}
\sum_{k=i+1}^{\infty} \lambda^{k-1} &= \sum_{m=i}^{\infty} \lambda^{m} \\
&= \lambda^i + \lambda^{i+1} + \cdots +\lambda^{\infty} \\
&= \frac{\lambda^i}{1-\lambda}
\end{aligned}
$$
全部代入，就得到了一个极其简洁的式子：
$$
\begin{aligned}
\hat{A}_{t}^{G A E} &= (1-\lambda)\sum_{i=0}^{\infty}\gamma^i \delta_{t+i}\left(  \frac{\lambda^i}{1-\lambda}\right)\\
&= \sum_{i=0}^{\infty}(\gamma\lambda)^i \delta_{t+i}
\end{aligned}
$$
这样可以理解为：GAE 本来是对“不同步数优势估计”的指数加权平均，展开后交换求和顺序，可以发现它等价于“把每一步的 TD 误差 $\delta_{t+i}$ 按 $(\gamma\lambda)^i$ 衰减加权求和”。这就是它和 TD 误差的直接联系。未来的 $\delta$ 权重被 $(\gamma\lambda)^i$ 快速衰减，远期的不确定性对优势估计的影响被大幅削弱，所以方差比 MC 小；同时又利用了多步信息，偏差比单步 TD 小。

### 在 AC 结构中，如何用 GAE？
有了 GAE 优势，Actor 和 Critic 的更新就变成：
**Actor 更新：**
$$
L_\theta = -\frac{1}{N}\sum_{i}\hat{A}_{t}^{G A E} \log \pi_\theta(a_i \mid s_i)
$$
注意：$\hat{A}_{t}^{G A E}$ 需要 detach，不参与对 $\theta$ 的梯度计算。

**Critic 更新：**
Critic 的目标是让 $V_\phi(s_t)$ 逼近“回报目标”。这个目标可以用 GAE 优势加上旧价值来构造：
$$
\hat{V}^{target}_t = \hat{A}^{GAE}_t + V_{\phi_{old}}(s_t)
$$
然后最小化：
$$
L_\phi = \frac{1}{2}(\hat{V}^{target}_t - V_\phi(s_t))^2
$$

***

## 延伸阅读 & 下一步

- 到这里，策略梯度从 REINFORCE 一路走到 Actor-Critic，又用 GAE 在方差与偏差之间做了折中。但 AC 仍是 **On-Policy**：策略一更新，旧数据就失效，样本利用率很低
- 下一篇 **PPO** 要解决的就是这个：用**重要性采样**让旧数据能重复利用，再用 **clip 裁剪**把策略更新幅度限制在一个小区间里，防止“一次更新过头”
- 📄 经典原文：
  - Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*（GAE）
  - Schulman et al., *Proximal Policy Optimization Algorithms*（PPO）


> 📌 本文为个人学习笔记，如有疏漏欢迎指正。
