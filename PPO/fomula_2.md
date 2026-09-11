## Bellman方程和TD误差
根据前文，可以先把奖励目标函数的梯度写成一个理论通用形式。
$$
\nabla J(\theta) = \mathbb{E}_\tau \left[
\sum_{t=0}^{T-1}(Q^\pi(s_t,a_t) - V^\pi(s_t)) \nabla \log \pi_\theta(a_t \mid s_t)
\right]
$$
其中优势函数的定义是
$$
A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^{\pi}(s_t)
$$
我们用 $V\phi(s)$ 近似 $V^\pi(s)$ ；用单次蒙特卡洛采样 $G_t$ 近似 $Q^\pi(s,a)$ ：
$$
\hat{A}_{t}^{M C}=G_{t}-V_{\phi}\left(s_{t}\right)
$$
但由于 $G_t$ 的局限性，就需要换种方式把 $G_t$ 替代掉。重新回到通用表达式，可以用Bellman方程把 $Q^\pi$ 递归表示出来：
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
在更新过程中，如果 $\delta_t > 0$ ，说明这一步比预期平均好，调整策略参数。 

此时我们可以看到
$$
\nabla J(\theta) = \mathbb{E}_{s_t,a_t}\left[
\sum_{t=0}^{T-1} [(r_t + \gamma V_{\phi}(s_{t+1})) - V_{\phi}(s_t)] \nabla \log \pi_\theta(a_t \mid s_t)
\right]
$$
注意：计算策略梯度时，$V_{\phi}$ 被视为常数，不参与对 $\theta$ 的求导。同时 $V_{\phi}$ 自身通过最小化TD误差的平方来更新。

这样一来，策略网络每一步都可以通过网络 $V\phi$ 提供梯度来更新，就不用等完整episode求出 $G_t$ 才能更新了。


## Actor-Critic 网络

## 从TD误差到GAE
