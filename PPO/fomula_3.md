# 🚧 PPO（完结）：重要性采样与Clip

> **一句话总结**：AC + GAE 的问题在于 On-Policy——策略一更新，旧数据就得扔掉，样本利用率极低。PPO 用**重要性采样**让旧数据能继续用，再用 **Clip 裁剪**限制新旧策略的差距、防止重要性权重爆炸。它绕开计算昂贵的 KL 散度约束，用一行 clip 就完成了“更新别太猛”这件事。

## 本节地图 🗺️

1. **重要性采样**：乘一个修正权重，用旧策略的数据更新新策略；
2. **引入 KL 散度**：先看看“限制更新幅度”的理论做法，以及它的代价；
3. **Clip 梯度裁剪**：用 clip 替代 KL，分正负优势两种情况拆解；
4. **PPO 完整流程**：从采样到多轮更新，附超参数与常见坑。

---

目前已经大致理解了：策略梯度 → 优势函数 → TD 误差 → Actor-Critic → GAE。PPO（Proximal Policy Optimization，近端策略优化）就是在 AC 结构 + GAE 的基础上，进一步优化，核心主要体现在两点：

（1）**重要性采样**：之前提到过，基于策略的梯度在同一局训练中是依赖当前策略的——玩一局得到数据、更新策略，之后便扔掉数据再玩。PPO 想让我们能利用旧策略的数据更新新策略，提高数据利用率。

（2）**梯度裁剪**：既然想用新策略跑旧数据，那最好让新旧两个策略差距不大。如果更新后策略变化太大，旧数据就不再能代表新策略的行为，继续用旧数据更新，梯度方向就错了，策略会越走越偏、甚至崩溃。因此 PPO 引入梯度裁剪，用最简单的 clip 函数代替复杂的 KL 散度约束，防止策略更新步子迈得太大。

上一节我们提到 Actor 的目标函数（损失函数取负）如下：
$$
L_\theta = \frac{1}{N}\sum_{i}\hat{A}_{t}^{G A E} \log \pi_\theta(a_i \mid s_i)
$$
但这个式子的更严谨的表达方式应该是这样：
$$
L_\theta = \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\hat{A}_{\theta}^{G A E}(s^t_n,a^t_n) \log \pi_\theta(a^t_n \mid s^t_n)
$$
梯度更新表达式为：
$$
\nabla L_\theta = \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\hat{A}_{\theta}^{G A E}(s^t_n,a^t_n) \nabla \log \pi_\theta(a^t_n \mid s^t_n)
$$
这样写也能看出在实际代码中是如何按 Step 和 episode 计算的。本节讲述的 PPO 就是在这个式子基础上做一些优化。
***

## 一、重要性采样：用旧数据更新新策略 🎲
试想这样一个问题：我们想求某个函数 $f(x)$ 在分布 $P(X)$ 下的期望：
$$
\mathbb{E}_{x\sim P}[f(x)] = \sum P(x) f(x)
$$
但是我们只有从另一个分布 $Q(x)$ 中采样的样本。我们可以在公式里强行乘一个 $Q(x)$ 再除一个 $Q(x)$：
$$
\sum P(x) f(x) = \sum Q(x) \frac{P(x)}{Q(x)} f(x) = \mathbb{E}_{x\sim Q}\left[\frac{P(x)}{Q(x)} f(x)\right]
$$
也就是说，可以用 $Q$ 的样本来算 $P$ 的期望，只要在每一项前面乘上一个权重 $\frac{P(x)}{Q(x)}$ 即可。类似地，在 PPO 中，$Q(x)$ 就相当于旧策略，$P(x)$ 就相当于新策略。
$$
\begin{aligned}
\nabla L_\theta &= \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\hat{A}_{\theta}^{G A E}(s^t_n,a^t_n) \nabla \log \pi_\theta(a^t_n \mid s^t_n) \\
&= \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\hat{A}_{\theta_{old}}^{G A E}(s^t_n,a^t_n) \frac{\pi_\theta(a^t_n \mid s^t_n)}{\pi_{\theta_{old}}(a^t_n \mid s^t_n)} \nabla \log \pi_\theta(a^t_n \mid s^t_n) \\
&= \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\hat{A}_{\theta_{old}}^{G A E}(s^t_n,a^t_n) \frac{\pi_\theta(a^t_n \mid s^t_n)}{\pi_{\theta_{old}}(a^t_n \mid s^t_n)} \frac{\nabla \pi_\theta(a^t_n \mid s^t_n)}{\pi_\theta(a^t_n \mid s^t_n)} \\
&= \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\hat{A}_{\theta_{old}}^{G A E}(s^t_n,a^t_n) \frac{\nabla \pi_\theta(a^t_n \mid s^t_n)}{\pi_{\theta_{old}}(a^t_n \mid s^t_n)}
\end{aligned}
$$
所以损失函数可以写成：
$$
Loss = -\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\hat{A}_{\theta_{old}}^{G A E}(s^t_n,a^t_n) \frac{\pi_\theta(a^t_n \mid s^t_n)}{\pi_{\theta_{old}}(a^t_n \mid s^t_n)}
$$
记
$$ r^t_n(\theta) = \frac{\pi_\theta(a^t_n \mid s^t_n)}{\pi_{\theta_{old}}(a^t_n \mid s^t_n)} 
$$
若 $r^t_n(\theta) >1 $，新策略更倾向于选这个动作；反之则更不倾向于选这个动作。也就是说，用旧策略采样，然后给每个样本乘一个“修正权重” $r^t_n(\theta)$，就能得到新策略下的无偏估计。但有一个问题：如果 $\pi_\theta$ 和 $\pi_{\theta_{old}}$ 差太远，$r^t_n(\theta)$ 会极大或极小，重要性采样的方差会爆炸。因此就需要有一种方式**限制新旧策略更新的幅度**，把 $r^t_n(\theta)$ 控制在小范围内。

***

## 二、引入 KL 散度 📏
为了限制新旧策略更新的幅度，也就是让 $\pi_\theta(a^t_n \mid s^t_n)$ 和 $\pi_{\theta_{old}}(a^t_n \mid s^t_n)$ 的分布接近，我们可以在损失函数中引入 KL 散度约束：
$$
Loss = -\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\hat{A}_{\theta_{old}}^{G A E}(s^t_n,a^t_n) \frac{\pi_\theta(a^t_n \mid s^t_n)}{\pi_{\theta_{old}}(a^t_n \mid s^t_n)} + \beta KL(\pi_\theta, \pi_{\theta_{old}})
$$
其中 
$$
KL(\pi_\theta, \pi_{\theta_{old}}) \approx \mathbb{E}_t\left[ 
  \log \frac{\pi_\theta(a^t_n \mid s^t_n)}{\pi_{\theta_{old}}(a^t_n \mid s^t_n)}
\right]
$$
或者用更精确的K3估计（数值更稳定）
$$
KL(\pi_\theta, \pi_{\theta_{old}}) \approx \mathbb{E}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)}-1-\log \frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)}\right]
$$

加 KL 散度这个思想在 **TRPO** 算法中就有体现，但 TRPO 是直接约束优化——**KL 硬约束** $\mathbb{E}[KL]<=\delta$；而 PPO-penalty 是一个**KL 软惩罚**，在损失里加 $\beta KL(\pi_\theta, \pi_{\theta_{old}})$。不过 $\beta$ 是一个超参数，需要通过实验来确定，后期引入自适应动态调整的 $\beta$，让 $KL$ 维持在目标值。

但是 KL 散度的计算复杂度很高，因此 PPO 采用了一种更简单实用的方法，就是 **Clip 梯度裁剪**。

***

## 三、Clip 梯度裁剪 ✂️
先列一个总览公式，带梯度裁剪的 PPO 损失函数如下（目标函数取负）：
$$
Loss = -\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}
\min \left(
  \hat{A}_{\theta_{old}}^{G A E}(s^t_n,a^t_n) 
  \frac{\pi_\theta(a^t_n \mid s^t_n)}{\pi_{\theta_{old}}(a^t_n \mid s^t_n)},
  clip (
    \frac{\pi_\theta(a^t_n \mid s^t_n)}{\pi_{\theta_{old}}(a^t_n \mid s^t_n)},
    1-\epsilon,
    1+\epsilon
  )
  \hat{A}_{\theta_{old}}^{G A E}(s^t_n,a^t_n)
\right)
$$

简化一下就是：
$$
L^{C L I P}(\theta)=\mathbb{E}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]
$$
Clip 的作用就是：重要性权重 $ r_{t}(\theta) $ 被限制在 $[1-\epsilon, 1+\epsilon]$ 范围内，可以偏离 1，但不能偏离太远，避免重要性采样的方差爆炸。下面分两种情况讨论具体裁剪：

**情况一：$\hat{A}_t>0$，这个动作比平均好**
如果没有clip：
$$
L = r_t(\theta)\hat{A}_{t}
$$
$r_t$ 越大，$L$ 也就越大，梯度也很大。但加了 clip 后：
$$
L=\min \left(r_{t} \hat{A}_{t}, \operatorname{clip}\left(r_{t}, 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)
$$
因为 $\hat{A}_t > 0$，所以 $\operatorname{clip}\left(r_{t}, 1-\epsilon, 1+\epsilon\right)\hat{A}_{t}$ 的最大值就是 $(1+\epsilon)\hat{A}_{t}$。如果 $r_t <= 1+\epsilon$，clip 没起作用，$\min$ 取第一项，梯度正常更新；如果 $r_t > 1+\epsilon $，clip 就把 $r_t$ 截断成 $1+\epsilon$，整个梯度就是 0。

也就是说，好动作可以鼓励，但概率最多只能提高到旧策略的 $1+\epsilon$ 倍。

**情况二：$\hat{A}_t<0$，这个动作比平均差**
如果没有clip：
$$
L = r_t(\theta)\hat{A}_{t}
$$
$r_t$ 越小，$L$ 也就越小，梯度也越小。但加了 clip 后：
$$
L=\min \left(r_{t} \hat{A}_{t}, \operatorname{clip}\left(r_{t}, 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)
$$
因为 $\hat{A}_t < 0$，所以 $\operatorname{clip}\left(r_{t}, 1-\epsilon, 1+\epsilon\right)\hat{A}_{t}$ 的最小值就是 $(1-\epsilon)\hat{A}_{t}$。如果 $r_t >= 1-\epsilon$，clip 没起作用，$\min$ 取第一项，梯度正常更新；如果 $r_t < 1-\epsilon $，clip 就把 $r_t$ 截断成 $1-\epsilon$，整个梯度就是 0。

也就是说，坏动作可以抑制，但概率最多只能降低到旧策略的 $1-\epsilon$ 倍。

***

## 四、PPO 完整流程：从采样到更新 🔧
到这里，PPO 的理论基本介绍完毕，接下来简单看一下代码中的实现步骤（和 AC 基本一致）：

**Step 1：收集数据**

用当前策略 $\pi_{\theta_{old}}$ 跑 N 个环境，采样一批轨迹：
$$
(s_t^n,\ a_t^n,\ r_t^n,\ s_{t+1}^n,\ done_t^n)
$$

同时记录：旧策略的 log 概率：$\log \pi_{\theta_{old}}(a_t^n \mid s_t^n)$ 和旧价值网络的输出：$V_{\phi_{old}}(s_t^n)$。注意：$\theta$ 是策略网络参数，$\phi$ 是价值网络参数，两者独立。

**Step 2：计算 GAE 优势和 Critic 目标**

用旧价值 $V_{\phi_{old}}$ 计算 TD 误差：
$$
\delta_t^n = r_t^n + \gamma \cdot V_{\phi_{old}}(s_{t+1}^n) \cdot (1 - done_t^n) - V_{\phi_{old}}(s_t^n)
$$

然后用 GAE 公式（从后往前递推）：
$$
\hat{A}^{GAE}(t,n) = \delta_t^n + \gamma \lambda \cdot (1 - done_t^n) \cdot \hat{A}^{GAE}(t+1,n)
$$

等价于：
$$
\hat{A}^{GAE}(t,n) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^n
$$

Critic 的回归目标：
$$
V_{target}(t,n) = \hat{A}^{GAE}(t,n) + V_{\phi_{old}}(s_t^n)
$$
所有优势与目标值均需 detach，视为常数。

**Step 3：多轮更新 Actor 和 Critic**
对同一批数据，重复 K 轮：
用当前策略 $\pi_{\theta}$ 重新计算 $\log \pi_{\theta}(a_t^n \mid s_t^n)$。算概率比值：
$$
r_t(\theta) = \frac{\pi_{\theta}(a_t^n \mid s_t^n)}{\pi_{\theta_{old}}(a_t^n \mid s_t^n)}
$$
算 PPO-Clip 损失：
$$
L^{CLIP} = -\frac{1}{N}\sum_{n,t} \min\left(r_t \hat{A}^{GAE}(t,n),\ \operatorname{clip}\left(r_t,\ 1-\epsilon,\ 1+\epsilon\right) \hat{A}^{GAE}(t,n)\right)
$$
算 Critic 损失：
$$
L^{VF} = \frac{1}{N}\sum_{n,t}\left(V_{target}(t,n) - V_{\phi}(s_t^n)\right)^2
$$
算熵奖励：
$$
L^{ENT} = -\frac{1}{N}\sum_{n,t} H\left(\pi_{\theta}(\cdot \mid s_t^n)\right)
$$
这一步之前没有细讲，熵奖励的作用是：防止策略过拟合，保持策略分布的多样性。我们知道熵用来衡量一个概率分布的不确定性或随机程度。熵如果很小，表示“很确定”，策略分布就比较单一。如果熵很大，表示“很不确定”，策略分布就比较分散。

PPO 就是想奖励“高熵”的策略。策略网络输出每个动作的 logits，经过 softmax 得到概率 $\pi_{\theta}(a \mid s)$：
$H\left(\pi_{\theta}(\cdot \mid s)\right) =  - \sum_{a} \pi_{\theta}(a \mid s) \log \pi_{\theta}(a \mid s)$

总损失：
$$
L = L^{CLIP} + c_1 L^{VF} + c_2 L^{ENT}
$$

反向传播，更新 $\theta$ 和 $\phi$。注意：log_probs_old、advantages、targets 均不参与梯度计算。这里可能有一点疑问：之前 AC 是两个网络各自有损失函数、各自更新，这里为什么写成统一的损失函数一起更新？

Actor 网络参数是 $\theta$，Critic 网络的参数是 $\phi$。如果两个网络完全独立，分开更新和一起更新在数学上完全等价。统一反向传播时，梯度会自动分别流向各自的网络，和分开更新完全等价。

**Step 4：丢弃旧数据，回到 Step 1**
因为策略已经更新很多，旧数据不再符合当前策略分布，必须重新采样。

### 超参数参考

- $\gamma = 0.99$
- $\lambda = 0.95$
- $\epsilon = 0.2$
- K_epochs = 4 到 10
- rollout_steps = 2048
- lr_actor = 3e-4
- lr_critic = 1e-3
- $c_1 = 0.5$
- $c_2 = 0.01$
- max_grad_norm = 0.5

### 注意事项

- log_probs_old 忘了 detach：会导致计算图保留旧数据，内存爆炸。
- advantages 忘了 detach：Actor 更新会影响 Critic 的优势，训练不稳定。
- targets 忘了 detach：Critic 目标跟着网络跑，训练崩溃。
- 优势没有归一化：训练极其不稳定，尤其在奖励尺度大的环境。
- 梯度没有裁剪：偶尔一个大梯度直接毁掉网络。
- K_epochs 设太大：策略偏离旧策略太远，Clip 失效，训练崩溃。

***

## 附：简单代码框架

```python
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Actor 输出动作 logits，Critic 输出状态价值 V(s)。"""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)   # logits
        self.critic = nn.Linear(hidden, 1)        # V(s)

    def forward(self, s):
        h = self.shared(s)
        return self.actor(h), self.critic(h).squeeze(-1)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """从后往前递推 GAE；values 需多存一个末端 V(s_T)=0。"""
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values[:-1]


def ppo_update(model, opt, states, actions, old_log_probs,
               advantages, returns, eps=0.2, c1=0.5, c2=0.01, K=4):
    """同一批数据重复 K 轮，同时更新 Actor 与 Critic。"""
    for _ in range(K):
        logits, values = model(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)             # r_t(θ)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
        loss_clip = -torch.min(surr1, surr2).mean()                  # PPO-Clip

        loss_vf = ((values - returns) ** 2).mean()                   # Critic 回归
        loss_ent = -dist.entropy().mean()                            # 熵奖励

        loss = loss_clip + c1 * loss_vf + c2 * loss_ent              # 总损失
        opt.zero_grad()
        loss.backward()
        opt.step()


# 主循环：采样 → 算 GAE → 多轮更新 → 丢弃旧数据重新采样
for update in range(num_updates):
    states, actions, rewards, dones, old_log_probs, values = collect(...)
    advantages, returns = compute_gae(rewards, values, dones)
    ppo_update(model, opt, states, actions, old_log_probs, advantages, returns)
```

至此，PPO 的基本理论就讲述完毕了。

***

> 📌 本文为个人学习笔记，如有疏漏欢迎指正。