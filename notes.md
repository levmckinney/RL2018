# RL 2018

## Chapter 2

### 2.9

Associative bandits -- think of being randomly given a multi armed bandit task along with some signal that  *is associated with this bandit* then you must select the correct action. These tasks do not take into account that current actions may affect future rewards but they do take into account that rewards may change over time.

It would be helpful to do a perimeter study like the one described in exersize 2.11

## Chapter 3

## Chapter 4
#### Exercise 4.2 
#### Exercise 4.3
$$
q_\pi(s,a) = \mathbb{E}_\pi[R_t + \gamma G_{t+1} | S_t=s, A_t = a]
$$
$$
\begin{aligned}
    q_\pi(s,a) &= \sum_{s',r}p(s',r|a,s)
  \left(r + \gamma\sum_{a'\in \mathcal{A}(s')}\pi(a'|s')q_\pi(s', a')\right)\\
               &= r(s, a) + \gamma\sum_{s'}p(s'|a,s)
  \sum_{a'\in \mathcal{A}(s')}\pi(a'|s')q_\pi(s', a')
\end{aligned}
$$
$$
q_k(s,a) = r(s, a) + \gamma\sum_{s'}p(s'|a,s)
  \sum_{a'\in \mathcal{A}(s')}\pi(a'|s')q_k(s', a')
$$

#### exercise 4.4
There are a few ways to go about this. The simplest way is to keep a copy of the old value function. if there has been any improvement, if the value of all sates are the same or less stop.

However since the value functions are aproximate this may cause you to stop early. Thus I would use a exponential-recency weighted estimate of the change in value at each state between iterations. Then I would see if it dropped bellow some threshold $epsilon$. The weighting and epsilon would then be hyper perameters that would tune how sensitive the algorithm was to stalling.

#### exercise 4.5
Note that here the policy $\pi$ is a deterministic policy
1. Initilization

$Q(s,a) \in \mathbb{R}$ and $\pi(s) \in \mathcal{A}(s)$ abitrarly for all $s \in \mathcal{S}$

2. Policy Evaluation

Loop:
- $\Delta \leftarrow 0$
- Loop for each $s,a \in \mathcal{S} \times \mathcal{A(s)}$:
  - $q \leftarrow Q(s,a)$
  - $Q(s,a) \leftarrow \sum_{s',r}p(r,s'|s, a)(r + \gamma Q(s',\pi(s')))$
  - $\Delta \leftarrow \max(\Delta, |q - Q(s,a)|)$

3. Policy Improvment
- *policy-stable* $\leftarrow$ true
- For each $s \in \mathcal{S}$
  - *old-action* $\leftarrow$ $\pi(s)$
  - $\pi$(s) $\leftarrow \argmax_a Q(s, a)$
  - if *old-action* $\neq$ $\pi(s)$, then *pollicy stable*  $\leftarrow$ false
- If *policy-stable*, then stop and return $Q$ $\approx q_*$ and $\pi \approx \pi_*$ else go to 2

### 4.4
For value iteration "[f]aster convergence is often achieved by interposing
multiple policy evaluation sweeps between each policy improvement sweep."

#### Exercise 4.8

#### Exercise 4.10
$$
q_{k + 1} = \sum_{r,s'}p(r,s'|s,a)(r +\gamma \max_a q_k(s',a))
$$
Notice this is the same as the bellman optimality operator.

## 5
### 5.5 Off policy learning
We use importance sampling to predict the value of a policy $pi$ from data following a policy $b$.
We have
$$
  v_b(s) = \mathbb{E}[G_t| S_t=s]S 
$$
We need $v_pi$.
So we replace out the probability of future sequences under $b$, with the probability of future sequences under
$pi$ by multiplying by the ratio. This ratio is 
$$
  \rho_{t:T-1} = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$
The state transition probabilities cancel out. Thus we get that.
$$
  v_\pi(s) =\mathbb{E}[\rho_{t:T-1}G_t|S_t=s]
$$
Very nice!

Considering first visit methods,

*Ordinary-importance-sampling*

Unbiased (i.e. has the correct expected value)  however the variance is unbounded and it is generally not preferred

*Weighted-importance-sampling*

biased, however the bias coverages to zero as we apply more iterations. In general this method is preferred.

Both are biased for every-visit methods.

#### Exercise 5.5
TODO

### 5.6
#### Exercise 5.6
TODO

#### Exercise 5.7
I think this has to do with the fact that the
#### Exercise 5.8

### 5.7
#### Exercise 5.11
It is correct since just above this we check to see if the action $A_t$ matches the greedy action. If it does not then since the policy is deterministic $pi(A_t|S_t) = 0$ and the remaining loops would not do anything. 
#### Exercise 5.12: Racetrack (programming)
It would be good to do this 

## Chapter 6
### 6.1
In temporal difference learning rather than waiting for the full return reward (Monte carlo) we use an approximation of the return i.e.
$$
 \gamma V(s') + R_t
$$
for $TD(0)$.
### 6.2
#### Exercise 6.3
Since we reduced the value of state A it seems like the first random walk ended with by going off the left side. Only A changing is an artifact of the boot straping that $TD(0)$ is doing. Since this is an undiscounded task and we initalize all none-terminal values to 0.5 the $V(s')$ and $V(s)$ terms in the td error cancel each other so long as $s'$ is not a terminal state. However since we transition from A to a terminal state A's estimated value does change. $V_0(A) = 0.5$ and $V_1(A)= V_0(A) - \alpha(R_1 + 0 - V_0(A)) =0.5 - 0.5 * 0.1 = 0.45$ so the value of a changed by 0.05.

#### Exercise 6.4
Im not really sure. For TD it apears that $\alpha=.05$ and $.15$ bracket the reasonable range of values since since TD(0) begins to converge more slowly than MC bellow $\alpha = 0.05$ and the TD error seems to grow over worse over time for large values of $\alpha$. Considering the monte carlo simulations it seems like values above $\alpha = 0.2$ converge quite slowly any larger start fail to converge at all. Assuming these trends the behaviors caused by alpha hold I would not expect either algorithm to preform much better than shown for other values of $\alpha$.

#### Exercise 6.6
These values could have been comuted using value iteration or by solving the bellman equations for this system. I suspect the writers solved directly since in this case they are fairly simple and by symmetry we can fix V(C) strait away as 0.5.

### 6.3
"Batch Monte Carlo methods always find the
estimates that minimize mean-squared error on the training set, whereas batch TD(0)
always finds the estimates that would be exactly correct for the maximum-likelihood
model of the Markov process."

## 6.4
#### Exercise 6.8
Trying to prove something analogous to equation (6.6) for action value functions. The TD error for action value functions is as follows:
$$
  \delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
$$
*Proof:*

Assuming action value is stable durning the episode.
Basically identical to the proof for 6.6.
$$
\begin{aligned}
  G_t - Q(S_t) &= R_{t + 1} + \gamma G_{t+1} - Q(S_t, A_t) + \gamma Q(S_{t+1}, A_{t+1}) - \gamma Q(S_{t+1}, A_{t+1})\\
               &= \delta_t + \gamma(G_{t+1}-Q(S_{t+1}, G_{t + 1}))\\
               &= \delta_t + \gamma \delta_t + \gamma^2(G_{t+1}-Q(S_{t+2}, G_{t + 2}))\\
               \text{Since $G_T = 0$}&\text{ and $Q(S_T, A_T) = 0$}\\
               &= \delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2} + ... +\gamma^{T-t}(0 - 0)\\
               &=  \sum_{k=t}^{T-1} \gamma^{k-t}\delta_k
\end{aligned}
$$
As needed.
## 6.5
#### Exercise 6.11
Q-learning is considered off policy because it does not learn the action value function for the policy learned actually interacting with the environment it instead learns the action value function for the optimal policy i.e. $q_*$.
#### Exercise 6.12
If action selection is greedy then SARSA and Q-learning are the same algorithm since the only real difference between the weight updates is 
$\max_a Q(S', a)$ v.s. $Q(S', A')$ which are equivalent if $A'$ is the greedy action.

### 6.6 Expected sarsa
Through it requires more computation per time step expected sarsa is performs better than sarsa in both asymptotically and int he near term. large values of $alpha$ seem to work best. Expected sarsa can be used as an off policy method simply by using $\pi$ as a target policy and some other policy $b$ to generate the behavior. Expected sarsa is also a generalization of Q-learning since for a greedy target policy it is equivalent to Q-learning.

### 6.7
Maximization bias example:
- Imagine if you where at a casino and you had the choice of going to play the slot machine or leave. Though most of the time you lose, money you remember that one time when you pulled the lever just so you won so you say to your self I will pull the lever just so again and then I will probably win. 

This is an example of maximization bias since you are approximating the true maximum over the action values of ways of playing with the maximum of your experiences with the machine. 

This is kind of bias badly effects q-learning and sarsa methods. It can be fixed by learning two separate action value functions using one to select an action and one to evaluate said action. This removes the bias.

We might will alternate which function we use for each task or select randomly.

In the above example this would be equivalent to asking a friend what happens when they pull the slot machine lever just so.

Update rule for double Q-learning,
$$
Q_1(S_t, A_t) = Q_1(S_t, A_t) + \alpha\left[R_{t+1} + \gamma Q_2(S_{t+1}, \argmax_a Q_1(S_{t+1}, a)) - Q_1(S_t, A_t)\right]
$$
### 6.8 After states
#### Exercise 6.14
To reformulated example 4.2 in terms of after states we could simply chose as the state the number of cars at each lot after moving cars between lots at the end of the day. Since the affect of moving the cars is totally deterministic. This would speed convergence since ...

## Chapter 7
### 7.1
#### Exercise 7.1
WTS: $G_{t:t+n} - V(S_t) = \sum_{}^{} \gamma^{}\delta_{}$
Remember that:
$$
\delta_t = R_{t+1} + \gamma V(S_{t+t}) - V(S_t)
$$
and that 
$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} - V_{t+n - 1}(S_{t+n})
$$
However since we are assuming an value function unchanging during each episode.
$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^n-1 R_{t+n} - V_(S_{t+n})
$$
Using this we can show that,
$$
\begin{aligned}
  G_{t:t+n} - V(S_t) =& R_{t+1} + \gamma G_{t+1:t+n} - V(S_t)\\
  =& R_{t+1} + \gamma G_{t+1:t+n} - V(S_t) + \gamma V(S_{t+1}) - \gamma V(S_{t+1})\\
  =& R_{t+1} + \gamma V(S_{t+1}) - V(S_t) + \gamma \left(G_{t+1:t+n} - V(S_{t+1})\right) \\
  =& \delta_t + \gamma \left(G_{t+1:t+n} - V(S_{t+1})\right) \\
  =& \delta_t + \gamma \left(\delta_{t+1} + \gamma \left(G_{t+2:t+n} - V(S_{t+2})\right)\right) \\
  =& \delta_t + \gamma \delta_{t+1} + \gamma \left(G_{t+2:t+n} - V(S_{t+2})\right)\\
  =& \delta_t + \gamma \delta_{t+1} + ... + \gamma^{t+n-1}\delta_{t+n-1} + \gamma^{t+n}(G_{t+n:t+n} - V(S_{t+n}))\\
  =& \delta_t + \gamma \delta_{t+1} + ... + \gamma^{t+n-1}\delta_{t+n-1} + \gamma^{t+n}(V(S_{t+n}) - V(S_{t+n}))\\
  &\text{if $t+n > T$ then since $G_T = 0$ and $V(terminal) = 0$ the following holds}\\
  =& \sum_{k = t}^{\min(t+n-1, T - 1)} \gamma^{k - t}\delta_k
\end{aligned}
$$
A very nice generalization of (6.6)!
#### Exercise 7.3
You need enough states to ensure that the random walks last long enough. Otherwise you would not be able to tell the difference between many of the n-step methods and monte carlo. Reducing the number of steps would likely cause the smaller values of n to be preform better while allowing. Im not as certain about change the right most states reward to -1.

#### Exercise 7.4
TODO

#### Exercise 7.5
TODO

#### Exercise 7.7
TODO

#### Exercise 7.8
TODO

#### Exercise 7.10 (program)
This one is going to take some time to figure out need to come up with a good experiment maybe somthing really. 
Use the random walk value aproximation task. Should be safe to use python for this one.

#### Exercise 7.11
TODO worth completing.

#### Exercise 8.1
Its possible that the tree backup algorithm be more efficient on this task then the one step algorithm used. 
... I need to think more on this one. I think that the tree back up method might do something similar to planing.

#### Exercise 8.2
The dyna-Q+ algorithm preformed better because the states on the other side of the wall where over time given an exploration bonus 
so the againt planed to explore them and discover the goal state, On the other hand the dyna-Q using an epsilon greedy policy simply took random actions which where less likely to lead to the small opening on the left and eventually the goal state.

#### Exercise 8.3
Taking a single random action is less likely to hurt the return then going on a wild goose chase to check a state that has not been seen in a while. Thus while no changes where happening to the environment the dyna-Q preforms a bit better then dyna-Q+.

#### Exercise 8.5
TODO