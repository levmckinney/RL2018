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

However since the value functions are aproximate this may cause you to stop early. Thus I would use a exponential-recency weigted estimate of the change in value at each state between iterations. Then I would see if it dropped bellow some threshold $epsilon$. The weighting and epsilon would then be hyper perameters that would tune how sensitive the algorithm was to stalling.

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