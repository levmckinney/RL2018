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