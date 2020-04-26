# RL 2018

## Chapter 2

### 2.9

Associative bandits -- think of being randomly given a multi armed bandit task along with some signal that  *is associated with this bandit* then you must select the correct action. These tasks do not take into account that current actions may affect future rewards but they do take into account that rewards may change over time.

We can 

It would be helpful to do a perimeter study like the one described in exersize 2.11

## Chapter 3

## Chapter 4
#### excercise 4.3
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

$$