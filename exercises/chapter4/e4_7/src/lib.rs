use rayon::prelude::*;
use dashmap::DashMap;
use std::vec::Vec;
use std::hash::Hash;
use std::cmp::Eq;
use std::cmp::min;

pub trait Environment <State, Action>: Send + Sync
    where State: Hash+Eq+Copy+Send+Sync, 
    Action: Hash+Eq+Copy+Send+Sync {
    fn states(&self) -> &[State]; // State set
    fn rewards(&self) -> &[i32]; // Reward set
    fn actions(&self, state: &State) -> Vec<Action>; // Action set
    //dynamics it is assumed that the action will be valid given the current state
    fn dynamics(&self, next_s: &State, r: &i32, s: &State, a: &Action) -> f64;
    fn gamma(&self) -> f64; // Discount factor in [0, 1]
}

impl <State, Action> dyn Environment<State, Action> 
    where State: Hash+Eq+Copy+Send+Sync, 
    Action: Hash+Eq+Copy+Send+Sync{    
    // returns the value of the left hand side of the bellman equation for value function
    pub fn bellman_value_update(
        &self,
        s:&State, 
        a:&Action, 
        value: &DashMap<State, f64>,
        ) -> f64{
        // sweet parallelism
        self.states().par_iter().map(
            |next_s| -> f64 {
                self.rewards().iter().map(
                    |r| {
                        self.dynamics(next_s, r, s, a) * ((*r as f64) + (self.gamma())*(*value.get(next_s).unwrap()))
                    }
                ).sum()
            }
        ).sum()
    }

    pub fn policy_iteration (
        &self,
        // These hyper parameters are explained in the notebook
        theta: f64,
        alpha: f64,
        epsilon: f64,
        start_action: Option<Action>,
        // n is the number of iterations to stop after.
        // if n is none then the process will run until convergence
        n:Option<i32>) -> Option<(DashMap<State, Action>, DashMap<State, f64>)> {
        let value: DashMap<State, f64> = DashMap::new(); 
        let policy: DashMap<State, Action> = DashMap::new();

        // used to determine convergence/when to stop
        let mut policy_stable: bool;
        let mut value_stable: bool;
        let mut last_value: DashMap<State, f64>;
        let d_value: DashMap<State, f64> = DashMap::new(); // exponential recent change in value
        let mut i = 0;

        // Initialize
        for s in self.states() {
            value.insert(*s, 0.0);
            d_value.insert(*s, 0.0);
            if let Some(action) = start_action{
                policy.insert(*s, action);
            } else {
                if self.actions(s).is_empty() {
                    println!("action set empty!");
                    return None;
                }
                policy.insert(*s, self.actions(s)[0]);
            }
        }

        // policy iteration
        while n.is_none() || i < n.unwrap() {
            // Save old values for convergence check
            last_value = value.clone();
            // policy evaluation
            println!("Starting evaluation");
            loop {
                let mut delta = f64::NEG_INFINITY;
                for s in self.states() {
                    let v = *value.get(s).unwrap();
                    value.insert(
                        *s, 
                        self.bellman_value_update(
                            &s,
                            &policy.get(s).unwrap(),
                            &value,
                        )
                    );
                    let change = (v - *value.get(s).unwrap()).abs();
                    delta = if delta >  change{
                        delta
                    } else {
                        change
                    }
                }
                if delta < theta {
                    break
                }
                println!("-({} > {})", delta, theta);
            }

            // policy improvement
            policy_stable = true;
            println!("Starting improvement");
            for (i, s) in self.states().iter().enumerate() {
                if i % 30 == 0 {println!("-({}/{})", i+1, self.states().len())}
                let old_action = *policy.get(s).unwrap();
                let mut greedy_action = old_action;
                let mut best_value = f64::NEG_INFINITY;
                for a in self.actions(s) {
                    let value = self.bellman_value_update(
                        &s,
                        &a,
                        &value,
                    );
                    if value > best_value {
                        best_value = value;
                        greedy_action = a;
                    }
                }
                policy.insert(*s, greedy_action);
                if greedy_action != old_action {
                    policy_stable = false;
                }
            }

            // check for convergence
            // policy stability
            if policy_stable {
                println!("Policy stabilized after {} iterations.", i);
                break;
            }
            
            // value function stability
            value_stable = true;
            let mut max_d_value = f64::NEG_INFINITY;
            for s in self.states() {
                let d_value_s = *d_value.get(s).unwrap();
                let value_s = *value.get(s).unwrap();
                let last_value_s = *last_value.get(s).unwrap();
                let d_value_s =  d_value_s + alpha*((value_s - last_value_s) - d_value_s);
                d_value.insert(*s, d_value_s);
                if d_value_s.abs() > max_d_value {
                    max_d_value = d_value_s.abs();
                }
                if d_value_s > epsilon {
                    value_stable = false;
                }
            }

            if value_stable {
                println!("Value stabilized after {} iterations", i + 1);
                break;
            }
            i += 1;
            println!("Completed {} iteration(s). Max recent value change {}.", i, max_d_value);
        }
        Some((policy, value))
    }
}

// Im using ints here since it makes caching simpler

fn ln_factorial(n: i32) -> f64 {
    let mut total = 0.0;
    for i in 1..=n {
        total += (i as f64).ln();
    }
    return total;
}

fn ln_poisson(n: i32, lamb: i32) -> f64 {
    let lamb = lamb as f64;
    let ln_prob = lamb.ln() * (n as f64) - lamb - ln_factorial(n);
    ln_prob
}

fn poisson(n:i32, lamb: i32) -> f64 {
    if n < 0 {
        0.0
    } else {
        ln_poisson(n, lamb).exp()
    }
}

// Truncated poisson distribution.
// if X is poisson(lamb) then min{X, c} is truncated poisson(c, lamb)
fn trunc_poisson(n:i32, lamb:i32, c:i32) -> f64 {
    if n < 0 {
        0.0
    } else if n < c {
        poisson(n, lamb)
    } else if n == c {
        let sum: f64 = (0..c).map(|i| { 
            poisson(i, lamb)
        }).sum();
        1.0 - sum
    } else {
        0.0
    }
}

struct CachedPoisson {
    trunc_poisson_cache: DashMap<(i32,i32,i32), f64>
}

impl CachedPoisson {
    fn new() -> CachedPoisson{
        CachedPoisson {
            trunc_poisson_cache:DashMap::new()
        }
    }

    fn trunc_poisson(&self, n:i32, lamb:i32, c:i32) -> f64 {
        let prob = *self.trunc_poisson_cache
        .entry((n, lamb, c))
        .or_insert_with(|| trunc_poisson(n, lamb, c));
        prob
    }
}
// This is the environment described in example 4.2 from RL 2018
pub struct SimpleCarEnv {
    states: Vec<(i32, i32)>,
    cache: CachedPoisson,
    rewards: Vec<i32>
}

impl SimpleCarEnv {
    pub fn new() -> SimpleCarEnv {
        SimpleCarEnv {
            states: {
                let mut states = Vec::new();
                for i in 0..=20 {
                    for j in 0..=20 {
                        states.push((i,j))
                    }
                }
                states
            },
            cache: CachedPoisson::new(),
            rewards: (-10..=400).step_by(2).collect()
        }
    }
}

impl Environment<(i32, i32), i32> for SimpleCarEnv {
    fn states(&self) -> &[(i32, i32)]{
        &self.states
    }

    fn actions(&self, s: &(i32, i32)) -> Vec<i32> {
        let vec: Vec<i32> = (-min(s.1, min(5, 20-s.0))..=min(s.0, min(5, 20-s.1))).collect();
        vec
    }

    fn rewards(&self) -> &[i32] {
        &self.rewards
    }

    fn dynamics(&self, next_s: &(i32, i32), r: &i32, s: &(i32, i32), a: &i32) -> f64 {
        let reward_from_cars = r + a.abs()*2;
        if reward_from_cars % 10 != 0 {
            // this is not possible since we should have sold a natural number of cars
            return 0.0;
        }
        let num_sold = reward_from_cars / 10;
        // sum over disjoint events
        let mut prob = 0.0;
        for sold_at_lot_0 in 0..=num_sold {
            // if we assume the number of cars sold at one lot is fixed then
            // we can compute the probability.
            // since each one of these events are disjoint we can sum their probabilities.
            let sold_at_lot_1 = num_sold - sold_at_lot_0;
            let parked_lot_0 = next_s.0 + sold_at_lot_0 + a - s.0;
            let parked_lot_1 = next_s.1 + sold_at_lot_1 - a - s.1;
            if sold_at_lot_0 < 0 
                || sold_at_lot_1 < 0
                || parked_lot_0 < 0
                || parked_lot_1 < 0 {
                continue;
            }
            prob += self.cache.trunc_poisson(sold_at_lot_0, 3, s.0 - a)
                    * self.cache.trunc_poisson(sold_at_lot_1, 4, s.1 + a)
                    // adding sold_at _lot may account for the decrepensy between this one and the text book
                    * self.cache.trunc_poisson(parked_lot_0, 3, 20 + sold_at_lot_0 - s.0 + a)
                    * self.cache.trunc_poisson(parked_lot_1, 2, 20 + sold_at_lot_1 - s.1 - a);
        }
        prob
    }

    fn gamma(&self) -> f64 {
        0.9
    }
}

// This is the environment described in exercise 4.7 in RL 2018
pub struct MoreComplexCarEnv {
    states: Vec<(i32, i32)>,
    cache: CachedPoisson,
    rewards: Vec<i32>
}

impl MoreComplexCarEnv {
    pub fn new() -> MoreComplexCarEnv {
        MoreComplexCarEnv {
            states: {
                let mut states = Vec::new();
                for i in 0..=20 {
                    for j in 0..=20 {
                        states.push((i,j))
                    }
                }
                states
            },
            cache: CachedPoisson::new(),
            rewards: (-18..=400).step_by(2).collect()
        }
    }
}

impl Environment<(i32, i32), i32> for MoreComplexCarEnv {
    fn states(&self) -> &[(i32, i32)]{
        &self.states
    }

    fn actions(&self, s: &(i32, i32)) -> Vec<i32> {
        let vec: Vec<i32> = (-min(s.1, min(5, 20-s.0))..=min(s.0, min(5, 20-s.1))).collect();
        vec
    }

    fn rewards(&self) -> &[i32] {
        &self.rewards
    }

    fn dynamics(&self, next_s: &(i32, i32), r: &i32, s: &(i32, i32), a: &i32) -> f64 {
        let cost_of_actions = if *a > 0 {
            2*(a - 1) // since the an employee can take one car
        } else {
            -a*2
        };
        let cost_of_storage = if s.0 > 10 {10} else {0} // cost of storage at lot 1
            + if s.1 > 10 {10} else {0}; // cost of storage at lot 2
        let reward_from_cars = r + cost_of_actions + cost_of_storage;
        if reward_from_cars % 10 != 0 {
            // this is not possible since we should have sold a natural number of cars
            return 0.0;
        }
        let num_sold = reward_from_cars / 10;
        // sum over disjoint events
        let mut prob = 0.0;
        for sold_at_lot_0 in 0..=num_sold {
            // if we assume the number of cars sold at one lot is fixed then
            // we can compute the probability.
            // since each one of these events are disjoint we can sum their probabilities.
            let sold_at_lot_1 = num_sold - sold_at_lot_0;
            let parked_lot_0 = next_s.0 + sold_at_lot_0 + a - s.0;
            let parked_lot_1 = next_s.1 + sold_at_lot_1 - a - s.1;
            if sold_at_lot_0 < 0 
                || sold_at_lot_1 < 0
                || parked_lot_0 < 0
                || parked_lot_1 < 0 {
                continue;
            }
            prob += self.cache.trunc_poisson(sold_at_lot_0, 3, s.0 - a)
                    * self.cache.trunc_poisson(sold_at_lot_1, 4, s.1 + a)
                    // adding sold_at _lot may account for the discrepancy between this one and the text book
                    * self.cache.trunc_poisson(parked_lot_0, 3, 20 + sold_at_lot_0 - s.0 + a)
                    * self.cache.trunc_poisson(parked_lot_1, 2, 20 + sold_at_lot_1 - s.1 - a);
        }
        prob
    }

    fn gamma(&self) -> f64 {
        0.9
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    fn sum_to_one(env: &Environment<(i32, i32), i32>) {
        env.states().par_iter().for_each(|s| {
            env.actions(&s).iter().for_each(|a| {
                println!("state {:?} and action {:?}", s, a);
                let prob: f64 = env.states().iter().map(
                    |next_s| -> f64 {
                        env.rewards().iter().map(
                            |r| env.dynamics(next_s, r, &s, &a)
                        ).sum()
                    }
                ).sum();
                assert!(0.98 < prob && prob  < 1.02, 
                    "failed for state {:?} and action {:?}", s, a)
            });
        });
    }
    #[test]
    fn simple_sum_to_one() {
        sum_to_one(&SimpleCarEnv::new());
    }

    #[test]
    fn complex_sum_to_one() {
        sum_to_one(&MoreComplexCarEnv::new());
    }
}
