#[macro_use] extern crate cached;
use rayon::prelude::*;
use std::vec::Vec;
use std::collections::HashMap;
use std::hash::Hash;
use std::cmp::Eq;
use std::cmp::min;

pub struct Environment <State, Action>
    where State: Hash+Eq+Copy+Send+Sync, Action: Hash+Eq+Copy+Send+Sync {
    pub states: Vec<State>, // State set
    pub e_reward: fn(&State, &Action) -> f64, // expected reward
    pub actions: fn(&State) -> Vec<Action>, // Action set
    // state_trans it is assumed that the action will be valid given the current state
    pub state_trans: fn(&State, &State, &Action) -> f64,
    pub gamma: f64, // Discount factor in [0, 1]
}

impl <State, Action> Environment<State, Action> 
    where State: Hash+Eq+Copy+Send+Sync, Action: Hash+Eq+Copy+Send+Sync {
    // returns the value of the left hand side of the bellman equation for value function
    pub fn bellman_value_update(
        &self,
        s:&State, 
        a:&Action, 
        value: &HashMap<State, f64>,
        ) -> f64{
        let Environment{states, e_reward, actions:_, state_trans, gamma } = self;
        // sweet parallelism
        states.par_iter().map(
            |next_s| -> f64 {
                rewards.iter().map(
                    |r| dynamics(next_s, r, s, a) * ((*r as f64) + (gamma)*value[next_s])
                ).sum()
            }
        ).sum()
    }

    pub fn policy_iteration (
        &self,
        // These hyper parameters are explained above
        theta: f64,
        alpha: f64,
        epsilon: f64,
        start_action: Option<Action>,
        // n is the number of iterations to stop after.
        // if n is none then the process will run until convergence
        n:Option<u64>) -> Option<(HashMap<State, Action>, HashMap<State, f64>)> {
        let Environment{states, e_reward:_, actions, state_trans:_, gamma:_ } = self;
        let mut value: HashMap<State, f64> = HashMap::new(); 
        let mut policy: HashMap<State, Action> = HashMap::new();

        // used to determine convergence/when to stop
        let mut policy_stable: bool;
        let mut value_stable: bool;
        let mut last_value: HashMap<State, f64>;
        let mut d_value: HashMap<State, f64> = HashMap::new(); // exponential recent change in value
        let mut i = 0;

        // Initialize
        for s in states {
            value.insert(*s, 0.0);
            d_value.insert(*s, 0.0);
            if let Some(action) = start_action{
                policy.insert(*s, action);
            } else {
                if actions(s).is_empty() {
                    println!("action set empty!");
                    return None;
                }
                policy.insert(*s, actions(s)[0]);
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
                for s in states {
                    let v = value[s];
                    value.insert(
                        *s, 
                        self.bellman_value_update(
                            &s,
                            &policy[s],
                            &value
                        )
                    );
                    delta = if delta > (v - value[s]).abs() {
                        delta
                    } else {
                        (v - value[s]).abs()
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
            for (i, s) in states.iter().enumerate() {
                if i % 30 == 0 {println!("-({}/{})", i+1, states.len())}
                let old_action = policy[s];
                let mut greedy_action = old_action;
                let mut best_value = f64::NEG_INFINITY;
                for a in actions(s) {
                    let value = self.bellman_value_update(
                        &s,
                        &a,
                        &value
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
            if policy_stable {
                println!("Policy stabilized after {} iterations.", i);
                break;
            }
            value_stable = true;
            for s in states {
                d_value.insert(*s, d_value[s] + alpha*((value[s] - last_value[s]) - d_value[s]));
                if d_value[s] > epsilon {
                    value_stable = false;
                }
            }
            if value_stable {
                println!("Value stabilized after {} iterations", i + 1);
                break;
            }
            i += 1;
            println!("Completed {} iteration(s).", i);// just lets you know its doing something
        }
        Some((policy, value))
    }
}

// Im using unsigned ints here since it makes caching simpler

fn ln_factorial(n: u32) -> f64 {
    let mut total = 0.0;
    for i in 1..=n {
        total += (i as f64).ln();
    }
    return total;
}

fn ln_poisson(n:u32, lamb:u32) -> f64 {
    let lamb = lamb as f64;
    let ln_prob = lamb.ln() * (n as f64) - lamb - ln_factorial(n);
    ln_prob
}

fn poisson(n:u32, lamb:u32) -> f64 {
    ln_poisson(n, lamb).exp()
}

// Truncated poisson distribution.
// if X is poisson(lamb) then min{X, c} is truncated poisson(c, lamb)
fn trunc_poisson(n:u32, lamb:u32, c:u32) -> f64 {
    if n < c{
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

// Expected value of truncated poisson
fn e_trunc_poisson(lamb:u32, c:u32) -> f64{
    let sum: f64 = (0..c).map(|i| { 
        ((i - c) as f64) * poisson(i, lamb)
    }).sum();
    1.0 + sum
}

// if X_1 follows trunc_poisson(lamb_1, c_1) and X_2 follows trunc_poisson(lamb_2, c_2)
// then Y = X_1 - X_2 follows dif_trunc_poisson(lamb_1, c_1, lamb_2, c_2)  
fn dif_trunc_poisson(n:i32, lamb_1:u32, c_1:u32, lamb_2:u32, c_2: u32) -> f64{
    if n >= 0 {
        let n = n as u32;
        (n..=min(c_1, c_2 + n)).map(|n_1| {
            let n_2 = n_1 - n;
            trunc_poisson(n_1, lamb_1, c_1) * trunc_poisson(n_2, lamb_2, c_2)
        }).sum()
    } else {
        dif_trunc_poisson(-n, lamb_2, c_2, lamb_1, c_1)
    }
}

pub fn car_env() -> Environment<(i32, i32), i32> {
    let states = {
        let mut states = Vec::new();
        for i in 0..=20 {
            for j in 0..=20 {
                states.push((i,j))
            }
    };
    let actions = |s| {
        (-min(s.1, min(5, 20-s.0))..=min(s.0, min(5, 20-s.1))).collect()
    };
    let e_reward_map = HashMap::new();
    let state_trans_map = HashMap::new();
    for s in states {
        for a in actions(s) {
            e_reward_map.insert((s, a), {
                10.0 * e_trunc_poisson(3.0, s.0) * e_trunc_poisson(4.0, s.1) - 2.0 * a
            });
            for next_s in states {
                let mut prob = 0.0;
                state_trans_map.insert((next_s, s, a),{
                    let delta_cars_lot_0 = next_s.0 - s.0 - a;
                    let delta_cars_lot_1 = next_s.1 - s.1 + a;    
                    let temp = dif_trunc_poisson(delta_cars_lot_0, 3, 20 - s.0, 3, s.0)
                               * dif_trunc_poisson(delta_cars_lot_1, 2, 20 - s.1, 4, s.1);
                });
                println!("{}", prob)
            }
        }
    }
    Environment {
        states: states
        // expected reward
        e_reward: |s, a| {

        },
        actions: actions, // Action set
        //state_trans it is assumed that the action will be valid given the current state
        state_trans: |next_s, s, a| {
            
        },
        gamma: 0.9 // Discount factor in [0, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert!(car_env().policy_iteration(
            0.1, // theta
            0.4, // alpha
            1e-4, // epsilon
            Some(0), //start action
            Some(10) //cut off after 10 iterations
        ).is_some());
    }
    #[test]
    fn sum_to_one() {
        let Environment{states, rewards, actions, dynamics, gamma:_ } = car_env();
        let prob: f64 = states.par_iter().map(
            |next_s| -> f64{
                rewards.iter().map(
                    |r| dynamics(next_s, r, &(0,0), &0)
                ).sum()
            }
        ).sum();
        assert_eq!(prob, 1.0);
    }
}