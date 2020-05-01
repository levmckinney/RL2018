#[macro_use] 
extern crate cached;

use cached::stores::UnboundCache;
use rayon::prelude::*;
use std::vec::Vec;
use std::collections::HashMap;
use std::hash::Hash;
use std::cmp::Eq;
use std::cmp::min;

pub struct Environment <State, Action>
    where State: Hash+Eq+Copy+Send+Sync, Action: Hash+Eq+Copy+Send+Sync {
    pub states: Vec<State>, // State set
    pub rewards: Vec<i32>, // Reward set
    pub actions: fn(&State) -> Vec<Action>, // Action set
    //dynamics it is assumed that the action will be valid given the current state
    pub dynamics: fn(&State, &i32, &State, &Action) -> f64,
    pub gamma: f64, // Discount factor in [0, 1]
}

impl <State, Action> Environment<State, Action> 
    where State: Hash+Eq+Copy+Send+Sync, Action: Hash+Eq+Copy+Send+Sync {
    // returns the value of the left hand side of the bellman equation for value function
    pub fn bellman_value(
        &self,
        s:&State, 
        a:&Action, 
        value: &HashMap<State, f64>,
        ) -> f64{
        let Environment{states, rewards, actions:_, dynamics, gamma } = self;
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
        let Environment{states, rewards:_, actions, dynamics:_, gamma:_ } = self;
        let mut value: HashMap<State, f64> = HashMap::new(); 
        let mut policy: HashMap<State, Action> = HashMap::new();

        // used to determine convergence/when to stop
        let mut policy_stable: bool;
        let mut value_stable: bool;
        let mut last_value: HashMap<State, f64>;
        let mut d_value: HashMap<State, f64> = HashMap::new(); // expetaily recent change in value
        let mut i = 0;

        // Initialize
        for s in states {
            value.insert(*s, 0.0);
            d_value.insert(*s, 0.0);
            if start_action.is_none() {
                if actions(s).is_empty() {
                    println!("action set empty!");
                    return None;
                }
                policy.insert(*s, actions(s)[0]);
            } else {
                policy.insert(*s, start_action.unwrap());
            }
        }

        // policy iteration
        while !n.is_some() || i < n.unwrap() {
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
                        self.bellman_value(
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
                    let value = self.bellman_value(
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

fn factorial(n: i32) -> f64 {
    if n < 2 {
        1.0
    } else {
        n as f64 * factorial(n - 1)
    }
}


pub fn poisson(n:i32, lamb:f64) -> f64{
    if n >= 0 { 
        (lamb.powi(n)/factorial(n))*(-lamb).exp()
    } else {
        0.0
    }
}


pub fn car_env() -> Environment<(i32, i32), i32> {
    Environment {
        states: {
            let mut states = Vec::new();
            for i in 0..=20 {
                for j in 0..=20 {
                    states.push((i,j))
                }
            }
            states
        },
        rewards: (-10..=400).step_by(2).collect(), // Reward set
        actions: |s| {
            (-min(s.1, min(5, 20-s.0))..=min(s.0, min(5, 20-s.1))).collect()
        }, // Action set
        //dynamics it is assumed that the action will be valid given the current state
        dynamics: |next_s, r, s, a| {
        if (r - a*2) % 10 != 0 {
            // this is not possible since we should have sold a natural number of cars
            return 0.0;
        }
        let num_sold = (r - a*2) / 10;
        // compute the probability
        // sum over disjoint events
        let mut prob = 0.0;
        for sold_at_lot_1 in 0..=num_sold {
            // if we assume the number of cars sold at one lot is fixed then
            // we can compute the probability.
            // since each one of these events are disjoint we can sum there probabilities.
            let sold_at_lot_2 = num_sold - sold_at_lot_1;
            let num_end_of_day_lot_1 = next_s.0 + a; // we assume this includes the newly arived cars
            let num_end_of_day_lot_2 = next_s.1 - a;
            let num_arrived_lot_1 = num_end_of_day_lot_1 + sold_at_lot_1 - s.0;
            let num_arrived_lot_2 = num_end_of_day_lot_2 + sold_at_lot_2 - s.1;
            prob += 
                poisson(sold_at_lot_1, 3.0)
                *poisson(sold_at_lot_2, 4.0)
                *poisson(num_arrived_lot_1, 3.0)
                *poisson(num_arrived_lot_2, 2.0);
        }
        return prob
        },
        gamma: 0.9 // Discount factor in [0, 1]
    }
}
