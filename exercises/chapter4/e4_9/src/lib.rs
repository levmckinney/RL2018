use dashmap::DashMap;
use std::vec::Vec;
use std::hash::Hash;
use std::cmp::Eq;
use std::cmp::min;


pub trait Environment<State, Action>: Send + Sync
    where State: Hash+Eq+Copy+Send+Sync + std::fmt::Debug, 
    Action: Hash+Eq+Copy+Send+Sync + std::fmt::Debug{
    fn states(&self) -> &[State]; // State set
    fn rewards(&self) -> &[i32]; // Reward set
    fn actions(&self, state: &State) -> Vec<Action>; // Action set
    //dynamics it is assumed that the action will be valid given the current state
    fn dynamics(&self, next_s: &State, r: &i32, s: &State, a: &Action) -> f64;
    fn gamma(&self) -> f64; // Discount factor in [0, 1]
    
    // returns the value of the left hand side of the bellman equation for value function
    fn bellman_value_update(
        &self,
        s:&State, 
        a:&Action, 
        value: &DashMap<State, f64>,
        ) -> f64{
        // sweet parallelism
        self.states().iter().map(
            |next_s| -> f64 {
                self.rewards().iter().map(
                    |r| {
                        self.dynamics(next_s, r, s, a) * ((*r as f64) + self.gamma()*(*value.get(next_s).unwrap()))
                    }
                ).sum()
            }
        ).sum()
    }

    fn value_iteration(&self, theta: f64) -> (DashMap<State, Action>, DashMap<State, f64>) {
        // Initialize value function
        let value = DashMap::new();
        for s in self.states() {
            value.insert(*s, 0.0);
        }
        // value iteration
        println!("Beginning value iteration.");
        let par_ord = |v_1:&f64, v_2:&f64| v_1.partial_cmp(v_2).expect("Nan in values");
        loop {
            let mut delta = 0.0;
            for s in self.states() {
                let v = *value.get(s).unwrap();
                value.insert(*s, 
                    self.actions(s).iter().map(|a|
                        self.bellman_value_update(s, a, &value)
                    ).max_by(par_ord)
                    .expect("empty action set")
                );
                let state_value_delta = (v - *value.get(s).unwrap()).abs();
                delta = if state_value_delta > delta {
                    state_value_delta
                } else {
                    delta 
                };
            }
            println!("-({} > {})", delta, theta);
            if delta < theta {
                break;
            }
        }
        // produce policy
        let policy = DashMap::new();
        let value_ord = |av_1: &(f64, _), av_2:&(f64, _)| {
            av_1.0.partial_cmp(&av_2.0).expect("Nan in values")
        };
        for s in self.states() {
            policy.insert(*s, {
                self.actions(s).iter().map(|a|
                    (self.bellman_value_update(s, a, &value), a.clone())
                ).max_by(value_ord)
                .expect("empty action set").1
            });
        }
        (policy, value)
    }
}

pub struct GamblersEnv {
    states: Vec<i32>,
    rewards: Vec<i32>,
    p_h: f64
}

impl GamblersEnv {
    pub fn new(p_h: f64) -> GamblersEnv {
        GamblersEnv {
            states: (0..=100).collect(),
            rewards: vec![0,1],
            p_h
        }
    }
}

impl Environment<i32, i32> for GamblersEnv {
    fn states(&self) -> &[i32] {
        self.states.as_slice()
    }
    
    fn rewards(&self) -> &[i32] {
        self.rewards.as_slice()
    }

    fn actions(&self, s: &i32) -> Vec<i32> {
        (0..=min(*s, 100-*s)).collect()
    }

    fn dynamics (&self, next_s: &i32, r: &i32, s: &i32, a: &i32) -> f64 {
        if *s == 0 || *s == 100 { // dead states game is over
            if *r == 0 && *next_s == *s {1.0} else {0.0}
        } else if *next_s != s + a && *next_s != s - a {
            0.0 // inconsistent transition
        } else if (*next_s != 100 && *r == 1) || (*next_s == 100 && *r == 0) {
            0.0 // inconsistent reward
        } else if s == next_s && *a == 0 {
            1.0 // we will always return to the same state if we wager nothing
        } else if *next_s == s + a {
            self.p_h // win wager
        } else if *next_s == s - a {
            1.0 - self.p_h // lost wager
        } else {
            0.0
        }
    }

    fn gamma(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sum_to_one(env: &dyn Environment<i32, i32>) {
        env.states().iter().for_each(|s| {
            env.actions(&s).iter().for_each(|a| {
                let prob: f64 = env.states().iter().map(
                    |next_s| -> f64 {
                        env.rewards().iter().map(
                            |r| {
                                env.dynamics(next_s, r, &s, &a)
                            }
                        ).sum()
                    }
                ).sum();
                assert!(0.98 < prob && prob  < 1.02, 
                    "failed for state {:?} and action {:?}, prob {}", s, a, prob)
            });
        });
    }
    #[test]
    fn gambler_sum_to_one() {
        sum_to_one(&GamblersEnv::new(/*theta=*/0.25));
    }

    #[test]
    fn reward() {
        assert_eq!(GamblersEnv::new(/*theta=*/0.25).rewards()[0], 0);
        assert_eq!(GamblersEnv::new(/*theta=*/0.25).rewards()[1], 1);
    }

    #[test]
    fn states() {
        let env = GamblersEnv::new(0.5);
        assert_eq!(env.states()[50], 50)
    }

    #[test]
    fn transition_to_win(){
        let p_h = 0.25;
        let env = GamblersEnv::new(p_h);
        let next_s = 100;
        let r = 1;
        let a = 50;
        let s = 50;
        assert!(env.actions(&s).contains(&a));
        assert!(env.states().contains(&s));
        assert!(env.states().contains(&next_s));
        assert!(env.rewards().contains(&r));
        assert_eq!(env.dynamics(&next_s, &r, &s, &a), p_h);
        assert!(env.dynamics(&next_s, &r, &s, &a) * ((r as f64) + env.gamma()*/*(*value.get(next_s).unwrap()=*/0.0) > 0.0);
    }
}
