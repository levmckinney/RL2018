use std::vec::Vec;
use std::hash::Hash;
use std::cmp::Eq;
use rand::prelude::*;
use rand::distributions::Bernoulli;
use image::{RgbImage, Rgb};
use indexmap::IndexMap;
use indexmap::IndexSet;
use indicatif::{ProgressBar, ProgressStyle};
use std::cmp::max;
use std::cmp::Ordering;
use std::fmt::Debug;
// Im using index map/set to allow for consistent runs.
type Set<T> = IndexSet<T>;
type Map<K, V> = IndexMap<K, V>;

// How recently waited the progress bar is.
const BAR_RECENCY: f64 = 0.001;

/// Preforms argmax as expected
pub fn argmax<T>(over: impl Iterator<Item=T>, func: impl Fn(T) -> f64) -> Option<T> 
where T: Copy {
   over
  .map(|item| (func(item), item))
  .max_by(|(v1, _), (v2, _)| {
    if v1.is_nan() || v2.is_nan() {
      panic!("cant order nan")
    }
    if v1 == v2 {
      Ordering::Equal
    } else if v1 > v2 {
      Ordering::Greater
    } else {
      Ordering::Less
    }
  })
  .map(|(_, argmax)| argmax)
}

/// Basic monte-carlo environment
pub trait MonteEnvironment<State, Action>:
where State: Hash+Eq+Copy+Send+Sync+Debug,
      Action: Hash+Eq+Copy+Send+Sync+Debug {
  fn start_states(&self) -> &Set<State>; // start states
  fn actions(&self, state: &State) -> &Set<Action>; // Action set
  fn next_state(&self, s:State, a:Action, rng: &mut StdRng) -> (f64, Option<State>);
  fn gamma(&self) -> f64; // Discount factor in [0, 1]

  // Generate episode in the form
  // [(S_0, A_0, R_1), (S_1, A_1, R_2), ..., (S_{T-1}, A_{T-1}, R_T)]
  fn episode<F> (&self, start: &State, 
    start_action: Option<&Action>,
    policy: F, 
    max_ep_len: Option<u32>,
    rng: &mut StdRng) -> Vec<(State, Action, f64)>
  where F: Fn(State, &mut StdRng) -> Action {
    let mut episode = Vec::new();
    let mut state = Some(*start);
    let mut action = *start_action.unwrap_or(&policy(state.unwrap(), rng));
    let mut i = 0;
    loop {
      i += 1;
      let result = self.next_state(state.unwrap(), action, rng);
      let reward = result.0;
      episode.push((state.unwrap(), action, reward));
      // next step
      state = result.1;
      if state.is_none() {break;} /* i.e. terminal state */
      action = policy(state.unwrap(), rng);
      if max_ep_len.is_none() || i > max_ep_len.unwrap() {
        break;
      }
    }
    episode
  }
}

/// An environment that provides is full state set.
pub trait StateFull<State>:
  where State: Hash+Eq+Copy+Send+Sync {
  fn states(&self) -> &Set<State>; // full state set
}

fn progress_bar(n: u32) -> ProgressBar {
  let bar = ProgressBar::new(n as u64);
  bar.set_style(ProgressStyle::default_bar()
  .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}"));
  bar
}

fn update_bar(bar:&ProgressBar, recent_ep_len: &mut Option<f64>, ep_len: usize,) {
  let ep_len = ep_len as f64;
  // Update the progress bar
  *recent_ep_len = match recent_ep_len {
    Some(recent) => Some(*recent + BAR_RECENCY*(ep_len - *recent)),
    None => Some(ep_len)
  };
  bar.inc(1);
  bar.set_message(&format!("recent ep len {}", recent_ep_len.unwrap()));  
}

/// Monte carlo exploring starts
/// n being the number of episodes
/// using first visit method
pub fn monte_carlo_es<State, Action, E>(env: &mut E, n: u32, 
  start_action: Option<Action>, 
  max_ep_len: u32,
  default_value: f64,
  rng: &mut StdRng) -> (impl Fn(State) -> Option<Action>, impl Fn(State, Action) -> f64)
  where State: Hash+Eq+Copy+Send+Sync+Debug,
        Action: Hash+Eq+Copy+Send+Sync+Debug, 
        E: MonteEnvironment<State, Action>+StateFull<State> {
  let mut policy: Map<State, Action> = Map::new();

  let mut action_value: Map<(State, Action), f64> = Map::new();
  macro_rules! q {
    ($state:expr, $action:expr) => {
      action_value.get(&($state, $action)).cloned().unwrap_or(default_value)
    };
  }
  
  // this is a value needed to in the incremental update later see RL 2018 chapter 5
  let mut updates: Map<(State, Action), f64> = Map::new();
  macro_rules! u {
    ($state:expr, $action:expr) => {
      updates.get(&($state, $action)).cloned().unwrap_or(0.0)
    };
  }

  // Don't need to initialize we will do this as we go
  println!("Starting Monte Carlo Simulations");
  let bar = progress_bar(n);
  let mut recent_ep_len = None;
  for _ in 0..n {
    let s_0 = env.states().iter().choose(rng).expect("no empty state set");
    let start_actions = env.actions(&s_0);
    let a_0 = start_actions.iter().choose(rng).expect("no empty action set");
    let ep = env.episode(&s_0, 
      Some(&a_0), 
      // behave using greedy policy
      |state, _| {
        policy.get(&state).cloned().unwrap_or_else(||
          //take default action
          start_action.unwrap_or_else(|| 
            *env.actions(&state).iter().last().expect("no empty action set")
          )
        )
      }, 
      Some(max_ep_len), 
      rng
    );

    update_bar(&bar, &mut recent_ep_len, ep.len());
    
    let terminal = ep.len() - 1;
    if terminal == 0 {
      continue;
    }
    let mut g = 0.0; // return so far
    for t in (0..terminal).rev() { // loop backwards along the states
      let (s_t, a_t, r_tplus1) = ep[t];
      g = env.gamma() * g + r_tplus1;
      if t == 0 || ep[0..(t-1)].iter().all(|(s, a, _)| (*s, *a) != (s_t, a_t)) {
        //first visit to s_t a_t during this episode
        updates.insert((s_t,a_t), 1.0 + u!(s_t,a_t));
        // Running average
        action_value.insert((s_t,a_t),
          q!(s_t, a_t) +(1.0/u!(s_t, a_t))*(g - q!(s_t, a_t))
        );
        policy.insert(s_t,
          *argmax(env.actions(&s_t).iter(), |a| q!(s_t, *a))
          .expect("argmax exists")
        );
      }
    }
  }
  bar.finish();
  println!("Simulation complete covered {} states", policy.len());
  (move |state| policy.get(&state).cloned(),
  move |state, action| q!(state, action))}

/// Uses off policy monte carlo to learn the optimal greedy policy
/// behaving according to an epsilon greedy policy.
pub fn mc_off_policy_epsilon_greedy<State, Action>(
  env: &mut impl MonteEnvironment<State, Action>, 
  n: u32,
  epsilon: f64,
  max_ep_len: u32,
  default_value: f64,
  rng: &mut StdRng) -> (impl Fn(State) -> Option<Action>, impl Fn(State, Action) -> f64)
  where State: Hash+Eq+Copy+Send+Sync+Debug,
        Action: Hash+Eq+Copy+Send+Sync+Debug {
  // Notice here we do not initialize since we do not expect to
  // explore most of the state space before stopping
  let mut action_value: Map<(State, Action), f64> = Map::new();
  macro_rules! q {
    ($state:expr, $action:expr) => {
      action_value.get(&($state, $action)).cloned().unwrap_or(default_value);
    };
  }
  
  // this is a value needed to in the incremental update later see RL 2018 5.6
  let mut sum_weights: Map<(State, Action), f64> = Map::new();
  macro_rules! c {
    ($state:expr, $action:expr) => {
      sum_weights.get(&($state, $action)).cloned().unwrap_or(0.0)
    };
  }
  
  // the target policy is a greedy policy
  let mut greedy_policy: Map<State, Action> = Map::new();
  
  //needed to create epsilon greedy behavior policy
  let explore = Bernoulli::new(epsilon).expect("epsilon in [0,1]");
  //helpful in batching updates to the target greedy policy
  let mut greedy_updates = Map::new();
  println!("Starting Monte Carlo simulations");
  let bar = progress_bar(n);
  let mut recent_ep_len = None;
  for _ in 0..n {
    /*lifetime of behavior policy*/{
      // initialize epsilon greedy behavior policy
      let behavior_policy = |state: State, rng: &mut StdRng| {
        let greedy_action = greedy_policy.get(&state);
        if explore.sample(rng) || greedy_action.is_none() {
          *env.actions(&state).iter().choose(rng).expect("actions not to be empty")
        } else {
          *greedy_action.unwrap()
        }
      };
      // probabilities of taking certain actions under behavior policy
      let p_behavior_policy = |s: State, a: Action| {
        let actions = env.actions(&s);
        let num_actions = actions.len() as f64;
        let greedy_action = greedy_policy.get(&s);
        if !actions.contains(&a) {
          0.0
        } else if greedy_action.is_none() {
          1.0/num_actions
        } else if *greedy_action.unwrap() == a {
          epsilon/num_actions + (1.0 - epsilon)
        } else {
          epsilon/num_actions
        }
      };
      
      // Generate episode starting from random position
      let ep = env.episode(
        env.start_states().iter().choose(rng).expect("start states not to be empty"),
        None, behavior_policy, Some(max_ep_len), rng
      );

      update_bar(&bar, &mut recent_ep_len, ep.len());

      let terminal = ep.len() - 1;
      let mut g = 0.0; // return so far
      let mut w = 1.0; // weighted importance sampling ratio so far
      for t in (0..terminal).rev() { // loop backwards along the states
        let (s_t, a_t, r_tplus1) = ep[t];
        
        g = env.gamma()*g + r_tplus1;
        sum_weights.insert((s_t, a_t), c!(s_t, a_t) + w);
        
        action_value.insert((s_t, a_t), 
          q!(s_t, a_t) + (w/c!(s_t, a_t))*(g - q!(s_t, a_t))
        );

        greedy_updates.insert(s_t, 
          *argmax(env.actions(&s_t).iter(), |a| q!(s_t, *a))
          .expect("not to have empty action set")
        );
        
        if a_t != greedy_updates[&s_t] { break; }
        
        w *= 1.0/p_behavior_policy(s_t, a_t);
      }
    }/*end of behavior policies lifetime*/
    //safe to update greedy policy
    greedy_policy.extend(greedy_updates.iter());
    greedy_updates.clear()
  }
  bar.finish();
    println!("Simulation complete covered {} states", greedy_policy.len());
    (move |state| greedy_policy.get(&state).cloned(),
    move |state, action| q!(state, action))
}

#[derive(Clone, PartialEq, Eq, Copy, Hash, Debug)]
pub struct CarState {
  position: (i32,i32),
  velocity: (i32, i32)
}
  
pub struct RaceTrackEnv {
    states: Set<CarState>,
    end_positions: Set<(i32, i32)>,
    actions: Set<(i32, i32)>,
    start_states: Set<CarState>,
  velocity_freeze: Bernoulli
}

impl RaceTrackEnv {
  /// Creates a new race track environment from an RGB image
  /// the start line is red the track is white and the finish area is green
  /// note that the finishing area should be more then a line since the 
  /// car may skip over it in its final step. All other areas should be black.
  pub fn new(img: &RgbImage) -> RaceTrackEnv {
    println!("Creating race track env");
    let mut states = Set::new();
    let mut end_positions = Set::new();
    let mut start_states = Set::new();
    let mut actions = Set::new();
    for a_x in -1..=1 {
      for a_y in -1..=1 {
        actions.insert((a_x, a_y));
      }
    }
    let dims = img.dimensions();
    let max_vel = {
      let mut p = 0; let mut v = 0; 
      while p < max(dims.0, dims.1) {
        v += 1;
        p += v;
      }
      v
    } as i32;
    println!("max velocity {}", max_vel);
    let mut add_states_at = |position: (i32, i32)| {
      for v_x in -max_vel..=max_vel {
        for v_y in -max_vel..=max_vel {
          states.insert(CarState {
            position, velocity: (v_x, v_y)
          });
        }
      }
    };

    let is_white = |rgb: Rgb<u8>| { 
      rgb[0] > 230 && rgb[1] > 230 && rgb[2] > 230
    };
    let is_green = |rgb: Rgb<u8>| {
      rgb[0] < 100 && rgb[1] > 100 && rgb[2] < 100
    };
    let is_red = |rgb: Rgb<u8>| {
      rgb[0] > 100 && rgb[1] < 100 && rgb[2] < 100
    };
    let mut set = Set::new();
    for y in 0..img.height() {
      for x in 0..img.width() {
        let pix = img.get_pixel(x, y);
        set.insert((pix[0], pix[1], pix[2]));
        if is_red(*pix) {
          print!("=");
          start_states.insert(CarState {
            position: (x as i32, y as i32),
            velocity: (0, 0)
          });
          add_states_at((x as i32, y as i32));
        } else if is_green(*pix) {
          print!("#");
          end_positions.insert((x as i32, y as i32));
        } else if is_white(*pix) {
          print!("X");
          add_states_at((x as i32, y as i32));
        } else {
          //not in track
          print!("_");
        }
      }
      print!("\n");
    }
    println!("There are a total of {} states", states.len());
    RaceTrackEnv {
      start_states,
      states,
      actions,
      end_positions,
      velocity_freeze: Bernoulli::new(0.1).unwrap()
    }
  }
}

// Environments that provide their full state set
impl StateFull<CarState> for RaceTrackEnv {
  fn states(&self) -> &Set<CarState> { 
    &self.states
  }
}

impl MonteEnvironment<CarState, (i32, i32)> for RaceTrackEnv {
  fn actions(&self, _: &CarState) ->  &Set<(i32,i32)>{
    &self.actions
  }
  fn start_states(&self) -> &Set<CarState> { 
    &self.start_states
  }
  fn gamma(&self) -> f64 { 1.0 }

  fn next_state(&self, s:CarState, a:(i32, i32), rng: &mut StdRng) -> (f64, Option<CarState>) {
    //update velocity
    let new_v = if self.velocity_freeze.sample(rng) {
      s.velocity
    } else {
      (s.velocity.0 + a.0, s.velocity.1 + a.1)
    };
    // update position
    let new_p = (new_v.0 + s.position.0, new_v.1 + s.position.1);
    let new_s = CarState {
      position: new_p,
      velocity: new_v
    };
    if self.end_positions.contains(&new_p) {
      (0.0, None)
    } else if self.states.contains(&new_s) {
      (-1.0, Some(new_s))
    } else {
      (-1.0, self.start_states.iter().choose(rng).cloned())
    }
  }
}

// plot example trajectory.
pub fn plot_example(
  env:&mut impl MonteEnvironment<CarState, (i32, i32)>, 
  img:&mut RgbImage,
  action_value: impl Fn(CarState, (i32, i32)) -> f64,
  policy: impl Fn(CarState) -> Option<(i32, i32)>,
  max_ep_len: u32,
  rng: &mut StdRng) {
    let policy_rng = |state: CarState, rng: &mut StdRng| {
      let action_value_tuples: Vec<_> = env.actions(&state).iter().map(|a| (a, action_value(state, *a))).collect();
      println!("state {:?}, action_values {:?}", state, action_value_tuples);
      policy(state).unwrap_or_else(|| {
        println!("encountered new state {:?} not seen in simulations\n taking random action", state);
        *env.actions(&state).iter().choose(rng).expect("actions not to be empty")
      })
    };
    
    let start_state = env.start_states().iter().choose(rng).expect("start states not to be empty");
    let episode = env.episode(
      start_state, 
      None,
      policy_rng, 
      Some(max_ep_len), 
      rng);
    for (state, _, _) in episode {
      let (x,y) = state.position;
      img.put_pixel(x as u32, y as u32, Rgb([0,0,255]));
    }
}

#[inline]
fn in_range(x:i32, range: (i32, i32)) -> bool {
  range.0 <= x && x <= range.1
}

//plots the absolute value of the action value function at velocity and action
pub fn plot_value(
  env:&mut (impl MonteEnvironment<CarState, (i32, i32)>+StateFull<CarState>), 
  img:&mut RgbImage,
  x_v_range: (i32, i32),
  y_v_range: (i32, i32),
  action_value: impl Fn(CarState, (i32, i32)) -> f64) {
    println!("deriving average position value");
    let mut log_value: Map<(i32, i32), f64> = Map::new();
    let mut updates: Map<(i32, i32), f64> = Map::new();
    for state in env.states() {
      for action in env.actions(state) {
        let log_v = action_value(*state, *action);
        let CarState {velocity:(v_x, v_y), position} = state;
        if in_range(*v_x, x_v_range) && in_range(*v_y, y_v_range) {
          updates.insert(*position, updates.get(position).unwrap_or(&0.0) + 1.0);
          log_value.insert(*position, 
            log_value.get(position).unwrap_or(&0.0) + (1.0/updates[position])*(log_v - log_value.get(position).unwrap_or(&0.0))
          );
        }
      }
    }
    let max_value = log_value.iter().map(|(_, v)| v)
      .max_by(|v1, v2| v1.partial_cmp(v2).expect("expect no nan")).unwrap_or(&0.0);
    let min_value = log_value.iter().map(|(_, v)| v)
      .min_by(|v1, v2| v1.partial_cmp(v2).expect("expect no nan")).unwrap_or(&-200000.0);
    println!("avg values where in the range [{}, {}]", min_value, max_value);
    for ((x,y), avg_v) in log_value.iter() {
      let intensity = ((((avg_v - min_value)/(max_value - min_value))) * 255.0) as u8;
      img.put_pixel(*x as u32, *y as u32, Rgb([intensity, 0, 0]));
    }
}