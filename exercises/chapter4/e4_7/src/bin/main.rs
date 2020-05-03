use e4_7::*;
// Useful for benching
fn main() {
  let car_env = SimpleCarEnv::new();
  Environment::policy_iteration(
    &car_env,
    20.0, // theta
    0.4, // alpha
    1e-4, // epsilon
    Some(0), //start action
    Some(1) //cut off after 10 iterations
  ).unwrap();
}