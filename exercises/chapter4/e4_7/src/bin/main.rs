use e4_7::*;

fn main() {
  car_env().policy_iteration(
    20.0, // theta
    0.4, // alpha
    1e-4, // epsilon
    Some(0), //start action
    Some(1) //cut off after 10 iterations
  ).is_some();
}