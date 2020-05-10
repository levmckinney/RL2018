use clap::{Arg, App};
use rand::thread_rng;
use image;
use e5_12::*;

fn main() {
  let matches = App::new("RL 2018 exercise 5.12")
  .version("0.1.0")
  .author("Lev McKinney. <levmckinney@gmail.com>")
  .about("Runs monte carlo")
  .arg(Arg::with_name("track")
       .short("t")
       .long("track")
       .value_name("FILE")
       .help("Input image that represents the track")
       .takes_value(true)
       .required(true))
  .arg(Arg::with_name("example")
       .help("file to output example of optimal policy")
       .short("e")
       .long("example")
       .value_name("FILE")
       .required(false))
  .arg(Arg::with_name("num_episodes")
       .help("Number of episodes to simulate")
       .short("n")
       .takes_value(true)
       .value_name("NUMBER")
       .required(true)
      )
   .arg(Arg::with_name("algo")
       .short("a")
       .long("algo")
       .value_name("exploring | off-policy")
       .default_value("off-policy")
       .takes_value(true)
      )
    .arg(Arg::with_name("epsilon")
        .long("epsilon")
        .default_value("0.1")
        .takes_value(true)
        .value_name("NUM IN [0,1]"))
    .arg(Arg::with_name("value")
        .long("value")
        .short("v")
        .takes_value(true)
        .value_names(&["V_X", "V_Y", "A_X", "A_Y","FILE"])
        .multiple(true)
        .number_of_values(5))
    .arg(Arg::with_name("max_ep_len")
      .long("max_ep_len")
      .default_value("1000000")
      .takes_value(true)
      .value_name("NAT NUM"))
  .get_matches();

  let race_track_path = matches.value_of("track").unwrap();
  let mut race_track_img = image::open(race_track_path).expect("file to exist").to_rgb();
  let mut env = RaceTrackEnv::new(&race_track_img);
  let num_episodes = matches.value_of("num_episodes")
                            .unwrap()
                            .parse::<u32>()
                            .expect("valid number");
  
  let mut rng = thread_rng();
  let algo = matches.value_of("algo").unwrap();
  let epsilon = matches.value_of("epsilon")
    .unwrap()
    .parse::<f64>()
    .expect("epsilon to be valid float in [0, 1]");
  let max_ep_len = matches.value_of("max_ep_len")
    .unwrap()
    .parse::<u32>()
    .expect("max_ep_len to be a positive number less than u32::MAX");

  macro_rules! save_example {
    ($policy:expr) => {
      println!("Saving example");
      matches.value_of("example").map(|example_path| {
        plot_example(&mut env, &mut race_track_img, $policy, max_ep_len);
          race_track_img.save(example_path).expect(
          "to be able to save"
        );
      });
      println!("Complete!");
    };
  }

  macro_rules! save_value {
    ($value:expr) => {
      println!("Saving value");
      matches.values_of("value").map(|args| {
        let args: Vec<_> = args.collect();
        let v_x = args[0].parse::<i32>().expect("V_X to be a integer of size 32");
        let v_y = args[1].parse::<i32>().expect("V_Y to be a integer of size 32");
        let a_x = args[2].parse::<i32>().expect("A_X to be a integer of size 32");
        let a_y = args[3].parse::<i32>().expect("A_Y to be a integer of size 32");
        println!("velocity: {:?}, acceleration(i.e. action):{:?}", (v_x, v_y), (a_x, a_y));
        let mut value_img = image::RgbImage::new(race_track_img.width(), race_track_img.height());

        plot_value_at_velocity(&mut env, &mut value_img, (v_x, v_y), (a_x, a_y), $value);
        value_img.save(args[4]).expect(
          "to be able to save"
        );
      });
      println!("Complete!");
    };
  }
  
  let default_value = -(((max_ep_len + 1)/2) as f64);
  assert!(default_value.is_finite());

  if algo == "exploring" {
    let (es_policy, es_value) = monte_carlo_es(&mut env, num_episodes, None, max_ep_len, default_value, &mut rng);
    save_example!(es_policy);
    save_value!(es_value);
  } else if algo == "off-policy" {
    let (off_policy, off_policy_value) = mc_off_policy_epsilon_greedy(&mut env, num_episodes, epsilon, max_ep_len, default_value,  &mut rng);
    save_example!(off_policy);
    save_value!(off_policy_value);
  } else {
    println!("algorithm {} not supported", algo);
    return;
  }
}