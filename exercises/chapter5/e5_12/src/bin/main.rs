use clap::{Arg, App};
use rand::prelude::*;
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
       .long("num_episodes")
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
        .allow_hyphen_values(true)
        .value_names(&["V_X_MIN", "V_X_MAX", "V_Y_MIN", "V_Y_MAX","FILE"])
        .multiple(true)
        .number_of_values(5))
    .arg(Arg::with_name("max_ep_len")
      .long("max_ep_len")
      .default_value("10000000")
      .takes_value(true)
      .value_name("NAT NUM"))
    .arg(Arg::with_name("seed")
    .long("seed")
    .short("s")
    .takes_value(true))
  .get_matches();

  let race_track_path = matches.value_of("track").unwrap();
  let mut race_track_img = image::open(race_track_path).expect("file to exist").to_rgb();
  let mut env = RaceTrackEnv::new(&race_track_img);
  let num_episodes = matches.value_of("num_episodes")
                            .unwrap()
                            .parse::<u32>()
                            .expect("valid number");
  
  let mut rng = matches.value_of("seed").map(|seed_str| {
    let seed = seed_str.parse::<u64>().unwrap();
    println!("running from seed");
    SeedableRng::seed_from_u64(seed)
  }).unwrap_or(SeedableRng::from_entropy());
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
    ($policy:expr, $value:expr) => {
      println!("Saving example");
      matches.value_of("example").map(|example_path| {
        plot_example(&mut env, &mut race_track_img, $value, $policy, max_ep_len, &mut rng);
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
        let v_x_min = args[0].parse::<i32>().expect("V_X_MIN to be a integer of size 32");
        let v_x_max = args[1].parse::<i32>().expect("V_X_MAX to be a integer of size 32");
        let v_y_min = args[2].parse::<i32>().expect("V_Y_MIN to be a integer of size 32");
        let v_y_max = args[3].parse::<i32>().expect("V_Y_MAX to be a integer of size 32");
        println!("x velocity range: {:?}, acceleration(i.e. action):{:?}", (v_x_min,v_x_max), (v_y_min, v_y_max));
        let mut value_img = image::RgbImage::new(race_track_img.width(), race_track_img.height());

        plot_value(&mut env, &mut value_img, (v_x_min,v_x_max), (v_y_min, v_y_max), $value);
        value_img.save(args[4]).expect(
          "to be able to save"
        );
      });
      println!("Complete!");
    };
  }
  
  let default_value = -(((max_ep_len + 1 + 1)/2) as f64);
  println!("Using default value {}", default_value);
  assert!(default_value.is_finite());

  if algo == "exploring" {
    let (es_policy, es_value) = monte_carlo_es(&mut env, num_episodes, None, max_ep_len, default_value, &mut rng);
    save_example!(&es_policy, &es_value);
    save_value!(es_value);
  } else if algo == "off-policy" {
    let (off_policy, off_policy_value) = mc_off_policy_epsilon_greedy(&mut env, num_episodes, epsilon, max_ep_len, default_value,  &mut rng);
    save_example!(&off_policy, &off_policy_value);
    save_value!(&off_policy_value);
  } else {
    println!("algorithm {} not supported", algo);
    return;
  }
}