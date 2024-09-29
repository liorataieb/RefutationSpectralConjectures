#![allow(non_snake_case)]

#[macro_use]
extern crate core;

use std::io::prelude::*;
use std::io::Read;
use std::io::Write;
use ndarray_linalg::Eig;
use std::time::Instant;
use rand::prelude::*;
use ndarray_rand::rand_distr::num_traits::real::Real;
use crate::models::conjectures::conjectures_wagner_1::State;

mod tools;
mod methods;
mod models;


fn main() {
    println!("Hello, world!\n");

    let level = 1;
    let terminal = 20;
    let heuristic = 10.0;
    let timeout = 60.0;
    let verbose = true;

    let total = Instant::now();

    for i in 1..69 {
        let mut st = State::new();
        st.size_terminal = terminal;
        st.conj = i as usize;

        let start = Instant::now();
        println!("Conjecture {}", i);

        let st1 = methods::NMCS::launch_nmcs(st.clone(), level, heuristic, verbose, timeout, String::from(format!("NMCS{}", level)));
        let st2 = methods::NRPA::launch_nrpa(level, st.clone(), timeout, verbose, String::from(format!("NRPA{}", level)));
        let st3 = methods::GRAVE::launch_grave(st.clone(), 50, 0.0, heuristic, timeout, verbose, "GRAVE".to_string());
        let st4 = methods::BFS::launch_bfs(st.clone(),heuristic, -1, timeout, verbose, "BFS".to_string());
        let st5 = methods::ILS::iterative_local_search(terminal, 4, i as usize, timeout, verbose, "ILS".to_string());
        let st6 = methods::CMAES::launch_CMAES(10, 2000, i as usize, terminal, verbose, "CMAES".to_string());

        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("Conjecture {}\nTime : {}s, {}min", i, duration.as_secs_f64(), duration.as_secs_f64() / 60.0);
    }

    let end = Instant::now();
    let duration = end.duration_since(total);
    println!("\n\nAll Time : {}s, {}min", duration.as_secs_f64(), duration.as_secs_f64() / 60.0);
}