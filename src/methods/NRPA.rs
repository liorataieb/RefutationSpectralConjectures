use crate::models::conjectures::conjectures_wagner_1::{State, Move};
use std::collections::HashMap;
use crate::tools::resultSaver::writeLine;
use std::time::Instant;
use rand::{Rng, thread_rng};
use crate::tools::{graphToDot, saveMatrix};

pub(crate) static PLAYOUT: usize = 100;
pub struct NRPA{
    pub best_yet : f64,
    pub timeout : f64,
    pub registerName : String,
    pub start_time : Instant,
    pub best_state: State
}

impl NRPA{
    pub fn new() -> Self {
        Self{
            start_time: Instant::now(),
            best_yet: f64::NEG_INFINITY,
            timeout: -1.0,
            registerName: String::new(),
            best_state: State::new()
        }
    }

    pub fn random_move(&self, moves: Vec<Move>, policy: &mut HashMap<Move, f64>) -> Move {
        let mut rng = thread_rng();

        let mut sum: f64 = 0.0;
        for &mv  in &moves {
            match policy.get(&mv){
                Some(v) => sum += v.exp(),
                None => {policy.insert(mv, 0.0); sum += 1.0}
            };

        }

        let stop = sum * rng.gen::<f64>();;
        sum = 0.0;
        for &mv in &moves {
            sum += policy.get(&mv).unwrap().exp();
            if sum > stop {
                return mv;
            }
        }

        return moves[0];
    }

    pub fn playout(&self, mut st : State, mut policy : HashMap<Move, f64>) -> State {
        let mut best_state: State = st.clone();
        let mut best_state_score = best_state.score();

        while !st.terminal() {
            let moves: Vec<Move> = st.legal_moves();
            if moves.len() == 0 {
                break;
            }
            let mv = self.random_move(moves, &mut policy);
            st.play(mv);

            if State::CONSIDER_NON_TERM {
                let sc = st.score();
                if sc > best_state_score {
                    best_state_score = sc;
                    best_state = st.clone();
                    best_state.best_score = sc;
                }
            }

            if st.score() > 0.0001 {
                return best_state
            }
        }

        if State::CONSIDER_NON_TERM{
            return best_state;
        }

        return st;
    }

    pub fn adapt(&self, mut policy: HashMap<Move, f64>, st: &mut State, ini_state: State) -> HashMap<Move, f64> {
        let mut s: State = ini_state.clone();
        let mut polp: HashMap<Move, f64> = policy.clone();

        for best in &mut st.seq[..] {
            let moves = s.legal_moves();
            let mut sum = 0.0;
            for &m in &moves {
                match policy.get(&m){
                    Some(v) => sum += v.exp(),
                    None => {policy.insert(m, 0.0); sum += 1.0}
                };
            }

            for &m in &moves {
                match polp.get(&m){
                    Some(v) => polp.insert(m, v - policy.get(&m).unwrap().exp()/sum),
                    None => polp.insert(m, -policy.get(&m).unwrap().exp()/sum)
                };

            }

            polp.insert(*best, polp.get(&best).unwrap() + 1.0);
            s.play(*best);
        }
        return polp;
    }

    pub fn nrpa(&mut self, level : i8, mut policy: HashMap<Move, f64>, ini_state : State, initial: bool, verbose: bool) -> State {
        let mut st: State = ini_state.clone();
        let mut stscore : f64 = st.score();

        if level == 0 || self.start_time.elapsed().as_secs_f64() > self.timeout {
            return self.playout(st, policy);
        }

        for i in 0..PLAYOUT {
            if initial {
                println!("NRPA loop {}, best score : {} {}", i, stscore, stscore);
            }

            let pol: HashMap<Move, f64> = policy.clone();
            let mut s = self.nrpa(level-1, pol, ini_state.clone(), false, verbose);
            let s_score = s.score();

            if stscore < s_score {
                st = s.clone();
                stscore = s_score;

                if stscore > self.best_yet {
                    self.best_yet = stscore;
                    self.best_state = s.clone();
                    self.best_state.best_score = self.best_yet;

                    let elapsed = self.start_time.elapsed().as_secs_f64();
                    println!("NRPA best score yet : {}", stscore);
                    if verbose {
                        let new_name = self.registerName.clone() + &*"_evolution".to_string();
                        writeLine("Conjecture ".to_owned() + &*st.conj.to_string()
                                      + " | NRPA best score yet : " + &*stscore.to_string()
                                      + " after " + &*elapsed.to_string()
                                      + "s, " + &*st.n_sommet.to_string()
                                      + " vertices\n", new_name);
                                      }

                    if s_score > 0.0001 {
                        let elapsed = self.start_time.elapsed().as_secs_f64();
                        if verbose {
                            writeLine("Conjecture ".to_owned() + &*st.conj.to_string()
                                          + "\n        Counterexample found in " + &*elapsed.to_string()
                                          + "s: best score = " + &*s_score.to_string()
                                          + "\n        With NRPA level " + &*level.to_string()
                                          + ", " + &*st.n_sommet.to_string()
                                          + " vertices\n\n", self.registerName.clone());
                                          }
                        println!("Conjecture {}\n   Counter-example found with NRPA level {} after {}s\n", st.conj, level, elapsed);

                        graphToDot::adj_matrix_to_dot(s.adj_mat.clone(), &*format!("{}/conj{}", self.registerName, st.conj));
                        saveMatrix::save_matrix(&*format!("{}/conj{}", self.registerName, st.conj), s.adj_mat.clone());

                        return self.best_state.clone()
                    }
                }
            }
            policy = self.adapt(policy, &mut st, ini_state.clone());
        }

        return self.best_state.clone()
    }
}


pub fn launch_nrpa(level: i8, ini_state: State, timeout: f64, verbose: bool, registerName: String) -> State {
    let policy = HashMap::new();

    let mut expe = NRPA::new();
    expe.timeout = timeout;
    expe.registerName = registerName;

    let st = expe.nrpa(level, policy, ini_state, true, verbose);

    return st;
}