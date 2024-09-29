use crate::models::conjectures::conjectures_wagner_1::{State};
use crate::tools::calc::softmaxChoice;
use crate::tools::resultSaver::writeLine;
use std::time::Instant;
use crate::tools::{graphToDot, saveMatrix};

pub struct NMCS{
    pub best_yet : f64,
    pub timeout : f64,
    pub registerName : String,
    pub start_time : Instant,
    pub best_state: State
}

impl NMCS{
    pub fn new() -> Self {
        Self{
            start_time: Instant::now(),
            best_yet: f64::NEG_INFINITY,
            timeout: -1.0,
            registerName: String::new(),
            best_state: State::new()
        }
    }

    pub fn playout(&mut self, mut st: State, heuristic_w : f64) -> State {
        let mut best_state: State = st.clone();
        let mut best_state_score = best_state.score();

        while !st.terminal() {
            let moves = st.legal_moves();
            if moves.len() == 0 {
                break
            }

            let mut i = ((moves.len() as f64)*rand::random::<f64>()) as usize;

            if heuristic_w != 0.0 {
                let mut weights = Vec::new();
                for &m in &moves {
                    weights.push(heuristic_w*st.heuristic(m));
                }

                i = softmaxChoice(weights);
            }

            let mv = moves[i];
            st.play(mv);

            if State::CONSIDER_NON_TERM {
                let sc = st.score();
                if sc > best_state_score {
                    best_state_score = sc;
                    best_state = st.clone();
                    best_state.best_score = best_state_score;
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

    pub fn nmcs(&mut self, mut st: State, n : i8, heuristic_w : f64, verbose : bool) -> State {
        let mut best_state: State = st.clone();
        let mut best_state_score = best_state.score();

        while !st.terminal(){
            let moves = st.legal_moves();
            if moves.len() == 0 {
                break
            }
            for &mv in &moves{
                if self.start_time.elapsed().as_secs_f64() > self.timeout && self.timeout > 0.0 {
                    return best_state
                }

                let mut new_st = st.clone();
                new_st.play(mv);
                if n <= 1 {
                    new_st = self.playout(new_st, heuristic_w);
                } else {
                    new_st = self.nmcs(new_st, n-1, heuristic_w, verbose);
                }
                let new_st_score = new_st.score();

                if new_st_score > best_state_score {
                    best_state = new_st.clone();
                    best_state_score = new_st_score;
                    best_state.best_score = best_state_score;

                    if best_state_score > self.best_yet {
                        self.best_yet = best_state_score;
                        self.best_state = best_state.clone();
                        self.best_state.best_score = self.best_yet;

                        let elapsed = self.start_time.elapsed().as_secs_f64();
                        println!("NMCS best score yet : {} after {}", best_state_score, elapsed);
                        if verbose {
                            let new_name = self.registerName.clone() + &*"_evolution".to_string();
                            writeLine("Conjecture ".to_owned() + &*best_state.conj.to_string()
                                          + " | NMCS best score yet : " + &*best_state_score.to_string()
                                          + " after " + &*elapsed.to_string()
                                          + "s, " + &*best_state.n_sommet.to_string()
                                          + " vertices\n", new_name);
                                          }
                    }

                    if new_st_score > 0.0001 {
                        let elapsed = self.start_time.elapsed().as_secs_f64();
                        if verbose {
                            writeLine("Conjecture ".to_owned() + &*best_state.conj.to_string()
                                          + "\n        Counterexample found in " + &*elapsed.to_string()
                                          + "s: best score = " + &*new_st_score.to_string()
                                          + "\n        With NMCS level " + &*n.to_string()
                                          + ", " + &*best_state.n_sommet.to_string()
                                          + " vertices \n\n", self.registerName.clone());

                        }
                        println!("Conjecture {}\n   Counter-example found with NMCS level {} after {}s\n\n", best_state.conj, n, elapsed);

                        graphToDot::adj_matrix_to_dot(best_state.adj_mat.clone(), &*format!("{}/conj{}", self.registerName, best_state.conj));
                        saveMatrix::save_matrix(&*format!("{}/conj{}", self.registerName, best_state.conj), best_state.adj_mat.clone());

                        return self.best_state.clone()
                    }
                }
            }

            if State::CONSIDER_NON_TERM {
                if best_state.seq.len() == st.seq.len() {
                    break
                }
            }

            st.play(best_state.seq[st.seq.len()]);
        }

        if State::CONSIDER_NON_TERM{
            return self.best_state.clone()
        }

        return st
    }
}

pub fn launch_nmcs(init_st: State, level: i8, heuristic_w: f64, verbose: bool, timeout: f64, registerName: String) -> State {
    let mut expe = NMCS::new();
    expe.timeout = timeout;
    expe.registerName = registerName;

    let st = expe.nmcs(init_st, level, heuristic_w, verbose);

    return st;
}
