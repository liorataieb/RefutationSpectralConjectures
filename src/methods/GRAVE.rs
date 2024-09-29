use std::collections::HashMap;
use crate::tools::calc::softmaxChoice;
use std::time::Instant;
use crate::tools::resultSaver::writeLine;
use crate::models::conjectures::conjectures_wagner_1::{State, Move};
use crate::tools::{graphToDot, saveMatrix};

#[derive(Clone)]
pub struct transEntry{
    pub wins: HashMap<Move, f64>,
    pub playouts: HashMap<Move, i32>,
    pub winsAMAF: HashMap<Move, f64>,
    pub playoutsAMAF: HashMap<Move, i32>,
    pub allplayouts: i32
}

pub struct GRAVE{
    pub start_time: Instant,
    pub transTable: HashMap<Vec<Move>, transEntry>,
    pub REF: i32,
    pub best_score_yet: f64,
    pub registerName: String,
    pub timeout: f64,
    pub best_state: State,

}

impl GRAVE{
    pub fn new() -> Self {
        Self{
            start_time: Instant::now(),
            transTable: HashMap::new(),
            REF: -1,
            best_score_yet: f64::NEG_INFINITY,
            timeout: -1.0,
            registerName: String::new(),
            best_state: State::new()
        }
    }

    pub fn playout(&mut self, mut st: State, heuristic_w : f64) -> State {
        let mut best_state: State = st.clone();
        let mut best_state_score = best_state.score();

        while !st.terminal() {
            let mut heuristic_possible = true;
            let moves = st.legal_moves();

            if moves.len() == 0 {
                return st
            }
            let mut i = ((moves.len() as f64)*rand::random::<f64>()) as usize;

            if heuristic_w != 0.0 {
                let mut weights = Vec::new();
                for &m in &moves{
                    let v = heuristic_w*st.heuristic(m);
                    if v.is_nan() {
                        heuristic_possible = false
                    }
                    weights.push(v);
                }

                if heuristic_possible {
                    i = softmaxChoice(weights);
                }
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

        if State::CONSIDER_NON_TERM {
            return best_state
        }

        return st
    }

    pub fn GRAVE(&mut self, mut st: State, initref: transEntry, heuristic_w: f64, playout_heuristic_w: f64, verbose: bool) -> (f64, State) {
        let mut tref = initref.clone();

        let moves = st.legal_moves();
        let mut res = f64::NEG_INFINITY;
        if st.terminal() || moves.len() == 0 {
            let res = st.score();
            if res > self.best_score_yet {
                self.best_score_yet = res;
                self.best_state = st.clone();
                self.best_state.best_score = self.best_score_yet;

                let elapsed = self.start_time.elapsed().as_secs_f64();
                println!("GRAVE best score yet : {} after {}", res, elapsed);
                if verbose {
                    let new_name = self.registerName.clone() + &*"_evolution".to_string();
                    writeLine("Conjecture ".to_owned() + &*st.conj.to_string()
                                  + " | GRAVE best score yet : " + &*res.to_string()
                                  + " after " + &*elapsed.to_string()
                                  + "s, " + &*self.best_state.n_sommet.to_string()
                                  + " vertices\n", new_name);
                                  }

                if res > 0.0001 {
                    let elapsed = self.start_time.elapsed().as_secs_f64();
                    if verbose {
                        writeLine("Conjecture ".to_owned() + &*st.conj.to_string()
                                      + "\n        Counterexample found in " + &*elapsed.to_string()
                                      + "s: best score = " + &*res.to_string()
                                      + "\n        With GRAVE "
                                      + ", " + &*self.best_state.n_sommet.to_string()
                                      + " vertices\n\n", self.registerName.clone());
                                      }
                    println!("Conjecture {}\n   Counter-example found with GRAVE after {}s\n\n", st.conj, elapsed);

                    graphToDot::adj_matrix_to_dot(st.adj_mat.clone(), &*format!("{}/conj{}", self.registerName, st.conj));
                    saveMatrix::save_matrix(&*format!("{}/conj{}", self.registerName, st.conj), st.adj_mat.clone());
                    }
            }

            return (res, st);
        }

        if self.transTable.contains_key(&st.seq) {
            let mut new_st = st.clone();
            let t = self.transTable.get(&st.seq).unwrap().clone();
            if t.allplayouts > self.REF {
                tref = t.clone();
            }

            let mut best_value = f64::NEG_INFINITY;
            let mut best_move = moves[0].clone();
            for m in moves{

                let mut mean = 0.0;
                let mut p = 0.0;
                let mut w = 0.0;
                if t.wins.get(&m).is_some() {
                    w = *t.wins.get(&m).unwrap();
                    p = *t.playouts.get(&m).unwrap() as f64;
                    mean = w/p;
                }

                let mut value = 1000000000000.0;

                if tref.winsAMAF.get(&m).is_some() {
                    let wa = *tref.winsAMAF.get(&m).unwrap();
                    let pa = *tref.playoutsAMAF.get(&m).unwrap() as f64;
                    let mut Bm = pa/(pa + p);
                    if heuristic_w != 0.0 {
                        Bm = pa/(pa + p + st.heuristic(m)*pa*p);
                    }

                    let AMAF = wa/pa;
                    value = (1.0 - Bm) * mean + Bm*AMAF;
                }

                if value > best_value {
                    best_value = value;
                    best_move = m.clone();
                }
            }

            new_st.play(best_move);
            let (res, resState) = self.GRAVE(new_st.clone(), tref, heuristic_w, playout_heuristic_w, verbose);

            //update transtable[board] with res

            let mut n : &i32 = &0;
            let mut f : &f64 = &0.0;
            let m = best_move.clone();


            let mut w = t.wins.clone();
            if w.contains_key(&m) {
                f = w.get(&m).unwrap();
            }
            w.insert(m, f+ res);

            n= &0;
            let mut p = t.playouts.clone();
            if p.contains_key(&m) {
                n = p.get(&m).unwrap();
            }
            p.insert(m, n+1);


            let mut wa = t.winsAMAF.clone();
            let mut pa = t.playoutsAMAF.clone();
            for i in st.seq.len()..resState.seq.len() {
                let m = resState.seq[i];

                f= &0.0;
                if wa.contains_key(&m) {
                    f = wa.get(&m).unwrap();
                }
                wa.insert(m, f + res);

                n= &0;
                if pa.contains_key(&m) {
                    n = pa.get(&m).unwrap();
                }
                pa.insert(m, n + 1);
            }

            let entry = transEntry{wins : w, playouts : p, winsAMAF : wa , playoutsAMAF : pa, allplayouts : t.allplayouts +1};
            self.transTable.insert(st.seq, entry);
            return (res, resState);

        } else {
            let mut new_st = st.clone();
            let moves = new_st.legal_moves();
            let m = moves[((moves.len() as f64)*rand::random::<f64>()) as usize];
            new_st.play(m);
            let mut pl = self.playout(new_st, playout_heuristic_w);
            res = pl.score();

            if res > self.best_score_yet {
                self.best_score_yet = res;
                self.best_state = pl.clone();
                self.best_state.best_score = self.best_score_yet;

                let elapsed = self.start_time.elapsed().as_secs_f64();
                println!("GRAVE best score yet : {} after {}", res, elapsed);
                if verbose {
                    let new_name = self.registerName.clone() + &*"_evolution".to_string();
                    writeLine("Conjecture ".to_owned() + &*st.conj.to_string()
                                  + " | GRAVE best score yet : " + &*res.to_string()
                                  + " after " + &*elapsed.to_string()
                                  + "s, " + &*self.best_state.n_sommet.to_string()
                                  + " vertices\n", new_name);
                                  }

                if res > 0.0001 {
                    let elapsed = self.start_time.elapsed().as_secs_f64();
                    if verbose {
                        writeLine("Conjecture ".to_owned() + &*st.conj.to_string()
                                      + "\n        Counterexample found in " + &*elapsed.to_string()
                                      + "s: best score = " + &*res.to_string()
                                      + "\n        With GRAVE "
                                      + ", " + &*self.best_state.n_sommet.to_string()
                                      + " vertices\n\n", self.registerName.clone());
                                      }
                    println!("Conjecture {}\n   Counter-example found with GRAVE after {}s\n\n", st.conj, elapsed);

                    graphToDot::adj_matrix_to_dot(pl.adj_mat.clone(), &*format!("{}/conj{}", self.registerName, st.conj));
                    saveMatrix::save_matrix(&*format!("{}/conj{}", self.registerName, st.conj), pl.adj_mat.clone());
                }
            }

            let mut w = HashMap::new();
            w.insert(m, res);

            let mut p = HashMap::new();
            p.insert(m, 1);

            let mut wa = tref.winsAMAF.clone();
            let mut pa = tref.playoutsAMAF.clone();
            for i in st.seq.len()..pl.seq.len() {
                let m = pl.seq[i];

                let mut f : &f64 = &0.0;
                if wa.contains_key(&m) {
                    f = wa.get(&m).unwrap();
                }
                wa.insert(m, f + res);

                let mut n : &i32 = &0;
                if pa.contains_key(&m) {
                    n = pa.get(&m).unwrap();
                }
                pa.insert(m, n + 1);
            }

            let entry = transEntry{wins : w, playouts : p, winsAMAF : wa , playoutsAMAF : pa, allplayouts : 1};
            self.transTable.insert(st.seq, entry);

            return (res, pl);
        }
    }
}

pub fn launch_grave(inist: State, rf: i32, heuristic_w: f64, playout_heuristic_w: f64, timeout: f64, verbose: bool, registerName: String) -> State {
    let mut expe = GRAVE::new();
    expe.timeout = timeout;
    expe.registerName = registerName;
    expe.REF = rf;

    let tref = transEntry{wins : HashMap::new(), playouts : HashMap::new(), winsAMAF : HashMap::new(), playoutsAMAF : HashMap::new(), allplayouts : 0};
    let (vic, st) = expe.GRAVE(inist.clone(), tref, heuristic_w, playout_heuristic_w, verbose);

    if st.score() > 0.0001 {
        return st
    }

    while expe.start_time.elapsed().as_secs_f64() < timeout {
        let mut tref2 = transEntry{wins : HashMap::new(), playouts : HashMap::new(), winsAMAF : HashMap::new(), playoutsAMAF : HashMap::new(), allplayouts : 0};
        let (vic, st) = expe.GRAVE(inist.clone(), tref2, heuristic_w, playout_heuristic_w, verbose);

        if st.score() > 0.0001 {
            return st
        }
    }

    return st;
}
