use crate::models::conjectures::conjectures_wagner_1::{State};
use crate::tools::calc::softmaxChoice;
use std::time::Instant;
use crate::tools::resultSaver::writeLine;
use std::collections::HashMap;
use crate::tools::{graphToDot, saveMatrix};

static SKIP_REPEATING_SCORES : bool = false;

#[derive(Clone)]
pub struct WS{ // Weight-State
    pub w: f64,
    pub s: State
}


pub fn insertDicho(l : &Vec<WS>, node : &WS) -> usize{
    let mut i = l.len()/2;
    let mut mi = 0;
    let mut ma = l.len()-1;

    while (i != 0 && l[i-1].w > node.w) || l[i].w < node.w {
        if l[i].w == node.w {
            return i;
        }
        if l[i].w < node.w {
            mi = i + 1;
            if mi > ma{
                return mi;
            }
        } else {
            ma = i;
        }
        i = (mi as f64/2.0 + (ma as f64)/2.0) as usize
    }

    return i;
}

pub fn playout(mut st: State, heuristic_w : f64) -> State {
    let mut best_state: State = st.clone();
    let mut best_state_score = best_state.score();

    while !st.terminal() {
        let moves = st.legal_moves();
        if moves.len() == 0 {
            return st
        }

        let mut i = ((moves.len() as f64)*rand::random::<f64>()) as usize;
        if heuristic_w != 0.0 {
            let mut weights = Vec::new();
            for &m in &moves{
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
                best_state.best_score = sc;
            }
        }
    }

    if State::CONSIDER_NON_TERM {
        return best_state
    }

    return st
}

pub fn BFS(inist: State, heuristic_w: f64, p:i32, timeout: f64, verbose: bool, registerName: String) -> State {
    if p < 0 {
        //println!("attention, utilisation des scores des état non finaux au lieu des scores de playouts pour déterminer la valeur d'un noeud")
    }

    let mut st = inist.clone();
    let mut start_time = Instant::now();

    let mut open_nodes = Vec::new();
    open_nodes.push(WS{w : 0.0, s : st.clone()});

    let mut best_score_yet = f64::NEG_INFINITY;
    let mut best_state_yet = st.clone();
    let mut opened_nodes = 0;

    let mut visitedScores: HashMap<i64, bool> = HashMap::new();

    while open_nodes.len() != 0 {
        //println!("open nodes : {}", open_nodes.len());

        opened_nodes += 1;
        if start_time.elapsed().as_secs_f64() > timeout && timeout > 0.0 {
            return best_state_yet
        }

        let mut node = open_nodes.pop().unwrap();
        //println!("node opened with score : {}", node.w);
        //println!("{}", node.s.adj_mat);

        if SKIP_REPEATING_SCORES {
            while visitedScores.contains_key(&((node.w*10000000000.0) as i64)) && node.w != 0.0 && open_nodes.len() != 0 {
                node = open_nodes.pop().unwrap();
            }
            visitedScores.insert(((node.w*10000000000.0) as i64), true);
        }

        for m in node.s.legal_moves() {
            let mut new_state = node.s.clone();
            new_state.play(m);

            if p >= 0 {
                let mut best_playout_state = playout(new_state.clone(), heuristic_w);
                let mut best_playout_state_score = best_playout_state.score();

                for _ in 0..p {
                    let mut playout_state = playout(new_state.clone(), heuristic_w);
                    let playout_state_score = playout_state.score();

                    if playout_state_score > best_playout_state_score {
                        best_playout_state = playout_state.clone();
                        best_playout_state_score = playout_state_score;
                    }
                }

                if best_playout_state_score > best_score_yet {
                    best_score_yet = best_playout_state_score;
                    best_state_yet = best_playout_state.clone();

                    println!("BFS best score yet : {} after {}", best_score_yet, start_time.elapsed().as_secs_f64());
                    if verbose {
                        let new_name = registerName.clone() + &*"_evolution".to_string();
                        writeLine("Conjecture ".to_owned() + &*best_state_yet.conj.to_string()
                                      + " | BFS best score yet : " + &*best_score_yet.to_string()
                                      + " after " + &*start_time.elapsed().as_secs_f64().to_string()
                                      + "s, " + &*best_state_yet.n_sommet.to_string()
                                      + " vertices\n", new_name);
                                      }

                    if best_playout_state_score > 0.0001 {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        if verbose {
                            writeLine("Conjecture ".to_owned() + &*inist.conj.to_string()
                                          + "\n        Counterexample found in " + &*elapsed.to_string()
                                          + "s: best score = " + &*best_playout_state_score.to_string()
                                          + "\n        With BFS with playout"
                                          + ", " + &*best_state_yet.n_sommet.to_string()
                                          + " vertices\n\n", registerName.clone());
                                          }
                        println!("Conjecture {}\n   Counter-example found with BFS after {}s\n\n", inist.conj, elapsed);

                        graphToDot::adj_matrix_to_dot(best_state_yet.adj_mat.clone(), &*format!("{}/conj{}", registerName, best_state_yet.conj));
                        saveMatrix::save_matrix(&*format!("{}/conj{}", registerName, best_state_yet.conj), best_state_yet.adj_mat.clone());

                        return best_state_yet
                    }
                }

                let new_ws = WS{w : best_playout_state_score, s : new_state};
                let mut i = 0;
                if open_nodes.len() != 0 {
                    i = insertDicho(&open_nodes, &new_ws);
                }

                open_nodes.insert(i, new_ws);

            } else {
                let sc = new_state.score();
                if sc > best_score_yet {
                    best_score_yet = sc;
                    best_state_yet = new_state.clone();

                    println!("BFS best score yet : {} after {}", best_score_yet, start_time.elapsed().as_secs_f64());
                    if verbose {
                        let new_name = registerName.clone() + &*"_evolution".to_string();
                        writeLine("Conjecture ".to_owned() + &*best_state_yet.conj.to_string()
                                      + " | BFS best score yet : " + &*best_score_yet.to_string()
                                      + " after " + &*start_time.elapsed().as_secs_f64().to_string()
                                      + "s, " + &*best_state_yet.n_sommet.to_string()
                                      + " vertices\n", new_name);
                                      }

                    if sc > 0.0001 {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        if verbose {
                            writeLine("Conjecture ".to_owned() + &*inist.conj.to_string()
                                          + "\n        Counterexample found in " + &*elapsed.to_string()
                                          + "s: best score = " + &*sc.to_string()
                                          + "\n        With BFS"
                                          + ", " + &*best_state_yet.n_sommet.to_string()
                                          + " vertices\n\n", registerName.clone());
                                          }
                        println!("Conjecture {}\n   Counter-example found with BFS after {}s\n\n", inist.conj, elapsed);

                        graphToDot::adj_matrix_to_dot(best_state_yet.adj_mat.clone(), &*format!("{}/conj{}", registerName, best_state_yet.conj));
                        saveMatrix::save_matrix(&*format!("{}/conj{}", registerName, best_state_yet.conj), best_state_yet.adj_mat.clone());

                        return best_state_yet
                    }
                }
                if !new_state.terminal() {
                    //println!("{}", new_state.n_sommet);
                    //println!("{}, {}", m.from, m.to);
                    let new_ws = WS{w: sc, s: new_state};

                    let mut i = 0;
                    if open_nodes.len() != 0 {
                        i = insertDicho(&open_nodes, &new_ws);
                    }
                    open_nodes.insert(i, new_ws);
                }
            }
        }
    }

    //println!(" exhausted all nodes : {}", opened_nodes);
    return best_state_yet
}

pub fn launch_bfs(init_stat: State, heuristic_w: f64, p:i32, timeout: f64, verbose: bool, registerName: String) -> State {
    return BFS(init_stat, heuristic_w, p, timeout, verbose, registerName);
}
