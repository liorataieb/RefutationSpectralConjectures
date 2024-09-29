use std::time::Instant;
use rand::prelude::SliceRandom;
use rand::{Rng, thread_rng};
use crate::models::conjectures::conjectures_wagner_1::{Move, State};
use crate::tools::{graphToDot, saveMatrix};
use crate::tools::resultSaver::writeLine;

pub fn create_random_regular_graph(n: usize, d: usize, fct: usize) -> State {
    let mut st = State::new();
    st.n_sommet = n;
    st.adj_mat.resize_mut(st.n_sommet, st.n_sommet, 0.0);
    st.conj = fct;

    let mut moves = vec![];
    let mut degrees = vec![];
    for i in 0..st.n_sommet {
        degrees.push(0);
        for j in (i+1)..st.n_sommet {
            moves.push((i, j));
        }
    }

    let mut rng = thread_rng();
    moves.shuffle(&mut rng);

    for m in moves {
        if degrees[m.0] <= d && degrees[m.1] <= d {
            st.adj_mat[(m.0, m.1)] = 1.0;
            st.adj_mat[(m.1, m.0)] = 1.0;
            degrees[m.0] += 1;
            degrees[m.1] += 1;
        }
    }

    let sc = st.score();
    st.best_score = sc;

    st
}

pub fn create_random_graph(n: usize, fct: usize) -> State {
    let mut rng = thread_rng();

    let mut st = State::new();
    st.n_sommet = n;
    st.conj = fct;

    st.adj_mat.resize_mut(st.n_sommet, st.n_sommet, 0.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let value = if rng.gen::<f64>() < 0.5 { 0.0 } else { 1.0 };
            st.adj_mat[(i, j)] = value;
            st.adj_mat[(j, i)] = value;
        }
    }

    let sc = st.score();
    st.best_score = sc;

    st
}

pub fn create_complete_graph(n: usize, fct: usize) -> State {
    let mut st = State::new();
    st.n_sommet = n;
    st.conj = fct;

    st.adj_mat.resize_mut(st.n_sommet, st.n_sommet, 0.0);
    for i in 0..n {
        for j in (i + 1)..n {
            st.adj_mat[(i, j)] = 1.0;
            st.adj_mat[(j, i)] = 1.0;
        }
    }

    let sc = st.score();
    st.best_score = sc;

    st
}

pub fn local_search(st: State) -> State {
    let mut rng = thread_rng();

    let mut st_clone = st.clone();
    let mut sc = st_clone.score();

    let mut possible_moves = Vec::new();
    for i in 0..st.n_sommet {
        for j in (i+1)..st.n_sommet {
            possible_moves.push((i, j));
        }
    }

    let mut possible_improvement = true;
    while possible_improvement {
        possible_moves.shuffle(&mut rng);

        possible_improvement = false;
        for m in &possible_moves {
            let mv = Move{ind: st.n_sommet, from: m.0, to: m.1 as i64};
            let mut new_st = st.clone();
            new_st.play(mv);

            let new_st_score = new_st.score();

            if new_st_score > sc {
                sc = new_st_score;
                st_clone = new_st.clone();
                st_clone.best_score = sc;

                possible_improvement = true;
                break;
            }
        }
    }

    return st_clone
}

pub fn perturbation(st: State) -> State {
    let mut rng = thread_rng();

    let mut pertubated_state = st.clone();

    let mut moves = vec![];
    for i in 0..st.n_sommet {
        for j in (i+1)..st.n_sommet {
            moves.push((i, j));
        }
    }
    moves.shuffle(&mut rng);

    let mut m1 = moves[0].0;
    let mut m2 = moves[0].1;
    let mut edge = 1.0;
    if pertubated_state.adj_mat[(m1, m2)] == 1.0 {
        edge = 0.0;
    }
    pertubated_state.adj_mat[(m1, m2)] == edge;
    pertubated_state.adj_mat[(m2, m1)] == edge;

    let mut i = 2;
    for &m in &moves[2..] {
        m1 = m.0;
        m2 = m.1;
        if pertubated_state.adj_mat[(m1, m2)] == edge {
            let mut second_edge = 0.0;
            if second_edge == edge {
                second_edge = 1.0;
            }
            pertubated_state.adj_mat[(m1, m2)] == second_edge;
            pertubated_state.adj_mat[(m2, m1)] == second_edge;
            break
        }

        i += 1;
    }

    /*let mut m1 = moves[moves.len()-1].0;
    let mut m2 = moves[moves.len()-1].1;
    let mut third_edge = 1.0;
    if pertubated_state.adj_mat[(m1, m2)] == 1.0 {
        third_edge = 0.0;
    }
    pertubated_state.adj_mat[(m1, m2)] == third_edge;
    pertubated_state.adj_mat[(m2, m1)] == third_edge;

    for &m in moves.iter().rev() {
        m1 = m.0;
        m2 = m.1;
        if pertubated_state.adj_mat[(m1, m2)] == edge {
            let mut fourth_edge = 0.0;
            if fourth_edge == edge {
                fourth_edge = 1.0;
            }
            pertubated_state.adj_mat[(m1, m2)] == fourth_edge;
            pertubated_state.adj_mat[(m2, m1)] == fourth_edge;
            break
        }
    }*/

    let sc = pertubated_state.score();
    pertubated_state.best_score = sc;

    pertubated_state
}

pub fn iterative_local_search(n: usize, d: usize, fct: usize, timeout: f64, verbose: bool, registerName: String) -> State {
    let start_time = Instant::now();

    //let mut st = create_random_regular_graph(n, d, fct);
    let mut st = create_random_graph(n, fct);
    let mut best_state = st.clone();
    let mut best_score = best_state.best_score;
    println!("First best_score {}", best_score);

    st = local_search(st);

    while st.best_score <= 0.0001 {
        if start_time.elapsed().as_secs_f64() > timeout && timeout > 0.0 {
            return best_state
        }

        let mut new_st = perturbation(st.clone());
        //st = perturbation(st, fct, &mut rng); #RW

        new_st = local_search(new_st);
        //st = local_search(st, fct, &mut rng); #RW
        //if st.best_score > best_score { #RW
        if new_st.best_score > best_score {
            //best_score = st.best_score; #RW
            best_score = new_st.best_score;
            //best_state = st.clone(); #RW
            best_state = new_st.clone();
            best_state.best_score = best_score;

            println!("ILS best score yet : {} after {}", best_score, start_time.elapsed().as_secs_f64());
            if verbose {
                let new_name = registerName.clone() + &*"_evolution".to_string();
                writeLine("Conjecture ".to_owned() + &*fct.to_string()
                              + " | ILS best score yet : " + &*best_score.to_string()
                              + " after " + &*start_time.elapsed().as_secs_f64().to_string()
                              + "s, " + &*best_state.n_sommet.to_string()
                              + " vertices\n", new_name);
                              }

            if new_st.best_score > 0.0001 {
                let elapsed = start_time.elapsed().as_secs_f64();
                if verbose {
                    writeLine("Conjecture ".to_owned() + &*st.conj.to_string()
                                  + "\n        Counterexample found in " + &*elapsed.to_string()
                                  + "s: best score = " + &*new_st.best_score.to_string()
                                  + "\n        With ILS"
                                  + ", " + &*best_state.n_sommet.to_string()
                                  + " vertices\n\n", registerName.clone());
                                  }
                println!("Conjecture {}\n   Counter-example found with ILS after {}s\n", st.conj, elapsed);

                graphToDot::adj_matrix_to_dot(new_st.adj_mat.clone(), &*format!("{}/conj{}", registerName, st.conj));
                saveMatrix::save_matrix(&*format!("{}/conj{}", registerName, st.conj), new_st.adj_mat.clone());

                return best_state
            }
        }

        if new_st.best_score > st.best_score {
            st = new_st.clone();
        }
    }

    st
}