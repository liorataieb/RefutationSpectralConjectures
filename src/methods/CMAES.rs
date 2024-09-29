use std::time::Instant;
use nalgebra::{DMatrix, DVector, Dynamic, OMatrix, OVector, SVD};
use rand::distributions::Distribution;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rand_distr::StandardNormal;
use stat::covariance;
use crate::models::conjectures::conjectures_wagner_1::State;
use crate::tools::{graphToDot, saveMatrix};
use crate::tools::resultSaver::writeLine;

pub fn encode(adj_mat: &DMatrix<f64>) -> Vec<f64> {
    let mut new_matrix = adj_mat.clone();
    for i in 0..adj_mat.nrows(){
        for j in 0.. adj_mat.nrows(){
            if new_matrix[(i, j)] == -1.0 {
                new_matrix[(i, j)] = 0.0;
            }
        }
    }

    let encoded_vec: Vec<f64> = new_matrix.as_slice().to_vec();

    encoded_vec
}

pub fn assemble_encodings(states: Vec<State>) -> Vec<Vec<f64>> {
    let mut ensemble: Vec<Vec<f64>> = vec![];
    for st in &states {
        let encoding = encode(&st.adj_mat);
        ensemble.push(encoding);
    }

    ensemble
}

pub fn ensemble_to_DMatrix(ensemble: &Vec<Vec<f64>>) -> DMatrix<f64> {
    let mut mat:Vec<f64> = vec![];
    for v in ensemble {
        mat.extend(v);
    }

    let matrix = DMatrix::from_row_slice(ensemble.len(), ensemble[0].len(), &mat);

    matrix
}

pub fn gaussian_mean(matrix: &DMatrix<f64>) -> Vec<f64> {
    let mut mean = vec![];
    for i in 0..matrix.ncols() {
        let mut sum = 0.0;
        for j in 0..matrix.nrows() {
            sum += matrix[(j, i)];
        }

        mean.push(sum);
    }

    let denom = matrix.ncols() as f64;
    mean = mean.iter().map(|x| *x/denom).collect();

    mean
}

pub fn gaussian_covariance(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let mut cov_mat = DMatrix::zeros(matrix.ncols(), matrix.ncols());

    for i in 0..matrix.ncols(){
        for j in i..matrix.ncols() {
            let mut v1 = vec![];
            let mut v2 = vec![];
            for k in 0..matrix.nrows() {
                v1.push(matrix[(k, i)]);
                v2.push(matrix[(k, j)]);
            }
            let covariance = covariance(&v1, &v2);
            cov_mat[(i, j)] = covariance;
        }
    }

    for i in 1..matrix.ncols() {
        for j in 0..i {
            cov_mat[(i, j)] = cov_mat[(j, i)];
        }
    }

    cov_mat
}

pub fn svd(cov_mat: DMatrix<f64>) -> (OMatrix<f64, Dynamic, Dynamic>, OVector<f64, Dynamic>) {
    let svd = SVD::new(cov_mat.clone(), true, false);
    let P = svd.u.unwrap();
    let delta = svd.singular_values;

    (P, delta)
}

pub fn construct_state(child: Vec<(&f64, &(usize, usize))>, size: usize) -> State {
    let mut new_state = State::new();
    new_state.n_sommet = size;
    new_state.adj_mat.resize_mut(size, size, 0.0);
    for &child in &child {
        let i = child.1.0;
        let j = child.1.1;
        let value = *child.0;
        if i != j && value == 1.0 {
            new_state.adj_mat[(i, j)] = 1.0;
            new_state.adj_mat[(j, i)] = 1.0;
            new_state.n_arete += 1;
        }
    }

    new_state
}

pub fn create_child(P: &OMatrix<f64, Dynamic, Dynamic>, delta: &OVector<f64, Dynamic>, mean: &Vec<f64>) -> State {
    let mut rng = thread_rng();
    let dist = StandardNormal;

    let mut child = vec![];
    let size = delta.len();
    for i in 0..size {
        let mut ele: f64 = dist.sample(&mut rng);
        ele = ele*(delta[i].sqrt());
        child.push(ele);
    }
    let child = DVector::from_vec(child);
    let coef = 1.0;
    let child: Vec<_> = (coef*P*child).as_slice().to_vec().iter().zip(mean.iter()).map(|(&a, &b)| a + b).collect();

    let mut binary_child: Vec<f64> = vec![];
    for k in 0..child.len() {
        if child[k] > mean[k] {
            binary_child.push(1.0);
        } else {
            binary_child.push(0.0);
        }
    }

    let mut coordinates = vec![];
    for i in 0..(f64::sqrt(child.len() as f64) as usize) {
        for j in 0..(f64::sqrt(child.len() as f64) as usize) {
            coordinates.push((i,j));
        }
    }

    let mut full_child: Vec<_> = binary_child.iter().zip(coordinates.iter()).collect();
    //let mut full_child: Vec<_> = child.iter().zip(coordinates.iter()).collect();
    full_child.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let new_state = construct_state(full_child, f64::sqrt(size as f64) as usize);

    new_state
}

pub fn select_candidates(mut candidates: Vec<State>, to_keep: usize) -> Vec<State> {
    candidates.sort_by(|(mut a), (mut  b)| b.clone().score().partial_cmp(&a.clone().score()).unwrap());
    let best_states: Vec<State> = candidates.into_iter().take(to_keep).collect();

    best_states
}

pub fn create_random_edges(n: usize) -> Vec<(f64, (usize, usize))> {
    let mut list = vec![];
    for i in 0..n {
        for j in (i+1)..n {
            if i != j {
                list.push((0.0, (i, j)));
            }
        }
    }

    let mut rng = thread_rng();
    list.shuffle(&mut rng);

    list
}

pub fn add_parents_small_curri(lambda: usize, n: usize, mut states: Vec<State> ) -> Vec<State> {
    let to_add = lambda - states.len();
    for i in 0..to_add {
        let list = create_random_edges(n);
        let vec: Vec<(&f64, &(usize, usize))>  = list.iter().map(|(x, y)| (x, y)).collect();
        let st = construct_state(vec, n);
        states.push(st);
    }

    states
}

pub fn launch_CMAES(lambda: usize, restart: i32, fct: usize, size_terminal: usize, verbose: bool, registerName: String) -> State {
    let mut best_state = State::new();
    best_state.conj = fct;
    let mut best_score = best_state.score();

    let start_time = Instant::now();

    let mut parents: Vec<State> = vec![];
    parents = add_parents_small_curri(lambda, size_terminal, parents);
    let mut num_reach = 0;

    while num_reach < restart {
        let ensemble: Vec<Vec<f64>> = assemble_encodings(parents.clone());
        let parents_matrix: DMatrix<f64> = ensemble_to_DMatrix(&ensemble);

        let mean = gaussian_mean(&parents_matrix);
        let covariance = gaussian_covariance(&parents_matrix);
        let (P, delta) = svd(covariance);

        let mut children: Vec<State> = vec![];
        for _ in 0..lambda {
            let child = create_child(&P, &delta, &mean);
            children.push(child);
        }

        for st in &children {
            let new_st = st.clone();
            let new_st_score = new_st.score();

            if new_st_score > best_score {
                best_state = new_st.clone();
                best_score = new_st_score;
                best_state.best_score = new_st_score;

                let elapsed = start_time.elapsed().as_secs_f64();
                println!("CMAES best score yet : {} after {}", new_st_score, elapsed);
                if verbose {
                    let new_name = registerName.clone() + &*"_evolution".to_string();
                    writeLine("Conjecture ".to_owned() + &*fct.to_string()
                                  + " | CMAES best score yet : " + &*new_st_score.to_string()
                                  + " after " + &*elapsed.to_string()
                                  + "s, " + &*best_state.n_sommet.to_string()
                                  + " vertices\n", new_name);
                                  }

                if new_st_score > 0.0001 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    if verbose {
                        writeLine("Conjecture ".to_owned() + &*fct.to_string()
                                      + "\n        Counterexample found in " + &*elapsed.to_string()
                                      + "s: best score = " + &*new_st_score.to_string()
                                      + "\n        With CMAES restart" + &*restart.to_string()
                                      + ", " + &*best_state.n_sommet.to_string()
                                      + "\n\n", registerName.clone());
                                      }
                    println!("Conjecture {}\n   Counter-example found with CMAES restart {} after {}s\n\n", fct, restart, elapsed);

                    graphToDot::adj_matrix_to_dot(new_st.adj_mat.clone(), &*format!("{}/conj{}", registerName, best_state.conj));
                    saveMatrix::save_matrix(&*format!("{}/conj{}", registerName, best_state.conj), new_st.adj_mat.clone());

                    return best_state
                }
            }
        }

        parents.extend(children);
        parents = select_candidates(parents.clone(), lambda);

        num_reach += 1;
    }

    return best_state
}