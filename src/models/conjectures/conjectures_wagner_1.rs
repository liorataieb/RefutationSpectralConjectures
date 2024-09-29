extern crate nalgebra;

use nalgebra::{DMatrix, DVector};
use nalgebra::linalg::SymmetricEigen;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct Move{
    pub ind : usize,
    pub from : usize,
    pub to: i64
}
#[derive(Clone)]
pub struct State{
    pub adj_mat: DMatrix<f64>,
    pub n_arete: usize,
    pub n_sommet: usize,
    pub size_terminal: usize,
    pub best_score: f64,
    pub seq: Vec<Move>,
    pub conj: usize
}

impl State{
    pub const CONSIDER_NON_TERM: bool = true;

    pub fn new() -> Self {
        Self {
            adj_mat: DMatrix::from_diagonal_element(1, 1, 0.0),
            n_arete: 0,
            n_sommet: 1,
            size_terminal: 1,
            best_score: f64::NEG_INFINITY,
            seq : Vec::new(),
            conj: 1
        }
    }

    pub fn add_arete(&mut self, from : usize, to : i32) {
        if from as i32 != to && self.n_sommet > from  {
            let mut true_to : usize = 0;
            if  to >= self.n_sommet as i32 || to == -1 {
                true_to = self.n_sommet;
                self.n_sommet += 1;
                self.adj_mat.resize_mut(self.n_sommet, self.n_sommet, 0.0)
            } else {
                true_to = to as usize;
                if self.adj_mat[(from, true_to)] != 0.0 {
                    return;
                }
            }
            self.n_arete += 1;
            self.adj_mat[(from, true_to)] = 1.0;
            self.adj_mat[(true_to, from)] = 1.0;
        }
    }

    pub fn play(&mut self, m : Move) {
        self.add_arete(m.from, m.to as i32);
        self.seq.push(m);
    }

    pub fn legal_moves(& self) -> Vec<Move> {
        let mut vec :Vec<Move> = Vec::new();

        //tree mode
        /*for i in 0..self.n_sommet {
            let m1 = Move{ind: self.n_sommet, from : i, to : self.n_sommet as i64 };
            vec.push(m1);
        }*/

        //any graph mode
        if self.n_sommet < self.size_terminal {
            for i in 0..self.n_sommet {
                let m1 = Move {ind: self.n_sommet, from: i, to: -1};
                vec.push(m1);
            }
        }

        for i in 0..self.n_sommet {
            for j in (i+1)..self.n_sommet {
                if self.adj_mat[(i, j)] == 0.0 {
                    let m1 = Move{ind: self.n_sommet, from: i, to: j as i64};
                    vec.push(m1);
                }
            }
        }

        return vec;
    }

    pub fn degree_matrix(& self) -> DMatrix<f64> {
        let row_sums = self.adj_mat.row_iter().map(|row| row.sum());
        let degrees = DVector::from_iterator(self.adj_mat.nrows(), row_sums);
        let deg_mat = DMatrix::from_diagonal(&degrees);

        return deg_mat;
    }

    pub fn largest_eigenvalue_laplacian_matrix(& self) -> f64 {
        let deg_mat = self.degree_matrix();
        let lap_mat = deg_mat - self.adj_mat.clone();

        let mut eigen = SymmetricEigen::new(lap_mat.clone()).eigenvalues;
        let mut eigenvalues: Vec<_> = eigen.iter().collect();
        eigenvalues.sort_by(|a: &&f64, b| a.total_cmp(b));

        return *eigenvalues[eigenvalues.len() - 1];
    }

    pub fn average_degree_neighbors_vec(& self) -> Vec<f64> {
        let deg_mat = self.degree_matrix();

        let mut mi_vec = Vec::new();
        for i in 0..self.n_sommet {
            let mut mi = 1.0 / deg_mat[(i, i)];

            let mut sum_degree_neighbors = 0.0;
            for j in 0..self.n_sommet {
                sum_degree_neighbors += self.adj_mat[(i, j)]*deg_mat[(j, j)];
            }

            mi *= sum_degree_neighbors;
            mi_vec.push(mi);
        }
        return mi_vec
    }

    pub fn heuristic(&mut self, m : Move) -> f64{
        let actual_score = self.score();

        let mut cl = self.clone();
        cl.play(m);
        let new_score = cl.score();

        return new_score - actual_score
    }

    pub fn score(& self) -> f64 {
        let mu = self.largest_eigenvalue_laplacian_matrix();
        let deg_mat = self.degree_matrix();
        let avg_deg_neighbors_vec = self.average_degree_neighbors_vec();

        let mut results_vec = Vec::new();
        if self.conj == 1 {
            for i in 0..self.n_sommet {
                let mut partial_result = (4.0*deg_mat[(i, i)].powf(3.0)/avg_deg_neighbors_vec[i]).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 2 {
            for i in 0..self.n_sommet {
                let mut partial_result = 2.0*avg_deg_neighbors_vec[i].powf(2.0)/deg_mat[(i, i)];
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 3 {
            for i in 0..self.n_sommet {
                let mut partial_result =  avg_deg_neighbors_vec[i].powf(2.0)/deg_mat[(i, i)] + avg_deg_neighbors_vec[i];
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 4 {
            for i in 0..self.n_sommet {
                let mut partial_result =  2.0*deg_mat[(i, i)].powf(2.0)/avg_deg_neighbors_vec[i];
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 5 {
            for i in 0..self.n_sommet {
                let mut partial_result = deg_mat[(i, i)].powf(2.0)/avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[i];
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 6 {
            for i in 0..self.n_sommet {
                let mut partial_result = (3.0*deg_mat[(i, i)].powf(2.0) + avg_deg_neighbors_vec[i].powf(2.0)).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 7 {
            for i in 0..self.n_sommet {
                let mut partial_result = deg_mat[(i, i)].powf(2.0)/avg_deg_neighbors_vec[i] + deg_mat[(i, i)];
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 8 {
            for i in 0..self.n_sommet {
                let mut partial_result = (deg_mat[(i, i)]*(avg_deg_neighbors_vec[i] + 3.0*deg_mat[(i, i)])).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 9 {
            for i in 0..self.n_sommet {
                let mut partial_result = (avg_deg_neighbors_vec[i] + 3.0*deg_mat[(i, i)])/2.0;
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 10 {
            for i in 0..self.n_sommet {
                let mut partial_result = (deg_mat[(i, i)]*(3.0*avg_deg_neighbors_vec[i] + deg_mat[(i, i)])).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 11 {
            for i in 0..self.n_sommet {
                let mut partial_result = 2.0*avg_deg_neighbors_vec[i].powf(3.0)/deg_mat[(i, i)].powf(2.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 12 {
            for i in 0..self.n_sommet {
                let mut partial_result = (2.0*deg_mat[(i, i)].powf(2.0) + 2.0*avg_deg_neighbors_vec[i].powf(2.0)).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 13 {
            for i in 0..self.n_sommet {
                let mut partial_result = 2.0*avg_deg_neighbors_vec[i].powf(4.0)/deg_mat[(i, i)].powf(3.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 14 {
            for i in 0..self.n_sommet {
                let mut partial_result = 2.0*deg_mat[(i, i)].powf(3.0)/avg_deg_neighbors_vec[i].powf(2.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 15 {
            for i in 0..self.n_sommet {
                let mut partial_result = (4.0*avg_deg_neighbors_vec[i].powf(3.0)/deg_mat[(i, i)]).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 16 {
            for i in 0..self.n_sommet {
                let mut partial_result = 2.0*deg_mat[(i, i)].powf(4.0)/avg_deg_neighbors_vec[i].powf(3.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 17 {
            for i in 0..self.n_sommet {
                let mut partial_result = (5.0*deg_mat[(i, i)].powf(4.0) + 11.0*avg_deg_neighbors_vec[i].powf(4.0)).powf(1.0/4.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 18 {
            for i in 0..self.n_sommet {
                let mut partial_result = (2.0*deg_mat[(i, i)].powf(2.0) + 2.0*avg_deg_neighbors_vec[i].powf(3.0)/deg_mat[(i, i)]).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 19 {
            for i in 0..self.n_sommet {
                let mut partial_result = (4.0*deg_mat[(i, i)].powf(4.0) + 12.0*avg_deg_neighbors_vec[i].powf(3.0)*deg_mat[(i, i)]).powf(1.0/4.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 20 {
            for i in 0..self.n_sommet {
                let mut partial_result = (7.0*deg_mat[(i, i)].powf(2.0) + 9.0*avg_deg_neighbors_vec[i].powf(2.0)).sqrt()/2.0;
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 21 {
            for i in 0..self.n_sommet {
                let mut partial_result = (3.0*avg_deg_neighbors_vec[i].powf(2.0) + deg_mat[(i, i)].powf(3.0)/avg_deg_neighbors_vec[i]).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 22 {
            for i in 0..self.n_sommet {
                let mut partial_result = (2.0*deg_mat[(i, i)].powf(4.0) + 14.0*avg_deg_neighbors_vec[i].powf(2.0)*deg_mat[(i, i)].powf(2.0)).powf(1.0/4.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 23 {
            for i in 0..self.n_sommet {
                let mut partial_result = (deg_mat[(i, i)].powf(2.0) + 3.0*avg_deg_neighbors_vec[i]*deg_mat[(i, i)]).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 24 {
            for i in 0..self.n_sommet {
                let mut partial_result = (6.0*deg_mat[(i, i)].powf(4.0) + 10.0*avg_deg_neighbors_vec[i].powf(4.0)).powf(1.0/4.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 25 {
            for i in 0..self.n_sommet {
                let mut partial_result = (3.0*deg_mat[(i, i)].powf(4.0) + 13.0*avg_deg_neighbors_vec[i].powf(2.0)*deg_mat[(i, i)].powf(2.0)).powf(1.0/4.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 26 {
            for i in 0..self.n_sommet {
                let mut partial_result = (5.0*deg_mat[(i, i)].powf(2.0) + 11.0*avg_deg_neighbors_vec[i]*deg_mat[(i, i)]).sqrt()/2.0;
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 27 {
            for i in 0..self.n_sommet {
                let mut partial_result = ((3.0*deg_mat[(i, i)].powf(2.0) + 5.0*avg_deg_neighbors_vec[i]*deg_mat[(i, i)])/2.0).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 28 {
            for i in 0..self.n_sommet {
                let mut partial_result = (2.0*deg_mat[(i, i)]*avg_deg_neighbors_vec[i] + 2.0*avg_deg_neighbors_vec[i].powf(4.0)/deg_mat[(i, i)].powf(2.0)).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 29 {
            for i in 0..self.n_sommet {
                let mut partial_result = (avg_deg_neighbors_vec[i].powf(2.0) + 3.0*avg_deg_neighbors_vec[i].powf(3.0)/deg_mat[(i, i)]).sqrt();
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 30 {
            for i in 0..self.n_sommet {
                let mut partial_result = deg_mat[(i, i)].powf(2.0)/avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[i].powf(3.0)/deg_mat[(i, i)].powf(2.0);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 31 {
            for i in 0..self.n_sommet {
                let mut partial_result = 4.0*avg_deg_neighbors_vec[i].powf(2.0)/(deg_mat[(i, i)] + avg_deg_neighbors_vec[i]);
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 32 {
            for i in 0..self.n_sommet {
                let mut partial_result = (avg_deg_neighbors_vec[i].powf(3.0)*(3.0*deg_mat[(i, i)] + avg_deg_neighbors_vec[i])).sqrt()/deg_mat[(i, i)];
                if partial_result.is_nan() {
                    partial_result = 0.0;
                }
                results_vec.push(partial_result);
            }
        } else if self.conj == 33 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0 * (deg_mat[(i, i)] + deg_mat[(j, j)]) - (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 34 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0)) / (deg_mat[(i, i)] + deg_mat[(j, j)]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 35 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0)) / (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 36 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[j].powf(2.0)) / (deg_mat[(i, i)] + deg_mat[(j, j)]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 37 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = (2.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 38 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0 + (2.0 * (deg_mat[(i, i)] - 1.0).powf(2.0) + 2.0 * (deg_mat[(j, j)] - 1.0).powf(2.0)).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 39 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (2.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            - 4.0 * (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 40 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (2.0 * ((avg_deg_neighbors_vec[i] - 1.0).powf(2.0) + (avg_deg_neighbors_vec[j] - 1.0).powf(2.0))
                            + (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            - (deg_mat[(i, i)] * avg_deg_neighbors_vec[i] + deg_mat[(j, j)] * avg_deg_neighbors_vec[j])).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 41 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            - (deg_mat[(i, i)] + deg_mat[(j, j)])
                            + (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            + (2.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            - 4.0 * (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0 - (deg_mat[(i, i)] + deg_mat[(j, j)]) + (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j]) ;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 42 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0)
                            + 2.0 * avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j]).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 43 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (3.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[j].powf(2.0))
                            - 2.0 * avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j]
                            - 4.0 * (deg_mat[(i, i)] + deg_mat[(j, j)])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 44 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (2.0 * ((deg_mat[(i, i)] - 1.0).powf(2.0)
                            + (deg_mat[(j, j)] - 1.0).powf(2.0)
                            + avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j]
                            - deg_mat[(i, i)] * deg_mat[(j, j)])).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 45 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + ((deg_mat[(i, i)] - deg_mat[(j, j)]).powf(2.0)
                            + 2.0 * (deg_mat[(i, i)] * avg_deg_neighbors_vec[i] + deg_mat[(j, j)] * avg_deg_neighbors_vec[j])
                            - 4.0 * (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 46 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (2.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            - 16.0 * (deg_mat[(i, i)] * deg_mat[(j, j)]) / (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 47 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = (2.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            - (avg_deg_neighbors_vec[i] - avg_deg_neighbors_vec[j]).powf(2.0))
                            / (deg_mat[(i, i)] + deg_mat[(j, j)]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 48 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            / (2.0
                            + (2.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            - 4.0 * (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            + 4.0).sqrt());
                        if partial_result.is_nan() {
                            partial_result = deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0);
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 49 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (2.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[j].powf(2.0))
                            + (deg_mat[(i, i)] - deg_mat[(j, j)]).powf(2.0)
                            - 4.0 * (deg_mat[(i, i)] + deg_mat[(j, j)])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 50 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0 * (
                            (deg_mat[(i, i)].powf(2.0)
                                + deg_mat[(j, j)].powf(2.0)
                                + avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j]
                                - deg_mat[(i, i)] * deg_mat[(j, j)])
                                / (deg_mat[(i, i)] + deg_mat[(j, j)]));
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 51 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0 * (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            - 4.0 * avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j] / (deg_mat[(i, i)] + deg_mat[(j, j)]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 52 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + ((8.0 * (avg_deg_neighbors_vec[i].powf(4.0) + avg_deg_neighbors_vec[j].powf(4.0))
                            - 8.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            + 4.0).sqrt()
                            - 4.0 * (deg_mat[(i, i)] + deg_mat[(j, j)])
                            + 6.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 53 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + ((8.0 * (avg_deg_neighbors_vec[i].powf(4.0) + avg_deg_neighbors_vec[j].powf(4.0))
                            - 8.0 * (deg_mat[(i, i)] * avg_deg_neighbors_vec[i] + deg_mat[(j, j)] * avg_deg_neighbors_vec[j])
                            + 4.0).sqrt()
                            - 4.0 * (deg_mat[(i, i)] + deg_mat[(j, j)])
                            + 6.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 54 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (2.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[j].powf(2.0))
                            + (deg_mat[(i, i)] * avg_deg_neighbors_vec[i] + deg_mat[(j, j)] * avg_deg_neighbors_vec[j])
                            - (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            - 4.0 * (deg_mat[(i, i)] + deg_mat[(j, j)])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 55 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (3.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[j].powf(2.0))
                            - (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            - 4.0 * (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 56 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = ((deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0)) * (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j]))
                            / (2.0 * deg_mat[(i, i)] * deg_mat[(j, j)]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 57 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (2.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[j].powf(2.0))
                            - 8.0 * (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0)) / (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 58 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (2.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j] + avg_deg_neighbors_vec[j].powf(2.0))
                            - (deg_mat[(i, i)] * avg_deg_neighbors_vec[i] + deg_mat[(j, j)] * avg_deg_neighbors_vec[j])
                            - 4.0 * (deg_mat[(i, i)] + deg_mat[(j, j)])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 59 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = (2.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j] + avg_deg_neighbors_vec[j].powf(2.0))
                            - (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0)))
                            / (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 60 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (2.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j] + avg_deg_neighbors_vec[j].powf(2.0))
                            - (deg_mat[(i, i)].powf(2.0) + deg_mat[(j, j)].powf(2.0))
                            - 4.0 * (deg_mat[(i, i)] + deg_mat[(j, j)])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 61 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0 * (avg_deg_neighbors_vec[i].powf(2.0) + avg_deg_neighbors_vec[j].powf(2.0))
                            / (2.0 + (2.0 * (deg_mat[(i, i)] - 1.0).powf(2.0) + 2.0 * (deg_mat[(j, j)] - 1.0).powf(2.0)).sqrt());
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 62 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + (avg_deg_neighbors_vec[i].powf(2.0)
                            + 4.0 * avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j]
                            + avg_deg_neighbors_vec[j].powf(2.0)
                            - 2.0 * deg_mat[(i, i)] * deg_mat[(j, j)]
                            - 4.0 * (deg_mat[(i, i)] + deg_mat[(j, j)])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 63 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = deg_mat[(i, i)] + deg_mat[(j, j)] + avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j]
                            - 4.0 * deg_mat[(i, i)] * deg_mat[(j, j)] / (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 64 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j] * (deg_mat[(i, i)] + deg_mat[(j, j)]) / (deg_mat[(i, i)] * deg_mat[(j, j)]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 65 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            * (deg_mat[(i, i)] * avg_deg_neighbors_vec[i] + deg_mat[(j, j)] * avg_deg_neighbors_vec[j])
                            / (2.0 * avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 66 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = (avg_deg_neighbors_vec[i].powf(2.0)
                            + 4.0 * avg_deg_neighbors_vec[i] * avg_deg_neighbors_vec[j]
                            + avg_deg_neighbors_vec[j].powf(2.0)
                            - (deg_mat[(i, i)] * avg_deg_neighbors_vec[i] + deg_mat[(j, j)] * avg_deg_neighbors_vec[j]))
                            / (deg_mat[(i, i)] + deg_mat[(j, j)]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 67 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            * (deg_mat[(i, i)] * avg_deg_neighbors_vec[i] + deg_mat[(j, j)] * avg_deg_neighbors_vec[j])
                            / (2.0 * deg_mat[(i, i)] * deg_mat[(j, j)]);
                        if partial_result.is_nan() {
                            partial_result = 0.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        } else if self.conj == 68 {
            for i in 0..self.n_sommet {
                for j in (i+1)..self.n_sommet {
                    if self.adj_mat[(i, j)] == 1.0 {
                        let mut partial_result = 2.0
                            + ((avg_deg_neighbors_vec[i] - avg_deg_neighbors_vec[j]).powf(2.0)
                            + 4.0 * deg_mat[(i, i)] * deg_mat[(j, j)]
                            - 4.0 * (avg_deg_neighbors_vec[i] + avg_deg_neighbors_vec[j])
                            + 4.0).sqrt();
                        if partial_result.is_nan() {
                            partial_result = 2.0;
                        }
                        results_vec.push(partial_result);
                    }
                }
            }
        }

        let result = if results_vec.is_empty() {
            0.0
        } else {
            *results_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        };
        let sc = mu - result;

        return sc
    }

    pub fn terminal(& self) -> bool {
        return self.n_sommet > self.size_terminal;
    }
}

pub fn construct_SQ_star(i: usize) -> State {
    let mut st1 = State::new();
    st1.n_sommet = 12;
    st1.conj = i;

    st1.adj_mat.resize_mut(st1.n_sommet, st1.n_sommet, 0.0);
    let aretes = vec![(0, 1), (0, 5), (0, 6),
                      (1, 2), (1, 7),
                      (2, 3), (2, 8),
                      (3, 4), (3, 9),
                      (4, 5), (4, 10),
                      (5, 11),
                      (6, 7), (6, 9), (6, 11),
                      (7, 8), (7, 10),
                      (8, 9), (8, 11),
                      (9, 10),
                      (10, 11)];
    for arete in aretes {
        let arete_from = arete.0;
        let arete_to = arete.1;
        st1.adj_mat[arete] = 1.0;
        st1.adj_mat[(arete_to, arete_from)] = 1.0;
    }

    st1
}

pub fn construct_SQ_17(i: usize) -> State {
    let mut st1 = State::new();
    st1.n_sommet = 12;
    st1.conj = i;

    st1.adj_mat.resize_mut(st1.n_sommet, st1.n_sommet, 0.0);
    let aretes = vec![(0, 1), (0, 11),
                      (1, 2), (1, 4), (1, 10),
                      (2, 3), (2, 5), (2, 11),
                      (3, 4),
                      (4, 5), (4, 7),
                      (5, 6), (5, 8),
                      (6, 7),
                      (7, 8), (7, 10),
                      (8, 9), (8, 11),
                      (9, 10),
                      (10, 11)];
    for arete in aretes {
        let arete_from = arete.0;
        let arete_to = arete.1;
        st1.adj_mat[arete] = 1.0;
        st1.adj_mat[(arete_to, arete_from)] = 1.0;
    }

    st1
}

pub fn construct_SQ_50(i: usize) -> State {
    let mut st1 = State::new();
    st1.n_sommet = 12;
    st1.conj = i;

    st1.adj_mat.resize_mut(st1.n_sommet, st1.n_sommet, 0.0);
    let aretes = vec![(0, 1), (0, 11),
                      (1, 2),
                      (2, 3), (2, 9),
                      (3, 4), (3, 8), (3, 10),
                      (4, 5), (4, 7), (4, 9),
                      (5, 6), (5, 8),
                      (6, 7),
                      (7, 8),
                      (8, 9),
                      (9, 10),
                      (10, 11)];
    for arete in aretes {
        let arete_from = arete.0;
        let arete_to = arete.1;
        st1.adj_mat[arete] = 1.0;
        st1.adj_mat[(arete_to, arete_from)] = 1.0;
    }

    st1
}

pub fn construct_SQ_66(i: usize) -> State {
    let mut st1 = State::new();
    st1.n_sommet = 12;
    st1.conj = i;

    st1.adj_mat.resize_mut(st1.n_sommet, st1.n_sommet, 0.0);
    let aretes = vec![(0, 1),
                      (1, 2),
                      (2, 3),
                      (3, 4), (3, 11),
                      (4, 5), (4, 10),
                      (5, 6), (5, 11),
                      (6, 7),
                      (7, 8),
                      (8, 9),
                      (9, 10),
                      (10, 11)];
    for arete in aretes {
        let arete_from = arete.0;
        let arete_to = arete.1;
        st1.adj_mat[arete] = 1.0;
        st1.adj_mat[(arete_to, arete_from)] = 1.0;
    }

    st1
}


