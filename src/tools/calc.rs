pub fn softmaxChoice(l : Vec<f64>) -> usize {
    let r = rand::random::<f64>();

    let mut sum = 0.0;
    for i in 0..l.len() {
        sum += l[i].exp();
    }

    let mut sum2 = 0.0;
    for i in 0..l.len() {
        sum2+= l[i].exp()/sum;
        if sum2 >= r{
            return i;
        }
    }

    println!("The heuristic may be too big, or the algo cannot differentiate paths.");
    return l.len() - 1;
}
