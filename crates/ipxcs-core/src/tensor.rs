use ndarray::ArrayD;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: ArrayD<f32>,
}
