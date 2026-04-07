use ndarray::{ArrayD, IxDyn};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: ArrayD<f32>,
}

impl Tensor {
    pub fn new(data: ArrayD<f32>) -> Self {
        Self { data }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn into_data(self) -> ArrayD<f32> {
        self.data
    }
}
