mod tensor;

pub use tensor::Tensor;

pub trait Op {
    fn name(&self) -> &str;
    fn run(&self, inputs: &[Tensor]) -> anyhow::Result<Tensor>;
}