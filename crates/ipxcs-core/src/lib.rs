mod ops;
mod tensor;

pub use ops::{apply_op, apply_ops, argmax, eval_expr, softmax, topk_indices, topk_values};
pub use tensor::Tensor;
