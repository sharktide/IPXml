use ipxcs_core::{apply_ops, argmax, softmax, topk_indices, Tensor};
use ipxml_schema::OpSpec;
use ndarray::array;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

#[test]
fn test_normalize_op() {
    let data = array![[1.0, 2.0, 3.0]].into_dyn();
    let tensor = Tensor::new(data);
    let ops = vec![OpSpec::Normalize {
        scale: None,
        mean: Some(vec![1.0, 1.0, 1.0]),
        std: Some(vec![1.0, 2.0, 1.0]),
    }];
    let out = apply_ops(tensor, &ops).unwrap();
    let values: Vec<f32> = out.data.iter().copied().collect();
    assert!(approx_eq(values[0], 0.0));
    assert!(approx_eq(values[1], 0.5));
    assert!(approx_eq(values[2], 2.0));
}

#[test]
fn test_softmax() {
    let data = array![0.0, 0.0].into_dyn();
    let tensor = Tensor::new(data);
    let out = softmax(&tensor, Some(0)).unwrap();
    let values: Vec<f32> = out.data.iter().copied().collect();
    assert!(approx_eq(values[0], 0.5));
    assert!(approx_eq(values[1], 0.5));
}

#[test]
fn test_argmax() {
    let data = array![1.0, 3.0, 2.0].into_dyn();
    let tensor = Tensor::new(data);
    let out = argmax(&tensor, Some(0)).unwrap();
    let values: Vec<f32> = out.data.iter().copied().collect();
    assert_eq!(values[0] as usize, 1);
}

#[test]
fn test_topk_indices() {
    let data = array![0.1, 0.9, 0.2].into_dyn();
    let tensor = Tensor::new(data);
    let scores = topk_indices(&tensor, 2, Some(0), true).unwrap();
    assert_eq!(scores[0].0, 1);
    assert_eq!(scores[1].0, 2);
}

#[test]
fn test_reshape_transpose_ops() {
    let data = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let tensor = Tensor::new(data);
    let ops = vec![
        OpSpec::Reshape { shape: vec![1, 2, 2] },
        OpSpec::Transpose { axes: vec![0, 2, 1] },
    ];
    let out = apply_ops(tensor, &ops).unwrap();
    assert_eq!(out.data.shape(), &[1, 2, 2]);
    let values: Vec<f32> = out.data.iter().copied().collect();
    assert!(approx_eq(values[0], 1.0));
    assert!(approx_eq(values[1], 3.0));
}

#[test]
fn test_expr_scale() {
    let data = array![1.0, 2.0].into_dyn();
    let tensor = Tensor::new(data);
    let ops = vec![OpSpec::Expr { code: "scale(x, 2.0)".to_string() }];
    let out = apply_ops(tensor, &ops).unwrap();
    let values: Vec<f32> = out.data.iter().copied().collect();
    assert!(approx_eq(values[0], 2.0));
    assert!(approx_eq(values[1], 4.0));
}
