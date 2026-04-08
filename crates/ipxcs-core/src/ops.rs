use anyhow::{Context, Result, anyhow};
use ipxml_schema::{OpSpec, RuleSpec, TensorLiteral};
use ndarray::{Array3, Array4, ArrayD, Axis, Ix2, Ix3, Ix4, IxDyn, Zip};
use rhai::{Engine, EvalAltResult, Scope};

use crate::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageLayout {
    Hwc,
    Chw,
    Nhwc,
    Nchw,
}

pub fn apply_ops(mut tensor: Tensor, ops: &[OpSpec]) -> Result<Tensor> {
    for op in ops {
        tensor = apply_op(tensor, op)?;
    }
    Ok(tensor)
}

pub fn apply_op(tensor: Tensor, op: &OpSpec) -> Result<Tensor> {
    match op {
        OpSpec::ApplyIf {
            when,
            rules,
            then_ops,
            otherwise,
        } => {
            if evaluate_condition(when.as_deref(), rules.as_deref()) {
                apply_ops(tensor, then_ops)
            } else if let Some(otherwise_ops) = otherwise {
                apply_ops(tensor, otherwise_ops)
            } else {
                Ok(tensor)
            }
        }
        OpSpec::Resize {
            width,
            height,
            layout,
        } => resize(tensor, *width, *height, layout.as_deref()),
        OpSpec::CenterCrop {
            width,
            height,
            layout,
        } => center_crop(tensor, *width, *height, layout.as_deref()),
        OpSpec::Normalize { scale, mean, std } => {
            normalize(tensor, *scale, mean.as_ref(), std.as_ref())
        }
        OpSpec::Scale { factor } => Ok(Tensor::new(tensor.data.mapv(|v| v * *factor))),
        OpSpec::Cast { dtype } => cast(tensor, dtype),
        OpSpec::Clip { min, max } => Ok(Tensor::new(tensor.data.mapv(|v| v.clamp(*min, *max)))),
        OpSpec::Transpose { axes } => transpose(tensor, axes),
        OpSpec::Reshape { shape } => reshape(tensor, shape),
        OpSpec::Squeeze { axes } => squeeze(tensor, axes.as_deref()),
        OpSpec::Unsqueeze { axes } => unsqueeze(tensor, axes),
        OpSpec::Softmax { axis } => softmax(&tensor, *axis),
        OpSpec::ArgMax { axis } => argmax(&tensor, *axis),
        OpSpec::TopK { k, axis, largest } => {
            topk_values(&tensor, *k, *axis, largest.unwrap_or(true))
        }
        OpSpec::MatMul { rhs } => matmul(tensor, rhs),
        OpSpec::Add { value, tensor: rhs } => binary_op(tensor, value, rhs, |a, b| a + b),
        OpSpec::Mul { value, tensor: rhs } => binary_op(tensor, value, rhs, |a, b| a * b),
        OpSpec::Div { value, tensor: rhs } => binary_op(tensor, value, rhs, |a, b| a / b),
        OpSpec::Sub { value, tensor: rhs } => binary_op(tensor, value, rhs, |a, b| a - b),
        OpSpec::Mean { axis, keepdims } => reduce_mean(tensor, *axis, keepdims.unwrap_or(false)),
        OpSpec::Std { axis, keepdims } => reduce_std(tensor, *axis, keepdims.unwrap_or(false)),
        OpSpec::Sum { axis, keepdims } => reduce_sum(tensor, *axis, keepdims.unwrap_or(false)),
        OpSpec::Expr { code } => eval_expr(tensor, code),
    }
}

fn evaluate_condition(when: Option<&str>, rules: Option<&[RuleSpec]>) -> bool {
    if let Some(expr) = when {
        return eval_bool(expr);
    }
    if let Some(rules) = rules {
        for rule in rules {
            if eval_bool(&rule.if_expr) {
                return rule.then.as_ref().and_then(|a| a.run).unwrap_or(true);
            }
            if let Some(otherwise) = &rule.otherwise {
                if let Some(run) = otherwise.run {
                    return run;
                }
            }
        }
    }
    true
}

fn eval_bool(expr: &str) -> bool {
    let engine = Engine::new();
    engine.eval_expression::<bool>(expr).unwrap_or(false)
}

pub fn softmax(tensor: &Tensor, axis: Option<isize>) -> Result<Tensor> {
    let axis = normalize_axis(axis.unwrap_or(-1), tensor.ndim())?;
    let mut data = tensor.data.clone();
    for mut lane in data.lanes_mut(Axis(axis)) {
        let max = lane.iter().fold(f32::NEG_INFINITY, |acc, v| acc.max(*v));
        let mut sum = 0.0;
        lane.mapv_inplace(|v| {
            let e = (v - max).exp();
            sum += e;
            e
        });
        if sum != 0.0 {
            lane.mapv_inplace(|v| v / sum);
        }
    }
    Ok(Tensor::new(data))
}

pub fn argmax(tensor: &Tensor, axis: Option<isize>) -> Result<Tensor> {
    let axis = normalize_axis(axis.unwrap_or(-1), tensor.ndim())?;
    let data = tensor
        .data
        .map_axis(Axis(axis), |lane| {
            lane.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as f32)
                .unwrap_or(0.0)
        })
        .into_dyn();
    Ok(Tensor::new(data))
}

pub fn topk_values(
    tensor: &Tensor,
    k: usize,
    axis: Option<isize>,
    largest: bool,
) -> Result<Tensor> {
    let axis = normalize_axis(axis.unwrap_or(-1), tensor.ndim())?;
    let mut output = Vec::new();
    for lane in tensor.data.lanes(Axis(axis)) {
        let mut pairs: Vec<(usize, f32)> = lane.iter().cloned().enumerate().collect();
        if largest {
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        } else {
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        output.extend(pairs.iter().take(k).map(|(_, v)| *v));
    }

    let mut shape = tensor.data.shape().to_vec();
    if axis < shape.len() {
        shape[axis] = k;
    }
    let array = ArrayD::from_shape_vec(IxDyn(&shape), output).context("topk reshape")?;
    Ok(Tensor::new(array))
}

pub fn topk_indices(
    tensor: &Tensor,
    k: usize,
    axis: Option<isize>,
    largest: bool,
) -> Result<Vec<(usize, f32)>> {
    let axis = normalize_axis(axis.unwrap_or(-1), tensor.ndim())?;
    let data = &tensor.data;
    if data.ndim() > 2 {
        return Err(anyhow!("TopK decoding supports only 1D or 2D tensors."));
    }

    let mut pairs: Vec<(usize, f32)> = if data.ndim() == 1 {
        data.iter().cloned().enumerate().collect()
    } else {
        let lane = if data.ndim() == 2 {
            data.index_axis(Axis(0), 0)
        } else {
            data.index_axis(Axis(axis), 0)
        };
        lane.iter().cloned().enumerate().collect()
    };
    if largest {
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    } else {
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    }
    Ok(pairs.into_iter().take(k).collect())
}

pub fn eval_expr(tensor: Tensor, code: &str) -> Result<Tensor> {
    let mut engine = Engine::new();
    engine.register_type::<Tensor>();
    engine.register_fn(
        "reshape",
        |t: Tensor, shape: rhai::Array| -> Result<Tensor, Box<EvalAltResult>> {
            reshape(t, &array_to_i64_vec(&shape)?).map_err(to_rhai_err)
        },
    );
    engine.register_fn(
        "transpose",
        |t: Tensor, axes: rhai::Array| -> Result<Tensor, Box<EvalAltResult>> {
            transpose(t, &array_to_usize_vec(&axes)?).map_err(to_rhai_err)
        },
    );
    engine.register_fn(
        "matmul",
        |a: Tensor, b: Tensor| -> Result<Tensor, Box<EvalAltResult>> {
            matmul_tensor(a, b).map_err(to_rhai_err)
        },
    );
    engine.register_fn("sum", |t: Tensor| -> Result<Tensor, Box<EvalAltResult>> {
        reduce_sum(t, None, false).map_err(to_rhai_err)
    });
    engine.register_fn("mean", |t: Tensor| -> Result<Tensor, Box<EvalAltResult>> {
        reduce_mean(t, None, false).map_err(to_rhai_err)
    });
    engine.register_fn("std", |t: Tensor| -> Result<Tensor, Box<EvalAltResult>> {
        reduce_std(t, None, false).map_err(to_rhai_err)
    });
    engine.register_fn(
        "softmax",
        |t: Tensor| -> Result<Tensor, Box<EvalAltResult>> {
            softmax(&t, None).map_err(to_rhai_err)
        },
    );
    engine.register_fn(
        "argmax",
        |t: Tensor| -> Result<Tensor, Box<EvalAltResult>> { argmax(&t, None).map_err(to_rhai_err) },
    );
    engine.register_fn(
        "topk",
        |t: Tensor, k: i64| -> Result<Tensor, Box<EvalAltResult>> {
            topk_values(&t, k as usize, None, true).map_err(to_rhai_err)
        },
    );
    engine.register_fn("clip", |t: Tensor, min: f64, max: f64| -> Tensor {
        Tensor::new(t.data.mapv(|v| v.clamp(min as f32, max as f32)))
    });
    engine.register_fn(
        "normalize",
        |t: Tensor,
         mean: rhai::Array,
         std: rhai::Array,
         scale: f64|
         -> Result<Tensor, Box<EvalAltResult>> {
            normalize(
                t,
                Some(scale as f32),
                Some(&array_to_f32_vec(&mean)?),
                Some(&array_to_f32_vec(&std)?),
            )
            .map_err(to_rhai_err)
        },
    );
    engine.register_fn("scale", |t: Tensor, factor: f64| -> Tensor {
        Tensor::new(t.data.mapv(|v| v * factor as f32))
    });

    let mut scope = Scope::new();
    scope.push("x", tensor);
    let ast = engine.compile(code).map_err(|e| anyhow!(e.to_string()))?;
    engine
        .eval_ast_with_scope::<Tensor>(&mut scope, &ast)
        .map_err(|e| anyhow!(e.to_string()))
}

fn to_rhai_err(err: anyhow::Error) -> Box<EvalAltResult> {
    err.to_string().into()
}

fn array_to_i64_vec(array: &rhai::Array) -> Result<Vec<i64>, Box<EvalAltResult>> {
    let mut out = Vec::with_capacity(array.len());
    for value in array {
        match value.as_int() {
            Ok(int) => out.push(int),
            Err(_) => return Err("Expected integer array".into()),
        }
    }
    Ok(out)
}

fn array_to_usize_vec(array: &rhai::Array) -> Result<Vec<usize>, Box<EvalAltResult>> {
    let mut out = Vec::with_capacity(array.len());
    for value in array {
        match value.as_int() {
            Ok(int) => out.push(int as usize),
            Err(_) => return Err("Expected integer array".into()),
        }
    }
    Ok(out)
}

fn array_to_f32_vec(array: &rhai::Array) -> Result<Vec<f32>, Box<EvalAltResult>> {
    let mut out = Vec::with_capacity(array.len());
    for value in array {
        if let Ok(float) = value.as_float() {
            out.push(float as f32);
        } else if let Ok(int) = value.as_int() {
            out.push(int as f32);
        } else {
            return Err("Expected float array".into());
        }
    }
    Ok(out)
}

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    let axis = if axis < 0 {
        (ndim as isize + axis) as isize
    } else {
        axis
    };
    if axis < 0 || axis as usize >= ndim {
        Err(anyhow!("Axis {axis} out of bounds for ndim {ndim}"))
    } else {
        Ok(axis as usize)
    }
}

fn cast(tensor: Tensor, dtype: &str) -> Result<Tensor> {
    match dtype.to_ascii_lowercase().as_str() {
        "f32" | "float" | "float32" => Ok(tensor),
        other => Err(anyhow!("Unsupported cast dtype: {other}")),
    }
}

fn reshape(tensor: Tensor, shape: &[i64]) -> Result<Tensor> {
    let mut shape = shape.to_vec();
    let mut infer_index = None;
    let mut known_product: i64 = 1;

    for (i, dim) in shape.iter().enumerate() {
        if *dim == -1 {
            if infer_index.is_some() {
                return Err(anyhow!("Only one -1 dimension is supported in reshape."));
            }
            infer_index = Some(i);
        } else if *dim <= 0 {
            return Err(anyhow!("Reshape dimensions must be positive or -1."));
        } else {
            known_product *= *dim;
        }
    }

    if let Some(idx) = infer_index {
        let total: i64 = tensor.data.len() as i64;
        shape[idx] = total / known_product;
    }

    let shape_usize: Vec<usize> = shape.iter().map(|v| *v as usize).collect();
    let data = tensor
        .data
        .into_shape_with_order(IxDyn(&shape_usize))
        .context("reshape tensor")?;
    Ok(Tensor::new(data))
}

fn transpose(tensor: Tensor, axes: &[usize]) -> Result<Tensor> {
    if axes.len() != tensor.ndim() {
        return Err(anyhow!("Transpose axes length must match tensor ndim."));
    }
    Ok(Tensor::new(tensor.data.permuted_axes(axes.to_vec())))
}

fn squeeze(tensor: Tensor, axes: Option<&[usize]>) -> Result<Tensor> {
    let mut shape = tensor.shape().to_vec();
    if let Some(axes) = axes {
        let mut axes = axes.to_vec();
        axes.sort_unstable_by(|a, b| b.cmp(a));
        for axis in axes {
            if axis >= shape.len() || shape[axis] != 1 {
                return Err(anyhow!(
                    "Cannot squeeze axis {axis} with size {}",
                    shape.get(axis).unwrap_or(&0)
                ));
            }
            shape.remove(axis);
        }
    } else {
        shape.retain(|dim| *dim != 1);
        if shape.is_empty() {
            shape.push(1);
        }
    }
    Ok(Tensor::new(
        tensor.data.into_shape_with_order(IxDyn(&shape))?,
    ))
}

fn unsqueeze(tensor: Tensor, axes: &[usize]) -> Result<Tensor> {
    let mut shape = tensor.shape().to_vec();
    let mut axes = axes.to_vec();
    axes.sort_unstable();
    for axis in axes {
        if axis > shape.len() {
            return Err(anyhow!(
                "Cannot unsqueeze axis {axis} for shape {:?}",
                shape
            ));
        }
        shape.insert(axis, 1);
    }
    Ok(Tensor::new(
        tensor.data.into_shape_with_order(IxDyn(&shape))?,
    ))
}

fn reduce_sum(tensor: Tensor, axis: Option<isize>, keepdims: bool) -> Result<Tensor> {
    reduce_axis(tensor, axis, keepdims, |data, axis| {
        Ok(data.sum_axis(Axis(axis)).into_dyn())
    })
}

fn reduce_mean(tensor: Tensor, axis: Option<isize>, keepdims: bool) -> Result<Tensor> {
    if axis.is_none() {
        let mean = tensor.data.mean().unwrap_or(0.0);
        return Ok(Tensor::new(ArrayD::from_elem(IxDyn(&[]), mean)));
    }
    reduce_axis(tensor, axis, keepdims, |data, axis| {
        Ok(data
            .mean_axis(Axis(axis))
            .context("mean reduction")?
            .into_dyn())
    })
}

fn reduce_std(tensor: Tensor, axis: Option<isize>, keepdims: bool) -> Result<Tensor> {
    if axis.is_none() {
        let mean = tensor.data.mean().unwrap_or(0.0);
        let var = tensor
            .data
            .mapv(|v| {
                let d = v - mean;
                d * d
            })
            .mean()
            .unwrap_or(0.0);
        return Ok(Tensor::new(ArrayD::from_elem(IxDyn(&[]), var.sqrt())));
    }
    reduce_axis(tensor, axis, keepdims, |data, axis| {
        let mean = data.mean_axis(Axis(axis)).context("mean reduction")?;
        let mean = mean.insert_axis(Axis(axis));
        let diff = &data - &mean;
        let var = diff
            .mapv(|v| v * v)
            .mean_axis(Axis(axis))
            .context("variance")?;
        Ok(var.mapv(|v| v.sqrt()).into_dyn())
    })
}

fn reduce_axis<F>(tensor: Tensor, axis: Option<isize>, keepdims: bool, op: F) -> Result<Tensor>
where
    F: FnOnce(ArrayD<f32>, usize) -> Result<ArrayD<f32>>,
{
    if let Some(axis) = axis {
        let axis = normalize_axis(axis, tensor.ndim())?;
        let reduced = op(tensor.data, axis)?;
        let output = if keepdims {
            reduced.insert_axis(Axis(axis)).into_dyn()
        } else {
            reduced.into_dyn()
        };
        Ok(Tensor::new(output))
    } else {
        let sum = tensor.data.sum();
        Ok(Tensor::new(ArrayD::from_elem(IxDyn(&[]), sum)))
    }
}

fn normalize(
    tensor: Tensor,
    scale: Option<f32>,
    mean: Option<&Vec<f32>>,
    std: Option<&Vec<f32>>,
) -> Result<Tensor> {
    let mut data = tensor.data;
    if let Some(scale) = scale {
        data.mapv_inplace(|v| v * scale);
    }

    if let (Some(mean), Some(std)) = (mean, std) {
        let axis = detect_channel_axis(&data, mean.len()).unwrap_or(data.ndim().saturating_sub(1));
        for c in 0..mean.len() {
            let mean_val = mean[c];
            let std_val = std.get(c).copied().unwrap_or(1.0);
            let mut slice = data.index_axis_mut(Axis(axis), c);
            slice.mapv_inplace(|v| (v - mean_val) / std_val);
        }
    }

    Ok(Tensor::new(data))
}

fn detect_channel_axis(data: &ArrayD<f32>, channels: usize) -> Option<usize> {
    let shape = data.shape();
    if shape.is_empty() {
        return None;
    }
    if let Some((idx, _)) = shape.iter().enumerate().find(|(_, dim)| **dim == channels) {
        Some(idx)
    } else {
        None
    }
}

fn binary_op<F>(
    tensor: Tensor,
    value: &Option<f32>,
    rhs: &Option<TensorLiteral>,
    op: F,
) -> Result<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    if let Some(value) = value {
        let data = tensor.data.mapv(|v| op(v, *value));
        return Ok(Tensor::new(data));
    }
    if let Some(rhs) = rhs {
        let rhs_tensor = tensor_from_literal(rhs)?;
        if let Some(lhs_broadcast) = tensor.data.broadcast(rhs_tensor.data.raw_dim()) {
            let rhs_broadcast = rhs_tensor
                .data
                .broadcast(lhs_broadcast.raw_dim())
                .ok_or_else(|| anyhow!("Broadcast shapes not compatible"))?;
            let mut out = lhs_broadcast.to_owned();
            Zip::from(&mut out).and(&rhs_broadcast).for_each(|a, b| {
                *a = op(*a, *b);
            });
            return Ok(Tensor::new(out));
        }
        if let Some(rhs_broadcast) = rhs_tensor.data.broadcast(tensor.data.raw_dim()) {
            let mut out = tensor.data.clone();
            Zip::from(&mut out).and(&rhs_broadcast).for_each(|a, b| {
                *a = op(*a, *b);
            });
            return Ok(Tensor::new(out));
        }
        return Err(anyhow!("Broadcast shapes not compatible"));
    }
    Err(anyhow!("Binary op requires value or tensor"))
}

fn matmul(tensor: Tensor, rhs: &TensorLiteral) -> Result<Tensor> {
    let rhs_tensor = tensor_from_literal(rhs)?;
    matmul_tensor(tensor, rhs_tensor)
}

fn matmul_tensor(lhs: Tensor, rhs: Tensor) -> Result<Tensor> {
    if lhs.ndim() != 2 || rhs.ndim() != 2 {
        return Err(anyhow!("MatMul currently supports 2D tensors only."));
    }
    let lhs = lhs.data.into_dimensionality::<Ix2>()?;
    let rhs = rhs.data.into_dimensionality::<Ix2>()?;
    let result = lhs.dot(&rhs);
    Ok(Tensor::new(result.into_dyn()))
}

fn tensor_from_literal(lit: &TensorLiteral) -> Result<Tensor> {
    let array = ArrayD::from_shape_vec(IxDyn(&lit.shape), lit.data.clone())
        .context("tensor literal shape")?;
    Ok(Tensor::new(array))
}

fn resize(tensor: Tensor, width: usize, height: usize, layout: Option<&str>) -> Result<Tensor> {
    let layout = infer_layout(tensor.shape(), layout);
    match layout {
        ImageLayout::Hwc => {
            let data = tensor.data.into_dimensionality::<Ix3>()?;
            Ok(Tensor::new(resize_hwc(data, width, height).into_dyn()))
        }
        ImageLayout::Chw => {
            let data = tensor.data.into_dimensionality::<Ix3>()?;
            Ok(Tensor::new(resize_chw(data, width, height).into_dyn()))
        }
        ImageLayout::Nhwc => {
            let data = tensor.data.into_dimensionality::<Ix4>()?;
            Ok(Tensor::new(resize_nhwc(data, width, height).into_dyn()))
        }
        ImageLayout::Nchw => {
            let data = tensor.data.into_dimensionality::<Ix4>()?;
            Ok(Tensor::new(resize_nchw(data, width, height).into_dyn()))
        }
    }
}

fn center_crop(
    tensor: Tensor,
    width: usize,
    height: usize,
    layout: Option<&str>,
) -> Result<Tensor> {
    let layout = infer_layout(tensor.shape(), layout);
    match layout {
        ImageLayout::Hwc => {
            let data = tensor.data.into_dimensionality::<Ix3>()?;
            Ok(Tensor::new(center_crop_hwc(data, width, height).into_dyn()))
        }
        ImageLayout::Chw => {
            let data = tensor.data.into_dimensionality::<Ix3>()?;
            Ok(Tensor::new(center_crop_chw(data, width, height).into_dyn()))
        }
        ImageLayout::Nhwc => {
            let data = tensor.data.into_dimensionality::<Ix4>()?;
            Ok(Tensor::new(
                center_crop_nhwc(data, width, height).into_dyn(),
            ))
        }
        ImageLayout::Nchw => {
            let data = tensor.data.into_dimensionality::<Ix4>()?;
            Ok(Tensor::new(
                center_crop_nchw(data, width, height).into_dyn(),
            ))
        }
    }
}

fn infer_layout(shape: &[usize], layout: Option<&str>) -> ImageLayout {
    if let Some(layout) = layout {
        return match layout.to_ascii_lowercase().as_str() {
            "nhwc" => ImageLayout::Nhwc,
            "nchw" => ImageLayout::Nchw,
            "chw" => ImageLayout::Chw,
            "hwc" => ImageLayout::Hwc,
            _ => ImageLayout::Nchw,
        };
    }
    match shape.len() {
        4 => {
            if shape[1] == 1 || shape[1] == 3 {
                ImageLayout::Nchw
            } else {
                ImageLayout::Nhwc
            }
        }
        3 => {
            if shape[0] == 1 || shape[0] == 3 {
                ImageLayout::Chw
            } else {
                ImageLayout::Hwc
            }
        }
        _ => ImageLayout::Hwc,
    }
}

fn resize_hwc(data: Array3<f32>, width: usize, height: usize) -> Array3<f32> {
    let (h, w, c) = data.dim();
    let mut out = Array3::<f32>::zeros((height, width, c));
    for y in 0..height {
        let src_y = y * h / height.max(1);
        for x in 0..width {
            let src_x = x * w / width.max(1);
            for ch in 0..c {
                out[[y, x, ch]] = data[[src_y, src_x, ch]];
            }
        }
    }
    out
}

fn resize_chw(data: Array3<f32>, width: usize, height: usize) -> Array3<f32> {
    let (c, h, w) = data.dim();
    let mut out = Array3::<f32>::zeros((c, height, width));
    for ch in 0..c {
        for y in 0..height {
            let src_y = y * h / height.max(1);
            for x in 0..width {
                let src_x = x * w / width.max(1);
                out[[ch, y, x]] = data[[ch, src_y, src_x]];
            }
        }
    }
    out
}

fn resize_nhwc(data: Array4<f32>, width: usize, height: usize) -> Array4<f32> {
    let (n, h, w, c) = data.dim();
    let mut out = Array4::<f32>::zeros((n, height, width, c));
    for b in 0..n {
        for y in 0..height {
            let src_y = y * h / height.max(1);
            for x in 0..width {
                let src_x = x * w / width.max(1);
                for ch in 0..c {
                    out[[b, y, x, ch]] = data[[b, src_y, src_x, ch]];
                }
            }
        }
    }
    out
}

fn resize_nchw(data: Array4<f32>, width: usize, height: usize) -> Array4<f32> {
    let (n, c, h, w) = data.dim();
    let mut out = Array4::<f32>::zeros((n, c, height, width));
    for b in 0..n {
        for ch in 0..c {
            for y in 0..height {
                let src_y = y * h / height.max(1);
                for x in 0..width {
                    let src_x = x * w / width.max(1);
                    out[[b, ch, y, x]] = data[[b, ch, src_y, src_x]];
                }
            }
        }
    }
    out
}

fn center_crop_hwc(data: Array3<f32>, width: usize, height: usize) -> Array3<f32> {
    let (h, w, c) = data.dim();
    let start_y = h.saturating_sub(height) / 2;
    let start_x = w.saturating_sub(width) / 2;
    let mut out = Array3::<f32>::zeros((height, width, c));
    for y in 0..height {
        for x in 0..width {
            for ch in 0..c {
                out[[y, x, ch]] = data[[start_y + y, start_x + x, ch]];
            }
        }
    }
    out
}

fn center_crop_chw(data: Array3<f32>, width: usize, height: usize) -> Array3<f32> {
    let (c, h, w) = data.dim();
    let start_y = h.saturating_sub(height) / 2;
    let start_x = w.saturating_sub(width) / 2;
    let mut out = Array3::<f32>::zeros((c, height, width));
    for ch in 0..c {
        for y in 0..height {
            for x in 0..width {
                out[[ch, y, x]] = data[[ch, start_y + y, start_x + x]];
            }
        }
    }
    out
}

fn center_crop_nhwc(data: Array4<f32>, width: usize, height: usize) -> Array4<f32> {
    let (n, h, w, c) = data.dim();
    let start_y = h.saturating_sub(height) / 2;
    let start_x = w.saturating_sub(width) / 2;
    let mut out = Array4::<f32>::zeros((n, height, width, c));
    for b in 0..n {
        for y in 0..height {
            for x in 0..width {
                for ch in 0..c {
                    out[[b, y, x, ch]] = data[[b, start_y + y, start_x + x, ch]];
                }
            }
        }
    }
    out
}

fn center_crop_nchw(data: Array4<f32>, width: usize, height: usize) -> Array4<f32> {
    let (n, c, h, w) = data.dim();
    let start_y = h.saturating_sub(height) / 2;
    let start_x = w.saturating_sub(width) / 2;
    let mut out = Array4::<f32>::zeros((n, c, height, width));
    for b in 0..n {
        for ch in 0..c {
            for y in 0..height {
                for x in 0..width {
                    out[[b, ch, y, x]] = data[[b, ch, start_y + y, start_x + x]];
                }
            }
        }
    }
    out
}
