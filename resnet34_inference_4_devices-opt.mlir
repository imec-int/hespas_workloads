module @pmap_inference_step attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {suggested_combiner_threshold = "38935336880"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.num_replicas = 4 : i32, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot.1_computation(%arg0: tensor<256x512xbf16>, %arg1: tensor<512x1000xf32>) -> tensor<256x1000xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<512x1000xf32>) -> tensor<512x1000xbf16>
    %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[1000,256]{0,1}"} : (tensor<512x1000xbf16>, tensor<256x512xbf16>) -> tensor<1000x256xbf16>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x256xbf16>) -> tensor<256x1000xbf16>
    return %2 : tensor<256x1000xbf16>
  }
  func.func private @region_1.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<256x7x7x512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<256x7x7x512xf32>, %arg4: tensor<512xf32>) -> tensor<256x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x7x7x512xf32>
    %7 = stablehlo.convert %6 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x7x7x512xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x512xf32>) -> tensor<256x49x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x49x512xf32>, tensor<f32>) -> tensor<256x512xf32>
    return %12 : tensor<256x512xf32>
  }
  func.func private @fused_convert(%arg0: tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
    return %1 : tensor<256x224x224x3xf32>
  }
  func.func private @region_0.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg0 : (tensor<f32>) -> tensor<bf16>
    %1 = stablehlo.convert %arg1 : (tensor<f32>) -> tensor<bf16>
    %2 = stablehlo.maximum %0, %1 : tensor<bf16>
    %3 = stablehlo.convert %2 : (tensor<bf16>) -> tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @fused_reduce_window(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x112x112x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x112x112x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x112x112x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x112x112x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x112x112x64xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %10 = stablehlo.convert %cst_0 : (tensor<bf16>) -> tensor<f32>
    %11 = "stablehlo.reduce_window"(%9, %10) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %12 = stablehlo.convert %arg4 : (tensor<f32>) -> tensor<bf16>
      %13 = stablehlo.convert %arg5 : (tensor<f32>) -> tensor<bf16>
      %14 = stablehlo.maximum %12, %13 : tensor<bf16>
      %15 = stablehlo.convert %14 : (tensor<bf16>) -> tensor<f32>
      stablehlo.return %15 : tensor<f32>
    }) : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<256x56x56x64xf32>
    return %11 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.1(%arg0: tensor<256x56x56x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x56x56x64xf32>, %arg4: tensor<64xf32>) -> tensor<256x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x56x56x64xf32>
    %7 = stablehlo.convert %6 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x56x56x64xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    return %10 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.2(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x56x56x64xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    return %9 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.3(%arg0: tensor<256x56x56x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x56x56x64xf32>, %arg4: tensor<64xf32>) -> tensor<256x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x56x56x64xf32>
    %7 = stablehlo.convert %6 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x56x56x64xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    return %10 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.4(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x56x56x64xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    return %9 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.5(%arg0: tensor<256x56x56x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x56x56x64xf32>, %arg4: tensor<64xf32>) -> tensor<256x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x56x56x64xf32>
    %7 = stablehlo.convert %6 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x56x56x64xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    return %10 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.6(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x56x56x64xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    return %9 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.7(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<256x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x28x28x128xf32>
    %7 = stablehlo.convert %6 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x28x28x128xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %10 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.8(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x28x28x128xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %9 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.9(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<256x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x28x28x128xf32>
    %7 = stablehlo.convert %6 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x28x28x128xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %10 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.10(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x28x28x128xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %9 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.11(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<256x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x28x28x128xf32>
    %7 = stablehlo.convert %6 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x28x28x128xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %10 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.12(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x28x28x128xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %9 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.13(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<256x28x28x128xf32>, %arg7: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %7 = stablehlo.subtract %arg2, %6 : tensor<256x28x28x128xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<256x28x28x128xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %11 = stablehlo.add %9, %10 : tensor<256x28x28x128xf32>
    %12 = stablehlo.add %5, %11 : tensor<256x28x28x128xf32>
    %13 = stablehlo.convert %12 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %15 = stablehlo.maximum %13, %14 : tensor<256x28x28x128xbf16>
    %16 = stablehlo.convert %15 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %16 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.14(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x28x28x128xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %9 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.15(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x14x14x256xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %9 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.16(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x14x14x256xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %9 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.17(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x14x14x256xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %9 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.18(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x14x14x256xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %9 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.19(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x14x14x256xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %9 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.20(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x14x14x256xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %9 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.21(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256x14x14x256xf32>, %arg7: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %7 = stablehlo.subtract %arg2, %6 : tensor<256x14x14x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<256x14x14x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %11 = stablehlo.add %9, %10 : tensor<256x14x14x256xf32>
    %12 = stablehlo.add %5, %11 : tensor<256x14x14x256xf32>
    %13 = stablehlo.convert %12 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %15 = stablehlo.maximum %13, %14 : tensor<256x14x14x256xbf16>
    %16 = stablehlo.convert %15 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %16 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.22(%arg0: tensor<256x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x14x14x256xf32>
    %7 = stablehlo.convert %6 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x14x14x256xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %10 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.23(%arg0: tensor<256x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x14x14x256xf32>
    %7 = stablehlo.convert %6 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x14x14x256xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %10 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.24(%arg0: tensor<256x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x14x14x256xf32>
    %7 = stablehlo.convert %6 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x14x14x256xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %10 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.25(%arg0: tensor<256x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x14x14x256xf32>
    %7 = stablehlo.convert %6 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x14x14x256xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %10 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.26(%arg0: tensor<256x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x14x14x256xf32>
    %7 = stablehlo.convert %6 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x14x14x256xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %10 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.27(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.convert %5 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x7x7x512xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    return %9 : tensor<256x7x7x512xf32>
  }
  func.func private @fused_convert.28(%arg0: tensor<256x7x7x512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<256x7x7x512xf32>, %arg4: tensor<512xf32>) -> tensor<256x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.add %arg0, %5 : tensor<256x7x7x512xf32>
    %7 = stablehlo.convert %6 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x7x7x512xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    return %10 : tensor<256x7x7x512xf32>
  }
  func.func private @fused_convert.29(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.convert %5 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x7x7x512xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    return %9 : tensor<256x7x7x512xf32>
  }
  func.func private @fused_convert.30(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<256x7x7x512xf32>, %arg7: tensor<512xf32>) -> tensor<256x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %7 = stablehlo.subtract %arg2, %6 : tensor<256x7x7x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<256x7x7x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %11 = stablehlo.add %9, %10 : tensor<256x7x7x512xf32>
    %12 = stablehlo.add %5, %11 : tensor<256x7x7x512xf32>
    %13 = stablehlo.convert %12 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %15 = stablehlo.maximum %13, %14 : tensor<256x7x7x512xbf16>
    %16 = stablehlo.convert %15 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    return %16 : tensor<256x7x7x512xf32>
  }
  func.func private @fused_convert.31(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.convert %5 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %8 = stablehlo.maximum %6, %7 : tensor<256x7x7x512xbf16>
    %9 = stablehlo.convert %8 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    return %9 : tensor<256x7x7x512xf32>
  }
  func.func private @fused_convert.32(%arg0: tensor<1000xf32>, %arg1: tensor<256x1000xbf16>) -> tensor<256x1000xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %2 = stablehlo.add %0, %1 : tensor<256x1000xf32>
    %3 = stablehlo.convert %2 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
    return %3 : tensor<256x1000xbf16>
  }
  func.func private @fused_convert.33(%arg0: tensor<256x512xf32>) -> tensor<256x512xbf16> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x512xf32>
    %2 = stablehlo.convert %1 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    return %2 : tensor<256x512xbf16>
  }
  func.func private @fused_multiply(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.25(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.26(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.27(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.28(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.29(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.30(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.31(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.32(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.33(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.34(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.35(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func @main(%arg0: tensor<i32> {mhlo.parameter_replication = [true]}, %arg1: tensor<1000xf32> {mhlo.parameter_replication = [true]}, %arg2: tensor<512x1000xf32> {mhlo.parameter_replication = [true]}, %arg3: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg4: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg5: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg6: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg7: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg8: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg9: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg10: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg11: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg12: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg13: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg14: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg15: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg16: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg17: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg18: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg19: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg20: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg21: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg22: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg23: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg24: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg25: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg26: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg27: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg28: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg29: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg30: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg31: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg32: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg33: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg34: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg35: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg36: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg37: tensor<3x3x256x512xf32> {mhlo.parameter_replication = [true]}, %arg38: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg39: tensor<1x1x256x512xf32> {mhlo.parameter_replication = [true]}, %arg40: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg41: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg42: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg43: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg44: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg45: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg46: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg47: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg48: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg49: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg50: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg51: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg52: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg53: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg54: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg55: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg56: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg57: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg58: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg59: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg60: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg61: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg62: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg63: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg64: tensor<3x3x64x128xf32> {mhlo.parameter_replication = [true]}, %arg65: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg66: tensor<1x1x64x128xf32> {mhlo.parameter_replication = [true]}, %arg67: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg68: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg69: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg70: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg71: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg72: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg73: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg74: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg75: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg76: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg77: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg78: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg79: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg80: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg81: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg82: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg83: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg84: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg85: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg86: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg87: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg88: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg89: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg90: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg91: tensor<3x3x128x256xf32> {mhlo.parameter_replication = [true]}, %arg92: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg93: tensor<1x1x128x256xf32> {mhlo.parameter_replication = [true]}, %arg94: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg95: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg96: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg97: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg98: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg99: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg100: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg101: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg102: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg103: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg104: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg105: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg106: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg107: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg108: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg109: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg110: tensor<7x7x3x64xf32> {mhlo.parameter_replication = [true]}, %arg111: tensor<1000xf32> {mhlo.parameter_replication = [true]}, %arg112: tensor<512x1000xf32> {mhlo.parameter_replication = [true]}, %arg113: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg114: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg115: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg116: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg117: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg118: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg119: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg120: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg121: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg122: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg123: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg124: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg125: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg126: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg127: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg128: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg129: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg130: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg131: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg132: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg133: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg134: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg135: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg136: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg137: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg138: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg139: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg140: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg141: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg142: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg143: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg144: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg145: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg146: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg147: tensor<3x3x256x512xf32> {mhlo.parameter_replication = [true]}, %arg148: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg149: tensor<1x1x256x512xf32> {mhlo.parameter_replication = [true]}, %arg150: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg151: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg152: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg153: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg154: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg155: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg156: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg157: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg158: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg159: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg160: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg161: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg162: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg163: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg164: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg165: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg166: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg167: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg168: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg169: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg170: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg171: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg172: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg173: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg174: tensor<3x3x64x128xf32> {mhlo.parameter_replication = [true]}, %arg175: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg176: tensor<1x1x64x128xf32> {mhlo.parameter_replication = [true]}, %arg177: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg178: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg179: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg180: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg181: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg182: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg183: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg184: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg185: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg186: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg187: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg188: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg189: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg190: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg191: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg192: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg193: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg194: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg195: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg196: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg197: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg198: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg199: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg200: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg201: tensor<3x3x128x256xf32> {mhlo.parameter_replication = [true]}, %arg202: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg203: tensor<1x1x128x256xf32> {mhlo.parameter_replication = [true]}, %arg204: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg205: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg206: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg207: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg208: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg209: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg210: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg211: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg212: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg213: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg214: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg215: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg216: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg217: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg218: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg219: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg220: tensor<7x7x3x64xf32> {mhlo.parameter_replication = [true]}, %arg221: tensor<i32> {mhlo.parameter_replication = [true]}, %arg222: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg223: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg224: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg225: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg226: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg227: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg228: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg229: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg230: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg231: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg232: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg233: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg234: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg235: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg236: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg237: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg238: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg239: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg240: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg241: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg242: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg243: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg244: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg245: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg246: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg247: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg248: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg249: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg250: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg251: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg252: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg253: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg254: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg255: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg256: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg257: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg258: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg259: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg260: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg261: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg262: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg263: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg264: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg265: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg266: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg267: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg268: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg269: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg270: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg271: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg272: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg273: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg274: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg275: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg276: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg277: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg278: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg279: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg280: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg281: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg282: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg283: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg284: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg285: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg286: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg287: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg288: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg289: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg290: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg291: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg292: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg293: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg294: tensor<i32> {mhlo.parameter_replication = [true]}, %arg295: tensor<f32> {mhlo.parameter_replication = [true]}, %arg296: tensor<1x256x224x224x3xbf16>, %arg297: tensor<1x256xi32>) -> tensor<1x256x1000xbf16> {
    %0 = "mhlo.fusion"(%arg36, %arg245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<512xf32>
      %112 = stablehlo.rsqrt %111 : tensor<512xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<512xf32>
      mhlo.return %113 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %1 = "mhlo.fusion"(%arg34, %arg243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<512xf32>
      %112 = stablehlo.rsqrt %111 : tensor<512xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<512xf32>
      mhlo.return %113 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %2 = "mhlo.fusion"(%arg90, %arg281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %3 = "mhlo.fusion"(%arg88, %arg279) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4 = "mhlo.fusion"(%arg63, %arg263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<128xf32>
      %112 = stablehlo.rsqrt %111 : tensor<128xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<128xf32>
      mhlo.return %113 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %5 = "mhlo.fusion"(%arg61, %arg261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<128xf32>
      %112 = stablehlo.rsqrt %111 : tensor<128xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<128xf32>
      mhlo.return %113 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %6 = "mhlo.fusion"(%arg109, %arg293) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<64xf32>
      %112 = stablehlo.rsqrt %111 : tensor<64xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<64xf32>
      mhlo.return %113 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7 = "mhlo.fusion"(%arg296) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<1x256x224x224x3xbf16>):
      %110 = mhlo.bitcast %arg298 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
      %111 = stablehlo.convert %110 : (tensor<256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
      mhlo.return %111 : tensor<256x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
    %8 = stablehlo.convolution(%7, %arg110) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<256x112x112x64xf32>
    %9 = "mhlo.fusion"(%arg108, %6, %8, %arg292) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>, %arg300: tensor<256x112x112x64xf32>, %arg301: tensor<64xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x112x112x64xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x112x112x64xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x112x112x64xf32>
      %116 = stablehlo.convert %115 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x112x112x64xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
      %120 = stablehlo.convert %cst_0 : (tensor<bf16>) -> tensor<f32>
      %121 = "stablehlo.reduce_window"(%119, %120) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg302: tensor<f32>, %arg303: tensor<f32>):
        %122 = stablehlo.convert %arg302 : (tensor<f32>) -> tensor<bf16>
        %123 = stablehlo.convert %arg303 : (tensor<f32>) -> tensor<bf16>
        %124 = stablehlo.maximum %122, %123 : tensor<bf16>
        %125 = stablehlo.convert %124 : (tensor<bf16>) -> tensor<f32>
        stablehlo.return %125 : tensor<f32>
      }) : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<256x56x56x64xf32>
      mhlo.return %121 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x112x112x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %10 = "mhlo.fusion"(%arg6, %arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<64xf32>
      %112 = stablehlo.rsqrt %111 : tensor<64xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<64xf32>
      mhlo.return %113 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %11 = "mhlo.fusion"(%arg4, %arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<64xf32>
      %112 = stablehlo.rsqrt %111 : tensor<64xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<64xf32>
      mhlo.return %113 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %12 = stablehlo.convolution(%9, %arg7) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %13 = "mhlo.fusion"(%arg3, %11, %12, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>, %arg300: tensor<256x56x56x64xf32>, %arg301: tensor<64xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x56x56x64xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x56x56x64xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x56x56x64xf32>
      %116 = stablehlo.convert %115 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x56x56x64xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %119 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %14 = stablehlo.convolution(%13, %arg8) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %15 = "mhlo.fusion"(%9, %arg5, %10, %14, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x56x56x64xf32>, %arg299: tensor<64xf32>, %arg300: tensor<64xf32>, %arg301: tensor<256x56x56x64xf32>, %arg302: tensor<64xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x56x56x64xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x56x56x64xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x56x56x64xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x56x56x64xf32>
      %117 = stablehlo.convert %116 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x56x56x64xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %120 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %16 = "mhlo.fusion"(%arg12, %arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<64xf32>
      %112 = stablehlo.rsqrt %111 : tensor<64xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<64xf32>
      mhlo.return %113 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %17 = "mhlo.fusion"(%arg10, %arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<64xf32>
      %112 = stablehlo.rsqrt %111 : tensor<64xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<64xf32>
      mhlo.return %113 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %18 = stablehlo.convolution(%15, %arg13) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %19 = "mhlo.fusion"(%arg9, %17, %18, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>, %arg300: tensor<256x56x56x64xf32>, %arg301: tensor<64xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x56x56x64xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x56x56x64xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x56x56x64xf32>
      %116 = stablehlo.convert %115 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x56x56x64xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %119 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %20 = stablehlo.convolution(%19, %arg14) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %21 = "mhlo.fusion"(%15, %arg11, %16, %20, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x56x56x64xf32>, %arg299: tensor<64xf32>, %arg300: tensor<64xf32>, %arg301: tensor<256x56x56x64xf32>, %arg302: tensor<64xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x56x56x64xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x56x56x64xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x56x56x64xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x56x56x64xf32>
      %117 = stablehlo.convert %116 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x56x56x64xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %120 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %22 = "mhlo.fusion"(%arg57, %arg259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<64xf32>
      %112 = stablehlo.rsqrt %111 : tensor<64xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<64xf32>
      mhlo.return %113 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %23 = "mhlo.fusion"(%arg55, %arg257) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<64xf32>
      %112 = stablehlo.rsqrt %111 : tensor<64xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<64xf32>
      mhlo.return %113 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %24 = stablehlo.convolution(%21, %arg58) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %25 = "mhlo.fusion"(%arg54, %23, %24, %arg256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<64xf32>, %arg299: tensor<64xf32>, %arg300: tensor<256x56x56x64xf32>, %arg301: tensor<64xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x56x56x64xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x56x56x64xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x56x56x64xf32>
      %116 = stablehlo.convert %115 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x56x56x64xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %119 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %26 = stablehlo.convolution(%25, %arg59) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %27 = "mhlo.fusion"(%21, %arg56, %22, %26, %arg258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x56x56x64xf32>, %arg299: tensor<64xf32>, %arg300: tensor<64xf32>, %arg301: tensor<256x56x56x64xf32>, %arg302: tensor<64xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x56x56x64xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x56x56x64xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x56x56x64xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x56x56x64xf32>
      %117 = stablehlo.convert %116 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x56x56x64xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %120 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %28 = stablehlo.convolution(%27, %arg64) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x128xf32>) -> tensor<256x28x28x128xf32>
    %29 = "mhlo.fusion"(%arg60, %5, %28, %arg260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<256x28x28x128xf32>, %arg301: tensor<128xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x28x28x128xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x28x28x128xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x28x28x128xf32>
      %116 = stablehlo.convert %115 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x28x28x128xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %119 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %30 = stablehlo.convolution(%29, %arg65) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %31 = "mhlo.fusion"(%arg68, %arg265) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<128xf32>
      %112 = stablehlo.rsqrt %111 : tensor<128xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<128xf32>
      mhlo.return %113 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %32 = stablehlo.convolution(%27, %arg66) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<1x1x64x128xf32>) -> tensor<256x28x28x128xf32>
    %33 = "mhlo.fusion"(%arg62, %4, %30, %arg262, %arg67, %31, %32, %arg264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<256x28x28x128xf32>, %arg301: tensor<128xf32>, %arg302: tensor<128xf32>, %arg303: tensor<128xf32>, %arg304: tensor<256x28x28x128xf32>, %arg305: tensor<128xf32>):
      %110 = stablehlo.broadcast_in_dim %arg305, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %111 = stablehlo.subtract %arg304, %110 : tensor<256x28x28x128xf32>
      %112 = stablehlo.broadcast_in_dim %arg303, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x28x28x128xf32>
      %114 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x28x28x128xf32>
      %116 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %117 = stablehlo.subtract %arg300, %116 : tensor<256x28x28x128xf32>
      %118 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %119 = stablehlo.multiply %117, %118 : tensor<256x28x28x128xf32>
      %120 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %121 = stablehlo.add %119, %120 : tensor<256x28x28x128xf32>
      %122 = stablehlo.add %115, %121 : tensor<256x28x28x128xf32>
      %123 = stablehlo.convert %122 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %124 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %125 = stablehlo.maximum %123, %124 : tensor<256x28x28x128xbf16>
      %126 = stablehlo.convert %125 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %126 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %34 = "mhlo.fusion"(%arg72, %arg269) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<128xf32>
      %112 = stablehlo.rsqrt %111 : tensor<128xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<128xf32>
      mhlo.return %113 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %35 = "mhlo.fusion"(%arg70, %arg267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<128xf32>
      %112 = stablehlo.rsqrt %111 : tensor<128xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<128xf32>
      mhlo.return %113 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %36 = stablehlo.convolution(%33, %arg73) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %37 = "mhlo.fusion"(%arg69, %35, %36, %arg266) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<256x28x28x128xf32>, %arg301: tensor<128xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x28x28x128xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x28x28x128xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x28x28x128xf32>
      %116 = stablehlo.convert %115 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x28x28x128xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %119 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %38 = stablehlo.convolution(%37, %arg74) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %39 = "mhlo.fusion"(%33, %arg71, %34, %38, %arg268) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x28x28x128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<128xf32>, %arg301: tensor<256x28x28x128xf32>, %arg302: tensor<128xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x28x28x128xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x28x28x128xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x28x28x128xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x28x28x128xf32>
      %117 = stablehlo.convert %116 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x28x28x128xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %120 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %40 = "mhlo.fusion"(%arg78, %arg273) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<128xf32>
      %112 = stablehlo.rsqrt %111 : tensor<128xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<128xf32>
      mhlo.return %113 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %41 = "mhlo.fusion"(%arg76, %arg271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<128xf32>
      %112 = stablehlo.rsqrt %111 : tensor<128xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<128xf32>
      mhlo.return %113 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %42 = stablehlo.convolution(%39, %arg79) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %43 = "mhlo.fusion"(%arg75, %41, %42, %arg270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<256x28x28x128xf32>, %arg301: tensor<128xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x28x28x128xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x28x28x128xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x28x28x128xf32>
      %116 = stablehlo.convert %115 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x28x28x128xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %119 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %44 = stablehlo.convolution(%43, %arg80) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %45 = "mhlo.fusion"(%39, %arg77, %40, %44, %arg272) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x28x28x128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<128xf32>, %arg301: tensor<256x28x28x128xf32>, %arg302: tensor<128xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x28x28x128xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x28x28x128xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x28x28x128xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x28x28x128xf32>
      %117 = stablehlo.convert %116 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x28x28x128xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %120 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %46 = "mhlo.fusion"(%arg84, %arg277) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<128xf32>
      %112 = stablehlo.rsqrt %111 : tensor<128xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<128xf32>
      mhlo.return %113 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %47 = "mhlo.fusion"(%arg82, %arg275) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<128xf32>
      %112 = stablehlo.rsqrt %111 : tensor<128xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<128xf32>
      mhlo.return %113 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %48 = stablehlo.convolution(%45, %arg85) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %49 = "mhlo.fusion"(%arg81, %47, %48, %arg274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<256x28x28x128xf32>, %arg301: tensor<128xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x28x28x128xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x28x28x128xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x28x28x128xf32>
      %116 = stablehlo.convert %115 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x28x28x128xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %119 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %50 = stablehlo.convolution(%49, %arg86) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %51 = "mhlo.fusion"(%45, %arg83, %46, %50, %arg276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x28x28x128xf32>, %arg299: tensor<128xf32>, %arg300: tensor<128xf32>, %arg301: tensor<256x28x28x128xf32>, %arg302: tensor<128xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x28x28x128xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x28x28x128xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x28x28x128xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x28x28x128xf32>
      %117 = stablehlo.convert %116 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x28x28x128xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %120 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %52 = stablehlo.convolution(%51, %arg91) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x256xf32>) -> tensor<256x14x14x256xf32>
    %53 = "mhlo.fusion"(%arg87, %3, %52, %arg278) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256x14x14x256xf32>, %arg301: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.convert %115 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x14x14x256xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %119 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %54 = stablehlo.convolution(%53, %arg92) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %55 = "mhlo.fusion"(%arg95, %arg283) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %56 = stablehlo.convolution(%51, %arg93) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<1x1x128x256xf32>) -> tensor<256x14x14x256xf32>
    %57 = "mhlo.fusion"(%arg89, %2, %54, %arg280, %arg94, %55, %56, %arg282) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256x14x14x256xf32>, %arg301: tensor<256xf32>, %arg302: tensor<256xf32>, %arg303: tensor<256xf32>, %arg304: tensor<256x14x14x256xf32>, %arg305: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg305, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg304, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg303, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %117 = stablehlo.subtract %arg300, %116 : tensor<256x14x14x256xf32>
      %118 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %119 = stablehlo.multiply %117, %118 : tensor<256x14x14x256xf32>
      %120 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %121 = stablehlo.add %119, %120 : tensor<256x14x14x256xf32>
      %122 = stablehlo.add %115, %121 : tensor<256x14x14x256xf32>
      %123 = stablehlo.convert %122 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %124 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %125 = stablehlo.maximum %123, %124 : tensor<256x14x14x256xbf16>
      %126 = stablehlo.convert %125 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %126 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %58 = "mhlo.fusion"(%arg99, %arg287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %59 = "mhlo.fusion"(%arg97, %arg285) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %60 = stablehlo.convolution(%57, %arg100) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %61 = "mhlo.fusion"(%arg96, %59, %60, %arg284) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256x14x14x256xf32>, %arg301: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.convert %115 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x14x14x256xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %119 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %62 = stablehlo.convolution(%61, %arg101) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %63 = "mhlo.fusion"(%57, %arg98, %58, %62, %arg286) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x14x14x256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256xf32>, %arg301: tensor<256x14x14x256xf32>, %arg302: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x14x14x256xf32>
      %117 = stablehlo.convert %116 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x14x14x256xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %120 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %64 = "mhlo.fusion"(%arg105, %arg291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %65 = "mhlo.fusion"(%arg103, %arg289) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %66 = stablehlo.convolution(%63, %arg106) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %67 = "mhlo.fusion"(%arg102, %65, %66, %arg288) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256x14x14x256xf32>, %arg301: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.convert %115 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x14x14x256xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %119 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %68 = stablehlo.convolution(%67, %arg107) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %69 = "mhlo.fusion"(%63, %arg104, %64, %68, %arg290) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x14x14x256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256xf32>, %arg301: tensor<256x14x14x256xf32>, %arg302: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x14x14x256xf32>
      %117 = stablehlo.convert %116 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x14x14x256xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %120 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %70 = "mhlo.fusion"(%arg18, %arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %71 = "mhlo.fusion"(%arg16, %arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %72 = stablehlo.convolution(%69, %arg19) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %73 = "mhlo.fusion"(%arg15, %71, %72, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256x14x14x256xf32>, %arg301: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.convert %115 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x14x14x256xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %119 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %74 = stablehlo.convolution(%73, %arg20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %75 = "mhlo.fusion"(%69, %arg17, %70, %74, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x14x14x256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256xf32>, %arg301: tensor<256x14x14x256xf32>, %arg302: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x14x14x256xf32>
      %117 = stablehlo.convert %116 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x14x14x256xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %120 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %76 = "mhlo.fusion"(%arg24, %arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %77 = "mhlo.fusion"(%arg22, %arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %78 = stablehlo.convolution(%75, %arg25) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %79 = "mhlo.fusion"(%arg21, %77, %78, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256x14x14x256xf32>, %arg301: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.convert %115 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x14x14x256xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %119 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %80 = stablehlo.convolution(%79, %arg26) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %81 = "mhlo.fusion"(%75, %arg23, %76, %80, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x14x14x256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256xf32>, %arg301: tensor<256x14x14x256xf32>, %arg302: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x14x14x256xf32>
      %117 = stablehlo.convert %116 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x14x14x256xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %120 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %82 = "mhlo.fusion"(%arg30, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %83 = "mhlo.fusion"(%arg28, %arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<256xf32>
      %112 = stablehlo.rsqrt %111 : tensor<256xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<256xf32>
      mhlo.return %113 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %84 = stablehlo.convolution(%81, %arg31) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %85 = "mhlo.fusion"(%arg27, %83, %84, %arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256x14x14x256xf32>, %arg301: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.convert %115 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x14x14x256xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %119 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %86 = stablehlo.convolution(%85, %arg32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %87 = "mhlo.fusion"(%81, %arg29, %82, %86, %arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x14x14x256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256xf32>, %arg301: tensor<256x14x14x256xf32>, %arg302: tensor<256xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x14x14x256xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x14x14x256xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x14x14x256xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x14x14x256xf32>
      %117 = stablehlo.convert %116 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x14x14x256xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %120 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %88 = stablehlo.convolution(%87, %arg37) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x512xf32>) -> tensor<256x7x7x512xf32>
    %89 = "mhlo.fusion"(%arg33, %1, %88, %arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>, %arg300: tensor<256x7x7x512xf32>, %arg301: tensor<512xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x7x7x512xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x7x7x512xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x7x7x512xf32>
      %116 = stablehlo.convert %115 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x7x7x512xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %119 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %90 = stablehlo.convolution(%89, %arg38) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %91 = "mhlo.fusion"(%arg41, %arg247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<512xf32>
      %112 = stablehlo.rsqrt %111 : tensor<512xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<512xf32>
      mhlo.return %113 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %92 = stablehlo.convolution(%87, %arg39) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<1x1x256x512xf32>) -> tensor<256x7x7x512xf32>
    %93 = "mhlo.fusion"(%arg35, %0, %90, %arg244, %arg40, %91, %92, %arg246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>, %arg300: tensor<256x7x7x512xf32>, %arg301: tensor<512xf32>, %arg302: tensor<512xf32>, %arg303: tensor<512xf32>, %arg304: tensor<256x7x7x512xf32>, %arg305: tensor<512xf32>):
      %110 = stablehlo.broadcast_in_dim %arg305, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %111 = stablehlo.subtract %arg304, %110 : tensor<256x7x7x512xf32>
      %112 = stablehlo.broadcast_in_dim %arg303, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x7x7x512xf32>
      %114 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x7x7x512xf32>
      %116 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %117 = stablehlo.subtract %arg300, %116 : tensor<256x7x7x512xf32>
      %118 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %119 = stablehlo.multiply %117, %118 : tensor<256x7x7x512xf32>
      %120 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %121 = stablehlo.add %119, %120 : tensor<256x7x7x512xf32>
      %122 = stablehlo.add %115, %121 : tensor<256x7x7x512xf32>
      %123 = stablehlo.convert %122 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %124 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %125 = stablehlo.maximum %123, %124 : tensor<256x7x7x512xbf16>
      %126 = stablehlo.convert %125 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %126 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %94 = "mhlo.fusion"(%arg45, %arg251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<512xf32>
      %112 = stablehlo.rsqrt %111 : tensor<512xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<512xf32>
      mhlo.return %113 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %95 = "mhlo.fusion"(%arg43, %arg249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<512xf32>
      %112 = stablehlo.rsqrt %111 : tensor<512xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<512xf32>
      mhlo.return %113 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %96 = stablehlo.convolution(%93, %arg46) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %97 = "mhlo.fusion"(%arg42, %95, %96, %arg248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>, %arg300: tensor<256x7x7x512xf32>, %arg301: tensor<512xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x7x7x512xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x7x7x512xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x7x7x512xf32>
      %116 = stablehlo.convert %115 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x7x7x512xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %119 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %98 = stablehlo.convolution(%97, %arg47) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %99 = "mhlo.fusion"(%93, %arg44, %94, %98, %arg250) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x7x7x512xf32>, %arg299: tensor<512xf32>, %arg300: tensor<512xf32>, %arg301: tensor<256x7x7x512xf32>, %arg302: tensor<512xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x7x7x512xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x7x7x512xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x7x7x512xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x7x7x512xf32>
      %117 = stablehlo.convert %116 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x7x7x512xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %120 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<256x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %100 = "mhlo.fusion"(%arg51, %arg255) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<512xf32>
      %112 = stablehlo.rsqrt %111 : tensor<512xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<512xf32>
      mhlo.return %113 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %101 = "mhlo.fusion"(%arg49, %arg253) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %111 = stablehlo.add %arg299, %110 : tensor<512xf32>
      %112 = stablehlo.rsqrt %111 : tensor<512xf32>
      %113 = stablehlo.multiply %112, %arg298 : tensor<512xf32>
      mhlo.return %113 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %102 = stablehlo.convolution(%99, %arg52) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %103 = "mhlo.fusion"(%arg48, %101, %102, %arg252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<512xf32>, %arg299: tensor<512xf32>, %arg300: tensor<256x7x7x512xf32>, %arg301: tensor<512xf32>):
      %110 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %111 = stablehlo.subtract %arg300, %110 : tensor<256x7x7x512xf32>
      %112 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x7x7x512xf32>
      %114 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x7x7x512xf32>
      %116 = stablehlo.convert %115 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %117 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %118 = stablehlo.maximum %116, %117 : tensor<256x7x7x512xbf16>
      %119 = stablehlo.convert %118 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %119 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %104 = stablehlo.convolution(%103, %arg53) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %105 = "mhlo.fusion"(%99, %arg50, %100, %104, %arg254) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg298: tensor<256x7x7x512xf32>, %arg299: tensor<512xf32>, %arg300: tensor<512xf32>, %arg301: tensor<256x7x7x512xf32>, %arg302: tensor<512xf32>):
      %110 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %111 = stablehlo.subtract %arg301, %110 : tensor<256x7x7x512xf32>
      %112 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<256x7x7x512xf32>
      %114 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %115 = stablehlo.add %113, %114 : tensor<256x7x7x512xf32>
      %116 = stablehlo.add %arg298, %115 : tensor<256x7x7x512xf32>
      %117 = stablehlo.convert %116 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %118 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %119 = stablehlo.maximum %117, %118 : tensor<256x7x7x512xbf16>
      %120 = stablehlo.convert %119 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      %121 = mhlo.bitcast %120 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x512xf32>) -> tensor<256x49x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %122 = stablehlo.reduce(%121 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x49x512xf32>, tensor<f32>) -> tensor<256x512xf32>
      mhlo.return %122 : tensor<256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<256x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x512xf32>
    %106 = "mhlo.fusion"(%105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<256x512xf32>):
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
      %111 = stablehlo.multiply %arg298, %110 : tensor<256x512xf32>
      %112 = stablehlo.convert %111 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
      mhlo.return %112 : tensor<256x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %107 = "mhlo.fusion"(%106, %arg2) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg298: tensor<256x512xbf16>, %arg299: tensor<512x1000xf32>):
      %110 = stablehlo.convert %arg299 : (tensor<512x1000xf32>) -> tensor<512x1000xbf16>
      %111 = stablehlo.dot_general %110, %arg298, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[1000,256]{0,1}"} : (tensor<512x1000xbf16>, tensor<256x512xbf16>) -> tensor<1000x256xbf16>
      %112 = mhlo.bitcast %111 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x256xbf16>) -> tensor<256x1000xbf16>
      mhlo.return %112 : tensor<256x1000xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x512xbf16>, tensor<512x1000xf32>) -> tensor<256x1000xbf16>
    %108 = "mhlo.fusion"(%arg1, %107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg298: tensor<1000xf32>, %arg299: tensor<256x1000xbf16>):
      %110 = stablehlo.convert %arg299 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
      %111 = stablehlo.broadcast_in_dim %arg298, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
      %112 = stablehlo.add %110, %111 : tensor<256x1000xf32>
      %113 = stablehlo.convert %112 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
      mhlo.return %113 : tensor<256x1000xbf16>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<256x1000xbf16>) -> tensor<256x1000xbf16>
    %109 = mhlo.bitcast %108 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x1000xbf16>) -> tensor<1x256x1000xbf16>
    return %109 : tensor<1x256x1000xbf16>
  }
}
