module @pmap_inference_step attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {suggested_combiner_threshold = "39016231856"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.num_replicas = 4 : i32, mhlo.use_auto_spmd_partitioning = false} {
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
  func.func private @fused_convert.5(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<256x28x28x128xf32>, %arg4: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
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
  func.func private @fused_convert.6(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
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
  func.func private @fused_convert.7(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<256x28x28x128xf32>, %arg7: tensor<128xf32>) -> tensor<256x28x28x128xf32> {
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
  func.func private @fused_convert.9(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.10(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.11(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256x14x14x256xf32>, %arg7: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.12(%arg0: tensor<256x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x14x14x256xf32>, %arg4: tensor<256xf32>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.13(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xf32> {
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
  func.func private @fused_convert.14(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<256x7x7x512xf32>, %arg7: tensor<512xf32>) -> tensor<256x7x7x512xf32> {
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
  func.func private @fused_convert.15(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xf32> {
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
  func.func private @fused_convert.16(%arg0: tensor<1000xf32>, %arg1: tensor<256x1000xbf16>) -> tensor<256x1000xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %2 = stablehlo.add %0, %1 : tensor<256x1000xf32>
    %3 = stablehlo.convert %2 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
    return %3 : tensor<256x1000xbf16>
  }
  func.func private @fused_convert.17(%arg0: tensor<256x512xf32>) -> tensor<256x512xbf16> {
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
  func.func private @fused_multiply.5(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
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
  func.func private @fused_multiply.10(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func @main(%arg0: tensor<i32> {mhlo.parameter_replication = [true]}, %arg1: tensor<1000xf32> {mhlo.parameter_replication = [true]}, %arg2: tensor<512x1000xf32> {mhlo.parameter_replication = [true]}, %arg3: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg4: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg5: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg6: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg7: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg8: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg9: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg10: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg11: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg12: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg13: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg14: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg15: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg16: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg17: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg18: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg19: tensor<3x3x64x128xf32> {mhlo.parameter_replication = [true]}, %arg20: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg21: tensor<1x1x64x128xf32> {mhlo.parameter_replication = [true]}, %arg22: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg23: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg24: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg25: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg26: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg27: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg28: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg29: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg30: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg31: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg32: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg33: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg34: tensor<3x3x128x256xf32> {mhlo.parameter_replication = [true]}, %arg35: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg36: tensor<1x1x128x256xf32> {mhlo.parameter_replication = [true]}, %arg37: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg38: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg39: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg40: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg41: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg42: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg43: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg44: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg45: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg46: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg47: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg48: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg49: tensor<3x3x256x512xf32> {mhlo.parameter_replication = [true]}, %arg50: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg51: tensor<1x1x256x512xf32> {mhlo.parameter_replication = [true]}, %arg52: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg53: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg54: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg55: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg56: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg57: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg58: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg59: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg60: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg61: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg62: tensor<7x7x3x64xf32> {mhlo.parameter_replication = [true]}, %arg63: tensor<1000xf32> {mhlo.parameter_replication = [true]}, %arg64: tensor<512x1000xf32> {mhlo.parameter_replication = [true]}, %arg65: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg66: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg67: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg68: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg69: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg70: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg71: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg72: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg73: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg74: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg75: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg76: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg77: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg78: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg79: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg80: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg81: tensor<3x3x64x128xf32> {mhlo.parameter_replication = [true]}, %arg82: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg83: tensor<1x1x64x128xf32> {mhlo.parameter_replication = [true]}, %arg84: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg85: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg86: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg87: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg88: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg89: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg90: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg91: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg92: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg93: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg94: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg95: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg96: tensor<3x3x128x256xf32> {mhlo.parameter_replication = [true]}, %arg97: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg98: tensor<1x1x128x256xf32> {mhlo.parameter_replication = [true]}, %arg99: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg100: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg101: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg102: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg103: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg104: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg105: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg106: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg107: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg108: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg109: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg110: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg111: tensor<3x3x256x512xf32> {mhlo.parameter_replication = [true]}, %arg112: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg113: tensor<1x1x256x512xf32> {mhlo.parameter_replication = [true]}, %arg114: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg115: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg116: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg117: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg118: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg119: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg120: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg121: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg122: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg123: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg124: tensor<7x7x3x64xf32> {mhlo.parameter_replication = [true]}, %arg125: tensor<i32> {mhlo.parameter_replication = [true]}, %arg126: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg127: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg128: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg129: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg130: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg131: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg132: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg133: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg134: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg135: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg136: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg137: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg138: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg139: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg140: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg141: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg142: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg143: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg144: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg145: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg146: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg147: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg148: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg149: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg150: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg151: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg152: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg153: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg154: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg155: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg156: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg157: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg158: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg159: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg160: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg161: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg162: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg163: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg164: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg165: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg166: tensor<i32> {mhlo.parameter_replication = [true]}, %arg167: tensor<f32> {mhlo.parameter_replication = [true]}, %arg168: tensor<1x256x224x224x3xbf16>, %arg169: tensor<1x256xi32>) -> tensor<1x256x1000xbf16> {
    %0 = "mhlo.fusion"(%arg48, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<512xf32>, %arg171: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<512xf32>
      %64 = stablehlo.rsqrt %63 : tensor<512xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<512xf32>
      mhlo.return %65 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %1 = "mhlo.fusion"(%arg46, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<512xf32>, %arg171: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<512xf32>
      %64 = stablehlo.rsqrt %63 : tensor<512xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<512xf32>
      mhlo.return %65 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %2 = "mhlo.fusion"(%arg33, %arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256xf32>, %arg171: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<256xf32>
      %64 = stablehlo.rsqrt %63 : tensor<256xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<256xf32>
      mhlo.return %65 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %3 = "mhlo.fusion"(%arg31, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256xf32>, %arg171: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<256xf32>
      %64 = stablehlo.rsqrt %63 : tensor<256xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<256xf32>
      mhlo.return %65 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4 = "mhlo.fusion"(%arg18, %arg137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<128xf32>, %arg171: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<128xf32>
      %64 = stablehlo.rsqrt %63 : tensor<128xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<128xf32>
      mhlo.return %65 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %5 = "mhlo.fusion"(%arg16, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<128xf32>, %arg171: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<128xf32>
      %64 = stablehlo.rsqrt %63 : tensor<128xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<128xf32>
      mhlo.return %65 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %6 = "mhlo.fusion"(%arg61, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<64xf32>, %arg171: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<64xf32>
      %64 = stablehlo.rsqrt %63 : tensor<64xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<64xf32>
      mhlo.return %65 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7 = "mhlo.fusion"(%arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<1x256x224x224x3xbf16>):
      %62 = mhlo.bitcast %arg170 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
      %63 = stablehlo.convert %62 : (tensor<256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
      mhlo.return %63 : tensor<256x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
    %8 = stablehlo.convolution(%7, %arg62) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<256x112x112x64xf32>
    %9 = "mhlo.fusion"(%arg60, %6, %8, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<64xf32>, %arg171: tensor<64xf32>, %arg172: tensor<256x112x112x64xf32>, %arg173: tensor<64xf32>):
      %62 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %63 = stablehlo.subtract %arg172, %62 : tensor<256x112x112x64xf32>
      %64 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x112x112x64xf32>
      %66 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x112x112x64xf32>
      %68 = stablehlo.convert %67 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
      %70 = stablehlo.maximum %68, %69 : tensor<256x112x112x64xbf16>
      %71 = stablehlo.convert %70 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
      %72 = stablehlo.convert %cst_0 : (tensor<bf16>) -> tensor<f32>
      %73 = "stablehlo.reduce_window"(%71, %72) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg174: tensor<f32>, %arg175: tensor<f32>):
        %74 = stablehlo.convert %arg174 : (tensor<f32>) -> tensor<bf16>
        %75 = stablehlo.convert %arg175 : (tensor<f32>) -> tensor<bf16>
        %76 = stablehlo.maximum %74, %75 : tensor<bf16>
        %77 = stablehlo.convert %76 : (tensor<bf16>) -> tensor<f32>
        stablehlo.return %77 : tensor<f32>
      }) : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<256x56x56x64xf32>
      mhlo.return %73 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x112x112x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %10 = "mhlo.fusion"(%arg6, %arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<64xf32>, %arg171: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<64xf32>
      %64 = stablehlo.rsqrt %63 : tensor<64xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<64xf32>
      mhlo.return %65 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %11 = "mhlo.fusion"(%arg4, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<64xf32>, %arg171: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<64xf32>
      %64 = stablehlo.rsqrt %63 : tensor<64xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<64xf32>
      mhlo.return %65 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %12 = stablehlo.convolution(%9, %arg7) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %13 = "mhlo.fusion"(%arg3, %11, %12, %arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<64xf32>, %arg171: tensor<64xf32>, %arg172: tensor<256x56x56x64xf32>, %arg173: tensor<64xf32>):
      %62 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %63 = stablehlo.subtract %arg172, %62 : tensor<256x56x56x64xf32>
      %64 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x56x56x64xf32>
      %66 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x56x56x64xf32>
      %68 = stablehlo.convert %67 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %70 = stablehlo.maximum %68, %69 : tensor<256x56x56x64xbf16>
      %71 = stablehlo.convert %70 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %71 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %14 = stablehlo.convolution(%13, %arg8) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %15 = "mhlo.fusion"(%9, %arg5, %10, %14, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256x56x56x64xf32>, %arg171: tensor<64xf32>, %arg172: tensor<64xf32>, %arg173: tensor<256x56x56x64xf32>, %arg174: tensor<64xf32>):
      %62 = stablehlo.broadcast_in_dim %arg174, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %63 = stablehlo.subtract %arg173, %62 : tensor<256x56x56x64xf32>
      %64 = stablehlo.broadcast_in_dim %arg172, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x56x56x64xf32>
      %66 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x56x56x64xf32>
      %68 = stablehlo.add %arg170, %67 : tensor<256x56x56x64xf32>
      %69 = stablehlo.convert %68 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %70 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %71 = stablehlo.maximum %69, %70 : tensor<256x56x56x64xbf16>
      %72 = stablehlo.convert %71 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %72 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %16 = "mhlo.fusion"(%arg12, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<64xf32>, %arg171: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<64xf32>
      %64 = stablehlo.rsqrt %63 : tensor<64xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<64xf32>
      mhlo.return %65 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %17 = "mhlo.fusion"(%arg10, %arg131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<64xf32>, %arg171: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<64xf32>
      %64 = stablehlo.rsqrt %63 : tensor<64xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<64xf32>
      mhlo.return %65 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %18 = stablehlo.convolution(%15, %arg13) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %19 = "mhlo.fusion"(%arg9, %17, %18, %arg130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<64xf32>, %arg171: tensor<64xf32>, %arg172: tensor<256x56x56x64xf32>, %arg173: tensor<64xf32>):
      %62 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %63 = stablehlo.subtract %arg172, %62 : tensor<256x56x56x64xf32>
      %64 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x56x56x64xf32>
      %66 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x56x56x64xf32>
      %68 = stablehlo.convert %67 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %70 = stablehlo.maximum %68, %69 : tensor<256x56x56x64xbf16>
      %71 = stablehlo.convert %70 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %71 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %20 = stablehlo.convolution(%19, %arg14) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %21 = "mhlo.fusion"(%15, %arg11, %16, %20, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256x56x56x64xf32>, %arg171: tensor<64xf32>, %arg172: tensor<64xf32>, %arg173: tensor<256x56x56x64xf32>, %arg174: tensor<64xf32>):
      %62 = stablehlo.broadcast_in_dim %arg174, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %63 = stablehlo.subtract %arg173, %62 : tensor<256x56x56x64xf32>
      %64 = stablehlo.broadcast_in_dim %arg172, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x56x56x64xf32>
      %66 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x56x56x64xf32>
      %68 = stablehlo.add %arg170, %67 : tensor<256x56x56x64xf32>
      %69 = stablehlo.convert %68 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %70 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %71 = stablehlo.maximum %69, %70 : tensor<256x56x56x64xbf16>
      %72 = stablehlo.convert %71 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %72 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %22 = stablehlo.convolution(%21, %arg19) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x128xf32>) -> tensor<256x28x28x128xf32>
    %23 = "mhlo.fusion"(%arg15, %5, %22, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<128xf32>, %arg171: tensor<128xf32>, %arg172: tensor<256x28x28x128xf32>, %arg173: tensor<128xf32>):
      %62 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %63 = stablehlo.subtract %arg172, %62 : tensor<256x28x28x128xf32>
      %64 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x28x28x128xf32>
      %66 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x28x28x128xf32>
      %68 = stablehlo.convert %67 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %70 = stablehlo.maximum %68, %69 : tensor<256x28x28x128xbf16>
      %71 = stablehlo.convert %70 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %71 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %24 = stablehlo.convolution(%23, %arg20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %25 = "mhlo.fusion"(%arg23, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<128xf32>, %arg171: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<128xf32>
      %64 = stablehlo.rsqrt %63 : tensor<128xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<128xf32>
      mhlo.return %65 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %26 = stablehlo.convolution(%21, %arg21) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<1x1x64x128xf32>) -> tensor<256x28x28x128xf32>
    %27 = "mhlo.fusion"(%arg17, %4, %24, %arg136, %arg22, %25, %26, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<128xf32>, %arg171: tensor<128xf32>, %arg172: tensor<256x28x28x128xf32>, %arg173: tensor<128xf32>, %arg174: tensor<128xf32>, %arg175: tensor<128xf32>, %arg176: tensor<256x28x28x128xf32>, %arg177: tensor<128xf32>):
      %62 = stablehlo.broadcast_in_dim %arg177, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %63 = stablehlo.subtract %arg176, %62 : tensor<256x28x28x128xf32>
      %64 = stablehlo.broadcast_in_dim %arg175, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x28x28x128xf32>
      %66 = stablehlo.broadcast_in_dim %arg174, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x28x28x128xf32>
      %68 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %69 = stablehlo.subtract %arg172, %68 : tensor<256x28x28x128xf32>
      %70 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %71 = stablehlo.multiply %69, %70 : tensor<256x28x28x128xf32>
      %72 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %73 = stablehlo.add %71, %72 : tensor<256x28x28x128xf32>
      %74 = stablehlo.add %67, %73 : tensor<256x28x28x128xf32>
      %75 = stablehlo.convert %74 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %76 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %77 = stablehlo.maximum %75, %76 : tensor<256x28x28x128xbf16>
      %78 = stablehlo.convert %77 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %78 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %28 = "mhlo.fusion"(%arg27, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<128xf32>, %arg171: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<128xf32>
      %64 = stablehlo.rsqrt %63 : tensor<128xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<128xf32>
      mhlo.return %65 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %29 = "mhlo.fusion"(%arg25, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<128xf32>, %arg171: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<128xf32>
      %64 = stablehlo.rsqrt %63 : tensor<128xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<128xf32>
      mhlo.return %65 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %30 = stablehlo.convolution(%27, %arg28) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %31 = "mhlo.fusion"(%arg24, %29, %30, %arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<128xf32>, %arg171: tensor<128xf32>, %arg172: tensor<256x28x28x128xf32>, %arg173: tensor<128xf32>):
      %62 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %63 = stablehlo.subtract %arg172, %62 : tensor<256x28x28x128xf32>
      %64 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x28x28x128xf32>
      %66 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x28x28x128xf32>
      %68 = stablehlo.convert %67 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %70 = stablehlo.maximum %68, %69 : tensor<256x28x28x128xbf16>
      %71 = stablehlo.convert %70 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %71 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %32 = stablehlo.convolution(%31, %arg29) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %33 = "mhlo.fusion"(%27, %arg26, %28, %32, %arg142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256x28x28x128xf32>, %arg171: tensor<128xf32>, %arg172: tensor<128xf32>, %arg173: tensor<256x28x28x128xf32>, %arg174: tensor<128xf32>):
      %62 = stablehlo.broadcast_in_dim %arg174, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %63 = stablehlo.subtract %arg173, %62 : tensor<256x28x28x128xf32>
      %64 = stablehlo.broadcast_in_dim %arg172, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x28x28x128xf32>
      %66 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x28x28x128xf32>
      %68 = stablehlo.add %arg170, %67 : tensor<256x28x28x128xf32>
      %69 = stablehlo.convert %68 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %70 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %71 = stablehlo.maximum %69, %70 : tensor<256x28x28x128xbf16>
      %72 = stablehlo.convert %71 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %72 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<256x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %34 = stablehlo.convolution(%33, %arg34) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x256xf32>) -> tensor<256x14x14x256xf32>
    %35 = "mhlo.fusion"(%arg30, %3, %34, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256x14x14x256xf32>, %arg173: tensor<256xf32>):
      %62 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %63 = stablehlo.subtract %arg172, %62 : tensor<256x14x14x256xf32>
      %64 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x14x14x256xf32>
      %66 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x14x14x256xf32>
      %68 = stablehlo.convert %67 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %70 = stablehlo.maximum %68, %69 : tensor<256x14x14x256xbf16>
      %71 = stablehlo.convert %70 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %71 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %36 = stablehlo.convolution(%35, %arg35) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %37 = "mhlo.fusion"(%arg38, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256xf32>, %arg171: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<256xf32>
      %64 = stablehlo.rsqrt %63 : tensor<256xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<256xf32>
      mhlo.return %65 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %38 = stablehlo.convolution(%33, %arg36) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<1x1x128x256xf32>) -> tensor<256x14x14x256xf32>
    %39 = "mhlo.fusion"(%arg32, %2, %36, %arg146, %arg37, %37, %38, %arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256x14x14x256xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256x14x14x256xf32>, %arg177: tensor<256xf32>):
      %62 = stablehlo.broadcast_in_dim %arg177, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %63 = stablehlo.subtract %arg176, %62 : tensor<256x14x14x256xf32>
      %64 = stablehlo.broadcast_in_dim %arg175, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x14x14x256xf32>
      %66 = stablehlo.broadcast_in_dim %arg174, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x14x14x256xf32>
      %68 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %69 = stablehlo.subtract %arg172, %68 : tensor<256x14x14x256xf32>
      %70 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %71 = stablehlo.multiply %69, %70 : tensor<256x14x14x256xf32>
      %72 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %73 = stablehlo.add %71, %72 : tensor<256x14x14x256xf32>
      %74 = stablehlo.add %67, %73 : tensor<256x14x14x256xf32>
      %75 = stablehlo.convert %74 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %76 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %77 = stablehlo.maximum %75, %76 : tensor<256x14x14x256xbf16>
      %78 = stablehlo.convert %77 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %78 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %40 = "mhlo.fusion"(%arg42, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256xf32>, %arg171: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<256xf32>
      %64 = stablehlo.rsqrt %63 : tensor<256xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<256xf32>
      mhlo.return %65 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %41 = "mhlo.fusion"(%arg40, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256xf32>, %arg171: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<256xf32>
      %64 = stablehlo.rsqrt %63 : tensor<256xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<256xf32>
      mhlo.return %65 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %42 = stablehlo.convolution(%39, %arg43) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %43 = "mhlo.fusion"(%arg39, %41, %42, %arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256x14x14x256xf32>, %arg173: tensor<256xf32>):
      %62 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %63 = stablehlo.subtract %arg172, %62 : tensor<256x14x14x256xf32>
      %64 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x14x14x256xf32>
      %66 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x14x14x256xf32>
      %68 = stablehlo.convert %67 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %70 = stablehlo.maximum %68, %69 : tensor<256x14x14x256xbf16>
      %71 = stablehlo.convert %70 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %71 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %44 = stablehlo.convolution(%43, %arg44) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %45 = "mhlo.fusion"(%39, %arg41, %40, %44, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256x14x14x256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256xf32>, %arg173: tensor<256x14x14x256xf32>, %arg174: tensor<256xf32>):
      %62 = stablehlo.broadcast_in_dim %arg174, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %63 = stablehlo.subtract %arg173, %62 : tensor<256x14x14x256xf32>
      %64 = stablehlo.broadcast_in_dim %arg172, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x14x14x256xf32>
      %66 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x14x14x256xf32>
      %68 = stablehlo.add %arg170, %67 : tensor<256x14x14x256xf32>
      %69 = stablehlo.convert %68 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %70 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %71 = stablehlo.maximum %69, %70 : tensor<256x14x14x256xbf16>
      %72 = stablehlo.convert %71 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %72 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %46 = stablehlo.convolution(%45, %arg49) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x512xf32>) -> tensor<256x7x7x512xf32>
    %47 = "mhlo.fusion"(%arg45, %1, %46, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<512xf32>, %arg171: tensor<512xf32>, %arg172: tensor<256x7x7x512xf32>, %arg173: tensor<512xf32>):
      %62 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %63 = stablehlo.subtract %arg172, %62 : tensor<256x7x7x512xf32>
      %64 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x7x7x512xf32>
      %66 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x7x7x512xf32>
      %68 = stablehlo.convert %67 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %70 = stablehlo.maximum %68, %69 : tensor<256x7x7x512xbf16>
      %71 = stablehlo.convert %70 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %71 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %48 = stablehlo.convolution(%47, %arg50) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %49 = "mhlo.fusion"(%arg53, %arg159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<512xf32>, %arg171: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<512xf32>
      %64 = stablehlo.rsqrt %63 : tensor<512xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<512xf32>
      mhlo.return %65 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %50 = stablehlo.convolution(%45, %arg51) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<1x1x256x512xf32>) -> tensor<256x7x7x512xf32>
    %51 = "mhlo.fusion"(%arg47, %0, %48, %arg156, %arg52, %49, %50, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<512xf32>, %arg171: tensor<512xf32>, %arg172: tensor<256x7x7x512xf32>, %arg173: tensor<512xf32>, %arg174: tensor<512xf32>, %arg175: tensor<512xf32>, %arg176: tensor<256x7x7x512xf32>, %arg177: tensor<512xf32>):
      %62 = stablehlo.broadcast_in_dim %arg177, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %63 = stablehlo.subtract %arg176, %62 : tensor<256x7x7x512xf32>
      %64 = stablehlo.broadcast_in_dim %arg175, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x7x7x512xf32>
      %66 = stablehlo.broadcast_in_dim %arg174, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x7x7x512xf32>
      %68 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %69 = stablehlo.subtract %arg172, %68 : tensor<256x7x7x512xf32>
      %70 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %71 = stablehlo.multiply %69, %70 : tensor<256x7x7x512xf32>
      %72 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %73 = stablehlo.add %71, %72 : tensor<256x7x7x512xf32>
      %74 = stablehlo.add %67, %73 : tensor<256x7x7x512xf32>
      %75 = stablehlo.convert %74 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %76 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %77 = stablehlo.maximum %75, %76 : tensor<256x7x7x512xbf16>
      %78 = stablehlo.convert %77 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %78 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %52 = "mhlo.fusion"(%arg57, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<512xf32>, %arg171: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<512xf32>
      %64 = stablehlo.rsqrt %63 : tensor<512xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<512xf32>
      mhlo.return %65 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %53 = "mhlo.fusion"(%arg55, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<512xf32>, %arg171: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %63 = stablehlo.add %arg171, %62 : tensor<512xf32>
      %64 = stablehlo.rsqrt %63 : tensor<512xf32>
      %65 = stablehlo.multiply %64, %arg170 : tensor<512xf32>
      mhlo.return %65 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %54 = stablehlo.convolution(%51, %arg58) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %55 = "mhlo.fusion"(%arg54, %53, %54, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<512xf32>, %arg171: tensor<512xf32>, %arg172: tensor<256x7x7x512xf32>, %arg173: tensor<512xf32>):
      %62 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %63 = stablehlo.subtract %arg172, %62 : tensor<256x7x7x512xf32>
      %64 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x7x7x512xf32>
      %66 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x7x7x512xf32>
      %68 = stablehlo.convert %67 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %70 = stablehlo.maximum %68, %69 : tensor<256x7x7x512xbf16>
      %71 = stablehlo.convert %70 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %71 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %56 = stablehlo.convolution(%55, %arg59) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %57 = "mhlo.fusion"(%51, %arg56, %52, %56, %arg162) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg170: tensor<256x7x7x512xf32>, %arg171: tensor<512xf32>, %arg172: tensor<512xf32>, %arg173: tensor<256x7x7x512xf32>, %arg174: tensor<512xf32>):
      %62 = stablehlo.broadcast_in_dim %arg174, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %63 = stablehlo.subtract %arg173, %62 : tensor<256x7x7x512xf32>
      %64 = stablehlo.broadcast_in_dim %arg172, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %65 = stablehlo.multiply %63, %64 : tensor<256x7x7x512xf32>
      %66 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %67 = stablehlo.add %65, %66 : tensor<256x7x7x512xf32>
      %68 = stablehlo.add %arg170, %67 : tensor<256x7x7x512xf32>
      %69 = stablehlo.convert %68 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %70 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %71 = stablehlo.maximum %69, %70 : tensor<256x7x7x512xbf16>
      %72 = stablehlo.convert %71 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      %73 = mhlo.bitcast %72 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x512xf32>) -> tensor<256x49x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %74 = stablehlo.reduce(%73 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x49x512xf32>, tensor<f32>) -> tensor<256x512xf32>
      mhlo.return %74 : tensor<256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<256x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x512xf32>
    %58 = "mhlo.fusion"(%57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<256x512xf32>):
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
      %63 = stablehlo.multiply %arg170, %62 : tensor<256x512xf32>
      %64 = stablehlo.convert %63 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
      mhlo.return %64 : tensor<256x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %59 = "mhlo.fusion"(%58, %arg2) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg170: tensor<256x512xbf16>, %arg171: tensor<512x1000xf32>):
      %62 = stablehlo.convert %arg171 : (tensor<512x1000xf32>) -> tensor<512x1000xbf16>
      %63 = stablehlo.dot_general %62, %arg170, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[1000,256]{0,1}"} : (tensor<512x1000xbf16>, tensor<256x512xbf16>) -> tensor<1000x256xbf16>
      %64 = mhlo.bitcast %63 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x256xbf16>) -> tensor<256x1000xbf16>
      mhlo.return %64 : tensor<256x1000xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x512xbf16>, tensor<512x1000xf32>) -> tensor<256x1000xbf16>
    %60 = "mhlo.fusion"(%arg1, %59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg170: tensor<1000xf32>, %arg171: tensor<256x1000xbf16>):
      %62 = stablehlo.convert %arg171 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
      %63 = stablehlo.broadcast_in_dim %arg170, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
      %64 = stablehlo.add %62, %63 : tensor<256x1000xf32>
      %65 = stablehlo.convert %64 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
      mhlo.return %65 : tensor<256x1000xbf16>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<256x1000xbf16>) -> tensor<256x1000xbf16>
    %61 = mhlo.bitcast %60 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x1000xbf16>) -> tensor<1x256x1000xbf16>
    return %61 : tensor<1x256x1000xbf16>
  }
}
