module @pmap_inference_step attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {suggested_combiner_threshold = "38288554144"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.num_replicas = 4 : i32, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_computation(%arg0: tensor<256x56x56x64xbf16>, %arg1: tensor<1x1x64x256xf32>) -> tensor<256x56x56x256xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xbf16>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xbf16>) -> tensor<64x256xbf16>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
    %3 = stablehlo.dot_general %1, %2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[256,802816]{0,1}"} : (tensor<64x256xbf16>, tensor<802816x64xbf16>) -> tensor<256x802816xbf16>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x802816xbf16>) -> tensor<802816x256xbf16>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
    return %5 : tensor<256x56x56x256xbf16>
  }
  func.func private @gemm_fusion_dot.2_computation(%arg0: tensor<256x56x56x64xbf16>, %arg1: tensor<1x1x64x64xf32>) -> tensor<256x56x56x64xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xbf16>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x64xbf16>) -> tensor<64x64xbf16>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
    %3 = stablehlo.dot_general %1, %2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[64,802816]{0,1}"} : (tensor<64x64xbf16>, tensor<802816x64xbf16>) -> tensor<64x802816xbf16>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<64x802816xbf16>) -> tensor<802816x64xbf16>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x64xbf16>) -> tensor<256x56x56x64xbf16>
    return %5 : tensor<256x56x56x64xbf16>
  }
  func.func private @maximum.99(%arg0: tensor<802816x64xbf16>) -> tensor<802816x64xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<802816x64xbf16>
    return %1 : tensor<802816x64xbf16>
  }
  func.func private @convert.2(%arg0: tensor<64x256xf32>) -> tensor<64x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
    return %0 : tensor<64x256xbf16>
  }
  func.func private @gemm_fusion_dot.3_computation(%arg0: tensor<802816x64xbf16>, %arg1: tensor<64x256xf32>) -> tensor<802816x256xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<802816x64xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<802816x64xbf16>
      mhlo.return %4 : tensor<802816x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>) -> tensor<802816x64xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<64x256xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
      mhlo.return %3 : tensor<64x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<802816x64xbf16>, tensor<64x256xbf16>) -> tensor<802816x256xbf16>
    return %2 : tensor<802816x256xbf16>
  }
  func.func private @maximum.100(%arg0: tensor<802816x256xbf16>) -> tensor<802816x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<802816x256xbf16>
    return %1 : tensor<802816x256xbf16>
  }
  func.func private @convert.3(%arg0: tensor<256x64xf32>) -> tensor<256x64xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
    return %0 : tensor<256x64xbf16>
  }
  func.func private @gemm_fusion_dot.4_computation(%arg0: tensor<802816x256xbf16>, %arg1: tensor<256x64xf32>) -> tensor<802816x64xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<802816x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<802816x256xbf16>
      mhlo.return %4 : tensor<802816x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>) -> tensor<802816x256xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<256x64xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
      mhlo.return %3 : tensor<256x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x64xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<802816x256xbf16>, tensor<256x64xbf16>) -> tensor<802816x64xbf16>
    return %2 : tensor<802816x64xbf16>
  }
  func.func private @maximum.101(%arg0: tensor<802816x64xbf16>) -> tensor<802816x64xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<802816x64xbf16>
    return %1 : tensor<802816x64xbf16>
  }
  func.func private @convert.4(%arg0: tensor<64x256xf32>) -> tensor<64x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
    return %0 : tensor<64x256xbf16>
  }
  func.func private @gemm_fusion_dot.5_computation(%arg0: tensor<802816x64xbf16>, %arg1: tensor<64x256xf32>) -> tensor<802816x256xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<802816x64xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<802816x64xbf16>
      mhlo.return %4 : tensor<802816x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>) -> tensor<802816x64xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<64x256xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
      mhlo.return %3 : tensor<64x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<802816x64xbf16>, tensor<64x256xbf16>) -> tensor<802816x256xbf16>
    return %2 : tensor<802816x256xbf16>
  }
  func.func private @maximum.102(%arg0: tensor<802816x256xbf16>) -> tensor<802816x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<802816x256xbf16>
    return %1 : tensor<802816x256xbf16>
  }
  func.func private @convert.5(%arg0: tensor<256x64xf32>) -> tensor<256x64xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
    return %0 : tensor<256x64xbf16>
  }
  func.func private @gemm_fusion_dot.6_computation(%arg0: tensor<802816x256xbf16>, %arg1: tensor<256x64xf32>) -> tensor<802816x64xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<802816x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<802816x256xbf16>
      mhlo.return %4 : tensor<802816x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>) -> tensor<802816x256xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<256x64xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
      mhlo.return %3 : tensor<256x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x64xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<802816x256xbf16>, tensor<256x64xbf16>) -> tensor<802816x64xbf16>
    return %2 : tensor<802816x64xbf16>
  }
  func.func private @maximum.103(%arg0: tensor<802816x64xbf16>) -> tensor<802816x64xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<802816x64xbf16>
    return %1 : tensor<802816x64xbf16>
  }
  func.func private @convert.6(%arg0: tensor<64x256xf32>) -> tensor<64x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
    return %0 : tensor<64x256xbf16>
  }
  func.func private @gemm_fusion_dot.7_computation(%arg0: tensor<802816x64xbf16>, %arg1: tensor<64x256xf32>) -> tensor<802816x256xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<802816x64xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<802816x64xbf16>
      mhlo.return %4 : tensor<802816x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>) -> tensor<802816x64xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<64x256xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
      mhlo.return %3 : tensor<64x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<802816x64xbf16>, tensor<64x256xbf16>) -> tensor<802816x256xbf16>
    return %2 : tensor<802816x256xbf16>
  }
  func.func private @maximum.104(%arg0: tensor<802816x256xbf16>) -> tensor<802816x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<802816x256xbf16>
    return %1 : tensor<802816x256xbf16>
  }
  func.func private @convert.7(%arg0: tensor<256x128xf32>) -> tensor<256x128xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x128xf32>) -> tensor<256x128xbf16>
    return %0 : tensor<256x128xbf16>
  }
  func.func private @gemm_fusion_dot.8_computation(%arg0: tensor<802816x256xbf16>, %arg1: tensor<256x128xf32>) -> tensor<802816x128xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<802816x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<802816x256xbf16>
      mhlo.return %4 : tensor<802816x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>) -> tensor<802816x256xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<256x128xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<256x128xf32>) -> tensor<256x128xbf16>
      mhlo.return %3 : tensor<256x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x128xf32>) -> tensor<256x128xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<802816x256xbf16>, tensor<256x128xbf16>) -> tensor<802816x128xbf16>
    return %2 : tensor<802816x128xbf16>
  }
  func.func private @maximum.105(%arg0: tensor<200704x128xbf16>) -> tensor<200704x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<200704x128xbf16>
    return %1 : tensor<200704x128xbf16>
  }
  func.func private @convert.8(%arg0: tensor<128x512xf32>) -> tensor<128x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
    return %0 : tensor<128x512xbf16>
  }
  func.func private @gemm_fusion_dot.9_computation(%arg0: tensor<200704x128xbf16>, %arg1: tensor<128x512xf32>) -> tensor<200704x512xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<200704x128xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<200704x128xbf16>
      mhlo.return %4 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<128x512xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      mhlo.return %3 : tensor<128x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
    return %2 : tensor<200704x512xbf16>
  }
  func.func private @maximum.106(%arg0: tensor<200704x512xbf16>) -> tensor<200704x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<200704x512xbf16>
    return %1 : tensor<200704x512xbf16>
  }
  func.func private @convert.9(%arg0: tensor<512x128xf32>) -> tensor<512x128xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
    return %0 : tensor<512x128xbf16>
  }
  func.func private @gemm_fusion_dot.10_computation(%arg0: tensor<200704x512xbf16>, %arg1: tensor<512x128xf32>) -> tensor<200704x128xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<200704x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<200704x512xbf16>
      mhlo.return %4 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<512x128xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
      mhlo.return %3 : tensor<512x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<512x128xf32>) -> tensor<512x128xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x128xbf16>) -> tensor<200704x128xbf16>
    return %2 : tensor<200704x128xbf16>
  }
  func.func private @maximum.107(%arg0: tensor<200704x128xbf16>) -> tensor<200704x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<200704x128xbf16>
    return %1 : tensor<200704x128xbf16>
  }
  func.func private @convert.10(%arg0: tensor<128x512xf32>) -> tensor<128x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
    return %0 : tensor<128x512xbf16>
  }
  func.func private @gemm_fusion_dot.11_computation(%arg0: tensor<200704x128xbf16>, %arg1: tensor<128x512xf32>) -> tensor<200704x512xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<200704x128xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<200704x128xbf16>
      mhlo.return %4 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<128x512xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      mhlo.return %3 : tensor<128x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
    return %2 : tensor<200704x512xbf16>
  }
  func.func private @maximum.108(%arg0: tensor<200704x512xbf16>) -> tensor<200704x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<200704x512xbf16>
    return %1 : tensor<200704x512xbf16>
  }
  func.func private @convert.11(%arg0: tensor<512x128xf32>) -> tensor<512x128xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
    return %0 : tensor<512x128xbf16>
  }
  func.func private @gemm_fusion_dot.12_computation(%arg0: tensor<200704x512xbf16>, %arg1: tensor<512x128xf32>) -> tensor<200704x128xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<200704x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<200704x512xbf16>
      mhlo.return %4 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<512x128xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
      mhlo.return %3 : tensor<512x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<512x128xf32>) -> tensor<512x128xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x128xbf16>) -> tensor<200704x128xbf16>
    return %2 : tensor<200704x128xbf16>
  }
  func.func private @maximum.109(%arg0: tensor<200704x128xbf16>) -> tensor<200704x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<200704x128xbf16>
    return %1 : tensor<200704x128xbf16>
  }
  func.func private @convert.12(%arg0: tensor<128x512xf32>) -> tensor<128x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
    return %0 : tensor<128x512xbf16>
  }
  func.func private @gemm_fusion_dot.13_computation(%arg0: tensor<200704x128xbf16>, %arg1: tensor<128x512xf32>) -> tensor<200704x512xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<200704x128xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<200704x128xbf16>
      mhlo.return %4 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<128x512xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      mhlo.return %3 : tensor<128x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
    return %2 : tensor<200704x512xbf16>
  }
  func.func private @maximum.110(%arg0: tensor<200704x512xbf16>) -> tensor<200704x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<200704x512xbf16>
    return %1 : tensor<200704x512xbf16>
  }
  func.func private @convert.13(%arg0: tensor<512x128xf32>) -> tensor<512x128xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
    return %0 : tensor<512x128xbf16>
  }
  func.func private @gemm_fusion_dot.14_computation(%arg0: tensor<200704x512xbf16>, %arg1: tensor<512x128xf32>) -> tensor<200704x128xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<200704x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<200704x512xbf16>
      mhlo.return %4 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<512x128xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
      mhlo.return %3 : tensor<512x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<512x128xf32>) -> tensor<512x128xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x128xbf16>) -> tensor<200704x128xbf16>
    return %2 : tensor<200704x128xbf16>
  }
  func.func private @maximum.111(%arg0: tensor<200704x128xbf16>) -> tensor<200704x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<200704x128xbf16>
    return %1 : tensor<200704x128xbf16>
  }
  func.func private @convert.14(%arg0: tensor<128x512xf32>) -> tensor<128x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
    return %0 : tensor<128x512xbf16>
  }
  func.func private @gemm_fusion_dot.15_computation(%arg0: tensor<200704x128xbf16>, %arg1: tensor<128x512xf32>) -> tensor<200704x512xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<200704x128xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<200704x128xbf16>
      mhlo.return %4 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<128x512xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      mhlo.return %3 : tensor<128x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
    return %2 : tensor<200704x512xbf16>
  }
  func.func private @maximum.112(%arg0: tensor<200704x512xbf16>) -> tensor<200704x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<200704x512xbf16>
    return %1 : tensor<200704x512xbf16>
  }
  func.func private @convert.15(%arg0: tensor<512x256xf32>) -> tensor<512x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x256xf32>) -> tensor<512x256xbf16>
    return %0 : tensor<512x256xbf16>
  }
  func.func private @gemm_fusion_dot.16_computation(%arg0: tensor<200704x512xbf16>, %arg1: tensor<512x256xf32>) -> tensor<200704x256xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<200704x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<200704x512xbf16>
      mhlo.return %4 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<512x256xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<512x256xf32>) -> tensor<512x256xbf16>
      mhlo.return %3 : tensor<512x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<512x256xf32>) -> tensor<512x256xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x256xbf16>) -> tensor<200704x256xbf16>
    return %2 : tensor<200704x256xbf16>
  }
  func.func private @maximum.113(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.16(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.17_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x256xbf16>
      mhlo.return %4 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<256x1024xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      mhlo.return %3 : tensor<256x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
    return %2 : tensor<50176x1024xbf16>
  }
  func.func private @maximum.114(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.17(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.18_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x1024xbf16>
      mhlo.return %4 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<1024x256xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      mhlo.return %3 : tensor<1024x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
    return %2 : tensor<50176x256xbf16>
  }
  func.func private @maximum.115(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.18(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.19_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x256xbf16>
      mhlo.return %4 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<256x1024xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      mhlo.return %3 : tensor<256x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
    return %2 : tensor<50176x1024xbf16>
  }
  func.func private @maximum.116(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.19(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.20_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x1024xbf16>
      mhlo.return %4 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<1024x256xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      mhlo.return %3 : tensor<1024x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
    return %2 : tensor<50176x256xbf16>
  }
  func.func private @maximum.117(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.20(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.21_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x256xbf16>
      mhlo.return %4 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<256x1024xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      mhlo.return %3 : tensor<256x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
    return %2 : tensor<50176x1024xbf16>
  }
  func.func private @maximum.118(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.21(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.22_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x1024xbf16>
      mhlo.return %4 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<1024x256xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      mhlo.return %3 : tensor<1024x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
    return %2 : tensor<50176x256xbf16>
  }
  func.func private @maximum.119(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.22(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.23_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x256xbf16>
      mhlo.return %4 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<256x1024xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      mhlo.return %3 : tensor<256x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
    return %2 : tensor<50176x1024xbf16>
  }
  func.func private @maximum.120(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.23(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.24_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x1024xbf16>
      mhlo.return %4 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<1024x256xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      mhlo.return %3 : tensor<1024x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
    return %2 : tensor<50176x256xbf16>
  }
  func.func private @maximum.121(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.24(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.25_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x256xbf16>
      mhlo.return %4 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<256x1024xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      mhlo.return %3 : tensor<256x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
    return %2 : tensor<50176x1024xbf16>
  }
  func.func private @maximum.122(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.25(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.26_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x1024xbf16>
      mhlo.return %4 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<1024x256xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      mhlo.return %3 : tensor<1024x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
    return %2 : tensor<50176x256xbf16>
  }
  func.func private @maximum.123(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.26(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.27_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x256xbf16>
      mhlo.return %4 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<256x1024xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      mhlo.return %3 : tensor<256x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
    return %2 : tensor<50176x1024xbf16>
  }
  func.func private @maximum.124(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.27(%arg0: tensor<1024x512xf32>) -> tensor<1024x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x512xf32>) -> tensor<1024x512xbf16>
    return %0 : tensor<1024x512xbf16>
  }
  func.func private @gemm_fusion_dot.28_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x512xf32>) -> tensor<50176x512xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<50176x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<50176x1024xbf16>
      mhlo.return %4 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<1024x512xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<1024x512xf32>) -> tensor<1024x512xbf16>
      mhlo.return %3 : tensor<1024x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<1024x512xf32>) -> tensor<1024x512xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x512xbf16>) -> tensor<50176x512xbf16>
    return %2 : tensor<50176x512xbf16>
  }
  func.func private @maximum.125(%arg0: tensor<12544x512xbf16>) -> tensor<12544x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x512xbf16>
    return %1 : tensor<12544x512xbf16>
  }
  func.func private @convert.28(%arg0: tensor<512x2048xf32>) -> tensor<512x2048xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
    return %0 : tensor<512x2048xbf16>
  }
  func.func private @gemm_fusion_dot.29_computation(%arg0: tensor<12544x512xbf16>, %arg1: tensor<512x2048xf32>) -> tensor<12544x2048xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<12544x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<12544x512xbf16>
      mhlo.return %4 : tensor<12544x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>) -> tensor<12544x512xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<512x2048xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
      mhlo.return %3 : tensor<512x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<12544x512xbf16>, tensor<512x2048xbf16>) -> tensor<12544x2048xbf16>
    return %2 : tensor<12544x2048xbf16>
  }
  func.func private @maximum.126(%arg0: tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x2048xbf16>
    return %1 : tensor<12544x2048xbf16>
  }
  func.func private @convert.29(%arg0: tensor<2048x512xf32>) -> tensor<2048x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
    return %0 : tensor<2048x512xbf16>
  }
  func.func private @gemm_fusion_dot.30_computation(%arg0: tensor<12544x2048xbf16>, %arg1: tensor<2048x512xf32>) -> tensor<12544x512xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<12544x2048xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<12544x2048xbf16>
      mhlo.return %4 : tensor<12544x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<2048x512xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
      mhlo.return %3 : tensor<2048x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<12544x2048xbf16>, tensor<2048x512xbf16>) -> tensor<12544x512xbf16>
    return %2 : tensor<12544x512xbf16>
  }
  func.func private @maximum.127(%arg0: tensor<12544x512xbf16>) -> tensor<12544x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x512xbf16>
    return %1 : tensor<12544x512xbf16>
  }
  func.func private @convert.30(%arg0: tensor<512x2048xf32>) -> tensor<512x2048xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
    return %0 : tensor<512x2048xbf16>
  }
  func.func private @gemm_fusion_dot.31_computation(%arg0: tensor<12544x512xbf16>, %arg1: tensor<512x2048xf32>) -> tensor<12544x2048xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<12544x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<12544x512xbf16>
      mhlo.return %4 : tensor<12544x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>) -> tensor<12544x512xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<512x2048xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
      mhlo.return %3 : tensor<512x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<12544x512xbf16>, tensor<512x2048xbf16>) -> tensor<12544x2048xbf16>
    return %2 : tensor<12544x2048xbf16>
  }
  func.func private @maximum.128(%arg0: tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x2048xbf16>
    return %1 : tensor<12544x2048xbf16>
  }
  func.func private @convert.31(%arg0: tensor<2048x512xf32>) -> tensor<2048x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
    return %0 : tensor<2048x512xbf16>
  }
  func.func private @gemm_fusion_dot.32_computation(%arg0: tensor<12544x2048xbf16>, %arg1: tensor<2048x512xf32>) -> tensor<12544x512xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<12544x2048xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<12544x2048xbf16>
      mhlo.return %4 : tensor<12544x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<2048x512xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
      mhlo.return %3 : tensor<2048x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<12544x2048xbf16>, tensor<2048x512xbf16>) -> tensor<12544x512xbf16>
    return %2 : tensor<12544x512xbf16>
  }
  func.func private @maximum.129(%arg0: tensor<12544x512xbf16>) -> tensor<12544x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x512xbf16>
    return %1 : tensor<12544x512xbf16>
  }
  func.func private @convert.32(%arg0: tensor<512x2048xf32>) -> tensor<512x2048xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
    return %0 : tensor<512x2048xbf16>
  }
  func.func private @gemm_fusion_dot.33_computation(%arg0: tensor<12544x512xbf16>, %arg1: tensor<512x2048xf32>) -> tensor<12544x2048xbf16> {
    %0 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<12544x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
      %4 = stablehlo.maximum %arg2, %3 : tensor<12544x512xbf16>
      mhlo.return %4 : tensor<12544x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>) -> tensor<12544x512xbf16>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg2: tensor<512x2048xf32>):
      %3 = stablehlo.convert %arg2 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
      mhlo.return %3 : tensor<512x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
    %2 = stablehlo.dot %0, %1, precision = [DEFAULT, DEFAULT] : (tensor<12544x512xbf16>, tensor<512x2048xbf16>) -> tensor<12544x2048xbf16>
    return %2 : tensor<12544x2048xbf16>
  }
  func.func private @gemm_fusion_dot.1_computation(%arg0: tensor<256x2048xbf16>, %arg1: tensor<2048x1000xf32>) -> tensor<256x1000xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<2048x1000xf32>) -> tensor<2048x1000xbf16>
    %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[1000,256]{0,1}"} : (tensor<2048x1000xbf16>, tensor<256x2048xbf16>) -> tensor<1000x256xbf16>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x256xbf16>) -> tensor<256x1000xbf16>
    return %2 : tensor<256x1000xbf16>
  }
  func.func private @region_1.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<12544x2048xbf16>, %arg4: tensor<256x7x7x2048xbf16>) -> tensor<256x2048xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x2048xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x7x7x2048xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %3 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x7x7x2048xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x7x7x2048xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x7x7x2048xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x7x7x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    %13 = stablehlo.maximum %12, %0 : tensor<256x7x7x2048xbf16>
    %14 = stablehlo.convert %13 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x2048xf32>) -> tensor<256x49x2048xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x49x2048xf32>, tensor<f32>) -> tensor<256x2048xf32>
    return %16 : tensor<256x2048xf32>
  }
  func.func private @fused_convert(%arg0: tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
    return %1 : tensor<256x224x224x3xf32>
  }
  func.func private @fused_convert.1(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x56x56x256xbf16>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16> {
    %0 = stablehlo.convert %arg3 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x56x56x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<256x56x56x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %6 = stablehlo.add %4, %5 : tensor<256x56x56x256xf32>
    %7 = mhlo.bitcast %arg7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
    %8 = stablehlo.convert %7 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %10 = stablehlo.subtract %8, %9 : tensor<256x56x56x256xf32>
    %11 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<256x56x56x256xf32>
    %13 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %14 = stablehlo.add %12, %13 : tensor<256x56x56x256xf32>
    %15 = stablehlo.add %6, %14 : tensor<256x56x56x256xf32>
    %16 = stablehlo.convert %15 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
    return %16 : tensor<256x56x56x256xbf16>
  }
  func.func private @fused_convert.2(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<802816x256xbf16>, %arg4: tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x256xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x56x56x256xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %3 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x56x56x256xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x56x56x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x56x56x256xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x56x56x256xf32>
    %12 = stablehlo.convert %11 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
    return %12 : tensor<256x56x56x256xbf16>
  }
  func.func private @fused_convert.3(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<802816x256xbf16>, %arg4: tensor<256x56x56x256xbf16>) -> (tensor<256x56x56x256xf32>, tensor<256x56x56x256xbf16>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x256xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x56x56x256xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %3 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x56x56x256xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x56x56x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x56x56x256xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x56x56x256xf32>
    %12 = stablehlo.convert %11 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
    %13 = stablehlo.maximum %12, %0 : tensor<256x56x56x256xbf16>
    %14 = stablehlo.convert %13 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    return %14, %12 : tensor<256x56x56x256xf32>, tensor<256x56x56x256xbf16>
  }
  func.func private @fused_convert.4(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<200704x512xbf16>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<256x28x28x512xf32>, %arg7: tensor<512xf32>) -> tensor<256x28x28x512xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<256x28x28x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x512xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
    %7 = stablehlo.convert %6 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %8 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %9 = stablehlo.subtract %7, %8 : tensor<256x28x28x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<256x28x28x512xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x28x28x512xf32>
    %14 = stablehlo.add %5, %13 : tensor<256x28x28x512xf32>
    %15 = stablehlo.convert %14 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    return %15 : tensor<256x28x28x512xbf16>
  }
  func.func private @fused_convert.5(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<200704x512xbf16>, %arg4: tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x28x28x512xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %3 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x28x28x512xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x28x28x512xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x28x28x512xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x28x28x512xf32>
    %12 = stablehlo.convert %11 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    return %12 : tensor<256x28x28x512xbf16>
  }
  func.func private @fused_convert.6(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<200704x512xbf16>, %arg4: tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x28x28x512xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %3 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x28x28x512xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x28x28x512xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x28x28x512xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x28x28x512xf32>
    %12 = stablehlo.convert %11 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    return %12 : tensor<256x28x28x512xbf16>
  }
  func.func private @fused_convert.7(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<200704x512xbf16>, %arg4: tensor<256x28x28x512xbf16>) -> (tensor<256x28x28x512xf32>, tensor<256x28x28x512xbf16>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x28x28x512xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %3 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x28x28x512xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x28x28x512xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x28x28x512xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x28x28x512xf32>
    %12 = stablehlo.convert %11 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %13 = stablehlo.maximum %12, %0 : tensor<256x28x28x512xbf16>
    %14 = stablehlo.convert %13 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    return %14, %12 : tensor<256x28x28x512xf32>, tensor<256x28x28x512xbf16>
  }
  func.func private @region_0.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg0 : (tensor<f32>) -> tensor<bf16>
    %1 = stablehlo.convert %arg1 : (tensor<f32>) -> tensor<bf16>
    %2 = stablehlo.maximum %0, %1 : tensor<bf16>
    %3 = stablehlo.convert %2 : (tensor<bf16>) -> tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @fused_convert.8(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x112x112x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xbf16> {
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
      %13 = stablehlo.convert %arg4 : (tensor<f32>) -> tensor<bf16>
      %14 = stablehlo.convert %arg5 : (tensor<f32>) -> tensor<bf16>
      %15 = stablehlo.maximum %13, %14 : tensor<bf16>
      %16 = stablehlo.convert %15 : (tensor<bf16>) -> tensor<f32>
      stablehlo.return %16 : tensor<f32>
    }) : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<256x56x56x64xf32>
    %12 = stablehlo.convert %11 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    return %12 : tensor<256x56x56x64xbf16>
  }
  func.func private @fused_convert.9(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<256x14x14x1024xf32>, %arg7: tensor<1024xf32>) -> tensor<256x14x14x1024xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<256x14x14x1024xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x1024xf32>
    %4 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x1024xf32>
    %6 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %7 = stablehlo.convert %6 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %9 = stablehlo.subtract %7, %8 : tensor<256x14x14x1024xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<256x14x14x1024xf32>
    %12 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x14x14x1024xf32>
    %14 = stablehlo.add %5, %13 : tensor<256x14x14x1024xf32>
    %15 = stablehlo.convert %14 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    return %15 : tensor<256x14x14x1024xbf16>
  }
  func.func private @fused_convert.10(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x14x14x1024xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %5 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x14x14x1024xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x14x14x1024xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x14x14x1024xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x14x14x1024xf32>
    %12 = stablehlo.convert %11 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    return %12 : tensor<256x14x14x1024xbf16>
  }
  func.func private @fused_convert.11(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x14x14x1024xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %5 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x14x14x1024xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x14x14x1024xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x14x14x1024xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x14x14x1024xf32>
    %12 = stablehlo.convert %11 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    return %12 : tensor<256x14x14x1024xbf16>
  }
  func.func private @fused_convert.12(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x14x14x1024xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %5 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x14x14x1024xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x14x14x1024xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x14x14x1024xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x14x14x1024xf32>
    %12 = stablehlo.convert %11 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    return %12 : tensor<256x14x14x1024xbf16>
  }
  func.func private @fused_convert.13(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x14x14x1024xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %5 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x14x14x1024xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x14x14x1024xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x14x14x1024xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x14x14x1024xf32>
    %12 = stablehlo.convert %11 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    return %12 : tensor<256x14x14x1024xbf16>
  }
  func.func private @fused_convert.14(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    return %6 : tensor<256x56x56x64xbf16>
  }
  func.func private @fused_convert.15(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    return %6 : tensor<256x56x56x64xbf16>
  }
  func.func private @fused_convert.16(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    return %6 : tensor<256x56x56x64xbf16>
  }
  func.func private @fused_convert.17(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<802816x128xbf16>) -> tensor<256x56x56x128xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x128xbf16>) -> tensor<256x56x56x128xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<256x56x56x128xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x56x56x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x56x56x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x56x56x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x56x56x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x56x56x128xf32>
    %8 = stablehlo.convert %7 : (tensor<256x56x56x128xf32>) -> tensor<256x56x56x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x128xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x56x56x128xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xf32>
    return %11 : tensor<256x56x56x128xf32>
  }
  func.func private @fused_convert.18(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<802816x64xbf16>) -> tensor<256x56x56x64xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x64xbf16>) -> tensor<256x56x56x64xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x56x56x64xf32>
    %8 = stablehlo.convert %7 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x56x56x64xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    return %11 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.19(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<802816x64xbf16>) -> tensor<256x56x56x64xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x64xbf16>) -> tensor<256x56x56x64xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x56x56x64xf32>
    %8 = stablehlo.convert %7 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x56x56x64xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    return %11 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.20(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32> {
    %0 = stablehlo.convert %arg3 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<256x56x56x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<256x56x56x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %6 = stablehlo.add %4, %5 : tensor<256x56x56x64xf32>
    %7 = stablehlo.convert %6 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %9 = stablehlo.maximum %7, %8 : tensor<256x56x56x64xbf16>
    %10 = stablehlo.convert %9 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    return %10 : tensor<256x56x56x64xf32>
  }
  func.func private @fused_convert.21(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<200704x256xbf16>) -> tensor<256x28x28x256xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x256xbf16>) -> tensor<256x28x28x256xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x28x28x256xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x28x28x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x28x28x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x28x28x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x28x28x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x28x28x256xf32>
    %8 = stablehlo.convert %7 : (tensor<256x28x28x256xf32>) -> tensor<256x28x28x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x256xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x28x28x256xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xf32>
    return %11 : tensor<256x28x28x256xf32>
  }
  func.func private @fused_convert.22(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<50176x1024xbf16>, %arg4: tensor<256x14x14x1024xbf16>) -> (tensor<256x14x14x1024xf32>, tensor<256x14x14x1024xbf16>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x14x14x1024xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x14x14x1024xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x14x14x1024xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x14x14x1024xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x14x14x1024xf32>
    %12 = stablehlo.convert %11 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %13 = stablehlo.maximum %12, %0 : tensor<256x14x14x1024xbf16>
    %14 = stablehlo.convert %13 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    return %14, %12 : tensor<256x14x14x1024xf32>, tensor<256x14x14x1024xbf16>
  }
  func.func private @fused_convert.23(%arg0: tensor<12544x2048xbf16>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>, %arg6: tensor<256x7x7x2048xf32>, %arg7: tensor<2048xf32>) -> tensor<256x7x7x2048xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %1 = stablehlo.subtract %arg6, %0 : tensor<256x7x7x2048xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x2048xf32>
    %4 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x2048xf32>
    %6 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %7 = stablehlo.convert %6 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %9 = stablehlo.subtract %7, %8 : tensor<256x7x7x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<256x7x7x2048xf32>
    %12 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %13 = stablehlo.add %11, %12 : tensor<256x7x7x2048xf32>
    %14 = stablehlo.add %5, %13 : tensor<256x7x7x2048xf32>
    %15 = stablehlo.convert %14 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    return %15 : tensor<256x7x7x2048xbf16>
  }
  func.func private @fused_convert.24(%arg0: tensor<12544x2048xbf16>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x2048xbf16>
    %1 = stablehlo.maximum %arg4, %0 : tensor<256x7x7x2048xbf16>
    %2 = stablehlo.convert %1 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %3 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %5 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %6 = stablehlo.subtract %4, %5 : tensor<256x7x7x2048xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<256x7x7x2048xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
    %10 = stablehlo.add %8, %9 : tensor<256x7x7x2048xf32>
    %11 = stablehlo.add %2, %10 : tensor<256x7x7x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    return %12 : tensor<256x7x7x2048xbf16>
  }
  func.func private @fused_convert.25(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    return %6 : tensor<256x28x28x128xbf16>
  }
  func.func private @fused_convert.26(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    return %6 : tensor<256x28x28x128xbf16>
  }
  func.func private @fused_convert.27(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    return %6 : tensor<256x28x28x128xbf16>
  }
  func.func private @fused_convert.28(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    return %6 : tensor<256x28x28x128xbf16>
  }
  func.func private @fused_convert.29(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x28x28x128xf32>
    %8 = stablehlo.convert %7 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x28x28x128xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %11 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.30(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x28x28x128xf32>
    %8 = stablehlo.convert %7 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x28x28x128xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %11 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.31(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x28x28x128xf32>
    %8 = stablehlo.convert %7 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x28x28x128xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    return %11 : tensor<256x28x28x128xf32>
  }
  func.func private @fused_convert.32(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<50176x512xbf16>) -> tensor<256x14x14x512xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x512xbf16>) -> tensor<256x14x14x512xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<256x14x14x512xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x14x14x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x14x14x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x14x14x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x14x14x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x14x14x512xf32>
    %8 = stablehlo.convert %7 : (tensor<256x14x14x512xf32>) -> tensor<256x14x14x512xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x512xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x14x14x512xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xf32>
    return %11 : tensor<256x14x14x512xf32>
  }
  func.func private @fused_convert.33(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.34(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.35(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.36(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.37(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.38(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.39(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.convert %5 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    return %6 : tensor<256x7x7x512xbf16>
  }
  func.func private @fused_convert.40(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.convert %5 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    return %6 : tensor<256x7x7x512xbf16>
  }
  func.func private @fused_convert.41(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.convert %5 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    return %6 : tensor<256x7x7x512xbf16>
  }
  func.func private @fused_convert.42(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x14x14x256xf32>
    %8 = stablehlo.convert %7 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x14x14x256xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %11 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.43(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x14x14x256xf32>
    %8 = stablehlo.convert %7 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x14x14x256xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %11 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.44(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x14x14x256xf32>
    %8 = stablehlo.convert %7 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x14x14x256xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %11 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.45(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x14x14x256xf32>
    %8 = stablehlo.convert %7 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x14x14x256xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %11 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.46(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x14x14x256xf32>
    %8 = stablehlo.convert %7 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x14x14x256xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    return %11 : tensor<256x14x14x256xf32>
  }
  func.func private @fused_convert.47(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<12544x512xbf16>) -> tensor<256x7x7x512xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x7x7x512xf32>
    %8 = stablehlo.convert %7 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x7x7x512xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    return %11 : tensor<256x7x7x512xf32>
  }
  func.func private @fused_convert.48(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<12544x512xbf16>) -> tensor<256x7x7x512xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1 = stablehlo.convert %0 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<256x7x7x512xf32>
    %8 = stablehlo.convert %7 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %10 = stablehlo.maximum %8, %9 : tensor<256x7x7x512xbf16>
    %11 = stablehlo.convert %10 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    return %11 : tensor<256x7x7x512xf32>
  }
  func.func private @fused_convert.49(%arg0: tensor<256x2048xf32>) -> tensor<256x2048xbf16> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x2048xf32>
    %2 = stablehlo.convert %1 : (tensor<256x2048xf32>) -> tensor<256x2048xbf16>
    return %2 : tensor<256x2048xbf16>
  }
  func.func private @fused_convert.50(%arg0: tensor<1000xf32>, %arg1: tensor<256x1000xbf16>) -> tensor<256x1000xbf16> {
    %0 = stablehlo.convert %arg1 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %2 = stablehlo.add %0, %1 : tensor<256x1000xf32>
    %3 = stablehlo.convert %2 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
    return %3 : tensor<256x1000xbf16>
  }
  func.func private @fused_multiply(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<2048xf32>
    return %3 : tensor<2048xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<2048xf32>
    return %3 : tensor<2048xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<2048xf32>
    return %3 : tensor<2048xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>) -> tensor<2048xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<2048xf32>
    %2 = stablehlo.rsqrt %1 : tensor<2048xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<2048xf32>
    return %3 : tensor<2048xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.25(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.26(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.27(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.28(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.29(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.30(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.31(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.32(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.33(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.34(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.35(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.36(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.37(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.38(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.39(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.40(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.41(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.42(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.43(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.44(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.45(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.46(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.47(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.48(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.49(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.50(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.51(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.52(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func @main(%arg0: tensor<i32> {mhlo.parameter_replication = [true]}, %arg1: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg2: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg3: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg4: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg5: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg6: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg7: tensor<1x1x64x64xf32> {mhlo.parameter_replication = [true]}, %arg8: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg9: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg10: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg11: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg12: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg13: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg14: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg15: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg16: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg17: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg18: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg19: tensor<1x1x256x64xf32> {mhlo.parameter_replication = [true]}, %arg20: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg21: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg22: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg23: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg24: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg25: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg26: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg27: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg28: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg29: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg30: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg31: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg32: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg33: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg34: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg35: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg36: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg37: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg38: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg39: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg40: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg41: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg42: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg43: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg44: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg45: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg46: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg47: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg48: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg49: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg50: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg51: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg52: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg53: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg54: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg55: tensor<1x1x1024x512xf32> {mhlo.parameter_replication = [true]}, %arg56: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg57: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg58: tensor<1x1x1024x2048xf32> {mhlo.parameter_replication = [true]}, %arg59: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg60: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg61: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg62: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg63: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg64: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg65: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg66: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg67: tensor<1x1x2048x512xf32> {mhlo.parameter_replication = [true]}, %arg68: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg69: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg70: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg71: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg72: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg73: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg74: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg75: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg76: tensor<1x1x2048x512xf32> {mhlo.parameter_replication = [true]}, %arg77: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg78: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg79: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg80: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg81: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg82: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg83: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg84: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg85: tensor<1x1x256x64xf32> {mhlo.parameter_replication = [true]}, %arg86: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg87: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg88: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg89: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg90: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg91: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg92: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg93: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg94: tensor<1x1x256x128xf32> {mhlo.parameter_replication = [true]}, %arg95: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg96: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg97: tensor<1x1x256x512xf32> {mhlo.parameter_replication = [true]}, %arg98: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg99: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg100: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg101: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg102: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg103: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg104: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg105: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg106: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg107: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg108: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg109: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg110: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg111: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg112: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg113: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg114: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg115: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg116: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg117: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg118: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg119: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg120: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg121: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg122: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg123: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg124: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg125: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg126: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg127: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg128: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg129: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg130: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg131: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg132: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg133: tensor<1x1x512x256xf32> {mhlo.parameter_replication = [true]}, %arg134: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg135: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg136: tensor<1x1x512x1024xf32> {mhlo.parameter_replication = [true]}, %arg137: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg138: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg139: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg140: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg141: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg142: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg143: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg144: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg145: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg146: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg147: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg148: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg149: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg150: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg151: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg152: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg153: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg154: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg155: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg156: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg157: tensor<1000xf32> {mhlo.parameter_replication = [true]}, %arg158: tensor<2048x1000xf32> {mhlo.parameter_replication = [true]}, %arg159: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg160: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg161: tensor<7x7x3x64xf32> {mhlo.parameter_replication = [true]}, %arg162: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg163: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg164: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg165: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg166: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg167: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg168: tensor<1x1x64x64xf32> {mhlo.parameter_replication = [true]}, %arg169: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg170: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg171: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg172: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg173: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg174: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg175: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg176: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg177: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg178: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg179: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg180: tensor<1x1x256x64xf32> {mhlo.parameter_replication = [true]}, %arg181: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg182: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg183: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg184: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg185: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg186: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg187: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg188: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg189: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg190: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg191: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg192: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg193: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg194: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg195: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg196: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg197: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg198: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg199: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg200: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg201: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg202: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg203: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg204: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg205: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg206: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg207: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg208: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg209: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg210: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg211: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg212: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg213: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg214: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg215: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg216: tensor<1x1x1024x512xf32> {mhlo.parameter_replication = [true]}, %arg217: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg218: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg219: tensor<1x1x1024x2048xf32> {mhlo.parameter_replication = [true]}, %arg220: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg221: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg222: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg223: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg224: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg225: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg226: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg227: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg228: tensor<1x1x2048x512xf32> {mhlo.parameter_replication = [true]}, %arg229: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg230: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg231: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg232: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg233: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg234: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg235: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg236: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg237: tensor<1x1x2048x512xf32> {mhlo.parameter_replication = [true]}, %arg238: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg239: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg240: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg241: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg242: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg243: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg244: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg245: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg246: tensor<1x1x256x64xf32> {mhlo.parameter_replication = [true]}, %arg247: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg248: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg249: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg250: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg251: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg252: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg253: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg254: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg255: tensor<1x1x256x128xf32> {mhlo.parameter_replication = [true]}, %arg256: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg257: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg258: tensor<1x1x256x512xf32> {mhlo.parameter_replication = [true]}, %arg259: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg260: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg261: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg262: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg263: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg264: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg265: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg266: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg267: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg268: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg269: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg270: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg271: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg272: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg273: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg274: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg275: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg276: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg277: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg278: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg279: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg280: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg281: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg282: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg283: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg284: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg285: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg286: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg287: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg288: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg289: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg290: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg291: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg292: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg293: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg294: tensor<1x1x512x256xf32> {mhlo.parameter_replication = [true]}, %arg295: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg296: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg297: tensor<1x1x512x1024xf32> {mhlo.parameter_replication = [true]}, %arg298: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg299: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg300: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg301: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg302: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg303: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg304: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg305: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg306: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg307: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg308: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg309: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg310: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg311: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg312: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg313: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg314: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg315: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg316: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg317: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg318: tensor<1000xf32> {mhlo.parameter_replication = [true]}, %arg319: tensor<2048x1000xf32> {mhlo.parameter_replication = [true]}, %arg320: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg321: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg322: tensor<7x7x3x64xf32> {mhlo.parameter_replication = [true]}, %arg323: tensor<i32> {mhlo.parameter_replication = [true]}, %arg324: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg325: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg326: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg327: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg328: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg329: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg330: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg331: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg332: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg333: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg334: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg335: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg336: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg337: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg338: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg339: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg340: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg341: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg342: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg343: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg344: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg345: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg346: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg347: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg348: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg349: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg350: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg351: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg352: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg353: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg354: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg355: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg356: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg357: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg358: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg359: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg360: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg361: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg362: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg363: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg364: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg365: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg366: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg367: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg368: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg369: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg370: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg371: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg372: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg373: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg374: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg375: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg376: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg377: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg378: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg379: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg380: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg381: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg382: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg383: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg384: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg385: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg386: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg387: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg388: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg389: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg390: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg391: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg392: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg393: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg394: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg395: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg396: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg397: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg398: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg399: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg400: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg401: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg402: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg403: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg404: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg405: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg406: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg407: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg408: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg409: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg410: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg411: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg412: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg413: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg414: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg415: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg416: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg417: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg418: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg419: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg420: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg421: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg422: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg423: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg424: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg425: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg426: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg427: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg428: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg429: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg430: tensor<i32> {mhlo.parameter_replication = [true]}, %arg431: tensor<f32> {mhlo.parameter_replication = [true]}, %arg432: tensor<1x256x224x224x3xbf16>, %arg433: tensor<1x256xi32>) -> tensor<1x256x1000xbf16> {
    %0 = "mhlo.fusion"(%arg75, %arg375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<2048xf32>, %arg435: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<2048xf32>
      %224 = stablehlo.rsqrt %223 : tensor<2048xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<2048xf32>
      mhlo.return %225 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %1 = "mhlo.fusion"(%arg73, %arg373) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %2 = "mhlo.fusion"(%arg71, %arg371) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %3 = "mhlo.fusion"(%arg64, %arg367) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4 = "mhlo.fusion"(%arg62, %arg365) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %5 = "mhlo.fusion"(%arg52, %arg359) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %6 = "mhlo.fusion"(%arg50, %arg357) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7 = "mhlo.fusion"(%arg45, %arg355) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1024xf32>, %arg435: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<1024xf32>
      %224 = stablehlo.rsqrt %223 : tensor<1024xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<1024xf32>
      mhlo.return %225 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %8 = "mhlo.fusion"(%arg43, %arg353) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %9 = "mhlo.fusion"(%arg41, %arg351) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %10 = "mhlo.fusion"(%arg34, %arg347) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %11 = "mhlo.fusion"(%arg32, %arg345) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %12 = "mhlo.fusion"(%arg25, %arg341) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %13 = "mhlo.fusion"(%arg23, %arg339) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %14 = "mhlo.fusion"(%arg151, %arg425) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %15 = "mhlo.fusion"(%arg149, %arg423) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %16 = "mhlo.fusion"(%arg142, %arg419) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %17 = "mhlo.fusion"(%arg140, %arg417) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %18 = "mhlo.fusion"(%arg130, %arg411) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %19 = "mhlo.fusion"(%arg128, %arg409) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %20 = "mhlo.fusion"(%arg123, %arg407) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %21 = "mhlo.fusion"(%arg121, %arg405) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<128xf32>
      %224 = stablehlo.rsqrt %223 : tensor<128xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<128xf32>
      mhlo.return %225 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %22 = "mhlo.fusion"(%arg119, %arg403) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<128xf32>
      %224 = stablehlo.rsqrt %223 : tensor<128xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<128xf32>
      mhlo.return %225 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %23 = "mhlo.fusion"(%arg114, %arg401) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %24 = "mhlo.fusion"(%arg112, %arg399) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<128xf32>
      %224 = stablehlo.rsqrt %223 : tensor<128xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<128xf32>
      mhlo.return %225 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %25 = "mhlo.fusion"(%arg110, %arg397) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<128xf32>
      %224 = stablehlo.rsqrt %223 : tensor<128xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<128xf32>
      mhlo.return %225 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %26 = "mhlo.fusion"(%arg105, %arg395) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %27 = "mhlo.fusion"(%arg103, %arg393) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<128xf32>
      %224 = stablehlo.rsqrt %223 : tensor<128xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<128xf32>
      mhlo.return %225 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %28 = "mhlo.fusion"(%arg101, %arg391) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<128xf32>
      %224 = stablehlo.rsqrt %223 : tensor<128xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<128xf32>
      mhlo.return %225 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %29 = "mhlo.fusion"(%arg93, %arg387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %30 = "mhlo.fusion"(%arg91, %arg385) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<128xf32>
      %224 = stablehlo.rsqrt %223 : tensor<128xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<128xf32>
      mhlo.return %225 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %31 = "mhlo.fusion"(%arg89, %arg383) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<128xf32>
      %224 = stablehlo.rsqrt %223 : tensor<128xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<128xf32>
      mhlo.return %225 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %32 = "mhlo.fusion"(%arg84, %arg381) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %33 = "mhlo.fusion"(%arg82, %arg379) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<64xf32>
      %224 = stablehlo.rsqrt %223 : tensor<64xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<64xf32>
      mhlo.return %225 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %34 = "mhlo.fusion"(%arg80, %arg377) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<64xf32>
      %224 = stablehlo.rsqrt %223 : tensor<64xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<64xf32>
      mhlo.return %225 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %35 = "mhlo.fusion"(%arg18, %arg337) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %36 = "mhlo.fusion"(%arg16, %arg335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<64xf32>
      %224 = stablehlo.rsqrt %223 : tensor<64xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<64xf32>
      mhlo.return %225 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %37 = "mhlo.fusion"(%arg14, %arg333) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<64xf32>
      %224 = stablehlo.rsqrt %223 : tensor<64xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<64xf32>
      mhlo.return %225 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %38 = "mhlo.fusion"(%arg12, %arg331) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %39 = "mhlo.fusion"(%arg160, %arg429) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<64xf32>
      %224 = stablehlo.rsqrt %223 : tensor<64xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<64xf32>
      mhlo.return %225 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %40 = "mhlo.fusion"(%arg432) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1x256x224x224x3xbf16>):
      %222 = mhlo.bitcast %arg434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
      mhlo.return %223 : tensor<256x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
    %41 = stablehlo.convolution(%40, %arg161) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<256x112x112x64xf32>
    %42 = "mhlo.fusion"(%arg159, %39, %41, %arg428) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>, %arg436: tensor<256x112x112x64xf32>, %arg437: tensor<64xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x112x112x64xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x112x112x64xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x112x112x64xf32>
      %228 = stablehlo.convert %227 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %229 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
      %230 = stablehlo.maximum %228, %229 : tensor<256x112x112x64xbf16>
      %231 = stablehlo.convert %230 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
      %232 = stablehlo.convert %cst_0 : (tensor<bf16>) -> tensor<f32>
      %233 = "stablehlo.reduce_window"(%231, %232) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg438: tensor<f32>, %arg439: tensor<f32>):
        %235 = stablehlo.convert %arg438 : (tensor<f32>) -> tensor<bf16>
        %236 = stablehlo.convert %arg439 : (tensor<f32>) -> tensor<bf16>
        %237 = stablehlo.maximum %235, %236 : tensor<bf16>
        %238 = stablehlo.convert %237 : (tensor<bf16>) -> tensor<f32>
        stablehlo.return %238 : tensor<f32>
      }) : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<256x56x56x64xf32>
      %234 = stablehlo.convert %233 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      mhlo.return %234 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x112x112x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xbf16>
    %43 = "mhlo.fusion"(%42, %arg10) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<256x56x56x64xbf16>, %arg435: tensor<1x1x64x256xf32>):
      %222 = stablehlo.convert %arg435 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xbf16>
      %223 = mhlo.bitcast %222 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xbf16>) -> tensor<64x256xbf16>
      %224 = mhlo.bitcast %arg434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
      %225 = stablehlo.dot_general %223, %224, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[256,802816]{0,1}"} : (tensor<64x256xbf16>, tensor<802816x64xbf16>) -> tensor<256x802816xbf16>
      %226 = mhlo.bitcast %225 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x802816xbf16>) -> tensor<802816x256xbf16>
      %227 = mhlo.bitcast %226 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
      mhlo.return %227 : tensor<256x56x56x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x256xf32>) -> tensor<256x56x56x256xbf16>
    %44 = "mhlo.fusion"(%arg6, %arg329) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<256xf32>
      %224 = stablehlo.rsqrt %223 : tensor<256xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<256xf32>
      mhlo.return %225 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %45 = "mhlo.fusion"(%arg4, %arg327) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<64xf32>
      %224 = stablehlo.rsqrt %223 : tensor<64xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<64xf32>
      mhlo.return %225 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %46 = "mhlo.fusion"(%arg2, %arg325) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<64xf32>
      %224 = stablehlo.rsqrt %223 : tensor<64xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<64xf32>
      mhlo.return %225 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %47 = "mhlo.fusion"(%42, %arg7) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<256x56x56x64xbf16>, %arg435: tensor<1x1x64x64xf32>):
      %222 = stablehlo.convert %arg435 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xbf16>
      %223 = mhlo.bitcast %222 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x64xbf16>) -> tensor<64x64xbf16>
      %224 = mhlo.bitcast %arg434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
      %225 = stablehlo.dot_general %223, %224, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[64,802816]{0,1}"} : (tensor<64x64xbf16>, tensor<802816x64xbf16>) -> tensor<64x802816xbf16>
      %226 = mhlo.bitcast %225 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<64x802816xbf16>) -> tensor<802816x64xbf16>
      %227 = mhlo.bitcast %226 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x64xbf16>) -> tensor<256x56x56x64xbf16>
      mhlo.return %227 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x64xf32>) -> tensor<256x56x56x64xbf16>
    %48 = "mhlo.fusion"(%arg1, %46, %arg324, %47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>, %arg436: tensor<64xf32>, %arg437: tensor<256x56x56x64xbf16>):
      %222 = stablehlo.convert %arg437 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      %223 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %224 = stablehlo.subtract %222, %223 : tensor<256x56x56x64xf32>
      %225 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %226 = stablehlo.multiply %224, %225 : tensor<256x56x56x64xf32>
      %227 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %228 = stablehlo.add %226, %227 : tensor<256x56x56x64xf32>
      %229 = stablehlo.convert %228 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %230 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %231 = stablehlo.maximum %229, %230 : tensor<256x56x56x64xbf16>
      %232 = stablehlo.convert %231 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %232 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %49 = stablehlo.convolution(%48, %arg8) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %50 = "mhlo.fusion"(%45, %arg3, %49, %arg326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>, %arg436: tensor<256x56x56x64xf32>, %arg437: tensor<64xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x56x56x64xf32>
      %224 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x56x56x64xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x56x56x64xf32>
      %228 = stablehlo.convert %227 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      mhlo.return %228 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xbf16>
    %51 = mhlo.bitcast %50 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
    %52 = mhlo.bitcast %arg9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %53 = "mhlo.fusion"(%51, %52) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<802816x64xbf16>, %arg435: tensor<64x256xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<802816x64xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<802816x64xbf16>
        mhlo.return %226 : tensor<802816x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>) -> tensor<802816x64xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<64x256xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
        mhlo.return %225 : tensor<64x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<802816x64xbf16>, tensor<64x256xbf16>) -> tensor<802816x256xbf16>
      mhlo.return %224 : tensor<802816x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>, tensor<64x256xf32>) -> tensor<802816x256xbf16>
    %54 = "mhlo.fusion"(%arg11, %38, %arg330, %43, %arg5, %44, %arg328, %53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<256x56x56x256xbf16>, %arg438: tensor<256xf32>, %arg439: tensor<256xf32>, %arg440: tensor<256xf32>, %arg441: tensor<802816x256xbf16>):
      %222 = stablehlo.convert %arg437 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %223 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %224 = stablehlo.subtract %222, %223 : tensor<256x56x56x256xf32>
      %225 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %226 = stablehlo.multiply %224, %225 : tensor<256x56x56x256xf32>
      %227 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %228 = stablehlo.add %226, %227 : tensor<256x56x56x256xf32>
      %229 = mhlo.bitcast %arg441 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
      %230 = stablehlo.convert %229 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %231 = stablehlo.broadcast_in_dim %arg440, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %232 = stablehlo.subtract %230, %231 : tensor<256x56x56x256xf32>
      %233 = stablehlo.broadcast_in_dim %arg439, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %234 = stablehlo.multiply %232, %233 : tensor<256x56x56x256xf32>
      %235 = stablehlo.broadcast_in_dim %arg438, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %236 = stablehlo.add %234, %235 : tensor<256x56x56x256xf32>
      %237 = stablehlo.add %228, %236 : tensor<256x56x56x256xf32>
      %238 = stablehlo.convert %237 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
      mhlo.return %238 : tensor<256x56x56x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x56x56x256xbf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
    %55 = mhlo.bitcast %54 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x256xbf16>) -> tensor<802816x256xbf16>
    %56 = mhlo.bitcast %arg19 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x64xf32>) -> tensor<256x64xf32>
    %57 = "mhlo.fusion"(%55, %56) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<802816x256xbf16>, %arg435: tensor<256x64xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<802816x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<802816x256xbf16>
        mhlo.return %226 : tensor<802816x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>) -> tensor<802816x256xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<256x64xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
        mhlo.return %225 : tensor<256x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x64xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<802816x256xbf16>, tensor<256x64xbf16>) -> tensor<802816x64xbf16>
      mhlo.return %224 : tensor<802816x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>, tensor<256x64xf32>) -> tensor<802816x64xbf16>
    %58 = "mhlo.fusion"(%arg13, %37, %arg332, %57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>, %arg436: tensor<64xf32>, %arg437: tensor<802816x64xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x64xbf16>) -> tensor<256x56x56x64xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x56x56x64xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x56x56x64xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x56x56x64xf32>
      %230 = stablehlo.convert %229 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x56x56x64xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %233 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<802816x64xbf16>) -> tensor<256x56x56x64xf32>
    %59 = stablehlo.convolution(%58, %arg20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %60 = "mhlo.fusion"(%36, %arg15, %59, %arg334) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>, %arg436: tensor<256x56x56x64xf32>, %arg437: tensor<64xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x56x56x64xf32>
      %224 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x56x56x64xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x56x56x64xf32>
      %228 = stablehlo.convert %227 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      mhlo.return %228 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xbf16>
    %61 = mhlo.bitcast %60 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
    %62 = mhlo.bitcast %arg21 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %63 = "mhlo.fusion"(%61, %62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<802816x64xbf16>, %arg435: tensor<64x256xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<802816x64xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<802816x64xbf16>
        mhlo.return %226 : tensor<802816x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>) -> tensor<802816x64xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<64x256xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
        mhlo.return %225 : tensor<64x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<802816x64xbf16>, tensor<64x256xbf16>) -> tensor<802816x256xbf16>
      mhlo.return %224 : tensor<802816x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>, tensor<64x256xf32>) -> tensor<802816x256xbf16>
    %64 = "mhlo.fusion"(%arg17, %35, %arg336, %63, %54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<802816x256xbf16>, %arg438: tensor<256x56x56x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x256xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x56x56x256xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %225 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %227 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x56x56x256xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x56x56x256xf32>
      %231 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x56x56x256xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x56x56x256xf32>
      %234 = stablehlo.convert %233 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
      mhlo.return %234 : tensor<256x56x56x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<802816x256xbf16>, tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xbf16>
    %65 = mhlo.bitcast %64 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x256xbf16>) -> tensor<802816x256xbf16>
    %66 = mhlo.bitcast %arg85 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x64xf32>) -> tensor<256x64xf32>
    %67 = "mhlo.fusion"(%65, %66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<802816x256xbf16>, %arg435: tensor<256x64xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<802816x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<802816x256xbf16>
        mhlo.return %226 : tensor<802816x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>) -> tensor<802816x256xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<256x64xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
        mhlo.return %225 : tensor<256x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x64xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<802816x256xbf16>, tensor<256x64xbf16>) -> tensor<802816x64xbf16>
      mhlo.return %224 : tensor<802816x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>, tensor<256x64xf32>) -> tensor<802816x64xbf16>
    %68 = "mhlo.fusion"(%arg79, %34, %arg376, %67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>, %arg436: tensor<64xf32>, %arg437: tensor<802816x64xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x64xbf16>) -> tensor<256x56x56x64xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x56x56x64xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x56x56x64xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x56x56x64xf32>
      %230 = stablehlo.convert %229 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x56x56x64xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %233 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<802816x64xbf16>) -> tensor<256x56x56x64xf32>
    %69 = stablehlo.convolution(%68, %arg86) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %70 = "mhlo.fusion"(%33, %arg81, %69, %arg378) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<64xf32>, %arg435: tensor<64xf32>, %arg436: tensor<256x56x56x64xf32>, %arg437: tensor<64xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x56x56x64xf32>
      %224 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x56x56x64xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x56x56x64xf32>
      %228 = stablehlo.convert %227 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      mhlo.return %228 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xbf16>
    %71 = mhlo.bitcast %70 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
    %72 = mhlo.bitcast %arg87 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %73 = "mhlo.fusion"(%71, %72) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<802816x64xbf16>, %arg435: tensor<64x256xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<802816x64xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<802816x64xbf16>
        mhlo.return %226 : tensor<802816x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>) -> tensor<802816x64xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<64x256xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
        mhlo.return %225 : tensor<64x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<802816x64xbf16>, tensor<64x256xbf16>) -> tensor<802816x256xbf16>
      mhlo.return %224 : tensor<802816x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>, tensor<64x256xf32>) -> tensor<802816x256xbf16>
    %74:2 = "mhlo.fusion"(%arg83, %32, %arg380, %73, %64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<802816x256xbf16>, %arg438: tensor<256x56x56x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x256xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x56x56x256xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %225 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %227 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x56x56x256xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x56x56x256xf32>
      %231 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x56x56x256xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x56x56x256xf32>
      %234 = stablehlo.convert %233 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
      %235 = stablehlo.maximum %234, %222 : tensor<256x56x56x256xbf16>
      %236 = stablehlo.convert %235 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      mhlo.return %236, %234 : tensor<256x56x56x256xf32>, tensor<256x56x56x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<802816x256xbf16>, tensor<256x56x56x256xbf16>) -> (tensor<256x56x56x256xf32>, tensor<256x56x56x256xbf16>)
    %75 = mhlo.bitcast %74#1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x256xbf16>) -> tensor<802816x256xbf16>
    %76 = mhlo.bitcast %arg94 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %77 = "mhlo.fusion"(%75, %76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<802816x256xbf16>, %arg435: tensor<256x128xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<802816x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<802816x256xbf16>
        mhlo.return %226 : tensor<802816x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>) -> tensor<802816x256xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<256x128xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<256x128xf32>) -> tensor<256x128xbf16>
        mhlo.return %225 : tensor<256x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x128xf32>) -> tensor<256x128xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<802816x256xbf16>, tensor<256x128xbf16>) -> tensor<802816x128xbf16>
      mhlo.return %224 : tensor<802816x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>, tensor<256x128xf32>) -> tensor<802816x128xbf16>
    %78 = "mhlo.fusion"(%arg88, %31, %arg382, %77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>, %arg436: tensor<128xf32>, %arg437: tensor<802816x128xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x128xbf16>) -> tensor<256x56x56x128xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<128xf32>) -> tensor<256x56x56x128xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x56x56x128xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<128xf32>) -> tensor<256x56x56x128xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x56x56x128xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<128xf32>) -> tensor<256x56x56x128xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x56x56x128xf32>
      %230 = stablehlo.convert %229 : (tensor<256x56x56x128xf32>) -> tensor<256x56x56x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x128xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x56x56x128xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xf32>
      mhlo.return %233 : tensor<256x56x56x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<802816x128xbf16>) -> tensor<256x56x56x128xf32>
    %79 = stablehlo.convolution(%78, %arg95) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %80 = "mhlo.fusion"(%30, %arg90, %79, %arg384) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>, %arg436: tensor<256x28x28x128xf32>, %arg437: tensor<128xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x28x28x128xf32>
      %224 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x28x28x128xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x28x28x128xf32>
      %228 = stablehlo.convert %227 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      mhlo.return %228 : tensor<256x28x28x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xbf16>
    %81 = mhlo.bitcast %80 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x128xbf16>) -> tensor<200704x128xbf16>
    %82 = mhlo.bitcast %arg96 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %83 = "mhlo.fusion"(%81, %82) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<200704x128xbf16>, %arg435: tensor<128x512xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<200704x128xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<200704x128xbf16>
        mhlo.return %226 : tensor<200704x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<128x512xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
        mhlo.return %225 : tensor<128x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
      mhlo.return %224 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>, tensor<128x512xf32>) -> tensor<200704x512xbf16>
    %84 = "mhlo.fusion"(%arg99, %arg389) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<512xf32>
      %224 = stablehlo.rsqrt %223 : tensor<512xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<512xf32>
      mhlo.return %225 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %85 = stablehlo.convolution(%74#0, %arg97) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf32>, tensor<1x1x256x512xf32>) -> tensor<256x28x28x512xf32>
    %86 = "mhlo.fusion"(%arg92, %29, %arg386, %83, %arg98, %84, %85, %arg388) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<512xf32>, %arg437: tensor<200704x512xbf16>, %arg438: tensor<512xf32>, %arg439: tensor<512xf32>, %arg440: tensor<256x28x28x512xf32>, %arg441: tensor<512xf32>):
      %222 = stablehlo.broadcast_in_dim %arg441, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %223 = stablehlo.subtract %arg440, %222 : tensor<256x28x28x512xf32>
      %224 = stablehlo.broadcast_in_dim %arg439, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x28x28x512xf32>
      %226 = stablehlo.broadcast_in_dim %arg438, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x28x28x512xf32>
      %228 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
      %229 = stablehlo.convert %228 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %230 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %231 = stablehlo.subtract %229, %230 : tensor<256x28x28x512xf32>
      %232 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %233 = stablehlo.multiply %231, %232 : tensor<256x28x28x512xf32>
      %234 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %235 = stablehlo.add %233, %234 : tensor<256x28x28x512xf32>
      %236 = stablehlo.add %227, %235 : tensor<256x28x28x512xf32>
      %237 = stablehlo.convert %236 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
      mhlo.return %237 : tensor<256x28x28x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<200704x512xbf16>, tensor<512xf32>, tensor<512xf32>, tensor<256x28x28x512xf32>, tensor<512xf32>) -> tensor<256x28x28x512xbf16>
    %87 = mhlo.bitcast %86 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x512xbf16>) -> tensor<200704x512xbf16>
    %88 = mhlo.bitcast %arg106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x128xf32>) -> tensor<512x128xf32>
    %89 = "mhlo.fusion"(%87, %88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<200704x512xbf16>, %arg435: tensor<512x128xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<200704x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<200704x512xbf16>
        mhlo.return %226 : tensor<200704x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<512x128xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
        mhlo.return %225 : tensor<512x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x128xf32>) -> tensor<512x128xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x128xbf16>) -> tensor<200704x128xbf16>
      mhlo.return %224 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>, tensor<512x128xf32>) -> tensor<200704x128xbf16>
    %90 = "mhlo.fusion"(%arg100, %28, %arg390, %89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>, %arg436: tensor<128xf32>, %arg437: tensor<200704x128xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x128xbf16>) -> tensor<256x28x28x128xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x28x28x128xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x28x28x128xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x28x28x128xf32>
      %230 = stablehlo.convert %229 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x28x28x128xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %233 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32>
    %91 = stablehlo.convolution(%90, %arg107) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %92 = "mhlo.fusion"(%27, %arg102, %91, %arg392) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>, %arg436: tensor<256x28x28x128xf32>, %arg437: tensor<128xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x28x28x128xf32>
      %224 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x28x28x128xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x28x28x128xf32>
      %228 = stablehlo.convert %227 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      mhlo.return %228 : tensor<256x28x28x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xbf16>
    %93 = mhlo.bitcast %92 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x128xbf16>) -> tensor<200704x128xbf16>
    %94 = mhlo.bitcast %arg108 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %95 = "mhlo.fusion"(%93, %94) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<200704x128xbf16>, %arg435: tensor<128x512xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<200704x128xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<200704x128xbf16>
        mhlo.return %226 : tensor<200704x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<128x512xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
        mhlo.return %225 : tensor<128x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
      mhlo.return %224 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>, tensor<128x512xf32>) -> tensor<200704x512xbf16>
    %96 = "mhlo.fusion"(%arg104, %26, %arg394, %95, %86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<512xf32>, %arg437: tensor<200704x512xbf16>, %arg438: tensor<256x28x28x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x28x28x512xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %225 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %227 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x28x28x512xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x28x28x512xf32>
      %231 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x28x28x512xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x28x28x512xf32>
      %234 = stablehlo.convert %233 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
      mhlo.return %234 : tensor<256x28x28x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<200704x512xbf16>, tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %97 = mhlo.bitcast %96 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x512xbf16>) -> tensor<200704x512xbf16>
    %98 = mhlo.bitcast %arg115 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x128xf32>) -> tensor<512x128xf32>
    %99 = "mhlo.fusion"(%97, %98) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<200704x512xbf16>, %arg435: tensor<512x128xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<200704x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<200704x512xbf16>
        mhlo.return %226 : tensor<200704x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<512x128xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
        mhlo.return %225 : tensor<512x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x128xf32>) -> tensor<512x128xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x128xbf16>) -> tensor<200704x128xbf16>
      mhlo.return %224 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>, tensor<512x128xf32>) -> tensor<200704x128xbf16>
    %100 = "mhlo.fusion"(%arg109, %25, %arg396, %99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>, %arg436: tensor<128xf32>, %arg437: tensor<200704x128xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x128xbf16>) -> tensor<256x28x28x128xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x28x28x128xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x28x28x128xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x28x28x128xf32>
      %230 = stablehlo.convert %229 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x28x28x128xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %233 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32>
    %101 = stablehlo.convolution(%100, %arg116) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %102 = "mhlo.fusion"(%24, %arg111, %101, %arg398) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>, %arg436: tensor<256x28x28x128xf32>, %arg437: tensor<128xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x28x28x128xf32>
      %224 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x28x28x128xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x28x28x128xf32>
      %228 = stablehlo.convert %227 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      mhlo.return %228 : tensor<256x28x28x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xbf16>
    %103 = mhlo.bitcast %102 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x128xbf16>) -> tensor<200704x128xbf16>
    %104 = mhlo.bitcast %arg117 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %105 = "mhlo.fusion"(%103, %104) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<200704x128xbf16>, %arg435: tensor<128x512xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<200704x128xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<200704x128xbf16>
        mhlo.return %226 : tensor<200704x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<128x512xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
        mhlo.return %225 : tensor<128x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
      mhlo.return %224 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>, tensor<128x512xf32>) -> tensor<200704x512xbf16>
    %106 = "mhlo.fusion"(%arg113, %23, %arg400, %105, %96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<512xf32>, %arg437: tensor<200704x512xbf16>, %arg438: tensor<256x28x28x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x28x28x512xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %225 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %227 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x28x28x512xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x28x28x512xf32>
      %231 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x28x28x512xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x28x28x512xf32>
      %234 = stablehlo.convert %233 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
      mhlo.return %234 : tensor<256x28x28x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<200704x512xbf16>, tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %107 = mhlo.bitcast %106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x512xbf16>) -> tensor<200704x512xbf16>
    %108 = mhlo.bitcast %arg124 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x128xf32>) -> tensor<512x128xf32>
    %109 = "mhlo.fusion"(%107, %108) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<200704x512xbf16>, %arg435: tensor<512x128xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<200704x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<200704x512xbf16>
        mhlo.return %226 : tensor<200704x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<512x128xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
        mhlo.return %225 : tensor<512x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x128xf32>) -> tensor<512x128xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x128xbf16>) -> tensor<200704x128xbf16>
      mhlo.return %224 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>, tensor<512x128xf32>) -> tensor<200704x128xbf16>
    %110 = "mhlo.fusion"(%arg118, %22, %arg402, %109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>, %arg436: tensor<128xf32>, %arg437: tensor<200704x128xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x128xbf16>) -> tensor<256x28x28x128xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x28x28x128xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x28x28x128xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x28x28x128xf32>
      %230 = stablehlo.convert %229 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x28x28x128xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %233 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32>
    %111 = stablehlo.convolution(%110, %arg125) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %112 = "mhlo.fusion"(%21, %arg120, %111, %arg404) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<128xf32>, %arg435: tensor<128xf32>, %arg436: tensor<256x28x28x128xf32>, %arg437: tensor<128xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x28x28x128xf32>
      %224 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x28x28x128xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x28x28x128xf32>
      %228 = stablehlo.convert %227 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      mhlo.return %228 : tensor<256x28x28x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xbf16>
    %113 = mhlo.bitcast %112 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x128xbf16>) -> tensor<200704x128xbf16>
    %114 = mhlo.bitcast %arg126 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %115 = "mhlo.fusion"(%113, %114) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<200704x128xbf16>, %arg435: tensor<128x512xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<200704x128xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<200704x128xbf16>
        mhlo.return %226 : tensor<200704x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<128x512xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
        mhlo.return %225 : tensor<128x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
      mhlo.return %224 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>, tensor<128x512xf32>) -> tensor<200704x512xbf16>
    %116:2 = "mhlo.fusion"(%arg122, %20, %arg406, %115, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<512xf32>, %arg437: tensor<200704x512xbf16>, %arg438: tensor<256x28x28x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x28x28x512xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %225 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %227 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x28x28x512xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x28x28x512xf32>
      %231 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x28x28x512xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x28x28x512xf32>
      %234 = stablehlo.convert %233 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
      %235 = stablehlo.maximum %234, %222 : tensor<256x28x28x512xbf16>
      %236 = stablehlo.convert %235 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      mhlo.return %236, %234 : tensor<256x28x28x512xf32>, tensor<256x28x28x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<200704x512xbf16>, tensor<256x28x28x512xbf16>) -> (tensor<256x28x28x512xf32>, tensor<256x28x28x512xbf16>)
    %117 = mhlo.bitcast %116#1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x512xbf16>) -> tensor<200704x512xbf16>
    %118 = mhlo.bitcast %arg133 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %119 = "mhlo.fusion"(%117, %118) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<200704x512xbf16>, %arg435: tensor<512x256xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<200704x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<200704x512xbf16>
        mhlo.return %226 : tensor<200704x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<512x256xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<512x256xf32>) -> tensor<512x256xbf16>
        mhlo.return %225 : tensor<512x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x256xf32>) -> tensor<512x256xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x256xbf16>) -> tensor<200704x256xbf16>
      mhlo.return %224 : tensor<200704x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>, tensor<512x256xf32>) -> tensor<200704x256xbf16>
    %120 = "mhlo.fusion"(%arg127, %19, %arg408, %119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<200704x256xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x256xbf16>) -> tensor<256x28x28x256xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<256x28x28x256xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x28x28x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x28x28x256xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x28x28x256xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x28x28x256xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x28x28x256xf32>
      %230 = stablehlo.convert %229 : (tensor<256x28x28x256xf32>) -> tensor<256x28x28x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x256xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x28x28x256xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xf32>
      mhlo.return %233 : tensor<256x28x28x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<200704x256xbf16>) -> tensor<256x28x28x256xf32>
    %121 = stablehlo.convolution(%120, %arg134) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %122 = "mhlo.fusion"(%arg129, %18, %121, %arg410) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256x14x14x256xf32>, %arg437: tensor<256xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.convert %227 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %228 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %123 = mhlo.bitcast %122 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %124 = mhlo.bitcast %arg135 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %125 = "mhlo.fusion"(%123, %124) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x256xbf16>, %arg435: tensor<256x1024xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x256xbf16>
        mhlo.return %226 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<256x1024xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %225 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %224 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %126 = "mhlo.fusion"(%arg132, %arg413) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1024xf32>, %arg435: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<1024xf32>
      %224 = stablehlo.rsqrt %223 : tensor<1024xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<1024xf32>
      mhlo.return %225 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %127 = "mhlo.fusion"(%arg138, %arg415) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1024xf32>, %arg435: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<1024xf32>
      %224 = stablehlo.rsqrt %223 : tensor<1024xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<1024xf32>
      mhlo.return %225 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %128 = stablehlo.convolution(%116#0, %arg136) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf32>, tensor<1x1x512x1024xf32>) -> tensor<256x14x14x1024xf32>
    %129 = "mhlo.fusion"(%125, %126, %127, %arg131, %arg412, %arg137, %128, %arg414) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024xf32>, %arg436: tensor<1024xf32>, %arg437: tensor<1024xf32>, %arg438: tensor<1024xf32>, %arg439: tensor<1024xf32>, %arg440: tensor<256x14x14x1024xf32>, %arg441: tensor<1024xf32>):
      %222 = stablehlo.broadcast_in_dim %arg441, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %223 = stablehlo.subtract %arg440, %222 : tensor<256x14x14x1024xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x14x14x1024xf32>
      %226 = stablehlo.broadcast_in_dim %arg439, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x14x14x1024xf32>
      %228 = mhlo.bitcast %arg434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %229 = stablehlo.convert %228 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %230 = stablehlo.broadcast_in_dim %arg438, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %231 = stablehlo.subtract %229, %230 : tensor<256x14x14x1024xf32>
      %232 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %233 = stablehlo.multiply %231, %232 : tensor<256x14x14x1024xf32>
      %234 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %235 = stablehlo.add %233, %234 : tensor<256x14x14x1024xf32>
      %236 = stablehlo.add %227, %235 : tensor<256x14x14x1024xf32>
      %237 = stablehlo.convert %236 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %237 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xf32>, tensor<1024xf32>) -> tensor<256x14x14x1024xbf16>
    %130 = mhlo.bitcast %129 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %131 = mhlo.bitcast %arg145 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %132 = "mhlo.fusion"(%130, %131) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024x256xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x1024xbf16>
        mhlo.return %226 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<1024x256xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %225 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %224 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %133 = "mhlo.fusion"(%arg139, %17, %arg416, %132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<50176x256xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x14x14x256xf32>
      %230 = stablehlo.convert %229 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x14x14x256xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %233 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %134 = stablehlo.convolution(%133, %arg146) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %135 = "mhlo.fusion"(%arg141, %16, %134, %arg418) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256x14x14x256xf32>, %arg437: tensor<256xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.convert %227 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %228 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %136 = mhlo.bitcast %135 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %137 = mhlo.bitcast %arg147 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %138 = "mhlo.fusion"(%136, %137) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x256xbf16>, %arg435: tensor<256x1024xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x256xbf16>
        mhlo.return %226 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<256x1024xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %225 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %224 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %139 = "mhlo.fusion"(%arg144, %arg421) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1024xf32>, %arg435: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<1024xf32>
      %224 = stablehlo.rsqrt %223 : tensor<1024xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<1024xf32>
      mhlo.return %225 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %140 = "mhlo.fusion"(%138, %139, %arg143, %arg420, %129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024xf32>, %arg436: tensor<1024xf32>, %arg437: tensor<1024xf32>, %arg438: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x14x14x1024xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %225 = mhlo.bitcast %arg434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %227 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x14x14x1024xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x14x14x1024xf32>
      %231 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x14x14x1024xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x14x14x1024xf32>
      %234 = stablehlo.convert %233 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %234 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %141 = mhlo.bitcast %140 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %142 = mhlo.bitcast %arg154 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %143 = "mhlo.fusion"(%141, %142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024x256xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x1024xbf16>
        mhlo.return %226 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<1024x256xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %225 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %224 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %144 = "mhlo.fusion"(%arg148, %15, %arg422, %143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<50176x256xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x14x14x256xf32>
      %230 = stablehlo.convert %229 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x14x14x256xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %233 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %145 = stablehlo.convolution(%144, %arg155) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %146 = "mhlo.fusion"(%arg150, %14, %145, %arg424) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256x14x14x256xf32>, %arg437: tensor<256xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.convert %227 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %228 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %147 = mhlo.bitcast %146 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %148 = mhlo.bitcast %arg156 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %149 = "mhlo.fusion"(%147, %148) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x256xbf16>, %arg435: tensor<256x1024xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x256xbf16>
        mhlo.return %226 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<256x1024xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %225 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %224 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %150 = "mhlo.fusion"(%arg153, %arg427) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1024xf32>, %arg435: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<1024xf32>
      %224 = stablehlo.rsqrt %223 : tensor<1024xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<1024xf32>
      mhlo.return %225 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %151 = "mhlo.fusion"(%149, %150, %arg152, %arg426, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024xf32>, %arg436: tensor<1024xf32>, %arg437: tensor<1024xf32>, %arg438: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x14x14x1024xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %225 = mhlo.bitcast %arg434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %227 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x14x14x1024xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x14x14x1024xf32>
      %231 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x14x14x1024xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x14x14x1024xf32>
      %234 = stablehlo.convert %233 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %234 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %152 = mhlo.bitcast %151 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %153 = mhlo.bitcast %arg28 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %154 = "mhlo.fusion"(%152, %153) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024x256xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x1024xbf16>
        mhlo.return %226 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<1024x256xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %225 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %224 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %155 = "mhlo.fusion"(%arg22, %13, %arg338, %154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<50176x256xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x14x14x256xf32>
      %230 = stablehlo.convert %229 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x14x14x256xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %233 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %156 = stablehlo.convolution(%155, %arg29) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %157 = "mhlo.fusion"(%arg24, %12, %156, %arg340) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256x14x14x256xf32>, %arg437: tensor<256xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.convert %227 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %228 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %158 = mhlo.bitcast %157 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %159 = mhlo.bitcast %arg30 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %160 = "mhlo.fusion"(%158, %159) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x256xbf16>, %arg435: tensor<256x1024xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x256xbf16>
        mhlo.return %226 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<256x1024xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %225 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %224 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %161 = "mhlo.fusion"(%arg27, %arg343) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1024xf32>, %arg435: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<1024xf32>
      %224 = stablehlo.rsqrt %223 : tensor<1024xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<1024xf32>
      mhlo.return %225 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %162 = "mhlo.fusion"(%160, %161, %arg26, %arg342, %151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024xf32>, %arg436: tensor<1024xf32>, %arg437: tensor<1024xf32>, %arg438: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x14x14x1024xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %225 = mhlo.bitcast %arg434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %227 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x14x14x1024xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x14x14x1024xf32>
      %231 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x14x14x1024xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x14x14x1024xf32>
      %234 = stablehlo.convert %233 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %234 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %163 = mhlo.bitcast %162 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %164 = mhlo.bitcast %arg37 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %165 = "mhlo.fusion"(%163, %164) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024x256xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x1024xbf16>
        mhlo.return %226 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<1024x256xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %225 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %224 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %166 = "mhlo.fusion"(%arg31, %11, %arg344, %165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<50176x256xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x14x14x256xf32>
      %230 = stablehlo.convert %229 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x14x14x256xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %233 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %167 = stablehlo.convolution(%166, %arg38) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %168 = "mhlo.fusion"(%arg33, %10, %167, %arg346) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256x14x14x256xf32>, %arg437: tensor<256xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.convert %227 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %228 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %169 = mhlo.bitcast %168 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %170 = mhlo.bitcast %arg39 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %171 = "mhlo.fusion"(%169, %170) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x256xbf16>, %arg435: tensor<256x1024xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x256xbf16>
        mhlo.return %226 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<256x1024xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %225 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %224 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %172 = "mhlo.fusion"(%arg36, %arg349) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1024xf32>, %arg435: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<1024xf32>
      %224 = stablehlo.rsqrt %223 : tensor<1024xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<1024xf32>
      mhlo.return %225 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %173 = "mhlo.fusion"(%171, %172, %arg35, %arg348, %162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024xf32>, %arg436: tensor<1024xf32>, %arg437: tensor<1024xf32>, %arg438: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x14x14x1024xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %225 = mhlo.bitcast %arg434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %227 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x14x14x1024xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x14x14x1024xf32>
      %231 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x14x14x1024xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x14x14x1024xf32>
      %234 = stablehlo.convert %233 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %234 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %174 = mhlo.bitcast %173 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %175 = mhlo.bitcast %arg46 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %176 = "mhlo.fusion"(%174, %175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024x256xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x1024xbf16>
        mhlo.return %226 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<1024x256xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %225 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %224 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %177 = "mhlo.fusion"(%arg40, %9, %arg350, %176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<50176x256xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x14x14x256xf32>
      %230 = stablehlo.convert %229 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x14x14x256xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %233 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %178 = stablehlo.convolution(%177, %arg47) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %179 = "mhlo.fusion"(%arg42, %8, %178, %arg352) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256x14x14x256xf32>, %arg437: tensor<256xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x14x14x256xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x14x14x256xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x14x14x256xf32>
      %228 = stablehlo.convert %227 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %228 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %180 = mhlo.bitcast %179 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %181 = mhlo.bitcast %arg48 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %182 = "mhlo.fusion"(%180, %181) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x256xbf16>, %arg435: tensor<256x1024xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x256xbf16>
        mhlo.return %226 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<256x1024xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %225 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %224 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %183:2 = "mhlo.fusion"(%arg44, %7, %arg354, %182, %173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1024xf32>, %arg435: tensor<1024xf32>, %arg436: tensor<1024xf32>, %arg437: tensor<50176x1024xbf16>, %arg438: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x14x14x1024xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %225 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %227 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x14x14x1024xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x14x14x1024xf32>
      %231 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x14x14x1024xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x14x14x1024xf32>
      %234 = stablehlo.convert %233 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      %235 = stablehlo.maximum %234, %222 : tensor<256x14x14x1024xbf16>
      %236 = stablehlo.convert %235 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      mhlo.return %236, %234 : tensor<256x14x14x1024xf32>, tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<50176x1024xbf16>, tensor<256x14x14x1024xbf16>) -> (tensor<256x14x14x1024xf32>, tensor<256x14x14x1024xbf16>)
    %184 = mhlo.bitcast %183#1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %185 = mhlo.bitcast %arg55 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x512xf32>) -> tensor<1024x512xf32>
    %186 = "mhlo.fusion"(%184, %185) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<50176x1024xbf16>, %arg435: tensor<1024x512xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<50176x1024xbf16>
        mhlo.return %226 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<1024x512xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<1024x512xf32>) -> tensor<1024x512xbf16>
        mhlo.return %225 : tensor<1024x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x512xf32>) -> tensor<1024x512xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x512xbf16>) -> tensor<50176x512xbf16>
      mhlo.return %224 : tensor<50176x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x512xf32>) -> tensor<50176x512xbf16>
    %187 = "mhlo.fusion"(%arg49, %6, %arg356, %186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<512xf32>, %arg437: tensor<50176x512xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x512xbf16>) -> tensor<256x14x14x512xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<512xf32>) -> tensor<256x14x14x512xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x14x14x512xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x14x14x512xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x14x14x512xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x14x14x512xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x14x14x512xf32>
      %230 = stablehlo.convert %229 : (tensor<256x14x14x512xf32>) -> tensor<256x14x14x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x512xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x14x14x512xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xf32>
      mhlo.return %233 : tensor<256x14x14x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<50176x512xbf16>) -> tensor<256x14x14x512xf32>
    %188 = stablehlo.convolution(%187, %arg56) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %189 = "mhlo.fusion"(%arg51, %5, %188, %arg358) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<256x7x7x512xf32>, %arg437: tensor<512xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x7x7x512xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x7x7x512xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x7x7x512xf32>
      %228 = stablehlo.convert %227 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      mhlo.return %228 : tensor<256x7x7x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xbf16>
    %190 = mhlo.bitcast %189 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x512xbf16>) -> tensor<12544x512xbf16>
    %191 = mhlo.bitcast %arg57 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x2048xf32>) -> tensor<512x2048xf32>
    %192 = "mhlo.fusion"(%190, %191) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<12544x512xbf16>, %arg435: tensor<512x2048xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<12544x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<12544x512xbf16>
        mhlo.return %226 : tensor<12544x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>) -> tensor<12544x512xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<512x2048xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
        mhlo.return %225 : tensor<512x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<12544x512xbf16>, tensor<512x2048xbf16>) -> tensor<12544x2048xbf16>
      mhlo.return %224 : tensor<12544x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>, tensor<512x2048xf32>) -> tensor<12544x2048xbf16>
    %193 = "mhlo.fusion"(%arg54, %arg361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<2048xf32>, %arg435: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<2048xf32>
      %224 = stablehlo.rsqrt %223 : tensor<2048xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<2048xf32>
      mhlo.return %225 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %194 = "mhlo.fusion"(%arg60, %arg363) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<2048xf32>, %arg435: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<2048xf32>
      %224 = stablehlo.rsqrt %223 : tensor<2048xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<2048xf32>
      mhlo.return %225 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %195 = stablehlo.convolution(%183#0, %arg58) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf32>, tensor<1x1x1024x2048xf32>) -> tensor<256x7x7x2048xf32>
    %196 = "mhlo.fusion"(%192, %193, %194, %arg53, %arg360, %arg59, %195, %arg362) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<12544x2048xbf16>, %arg435: tensor<2048xf32>, %arg436: tensor<2048xf32>, %arg437: tensor<2048xf32>, %arg438: tensor<2048xf32>, %arg439: tensor<2048xf32>, %arg440: tensor<256x7x7x2048xf32>, %arg441: tensor<2048xf32>):
      %222 = stablehlo.broadcast_in_dim %arg441, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %223 = stablehlo.subtract %arg440, %222 : tensor<256x7x7x2048xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x7x7x2048xf32>
      %226 = stablehlo.broadcast_in_dim %arg439, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x7x7x2048xf32>
      %228 = mhlo.bitcast %arg434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x2048xbf16>) -> tensor<256x7x7x2048xbf16>
      %229 = stablehlo.convert %228 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %230 = stablehlo.broadcast_in_dim %arg438, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %231 = stablehlo.subtract %229, %230 : tensor<256x7x7x2048xf32>
      %232 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %233 = stablehlo.multiply %231, %232 : tensor<256x7x7x2048xf32>
      %234 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %235 = stablehlo.add %233, %234 : tensor<256x7x7x2048xf32>
      %236 = stablehlo.add %227, %235 : tensor<256x7x7x2048xf32>
      %237 = stablehlo.convert %236 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
      mhlo.return %237 : tensor<256x7x7x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<256x7x7x2048xf32>, tensor<2048xf32>) -> tensor<256x7x7x2048xbf16>
    %197 = mhlo.bitcast %196 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x2048xbf16>) -> tensor<12544x2048xbf16>
    %198 = mhlo.bitcast %arg67 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2048x512xf32>) -> tensor<2048x512xf32>
    %199 = "mhlo.fusion"(%197, %198) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<12544x2048xbf16>, %arg435: tensor<2048x512xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<12544x2048xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<12544x2048xbf16>
        mhlo.return %226 : tensor<12544x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<2048x512xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
        mhlo.return %225 : tensor<2048x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<12544x2048xbf16>, tensor<2048x512xbf16>) -> tensor<12544x512xbf16>
      mhlo.return %224 : tensor<12544x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>, tensor<2048x512xf32>) -> tensor<12544x512xbf16>
    %200 = "mhlo.fusion"(%arg61, %4, %arg364, %199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<512xf32>, %arg437: tensor<12544x512xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x512xbf16>) -> tensor<256x7x7x512xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x7x7x512xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x7x7x512xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x7x7x512xf32>
      %230 = stablehlo.convert %229 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x7x7x512xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %233 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<12544x512xbf16>) -> tensor<256x7x7x512xf32>
    %201 = stablehlo.convolution(%200, %arg68) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %202 = "mhlo.fusion"(%arg63, %3, %201, %arg366) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<256x7x7x512xf32>, %arg437: tensor<512xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x7x7x512xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x7x7x512xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x7x7x512xf32>
      %228 = stablehlo.convert %227 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      mhlo.return %228 : tensor<256x7x7x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xbf16>
    %203 = mhlo.bitcast %202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x512xbf16>) -> tensor<12544x512xbf16>
    %204 = mhlo.bitcast %arg69 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x2048xf32>) -> tensor<512x2048xf32>
    %205 = "mhlo.fusion"(%203, %204) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<12544x512xbf16>, %arg435: tensor<512x2048xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<12544x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<12544x512xbf16>
        mhlo.return %226 : tensor<12544x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>) -> tensor<12544x512xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<512x2048xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
        mhlo.return %225 : tensor<512x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<12544x512xbf16>, tensor<512x2048xbf16>) -> tensor<12544x2048xbf16>
      mhlo.return %224 : tensor<12544x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>, tensor<512x2048xf32>) -> tensor<12544x2048xbf16>
    %206 = "mhlo.fusion"(%arg66, %arg369) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<2048xf32>, %arg435: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %223 = stablehlo.add %arg435, %222 : tensor<2048xf32>
      %224 = stablehlo.rsqrt %223 : tensor<2048xf32>
      %225 = stablehlo.multiply %224, %arg434 : tensor<2048xf32>
      mhlo.return %225 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %207 = "mhlo.fusion"(%205, %206, %arg65, %arg368, %196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<12544x2048xbf16>, %arg435: tensor<2048xf32>, %arg436: tensor<2048xf32>, %arg437: tensor<2048xf32>, %arg438: tensor<256x7x7x2048xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x2048xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x7x7x2048xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %225 = mhlo.bitcast %arg434 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x2048xbf16>) -> tensor<256x7x7x2048xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %227 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x7x7x2048xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x7x7x2048xf32>
      %231 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x7x7x2048xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x7x7x2048xf32>
      %234 = stablehlo.convert %233 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
      mhlo.return %234 : tensor<256x7x7x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %208 = mhlo.bitcast %207 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x2048xbf16>) -> tensor<12544x2048xbf16>
    %209 = mhlo.bitcast %arg76 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2048x512xf32>) -> tensor<2048x512xf32>
    %210 = "mhlo.fusion"(%208, %209) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<12544x2048xbf16>, %arg435: tensor<2048x512xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<12544x2048xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<12544x2048xbf16>
        mhlo.return %226 : tensor<12544x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<2048x512xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
        mhlo.return %225 : tensor<2048x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<12544x2048xbf16>, tensor<2048x512xbf16>) -> tensor<12544x512xbf16>
      mhlo.return %224 : tensor<12544x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>, tensor<2048x512xf32>) -> tensor<12544x512xbf16>
    %211 = "mhlo.fusion"(%arg70, %2, %arg370, %210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<512xf32>, %arg437: tensor<12544x512xbf16>):
      %222 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x512xbf16>) -> tensor<256x7x7x512xbf16>
      %223 = stablehlo.convert %222 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      %224 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %225 = stablehlo.subtract %223, %224 : tensor<256x7x7x512xf32>
      %226 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<256x7x7x512xf32>
      %228 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %229 = stablehlo.add %227, %228 : tensor<256x7x7x512xf32>
      %230 = stablehlo.convert %229 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %231 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %232 = stablehlo.maximum %230, %231 : tensor<256x7x7x512xbf16>
      %233 = stablehlo.convert %232 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %233 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<12544x512xbf16>) -> tensor<256x7x7x512xf32>
    %212 = stablehlo.convolution(%211, %arg77) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %213 = "mhlo.fusion"(%arg72, %1, %212, %arg372) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<512xf32>, %arg435: tensor<512xf32>, %arg436: tensor<256x7x7x512xf32>, %arg437: tensor<512xf32>):
      %222 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %223 = stablehlo.subtract %arg436, %222 : tensor<256x7x7x512xf32>
      %224 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %225 = stablehlo.multiply %223, %224 : tensor<256x7x7x512xf32>
      %226 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %227 = stablehlo.add %225, %226 : tensor<256x7x7x512xf32>
      %228 = stablehlo.convert %227 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      mhlo.return %228 : tensor<256x7x7x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xbf16>
    %214 = mhlo.bitcast %213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x512xbf16>) -> tensor<12544x512xbf16>
    %215 = mhlo.bitcast %arg78 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x2048xf32>) -> tensor<512x2048xf32>
    %216 = "mhlo.fusion"(%214, %215) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<12544x512xbf16>, %arg435: tensor<512x2048xf32>):
      %222 = "mhlo.fusion"(%arg434) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<12544x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %225 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
        %226 = stablehlo.maximum %arg436, %225 : tensor<12544x512xbf16>
        mhlo.return %226 : tensor<12544x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>) -> tensor<12544x512xbf16>
      %223 = "mhlo.fusion"(%arg435) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg436: tensor<512x2048xf32>):
        %225 = stablehlo.convert %arg436 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
        mhlo.return %225 : tensor<512x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
      %224 = stablehlo.dot %222, %223, precision = [DEFAULT, DEFAULT] : (tensor<12544x512xbf16>, tensor<512x2048xbf16>) -> tensor<12544x2048xbf16>
      mhlo.return %224 : tensor<12544x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>, tensor<512x2048xf32>) -> tensor<12544x2048xbf16>
    %217 = "mhlo.fusion"(%arg74, %0, %arg374, %216, %207) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg434: tensor<2048xf32>, %arg435: tensor<2048xf32>, %arg436: tensor<2048xf32>, %arg437: tensor<12544x2048xbf16>, %arg438: tensor<256x7x7x2048xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x2048xbf16>
      %223 = stablehlo.maximum %arg438, %222 : tensor<256x7x7x2048xbf16>
      %224 = stablehlo.convert %223 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %225 = mhlo.bitcast %arg437 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x2048xbf16>) -> tensor<256x7x7x2048xbf16>
      %226 = stablehlo.convert %225 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %227 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %228 = stablehlo.subtract %226, %227 : tensor<256x7x7x2048xf32>
      %229 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %230 = stablehlo.multiply %228, %229 : tensor<256x7x7x2048xf32>
      %231 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %232 = stablehlo.add %230, %231 : tensor<256x7x7x2048xf32>
      %233 = stablehlo.add %224, %232 : tensor<256x7x7x2048xf32>
      %234 = stablehlo.convert %233 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
      %235 = stablehlo.maximum %234, %222 : tensor<256x7x7x2048xbf16>
      %236 = stablehlo.convert %235 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %237 = mhlo.bitcast %236 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x2048xf32>) -> tensor<256x49x2048xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %238 = stablehlo.reduce(%237 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x49x2048xf32>, tensor<f32>) -> tensor<256x2048xf32>
      mhlo.return %238 : tensor<256x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<12544x2048xbf16>, tensor<256x7x7x2048xbf16>) -> tensor<256x2048xf32>
    %218 = "mhlo.fusion"(%217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<256x2048xf32>):
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
      %223 = stablehlo.multiply %arg434, %222 : tensor<256x2048xf32>
      %224 = stablehlo.convert %223 : (tensor<256x2048xf32>) -> tensor<256x2048xbf16>
      mhlo.return %224 : tensor<256x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x2048xf32>) -> tensor<256x2048xbf16>
    %219 = "mhlo.fusion"(%218, %arg158) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg434: tensor<256x2048xbf16>, %arg435: tensor<2048x1000xf32>):
      %222 = stablehlo.convert %arg435 : (tensor<2048x1000xf32>) -> tensor<2048x1000xbf16>
      %223 = stablehlo.dot_general %222, %arg434, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[1000,256]{0,1}"} : (tensor<2048x1000xbf16>, tensor<256x2048xbf16>) -> tensor<1000x256xbf16>
      %224 = mhlo.bitcast %223 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x256xbf16>) -> tensor<256x1000xbf16>
      mhlo.return %224 : tensor<256x1000xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x2048xbf16>, tensor<2048x1000xf32>) -> tensor<256x1000xbf16>
    %220 = "mhlo.fusion"(%arg157, %219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg434: tensor<1000xf32>, %arg435: tensor<256x1000xbf16>):
      %222 = stablehlo.convert %arg435 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
      %223 = stablehlo.broadcast_in_dim %arg434, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
      %224 = stablehlo.add %222, %223 : tensor<256x1000xf32>
      %225 = stablehlo.convert %224 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
      mhlo.return %225 : tensor<256x1000xbf16>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<256x1000xbf16>) -> tensor<256x1000xbf16>
    %221 = mhlo.bitcast %220 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x1000xbf16>) -> tensor<1x256x1000xbf16>
    return %221 : tensor<1x256x1000xbf16>
  }
}
