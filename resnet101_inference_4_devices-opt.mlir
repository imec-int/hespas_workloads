module @pmap_inference_step attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {suggested_combiner_threshold = "38136408224"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.num_replicas = 4 : i32, mhlo.use_auto_spmd_partitioning = false} {
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
  func.func private @maximum.201(%arg0: tensor<802816x64xbf16>) -> tensor<802816x64xbf16> {
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
  func.func private @maximum.202(%arg0: tensor<802816x256xbf16>) -> tensor<802816x256xbf16> {
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
  func.func private @maximum.203(%arg0: tensor<802816x64xbf16>) -> tensor<802816x64xbf16> {
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
  func.func private @maximum.204(%arg0: tensor<802816x256xbf16>) -> tensor<802816x256xbf16> {
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
  func.func private @maximum.205(%arg0: tensor<802816x64xbf16>) -> tensor<802816x64xbf16> {
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
  func.func private @maximum.206(%arg0: tensor<802816x256xbf16>) -> tensor<802816x256xbf16> {
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
  func.func private @maximum.207(%arg0: tensor<200704x128xbf16>) -> tensor<200704x128xbf16> {
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
  func.func private @maximum.208(%arg0: tensor<200704x512xbf16>) -> tensor<200704x512xbf16> {
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
  func.func private @maximum.209(%arg0: tensor<200704x128xbf16>) -> tensor<200704x128xbf16> {
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
  func.func private @maximum.210(%arg0: tensor<200704x512xbf16>) -> tensor<200704x512xbf16> {
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
  func.func private @maximum.211(%arg0: tensor<200704x128xbf16>) -> tensor<200704x128xbf16> {
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
  func.func private @maximum.212(%arg0: tensor<200704x512xbf16>) -> tensor<200704x512xbf16> {
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
  func.func private @maximum.213(%arg0: tensor<200704x128xbf16>) -> tensor<200704x128xbf16> {
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
  func.func private @maximum.214(%arg0: tensor<200704x512xbf16>) -> tensor<200704x512xbf16> {
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
  func.func private @maximum.215(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.216(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.217(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.218(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.219(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.220(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.221(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.222(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.223(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.224(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.225(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.226(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.27(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.28_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.227(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.28(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.29_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.228(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.29(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.30_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.229(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.30(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.31_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.230(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.31(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.32_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.231(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.32(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.33_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.232(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.33(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.34_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.233(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.34(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.35_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.234(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.35(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.36_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.235(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.36(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.37_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.236(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.37(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.38_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.237(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.38(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.39_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.238(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.39(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.40_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.239(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.40(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.41_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.240(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.41(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.42_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.241(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.42(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.43_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.242(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.43(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.44_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.243(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.44(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.45_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.244(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.45(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.46_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.245(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.46(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.47_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.246(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.47(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.48_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.247(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.48(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.49_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.248(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.49(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.50_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.249(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.50(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.51_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.250(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.51(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.52_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.251(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.52(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.53_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.252(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.53(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.54_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.253(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.54(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.55_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.254(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.55(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.56_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.255(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.56(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.57_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.256(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.57(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.58_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.257(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.58(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.59_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.258(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.59(%arg0: tensor<1024x256xf32>) -> tensor<1024x256xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
    return %0 : tensor<1024x256xbf16>
  }
  func.func private @gemm_fusion_dot.60_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x256xf32>) -> tensor<50176x256xbf16> {
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
  func.func private @maximum.259(%arg0: tensor<50176x256xbf16>) -> tensor<50176x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x256xbf16>
    return %1 : tensor<50176x256xbf16>
  }
  func.func private @convert.60(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
  }
  func.func private @gemm_fusion_dot.61_computation(%arg0: tensor<50176x256xbf16>, %arg1: tensor<256x1024xf32>) -> tensor<50176x1024xbf16> {
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
  func.func private @maximum.260(%arg0: tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<50176x1024xbf16>
    return %1 : tensor<50176x1024xbf16>
  }
  func.func private @convert.61(%arg0: tensor<1024x512xf32>) -> tensor<1024x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<1024x512xf32>) -> tensor<1024x512xbf16>
    return %0 : tensor<1024x512xbf16>
  }
  func.func private @gemm_fusion_dot.62_computation(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024x512xf32>) -> tensor<50176x512xbf16> {
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
  func.func private @maximum.261(%arg0: tensor<12544x512xbf16>) -> tensor<12544x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x512xbf16>
    return %1 : tensor<12544x512xbf16>
  }
  func.func private @convert.62(%arg0: tensor<512x2048xf32>) -> tensor<512x2048xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
    return %0 : tensor<512x2048xbf16>
  }
  func.func private @gemm_fusion_dot.63_computation(%arg0: tensor<12544x512xbf16>, %arg1: tensor<512x2048xf32>) -> tensor<12544x2048xbf16> {
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
  func.func private @maximum.262(%arg0: tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x2048xbf16>
    return %1 : tensor<12544x2048xbf16>
  }
  func.func private @convert.63(%arg0: tensor<2048x512xf32>) -> tensor<2048x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
    return %0 : tensor<2048x512xbf16>
  }
  func.func private @gemm_fusion_dot.64_computation(%arg0: tensor<12544x2048xbf16>, %arg1: tensor<2048x512xf32>) -> tensor<12544x512xbf16> {
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
  func.func private @maximum.263(%arg0: tensor<12544x512xbf16>) -> tensor<12544x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x512xbf16>
    return %1 : tensor<12544x512xbf16>
  }
  func.func private @convert.64(%arg0: tensor<512x2048xf32>) -> tensor<512x2048xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
    return %0 : tensor<512x2048xbf16>
  }
  func.func private @gemm_fusion_dot.65_computation(%arg0: tensor<12544x512xbf16>, %arg1: tensor<512x2048xf32>) -> tensor<12544x2048xbf16> {
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
  func.func private @maximum.264(%arg0: tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x2048xbf16>
    return %1 : tensor<12544x2048xbf16>
  }
  func.func private @convert.65(%arg0: tensor<2048x512xf32>) -> tensor<2048x512xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
    return %0 : tensor<2048x512xbf16>
  }
  func.func private @gemm_fusion_dot.66_computation(%arg0: tensor<12544x2048xbf16>, %arg1: tensor<2048x512xf32>) -> tensor<12544x512xbf16> {
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
  func.func private @maximum.265(%arg0: tensor<12544x512xbf16>) -> tensor<12544x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<12544x512xbf16>
    return %1 : tensor<12544x512xbf16>
  }
  func.func private @convert.66(%arg0: tensor<512x2048xf32>) -> tensor<512x2048xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
    return %0 : tensor<512x2048xbf16>
  }
  func.func private @gemm_fusion_dot.67_computation(%arg0: tensor<12544x512xbf16>, %arg1: tensor<512x2048xf32>) -> tensor<12544x2048xbf16> {
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
  func.func private @fused_convert.14(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.15(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.16(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.17(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.18(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.19(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.20(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.21(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.22(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.23(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.24(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.25(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.26(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.27(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.28(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.29(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.30(%arg0: tensor<50176x1024xbf16>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
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
  func.func private @fused_convert.31(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    return %6 : tensor<256x56x56x64xbf16>
  }
  func.func private @fused_convert.32(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    return %6 : tensor<256x56x56x64xbf16>
  }
  func.func private @fused_convert.33(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<256x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<256x56x56x64xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x56x56x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x56x56x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x56x56x64xf32>
    %6 = stablehlo.convert %5 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    return %6 : tensor<256x56x56x64xbf16>
  }
  func.func private @fused_convert.34(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<802816x128xbf16>) -> tensor<256x56x56x128xf32> {
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
  func.func private @fused_convert.35(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<802816x64xbf16>) -> tensor<256x56x56x64xf32> {
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
  func.func private @fused_convert.36(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<802816x64xbf16>) -> tensor<256x56x56x64xf32> {
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
  func.func private @fused_convert.37(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32> {
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
  func.func private @fused_convert.38(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<200704x256xbf16>) -> tensor<256x28x28x256xf32> {
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
  func.func private @fused_convert.39(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<50176x1024xbf16>, %arg4: tensor<256x14x14x1024xbf16>) -> (tensor<256x14x14x1024xf32>, tensor<256x14x14x1024xbf16>) {
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
  func.func private @fused_convert.40(%arg0: tensor<12544x2048xbf16>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<2048xf32>, %arg5: tensor<2048xf32>, %arg6: tensor<256x7x7x2048xf32>, %arg7: tensor<2048xf32>) -> tensor<256x7x7x2048xbf16> {
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
  func.func private @fused_convert.41(%arg0: tensor<12544x2048xbf16>, %arg1: tensor<2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048xf32>, %arg4: tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16> {
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
  func.func private @fused_convert.42(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    return %6 : tensor<256x28x28x128xbf16>
  }
  func.func private @fused_convert.43(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    return %6 : tensor<256x28x28x128xbf16>
  }
  func.func private @fused_convert.44(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    return %6 : tensor<256x28x28x128xbf16>
  }
  func.func private @fused_convert.45(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<256x28x28x128xf32>, %arg3: tensor<128xf32>) -> tensor<256x28x28x128xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x28x28x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x28x28x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x28x28x128xf32>
    %6 = stablehlo.convert %5 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    return %6 : tensor<256x28x28x128xbf16>
  }
  func.func private @fused_convert.46(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32> {
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
  func.func private @fused_convert.47(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32> {
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
  func.func private @fused_convert.48(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32> {
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
  func.func private @fused_convert.49(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<50176x512xbf16>) -> tensor<256x14x14x512xf32> {
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
  func.func private @fused_convert.50(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.51(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.52(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.53(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.54(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.55(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.56(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.57(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.58(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.59(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.60(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.61(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.62(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.63(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.64(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.65(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.66(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.67(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.68(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.69(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.70(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.71(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.72(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256x14x14x256xf32>, %arg3: tensor<256xf32>) -> tensor<256x14x14x256xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x14x14x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x14x14x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x14x14x256xf32>
    %6 = stablehlo.convert %5 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    return %6 : tensor<256x14x14x256xbf16>
  }
  func.func private @fused_convert.73(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.convert %5 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    return %6 : tensor<256x7x7x512xbf16>
  }
  func.func private @fused_convert.74(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.convert %5 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    return %6 : tensor<256x7x7x512xbf16>
  }
  func.func private @fused_convert.75(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<256x7x7x512xf32>, %arg3: tensor<512xf32>) -> tensor<256x7x7x512xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<256x7x7x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<256x7x7x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %5 = stablehlo.add %3, %4 : tensor<256x7x7x512xf32>
    %6 = stablehlo.convert %5 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    return %6 : tensor<256x7x7x512xbf16>
  }
  func.func private @fused_convert.76(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.77(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.78(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.79(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.80(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.81(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.82(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.83(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.84(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.85(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.86(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.87(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.88(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.89(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.90(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.91(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.92(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.93(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.94(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.95(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.96(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.97(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32> {
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
  func.func private @fused_convert.98(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<12544x512xbf16>) -> tensor<256x7x7x512xf32> {
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
  func.func private @fused_convert.99(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<12544x512xbf16>) -> tensor<256x7x7x512xf32> {
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
  func.func private @fused_convert.100(%arg0: tensor<256x2048xf32>) -> tensor<256x2048xbf16> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<256x2048xf32>
    %2 = stablehlo.convert %1 : (tensor<256x2048xf32>) -> tensor<256x2048xbf16>
    return %2 : tensor<256x2048xbf16>
  }
  func.func private @fused_convert.101(%arg0: tensor<1000xf32>, %arg1: tensor<256x1000xbf16>) -> tensor<256x1000xbf16> {
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
  func.func private @fused_multiply.11(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.25(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.26(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.27(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1024xf32>
    %2 = stablehlo.rsqrt %1 : tensor<1024xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
  func.func private @fused_multiply.28(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.29(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.30(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.31(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.32(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.33(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.34(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.35(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.36(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.37(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.38(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<512xf32>
    return %3 : tensor<512xf32>
  }
  func.func private @fused_multiply.39(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.40(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.41(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.42(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.43(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.44(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.45(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.46(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.47(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.48(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.49(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.50(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.51(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.52(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.53(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.54(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.55(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.56(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.57(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.58(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.59(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.60(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.61(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.62(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.63(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.64(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.65(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.66(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.67(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.68(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.69(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.70(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.71(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.72(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.73(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.74(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.75(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.76(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.77(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.78(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.79(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.80(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.81(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.82(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.83(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.84(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.85(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.86(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.87(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.88(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<256xf32>
    return %3 : tensor<256xf32>
  }
  func.func private @fused_multiply.89(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.90(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.91(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.92(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.93(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.94(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.95(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.96(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_multiply.97(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.98(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.99(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.100(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.101(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.102(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func private @fused_multiply.103(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    %3 = stablehlo.multiply %2, %arg0 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func @main(%arg0: tensor<i32> {mhlo.parameter_replication = [true]}, %arg1: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg2: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg3: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg4: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg5: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg6: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg7: tensor<1x1x64x64xf32> {mhlo.parameter_replication = [true]}, %arg8: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg9: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg10: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg11: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg12: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg13: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg14: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg15: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg16: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg17: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg18: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg19: tensor<1x1x256x64xf32> {mhlo.parameter_replication = [true]}, %arg20: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg21: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg22: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg23: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg24: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg25: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg26: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg27: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg28: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg29: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg30: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg31: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg32: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg33: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg34: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg35: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg36: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg37: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg38: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg39: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg40: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg41: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg42: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg43: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg44: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg45: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg46: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg47: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg48: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg49: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg50: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg51: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg52: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg53: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg54: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg55: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg56: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg57: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg58: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg59: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg60: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg61: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg62: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg63: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg64: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg65: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg66: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg67: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg68: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg69: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg70: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg71: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg72: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg73: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg74: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg75: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg76: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg77: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg78: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg79: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg80: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg81: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg82: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg83: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg84: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg85: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg86: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg87: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg88: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg89: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg90: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg91: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg92: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg93: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg94: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg95: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg96: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg97: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg98: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg99: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg100: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg101: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg102: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg103: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg104: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg105: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg106: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg107: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg108: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg109: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg110: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg111: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg112: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg113: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg114: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg115: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg116: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg117: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg118: tensor<1x1x256x64xf32> {mhlo.parameter_replication = [true]}, %arg119: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg120: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg121: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg122: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg123: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg124: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg125: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg126: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg127: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg128: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg129: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg130: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg131: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg132: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg133: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg134: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg135: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg136: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg137: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg138: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg139: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg140: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg141: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg142: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg143: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg144: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg145: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg146: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg147: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg148: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg149: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg150: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg151: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg152: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg153: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg154: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg155: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg156: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg157: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg158: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg159: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg160: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg161: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg162: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg163: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg164: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg165: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg166: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg167: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg168: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg169: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg170: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg171: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg172: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg173: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg174: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg175: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg176: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg177: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg178: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg179: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg180: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg181: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg182: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg183: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg184: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg185: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg186: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg187: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg188: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg189: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg190: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg191: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg192: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg193: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg194: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg195: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg196: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg197: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg198: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg199: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg200: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg201: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg202: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg203: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg204: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg205: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg206: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg207: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg208: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg209: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg210: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg211: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg212: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg213: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg214: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg215: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg216: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg217: tensor<1x1x256x128xf32> {mhlo.parameter_replication = [true]}, %arg218: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg219: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg220: tensor<1x1x256x512xf32> {mhlo.parameter_replication = [true]}, %arg221: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg222: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg223: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg224: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg225: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg226: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg227: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg228: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg229: tensor<1x1x1024x512xf32> {mhlo.parameter_replication = [true]}, %arg230: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg231: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg232: tensor<1x1x1024x2048xf32> {mhlo.parameter_replication = [true]}, %arg233: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg234: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg235: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg236: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg237: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg238: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg239: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg240: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg241: tensor<1x1x2048x512xf32> {mhlo.parameter_replication = [true]}, %arg242: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg243: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg244: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg245: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg246: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg247: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg248: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg249: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg250: tensor<1x1x2048x512xf32> {mhlo.parameter_replication = [true]}, %arg251: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg252: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg253: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg254: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg255: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg256: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg257: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg258: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg259: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg260: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg261: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg262: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg263: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg264: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg265: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg266: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg267: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg268: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg269: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg270: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg271: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg272: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg273: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg274: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg275: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg276: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg277: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg278: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg279: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg280: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg281: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg282: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg283: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg284: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg285: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg286: tensor<1x1x512x256xf32> {mhlo.parameter_replication = [true]}, %arg287: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg288: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg289: tensor<1x1x512x1024xf32> {mhlo.parameter_replication = [true]}, %arg290: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg291: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg292: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg293: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg294: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg295: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg296: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg297: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg298: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg299: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg300: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg301: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg302: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg303: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg304: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg305: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg306: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg307: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg308: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg309: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg310: tensor<1000xf32> {mhlo.parameter_replication = [true]}, %arg311: tensor<2048x1000xf32> {mhlo.parameter_replication = [true]}, %arg312: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg313: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg314: tensor<7x7x3x64xf32> {mhlo.parameter_replication = [true]}, %arg315: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg316: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg317: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg318: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg319: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg320: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg321: tensor<1x1x64x64xf32> {mhlo.parameter_replication = [true]}, %arg322: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg323: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg324: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg325: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg326: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg327: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg328: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg329: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg330: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg331: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg332: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg333: tensor<1x1x256x64xf32> {mhlo.parameter_replication = [true]}, %arg334: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg335: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg336: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg337: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg338: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg339: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg340: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg341: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg342: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg343: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg344: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg345: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg346: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg347: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg348: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg349: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg350: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg351: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg352: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg353: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg354: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg355: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg356: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg357: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg358: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg359: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg360: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg361: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg362: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg363: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg364: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg365: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg366: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg367: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg368: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg369: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg370: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg371: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg372: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg373: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg374: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg375: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg376: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg377: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg378: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg379: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg380: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg381: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg382: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg383: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg384: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg385: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg386: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg387: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg388: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg389: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg390: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg391: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg392: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg393: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg394: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg395: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg396: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg397: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg398: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg399: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg400: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg401: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg402: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg403: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg404: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg405: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg406: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg407: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg408: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg409: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg410: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg411: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg412: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg413: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg414: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg415: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg416: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg417: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg418: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg419: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg420: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg421: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg422: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg423: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg424: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg425: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg426: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg427: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg428: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg429: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg430: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg431: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg432: tensor<1x1x256x64xf32> {mhlo.parameter_replication = [true]}, %arg433: tensor<3x3x64x64xf32> {mhlo.parameter_replication = [true]}, %arg434: tensor<1x1x64x256xf32> {mhlo.parameter_replication = [true]}, %arg435: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg436: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg437: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg438: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg439: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg440: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg441: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg442: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg443: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg444: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg445: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg446: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg447: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg448: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg449: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg450: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg451: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg452: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg453: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg454: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg455: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg456: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg457: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg458: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg459: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg460: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg461: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg462: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg463: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg464: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg465: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg466: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg467: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg468: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg469: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg470: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg471: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg472: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg473: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg474: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg475: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg476: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg477: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg478: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg479: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg480: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg481: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg482: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg483: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg484: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg485: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg486: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg487: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg488: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg489: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg490: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg491: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg492: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg493: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg494: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg495: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg496: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg497: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg498: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg499: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg500: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg501: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg502: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg503: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg504: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg505: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg506: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg507: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg508: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg509: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg510: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg511: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg512: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg513: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg514: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg515: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg516: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg517: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg518: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg519: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg520: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg521: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg522: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg523: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg524: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg525: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg526: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg527: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg528: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg529: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg530: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg531: tensor<1x1x256x128xf32> {mhlo.parameter_replication = [true]}, %arg532: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg533: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg534: tensor<1x1x256x512xf32> {mhlo.parameter_replication = [true]}, %arg535: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg536: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg537: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg538: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg539: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg540: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg541: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg542: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg543: tensor<1x1x1024x512xf32> {mhlo.parameter_replication = [true]}, %arg544: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg545: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg546: tensor<1x1x1024x2048xf32> {mhlo.parameter_replication = [true]}, %arg547: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg548: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg549: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg550: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg551: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg552: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg553: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg554: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg555: tensor<1x1x2048x512xf32> {mhlo.parameter_replication = [true]}, %arg556: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg557: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg558: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg559: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg560: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg561: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg562: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg563: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg564: tensor<1x1x2048x512xf32> {mhlo.parameter_replication = [true]}, %arg565: tensor<3x3x512x512xf32> {mhlo.parameter_replication = [true]}, %arg566: tensor<1x1x512x2048xf32> {mhlo.parameter_replication = [true]}, %arg567: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg568: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg569: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg570: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg571: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg572: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg573: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg574: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg575: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg576: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg577: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg578: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg579: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg580: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg581: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg582: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg583: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg584: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg585: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg586: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg587: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg588: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg589: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg590: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg591: tensor<1x1x512x128xf32> {mhlo.parameter_replication = [true]}, %arg592: tensor<3x3x128x128xf32> {mhlo.parameter_replication = [true]}, %arg593: tensor<1x1x128x512xf32> {mhlo.parameter_replication = [true]}, %arg594: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg595: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg596: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg597: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg598: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg599: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg600: tensor<1x1x512x256xf32> {mhlo.parameter_replication = [true]}, %arg601: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg602: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg603: tensor<1x1x512x1024xf32> {mhlo.parameter_replication = [true]}, %arg604: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg605: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg606: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg607: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg608: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg609: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg610: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg611: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg612: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg613: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg614: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg615: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg616: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg617: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg618: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg619: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg620: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg621: tensor<1x1x1024x256xf32> {mhlo.parameter_replication = [true]}, %arg622: tensor<3x3x256x256xf32> {mhlo.parameter_replication = [true]}, %arg623: tensor<1x1x256x1024xf32> {mhlo.parameter_replication = [true]}, %arg624: tensor<1000xf32> {mhlo.parameter_replication = [true]}, %arg625: tensor<2048x1000xf32> {mhlo.parameter_replication = [true]}, %arg626: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg627: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg628: tensor<7x7x3x64xf32> {mhlo.parameter_replication = [true]}, %arg629: tensor<i32> {mhlo.parameter_replication = [true]}, %arg630: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg631: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg632: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg633: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg634: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg635: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg636: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg637: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg638: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg639: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg640: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg641: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg642: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg643: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg644: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg645: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg646: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg647: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg648: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg649: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg650: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg651: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg652: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg653: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg654: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg655: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg656: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg657: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg658: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg659: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg660: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg661: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg662: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg663: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg664: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg665: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg666: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg667: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg668: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg669: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg670: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg671: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg672: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg673: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg674: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg675: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg676: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg677: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg678: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg679: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg680: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg681: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg682: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg683: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg684: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg685: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg686: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg687: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg688: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg689: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg690: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg691: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg692: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg693: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg694: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg695: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg696: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg697: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg698: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg699: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg700: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg701: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg702: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg703: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg704: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg705: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg706: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg707: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg708: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg709: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg710: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg711: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg712: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg713: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg714: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg715: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg716: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg717: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg718: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg719: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg720: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg721: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg722: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg723: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg724: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg725: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg726: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg727: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg728: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg729: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg730: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg731: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg732: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg733: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg734: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg735: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg736: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg737: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg738: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg739: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg740: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg741: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg742: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg743: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg744: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg745: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg746: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg747: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg748: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg749: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg750: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg751: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg752: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg753: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg754: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg755: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg756: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg757: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg758: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg759: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg760: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg761: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg762: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg763: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg764: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg765: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg766: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg767: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg768: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg769: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg770: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg771: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg772: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg773: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg774: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg775: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg776: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg777: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg778: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg779: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg780: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg781: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg782: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg783: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg784: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg785: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg786: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg787: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg788: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg789: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg790: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg791: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg792: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg793: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg794: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg795: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg796: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg797: tensor<2048xf32> {mhlo.parameter_replication = [true]}, %arg798: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg799: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg800: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg801: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg802: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg803: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg804: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg805: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg806: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg807: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg808: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg809: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg810: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg811: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg812: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg813: tensor<128xf32> {mhlo.parameter_replication = [true]}, %arg814: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg815: tensor<512xf32> {mhlo.parameter_replication = [true]}, %arg816: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg817: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg818: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg819: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg820: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg821: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg822: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg823: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg824: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg825: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg826: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg827: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg828: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg829: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg830: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg831: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg832: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg833: tensor<256xf32> {mhlo.parameter_replication = [true]}, %arg834: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg835: tensor<1024xf32> {mhlo.parameter_replication = [true]}, %arg836: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg837: tensor<64xf32> {mhlo.parameter_replication = [true]}, %arg838: tensor<i32> {mhlo.parameter_replication = [true]}, %arg839: tensor<f32> {mhlo.parameter_replication = [true]}, %arg840: tensor<1x256x224x224x3xbf16>, %arg841: tensor<1x256xi32>) -> tensor<1x256x1000xbf16> {
    %0 = "mhlo.fusion"(%arg249, %arg797) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<2048xf32>, %arg843: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<2048xf32>
      %445 = stablehlo.rsqrt %444 : tensor<2048xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<2048xf32>
      mhlo.return %446 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %1 = "mhlo.fusion"(%arg247, %arg795) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %2 = "mhlo.fusion"(%arg245, %arg793) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %3 = "mhlo.fusion"(%arg238, %arg789) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4 = "mhlo.fusion"(%arg236, %arg787) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %5 = "mhlo.fusion"(%arg226, %arg781) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %6 = "mhlo.fusion"(%arg224, %arg779) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7 = "mhlo.fusion"(%arg207, %arg769) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %8 = "mhlo.fusion"(%arg205, %arg767) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %9 = "mhlo.fusion"(%arg203, %arg765) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %10 = "mhlo.fusion"(%arg196, %arg761) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %11 = "mhlo.fusion"(%arg194, %arg759) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %12 = "mhlo.fusion"(%arg187, %arg755) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %13 = "mhlo.fusion"(%arg185, %arg753) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %14 = "mhlo.fusion"(%arg178, %arg749) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %15 = "mhlo.fusion"(%arg176, %arg747) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %16 = "mhlo.fusion"(%arg169, %arg743) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %17 = "mhlo.fusion"(%arg167, %arg741) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %18 = "mhlo.fusion"(%arg160, %arg737) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %19 = "mhlo.fusion"(%arg158, %arg735) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %20 = "mhlo.fusion"(%arg151, %arg731) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %21 = "mhlo.fusion"(%arg149, %arg729) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %22 = "mhlo.fusion"(%arg142, %arg725) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %23 = "mhlo.fusion"(%arg140, %arg723) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %24 = "mhlo.fusion"(%arg133, %arg719) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %25 = "mhlo.fusion"(%arg131, %arg717) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %26 = "mhlo.fusion"(%arg124, %arg713) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %27 = "mhlo.fusion"(%arg122, %arg711) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %28 = "mhlo.fusion"(%arg106, %arg701) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %29 = "mhlo.fusion"(%arg104, %arg699) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %30 = "mhlo.fusion"(%arg97, %arg695) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %31 = "mhlo.fusion"(%arg95, %arg693) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %32 = "mhlo.fusion"(%arg88, %arg689) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %33 = "mhlo.fusion"(%arg86, %arg687) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %34 = "mhlo.fusion"(%arg79, %arg683) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %35 = "mhlo.fusion"(%arg77, %arg681) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %36 = "mhlo.fusion"(%arg70, %arg677) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %37 = "mhlo.fusion"(%arg68, %arg675) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %38 = "mhlo.fusion"(%arg61, %arg671) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %39 = "mhlo.fusion"(%arg59, %arg669) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %40 = "mhlo.fusion"(%arg52, %arg665) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %41 = "mhlo.fusion"(%arg50, %arg663) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %42 = "mhlo.fusion"(%arg43, %arg659) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %43 = "mhlo.fusion"(%arg41, %arg657) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %44 = "mhlo.fusion"(%arg34, %arg653) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %45 = "mhlo.fusion"(%arg32, %arg651) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %46 = "mhlo.fusion"(%arg25, %arg647) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %47 = "mhlo.fusion"(%arg23, %arg645) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %48 = "mhlo.fusion"(%arg304, %arg833) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %49 = "mhlo.fusion"(%arg302, %arg831) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %50 = "mhlo.fusion"(%arg295, %arg827) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %51 = "mhlo.fusion"(%arg293, %arg825) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %52 = "mhlo.fusion"(%arg283, %arg819) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %53 = "mhlo.fusion"(%arg281, %arg817) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %54 = "mhlo.fusion"(%arg276, %arg815) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %55 = "mhlo.fusion"(%arg274, %arg813) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<128xf32>
      %445 = stablehlo.rsqrt %444 : tensor<128xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<128xf32>
      mhlo.return %446 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %56 = "mhlo.fusion"(%arg272, %arg811) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<128xf32>
      %445 = stablehlo.rsqrt %444 : tensor<128xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<128xf32>
      mhlo.return %446 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %57 = "mhlo.fusion"(%arg267, %arg809) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %58 = "mhlo.fusion"(%arg265, %arg807) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<128xf32>
      %445 = stablehlo.rsqrt %444 : tensor<128xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<128xf32>
      mhlo.return %446 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %59 = "mhlo.fusion"(%arg263, %arg805) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<128xf32>
      %445 = stablehlo.rsqrt %444 : tensor<128xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<128xf32>
      mhlo.return %446 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %60 = "mhlo.fusion"(%arg258, %arg803) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %61 = "mhlo.fusion"(%arg256, %arg801) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<128xf32>
      %445 = stablehlo.rsqrt %444 : tensor<128xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<128xf32>
      mhlo.return %446 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %62 = "mhlo.fusion"(%arg254, %arg799) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<128xf32>
      %445 = stablehlo.rsqrt %444 : tensor<128xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<128xf32>
      mhlo.return %446 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %63 = "mhlo.fusion"(%arg216, %arg775) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %64 = "mhlo.fusion"(%arg214, %arg773) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<128xf32>
      %445 = stablehlo.rsqrt %444 : tensor<128xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<128xf32>
      mhlo.return %446 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %65 = "mhlo.fusion"(%arg212, %arg771) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<128xf32>
      %445 = stablehlo.rsqrt %444 : tensor<128xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<128xf32>
      mhlo.return %446 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %66 = "mhlo.fusion"(%arg117, %arg709) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %67 = "mhlo.fusion"(%arg115, %arg707) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<64xf32>
      %445 = stablehlo.rsqrt %444 : tensor<64xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<64xf32>
      mhlo.return %446 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %68 = "mhlo.fusion"(%arg113, %arg705) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<64xf32>
      %445 = stablehlo.rsqrt %444 : tensor<64xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<64xf32>
      mhlo.return %446 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %69 = "mhlo.fusion"(%arg18, %arg643) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %70 = "mhlo.fusion"(%arg16, %arg641) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<64xf32>
      %445 = stablehlo.rsqrt %444 : tensor<64xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<64xf32>
      mhlo.return %446 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %71 = "mhlo.fusion"(%arg14, %arg639) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<64xf32>
      %445 = stablehlo.rsqrt %444 : tensor<64xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<64xf32>
      mhlo.return %446 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %72 = "mhlo.fusion"(%arg12, %arg637) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %73 = "mhlo.fusion"(%arg313, %arg837) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<64xf32>
      %445 = stablehlo.rsqrt %444 : tensor<64xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<64xf32>
      mhlo.return %446 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %74 = "mhlo.fusion"(%arg840) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1x256x224x224x3xbf16>):
      %443 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
      mhlo.return %444 : tensor<256x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xf32>
    %75 = stablehlo.convolution(%74, %arg314) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<256x112x112x64xf32>
    %76 = "mhlo.fusion"(%arg312, %73, %75, %arg836) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>, %arg844: tensor<256x112x112x64xf32>, %arg845: tensor<64xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x112x112x64xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x112x112x64xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x112x112x64xf32>
      %449 = stablehlo.convert %448 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %450 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
      %451 = stablehlo.maximum %449, %450 : tensor<256x112x112x64xbf16>
      %452 = stablehlo.convert %451 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
      %453 = stablehlo.convert %cst_0 : (tensor<bf16>) -> tensor<f32>
      %454 = "stablehlo.reduce_window"(%452, %453) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg846: tensor<f32>, %arg847: tensor<f32>):
        %456 = stablehlo.convert %arg846 : (tensor<f32>) -> tensor<bf16>
        %457 = stablehlo.convert %arg847 : (tensor<f32>) -> tensor<bf16>
        %458 = stablehlo.maximum %456, %457 : tensor<bf16>
        %459 = stablehlo.convert %458 : (tensor<bf16>) -> tensor<f32>
        stablehlo.return %459 : tensor<f32>
      }) : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<256x56x56x64xf32>
      %455 = stablehlo.convert %454 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      mhlo.return %455 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x112x112x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xbf16>
    %77 = "mhlo.fusion"(%76, %arg10) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<256x56x56x64xbf16>, %arg843: tensor<1x1x64x256xf32>):
      %443 = stablehlo.convert %arg843 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xbf16>
      %444 = mhlo.bitcast %443 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xbf16>) -> tensor<64x256xbf16>
      %445 = mhlo.bitcast %arg842 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
      %446 = stablehlo.dot_general %444, %445, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[256,802816]{0,1}"} : (tensor<64x256xbf16>, tensor<802816x64xbf16>) -> tensor<256x802816xbf16>
      %447 = mhlo.bitcast %446 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x802816xbf16>) -> tensor<802816x256xbf16>
      %448 = mhlo.bitcast %447 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
      mhlo.return %448 : tensor<256x56x56x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x256xf32>) -> tensor<256x56x56x256xbf16>
    %78 = "mhlo.fusion"(%arg6, %arg635) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<256xf32>
      %445 = stablehlo.rsqrt %444 : tensor<256xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<256xf32>
      mhlo.return %446 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %79 = "mhlo.fusion"(%arg4, %arg633) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<64xf32>
      %445 = stablehlo.rsqrt %444 : tensor<64xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<64xf32>
      mhlo.return %446 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %80 = "mhlo.fusion"(%arg2, %arg631) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<64xf32>
      %445 = stablehlo.rsqrt %444 : tensor<64xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<64xf32>
      mhlo.return %446 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %81 = "mhlo.fusion"(%76, %arg7) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<256x56x56x64xbf16>, %arg843: tensor<1x1x64x64xf32>):
      %443 = stablehlo.convert %arg843 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xbf16>
      %444 = mhlo.bitcast %443 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x64xbf16>) -> tensor<64x64xbf16>
      %445 = mhlo.bitcast %arg842 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
      %446 = stablehlo.dot_general %444, %445, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[64,802816]{0,1}"} : (tensor<64x64xbf16>, tensor<802816x64xbf16>) -> tensor<64x802816xbf16>
      %447 = mhlo.bitcast %446 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<64x802816xbf16>) -> tensor<802816x64xbf16>
      %448 = mhlo.bitcast %447 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x64xbf16>) -> tensor<256x56x56x64xbf16>
      mhlo.return %448 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x64xf32>) -> tensor<256x56x56x64xbf16>
    %82 = "mhlo.fusion"(%arg1, %80, %arg630, %81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>, %arg844: tensor<64xf32>, %arg845: tensor<256x56x56x64xbf16>):
      %443 = stablehlo.convert %arg845 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      %444 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %445 = stablehlo.subtract %443, %444 : tensor<256x56x56x64xf32>
      %446 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %447 = stablehlo.multiply %445, %446 : tensor<256x56x56x64xf32>
      %448 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %449 = stablehlo.add %447, %448 : tensor<256x56x56x64xf32>
      %450 = stablehlo.convert %449 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %451 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %452 = stablehlo.maximum %450, %451 : tensor<256x56x56x64xbf16>
      %453 = stablehlo.convert %452 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %453 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %83 = stablehlo.convolution(%82, %arg8) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %84 = "mhlo.fusion"(%79, %arg3, %83, %arg632) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>, %arg844: tensor<256x56x56x64xf32>, %arg845: tensor<64xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x56x56x64xf32>
      %445 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x56x56x64xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x56x56x64xf32>
      %449 = stablehlo.convert %448 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      mhlo.return %449 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xbf16>
    %85 = mhlo.bitcast %84 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
    %86 = mhlo.bitcast %arg9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %87 = "mhlo.fusion"(%85, %86) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<802816x64xbf16>, %arg843: tensor<64x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<802816x64xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<802816x64xbf16>
        mhlo.return %447 : tensor<802816x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>) -> tensor<802816x64xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<64x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
        mhlo.return %446 : tensor<64x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<802816x64xbf16>, tensor<64x256xbf16>) -> tensor<802816x256xbf16>
      mhlo.return %445 : tensor<802816x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>, tensor<64x256xf32>) -> tensor<802816x256xbf16>
    %88 = "mhlo.fusion"(%arg11, %72, %arg636, %77, %arg5, %78, %arg634, %87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<256x56x56x256xbf16>, %arg846: tensor<256xf32>, %arg847: tensor<256xf32>, %arg848: tensor<256xf32>, %arg849: tensor<802816x256xbf16>):
      %443 = stablehlo.convert %arg845 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %444 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %445 = stablehlo.subtract %443, %444 : tensor<256x56x56x256xf32>
      %446 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %447 = stablehlo.multiply %445, %446 : tensor<256x56x56x256xf32>
      %448 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %449 = stablehlo.add %447, %448 : tensor<256x56x56x256xf32>
      %450 = mhlo.bitcast %arg849 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
      %451 = stablehlo.convert %450 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %452 = stablehlo.broadcast_in_dim %arg848, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %453 = stablehlo.subtract %451, %452 : tensor<256x56x56x256xf32>
      %454 = stablehlo.broadcast_in_dim %arg847, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %455 = stablehlo.multiply %453, %454 : tensor<256x56x56x256xf32>
      %456 = stablehlo.broadcast_in_dim %arg846, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %457 = stablehlo.add %455, %456 : tensor<256x56x56x256xf32>
      %458 = stablehlo.add %449, %457 : tensor<256x56x56x256xf32>
      %459 = stablehlo.convert %458 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
      mhlo.return %459 : tensor<256x56x56x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x56x56x256xbf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
    %89 = mhlo.bitcast %88 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x256xbf16>) -> tensor<802816x256xbf16>
    %90 = mhlo.bitcast %arg19 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x64xf32>) -> tensor<256x64xf32>
    %91 = "mhlo.fusion"(%89, %90) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<802816x256xbf16>, %arg843: tensor<256x64xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<802816x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<802816x256xbf16>
        mhlo.return %447 : tensor<802816x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>) -> tensor<802816x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x64xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
        mhlo.return %446 : tensor<256x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x64xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<802816x256xbf16>, tensor<256x64xbf16>) -> tensor<802816x64xbf16>
      mhlo.return %445 : tensor<802816x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>, tensor<256x64xf32>) -> tensor<802816x64xbf16>
    %92 = "mhlo.fusion"(%arg13, %71, %arg638, %91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>, %arg844: tensor<64xf32>, %arg845: tensor<802816x64xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x64xbf16>) -> tensor<256x56x56x64xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x56x56x64xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x56x56x64xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x56x56x64xf32>
      %451 = stablehlo.convert %450 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x56x56x64xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %454 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<802816x64xbf16>) -> tensor<256x56x56x64xf32>
    %93 = stablehlo.convolution(%92, %arg20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %94 = "mhlo.fusion"(%70, %arg15, %93, %arg640) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>, %arg844: tensor<256x56x56x64xf32>, %arg845: tensor<64xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x56x56x64xf32>
      %445 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x56x56x64xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x56x56x64xf32>
      %449 = stablehlo.convert %448 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      mhlo.return %449 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xbf16>
    %95 = mhlo.bitcast %94 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
    %96 = mhlo.bitcast %arg21 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %97 = "mhlo.fusion"(%95, %96) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<802816x64xbf16>, %arg843: tensor<64x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<802816x64xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<802816x64xbf16>
        mhlo.return %447 : tensor<802816x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>) -> tensor<802816x64xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<64x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
        mhlo.return %446 : tensor<64x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<802816x64xbf16>, tensor<64x256xbf16>) -> tensor<802816x256xbf16>
      mhlo.return %445 : tensor<802816x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>, tensor<64x256xf32>) -> tensor<802816x256xbf16>
    %98 = "mhlo.fusion"(%arg17, %69, %arg642, %97, %88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<802816x256xbf16>, %arg846: tensor<256x56x56x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x256xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x56x56x256xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %446 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %448 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x56x56x256xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x56x56x256xf32>
      %452 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x56x56x256xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x56x56x256xf32>
      %455 = stablehlo.convert %454 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
      mhlo.return %455 : tensor<256x56x56x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<802816x256xbf16>, tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xbf16>
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x256xbf16>) -> tensor<802816x256xbf16>
    %100 = mhlo.bitcast %arg118 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x64xf32>) -> tensor<256x64xf32>
    %101 = "mhlo.fusion"(%99, %100) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<802816x256xbf16>, %arg843: tensor<256x64xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<802816x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<802816x256xbf16>
        mhlo.return %447 : tensor<802816x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>) -> tensor<802816x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x64xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x64xf32>) -> tensor<256x64xbf16>
        mhlo.return %446 : tensor<256x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<256x64xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<802816x256xbf16>, tensor<256x64xbf16>) -> tensor<802816x64xbf16>
      mhlo.return %445 : tensor<802816x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>, tensor<256x64xf32>) -> tensor<802816x64xbf16>
    %102 = "mhlo.fusion"(%arg112, %68, %arg704, %101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>, %arg844: tensor<64xf32>, %arg845: tensor<802816x64xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x64xbf16>) -> tensor<256x56x56x64xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x56x56x64xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x56x56x64xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x56x56x64xf32>
      %451 = stablehlo.convert %450 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x56x56x64xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
      mhlo.return %454 : tensor<256x56x56x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<802816x64xbf16>) -> tensor<256x56x56x64xf32>
    %103 = stablehlo.convolution(%102, %arg119) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<256x56x56x64xf32>
    %104 = "mhlo.fusion"(%67, %arg114, %103, %arg706) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<64xf32>, %arg843: tensor<64xf32>, %arg844: tensor<256x56x56x64xf32>, %arg845: tensor<64xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x56x56x64xf32>
      %445 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x56x56x64xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x56x56x64xf32>
      %449 = stablehlo.convert %448 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
      mhlo.return %449 : tensor<256x56x56x64xbf16>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<256x56x56x64xf32>, tensor<64xf32>) -> tensor<256x56x56x64xbf16>
    %105 = mhlo.bitcast %104 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x64xbf16>) -> tensor<802816x64xbf16>
    %106 = mhlo.bitcast %arg120 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x256xf32>) -> tensor<64x256xf32>
    %107 = "mhlo.fusion"(%105, %106) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<802816x64xbf16>, %arg843: tensor<64x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<802816x64xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x64xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<802816x64xbf16>
        mhlo.return %447 : tensor<802816x64xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>) -> tensor<802816x64xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<64x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<64x256xf32>) -> tensor<64x256xbf16>
        mhlo.return %446 : tensor<64x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<802816x64xbf16>, tensor<64x256xbf16>) -> tensor<802816x256xbf16>
      mhlo.return %445 : tensor<802816x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x64xbf16>, tensor<64x256xf32>) -> tensor<802816x256xbf16>
    %108:2 = "mhlo.fusion"(%arg116, %66, %arg708, %107, %98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<802816x256xbf16>, %arg846: tensor<256x56x56x256xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x256xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x56x56x256xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %446 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x256xbf16>) -> tensor<256x56x56x256xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      %448 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x56x56x256xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x56x56x256xf32>
      %452 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x56x56x256xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x56x56x256xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x56x56x256xf32>
      %455 = stablehlo.convert %454 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
      %456 = stablehlo.maximum %455, %443 : tensor<256x56x56x256xbf16>
      %457 = stablehlo.convert %456 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
      mhlo.return %457, %455 : tensor<256x56x56x256xf32>, tensor<256x56x56x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<802816x256xbf16>, tensor<256x56x56x256xbf16>) -> (tensor<256x56x56x256xf32>, tensor<256x56x56x256xbf16>)
    %109 = mhlo.bitcast %108#1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x56x56x256xbf16>) -> tensor<802816x256xbf16>
    %110 = mhlo.bitcast %arg217 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %111 = "mhlo.fusion"(%109, %110) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<802816x256xbf16>, %arg843: tensor<256x128xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<802816x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<802816x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<802816x256xbf16>
        mhlo.return %447 : tensor<802816x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>) -> tensor<802816x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x128xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x128xf32>) -> tensor<256x128xbf16>
        mhlo.return %446 : tensor<256x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x128xf32>) -> tensor<256x128xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<802816x256xbf16>, tensor<256x128xbf16>) -> tensor<802816x128xbf16>
      mhlo.return %445 : tensor<802816x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<802816x256xbf16>, tensor<256x128xf32>) -> tensor<802816x128xbf16>
    %112 = "mhlo.fusion"(%arg211, %65, %arg770, %111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>, %arg844: tensor<128xf32>, %arg845: tensor<802816x128xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<802816x128xbf16>) -> tensor<256x56x56x128xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<128xf32>) -> tensor<256x56x56x128xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x56x56x128xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<128xf32>) -> tensor<256x56x56x128xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x56x56x128xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<128xf32>) -> tensor<256x56x56x128xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x56x56x128xf32>
      %451 = stablehlo.convert %450 : (tensor<256x56x56x128xf32>) -> tensor<256x56x56x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x128xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x56x56x128xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xf32>
      mhlo.return %454 : tensor<256x56x56x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<802816x128xbf16>) -> tensor<256x56x56x128xf32>
    %113 = stablehlo.convolution(%112, %arg218) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %114 = "mhlo.fusion"(%64, %arg213, %113, %arg772) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>, %arg844: tensor<256x28x28x128xf32>, %arg845: tensor<128xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x28x28x128xf32>
      %445 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x28x28x128xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x28x28x128xf32>
      %449 = stablehlo.convert %448 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      mhlo.return %449 : tensor<256x28x28x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xbf16>
    %115 = mhlo.bitcast %114 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x128xbf16>) -> tensor<200704x128xbf16>
    %116 = mhlo.bitcast %arg219 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %117 = "mhlo.fusion"(%115, %116) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<200704x128xbf16>, %arg843: tensor<128x512xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<200704x128xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<200704x128xbf16>
        mhlo.return %447 : tensor<200704x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<128x512xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
        mhlo.return %446 : tensor<128x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
      mhlo.return %445 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>, tensor<128x512xf32>) -> tensor<200704x512xbf16>
    %118 = "mhlo.fusion"(%arg222, %arg777) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<512xf32>
      %445 = stablehlo.rsqrt %444 : tensor<512xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<512xf32>
      mhlo.return %446 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %119 = stablehlo.convolution(%108#0, %arg220) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf32>, tensor<1x1x256x512xf32>) -> tensor<256x28x28x512xf32>
    %120 = "mhlo.fusion"(%arg215, %63, %arg774, %117, %arg221, %118, %119, %arg776) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<512xf32>, %arg845: tensor<200704x512xbf16>, %arg846: tensor<512xf32>, %arg847: tensor<512xf32>, %arg848: tensor<256x28x28x512xf32>, %arg849: tensor<512xf32>):
      %443 = stablehlo.broadcast_in_dim %arg849, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %444 = stablehlo.subtract %arg848, %443 : tensor<256x28x28x512xf32>
      %445 = stablehlo.broadcast_in_dim %arg847, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x28x28x512xf32>
      %447 = stablehlo.broadcast_in_dim %arg846, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x28x28x512xf32>
      %449 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
      %450 = stablehlo.convert %449 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %451 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %452 = stablehlo.subtract %450, %451 : tensor<256x28x28x512xf32>
      %453 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %454 = stablehlo.multiply %452, %453 : tensor<256x28x28x512xf32>
      %455 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %456 = stablehlo.add %454, %455 : tensor<256x28x28x512xf32>
      %457 = stablehlo.add %448, %456 : tensor<256x28x28x512xf32>
      %458 = stablehlo.convert %457 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
      mhlo.return %458 : tensor<256x28x28x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<200704x512xbf16>, tensor<512xf32>, tensor<512xf32>, tensor<256x28x28x512xf32>, tensor<512xf32>) -> tensor<256x28x28x512xbf16>
    %121 = mhlo.bitcast %120 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x512xbf16>) -> tensor<200704x512xbf16>
    %122 = mhlo.bitcast %arg259 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x128xf32>) -> tensor<512x128xf32>
    %123 = "mhlo.fusion"(%121, %122) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<200704x512xbf16>, %arg843: tensor<512x128xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<200704x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<200704x512xbf16>
        mhlo.return %447 : tensor<200704x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<512x128xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
        mhlo.return %446 : tensor<512x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x128xf32>) -> tensor<512x128xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x128xbf16>) -> tensor<200704x128xbf16>
      mhlo.return %445 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>, tensor<512x128xf32>) -> tensor<200704x128xbf16>
    %124 = "mhlo.fusion"(%arg253, %62, %arg798, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>, %arg844: tensor<128xf32>, %arg845: tensor<200704x128xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x128xbf16>) -> tensor<256x28x28x128xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x28x28x128xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x28x28x128xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x28x28x128xf32>
      %451 = stablehlo.convert %450 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x28x28x128xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %454 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32>
    %125 = stablehlo.convolution(%124, %arg260) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %126 = "mhlo.fusion"(%61, %arg255, %125, %arg800) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>, %arg844: tensor<256x28x28x128xf32>, %arg845: tensor<128xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x28x28x128xf32>
      %445 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x28x28x128xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x28x28x128xf32>
      %449 = stablehlo.convert %448 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      mhlo.return %449 : tensor<256x28x28x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xbf16>
    %127 = mhlo.bitcast %126 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x128xbf16>) -> tensor<200704x128xbf16>
    %128 = mhlo.bitcast %arg261 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %129 = "mhlo.fusion"(%127, %128) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<200704x128xbf16>, %arg843: tensor<128x512xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<200704x128xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<200704x128xbf16>
        mhlo.return %447 : tensor<200704x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<128x512xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
        mhlo.return %446 : tensor<128x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
      mhlo.return %445 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>, tensor<128x512xf32>) -> tensor<200704x512xbf16>
    %130 = "mhlo.fusion"(%arg257, %60, %arg802, %129, %120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<512xf32>, %arg845: tensor<200704x512xbf16>, %arg846: tensor<256x28x28x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x28x28x512xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %446 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %448 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x28x28x512xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x28x28x512xf32>
      %452 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x28x28x512xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x28x28x512xf32>
      %455 = stablehlo.convert %454 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
      mhlo.return %455 : tensor<256x28x28x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<200704x512xbf16>, tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %131 = mhlo.bitcast %130 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x512xbf16>) -> tensor<200704x512xbf16>
    %132 = mhlo.bitcast %arg268 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x128xf32>) -> tensor<512x128xf32>
    %133 = "mhlo.fusion"(%131, %132) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<200704x512xbf16>, %arg843: tensor<512x128xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<200704x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<200704x512xbf16>
        mhlo.return %447 : tensor<200704x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<512x128xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
        mhlo.return %446 : tensor<512x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x128xf32>) -> tensor<512x128xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x128xbf16>) -> tensor<200704x128xbf16>
      mhlo.return %445 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>, tensor<512x128xf32>) -> tensor<200704x128xbf16>
    %134 = "mhlo.fusion"(%arg262, %59, %arg804, %133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>, %arg844: tensor<128xf32>, %arg845: tensor<200704x128xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x128xbf16>) -> tensor<256x28x28x128xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x28x28x128xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x28x28x128xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x28x28x128xf32>
      %451 = stablehlo.convert %450 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x28x28x128xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %454 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32>
    %135 = stablehlo.convolution(%134, %arg269) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %136 = "mhlo.fusion"(%58, %arg264, %135, %arg806) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>, %arg844: tensor<256x28x28x128xf32>, %arg845: tensor<128xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x28x28x128xf32>
      %445 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x28x28x128xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x28x28x128xf32>
      %449 = stablehlo.convert %448 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      mhlo.return %449 : tensor<256x28x28x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xbf16>
    %137 = mhlo.bitcast %136 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x128xbf16>) -> tensor<200704x128xbf16>
    %138 = mhlo.bitcast %arg270 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %139 = "mhlo.fusion"(%137, %138) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<200704x128xbf16>, %arg843: tensor<128x512xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<200704x128xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<200704x128xbf16>
        mhlo.return %447 : tensor<200704x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<128x512xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
        mhlo.return %446 : tensor<128x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
      mhlo.return %445 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>, tensor<128x512xf32>) -> tensor<200704x512xbf16>
    %140 = "mhlo.fusion"(%arg266, %57, %arg808, %139, %130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<512xf32>, %arg845: tensor<200704x512xbf16>, %arg846: tensor<256x28x28x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x28x28x512xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %446 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %448 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x28x28x512xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x28x28x512xf32>
      %452 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x28x28x512xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x28x28x512xf32>
      %455 = stablehlo.convert %454 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
      mhlo.return %455 : tensor<256x28x28x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<200704x512xbf16>, tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %141 = mhlo.bitcast %140 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x512xbf16>) -> tensor<200704x512xbf16>
    %142 = mhlo.bitcast %arg277 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x128xf32>) -> tensor<512x128xf32>
    %143 = "mhlo.fusion"(%141, %142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<200704x512xbf16>, %arg843: tensor<512x128xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<200704x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<200704x512xbf16>
        mhlo.return %447 : tensor<200704x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<512x128xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<512x128xf32>) -> tensor<512x128xbf16>
        mhlo.return %446 : tensor<512x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x128xf32>) -> tensor<512x128xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x128xbf16>) -> tensor<200704x128xbf16>
      mhlo.return %445 : tensor<200704x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>, tensor<512x128xf32>) -> tensor<200704x128xbf16>
    %144 = "mhlo.fusion"(%arg271, %56, %arg810, %143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>, %arg844: tensor<128xf32>, %arg845: tensor<200704x128xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x128xbf16>) -> tensor<256x28x28x128xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x28x28x128xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x28x28x128xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x28x28x128xf32>
      %451 = stablehlo.convert %450 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x28x28x128xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
      mhlo.return %454 : tensor<256x28x28x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<200704x128xbf16>) -> tensor<256x28x28x128xf32>
    %145 = stablehlo.convolution(%144, %arg278) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<256x28x28x128xf32>
    %146 = "mhlo.fusion"(%55, %arg273, %145, %arg812) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<128xf32>, %arg843: tensor<128xf32>, %arg844: tensor<256x28x28x128xf32>, %arg845: tensor<128xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x28x28x128xf32>
      %445 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x28x28x128xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x28x28x128xf32>
      %449 = stablehlo.convert %448 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
      mhlo.return %449 : tensor<256x28x28x128xbf16>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<256x28x28x128xf32>, tensor<128xf32>) -> tensor<256x28x28x128xbf16>
    %147 = mhlo.bitcast %146 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x128xbf16>) -> tensor<200704x128xbf16>
    %148 = mhlo.bitcast %arg279 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x512xf32>) -> tensor<128x512xf32>
    %149 = "mhlo.fusion"(%147, %148) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<200704x128xbf16>, %arg843: tensor<128x512xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<200704x128xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x128xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<200704x128xbf16>
        mhlo.return %447 : tensor<200704x128xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>) -> tensor<200704x128xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<128x512xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<128x512xf32>) -> tensor<128x512xbf16>
        mhlo.return %446 : tensor<128x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<128x512xf32>) -> tensor<128x512xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<200704x128xbf16>, tensor<128x512xbf16>) -> tensor<200704x512xbf16>
      mhlo.return %445 : tensor<200704x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x128xbf16>, tensor<128x512xf32>) -> tensor<200704x512xbf16>
    %150:2 = "mhlo.fusion"(%arg275, %54, %arg814, %149, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<512xf32>, %arg845: tensor<200704x512xbf16>, %arg846: tensor<256x28x28x512xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x28x28x512xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %446 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x512xbf16>) -> tensor<256x28x28x512xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      %448 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x28x28x512xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x28x28x512xf32>
      %452 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x28x28x512xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x28x28x512xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x28x28x512xf32>
      %455 = stablehlo.convert %454 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
      %456 = stablehlo.maximum %455, %443 : tensor<256x28x28x512xbf16>
      %457 = stablehlo.convert %456 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
      mhlo.return %457, %455 : tensor<256x28x28x512xf32>, tensor<256x28x28x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<200704x512xbf16>, tensor<256x28x28x512xbf16>) -> (tensor<256x28x28x512xf32>, tensor<256x28x28x512xbf16>)
    %151 = mhlo.bitcast %150#1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x28x28x512xbf16>) -> tensor<200704x512xbf16>
    %152 = mhlo.bitcast %arg286 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %153 = "mhlo.fusion"(%151, %152) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<200704x512xbf16>, %arg843: tensor<512x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<200704x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<200704x512xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<200704x512xbf16>
        mhlo.return %447 : tensor<200704x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>) -> tensor<200704x512xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<512x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<512x256xf32>) -> tensor<512x256xbf16>
        mhlo.return %446 : tensor<512x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x256xf32>) -> tensor<512x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<200704x512xbf16>, tensor<512x256xbf16>) -> tensor<200704x256xbf16>
      mhlo.return %445 : tensor<200704x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<200704x512xbf16>, tensor<512x256xf32>) -> tensor<200704x256xbf16>
    %154 = "mhlo.fusion"(%arg280, %53, %arg816, %153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<200704x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<200704x256xbf16>) -> tensor<256x28x28x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x28x28x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x28x28x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x28x28x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x28x28x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x28x28x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x28x28x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x28x28x256xf32>) -> tensor<256x28x28x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x28x28x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xf32>
      mhlo.return %454 : tensor<256x28x28x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<200704x256xbf16>) -> tensor<256x28x28x256xf32>
    %155 = stablehlo.convolution(%154, %arg287) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %156 = "mhlo.fusion"(%arg282, %52, %155, %arg818) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %157 = mhlo.bitcast %156 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %158 = mhlo.bitcast %arg288 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %159 = "mhlo.fusion"(%157, %158) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %160 = "mhlo.fusion"(%arg285, %arg821) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %161 = "mhlo.fusion"(%arg291, %arg823) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %162 = stablehlo.convolution(%150#0, %arg289) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf32>, tensor<1x1x512x1024xf32>) -> tensor<256x14x14x1024xf32>
    %163 = "mhlo.fusion"(%159, %160, %161, %arg284, %arg820, %arg290, %162, %arg822) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<1024xf32>, %arg847: tensor<1024xf32>, %arg848: tensor<256x14x14x1024xf32>, %arg849: tensor<1024xf32>):
      %443 = stablehlo.broadcast_in_dim %arg849, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %444 = stablehlo.subtract %arg848, %443 : tensor<256x14x14x1024xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x1024xf32>
      %447 = stablehlo.broadcast_in_dim %arg847, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x1024xf32>
      %449 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %450 = stablehlo.convert %449 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.broadcast_in_dim %arg846, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %452 = stablehlo.subtract %450, %451 : tensor<256x14x14x1024xf32>
      %453 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %454 = stablehlo.multiply %452, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %456 = stablehlo.add %454, %455 : tensor<256x14x14x1024xf32>
      %457 = stablehlo.add %448, %456 : tensor<256x14x14x1024xf32>
      %458 = stablehlo.convert %457 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %458 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xf32>, tensor<1024xf32>) -> tensor<256x14x14x1024xbf16>
    %164 = mhlo.bitcast %163 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %165 = mhlo.bitcast %arg298 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %166 = "mhlo.fusion"(%164, %165) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %167 = "mhlo.fusion"(%arg292, %51, %arg824, %166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %168 = stablehlo.convolution(%167, %arg299) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %169 = "mhlo.fusion"(%arg294, %50, %168, %arg826) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %170 = mhlo.bitcast %169 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %171 = mhlo.bitcast %arg300 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %172 = "mhlo.fusion"(%170, %171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %173 = "mhlo.fusion"(%arg297, %arg829) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %174 = "mhlo.fusion"(%172, %173, %arg296, %arg828, %163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %175 = mhlo.bitcast %174 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %176 = mhlo.bitcast %arg307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %177 = "mhlo.fusion"(%175, %176) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %178 = "mhlo.fusion"(%arg301, %49, %arg830, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %179 = stablehlo.convolution(%178, %arg308) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %180 = "mhlo.fusion"(%arg303, %48, %179, %arg832) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %181 = mhlo.bitcast %180 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %182 = mhlo.bitcast %arg309 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %183 = "mhlo.fusion"(%181, %182) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %184 = "mhlo.fusion"(%arg306, %arg835) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %185 = "mhlo.fusion"(%183, %184, %arg305, %arg834, %174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %186 = mhlo.bitcast %185 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %187 = mhlo.bitcast %arg28 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %188 = "mhlo.fusion"(%186, %187) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %189 = "mhlo.fusion"(%arg22, %47, %arg644, %188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %190 = stablehlo.convolution(%189, %arg29) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %191 = "mhlo.fusion"(%arg24, %46, %190, %arg646) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %192 = mhlo.bitcast %191 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %193 = mhlo.bitcast %arg30 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %194 = "mhlo.fusion"(%192, %193) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %195 = "mhlo.fusion"(%arg27, %arg649) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %196 = "mhlo.fusion"(%194, %195, %arg26, %arg648, %185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %197 = mhlo.bitcast %196 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %198 = mhlo.bitcast %arg37 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %199 = "mhlo.fusion"(%197, %198) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %200 = "mhlo.fusion"(%arg31, %45, %arg650, %199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %201 = stablehlo.convolution(%200, %arg38) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %202 = "mhlo.fusion"(%arg33, %44, %201, %arg652) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %203 = mhlo.bitcast %202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %204 = mhlo.bitcast %arg39 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %205 = "mhlo.fusion"(%203, %204) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %206 = "mhlo.fusion"(%arg36, %arg655) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %207 = "mhlo.fusion"(%205, %206, %arg35, %arg654, %196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %208 = mhlo.bitcast %207 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %209 = mhlo.bitcast %arg46 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %210 = "mhlo.fusion"(%208, %209) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %211 = "mhlo.fusion"(%arg40, %43, %arg656, %210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %212 = stablehlo.convolution(%211, %arg47) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %213 = "mhlo.fusion"(%arg42, %42, %212, %arg658) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %214 = mhlo.bitcast %213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %215 = mhlo.bitcast %arg48 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %216 = "mhlo.fusion"(%214, %215) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %217 = "mhlo.fusion"(%arg45, %arg661) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %218 = "mhlo.fusion"(%216, %217, %arg44, %arg660, %207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %219 = mhlo.bitcast %218 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %220 = mhlo.bitcast %arg55 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %221 = "mhlo.fusion"(%219, %220) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %222 = "mhlo.fusion"(%arg49, %41, %arg662, %221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %223 = stablehlo.convolution(%222, %arg56) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %224 = "mhlo.fusion"(%arg51, %40, %223, %arg664) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %225 = mhlo.bitcast %224 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %226 = mhlo.bitcast %arg57 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %227 = "mhlo.fusion"(%225, %226) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %228 = "mhlo.fusion"(%arg54, %arg667) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %229 = "mhlo.fusion"(%227, %228, %arg53, %arg666, %218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %230 = mhlo.bitcast %229 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %231 = mhlo.bitcast %arg64 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %232 = "mhlo.fusion"(%230, %231) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %233 = "mhlo.fusion"(%arg58, %39, %arg668, %232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %234 = stablehlo.convolution(%233, %arg65) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %235 = "mhlo.fusion"(%arg60, %38, %234, %arg670) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %236 = mhlo.bitcast %235 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %237 = mhlo.bitcast %arg66 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %238 = "mhlo.fusion"(%236, %237) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %239 = "mhlo.fusion"(%arg63, %arg673) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %240 = "mhlo.fusion"(%238, %239, %arg62, %arg672, %229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %241 = mhlo.bitcast %240 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %242 = mhlo.bitcast %arg73 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %243 = "mhlo.fusion"(%241, %242) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %244 = "mhlo.fusion"(%arg67, %37, %arg674, %243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %245 = stablehlo.convolution(%244, %arg74) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %246 = "mhlo.fusion"(%arg69, %36, %245, %arg676) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %247 = mhlo.bitcast %246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %248 = mhlo.bitcast %arg75 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %249 = "mhlo.fusion"(%247, %248) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %250 = "mhlo.fusion"(%arg72, %arg679) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %251 = "mhlo.fusion"(%249, %250, %arg71, %arg678, %240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %252 = mhlo.bitcast %251 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %253 = mhlo.bitcast %arg82 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %254 = "mhlo.fusion"(%252, %253) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %255 = "mhlo.fusion"(%arg76, %35, %arg680, %254) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %256 = stablehlo.convolution(%255, %arg83) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %257 = "mhlo.fusion"(%arg78, %34, %256, %arg682) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %258 = mhlo.bitcast %257 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %259 = mhlo.bitcast %arg84 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %260 = "mhlo.fusion"(%258, %259) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %261 = "mhlo.fusion"(%arg81, %arg685) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %262 = "mhlo.fusion"(%260, %261, %arg80, %arg684, %251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %263 = mhlo.bitcast %262 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %264 = mhlo.bitcast %arg91 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %265 = "mhlo.fusion"(%263, %264) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %266 = "mhlo.fusion"(%arg85, %33, %arg686, %265) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %267 = stablehlo.convolution(%266, %arg92) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %268 = "mhlo.fusion"(%arg87, %32, %267, %arg688) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %269 = mhlo.bitcast %268 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %270 = mhlo.bitcast %arg93 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %271 = "mhlo.fusion"(%269, %270) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %272 = "mhlo.fusion"(%arg90, %arg691) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %273 = "mhlo.fusion"(%271, %272, %arg89, %arg690, %262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %274 = mhlo.bitcast %273 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %275 = mhlo.bitcast %arg100 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %276 = "mhlo.fusion"(%274, %275) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %277 = "mhlo.fusion"(%arg94, %31, %arg692, %276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %278 = stablehlo.convolution(%277, %arg101) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %279 = "mhlo.fusion"(%arg96, %30, %278, %arg694) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %280 = mhlo.bitcast %279 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %281 = mhlo.bitcast %arg102 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %282 = "mhlo.fusion"(%280, %281) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %283 = "mhlo.fusion"(%arg99, %arg697) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %284 = "mhlo.fusion"(%282, %283, %arg98, %arg696, %273) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %285 = mhlo.bitcast %284 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %286 = mhlo.bitcast %arg109 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %287 = "mhlo.fusion"(%285, %286) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %288 = "mhlo.fusion"(%arg103, %29, %arg698, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %289 = stablehlo.convolution(%288, %arg110) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %290 = "mhlo.fusion"(%arg105, %28, %289, %arg700) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %291 = mhlo.bitcast %290 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %292 = mhlo.bitcast %arg111 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %293 = "mhlo.fusion"(%291, %292) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %294 = "mhlo.fusion"(%arg108, %arg703) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %295 = "mhlo.fusion"(%293, %294, %arg107, %arg702, %284) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %297 = mhlo.bitcast %arg127 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %298 = "mhlo.fusion"(%296, %297) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %299 = "mhlo.fusion"(%arg121, %27, %arg710, %298) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %300 = stablehlo.convolution(%299, %arg128) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %301 = "mhlo.fusion"(%arg123, %26, %300, %arg712) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %302 = mhlo.bitcast %301 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %303 = mhlo.bitcast %arg129 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %304 = "mhlo.fusion"(%302, %303) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %305 = "mhlo.fusion"(%arg126, %arg715) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %306 = "mhlo.fusion"(%304, %305, %arg125, %arg714, %295) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %307 = mhlo.bitcast %306 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %308 = mhlo.bitcast %arg136 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %309 = "mhlo.fusion"(%307, %308) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %310 = "mhlo.fusion"(%arg130, %25, %arg716, %309) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %311 = stablehlo.convolution(%310, %arg137) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %312 = "mhlo.fusion"(%arg132, %24, %311, %arg718) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %313 = mhlo.bitcast %312 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %314 = mhlo.bitcast %arg138 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %315 = "mhlo.fusion"(%313, %314) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %316 = "mhlo.fusion"(%arg135, %arg721) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %317 = "mhlo.fusion"(%315, %316, %arg134, %arg720, %306) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %318 = mhlo.bitcast %317 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %319 = mhlo.bitcast %arg145 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %320 = "mhlo.fusion"(%318, %319) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %321 = "mhlo.fusion"(%arg139, %23, %arg722, %320) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %322 = stablehlo.convolution(%321, %arg146) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %323 = "mhlo.fusion"(%arg141, %22, %322, %arg724) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %324 = mhlo.bitcast %323 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %325 = mhlo.bitcast %arg147 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %326 = "mhlo.fusion"(%324, %325) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %327 = "mhlo.fusion"(%arg144, %arg727) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %328 = "mhlo.fusion"(%326, %327, %arg143, %arg726, %317) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %329 = mhlo.bitcast %328 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %330 = mhlo.bitcast %arg154 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %331 = "mhlo.fusion"(%329, %330) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %332 = "mhlo.fusion"(%arg148, %21, %arg728, %331) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %333 = stablehlo.convolution(%332, %arg155) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %334 = "mhlo.fusion"(%arg150, %20, %333, %arg730) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %335 = mhlo.bitcast %334 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %336 = mhlo.bitcast %arg156 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %337 = "mhlo.fusion"(%335, %336) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %338 = "mhlo.fusion"(%arg153, %arg733) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %339 = "mhlo.fusion"(%337, %338, %arg152, %arg732, %328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %340 = mhlo.bitcast %339 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %341 = mhlo.bitcast %arg163 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %342 = "mhlo.fusion"(%340, %341) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %343 = "mhlo.fusion"(%arg157, %19, %arg734, %342) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %344 = stablehlo.convolution(%343, %arg164) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %345 = "mhlo.fusion"(%arg159, %18, %344, %arg736) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %346 = mhlo.bitcast %345 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %347 = mhlo.bitcast %arg165 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %348 = "mhlo.fusion"(%346, %347) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %349 = "mhlo.fusion"(%arg162, %arg739) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %350 = "mhlo.fusion"(%348, %349, %arg161, %arg738, %339) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %351 = mhlo.bitcast %350 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %352 = mhlo.bitcast %arg172 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %353 = "mhlo.fusion"(%351, %352) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %354 = "mhlo.fusion"(%arg166, %17, %arg740, %353) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %355 = stablehlo.convolution(%354, %arg173) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %356 = "mhlo.fusion"(%arg168, %16, %355, %arg742) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %357 = mhlo.bitcast %356 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %358 = mhlo.bitcast %arg174 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %359 = "mhlo.fusion"(%357, %358) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %360 = "mhlo.fusion"(%arg171, %arg745) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %361 = "mhlo.fusion"(%359, %360, %arg170, %arg744, %350) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %362 = mhlo.bitcast %361 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %363 = mhlo.bitcast %arg181 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %364 = "mhlo.fusion"(%362, %363) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %365 = "mhlo.fusion"(%arg175, %15, %arg746, %364) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %366 = stablehlo.convolution(%365, %arg182) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %367 = "mhlo.fusion"(%arg177, %14, %366, %arg748) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %368 = mhlo.bitcast %367 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %369 = mhlo.bitcast %arg183 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %370 = "mhlo.fusion"(%368, %369) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %371 = "mhlo.fusion"(%arg180, %arg751) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %372 = "mhlo.fusion"(%370, %371, %arg179, %arg750, %361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %373 = mhlo.bitcast %372 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %374 = mhlo.bitcast %arg190 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %375 = "mhlo.fusion"(%373, %374) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %376 = "mhlo.fusion"(%arg184, %13, %arg752, %375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %377 = stablehlo.convolution(%376, %arg191) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %378 = "mhlo.fusion"(%arg186, %12, %377, %arg754) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %379 = mhlo.bitcast %378 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %380 = mhlo.bitcast %arg192 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %381 = "mhlo.fusion"(%379, %380) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %382 = "mhlo.fusion"(%arg189, %arg757) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %383 = "mhlo.fusion"(%381, %382, %arg188, %arg756, %372) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %385 = mhlo.bitcast %arg199 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %386 = "mhlo.fusion"(%384, %385) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %387 = "mhlo.fusion"(%arg193, %11, %arg758, %386) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %388 = stablehlo.convolution(%387, %arg200) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %389 = "mhlo.fusion"(%arg195, %10, %388, %arg760) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %391 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %392 = "mhlo.fusion"(%390, %391) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %393 = "mhlo.fusion"(%arg198, %arg763) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<1024xf32>
      %445 = stablehlo.rsqrt %444 : tensor<1024xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<1024xf32>
      mhlo.return %446 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %394 = "mhlo.fusion"(%392, %393, %arg197, %arg762, %383) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<1024xf32>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      mhlo.return %455 : tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %395 = mhlo.bitcast %394 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %396 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %397 = "mhlo.fusion"(%395, %396) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x256xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x256xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
        mhlo.return %446 : tensor<1024x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024x256xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x256xbf16>) -> tensor<50176x256xbf16>
      mhlo.return %445 : tensor<50176x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x256xf32>) -> tensor<50176x256xbf16>
    %398 = "mhlo.fusion"(%arg202, %9, %arg764, %397) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<50176x256xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x256xbf16>) -> tensor<256x14x14x256xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x256xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x256xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
      mhlo.return %454 : tensor<256x14x14x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<50176x256xbf16>) -> tensor<256x14x14x256xf32>
    %399 = stablehlo.convolution(%398, %arg209) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<256x14x14x256xf32>
    %400 = "mhlo.fusion"(%arg204, %8, %399, %arg766) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256x14x14x256xf32>, %arg845: tensor<256xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x14x14x256xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x14x14x256xf32>
      %449 = stablehlo.convert %448 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
      mhlo.return %449 : tensor<256x14x14x256xbf16>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256x14x14x256xf32>, tensor<256xf32>) -> tensor<256x14x14x256xbf16>
    %401 = mhlo.bitcast %400 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x256xbf16>) -> tensor<50176x256xbf16>
    %402 = mhlo.bitcast %arg210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x1024xf32>) -> tensor<256x1024xf32>
    %403 = "mhlo.fusion"(%401, %402) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x256xbf16>, %arg843: tensor<256x1024xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x256xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x256xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x256xbf16>
        mhlo.return %447 : tensor<50176x256xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>) -> tensor<50176x256xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<256x1024xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
        mhlo.return %446 : tensor<256x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<256x1024xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x256xbf16>, tensor<256x1024xbf16>) -> tensor<50176x1024xbf16>
      mhlo.return %445 : tensor<50176x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x256xbf16>, tensor<256x1024xf32>) -> tensor<50176x1024xbf16>
    %404:2 = "mhlo.fusion"(%arg206, %7, %arg768, %403, %394) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1024xf32>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<50176x1024xbf16>, %arg846: tensor<256x14x14x1024xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x14x14x1024xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %446 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x1024xbf16>) -> tensor<256x14x14x1024xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      %448 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x14x14x1024xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x14x14x1024xf32>
      %452 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<1024xf32>) -> tensor<256x14x14x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x14x14x1024xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x14x14x1024xf32>
      %455 = stablehlo.convert %454 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
      %456 = stablehlo.maximum %455, %443 : tensor<256x14x14x1024xbf16>
      %457 = stablehlo.convert %456 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
      mhlo.return %457, %455 : tensor<256x14x14x1024xf32>, tensor<256x14x14x1024xbf16>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<50176x1024xbf16>, tensor<256x14x14x1024xbf16>) -> (tensor<256x14x14x1024xf32>, tensor<256x14x14x1024xbf16>)
    %405 = mhlo.bitcast %404#1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x14x14x1024xbf16>) -> tensor<50176x1024xbf16>
    %406 = mhlo.bitcast %arg229 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x512xf32>) -> tensor<1024x512xf32>
    %407 = "mhlo.fusion"(%405, %406) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<50176x1024xbf16>, %arg843: tensor<1024x512xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<50176x1024xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<50176x1024xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<50176x1024xbf16>
        mhlo.return %447 : tensor<50176x1024xbf16>
      }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>) -> tensor<50176x1024xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<1024x512xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<1024x512xf32>) -> tensor<1024x512xbf16>
        mhlo.return %446 : tensor<1024x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<1024x512xf32>) -> tensor<1024x512xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<50176x1024xbf16>, tensor<1024x512xbf16>) -> tensor<50176x512xbf16>
      mhlo.return %445 : tensor<50176x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<50176x1024xbf16>, tensor<1024x512xf32>) -> tensor<50176x512xbf16>
    %408 = "mhlo.fusion"(%arg223, %6, %arg778, %407) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<512xf32>, %arg845: tensor<50176x512xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50176x512xbf16>) -> tensor<256x14x14x512xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<512xf32>) -> tensor<256x14x14x512xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x14x14x512xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x14x14x512xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x14x14x512xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x14x14x512xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x14x14x512xf32>
      %451 = stablehlo.convert %450 : (tensor<256x14x14x512xf32>) -> tensor<256x14x14x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x512xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x14x14x512xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xf32>
      mhlo.return %454 : tensor<256x14x14x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<50176x512xbf16>) -> tensor<256x14x14x512xf32>
    %409 = stablehlo.convolution(%408, %arg230) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %410 = "mhlo.fusion"(%arg225, %5, %409, %arg780) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<256x7x7x512xf32>, %arg845: tensor<512xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x7x7x512xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x7x7x512xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x7x7x512xf32>
      %449 = stablehlo.convert %448 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      mhlo.return %449 : tensor<256x7x7x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xbf16>
    %411 = mhlo.bitcast %410 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x512xbf16>) -> tensor<12544x512xbf16>
    %412 = mhlo.bitcast %arg231 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x2048xf32>) -> tensor<512x2048xf32>
    %413 = "mhlo.fusion"(%411, %412) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<12544x512xbf16>, %arg843: tensor<512x2048xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<12544x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<12544x512xbf16>
        mhlo.return %447 : tensor<12544x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>) -> tensor<12544x512xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<512x2048xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
        mhlo.return %446 : tensor<512x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<12544x512xbf16>, tensor<512x2048xbf16>) -> tensor<12544x2048xbf16>
      mhlo.return %445 : tensor<12544x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>, tensor<512x2048xf32>) -> tensor<12544x2048xbf16>
    %414 = "mhlo.fusion"(%arg228, %arg783) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<2048xf32>, %arg843: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<2048xf32>
      %445 = stablehlo.rsqrt %444 : tensor<2048xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<2048xf32>
      mhlo.return %446 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %415 = "mhlo.fusion"(%arg234, %arg785) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<2048xf32>, %arg843: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<2048xf32>
      %445 = stablehlo.rsqrt %444 : tensor<2048xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<2048xf32>
      mhlo.return %446 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %416 = stablehlo.convolution(%404#0, %arg232) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf32>, tensor<1x1x1024x2048xf32>) -> tensor<256x7x7x2048xf32>
    %417 = "mhlo.fusion"(%413, %414, %415, %arg227, %arg782, %arg233, %416, %arg784) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<12544x2048xbf16>, %arg843: tensor<2048xf32>, %arg844: tensor<2048xf32>, %arg845: tensor<2048xf32>, %arg846: tensor<2048xf32>, %arg847: tensor<2048xf32>, %arg848: tensor<256x7x7x2048xf32>, %arg849: tensor<2048xf32>):
      %443 = stablehlo.broadcast_in_dim %arg849, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %444 = stablehlo.subtract %arg848, %443 : tensor<256x7x7x2048xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x7x7x2048xf32>
      %447 = stablehlo.broadcast_in_dim %arg847, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x7x7x2048xf32>
      %449 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x2048xbf16>) -> tensor<256x7x7x2048xbf16>
      %450 = stablehlo.convert %449 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %451 = stablehlo.broadcast_in_dim %arg846, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %452 = stablehlo.subtract %450, %451 : tensor<256x7x7x2048xf32>
      %453 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %454 = stablehlo.multiply %452, %453 : tensor<256x7x7x2048xf32>
      %455 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %456 = stablehlo.add %454, %455 : tensor<256x7x7x2048xf32>
      %457 = stablehlo.add %448, %456 : tensor<256x7x7x2048xf32>
      %458 = stablehlo.convert %457 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
      mhlo.return %458 : tensor<256x7x7x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<256x7x7x2048xf32>, tensor<2048xf32>) -> tensor<256x7x7x2048xbf16>
    %418 = mhlo.bitcast %417 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x2048xbf16>) -> tensor<12544x2048xbf16>
    %419 = mhlo.bitcast %arg241 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2048x512xf32>) -> tensor<2048x512xf32>
    %420 = "mhlo.fusion"(%418, %419) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<12544x2048xbf16>, %arg843: tensor<2048x512xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<12544x2048xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<12544x2048xbf16>
        mhlo.return %447 : tensor<12544x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<2048x512xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
        mhlo.return %446 : tensor<2048x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<12544x2048xbf16>, tensor<2048x512xbf16>) -> tensor<12544x512xbf16>
      mhlo.return %445 : tensor<12544x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>, tensor<2048x512xf32>) -> tensor<12544x512xbf16>
    %421 = "mhlo.fusion"(%arg235, %4, %arg786, %420) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<512xf32>, %arg845: tensor<12544x512xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x512xbf16>) -> tensor<256x7x7x512xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x7x7x512xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x7x7x512xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x7x7x512xf32>
      %451 = stablehlo.convert %450 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x7x7x512xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %454 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<12544x512xbf16>) -> tensor<256x7x7x512xf32>
    %422 = stablehlo.convolution(%421, %arg242) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %423 = "mhlo.fusion"(%arg237, %3, %422, %arg788) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<256x7x7x512xf32>, %arg845: tensor<512xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x7x7x512xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x7x7x512xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x7x7x512xf32>
      %449 = stablehlo.convert %448 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      mhlo.return %449 : tensor<256x7x7x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xbf16>
    %424 = mhlo.bitcast %423 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x512xbf16>) -> tensor<12544x512xbf16>
    %425 = mhlo.bitcast %arg243 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x2048xf32>) -> tensor<512x2048xf32>
    %426 = "mhlo.fusion"(%424, %425) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<12544x512xbf16>, %arg843: tensor<512x2048xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<12544x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<12544x512xbf16>
        mhlo.return %447 : tensor<12544x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>) -> tensor<12544x512xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<512x2048xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
        mhlo.return %446 : tensor<512x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<12544x512xbf16>, tensor<512x2048xbf16>) -> tensor<12544x2048xbf16>
      mhlo.return %445 : tensor<12544x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>, tensor<512x2048xf32>) -> tensor<12544x2048xbf16>
    %427 = "mhlo.fusion"(%arg240, %arg791) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<2048xf32>, %arg843: tensor<2048xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2048xf32>
      %444 = stablehlo.add %arg843, %443 : tensor<2048xf32>
      %445 = stablehlo.rsqrt %444 : tensor<2048xf32>
      %446 = stablehlo.multiply %445, %arg842 : tensor<2048xf32>
      mhlo.return %446 : tensor<2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    %428 = "mhlo.fusion"(%426, %427, %arg239, %arg790, %417) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<12544x2048xbf16>, %arg843: tensor<2048xf32>, %arg844: tensor<2048xf32>, %arg845: tensor<2048xf32>, %arg846: tensor<256x7x7x2048xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x2048xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x7x7x2048xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %446 = mhlo.bitcast %arg842 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x2048xbf16>) -> tensor<256x7x7x2048xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %448 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x7x7x2048xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x7x7x2048xf32>
      %452 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x7x7x2048xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x7x7x2048xf32>
      %455 = stablehlo.convert %454 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
      mhlo.return %455 : tensor<256x7x7x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %429 = mhlo.bitcast %428 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x2048xbf16>) -> tensor<12544x2048xbf16>
    %430 = mhlo.bitcast %arg250 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2048x512xf32>) -> tensor<2048x512xf32>
    %431 = "mhlo.fusion"(%429, %430) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<12544x2048xbf16>, %arg843: tensor<2048x512xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<12544x2048xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x2048xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<12544x2048xbf16>
        mhlo.return %447 : tensor<12544x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>) -> tensor<12544x2048xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<2048x512xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
        mhlo.return %446 : tensor<2048x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<2048x512xf32>) -> tensor<2048x512xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<12544x2048xbf16>, tensor<2048x512xbf16>) -> tensor<12544x512xbf16>
      mhlo.return %445 : tensor<12544x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x2048xbf16>, tensor<2048x512xf32>) -> tensor<12544x512xbf16>
    %432 = "mhlo.fusion"(%arg244, %2, %arg792, %431) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<512xf32>, %arg845: tensor<12544x512xbf16>):
      %443 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x512xbf16>) -> tensor<256x7x7x512xbf16>
      %444 = stablehlo.convert %443 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      %445 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %446 = stablehlo.subtract %444, %445 : tensor<256x7x7x512xf32>
      %447 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %448 = stablehlo.multiply %446, %447 : tensor<256x7x7x512xf32>
      %449 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %450 = stablehlo.add %448, %449 : tensor<256x7x7x512xf32>
      %451 = stablehlo.convert %450 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
      %453 = stablehlo.maximum %451, %452 : tensor<256x7x7x512xbf16>
      %454 = stablehlo.convert %453 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
      mhlo.return %454 : tensor<256x7x7x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<12544x512xbf16>) -> tensor<256x7x7x512xf32>
    %433 = stablehlo.convolution(%432, %arg251) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<256x7x7x512xf32>
    %434 = "mhlo.fusion"(%arg246, %1, %433, %arg794) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<512xf32>, %arg843: tensor<512xf32>, %arg844: tensor<256x7x7x512xf32>, %arg845: tensor<512xf32>):
      %443 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %444 = stablehlo.subtract %arg844, %443 : tensor<256x7x7x512xf32>
      %445 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %446 = stablehlo.multiply %444, %445 : tensor<256x7x7x512xf32>
      %447 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %448 = stablehlo.add %446, %447 : tensor<256x7x7x512xf32>
      %449 = stablehlo.convert %448 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
      mhlo.return %449 : tensor<256x7x7x512xbf16>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<256x7x7x512xf32>, tensor<512xf32>) -> tensor<256x7x7x512xbf16>
    %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x512xbf16>) -> tensor<12544x512xbf16>
    %436 = mhlo.bitcast %arg252 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x2048xf32>) -> tensor<512x2048xf32>
    %437 = "mhlo.fusion"(%435, %436) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<12544x512xbf16>, %arg843: tensor<512x2048xf32>):
      %443 = "mhlo.fusion"(%arg842) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<12544x512xbf16>):
        %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %446 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<12544x512xbf16>
        %447 = stablehlo.maximum %arg844, %446 : tensor<12544x512xbf16>
        mhlo.return %447 : tensor<12544x512xbf16>
      }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>) -> tensor<12544x512xbf16>
      %444 = "mhlo.fusion"(%arg843) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
      ^bb0(%arg844: tensor<512x2048xf32>):
        %446 = stablehlo.convert %arg844 : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
        mhlo.return %446 : tensor<512x2048xbf16>
      }) {output_operand_aliasing = []} : (tensor<512x2048xf32>) -> tensor<512x2048xbf16>
      %445 = stablehlo.dot %443, %444, precision = [DEFAULT, DEFAULT] : (tensor<12544x512xbf16>, tensor<512x2048xbf16>) -> tensor<12544x2048xbf16>
      mhlo.return %445 : tensor<12544x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<12544x512xbf16>, tensor<512x2048xf32>) -> tensor<12544x2048xbf16>
    %438 = "mhlo.fusion"(%arg248, %0, %arg796, %437, %428) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg842: tensor<2048xf32>, %arg843: tensor<2048xf32>, %arg844: tensor<2048xf32>, %arg845: tensor<12544x2048xbf16>, %arg846: tensor<256x7x7x2048xbf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x2048xbf16>
      %444 = stablehlo.maximum %arg846, %443 : tensor<256x7x7x2048xbf16>
      %445 = stablehlo.convert %444 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %446 = mhlo.bitcast %arg845 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<12544x2048xbf16>) -> tensor<256x7x7x2048xbf16>
      %447 = stablehlo.convert %446 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %448 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %449 = stablehlo.subtract %447, %448 : tensor<256x7x7x2048xf32>
      %450 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %451 = stablehlo.multiply %449, %450 : tensor<256x7x7x2048xf32>
      %452 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<2048xf32>) -> tensor<256x7x7x2048xf32>
      %453 = stablehlo.add %451, %452 : tensor<256x7x7x2048xf32>
      %454 = stablehlo.add %445, %453 : tensor<256x7x7x2048xf32>
      %455 = stablehlo.convert %454 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
      %456 = stablehlo.maximum %455, %443 : tensor<256x7x7x2048xbf16>
      %457 = stablehlo.convert %456 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
      %458 = mhlo.bitcast %457 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<256x7x7x2048xf32>) -> tensor<256x49x2048xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.reduce(%458 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x49x2048xf32>, tensor<f32>) -> tensor<256x2048xf32>
      mhlo.return %459 : tensor<256x2048xf32>
    }) {output_operand_aliasing = []} : (tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<12544x2048xbf16>, tensor<256x7x7x2048xbf16>) -> tensor<256x2048xf32>
    %439 = "mhlo.fusion"(%438) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<256x2048xf32>):
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
      %444 = stablehlo.multiply %arg842, %443 : tensor<256x2048xf32>
      %445 = stablehlo.convert %444 : (tensor<256x2048xf32>) -> tensor<256x2048xbf16>
      mhlo.return %445 : tensor<256x2048xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x2048xf32>) -> tensor<256x2048xbf16>
    %440 = "mhlo.fusion"(%439, %arg311) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg842: tensor<256x2048xbf16>, %arg843: tensor<2048x1000xf32>):
      %443 = stablehlo.convert %arg843 : (tensor<2048x1000xf32>) -> tensor<2048x1000xbf16>
      %444 = stablehlo.dot_general %443, %arg842, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[1000,256]{0,1}"} : (tensor<2048x1000xbf16>, tensor<256x2048xbf16>) -> tensor<1000x256xbf16>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x256xbf16>) -> tensor<256x1000xbf16>
      mhlo.return %445 : tensor<256x1000xbf16>
    }) {output_operand_aliasing = []} : (tensor<256x2048xbf16>, tensor<2048x1000xf32>) -> tensor<256x1000xbf16>
    %441 = "mhlo.fusion"(%arg310, %440) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg842: tensor<1000xf32>, %arg843: tensor<256x1000xbf16>):
      %443 = stablehlo.convert %arg843 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
      %444 = stablehlo.broadcast_in_dim %arg842, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
      %445 = stablehlo.add %443, %444 : tensor<256x1000xf32>
      %446 = stablehlo.convert %445 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
      mhlo.return %446 : tensor<256x1000xbf16>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<256x1000xbf16>) -> tensor<256x1000xbf16>
    %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<256x1000xbf16>) -> tensor<1x256x1000xbf16>
    return %442 : tensor<1x256x1000xbf16>
  }
}
