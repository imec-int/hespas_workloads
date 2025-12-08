module @pmap_inference_step attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg1: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg2: tensor<512x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg3: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg4: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg5: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg6: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg7: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg8: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg9: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg10: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg11: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg12: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg13: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg14: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg15: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg16: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg17: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg18: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg19: tensor<3x3x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg20: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg21: tensor<1x1x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg22: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg23: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg24: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg25: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg26: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg27: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg28: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg29: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg30: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg31: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg32: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg33: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg34: tensor<3x3x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg35: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg36: tensor<1x1x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg37: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg38: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg39: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg40: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg41: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg42: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg43: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg44: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg45: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg46: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg47: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg48: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg49: tensor<3x3x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg50: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg51: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg52: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg53: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg54: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg55: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg56: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg57: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg58: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg59: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg60: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg61: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg62: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg63: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg64: tensor<512x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg65: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg66: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg67: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg68: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg69: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg70: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg71: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg72: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg73: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg74: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg75: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg76: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg77: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg78: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg79: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg80: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg81: tensor<3x3x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg82: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg83: tensor<1x1x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg84: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg85: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg86: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg87: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg88: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg89: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg90: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg91: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg92: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg93: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg94: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg95: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg96: tensor<3x3x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg97: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg98: tensor<1x1x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg99: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg100: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg101: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg102: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg103: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg104: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg105: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg106: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg107: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg108: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg109: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg110: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg111: tensor<3x3x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg112: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg113: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg114: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg115: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg116: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg117: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg118: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg119: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg120: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg121: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg122: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg123: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg124: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg125: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg126: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg127: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg128: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg129: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg130: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg131: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg132: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg133: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg134: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg135: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg136: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg137: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg138: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg139: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg140: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg141: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg142: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg143: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg144: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg145: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg146: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg147: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg148: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg149: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg150: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg151: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg152: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg153: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg154: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg155: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg156: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg157: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg158: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg159: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg160: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg161: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg162: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg163: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg164: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg165: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg166: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg167: tensor<f32> {mhlo.is_same_data_across_replicas = true}, %arg168: tensor<1x256x224x224x3xbf16>, %arg169: tensor<1x256xi32>) -> (tensor<1x256x1000xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.reshape %arg168 : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
    %1 = stablehlo.reshape %arg169 : (tensor<1x256xi32>) -> tensor<256xi32>
    %2 = stablehlo.convert %arg62 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xbf16>
    %3 = stablehlo.convolution(%0, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xbf16>, tensor<7x7x3x64xbf16>) -> tensor<256x112x112x64xbf16>
    %4 = stablehlo.broadcast_in_dim %arg164, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg165, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %6 = stablehlo.convert %3 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
    %7 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %8 = stablehlo.subtract %6, %7 : tensor<256x112x112x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %10 = stablehlo.add %5, %9 : tensor<1x1x1x64xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x1x1x64xf32>
    %12 = stablehlo.reshape %arg61 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<1x1x1x64xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %15 = stablehlo.multiply %8, %14 : tensor<256x112x112x64xf32>
    %16 = stablehlo.reshape %arg60 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %18 = stablehlo.add %15, %17 : tensor<256x112x112x64xf32>
    %19 = stablehlo.convert %18 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %20 = call @relu(%19) : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xbf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %21 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %22 = "stablehlo.reduce_window"(%20, %21) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg170: tensor<bf16>, %arg171: tensor<bf16>):
      %401 = stablehlo.maximum %arg170, %arg171 : tensor<bf16>
      stablehlo.return %401 : tensor<bf16>
    }) : (tensor<256x112x112x64xbf16>, tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %23 = stablehlo.convert %arg7 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %24 = stablehlo.convolution(%22, %23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %25 = stablehlo.broadcast_in_dim %arg126, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %26 = stablehlo.broadcast_in_dim %arg127, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %27 = stablehlo.convert %24 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %29 = stablehlo.subtract %27, %28 : tensor<256x56x56x64xf32>
    %30 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %31 = stablehlo.add %26, %30 : tensor<1x1x1x64xf32>
    %32 = stablehlo.rsqrt %31 : tensor<1x1x1x64xf32>
    %33 = stablehlo.reshape %arg4 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<1x1x1x64xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %36 = stablehlo.multiply %29, %35 : tensor<256x56x56x64xf32>
    %37 = stablehlo.reshape %arg3 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %39 = stablehlo.add %36, %38 : tensor<256x56x56x64xf32>
    %40 = stablehlo.convert %39 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %41 = call @relu_13(%40) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %42 = stablehlo.convert %arg8 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %43 = stablehlo.convolution(%41, %42) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %44 = stablehlo.broadcast_in_dim %arg128, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %45 = stablehlo.broadcast_in_dim %arg129, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %46 = stablehlo.convert %43 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %47 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %48 = stablehlo.subtract %46, %47 : tensor<256x56x56x64xf32>
    %49 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %50 = stablehlo.add %45, %49 : tensor<1x1x1x64xf32>
    %51 = stablehlo.rsqrt %50 : tensor<1x1x1x64xf32>
    %52 = stablehlo.reshape %arg6 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %53 = stablehlo.multiply %51, %52 : tensor<1x1x1x64xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %55 = stablehlo.multiply %48, %54 : tensor<256x56x56x64xf32>
    %56 = stablehlo.reshape %arg5 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %58 = stablehlo.add %55, %57 : tensor<256x56x56x64xf32>
    %59 = stablehlo.convert %58 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %60 = stablehlo.add %22, %59 : tensor<256x56x56x64xbf16>
    %61 = call @relu_13(%60) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %62 = stablehlo.convert %arg13 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %63 = stablehlo.convolution(%61, %62) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %64 = stablehlo.broadcast_in_dim %arg130, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %65 = stablehlo.broadcast_in_dim %arg131, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %66 = stablehlo.convert %63 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %67 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %68 = stablehlo.subtract %66, %67 : tensor<256x56x56x64xf32>
    %69 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %70 = stablehlo.add %65, %69 : tensor<1x1x1x64xf32>
    %71 = stablehlo.rsqrt %70 : tensor<1x1x1x64xf32>
    %72 = stablehlo.reshape %arg10 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %73 = stablehlo.multiply %71, %72 : tensor<1x1x1x64xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %75 = stablehlo.multiply %68, %74 : tensor<256x56x56x64xf32>
    %76 = stablehlo.reshape %arg9 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %78 = stablehlo.add %75, %77 : tensor<256x56x56x64xf32>
    %79 = stablehlo.convert %78 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %80 = call @relu_13(%79) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %81 = stablehlo.convert %arg14 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %82 = stablehlo.convolution(%80, %81) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %83 = stablehlo.broadcast_in_dim %arg132, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %84 = stablehlo.broadcast_in_dim %arg133, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %85 = stablehlo.convert %82 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %86 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %87 = stablehlo.subtract %85, %86 : tensor<256x56x56x64xf32>
    %88 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %89 = stablehlo.add %84, %88 : tensor<1x1x1x64xf32>
    %90 = stablehlo.rsqrt %89 : tensor<1x1x1x64xf32>
    %91 = stablehlo.reshape %arg12 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %92 = stablehlo.multiply %90, %91 : tensor<1x1x1x64xf32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %94 = stablehlo.multiply %87, %93 : tensor<256x56x56x64xf32>
    %95 = stablehlo.reshape %arg11 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %97 = stablehlo.add %94, %96 : tensor<256x56x56x64xf32>
    %98 = stablehlo.convert %97 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %99 = stablehlo.add %61, %98 : tensor<256x56x56x64xbf16>
    %100 = call @relu_13(%99) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %101 = stablehlo.convert %arg19 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xbf16>
    %102 = stablehlo.convolution(%100, %101) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x128xbf16>) -> tensor<256x28x28x128xbf16>
    %103 = stablehlo.broadcast_in_dim %arg134, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %104 = stablehlo.broadcast_in_dim %arg135, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %105 = stablehlo.convert %102 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %106 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %107 = stablehlo.subtract %105, %106 : tensor<256x28x28x128xf32>
    %108 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %109 = stablehlo.add %104, %108 : tensor<1x1x1x128xf32>
    %110 = stablehlo.rsqrt %109 : tensor<1x1x1x128xf32>
    %111 = stablehlo.reshape %arg16 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %112 = stablehlo.multiply %110, %111 : tensor<1x1x1x128xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %114 = stablehlo.multiply %107, %113 : tensor<256x28x28x128xf32>
    %115 = stablehlo.reshape %arg15 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %117 = stablehlo.add %114, %116 : tensor<256x28x28x128xf32>
    %118 = stablehlo.convert %117 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %119 = call @relu_32(%118) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %120 = stablehlo.convert %arg20 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %121 = stablehlo.convolution(%119, %120) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %122 = stablehlo.broadcast_in_dim %arg136, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %123 = stablehlo.broadcast_in_dim %arg137, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %124 = stablehlo.convert %121 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %125 = stablehlo.broadcast_in_dim %122, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %126 = stablehlo.subtract %124, %125 : tensor<256x28x28x128xf32>
    %127 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %128 = stablehlo.add %123, %127 : tensor<1x1x1x128xf32>
    %129 = stablehlo.rsqrt %128 : tensor<1x1x1x128xf32>
    %130 = stablehlo.reshape %arg18 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %131 = stablehlo.multiply %129, %130 : tensor<1x1x1x128xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %133 = stablehlo.multiply %126, %132 : tensor<256x28x28x128xf32>
    %134 = stablehlo.reshape %arg17 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %135 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %136 = stablehlo.add %133, %135 : tensor<256x28x28x128xf32>
    %137 = stablehlo.convert %136 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %138 = stablehlo.convert %arg21 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xbf16>
    %139 = stablehlo.convolution(%100, %138) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x128xbf16>) -> tensor<256x28x28x128xbf16>
    %140 = stablehlo.broadcast_in_dim %arg138, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %141 = stablehlo.broadcast_in_dim %arg139, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %142 = stablehlo.convert %139 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %143 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %144 = stablehlo.subtract %142, %143 : tensor<256x28x28x128xf32>
    %145 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %146 = stablehlo.add %141, %145 : tensor<1x1x1x128xf32>
    %147 = stablehlo.rsqrt %146 : tensor<1x1x1x128xf32>
    %148 = stablehlo.reshape %arg23 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %149 = stablehlo.multiply %147, %148 : tensor<1x1x1x128xf32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %151 = stablehlo.multiply %144, %150 : tensor<256x28x28x128xf32>
    %152 = stablehlo.reshape %arg22 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %154 = stablehlo.add %151, %153 : tensor<256x28x28x128xf32>
    %155 = stablehlo.convert %154 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %156 = stablehlo.add %155, %137 : tensor<256x28x28x128xbf16>
    %157 = call @relu_32(%156) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %158 = stablehlo.convert %arg28 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %159 = stablehlo.convolution(%157, %158) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %160 = stablehlo.broadcast_in_dim %arg140, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %161 = stablehlo.broadcast_in_dim %arg141, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %162 = stablehlo.convert %159 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %163 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %164 = stablehlo.subtract %162, %163 : tensor<256x28x28x128xf32>
    %165 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %166 = stablehlo.add %161, %165 : tensor<1x1x1x128xf32>
    %167 = stablehlo.rsqrt %166 : tensor<1x1x1x128xf32>
    %168 = stablehlo.reshape %arg25 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %169 = stablehlo.multiply %167, %168 : tensor<1x1x1x128xf32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %171 = stablehlo.multiply %164, %170 : tensor<256x28x28x128xf32>
    %172 = stablehlo.reshape %arg24 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %173 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %174 = stablehlo.add %171, %173 : tensor<256x28x28x128xf32>
    %175 = stablehlo.convert %174 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %176 = call @relu_32(%175) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %177 = stablehlo.convert %arg29 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %178 = stablehlo.convolution(%176, %177) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %179 = stablehlo.broadcast_in_dim %arg142, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %180 = stablehlo.broadcast_in_dim %arg143, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %181 = stablehlo.convert %178 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %182 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %183 = stablehlo.subtract %181, %182 : tensor<256x28x28x128xf32>
    %184 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %185 = stablehlo.add %180, %184 : tensor<1x1x1x128xf32>
    %186 = stablehlo.rsqrt %185 : tensor<1x1x1x128xf32>
    %187 = stablehlo.reshape %arg27 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %188 = stablehlo.multiply %186, %187 : tensor<1x1x1x128xf32>
    %189 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %190 = stablehlo.multiply %183, %189 : tensor<256x28x28x128xf32>
    %191 = stablehlo.reshape %arg26 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %193 = stablehlo.add %190, %192 : tensor<256x28x28x128xf32>
    %194 = stablehlo.convert %193 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %195 = stablehlo.add %157, %194 : tensor<256x28x28x128xbf16>
    %196 = call @relu_32(%195) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %197 = stablehlo.convert %arg34 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xbf16>
    %198 = stablehlo.convolution(%196, %197) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x256xbf16>) -> tensor<256x14x14x256xbf16>
    %199 = stablehlo.broadcast_in_dim %arg144, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %200 = stablehlo.broadcast_in_dim %arg145, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %201 = stablehlo.convert %198 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %202 = stablehlo.broadcast_in_dim %199, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %203 = stablehlo.subtract %201, %202 : tensor<256x14x14x256xf32>
    %204 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %205 = stablehlo.add %200, %204 : tensor<1x1x1x256xf32>
    %206 = stablehlo.rsqrt %205 : tensor<1x1x1x256xf32>
    %207 = stablehlo.reshape %arg31 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %208 = stablehlo.multiply %206, %207 : tensor<1x1x1x256xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %210 = stablehlo.multiply %203, %209 : tensor<256x14x14x256xf32>
    %211 = stablehlo.reshape %arg30 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %212 = stablehlo.broadcast_in_dim %211, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %213 = stablehlo.add %210, %212 : tensor<256x14x14x256xf32>
    %214 = stablehlo.convert %213 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %215 = call @relu_55(%214) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %216 = stablehlo.convert %arg35 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %217 = stablehlo.convolution(%215, %216) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %218 = stablehlo.broadcast_in_dim %arg146, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %219 = stablehlo.broadcast_in_dim %arg147, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %220 = stablehlo.convert %217 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %221 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %222 = stablehlo.subtract %220, %221 : tensor<256x14x14x256xf32>
    %223 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %224 = stablehlo.add %219, %223 : tensor<1x1x1x256xf32>
    %225 = stablehlo.rsqrt %224 : tensor<1x1x1x256xf32>
    %226 = stablehlo.reshape %arg33 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %227 = stablehlo.multiply %225, %226 : tensor<1x1x1x256xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %229 = stablehlo.multiply %222, %228 : tensor<256x14x14x256xf32>
    %230 = stablehlo.reshape %arg32 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %231 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %232 = stablehlo.add %229, %231 : tensor<256x14x14x256xf32>
    %233 = stablehlo.convert %232 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %234 = stablehlo.convert %arg36 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xbf16>
    %235 = stablehlo.convolution(%196, %234) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x256xbf16>) -> tensor<256x14x14x256xbf16>
    %236 = stablehlo.broadcast_in_dim %arg148, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %237 = stablehlo.broadcast_in_dim %arg149, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %238 = stablehlo.convert %235 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %239 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %240 = stablehlo.subtract %238, %239 : tensor<256x14x14x256xf32>
    %241 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %242 = stablehlo.add %237, %241 : tensor<1x1x1x256xf32>
    %243 = stablehlo.rsqrt %242 : tensor<1x1x1x256xf32>
    %244 = stablehlo.reshape %arg38 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %245 = stablehlo.multiply %243, %244 : tensor<1x1x1x256xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %247 = stablehlo.multiply %240, %246 : tensor<256x14x14x256xf32>
    %248 = stablehlo.reshape %arg37 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %249 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %250 = stablehlo.add %247, %249 : tensor<256x14x14x256xf32>
    %251 = stablehlo.convert %250 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %252 = stablehlo.add %251, %233 : tensor<256x14x14x256xbf16>
    %253 = call @relu_55(%252) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %254 = stablehlo.convert %arg43 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %255 = stablehlo.convolution(%253, %254) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %256 = stablehlo.broadcast_in_dim %arg150, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %257 = stablehlo.broadcast_in_dim %arg151, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %258 = stablehlo.convert %255 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %259 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %260 = stablehlo.subtract %258, %259 : tensor<256x14x14x256xf32>
    %261 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %262 = stablehlo.add %257, %261 : tensor<1x1x1x256xf32>
    %263 = stablehlo.rsqrt %262 : tensor<1x1x1x256xf32>
    %264 = stablehlo.reshape %arg40 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %265 = stablehlo.multiply %263, %264 : tensor<1x1x1x256xf32>
    %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %267 = stablehlo.multiply %260, %266 : tensor<256x14x14x256xf32>
    %268 = stablehlo.reshape %arg39 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %270 = stablehlo.add %267, %269 : tensor<256x14x14x256xf32>
    %271 = stablehlo.convert %270 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %272 = call @relu_55(%271) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %273 = stablehlo.convert %arg44 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %274 = stablehlo.convolution(%272, %273) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %275 = stablehlo.broadcast_in_dim %arg152, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %276 = stablehlo.broadcast_in_dim %arg153, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %277 = stablehlo.convert %274 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %278 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %279 = stablehlo.subtract %277, %278 : tensor<256x14x14x256xf32>
    %280 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %281 = stablehlo.add %276, %280 : tensor<1x1x1x256xf32>
    %282 = stablehlo.rsqrt %281 : tensor<1x1x1x256xf32>
    %283 = stablehlo.reshape %arg42 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %284 = stablehlo.multiply %282, %283 : tensor<1x1x1x256xf32>
    %285 = stablehlo.broadcast_in_dim %284, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %286 = stablehlo.multiply %279, %285 : tensor<256x14x14x256xf32>
    %287 = stablehlo.reshape %arg41 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %288 = stablehlo.broadcast_in_dim %287, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %289 = stablehlo.add %286, %288 : tensor<256x14x14x256xf32>
    %290 = stablehlo.convert %289 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %291 = stablehlo.add %253, %290 : tensor<256x14x14x256xbf16>
    %292 = call @relu_55(%291) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %293 = stablehlo.convert %arg49 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xbf16>
    %294 = stablehlo.convolution(%292, %293) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x512xbf16>) -> tensor<256x7x7x512xbf16>
    %295 = stablehlo.broadcast_in_dim %arg154, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %296 = stablehlo.broadcast_in_dim %arg155, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %297 = stablehlo.convert %294 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %298 = stablehlo.broadcast_in_dim %295, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %299 = stablehlo.subtract %297, %298 : tensor<256x7x7x512xf32>
    %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %301 = stablehlo.add %296, %300 : tensor<1x1x1x512xf32>
    %302 = stablehlo.rsqrt %301 : tensor<1x1x1x512xf32>
    %303 = stablehlo.reshape %arg46 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %304 = stablehlo.multiply %302, %303 : tensor<1x1x1x512xf32>
    %305 = stablehlo.broadcast_in_dim %304, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %306 = stablehlo.multiply %299, %305 : tensor<256x7x7x512xf32>
    %307 = stablehlo.reshape %arg45 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %309 = stablehlo.add %306, %308 : tensor<256x7x7x512xf32>
    %310 = stablehlo.convert %309 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %311 = call @relu_78(%310) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %312 = stablehlo.convert %arg50 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %313 = stablehlo.convolution(%311, %312) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %314 = stablehlo.broadcast_in_dim %arg156, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %315 = stablehlo.broadcast_in_dim %arg157, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %316 = stablehlo.convert %313 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %317 = stablehlo.broadcast_in_dim %314, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %318 = stablehlo.subtract %316, %317 : tensor<256x7x7x512xf32>
    %319 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %320 = stablehlo.add %315, %319 : tensor<1x1x1x512xf32>
    %321 = stablehlo.rsqrt %320 : tensor<1x1x1x512xf32>
    %322 = stablehlo.reshape %arg48 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %323 = stablehlo.multiply %321, %322 : tensor<1x1x1x512xf32>
    %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %325 = stablehlo.multiply %318, %324 : tensor<256x7x7x512xf32>
    %326 = stablehlo.reshape %arg47 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %327 = stablehlo.broadcast_in_dim %326, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %328 = stablehlo.add %325, %327 : tensor<256x7x7x512xf32>
    %329 = stablehlo.convert %328 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %330 = stablehlo.convert %arg51 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xbf16>
    %331 = stablehlo.convolution(%292, %330) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x512xbf16>) -> tensor<256x7x7x512xbf16>
    %332 = stablehlo.broadcast_in_dim %arg158, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %333 = stablehlo.broadcast_in_dim %arg159, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %334 = stablehlo.convert %331 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %335 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %336 = stablehlo.subtract %334, %335 : tensor<256x7x7x512xf32>
    %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %338 = stablehlo.add %333, %337 : tensor<1x1x1x512xf32>
    %339 = stablehlo.rsqrt %338 : tensor<1x1x1x512xf32>
    %340 = stablehlo.reshape %arg53 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %341 = stablehlo.multiply %339, %340 : tensor<1x1x1x512xf32>
    %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %343 = stablehlo.multiply %336, %342 : tensor<256x7x7x512xf32>
    %344 = stablehlo.reshape %arg52 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %346 = stablehlo.add %343, %345 : tensor<256x7x7x512xf32>
    %347 = stablehlo.convert %346 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %348 = stablehlo.add %347, %329 : tensor<256x7x7x512xbf16>
    %349 = call @relu_78(%348) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %350 = stablehlo.convert %arg58 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %351 = stablehlo.convolution(%349, %350) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %352 = stablehlo.broadcast_in_dim %arg160, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %353 = stablehlo.broadcast_in_dim %arg161, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %354 = stablehlo.convert %351 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %355 = stablehlo.broadcast_in_dim %352, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %356 = stablehlo.subtract %354, %355 : tensor<256x7x7x512xf32>
    %357 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %358 = stablehlo.add %353, %357 : tensor<1x1x1x512xf32>
    %359 = stablehlo.rsqrt %358 : tensor<1x1x1x512xf32>
    %360 = stablehlo.reshape %arg55 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %361 = stablehlo.multiply %359, %360 : tensor<1x1x1x512xf32>
    %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %363 = stablehlo.multiply %356, %362 : tensor<256x7x7x512xf32>
    %364 = stablehlo.reshape %arg54 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %366 = stablehlo.add %363, %365 : tensor<256x7x7x512xf32>
    %367 = stablehlo.convert %366 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %368 = call @relu_78(%367) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %369 = stablehlo.convert %arg59 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %370 = stablehlo.convolution(%368, %369) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %371 = stablehlo.broadcast_in_dim %arg162, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %372 = stablehlo.broadcast_in_dim %arg163, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %373 = stablehlo.convert %370 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %374 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %375 = stablehlo.subtract %373, %374 : tensor<256x7x7x512xf32>
    %376 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %377 = stablehlo.add %372, %376 : tensor<1x1x1x512xf32>
    %378 = stablehlo.rsqrt %377 : tensor<1x1x1x512xf32>
    %379 = stablehlo.reshape %arg57 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %380 = stablehlo.multiply %378, %379 : tensor<1x1x1x512xf32>
    %381 = stablehlo.broadcast_in_dim %380, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %382 = stablehlo.multiply %375, %381 : tensor<256x7x7x512xf32>
    %383 = stablehlo.reshape %arg56 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %384 = stablehlo.broadcast_in_dim %383, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %385 = stablehlo.add %382, %384 : tensor<256x7x7x512xf32>
    %386 = stablehlo.convert %385 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %387 = stablehlo.add %349, %386 : tensor<256x7x7x512xbf16>
    %388 = call @relu_78(%387) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %389 = stablehlo.convert %388 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %390 = stablehlo.reduce(%389 init: %cst_1) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<256x512xf32>
    %cst_2 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
    %391 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
    %392 = stablehlo.divide %390, %391 : tensor<256x512xf32>
    %393 = stablehlo.convert %392 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %394 = stablehlo.convert %arg2 : (tensor<512x1000xf32>) -> tensor<512x1000xbf16>
    %395 = stablehlo.convert %arg1 : (tensor<1000xf32>) -> tensor<1000xbf16>
    %396 = stablehlo.dot_general %393, %394, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xbf16>, tensor<512x1000xbf16>) -> tensor<256x1000xbf16>
    %397 = stablehlo.reshape %395 : (tensor<1000xbf16>) -> tensor<1x1000xbf16>
    %398 = stablehlo.broadcast_in_dim %397, dims = [0, 1] : (tensor<1x1000xbf16>) -> tensor<256x1000xbf16>
    %399 = stablehlo.add %396, %398 : tensor<256x1000xbf16>
    %400 = stablehlo.broadcast_in_dim %399, dims = [1, 2] : (tensor<256x1000xbf16>) -> tensor<1x256x1000xbf16>
    return %400 : tensor<1x256x1000xbf16>
  }
  func.func private @relu(%arg0: tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x112x112x64xbf16>
    return %1 : tensor<256x112x112x64xbf16>
  }
  func.func private @relu_13(%arg0: tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x64xbf16>
    return %1 : tensor<256x56x56x64xbf16>
  }
  func.func private @relu_32(%arg0: tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xbf16>
    return %1 : tensor<256x28x28x128xbf16>
  }
  func.func private @relu_55(%arg0: tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xbf16>
    return %1 : tensor<256x14x14x256xbf16>
  }
  func.func private @relu_78(%arg0: tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xbf16>
    return %1 : tensor<256x7x7x512xbf16>
  }
}
