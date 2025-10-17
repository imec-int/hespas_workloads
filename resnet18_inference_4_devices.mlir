module @jit_call_wrapped attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=4]>
  func.func public @main(%arg0: tensor<1000xf32>, %arg1: tensor<512x1000xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<3x3x64x64xf32>, %arg7: tensor<3x3x64x64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<3x3x64x64xf32>, %arg13: tensor<3x3x64x64xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<3x3x64x128xf32>, %arg19: tensor<3x3x128x128xf32>, %arg20: tensor<1x1x64x128xf32>, %arg21: tensor<128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<3x3x128x128xf32>, %arg28: tensor<3x3x128x128xf32>, %arg29: tensor<256xf32>, %arg30: tensor<256xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<3x3x128x256xf32>, %arg34: tensor<3x3x256x256xf32>, %arg35: tensor<1x1x128x256xf32>, %arg36: tensor<256xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<3x3x256x256xf32>, %arg43: tensor<3x3x256x256xf32>, %arg44: tensor<512xf32>, %arg45: tensor<512xf32>, %arg46: tensor<512xf32>, %arg47: tensor<512xf32>, %arg48: tensor<3x3x256x512xf32>, %arg49: tensor<3x3x512x512xf32>, %arg50: tensor<1x1x256x512xf32>, %arg51: tensor<512xf32>, %arg52: tensor<512xf32>, %arg53: tensor<512xf32>, %arg54: tensor<512xf32>, %arg55: tensor<512xf32>, %arg56: tensor<512xf32>, %arg57: tensor<3x3x512x512xf32>, %arg58: tensor<3x3x512x512xf32>, %arg59: tensor<64xf32>, %arg60: tensor<64xf32>, %arg61: tensor<7x7x3x64xf32>, %arg62: tensor<64xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64xf32>, %arg65: tensor<64xf32>, %arg66: tensor<64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<64xf32>, %arg69: tensor<64xf32>, %arg70: tensor<128xf32>, %arg71: tensor<128xf32>, %arg72: tensor<128xf32>, %arg73: tensor<128xf32>, %arg74: tensor<128xf32>, %arg75: tensor<128xf32>, %arg76: tensor<128xf32>, %arg77: tensor<128xf32>, %arg78: tensor<128xf32>, %arg79: tensor<128xf32>, %arg80: tensor<256xf32>, %arg81: tensor<256xf32>, %arg82: tensor<256xf32>, %arg83: tensor<256xf32>, %arg84: tensor<256xf32>, %arg85: tensor<256xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<512xf32>, %arg91: tensor<512xf32>, %arg92: tensor<512xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512xf32>, %arg97: tensor<512xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<64xf32>, %arg101: tensor<64xf32>, %arg102: tensor<4x256x224x224x3xf16>, %arg103: tensor<4x256xi32>) -> (tensor<4xf32> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %0:2 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103) in_shardings=[<@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{"batch"}, {}, {}, {}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>] manual_axes={"batch"} (%arg104: tensor<1000xf32>, %arg105: tensor<512x1000xf32>, %arg106: tensor<64xf32>, %arg107: tensor<64xf32>, %arg108: tensor<64xf32>, %arg109: tensor<64xf32>, %arg110: tensor<3x3x64x64xf32>, %arg111: tensor<3x3x64x64xf32>, %arg112: tensor<64xf32>, %arg113: tensor<64xf32>, %arg114: tensor<64xf32>, %arg115: tensor<64xf32>, %arg116: tensor<3x3x64x64xf32>, %arg117: tensor<3x3x64x64xf32>, %arg118: tensor<128xf32>, %arg119: tensor<128xf32>, %arg120: tensor<128xf32>, %arg121: tensor<128xf32>, %arg122: tensor<3x3x64x128xf32>, %arg123: tensor<3x3x128x128xf32>, %arg124: tensor<1x1x64x128xf32>, %arg125: tensor<128xf32>, %arg126: tensor<128xf32>, %arg127: tensor<128xf32>, %arg128: tensor<128xf32>, %arg129: tensor<128xf32>, %arg130: tensor<128xf32>, %arg131: tensor<3x3x128x128xf32>, %arg132: tensor<3x3x128x128xf32>, %arg133: tensor<256xf32>, %arg134: tensor<256xf32>, %arg135: tensor<256xf32>, %arg136: tensor<256xf32>, %arg137: tensor<3x3x128x256xf32>, %arg138: tensor<3x3x256x256xf32>, %arg139: tensor<1x1x128x256xf32>, %arg140: tensor<256xf32>, %arg141: tensor<256xf32>, %arg142: tensor<256xf32>, %arg143: tensor<256xf32>, %arg144: tensor<256xf32>, %arg145: tensor<256xf32>, %arg146: tensor<3x3x256x256xf32>, %arg147: tensor<3x3x256x256xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512xf32>, %arg150: tensor<512xf32>, %arg151: tensor<512xf32>, %arg152: tensor<3x3x256x512xf32>, %arg153: tensor<3x3x512x512xf32>, %arg154: tensor<1x1x256x512xf32>, %arg155: tensor<512xf32>, %arg156: tensor<512xf32>, %arg157: tensor<512xf32>, %arg158: tensor<512xf32>, %arg159: tensor<512xf32>, %arg160: tensor<512xf32>, %arg161: tensor<3x3x512x512xf32>, %arg162: tensor<3x3x512x512xf32>, %arg163: tensor<64xf32>, %arg164: tensor<64xf32>, %arg165: tensor<7x7x3x64xf32>, %arg166: tensor<64xf32>, %arg167: tensor<64xf32>, %arg168: tensor<64xf32>, %arg169: tensor<64xf32>, %arg170: tensor<64xf32>, %arg171: tensor<64xf32>, %arg172: tensor<64xf32>, %arg173: tensor<64xf32>, %arg174: tensor<128xf32>, %arg175: tensor<128xf32>, %arg176: tensor<128xf32>, %arg177: tensor<128xf32>, %arg178: tensor<128xf32>, %arg179: tensor<128xf32>, %arg180: tensor<128xf32>, %arg181: tensor<128xf32>, %arg182: tensor<128xf32>, %arg183: tensor<128xf32>, %arg184: tensor<256xf32>, %arg185: tensor<256xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<256xf32>, %arg190: tensor<256xf32>, %arg191: tensor<256xf32>, %arg192: tensor<256xf32>, %arg193: tensor<256xf32>, %arg194: tensor<512xf32>, %arg195: tensor<512xf32>, %arg196: tensor<512xf32>, %arg197: tensor<512xf32>, %arg198: tensor<512xf32>, %arg199: tensor<512xf32>, %arg200: tensor<512xf32>, %arg201: tensor<512xf32>, %arg202: tensor<512xf32>, %arg203: tensor<512xf32>, %arg204: tensor<64xf32>, %arg205: tensor<64xf32>, %arg206: tensor<1x256x224x224x3xf16>, %arg207: tensor<1x256xi32>) {
      %1 = stablehlo.reshape %arg206 : (tensor<1x256x224x224x3xf16>) -> tensor<256x224x224x3xf16>
      %2 = stablehlo.reshape %arg207 : (tensor<1x256xi32>) -> tensor<256xi32>
      %3 = stablehlo.convert %arg165 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf16>
      %4 = stablehlo.convolution(%1, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf16>, tensor<7x7x3x64xf16>) -> tensor<256x112x112x64xf16>
      %5 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %6 = stablehlo.broadcast_in_dim %arg205, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %7 = stablehlo.convert %4 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
      %8 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %9 = stablehlo.subtract %7, %8 : tensor<256x112x112x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %11 = stablehlo.add %6, %10 : tensor<1x1x1x64xf32>
      %12 = stablehlo.rsqrt %11 : tensor<1x1x1x64xf32>
      %13 = stablehlo.reshape %arg164 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %14 = stablehlo.multiply %12, %13 : tensor<1x1x1x64xf32>
      %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %16 = stablehlo.multiply %9, %15 : tensor<256x112x112x64xf32>
      %17 = stablehlo.reshape %arg163 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %19 = stablehlo.add %16, %18 : tensor<256x112x112x64xf32>
      %20 = stablehlo.convert %19 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %21 = func.call @relu(%20) : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16>
      %cst_0 = stablehlo.constant dense<0xFC00> : tensor<f16>
      %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f16>) -> tensor<f16>
      %23 = "stablehlo.reduce_window"(%21, %22) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg208: tensor<f16>, %arg209: tensor<f16>):
        %430 = stablehlo.maximum %arg208, %arg209 : tensor<f16>
        stablehlo.return %430 : tensor<f16>
      }) : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x56x56x64xf16>
      %24 = stablehlo.convert %arg110 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %25 = stablehlo.convolution(%23, %24) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %26 = stablehlo.broadcast_in_dim %arg166, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %27 = stablehlo.broadcast_in_dim %arg167, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %28 = stablehlo.convert %25 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %29 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %30 = stablehlo.subtract %28, %29 : tensor<256x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %31 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %32 = stablehlo.add %27, %31 : tensor<1x1x1x64xf32>
      %33 = stablehlo.rsqrt %32 : tensor<1x1x1x64xf32>
      %34 = stablehlo.reshape %arg107 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %35 = stablehlo.multiply %33, %34 : tensor<1x1x1x64xf32>
      %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %37 = stablehlo.multiply %30, %36 : tensor<256x56x56x64xf32>
      %38 = stablehlo.reshape %arg106 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %40 = stablehlo.add %37, %39 : tensor<256x56x56x64xf32>
      %41 = stablehlo.convert %40 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %42 = func.call @relu_13(%41) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %43 = stablehlo.convert %arg111 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %44 = stablehlo.convolution(%42, %43) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %45 = stablehlo.broadcast_in_dim %arg168, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %46 = stablehlo.broadcast_in_dim %arg169, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %47 = stablehlo.convert %44 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %48 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %49 = stablehlo.subtract %47, %48 : tensor<256x56x56x64xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %50 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %51 = stablehlo.add %46, %50 : tensor<1x1x1x64xf32>
      %52 = stablehlo.rsqrt %51 : tensor<1x1x1x64xf32>
      %53 = stablehlo.reshape %arg109 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %54 = stablehlo.multiply %52, %53 : tensor<1x1x1x64xf32>
      %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %56 = stablehlo.multiply %49, %55 : tensor<256x56x56x64xf32>
      %57 = stablehlo.reshape %arg108 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %59 = stablehlo.add %56, %58 : tensor<256x56x56x64xf32>
      %60 = stablehlo.convert %59 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %61 = stablehlo.add %23, %60 : tensor<256x56x56x64xf16>
      %62 = func.call @relu_13(%61) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %63 = stablehlo.convert %arg116 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %64 = stablehlo.convolution(%62, %63) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %65 = stablehlo.broadcast_in_dim %arg170, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %66 = stablehlo.broadcast_in_dim %arg171, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %67 = stablehlo.convert %64 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %68 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %69 = stablehlo.subtract %67, %68 : tensor<256x56x56x64xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %70 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %71 = stablehlo.add %66, %70 : tensor<1x1x1x64xf32>
      %72 = stablehlo.rsqrt %71 : tensor<1x1x1x64xf32>
      %73 = stablehlo.reshape %arg113 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %74 = stablehlo.multiply %72, %73 : tensor<1x1x1x64xf32>
      %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %76 = stablehlo.multiply %69, %75 : tensor<256x56x56x64xf32>
      %77 = stablehlo.reshape %arg112 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %79 = stablehlo.add %76, %78 : tensor<256x56x56x64xf32>
      %80 = stablehlo.convert %79 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %81 = func.call @relu_13(%80) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %82 = stablehlo.convert %arg117 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %83 = stablehlo.convolution(%81, %82) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %84 = stablehlo.broadcast_in_dim %arg172, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %85 = stablehlo.broadcast_in_dim %arg173, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %86 = stablehlo.convert %83 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %87 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %88 = stablehlo.subtract %86, %87 : tensor<256x56x56x64xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %89 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %90 = stablehlo.add %85, %89 : tensor<1x1x1x64xf32>
      %91 = stablehlo.rsqrt %90 : tensor<1x1x1x64xf32>
      %92 = stablehlo.reshape %arg115 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %93 = stablehlo.multiply %91, %92 : tensor<1x1x1x64xf32>
      %94 = stablehlo.broadcast_in_dim %93, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %95 = stablehlo.multiply %88, %94 : tensor<256x56x56x64xf32>
      %96 = stablehlo.reshape %arg114 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %98 = stablehlo.add %95, %97 : tensor<256x56x56x64xf32>
      %99 = stablehlo.convert %98 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %100 = stablehlo.add %62, %99 : tensor<256x56x56x64xf16>
      %101 = func.call @relu_13(%100) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %102 = stablehlo.convert %arg122 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf16>
      %103 = stablehlo.convolution(%101, %102) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x128xf16>) -> tensor<256x28x28x128xf16>
      %104 = stablehlo.broadcast_in_dim %arg174, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %105 = stablehlo.broadcast_in_dim %arg175, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %106 = stablehlo.convert %103 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %107 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %108 = stablehlo.subtract %106, %107 : tensor<256x28x28x128xf32>
      %cst_5 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %109 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %110 = stablehlo.add %105, %109 : tensor<1x1x1x128xf32>
      %111 = stablehlo.rsqrt %110 : tensor<1x1x1x128xf32>
      %112 = stablehlo.reshape %arg119 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<1x1x1x128xf32>
      %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %115 = stablehlo.multiply %108, %114 : tensor<256x28x28x128xf32>
      %116 = stablehlo.reshape %arg118 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %117 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %118 = stablehlo.add %115, %117 : tensor<256x28x28x128xf32>
      %119 = stablehlo.convert %118 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %120 = func.call @relu_32(%119) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %121 = stablehlo.convert %arg123 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %122 = stablehlo.convolution(%120, %121) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %123 = stablehlo.broadcast_in_dim %arg176, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %124 = stablehlo.broadcast_in_dim %arg177, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %125 = stablehlo.convert %122 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %126 = stablehlo.broadcast_in_dim %123, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %127 = stablehlo.subtract %125, %126 : tensor<256x28x28x128xf32>
      %cst_6 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %128 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %129 = stablehlo.add %124, %128 : tensor<1x1x1x128xf32>
      %130 = stablehlo.rsqrt %129 : tensor<1x1x1x128xf32>
      %131 = stablehlo.reshape %arg121 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %132 = stablehlo.multiply %130, %131 : tensor<1x1x1x128xf32>
      %133 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %134 = stablehlo.multiply %127, %133 : tensor<256x28x28x128xf32>
      %135 = stablehlo.reshape %arg120 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %137 = stablehlo.add %134, %136 : tensor<256x28x28x128xf32>
      %138 = stablehlo.convert %137 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %139 = stablehlo.convert %arg124 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf16>
      %140 = stablehlo.convolution(%101, %139) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x128xf16>) -> tensor<256x28x28x128xf16>
      %141 = stablehlo.broadcast_in_dim %arg178, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %142 = stablehlo.broadcast_in_dim %arg179, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %143 = stablehlo.convert %140 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %144 = stablehlo.broadcast_in_dim %141, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %145 = stablehlo.subtract %143, %144 : tensor<256x28x28x128xf32>
      %cst_7 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %146 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %147 = stablehlo.add %142, %146 : tensor<1x1x1x128xf32>
      %148 = stablehlo.rsqrt %147 : tensor<1x1x1x128xf32>
      %149 = stablehlo.reshape %arg126 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %150 = stablehlo.multiply %148, %149 : tensor<1x1x1x128xf32>
      %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %152 = stablehlo.multiply %145, %151 : tensor<256x28x28x128xf32>
      %153 = stablehlo.reshape %arg125 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %154 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %155 = stablehlo.add %152, %154 : tensor<256x28x28x128xf32>
      %156 = stablehlo.convert %155 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %157 = stablehlo.add %156, %138 : tensor<256x28x28x128xf16>
      %158 = func.call @relu_32(%157) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %159 = stablehlo.convert %arg131 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %160 = stablehlo.convolution(%158, %159) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %161 = stablehlo.broadcast_in_dim %arg180, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %162 = stablehlo.broadcast_in_dim %arg181, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %163 = stablehlo.convert %160 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %164 = stablehlo.broadcast_in_dim %161, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %165 = stablehlo.subtract %163, %164 : tensor<256x28x28x128xf32>
      %cst_8 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %166 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %167 = stablehlo.add %162, %166 : tensor<1x1x1x128xf32>
      %168 = stablehlo.rsqrt %167 : tensor<1x1x1x128xf32>
      %169 = stablehlo.reshape %arg128 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %170 = stablehlo.multiply %168, %169 : tensor<1x1x1x128xf32>
      %171 = stablehlo.broadcast_in_dim %170, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %172 = stablehlo.multiply %165, %171 : tensor<256x28x28x128xf32>
      %173 = stablehlo.reshape %arg127 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %174 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %175 = stablehlo.add %172, %174 : tensor<256x28x28x128xf32>
      %176 = stablehlo.convert %175 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %177 = func.call @relu_32(%176) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %178 = stablehlo.convert %arg132 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %179 = stablehlo.convolution(%177, %178) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %180 = stablehlo.broadcast_in_dim %arg182, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %181 = stablehlo.broadcast_in_dim %arg183, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %182 = stablehlo.convert %179 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %183 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %184 = stablehlo.subtract %182, %183 : tensor<256x28x28x128xf32>
      %cst_9 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %185 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %186 = stablehlo.add %181, %185 : tensor<1x1x1x128xf32>
      %187 = stablehlo.rsqrt %186 : tensor<1x1x1x128xf32>
      %188 = stablehlo.reshape %arg130 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %189 = stablehlo.multiply %187, %188 : tensor<1x1x1x128xf32>
      %190 = stablehlo.broadcast_in_dim %189, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %191 = stablehlo.multiply %184, %190 : tensor<256x28x28x128xf32>
      %192 = stablehlo.reshape %arg129 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %194 = stablehlo.add %191, %193 : tensor<256x28x28x128xf32>
      %195 = stablehlo.convert %194 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %196 = stablehlo.add %158, %195 : tensor<256x28x28x128xf16>
      %197 = func.call @relu_32(%196) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %198 = stablehlo.convert %arg137 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf16>
      %199 = stablehlo.convolution(%197, %198) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x256xf16>) -> tensor<256x14x14x256xf16>
      %200 = stablehlo.broadcast_in_dim %arg184, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %201 = stablehlo.broadcast_in_dim %arg185, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %202 = stablehlo.convert %199 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %203 = stablehlo.broadcast_in_dim %200, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %204 = stablehlo.subtract %202, %203 : tensor<256x14x14x256xf32>
      %cst_10 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %206 = stablehlo.add %201, %205 : tensor<1x1x1x256xf32>
      %207 = stablehlo.rsqrt %206 : tensor<1x1x1x256xf32>
      %208 = stablehlo.reshape %arg134 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %209 = stablehlo.multiply %207, %208 : tensor<1x1x1x256xf32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %211 = stablehlo.multiply %204, %210 : tensor<256x14x14x256xf32>
      %212 = stablehlo.reshape %arg133 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %214 = stablehlo.add %211, %213 : tensor<256x14x14x256xf32>
      %215 = stablehlo.convert %214 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %216 = func.call @relu_55(%215) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %217 = stablehlo.convert %arg138 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %218 = stablehlo.convolution(%216, %217) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %219 = stablehlo.broadcast_in_dim %arg186, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %220 = stablehlo.broadcast_in_dim %arg187, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %221 = stablehlo.convert %218 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %222 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %223 = stablehlo.subtract %221, %222 : tensor<256x14x14x256xf32>
      %cst_11 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %224 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %225 = stablehlo.add %220, %224 : tensor<1x1x1x256xf32>
      %226 = stablehlo.rsqrt %225 : tensor<1x1x1x256xf32>
      %227 = stablehlo.reshape %arg136 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %228 = stablehlo.multiply %226, %227 : tensor<1x1x1x256xf32>
      %229 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %230 = stablehlo.multiply %223, %229 : tensor<256x14x14x256xf32>
      %231 = stablehlo.reshape %arg135 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %232 = stablehlo.broadcast_in_dim %231, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %233 = stablehlo.add %230, %232 : tensor<256x14x14x256xf32>
      %234 = stablehlo.convert %233 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %235 = stablehlo.convert %arg139 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf16>
      %236 = stablehlo.convolution(%197, %235) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x256xf16>) -> tensor<256x14x14x256xf16>
      %237 = stablehlo.broadcast_in_dim %arg188, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %238 = stablehlo.broadcast_in_dim %arg189, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %239 = stablehlo.convert %236 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %240 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %241 = stablehlo.subtract %239, %240 : tensor<256x14x14x256xf32>
      %cst_12 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %242 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %243 = stablehlo.add %238, %242 : tensor<1x1x1x256xf32>
      %244 = stablehlo.rsqrt %243 : tensor<1x1x1x256xf32>
      %245 = stablehlo.reshape %arg141 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %246 = stablehlo.multiply %244, %245 : tensor<1x1x1x256xf32>
      %247 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %248 = stablehlo.multiply %241, %247 : tensor<256x14x14x256xf32>
      %249 = stablehlo.reshape %arg140 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %251 = stablehlo.add %248, %250 : tensor<256x14x14x256xf32>
      %252 = stablehlo.convert %251 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %253 = stablehlo.add %252, %234 : tensor<256x14x14x256xf16>
      %254 = func.call @relu_55(%253) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %255 = stablehlo.convert %arg146 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %256 = stablehlo.convolution(%254, %255) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %257 = stablehlo.broadcast_in_dim %arg190, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %258 = stablehlo.broadcast_in_dim %arg191, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %259 = stablehlo.convert %256 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %260 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %261 = stablehlo.subtract %259, %260 : tensor<256x14x14x256xf32>
      %cst_13 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %262 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %263 = stablehlo.add %258, %262 : tensor<1x1x1x256xf32>
      %264 = stablehlo.rsqrt %263 : tensor<1x1x1x256xf32>
      %265 = stablehlo.reshape %arg143 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %266 = stablehlo.multiply %264, %265 : tensor<1x1x1x256xf32>
      %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %268 = stablehlo.multiply %261, %267 : tensor<256x14x14x256xf32>
      %269 = stablehlo.reshape %arg142 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %271 = stablehlo.add %268, %270 : tensor<256x14x14x256xf32>
      %272 = stablehlo.convert %271 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %273 = func.call @relu_55(%272) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %274 = stablehlo.convert %arg147 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %275 = stablehlo.convolution(%273, %274) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %276 = stablehlo.broadcast_in_dim %arg192, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %277 = stablehlo.broadcast_in_dim %arg193, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %278 = stablehlo.convert %275 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %279 = stablehlo.broadcast_in_dim %276, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %280 = stablehlo.subtract %278, %279 : tensor<256x14x14x256xf32>
      %cst_14 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %281 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %282 = stablehlo.add %277, %281 : tensor<1x1x1x256xf32>
      %283 = stablehlo.rsqrt %282 : tensor<1x1x1x256xf32>
      %284 = stablehlo.reshape %arg145 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %285 = stablehlo.multiply %283, %284 : tensor<1x1x1x256xf32>
      %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %287 = stablehlo.multiply %280, %286 : tensor<256x14x14x256xf32>
      %288 = stablehlo.reshape %arg144 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %289 = stablehlo.broadcast_in_dim %288, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %290 = stablehlo.add %287, %289 : tensor<256x14x14x256xf32>
      %291 = stablehlo.convert %290 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %292 = stablehlo.add %254, %291 : tensor<256x14x14x256xf16>
      %293 = func.call @relu_55(%292) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %294 = stablehlo.convert %arg152 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf16>
      %295 = stablehlo.convolution(%293, %294) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x512xf16>) -> tensor<256x7x7x512xf16>
      %296 = stablehlo.broadcast_in_dim %arg194, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %297 = stablehlo.broadcast_in_dim %arg195, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %298 = stablehlo.convert %295 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %299 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %300 = stablehlo.subtract %298, %299 : tensor<256x7x7x512xf32>
      %cst_15 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %302 = stablehlo.add %297, %301 : tensor<1x1x1x512xf32>
      %303 = stablehlo.rsqrt %302 : tensor<1x1x1x512xf32>
      %304 = stablehlo.reshape %arg149 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %305 = stablehlo.multiply %303, %304 : tensor<1x1x1x512xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %307 = stablehlo.multiply %300, %306 : tensor<256x7x7x512xf32>
      %308 = stablehlo.reshape %arg148 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %310 = stablehlo.add %307, %309 : tensor<256x7x7x512xf32>
      %311 = stablehlo.convert %310 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %312 = func.call @relu_78(%311) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %313 = stablehlo.convert %arg153 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %314 = stablehlo.convolution(%312, %313) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %315 = stablehlo.broadcast_in_dim %arg196, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %316 = stablehlo.broadcast_in_dim %arg197, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %317 = stablehlo.convert %314 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %318 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %319 = stablehlo.subtract %317, %318 : tensor<256x7x7x512xf32>
      %cst_16 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %320 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %321 = stablehlo.add %316, %320 : tensor<1x1x1x512xf32>
      %322 = stablehlo.rsqrt %321 : tensor<1x1x1x512xf32>
      %323 = stablehlo.reshape %arg151 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<1x1x1x512xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %326 = stablehlo.multiply %319, %325 : tensor<256x7x7x512xf32>
      %327 = stablehlo.reshape %arg150 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %329 = stablehlo.add %326, %328 : tensor<256x7x7x512xf32>
      %330 = stablehlo.convert %329 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %331 = stablehlo.convert %arg154 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf16>
      %332 = stablehlo.convolution(%293, %331) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x512xf16>) -> tensor<256x7x7x512xf16>
      %333 = stablehlo.broadcast_in_dim %arg198, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %334 = stablehlo.broadcast_in_dim %arg199, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %335 = stablehlo.convert %332 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %336 = stablehlo.broadcast_in_dim %333, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %337 = stablehlo.subtract %335, %336 : tensor<256x7x7x512xf32>
      %cst_17 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %339 = stablehlo.add %334, %338 : tensor<1x1x1x512xf32>
      %340 = stablehlo.rsqrt %339 : tensor<1x1x1x512xf32>
      %341 = stablehlo.reshape %arg156 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<1x1x1x512xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %344 = stablehlo.multiply %337, %343 : tensor<256x7x7x512xf32>
      %345 = stablehlo.reshape %arg155 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %347 = stablehlo.add %344, %346 : tensor<256x7x7x512xf32>
      %348 = stablehlo.convert %347 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %349 = stablehlo.add %348, %330 : tensor<256x7x7x512xf16>
      %350 = func.call @relu_78(%349) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %351 = stablehlo.convert %arg161 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %352 = stablehlo.convolution(%350, %351) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %353 = stablehlo.broadcast_in_dim %arg200, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %354 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %355 = stablehlo.convert %352 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %356 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %357 = stablehlo.subtract %355, %356 : tensor<256x7x7x512xf32>
      %cst_18 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %359 = stablehlo.add %354, %358 : tensor<1x1x1x512xf32>
      %360 = stablehlo.rsqrt %359 : tensor<1x1x1x512xf32>
      %361 = stablehlo.reshape %arg158 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %362 = stablehlo.multiply %360, %361 : tensor<1x1x1x512xf32>
      %363 = stablehlo.broadcast_in_dim %362, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %364 = stablehlo.multiply %357, %363 : tensor<256x7x7x512xf32>
      %365 = stablehlo.reshape %arg157 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %366 = stablehlo.broadcast_in_dim %365, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %367 = stablehlo.add %364, %366 : tensor<256x7x7x512xf32>
      %368 = stablehlo.convert %367 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %369 = func.call @relu_78(%368) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %370 = stablehlo.convert %arg162 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %371 = stablehlo.convolution(%369, %370) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %372 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %373 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %374 = stablehlo.convert %371 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %375 = stablehlo.broadcast_in_dim %372, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %376 = stablehlo.subtract %374, %375 : tensor<256x7x7x512xf32>
      %cst_19 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %378 = stablehlo.add %373, %377 : tensor<1x1x1x512xf32>
      %379 = stablehlo.rsqrt %378 : tensor<1x1x1x512xf32>
      %380 = stablehlo.reshape %arg160 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %381 = stablehlo.multiply %379, %380 : tensor<1x1x1x512xf32>
      %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %383 = stablehlo.multiply %376, %382 : tensor<256x7x7x512xf32>
      %384 = stablehlo.reshape %arg159 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %385 = stablehlo.broadcast_in_dim %384, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %386 = stablehlo.add %383, %385 : tensor<256x7x7x512xf32>
      %387 = stablehlo.convert %386 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %388 = stablehlo.add %350, %387 : tensor<256x7x7x512xf16>
      %389 = func.call @relu_78(%388) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %390 = stablehlo.convert %389 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %391 = stablehlo.reduce(%390 init: %cst_20) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<256x512xf32>
      %cst_21 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
      %393 = stablehlo.divide %391, %392 : tensor<256x512xf32>
      %394 = stablehlo.convert %393 : (tensor<256x512xf32>) -> tensor<256x512xf16>
      %395 = stablehlo.convert %arg105 : (tensor<512x1000xf32>) -> tensor<512x1000xf16>
      %396 = stablehlo.convert %arg104 : (tensor<1000xf32>) -> tensor<1000xf16>
      %397 = stablehlo.dot_general %394, %395, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf16>, tensor<512x1000xf16>) -> tensor<256x1000xf16>
      %398 = stablehlo.reshape %396 : (tensor<1000xf16>) -> tensor<1x1000xf16>
      %399 = stablehlo.broadcast_in_dim %398, dims = [0, 1] : (tensor<1x1000xf16>) -> tensor<256x1000xf16>
      %400 = stablehlo.add %397, %399 : tensor<256x1000xf16>
      %401 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
      %402 = stablehlo.iota dim = 0 : tensor<1000xi32>
      %403 = stablehlo.reshape %402 : (tensor<1000xi32>) -> tensor<1x1000xi32>
      %404 = stablehlo.broadcast_in_dim %401, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
      %405 = stablehlo.broadcast_in_dim %403, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
      %406 = stablehlo.compare  EQ, %404, %405,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
      %cst_22 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %409 = stablehlo.select %406, %407, %408 : tensor<256x1000xi1>, tensor<256x1000xf32>
      %410 = stablehlo.convert %409 : tensor<256x1000xf32>
      %411 = func.call @log_softmax(%400) : (tensor<256x1000xf16>) -> tensor<256x1000xf16>
      %412 = stablehlo.convert %411 : (tensor<256x1000xf16>) -> tensor<256x1000xf32>
      %413 = stablehlo.multiply %410, %412 : tensor<256x1000xf32>
      %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %414 = stablehlo.reduce(%413 init: %cst_24) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
      %415 = stablehlo.negate %414 : tensor<256xf32>
      %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %416 = stablehlo.reduce(%415 init: %cst_25) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_26 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %417 = stablehlo.divide %416, %cst_26 : tensor<f32>
      %418 = func.call @argmax(%400) : (tensor<256x1000xf16>) -> tensor<256xi32>
      %419 = stablehlo.compare  EQ, %418, %2,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
      %420 = stablehlo.convert %419 : (tensor<256xi1>) -> tensor<256xi32>
      %421 = stablehlo.convert %420 : (tensor<256xi32>) -> tensor<256xf32>
      %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %422 = stablehlo.reduce(%421 init: %cst_27) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_28 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %423 = stablehlo.divide %422, %cst_28 : tensor<f32>
      %424 = "stablehlo.all_reduce"(%423) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg208: tensor<f32>, %arg209: tensor<f32>):
        %430 = stablehlo.add %arg208, %arg209 : tensor<f32>
        stablehlo.return %430 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %425 = "stablehlo.all_reduce"(%417) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg208: tensor<f32>, %arg209: tensor<f32>):
        %430 = stablehlo.add %arg208, %arg209 : tensor<f32>
        stablehlo.return %430 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_29 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %426 = stablehlo.divide %424, %cst_29 : tensor<f32>
      %cst_30 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %427 = stablehlo.divide %425, %cst_30 : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %426, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %429 = stablehlo.broadcast_in_dim %427, dims = [] : (tensor<f32>) -> tensor<1xf32>
      sdy.return %428, %429 : tensor<1xf32>, tensor<1xf32>
    } : (tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x256x224x224x3xf16>, tensor<4x256xi32>) -> (tensor<4xf32>, tensor<4xf32>)
    return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }
  func.func private @relu(%arg0: tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x112x112x64xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x112x112x64xf16>
    return %1 : tensor<256x112x112x64xf16>
  }
  func.func private @relu_13(%arg0: tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x64xf16>
    return %1 : tensor<256x56x56x64xf16>
  }
  func.func private @relu_32(%arg0: tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xf16>
    return %1 : tensor<256x28x28x128xf16>
  }
  func.func private @relu_55(%arg0: tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xf16>
    return %1 : tensor<256x14x14x256xf16>
  }
  func.func private @relu_78(%arg0: tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xf16>
    return %1 : tensor<256x7x7x512xf16>
  }
  func.func private @log_softmax(%arg0: tensor<256x1000xf16>) -> tensor<256x1000xf16> {
    %cst = stablehlo.constant dense<0xFC00> : tensor<f16>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<256x1000xf16>, tensor<f16>) -> tensor<256xf16>
    %cst_0 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f16>) -> tensor<256xf16>
    %2 = stablehlo.maximum %1, %0 : tensor<256xf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xf16>) -> tensor<256x1xf16>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<256x1xf16>) -> tensor<256x1000xf16>
    %5 = stablehlo.subtract %arg0, %4 : tensor<256x1000xf16>
    %6 = stablehlo.exponential %5 : tensor<256x1000xf16>
    %7 = stablehlo.convert %6 : (tensor<256x1000xf16>) -> tensor<256x1000xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
    %10 = stablehlo.convert %9 : (tensor<256x1xf32>) -> tensor<256x1xf16>
    %11 = stablehlo.log %10 : tensor<256x1xf16>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<256x1xf16>) -> tensor<256x1000xf16>
    %13 = stablehlo.subtract %5, %12 : tensor<256x1000xf16>
    return %13 : tensor<256x1000xf16>
  }
  func.func private @argmax(%arg0: tensor<256x1000xf16>) -> tensor<256xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<256x1000xi32>
    %cst = stablehlo.constant dense<0xFC00> : tensor<f16>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<256x1000xf16>, tensor<256x1000xi32>, tensor<f16>, tensor<i32>) -> (tensor<256xf16>, tensor<256xi32>)
     reducer(%arg1: tensor<f16>, %arg3: tensor<f16>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f16>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<f16>, tensor<i32>
    }
    return %1#1 : tensor<256xi32>
  }
}
