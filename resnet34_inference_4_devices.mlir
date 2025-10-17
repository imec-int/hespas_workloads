module @jit_call_wrapped attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=4]>
  func.func public @main(%arg0: tensor<1000xf32>, %arg1: tensor<512x1000xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<3x3x64x64xf32>, %arg7: tensor<3x3x64x64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<3x3x64x64xf32>, %arg13: tensor<3x3x64x64xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<256xf32>, %arg17: tensor<256xf32>, %arg18: tensor<3x3x256x256xf32>, %arg19: tensor<3x3x256x256xf32>, %arg20: tensor<256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<3x3x256x256xf32>, %arg25: tensor<3x3x256x256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<256xf32>, %arg28: tensor<256xf32>, %arg29: tensor<256xf32>, %arg30: tensor<3x3x256x256xf32>, %arg31: tensor<3x3x256x256xf32>, %arg32: tensor<512xf32>, %arg33: tensor<512xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<3x3x256x512xf32>, %arg37: tensor<3x3x512x512xf32>, %arg38: tensor<1x1x256x512xf32>, %arg39: tensor<512xf32>, %arg40: tensor<512xf32>, %arg41: tensor<512xf32>, %arg42: tensor<512xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<3x3x512x512xf32>, %arg46: tensor<3x3x512x512xf32>, %arg47: tensor<512xf32>, %arg48: tensor<512xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<3x3x512x512xf32>, %arg52: tensor<3x3x512x512xf32>, %arg53: tensor<64xf32>, %arg54: tensor<64xf32>, %arg55: tensor<64xf32>, %arg56: tensor<64xf32>, %arg57: tensor<3x3x64x64xf32>, %arg58: tensor<3x3x64x64xf32>, %arg59: tensor<128xf32>, %arg60: tensor<128xf32>, %arg61: tensor<128xf32>, %arg62: tensor<128xf32>, %arg63: tensor<3x3x64x128xf32>, %arg64: tensor<3x3x128x128xf32>, %arg65: tensor<1x1x64x128xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<128xf32>, %arg70: tensor<128xf32>, %arg71: tensor<128xf32>, %arg72: tensor<3x3x128x128xf32>, %arg73: tensor<3x3x128x128xf32>, %arg74: tensor<128xf32>, %arg75: tensor<128xf32>, %arg76: tensor<128xf32>, %arg77: tensor<128xf32>, %arg78: tensor<3x3x128x128xf32>, %arg79: tensor<3x3x128x128xf32>, %arg80: tensor<128xf32>, %arg81: tensor<128xf32>, %arg82: tensor<128xf32>, %arg83: tensor<128xf32>, %arg84: tensor<3x3x128x128xf32>, %arg85: tensor<3x3x128x128xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<3x3x128x256xf32>, %arg91: tensor<3x3x256x256xf32>, %arg92: tensor<1x1x128x256xf32>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<3x3x256x256xf32>, %arg100: tensor<3x3x256x256xf32>, %arg101: tensor<256xf32>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256xf32>, %arg105: tensor<3x3x256x256xf32>, %arg106: tensor<3x3x256x256xf32>, %arg107: tensor<64xf32>, %arg108: tensor<64xf32>, %arg109: tensor<7x7x3x64xf32>, %arg110: tensor<64xf32>, %arg111: tensor<64xf32>, %arg112: tensor<64xf32>, %arg113: tensor<64xf32>, %arg114: tensor<64xf32>, %arg115: tensor<64xf32>, %arg116: tensor<64xf32>, %arg117: tensor<64xf32>, %arg118: tensor<256xf32>, %arg119: tensor<256xf32>, %arg120: tensor<256xf32>, %arg121: tensor<256xf32>, %arg122: tensor<256xf32>, %arg123: tensor<256xf32>, %arg124: tensor<256xf32>, %arg125: tensor<256xf32>, %arg126: tensor<256xf32>, %arg127: tensor<256xf32>, %arg128: tensor<256xf32>, %arg129: tensor<256xf32>, %arg130: tensor<512xf32>, %arg131: tensor<512xf32>, %arg132: tensor<512xf32>, %arg133: tensor<512xf32>, %arg134: tensor<512xf32>, %arg135: tensor<512xf32>, %arg136: tensor<512xf32>, %arg137: tensor<512xf32>, %arg138: tensor<512xf32>, %arg139: tensor<512xf32>, %arg140: tensor<512xf32>, %arg141: tensor<512xf32>, %arg142: tensor<512xf32>, %arg143: tensor<512xf32>, %arg144: tensor<64xf32>, %arg145: tensor<64xf32>, %arg146: tensor<64xf32>, %arg147: tensor<64xf32>, %arg148: tensor<128xf32>, %arg149: tensor<128xf32>, %arg150: tensor<128xf32>, %arg151: tensor<128xf32>, %arg152: tensor<128xf32>, %arg153: tensor<128xf32>, %arg154: tensor<128xf32>, %arg155: tensor<128xf32>, %arg156: tensor<128xf32>, %arg157: tensor<128xf32>, %arg158: tensor<128xf32>, %arg159: tensor<128xf32>, %arg160: tensor<128xf32>, %arg161: tensor<128xf32>, %arg162: tensor<128xf32>, %arg163: tensor<128xf32>, %arg164: tensor<128xf32>, %arg165: tensor<128xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<256xf32>, %arg179: tensor<256xf32>, %arg180: tensor<64xf32>, %arg181: tensor<64xf32>, %arg182: tensor<4x256x224x224x3xf16>, %arg183: tensor<4x256xi32>) -> (tensor<4xf32> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %0:2 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg181, %arg182, %arg183) in_shardings=[<@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{"batch"}, {}, {}, {}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>] manual_axes={"batch"} (%arg184: tensor<1000xf32>, %arg185: tensor<512x1000xf32>, %arg186: tensor<64xf32>, %arg187: tensor<64xf32>, %arg188: tensor<64xf32>, %arg189: tensor<64xf32>, %arg190: tensor<3x3x64x64xf32>, %arg191: tensor<3x3x64x64xf32>, %arg192: tensor<64xf32>, %arg193: tensor<64xf32>, %arg194: tensor<64xf32>, %arg195: tensor<64xf32>, %arg196: tensor<3x3x64x64xf32>, %arg197: tensor<3x3x64x64xf32>, %arg198: tensor<256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256xf32>, %arg201: tensor<256xf32>, %arg202: tensor<3x3x256x256xf32>, %arg203: tensor<3x3x256x256xf32>, %arg204: tensor<256xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<256xf32>, %arg208: tensor<3x3x256x256xf32>, %arg209: tensor<3x3x256x256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<3x3x256x256xf32>, %arg215: tensor<3x3x256x256xf32>, %arg216: tensor<512xf32>, %arg217: tensor<512xf32>, %arg218: tensor<512xf32>, %arg219: tensor<512xf32>, %arg220: tensor<3x3x256x512xf32>, %arg221: tensor<3x3x512x512xf32>, %arg222: tensor<1x1x256x512xf32>, %arg223: tensor<512xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<512xf32>, %arg228: tensor<512xf32>, %arg229: tensor<3x3x512x512xf32>, %arg230: tensor<3x3x512x512xf32>, %arg231: tensor<512xf32>, %arg232: tensor<512xf32>, %arg233: tensor<512xf32>, %arg234: tensor<512xf32>, %arg235: tensor<3x3x512x512xf32>, %arg236: tensor<3x3x512x512xf32>, %arg237: tensor<64xf32>, %arg238: tensor<64xf32>, %arg239: tensor<64xf32>, %arg240: tensor<64xf32>, %arg241: tensor<3x3x64x64xf32>, %arg242: tensor<3x3x64x64xf32>, %arg243: tensor<128xf32>, %arg244: tensor<128xf32>, %arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<3x3x64x128xf32>, %arg248: tensor<3x3x128x128xf32>, %arg249: tensor<1x1x64x128xf32>, %arg250: tensor<128xf32>, %arg251: tensor<128xf32>, %arg252: tensor<128xf32>, %arg253: tensor<128xf32>, %arg254: tensor<128xf32>, %arg255: tensor<128xf32>, %arg256: tensor<3x3x128x128xf32>, %arg257: tensor<3x3x128x128xf32>, %arg258: tensor<128xf32>, %arg259: tensor<128xf32>, %arg260: tensor<128xf32>, %arg261: tensor<128xf32>, %arg262: tensor<3x3x128x128xf32>, %arg263: tensor<3x3x128x128xf32>, %arg264: tensor<128xf32>, %arg265: tensor<128xf32>, %arg266: tensor<128xf32>, %arg267: tensor<128xf32>, %arg268: tensor<3x3x128x128xf32>, %arg269: tensor<3x3x128x128xf32>, %arg270: tensor<256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<3x3x128x256xf32>, %arg275: tensor<3x3x256x256xf32>, %arg276: tensor<1x1x128x256xf32>, %arg277: tensor<256xf32>, %arg278: tensor<256xf32>, %arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<3x3x256x256xf32>, %arg284: tensor<3x3x256x256xf32>, %arg285: tensor<256xf32>, %arg286: tensor<256xf32>, %arg287: tensor<256xf32>, %arg288: tensor<256xf32>, %arg289: tensor<3x3x256x256xf32>, %arg290: tensor<3x3x256x256xf32>, %arg291: tensor<64xf32>, %arg292: tensor<64xf32>, %arg293: tensor<7x7x3x64xf32>, %arg294: tensor<64xf32>, %arg295: tensor<64xf32>, %arg296: tensor<64xf32>, %arg297: tensor<64xf32>, %arg298: tensor<64xf32>, %arg299: tensor<64xf32>, %arg300: tensor<64xf32>, %arg301: tensor<64xf32>, %arg302: tensor<256xf32>, %arg303: tensor<256xf32>, %arg304: tensor<256xf32>, %arg305: tensor<256xf32>, %arg306: tensor<256xf32>, %arg307: tensor<256xf32>, %arg308: tensor<256xf32>, %arg309: tensor<256xf32>, %arg310: tensor<256xf32>, %arg311: tensor<256xf32>, %arg312: tensor<256xf32>, %arg313: tensor<256xf32>, %arg314: tensor<512xf32>, %arg315: tensor<512xf32>, %arg316: tensor<512xf32>, %arg317: tensor<512xf32>, %arg318: tensor<512xf32>, %arg319: tensor<512xf32>, %arg320: tensor<512xf32>, %arg321: tensor<512xf32>, %arg322: tensor<512xf32>, %arg323: tensor<512xf32>, %arg324: tensor<512xf32>, %arg325: tensor<512xf32>, %arg326: tensor<512xf32>, %arg327: tensor<512xf32>, %arg328: tensor<64xf32>, %arg329: tensor<64xf32>, %arg330: tensor<64xf32>, %arg331: tensor<64xf32>, %arg332: tensor<128xf32>, %arg333: tensor<128xf32>, %arg334: tensor<128xf32>, %arg335: tensor<128xf32>, %arg336: tensor<128xf32>, %arg337: tensor<128xf32>, %arg338: tensor<128xf32>, %arg339: tensor<128xf32>, %arg340: tensor<128xf32>, %arg341: tensor<128xf32>, %arg342: tensor<128xf32>, %arg343: tensor<128xf32>, %arg344: tensor<128xf32>, %arg345: tensor<128xf32>, %arg346: tensor<128xf32>, %arg347: tensor<128xf32>, %arg348: tensor<128xf32>, %arg349: tensor<128xf32>, %arg350: tensor<256xf32>, %arg351: tensor<256xf32>, %arg352: tensor<256xf32>, %arg353: tensor<256xf32>, %arg354: tensor<256xf32>, %arg355: tensor<256xf32>, %arg356: tensor<256xf32>, %arg357: tensor<256xf32>, %arg358: tensor<256xf32>, %arg359: tensor<256xf32>, %arg360: tensor<256xf32>, %arg361: tensor<256xf32>, %arg362: tensor<256xf32>, %arg363: tensor<256xf32>, %arg364: tensor<64xf32>, %arg365: tensor<64xf32>, %arg366: tensor<1x256x224x224x3xf16>, %arg367: tensor<1x256xi32>) {
      %1 = stablehlo.reshape %arg366 : (tensor<1x256x224x224x3xf16>) -> tensor<256x224x224x3xf16>
      %2 = stablehlo.reshape %arg367 : (tensor<1x256xi32>) -> tensor<256xi32>
      %3 = stablehlo.convert %arg293 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf16>
      %4 = stablehlo.convolution(%1, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf16>, tensor<7x7x3x64xf16>) -> tensor<256x112x112x64xf16>
      %5 = stablehlo.broadcast_in_dim %arg364, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %6 = stablehlo.broadcast_in_dim %arg365, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %7 = stablehlo.convert %4 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
      %8 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %9 = stablehlo.subtract %7, %8 : tensor<256x112x112x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %11 = stablehlo.add %6, %10 : tensor<1x1x1x64xf32>
      %12 = stablehlo.rsqrt %11 : tensor<1x1x1x64xf32>
      %13 = stablehlo.reshape %arg292 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %14 = stablehlo.multiply %12, %13 : tensor<1x1x1x64xf32>
      %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %16 = stablehlo.multiply %9, %15 : tensor<256x112x112x64xf32>
      %17 = stablehlo.reshape %arg291 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %19 = stablehlo.add %16, %18 : tensor<256x112x112x64xf32>
      %20 = stablehlo.convert %19 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %21 = func.call @relu(%20) : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16>
      %cst_0 = stablehlo.constant dense<0xFC00> : tensor<f16>
      %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f16>) -> tensor<f16>
      %23 = "stablehlo.reduce_window"(%21, %22) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg368: tensor<f16>, %arg369: tensor<f16>):
        %742 = stablehlo.maximum %arg368, %arg369 : tensor<f16>
        stablehlo.return %742 : tensor<f16>
      }) : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x56x56x64xf16>
      %24 = stablehlo.convert %arg190 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %25 = stablehlo.convolution(%23, %24) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %26 = stablehlo.broadcast_in_dim %arg294, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %27 = stablehlo.broadcast_in_dim %arg295, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %28 = stablehlo.convert %25 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %29 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %30 = stablehlo.subtract %28, %29 : tensor<256x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %31 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %32 = stablehlo.add %27, %31 : tensor<1x1x1x64xf32>
      %33 = stablehlo.rsqrt %32 : tensor<1x1x1x64xf32>
      %34 = stablehlo.reshape %arg187 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %35 = stablehlo.multiply %33, %34 : tensor<1x1x1x64xf32>
      %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %37 = stablehlo.multiply %30, %36 : tensor<256x56x56x64xf32>
      %38 = stablehlo.reshape %arg186 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %40 = stablehlo.add %37, %39 : tensor<256x56x56x64xf32>
      %41 = stablehlo.convert %40 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %42 = func.call @relu_13(%41) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %43 = stablehlo.convert %arg191 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %44 = stablehlo.convolution(%42, %43) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %45 = stablehlo.broadcast_in_dim %arg296, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %46 = stablehlo.broadcast_in_dim %arg297, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %47 = stablehlo.convert %44 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %48 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %49 = stablehlo.subtract %47, %48 : tensor<256x56x56x64xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %50 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %51 = stablehlo.add %46, %50 : tensor<1x1x1x64xf32>
      %52 = stablehlo.rsqrt %51 : tensor<1x1x1x64xf32>
      %53 = stablehlo.reshape %arg189 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %54 = stablehlo.multiply %52, %53 : tensor<1x1x1x64xf32>
      %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %56 = stablehlo.multiply %49, %55 : tensor<256x56x56x64xf32>
      %57 = stablehlo.reshape %arg188 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %59 = stablehlo.add %56, %58 : tensor<256x56x56x64xf32>
      %60 = stablehlo.convert %59 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %61 = stablehlo.add %23, %60 : tensor<256x56x56x64xf16>
      %62 = func.call @relu_13(%61) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %63 = stablehlo.convert %arg196 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %64 = stablehlo.convolution(%62, %63) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %65 = stablehlo.broadcast_in_dim %arg298, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %66 = stablehlo.broadcast_in_dim %arg299, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %67 = stablehlo.convert %64 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %68 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %69 = stablehlo.subtract %67, %68 : tensor<256x56x56x64xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %70 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %71 = stablehlo.add %66, %70 : tensor<1x1x1x64xf32>
      %72 = stablehlo.rsqrt %71 : tensor<1x1x1x64xf32>
      %73 = stablehlo.reshape %arg193 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %74 = stablehlo.multiply %72, %73 : tensor<1x1x1x64xf32>
      %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %76 = stablehlo.multiply %69, %75 : tensor<256x56x56x64xf32>
      %77 = stablehlo.reshape %arg192 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %79 = stablehlo.add %76, %78 : tensor<256x56x56x64xf32>
      %80 = stablehlo.convert %79 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %81 = func.call @relu_13(%80) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %82 = stablehlo.convert %arg197 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %83 = stablehlo.convolution(%81, %82) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %84 = stablehlo.broadcast_in_dim %arg300, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %85 = stablehlo.broadcast_in_dim %arg301, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %86 = stablehlo.convert %83 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %87 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %88 = stablehlo.subtract %86, %87 : tensor<256x56x56x64xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %89 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %90 = stablehlo.add %85, %89 : tensor<1x1x1x64xf32>
      %91 = stablehlo.rsqrt %90 : tensor<1x1x1x64xf32>
      %92 = stablehlo.reshape %arg195 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %93 = stablehlo.multiply %91, %92 : tensor<1x1x1x64xf32>
      %94 = stablehlo.broadcast_in_dim %93, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %95 = stablehlo.multiply %88, %94 : tensor<256x56x56x64xf32>
      %96 = stablehlo.reshape %arg194 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %98 = stablehlo.add %95, %97 : tensor<256x56x56x64xf32>
      %99 = stablehlo.convert %98 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %100 = stablehlo.add %62, %99 : tensor<256x56x56x64xf16>
      %101 = func.call @relu_13(%100) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %102 = stablehlo.convert %arg241 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %103 = stablehlo.convolution(%101, %102) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %104 = stablehlo.broadcast_in_dim %arg328, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %105 = stablehlo.broadcast_in_dim %arg329, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %106 = stablehlo.convert %103 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %107 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %108 = stablehlo.subtract %106, %107 : tensor<256x56x56x64xf32>
      %cst_5 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %109 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %110 = stablehlo.add %105, %109 : tensor<1x1x1x64xf32>
      %111 = stablehlo.rsqrt %110 : tensor<1x1x1x64xf32>
      %112 = stablehlo.reshape %arg238 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %113 = stablehlo.multiply %111, %112 : tensor<1x1x1x64xf32>
      %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %115 = stablehlo.multiply %108, %114 : tensor<256x56x56x64xf32>
      %116 = stablehlo.reshape %arg237 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %117 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %118 = stablehlo.add %115, %117 : tensor<256x56x56x64xf32>
      %119 = stablehlo.convert %118 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %120 = func.call @relu_13(%119) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %121 = stablehlo.convert %arg242 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %122 = stablehlo.convolution(%120, %121) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %123 = stablehlo.broadcast_in_dim %arg330, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %124 = stablehlo.broadcast_in_dim %arg331, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %125 = stablehlo.convert %122 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %126 = stablehlo.broadcast_in_dim %123, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %127 = stablehlo.subtract %125, %126 : tensor<256x56x56x64xf32>
      %cst_6 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %128 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %129 = stablehlo.add %124, %128 : tensor<1x1x1x64xf32>
      %130 = stablehlo.rsqrt %129 : tensor<1x1x1x64xf32>
      %131 = stablehlo.reshape %arg240 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %132 = stablehlo.multiply %130, %131 : tensor<1x1x1x64xf32>
      %133 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %134 = stablehlo.multiply %127, %133 : tensor<256x56x56x64xf32>
      %135 = stablehlo.reshape %arg239 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %137 = stablehlo.add %134, %136 : tensor<256x56x56x64xf32>
      %138 = stablehlo.convert %137 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %139 = stablehlo.add %101, %138 : tensor<256x56x56x64xf16>
      %140 = func.call @relu_13(%139) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %141 = stablehlo.convert %arg247 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf16>
      %142 = stablehlo.convolution(%140, %141) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x128xf16>) -> tensor<256x28x28x128xf16>
      %143 = stablehlo.broadcast_in_dim %arg332, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %144 = stablehlo.broadcast_in_dim %arg333, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %145 = stablehlo.convert %142 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %146 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %147 = stablehlo.subtract %145, %146 : tensor<256x28x28x128xf32>
      %cst_7 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %148 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %149 = stablehlo.add %144, %148 : tensor<1x1x1x128xf32>
      %150 = stablehlo.rsqrt %149 : tensor<1x1x1x128xf32>
      %151 = stablehlo.reshape %arg244 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %152 = stablehlo.multiply %150, %151 : tensor<1x1x1x128xf32>
      %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %154 = stablehlo.multiply %147, %153 : tensor<256x28x28x128xf32>
      %155 = stablehlo.reshape %arg243 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %156 = stablehlo.broadcast_in_dim %155, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %157 = stablehlo.add %154, %156 : tensor<256x28x28x128xf32>
      %158 = stablehlo.convert %157 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %159 = func.call @relu_34(%158) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %160 = stablehlo.convert %arg248 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %161 = stablehlo.convolution(%159, %160) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %162 = stablehlo.broadcast_in_dim %arg334, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %163 = stablehlo.broadcast_in_dim %arg335, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %164 = stablehlo.convert %161 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %165 = stablehlo.broadcast_in_dim %162, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %166 = stablehlo.subtract %164, %165 : tensor<256x28x28x128xf32>
      %cst_8 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %167 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %168 = stablehlo.add %163, %167 : tensor<1x1x1x128xf32>
      %169 = stablehlo.rsqrt %168 : tensor<1x1x1x128xf32>
      %170 = stablehlo.reshape %arg246 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %171 = stablehlo.multiply %169, %170 : tensor<1x1x1x128xf32>
      %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %173 = stablehlo.multiply %166, %172 : tensor<256x28x28x128xf32>
      %174 = stablehlo.reshape %arg245 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %175 = stablehlo.broadcast_in_dim %174, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %176 = stablehlo.add %173, %175 : tensor<256x28x28x128xf32>
      %177 = stablehlo.convert %176 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %178 = stablehlo.convert %arg249 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf16>
      %179 = stablehlo.convolution(%140, %178) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x128xf16>) -> tensor<256x28x28x128xf16>
      %180 = stablehlo.broadcast_in_dim %arg336, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %181 = stablehlo.broadcast_in_dim %arg337, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %182 = stablehlo.convert %179 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %183 = stablehlo.broadcast_in_dim %180, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %184 = stablehlo.subtract %182, %183 : tensor<256x28x28x128xf32>
      %cst_9 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %185 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %186 = stablehlo.add %181, %185 : tensor<1x1x1x128xf32>
      %187 = stablehlo.rsqrt %186 : tensor<1x1x1x128xf32>
      %188 = stablehlo.reshape %arg251 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %189 = stablehlo.multiply %187, %188 : tensor<1x1x1x128xf32>
      %190 = stablehlo.broadcast_in_dim %189, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %191 = stablehlo.multiply %184, %190 : tensor<256x28x28x128xf32>
      %192 = stablehlo.reshape %arg250 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %194 = stablehlo.add %191, %193 : tensor<256x28x28x128xf32>
      %195 = stablehlo.convert %194 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %196 = stablehlo.add %195, %177 : tensor<256x28x28x128xf16>
      %197 = func.call @relu_34(%196) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %198 = stablehlo.convert %arg256 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %199 = stablehlo.convolution(%197, %198) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %200 = stablehlo.broadcast_in_dim %arg338, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %201 = stablehlo.broadcast_in_dim %arg339, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %202 = stablehlo.convert %199 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %203 = stablehlo.broadcast_in_dim %200, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %204 = stablehlo.subtract %202, %203 : tensor<256x28x28x128xf32>
      %cst_10 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %205 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %206 = stablehlo.add %201, %205 : tensor<1x1x1x128xf32>
      %207 = stablehlo.rsqrt %206 : tensor<1x1x1x128xf32>
      %208 = stablehlo.reshape %arg253 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %209 = stablehlo.multiply %207, %208 : tensor<1x1x1x128xf32>
      %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %211 = stablehlo.multiply %204, %210 : tensor<256x28x28x128xf32>
      %212 = stablehlo.reshape %arg252 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %214 = stablehlo.add %211, %213 : tensor<256x28x28x128xf32>
      %215 = stablehlo.convert %214 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %216 = func.call @relu_34(%215) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %217 = stablehlo.convert %arg257 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %218 = stablehlo.convolution(%216, %217) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %219 = stablehlo.broadcast_in_dim %arg340, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %220 = stablehlo.broadcast_in_dim %arg341, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %221 = stablehlo.convert %218 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %222 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %223 = stablehlo.subtract %221, %222 : tensor<256x28x28x128xf32>
      %cst_11 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %224 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %225 = stablehlo.add %220, %224 : tensor<1x1x1x128xf32>
      %226 = stablehlo.rsqrt %225 : tensor<1x1x1x128xf32>
      %227 = stablehlo.reshape %arg255 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %228 = stablehlo.multiply %226, %227 : tensor<1x1x1x128xf32>
      %229 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %230 = stablehlo.multiply %223, %229 : tensor<256x28x28x128xf32>
      %231 = stablehlo.reshape %arg254 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %232 = stablehlo.broadcast_in_dim %231, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %233 = stablehlo.add %230, %232 : tensor<256x28x28x128xf32>
      %234 = stablehlo.convert %233 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %235 = stablehlo.add %197, %234 : tensor<256x28x28x128xf16>
      %236 = func.call @relu_34(%235) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %237 = stablehlo.convert %arg262 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %238 = stablehlo.convolution(%236, %237) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %239 = stablehlo.broadcast_in_dim %arg342, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %240 = stablehlo.broadcast_in_dim %arg343, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %241 = stablehlo.convert %238 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %242 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %243 = stablehlo.subtract %241, %242 : tensor<256x28x28x128xf32>
      %cst_12 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %244 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %245 = stablehlo.add %240, %244 : tensor<1x1x1x128xf32>
      %246 = stablehlo.rsqrt %245 : tensor<1x1x1x128xf32>
      %247 = stablehlo.reshape %arg259 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %248 = stablehlo.multiply %246, %247 : tensor<1x1x1x128xf32>
      %249 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %250 = stablehlo.multiply %243, %249 : tensor<256x28x28x128xf32>
      %251 = stablehlo.reshape %arg258 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %252 = stablehlo.broadcast_in_dim %251, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %253 = stablehlo.add %250, %252 : tensor<256x28x28x128xf32>
      %254 = stablehlo.convert %253 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %255 = func.call @relu_34(%254) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %256 = stablehlo.convert %arg263 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %257 = stablehlo.convolution(%255, %256) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %258 = stablehlo.broadcast_in_dim %arg344, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %259 = stablehlo.broadcast_in_dim %arg345, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %260 = stablehlo.convert %257 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %261 = stablehlo.broadcast_in_dim %258, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %262 = stablehlo.subtract %260, %261 : tensor<256x28x28x128xf32>
      %cst_13 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %264 = stablehlo.add %259, %263 : tensor<1x1x1x128xf32>
      %265 = stablehlo.rsqrt %264 : tensor<1x1x1x128xf32>
      %266 = stablehlo.reshape %arg261 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %267 = stablehlo.multiply %265, %266 : tensor<1x1x1x128xf32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %269 = stablehlo.multiply %262, %268 : tensor<256x28x28x128xf32>
      %270 = stablehlo.reshape %arg260 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %271 = stablehlo.broadcast_in_dim %270, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %272 = stablehlo.add %269, %271 : tensor<256x28x28x128xf32>
      %273 = stablehlo.convert %272 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %274 = stablehlo.add %236, %273 : tensor<256x28x28x128xf16>
      %275 = func.call @relu_34(%274) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %276 = stablehlo.convert %arg268 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %277 = stablehlo.convolution(%275, %276) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %278 = stablehlo.broadcast_in_dim %arg346, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %279 = stablehlo.broadcast_in_dim %arg347, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %280 = stablehlo.convert %277 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %281 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %282 = stablehlo.subtract %280, %281 : tensor<256x28x28x128xf32>
      %cst_14 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %283 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %284 = stablehlo.add %279, %283 : tensor<1x1x1x128xf32>
      %285 = stablehlo.rsqrt %284 : tensor<1x1x1x128xf32>
      %286 = stablehlo.reshape %arg265 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %287 = stablehlo.multiply %285, %286 : tensor<1x1x1x128xf32>
      %288 = stablehlo.broadcast_in_dim %287, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %289 = stablehlo.multiply %282, %288 : tensor<256x28x28x128xf32>
      %290 = stablehlo.reshape %arg264 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %291 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %292 = stablehlo.add %289, %291 : tensor<256x28x28x128xf32>
      %293 = stablehlo.convert %292 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %294 = func.call @relu_34(%293) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %295 = stablehlo.convert %arg269 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %296 = stablehlo.convolution(%294, %295) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %297 = stablehlo.broadcast_in_dim %arg348, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %298 = stablehlo.broadcast_in_dim %arg349, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %299 = stablehlo.convert %296 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %300 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %301 = stablehlo.subtract %299, %300 : tensor<256x28x28x128xf32>
      %cst_15 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %303 = stablehlo.add %298, %302 : tensor<1x1x1x128xf32>
      %304 = stablehlo.rsqrt %303 : tensor<1x1x1x128xf32>
      %305 = stablehlo.reshape %arg267 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x1x1x128xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %308 = stablehlo.multiply %301, %307 : tensor<256x28x28x128xf32>
      %309 = stablehlo.reshape %arg266 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %311 = stablehlo.add %308, %310 : tensor<256x28x28x128xf32>
      %312 = stablehlo.convert %311 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %313 = stablehlo.add %275, %312 : tensor<256x28x28x128xf16>
      %314 = func.call @relu_34(%313) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %315 = stablehlo.convert %arg274 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf16>
      %316 = stablehlo.convolution(%314, %315) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x256xf16>) -> tensor<256x14x14x256xf16>
      %317 = stablehlo.broadcast_in_dim %arg350, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %318 = stablehlo.broadcast_in_dim %arg351, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %319 = stablehlo.convert %316 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %320 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %321 = stablehlo.subtract %319, %320 : tensor<256x14x14x256xf32>
      %cst_16 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %323 = stablehlo.add %318, %322 : tensor<1x1x1x256xf32>
      %324 = stablehlo.rsqrt %323 : tensor<1x1x1x256xf32>
      %325 = stablehlo.reshape %arg271 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %326 = stablehlo.multiply %324, %325 : tensor<1x1x1x256xf32>
      %327 = stablehlo.broadcast_in_dim %326, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %328 = stablehlo.multiply %321, %327 : tensor<256x14x14x256xf32>
      %329 = stablehlo.reshape %arg270 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %331 = stablehlo.add %328, %330 : tensor<256x14x14x256xf32>
      %332 = stablehlo.convert %331 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %333 = func.call @relu_61(%332) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %334 = stablehlo.convert %arg275 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %335 = stablehlo.convolution(%333, %334) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %336 = stablehlo.broadcast_in_dim %arg352, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %337 = stablehlo.broadcast_in_dim %arg353, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %338 = stablehlo.convert %335 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %339 = stablehlo.broadcast_in_dim %336, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %340 = stablehlo.subtract %338, %339 : tensor<256x14x14x256xf32>
      %cst_17 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %342 = stablehlo.add %337, %341 : tensor<1x1x1x256xf32>
      %343 = stablehlo.rsqrt %342 : tensor<1x1x1x256xf32>
      %344 = stablehlo.reshape %arg273 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %345 = stablehlo.multiply %343, %344 : tensor<1x1x1x256xf32>
      %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %347 = stablehlo.multiply %340, %346 : tensor<256x14x14x256xf32>
      %348 = stablehlo.reshape %arg272 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %350 = stablehlo.add %347, %349 : tensor<256x14x14x256xf32>
      %351 = stablehlo.convert %350 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %352 = stablehlo.convert %arg276 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf16>
      %353 = stablehlo.convolution(%314, %352) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x256xf16>) -> tensor<256x14x14x256xf16>
      %354 = stablehlo.broadcast_in_dim %arg354, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %355 = stablehlo.broadcast_in_dim %arg355, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %356 = stablehlo.convert %353 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %357 = stablehlo.broadcast_in_dim %354, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %358 = stablehlo.subtract %356, %357 : tensor<256x14x14x256xf32>
      %cst_18 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %360 = stablehlo.add %355, %359 : tensor<1x1x1x256xf32>
      %361 = stablehlo.rsqrt %360 : tensor<1x1x1x256xf32>
      %362 = stablehlo.reshape %arg278 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %363 = stablehlo.multiply %361, %362 : tensor<1x1x1x256xf32>
      %364 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %365 = stablehlo.multiply %358, %364 : tensor<256x14x14x256xf32>
      %366 = stablehlo.reshape %arg277 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %367 = stablehlo.broadcast_in_dim %366, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %368 = stablehlo.add %365, %367 : tensor<256x14x14x256xf32>
      %369 = stablehlo.convert %368 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %370 = stablehlo.add %369, %351 : tensor<256x14x14x256xf16>
      %371 = func.call @relu_61(%370) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %372 = stablehlo.convert %arg283 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %373 = stablehlo.convolution(%371, %372) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %374 = stablehlo.broadcast_in_dim %arg356, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %375 = stablehlo.broadcast_in_dim %arg357, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %376 = stablehlo.convert %373 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %377 = stablehlo.broadcast_in_dim %374, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %378 = stablehlo.subtract %376, %377 : tensor<256x14x14x256xf32>
      %cst_19 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %380 = stablehlo.add %375, %379 : tensor<1x1x1x256xf32>
      %381 = stablehlo.rsqrt %380 : tensor<1x1x1x256xf32>
      %382 = stablehlo.reshape %arg280 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %383 = stablehlo.multiply %381, %382 : tensor<1x1x1x256xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %385 = stablehlo.multiply %378, %384 : tensor<256x14x14x256xf32>
      %386 = stablehlo.reshape %arg279 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %388 = stablehlo.add %385, %387 : tensor<256x14x14x256xf32>
      %389 = stablehlo.convert %388 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %390 = func.call @relu_61(%389) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %391 = stablehlo.convert %arg284 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %392 = stablehlo.convolution(%390, %391) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %393 = stablehlo.broadcast_in_dim %arg358, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %394 = stablehlo.broadcast_in_dim %arg359, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %395 = stablehlo.convert %392 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %396 = stablehlo.broadcast_in_dim %393, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %397 = stablehlo.subtract %395, %396 : tensor<256x14x14x256xf32>
      %cst_20 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %399 = stablehlo.add %394, %398 : tensor<1x1x1x256xf32>
      %400 = stablehlo.rsqrt %399 : tensor<1x1x1x256xf32>
      %401 = stablehlo.reshape %arg282 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %402 = stablehlo.multiply %400, %401 : tensor<1x1x1x256xf32>
      %403 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %404 = stablehlo.multiply %397, %403 : tensor<256x14x14x256xf32>
      %405 = stablehlo.reshape %arg281 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %406 = stablehlo.broadcast_in_dim %405, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %407 = stablehlo.add %404, %406 : tensor<256x14x14x256xf32>
      %408 = stablehlo.convert %407 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %409 = stablehlo.add %371, %408 : tensor<256x14x14x256xf16>
      %410 = func.call @relu_61(%409) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %411 = stablehlo.convert %arg289 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %412 = stablehlo.convolution(%410, %411) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %413 = stablehlo.broadcast_in_dim %arg360, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %414 = stablehlo.broadcast_in_dim %arg361, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %415 = stablehlo.convert %412 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %416 = stablehlo.broadcast_in_dim %413, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %417 = stablehlo.subtract %415, %416 : tensor<256x14x14x256xf32>
      %cst_21 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %419 = stablehlo.add %414, %418 : tensor<1x1x1x256xf32>
      %420 = stablehlo.rsqrt %419 : tensor<1x1x1x256xf32>
      %421 = stablehlo.reshape %arg286 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x1x1x256xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %424 = stablehlo.multiply %417, %423 : tensor<256x14x14x256xf32>
      %425 = stablehlo.reshape %arg285 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %427 = stablehlo.add %424, %426 : tensor<256x14x14x256xf32>
      %428 = stablehlo.convert %427 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %429 = func.call @relu_61(%428) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %430 = stablehlo.convert %arg290 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %431 = stablehlo.convolution(%429, %430) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %432 = stablehlo.broadcast_in_dim %arg362, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %433 = stablehlo.broadcast_in_dim %arg363, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %434 = stablehlo.convert %431 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %435 = stablehlo.broadcast_in_dim %432, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %436 = stablehlo.subtract %434, %435 : tensor<256x14x14x256xf32>
      %cst_22 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %438 = stablehlo.add %433, %437 : tensor<1x1x1x256xf32>
      %439 = stablehlo.rsqrt %438 : tensor<1x1x1x256xf32>
      %440 = stablehlo.reshape %arg288 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %441 = stablehlo.multiply %439, %440 : tensor<1x1x1x256xf32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %443 = stablehlo.multiply %436, %442 : tensor<256x14x14x256xf32>
      %444 = stablehlo.reshape %arg287 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %445 = stablehlo.broadcast_in_dim %444, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %446 = stablehlo.add %443, %445 : tensor<256x14x14x256xf32>
      %447 = stablehlo.convert %446 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %448 = stablehlo.add %410, %447 : tensor<256x14x14x256xf16>
      %449 = func.call @relu_61(%448) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %450 = stablehlo.convert %arg202 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %451 = stablehlo.convolution(%449, %450) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %452 = stablehlo.broadcast_in_dim %arg302, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %453 = stablehlo.broadcast_in_dim %arg303, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %454 = stablehlo.convert %451 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %455 = stablehlo.broadcast_in_dim %452, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %456 = stablehlo.subtract %454, %455 : tensor<256x14x14x256xf32>
      %cst_23 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %458 = stablehlo.add %453, %457 : tensor<1x1x1x256xf32>
      %459 = stablehlo.rsqrt %458 : tensor<1x1x1x256xf32>
      %460 = stablehlo.reshape %arg199 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %461 = stablehlo.multiply %459, %460 : tensor<1x1x1x256xf32>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %463 = stablehlo.multiply %456, %462 : tensor<256x14x14x256xf32>
      %464 = stablehlo.reshape %arg198 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %465 = stablehlo.broadcast_in_dim %464, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %466 = stablehlo.add %463, %465 : tensor<256x14x14x256xf32>
      %467 = stablehlo.convert %466 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %468 = func.call @relu_61(%467) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %469 = stablehlo.convert %arg203 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %470 = stablehlo.convolution(%468, %469) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %471 = stablehlo.broadcast_in_dim %arg304, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %472 = stablehlo.broadcast_in_dim %arg305, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %473 = stablehlo.convert %470 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %474 = stablehlo.broadcast_in_dim %471, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %475 = stablehlo.subtract %473, %474 : tensor<256x14x14x256xf32>
      %cst_24 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %477 = stablehlo.add %472, %476 : tensor<1x1x1x256xf32>
      %478 = stablehlo.rsqrt %477 : tensor<1x1x1x256xf32>
      %479 = stablehlo.reshape %arg201 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %480 = stablehlo.multiply %478, %479 : tensor<1x1x1x256xf32>
      %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %482 = stablehlo.multiply %475, %481 : tensor<256x14x14x256xf32>
      %483 = stablehlo.reshape %arg200 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %484 = stablehlo.broadcast_in_dim %483, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %485 = stablehlo.add %482, %484 : tensor<256x14x14x256xf32>
      %486 = stablehlo.convert %485 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %487 = stablehlo.add %449, %486 : tensor<256x14x14x256xf16>
      %488 = func.call @relu_61(%487) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %489 = stablehlo.convert %arg208 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %490 = stablehlo.convolution(%488, %489) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %491 = stablehlo.broadcast_in_dim %arg306, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %492 = stablehlo.broadcast_in_dim %arg307, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %493 = stablehlo.convert %490 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %494 = stablehlo.broadcast_in_dim %491, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %495 = stablehlo.subtract %493, %494 : tensor<256x14x14x256xf32>
      %cst_25 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %496 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %497 = stablehlo.add %492, %496 : tensor<1x1x1x256xf32>
      %498 = stablehlo.rsqrt %497 : tensor<1x1x1x256xf32>
      %499 = stablehlo.reshape %arg205 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %500 = stablehlo.multiply %498, %499 : tensor<1x1x1x256xf32>
      %501 = stablehlo.broadcast_in_dim %500, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %502 = stablehlo.multiply %495, %501 : tensor<256x14x14x256xf32>
      %503 = stablehlo.reshape %arg204 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %504 = stablehlo.broadcast_in_dim %503, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %505 = stablehlo.add %502, %504 : tensor<256x14x14x256xf32>
      %506 = stablehlo.convert %505 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %507 = func.call @relu_61(%506) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %508 = stablehlo.convert %arg209 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %509 = stablehlo.convolution(%507, %508) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %510 = stablehlo.broadcast_in_dim %arg308, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %511 = stablehlo.broadcast_in_dim %arg309, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %512 = stablehlo.convert %509 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %513 = stablehlo.broadcast_in_dim %510, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %514 = stablehlo.subtract %512, %513 : tensor<256x14x14x256xf32>
      %cst_26 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %515 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %516 = stablehlo.add %511, %515 : tensor<1x1x1x256xf32>
      %517 = stablehlo.rsqrt %516 : tensor<1x1x1x256xf32>
      %518 = stablehlo.reshape %arg207 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %519 = stablehlo.multiply %517, %518 : tensor<1x1x1x256xf32>
      %520 = stablehlo.broadcast_in_dim %519, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %521 = stablehlo.multiply %514, %520 : tensor<256x14x14x256xf32>
      %522 = stablehlo.reshape %arg206 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %523 = stablehlo.broadcast_in_dim %522, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %524 = stablehlo.add %521, %523 : tensor<256x14x14x256xf32>
      %525 = stablehlo.convert %524 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %526 = stablehlo.add %488, %525 : tensor<256x14x14x256xf16>
      %527 = func.call @relu_61(%526) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %528 = stablehlo.convert %arg214 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %529 = stablehlo.convolution(%527, %528) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %530 = stablehlo.broadcast_in_dim %arg310, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %531 = stablehlo.broadcast_in_dim %arg311, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %532 = stablehlo.convert %529 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %533 = stablehlo.broadcast_in_dim %530, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %534 = stablehlo.subtract %532, %533 : tensor<256x14x14x256xf32>
      %cst_27 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %536 = stablehlo.add %531, %535 : tensor<1x1x1x256xf32>
      %537 = stablehlo.rsqrt %536 : tensor<1x1x1x256xf32>
      %538 = stablehlo.reshape %arg211 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %539 = stablehlo.multiply %537, %538 : tensor<1x1x1x256xf32>
      %540 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %541 = stablehlo.multiply %534, %540 : tensor<256x14x14x256xf32>
      %542 = stablehlo.reshape %arg210 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %543 = stablehlo.broadcast_in_dim %542, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %544 = stablehlo.add %541, %543 : tensor<256x14x14x256xf32>
      %545 = stablehlo.convert %544 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %546 = func.call @relu_61(%545) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %547 = stablehlo.convert %arg215 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %548 = stablehlo.convolution(%546, %547) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %549 = stablehlo.broadcast_in_dim %arg312, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %550 = stablehlo.broadcast_in_dim %arg313, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %551 = stablehlo.convert %548 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %552 = stablehlo.broadcast_in_dim %549, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %553 = stablehlo.subtract %551, %552 : tensor<256x14x14x256xf32>
      %cst_28 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %554 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %555 = stablehlo.add %550, %554 : tensor<1x1x1x256xf32>
      %556 = stablehlo.rsqrt %555 : tensor<1x1x1x256xf32>
      %557 = stablehlo.reshape %arg213 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %558 = stablehlo.multiply %556, %557 : tensor<1x1x1x256xf32>
      %559 = stablehlo.broadcast_in_dim %558, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %560 = stablehlo.multiply %553, %559 : tensor<256x14x14x256xf32>
      %561 = stablehlo.reshape %arg212 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %562 = stablehlo.broadcast_in_dim %561, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %563 = stablehlo.add %560, %562 : tensor<256x14x14x256xf32>
      %564 = stablehlo.convert %563 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %565 = stablehlo.add %527, %564 : tensor<256x14x14x256xf16>
      %566 = func.call @relu_61(%565) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %567 = stablehlo.convert %arg220 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf16>
      %568 = stablehlo.convolution(%566, %567) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x512xf16>) -> tensor<256x7x7x512xf16>
      %569 = stablehlo.broadcast_in_dim %arg314, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %570 = stablehlo.broadcast_in_dim %arg315, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %571 = stablehlo.convert %568 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %572 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %573 = stablehlo.subtract %571, %572 : tensor<256x7x7x512xf32>
      %cst_29 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %574 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %575 = stablehlo.add %570, %574 : tensor<1x1x1x512xf32>
      %576 = stablehlo.rsqrt %575 : tensor<1x1x1x512xf32>
      %577 = stablehlo.reshape %arg217 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %578 = stablehlo.multiply %576, %577 : tensor<1x1x1x512xf32>
      %579 = stablehlo.broadcast_in_dim %578, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %580 = stablehlo.multiply %573, %579 : tensor<256x7x7x512xf32>
      %581 = stablehlo.reshape %arg216 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %582 = stablehlo.broadcast_in_dim %581, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %583 = stablehlo.add %580, %582 : tensor<256x7x7x512xf32>
      %584 = stablehlo.convert %583 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %585 = func.call @relu_92(%584) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %586 = stablehlo.convert %arg221 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %587 = stablehlo.convolution(%585, %586) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %588 = stablehlo.broadcast_in_dim %arg316, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %589 = stablehlo.broadcast_in_dim %arg317, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %590 = stablehlo.convert %587 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %591 = stablehlo.broadcast_in_dim %588, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %592 = stablehlo.subtract %590, %591 : tensor<256x7x7x512xf32>
      %cst_30 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %593 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %594 = stablehlo.add %589, %593 : tensor<1x1x1x512xf32>
      %595 = stablehlo.rsqrt %594 : tensor<1x1x1x512xf32>
      %596 = stablehlo.reshape %arg219 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %597 = stablehlo.multiply %595, %596 : tensor<1x1x1x512xf32>
      %598 = stablehlo.broadcast_in_dim %597, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %599 = stablehlo.multiply %592, %598 : tensor<256x7x7x512xf32>
      %600 = stablehlo.reshape %arg218 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %601 = stablehlo.broadcast_in_dim %600, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %602 = stablehlo.add %599, %601 : tensor<256x7x7x512xf32>
      %603 = stablehlo.convert %602 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %604 = stablehlo.convert %arg222 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf16>
      %605 = stablehlo.convolution(%566, %604) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x512xf16>) -> tensor<256x7x7x512xf16>
      %606 = stablehlo.broadcast_in_dim %arg318, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %607 = stablehlo.broadcast_in_dim %arg319, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %608 = stablehlo.convert %605 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %609 = stablehlo.broadcast_in_dim %606, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %610 = stablehlo.subtract %608, %609 : tensor<256x7x7x512xf32>
      %cst_31 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %611 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %612 = stablehlo.add %607, %611 : tensor<1x1x1x512xf32>
      %613 = stablehlo.rsqrt %612 : tensor<1x1x1x512xf32>
      %614 = stablehlo.reshape %arg224 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %615 = stablehlo.multiply %613, %614 : tensor<1x1x1x512xf32>
      %616 = stablehlo.broadcast_in_dim %615, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %617 = stablehlo.multiply %610, %616 : tensor<256x7x7x512xf32>
      %618 = stablehlo.reshape %arg223 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %619 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %620 = stablehlo.add %617, %619 : tensor<256x7x7x512xf32>
      %621 = stablehlo.convert %620 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %622 = stablehlo.add %621, %603 : tensor<256x7x7x512xf16>
      %623 = func.call @relu_92(%622) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %624 = stablehlo.convert %arg229 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %625 = stablehlo.convolution(%623, %624) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %626 = stablehlo.broadcast_in_dim %arg320, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %627 = stablehlo.broadcast_in_dim %arg321, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %628 = stablehlo.convert %625 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %629 = stablehlo.broadcast_in_dim %626, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %630 = stablehlo.subtract %628, %629 : tensor<256x7x7x512xf32>
      %cst_32 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %631 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %632 = stablehlo.add %627, %631 : tensor<1x1x1x512xf32>
      %633 = stablehlo.rsqrt %632 : tensor<1x1x1x512xf32>
      %634 = stablehlo.reshape %arg226 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %635 = stablehlo.multiply %633, %634 : tensor<1x1x1x512xf32>
      %636 = stablehlo.broadcast_in_dim %635, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %637 = stablehlo.multiply %630, %636 : tensor<256x7x7x512xf32>
      %638 = stablehlo.reshape %arg225 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %639 = stablehlo.broadcast_in_dim %638, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %640 = stablehlo.add %637, %639 : tensor<256x7x7x512xf32>
      %641 = stablehlo.convert %640 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %642 = func.call @relu_92(%641) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %643 = stablehlo.convert %arg230 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %644 = stablehlo.convolution(%642, %643) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %645 = stablehlo.broadcast_in_dim %arg322, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %646 = stablehlo.broadcast_in_dim %arg323, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %647 = stablehlo.convert %644 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %648 = stablehlo.broadcast_in_dim %645, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %649 = stablehlo.subtract %647, %648 : tensor<256x7x7x512xf32>
      %cst_33 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %650 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %651 = stablehlo.add %646, %650 : tensor<1x1x1x512xf32>
      %652 = stablehlo.rsqrt %651 : tensor<1x1x1x512xf32>
      %653 = stablehlo.reshape %arg228 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %654 = stablehlo.multiply %652, %653 : tensor<1x1x1x512xf32>
      %655 = stablehlo.broadcast_in_dim %654, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %656 = stablehlo.multiply %649, %655 : tensor<256x7x7x512xf32>
      %657 = stablehlo.reshape %arg227 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %658 = stablehlo.broadcast_in_dim %657, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %659 = stablehlo.add %656, %658 : tensor<256x7x7x512xf32>
      %660 = stablehlo.convert %659 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %661 = stablehlo.add %623, %660 : tensor<256x7x7x512xf16>
      %662 = func.call @relu_92(%661) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %663 = stablehlo.convert %arg235 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %664 = stablehlo.convolution(%662, %663) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %665 = stablehlo.broadcast_in_dim %arg324, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %666 = stablehlo.broadcast_in_dim %arg325, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %667 = stablehlo.convert %664 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %668 = stablehlo.broadcast_in_dim %665, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %669 = stablehlo.subtract %667, %668 : tensor<256x7x7x512xf32>
      %cst_34 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %670 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %671 = stablehlo.add %666, %670 : tensor<1x1x1x512xf32>
      %672 = stablehlo.rsqrt %671 : tensor<1x1x1x512xf32>
      %673 = stablehlo.reshape %arg232 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %674 = stablehlo.multiply %672, %673 : tensor<1x1x1x512xf32>
      %675 = stablehlo.broadcast_in_dim %674, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %676 = stablehlo.multiply %669, %675 : tensor<256x7x7x512xf32>
      %677 = stablehlo.reshape %arg231 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %678 = stablehlo.broadcast_in_dim %677, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %679 = stablehlo.add %676, %678 : tensor<256x7x7x512xf32>
      %680 = stablehlo.convert %679 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %681 = func.call @relu_92(%680) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %682 = stablehlo.convert %arg236 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %683 = stablehlo.convolution(%681, %682) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %684 = stablehlo.broadcast_in_dim %arg326, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %685 = stablehlo.broadcast_in_dim %arg327, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %686 = stablehlo.convert %683 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %687 = stablehlo.broadcast_in_dim %684, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %688 = stablehlo.subtract %686, %687 : tensor<256x7x7x512xf32>
      %cst_35 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %689 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %690 = stablehlo.add %685, %689 : tensor<1x1x1x512xf32>
      %691 = stablehlo.rsqrt %690 : tensor<1x1x1x512xf32>
      %692 = stablehlo.reshape %arg234 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %693 = stablehlo.multiply %691, %692 : tensor<1x1x1x512xf32>
      %694 = stablehlo.broadcast_in_dim %693, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %695 = stablehlo.multiply %688, %694 : tensor<256x7x7x512xf32>
      %696 = stablehlo.reshape %arg233 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %697 = stablehlo.broadcast_in_dim %696, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %698 = stablehlo.add %695, %697 : tensor<256x7x7x512xf32>
      %699 = stablehlo.convert %698 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %700 = stablehlo.add %662, %699 : tensor<256x7x7x512xf16>
      %701 = func.call @relu_92(%700) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %702 = stablehlo.convert %701 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %703 = stablehlo.reduce(%702 init: %cst_36) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<256x512xf32>
      %cst_37 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
      %704 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
      %705 = stablehlo.divide %703, %704 : tensor<256x512xf32>
      %706 = stablehlo.convert %705 : (tensor<256x512xf32>) -> tensor<256x512xf16>
      %707 = stablehlo.convert %arg185 : (tensor<512x1000xf32>) -> tensor<512x1000xf16>
      %708 = stablehlo.convert %arg184 : (tensor<1000xf32>) -> tensor<1000xf16>
      %709 = stablehlo.dot_general %706, %707, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf16>, tensor<512x1000xf16>) -> tensor<256x1000xf16>
      %710 = stablehlo.reshape %708 : (tensor<1000xf16>) -> tensor<1x1000xf16>
      %711 = stablehlo.broadcast_in_dim %710, dims = [0, 1] : (tensor<1x1000xf16>) -> tensor<256x1000xf16>
      %712 = stablehlo.add %709, %711 : tensor<256x1000xf16>
      %713 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
      %714 = stablehlo.iota dim = 0 : tensor<1000xi32>
      %715 = stablehlo.reshape %714 : (tensor<1000xi32>) -> tensor<1x1000xi32>
      %716 = stablehlo.broadcast_in_dim %713, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
      %717 = stablehlo.broadcast_in_dim %715, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
      %718 = stablehlo.compare  EQ, %716, %717,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
      %cst_38 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %719 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %720 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %721 = stablehlo.select %718, %719, %720 : tensor<256x1000xi1>, tensor<256x1000xf32>
      %722 = stablehlo.convert %721 : tensor<256x1000xf32>
      %723 = func.call @log_softmax(%712) : (tensor<256x1000xf16>) -> tensor<256x1000xf16>
      %724 = stablehlo.convert %723 : (tensor<256x1000xf16>) -> tensor<256x1000xf32>
      %725 = stablehlo.multiply %722, %724 : tensor<256x1000xf32>
      %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %726 = stablehlo.reduce(%725 init: %cst_40) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
      %727 = stablehlo.negate %726 : tensor<256xf32>
      %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %728 = stablehlo.reduce(%727 init: %cst_41) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_42 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %729 = stablehlo.divide %728, %cst_42 : tensor<f32>
      %730 = func.call @argmax(%712) : (tensor<256x1000xf16>) -> tensor<256xi32>
      %731 = stablehlo.compare  EQ, %730, %2,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
      %732 = stablehlo.convert %731 : (tensor<256xi1>) -> tensor<256xi32>
      %733 = stablehlo.convert %732 : (tensor<256xi32>) -> tensor<256xf32>
      %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %734 = stablehlo.reduce(%733 init: %cst_43) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_44 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %735 = stablehlo.divide %734, %cst_44 : tensor<f32>
      %736 = "stablehlo.all_reduce"(%735) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg368: tensor<f32>, %arg369: tensor<f32>):
        %742 = stablehlo.add %arg368, %arg369 : tensor<f32>
        stablehlo.return %742 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %737 = "stablehlo.all_reduce"(%729) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg368: tensor<f32>, %arg369: tensor<f32>):
        %742 = stablehlo.add %arg368, %arg369 : tensor<f32>
        stablehlo.return %742 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_45 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %738 = stablehlo.divide %736, %cst_45 : tensor<f32>
      %cst_46 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %739 = stablehlo.divide %737, %cst_46 : tensor<f32>
      %740 = stablehlo.broadcast_in_dim %738, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %741 = stablehlo.broadcast_in_dim %739, dims = [] : (tensor<f32>) -> tensor<1xf32>
      sdy.return %740, %741 : tensor<1xf32>, tensor<1xf32>
    } : (tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x256x224x224x3xf16>, tensor<4x256xi32>) -> (tensor<4xf32>, tensor<4xf32>)
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
  func.func private @relu_34(%arg0: tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xf16>
    return %1 : tensor<256x28x28x128xf16>
  }
  func.func private @relu_61(%arg0: tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xf16>
    return %1 : tensor<256x14x14x256xf16>
  }
  func.func private @relu_92(%arg0: tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16> {
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
