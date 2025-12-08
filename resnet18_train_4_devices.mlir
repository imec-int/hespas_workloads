module @pmap_train_step attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg1: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg2: tensor<512x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg3: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg4: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg5: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg6: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg7: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg8: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg9: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg10: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg11: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg12: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg13: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg14: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg15: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg16: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg17: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg18: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg19: tensor<3x3x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg20: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg21: tensor<1x1x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg22: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg23: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg24: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg25: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg26: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg27: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg28: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg29: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg30: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg31: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg32: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg33: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg34: tensor<3x3x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg35: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg36: tensor<1x1x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg37: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg38: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg39: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg40: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg41: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg42: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg43: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg44: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg45: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg46: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg47: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg48: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg49: tensor<3x3x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg50: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg51: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg52: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg53: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg54: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg55: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg56: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg57: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg58: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg59: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg60: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg61: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg62: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg63: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg64: tensor<512x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg65: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg66: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg67: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg68: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg69: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg70: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg71: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg72: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg73: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg74: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg75: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg76: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg77: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg78: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg79: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg80: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg81: tensor<3x3x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg82: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg83: tensor<1x1x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg84: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg85: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg86: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg87: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg88: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg89: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg90: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg91: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg92: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg93: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg94: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg95: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg96: tensor<3x3x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg97: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg98: tensor<1x1x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg99: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg100: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg101: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg102: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg103: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg104: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg105: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg106: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg107: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg108: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg109: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg110: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg111: tensor<3x3x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg112: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg113: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg114: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg115: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg116: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg117: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg118: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg119: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg120: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg121: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg122: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg123: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg124: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg125: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg126: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg127: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg128: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg129: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg130: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg131: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg132: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg133: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg134: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg135: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg136: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg137: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg138: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg139: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg140: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg141: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg142: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg143: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg144: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg145: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg146: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg147: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg148: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg149: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg150: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg151: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg152: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg153: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg154: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg155: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg156: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg157: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg158: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg159: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg160: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg161: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg162: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg163: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg164: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg165: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg166: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg167: tensor<f32> {mhlo.is_same_data_across_replicas = true}, %arg168: tensor<1x256x224x224x3xbf16>, %arg169: tensor<1x256xi32>) -> (tensor<i32> {jax.result_info = "result[0].step"}, tensor<1000xf32> {jax.result_info = "result[0].params['Dense_0']['bias']"}, tensor<512x1000xf32> {jax.result_info = "result[0].params['Dense_0']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['Conv_1']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['BatchNorm_1']['scale']"}, tensor<3x3x64x128xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['Conv_1']['kernel']"}, tensor<1x1x64x128xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['conv_proj']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['norm_proj']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['norm_proj']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['BatchNorm_1']['scale']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['BatchNorm_1']['scale']"}, tensor<3x3x128x256xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['Conv_1']['kernel']"}, tensor<1x1x128x256xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['conv_proj']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['norm_proj']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['norm_proj']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['Conv_1']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['BatchNorm_1']['scale']"}, tensor<3x3x256x512xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['Conv_1']['kernel']"}, tensor<1x1x256x512xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['conv_proj']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['norm_proj']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['norm_proj']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['BatchNorm_1']['scale']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].params['bn_init']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['bn_init']['scale']"}, tensor<7x7x3x64xf32> {jax.result_info = "result[0].params['conv_init']['kernel']"}, tensor<1000xf32> {jax.result_info = "result[0].opt_state[0].trace['Dense_0']['bias']"}, tensor<512x1000xf32> {jax.result_info = "result[0].opt_state[0].trace['Dense_0']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['Conv_1']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['BatchNorm_1']['scale']"}, tensor<3x3x64x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['Conv_1']['kernel']"}, tensor<1x1x64x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['conv_proj']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['norm_proj']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['norm_proj']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['BatchNorm_1']['scale']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['BatchNorm_1']['scale']"}, tensor<3x3x128x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['Conv_1']['kernel']"}, tensor<1x1x128x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['conv_proj']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['norm_proj']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['norm_proj']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['Conv_1']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['BatchNorm_1']['scale']"}, tensor<3x3x256x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['Conv_1']['kernel']"}, tensor<1x1x256x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['conv_proj']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['norm_proj']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['norm_proj']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['BatchNorm_1']['scale']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['bn_init']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['bn_init']['scale']"}, tensor<7x7x3x64xf32> {jax.result_info = "result[0].opt_state[0].trace['conv_init']['kernel']"}, tensor<i32> {jax.result_info = "result[0].opt_state[1].count"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_0']['BatchNorm_0']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_0']['BatchNorm_0']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_0']['BatchNorm_1']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_0']['BatchNorm_1']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_1']['BatchNorm_0']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_1']['BatchNorm_0']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_1']['BatchNorm_1']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_1']['BatchNorm_1']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['BatchNorm_0']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['BatchNorm_0']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['BatchNorm_1']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['BatchNorm_1']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['norm_proj']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['norm_proj']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['BatchNorm_0']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['BatchNorm_0']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['BatchNorm_1']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['BatchNorm_1']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['BatchNorm_0']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['BatchNorm_0']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['BatchNorm_1']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['BatchNorm_1']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['norm_proj']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['norm_proj']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_5']['BatchNorm_0']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_5']['BatchNorm_0']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_5']['BatchNorm_1']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_5']['BatchNorm_1']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['BatchNorm_0']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['BatchNorm_0']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['BatchNorm_1']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['BatchNorm_1']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['norm_proj']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['norm_proj']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['BatchNorm_0']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['BatchNorm_0']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['BatchNorm_1']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['BatchNorm_1']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['bn_init']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['bn_init']['var']"}, tensor<i32> {jax.result_info = "result[0].dynamic_scale.fin_steps"}, tensor<f32> {jax.result_info = "result[0].dynamic_scale.scale"}, tensor<1xf32> {jax.result_info = "result[1]['accuracy']"}, tensor<1xf32> {jax.result_info = "result[1]['learning_rate']"}, tensor<1xf32> {jax.result_info = "result[1]['loss']"}, tensor<1xf32> {jax.result_info = "result[1]['scale']"}) {
    %0 = stablehlo.reshape %arg168 : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
    %1 = stablehlo.reshape %arg169 : (tensor<1x256xi32>) -> tensor<256xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.subtract %arg0, %c : tensor<i32>
    %cst = stablehlo.constant dense<5.000000e+02> : tensor<f32>
    %3 = call @clip(%2, %c, %cst) : (tensor<i32>, tensor<i32>, tensor<f32>) -> tensor<f32>
    %4 = stablehlo.divide %3, %cst : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.subtract %cst_0, %4 : tensor<f32>
    %cst_1 = stablehlo.constant dense<-4.000000e-01> : tensor<f32>
    %6 = stablehlo.multiply %cst_1, %5 : tensor<f32>
    %cst_2 = stablehlo.constant dense<4.000000e-01> : tensor<f32>
    %7 = stablehlo.add %6, %cst_2 : tensor<f32>
    %8 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %9 = stablehlo.compare  LT, %8, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %10 = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<f32>
    %11 = stablehlo.subtract %10, %cst : tensor<f32>
    %cst_3 = stablehlo.constant dense<9.500000e+03> : tensor<f32>
    %12 = stablehlo.minimum %11, %cst_3 : tensor<f32>
    %cst_4 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %13 = stablehlo.multiply %cst_4, %12 : tensor<f32>
    %14 = stablehlo.divide %13, %cst_3 : tensor<f32>
    %15 = stablehlo.cosine %14 : tensor<f32>
    %16 = stablehlo.add %cst_0, %15 : tensor<f32>
    %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %17 = stablehlo.multiply %cst_5, %16 : tensor<f32>
    %18 = stablehlo.power %17, %cst_0 : tensor<f32>
    %19 = stablehlo.multiply %cst_0, %18 : tensor<f32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = stablehlo.add %19, %cst_6 : tensor<f32>
    %21 = stablehlo.multiply %cst_2, %20 : tensor<f32>
    %22 = call @_where(%9, %7, %21) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.convert %arg62 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xbf16>
    %24 = stablehlo.convolution(%0, %23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xbf16>, tensor<7x7x3x64xbf16>) -> tensor<256x112x112x64xbf16>
    %25 = stablehlo.convert %24 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
    %26 = stablehlo.multiply %25, %25 : tensor<256x112x112x64xf32>
    %cst_7 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x112x112x64xf32>
    %28 = stablehlo.multiply %27, %25 : tensor<256x112x112x64xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %29 = stablehlo.reduce(%25 init: %cst_8) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_9 = stablehlo.constant dense<0x4A440000> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %31 = stablehlo.divide %29, %30 : tensor<64xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %32 = stablehlo.reduce(%26 init: %cst_10) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
    %33 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %34 = stablehlo.divide %32, %33 : tensor<64xf32>
    %35 = stablehlo.broadcast_in_dim %31, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %36 = stablehlo.broadcast_in_dim %34, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %37 = stablehlo.concatenate %35, %36, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
    %38 = "stablehlo.all_reduce"(%37) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %cst_11 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %40 = stablehlo.divide %38, %39 : tensor<2x64xf32>
    %41 = stablehlo.slice %40 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %42 = stablehlo.reshape %41 : (tensor<1x64xf32>) -> tensor<64xf32>
    %43 = stablehlo.slice %40 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %44 = stablehlo.reshape %43 : (tensor<1x64xf32>) -> tensor<64xf32>
    %45 = stablehlo.multiply %42, %42 : tensor<64xf32>
    %46 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %47 = stablehlo.multiply %46, %42 : tensor<64xf32>
    %48 = stablehlo.subtract %44, %45 : tensor<64xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %49 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %50 = stablehlo.maximum %49, %48 : tensor<64xf32>
    %51 = stablehlo.compare  EQ, %48, %50,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %53 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %54 = stablehlo.select %51, %52, %53 : tensor<64xi1>, tensor<64xf32>
    %55 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %56 = stablehlo.compare  EQ, %55, %50,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %57 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %58 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %59 = stablehlo.select %56, %57, %58 : tensor<64xi1>, tensor<64xf32>
    %60 = stablehlo.divide %54, %59 : tensor<64xf32>
    %cst_14 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %62 = stablehlo.multiply %61, %arg164 : tensor<64xf32>
    %cst_15 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %63 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %64 = stablehlo.multiply %63, %42 : tensor<64xf32>
    %65 = stablehlo.add %62, %64 : tensor<64xf32>
    %66 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %67 = stablehlo.multiply %66, %arg165 : tensor<64xf32>
    %68 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %69 = stablehlo.multiply %68, %50 : tensor<64xf32>
    %70 = stablehlo.add %67, %69 : tensor<64xf32>
    %71 = stablehlo.broadcast_in_dim %42, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %72 = stablehlo.broadcast_in_dim %50, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %73 = stablehlo.convert %24 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
    %74 = stablehlo.broadcast_in_dim %71, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %75 = stablehlo.subtract %73, %74 : tensor<256x112x112x64xf32>
    %cst_16 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %77 = stablehlo.add %72, %76 : tensor<1x1x1x64xf32>
    %78 = stablehlo.rsqrt %77 : tensor<1x1x1x64xf32>
    %79 = stablehlo.divide %78, %77 : tensor<1x1x1x64xf32>
    %cst_17 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %81 = stablehlo.multiply %80, %79 : tensor<1x1x1x64xf32>
    %82 = stablehlo.reshape %arg61 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %83 = stablehlo.multiply %78, %82 : tensor<1x1x1x64xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %85 = stablehlo.multiply %75, %84 : tensor<256x112x112x64xf32>
    %86 = stablehlo.reshape %arg60 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %88 = stablehlo.add %85, %87 : tensor<256x112x112x64xf32>
    %89 = stablehlo.convert %88 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %90 = call @relu(%89) : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xbf16>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %91 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
    %92 = stablehlo.compare  GT, %89, %91,  FLOAT : (tensor<256x112x112x64xbf16>, tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xi1>
    %93 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
    %cst_19 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %94 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %95 = "stablehlo.reduce_window"(%90, %94) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg170: tensor<bf16>, %arg171: tensor<bf16>):
      %4188 = stablehlo.maximum %arg170, %arg171 : tensor<bf16>
      stablehlo.return %4188 : tensor<bf16>
    }) : (tensor<256x112x112x64xbf16>, tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %96 = stablehlo.convert %arg7 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %97 = stablehlo.convolution(%95, %96) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %98 = stablehlo.convert %97 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %99 = stablehlo.multiply %98, %98 : tensor<256x56x56x64xf32>
    %100 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
    %101 = stablehlo.multiply %100, %98 : tensor<256x56x56x64xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %102 = stablehlo.reduce(%98 init: %cst_20) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_21 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
    %103 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %104 = stablehlo.divide %102, %103 : tensor<64xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %105 = stablehlo.reduce(%99 init: %cst_22) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %106 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %107 = stablehlo.divide %105, %106 : tensor<64xf32>
    %108 = stablehlo.broadcast_in_dim %104, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %109 = stablehlo.broadcast_in_dim %107, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %110 = stablehlo.concatenate %108, %109, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
    %111 = "stablehlo.all_reduce"(%110) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %112 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %113 = stablehlo.divide %111, %112 : tensor<2x64xf32>
    %114 = stablehlo.slice %113 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %115 = stablehlo.reshape %114 : (tensor<1x64xf32>) -> tensor<64xf32>
    %116 = stablehlo.slice %113 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %117 = stablehlo.reshape %116 : (tensor<1x64xf32>) -> tensor<64xf32>
    %118 = stablehlo.multiply %115, %115 : tensor<64xf32>
    %119 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %120 = stablehlo.multiply %119, %115 : tensor<64xf32>
    %121 = stablehlo.subtract %117, %118 : tensor<64xf32>
    %122 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %123 = stablehlo.maximum %122, %121 : tensor<64xf32>
    %124 = stablehlo.compare  EQ, %121, %123,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %125 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %126 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %127 = stablehlo.select %124, %125, %126 : tensor<64xi1>, tensor<64xf32>
    %128 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %129 = stablehlo.compare  EQ, %128, %123,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %130 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %131 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %132 = stablehlo.select %129, %130, %131 : tensor<64xi1>, tensor<64xf32>
    %133 = stablehlo.divide %127, %132 : tensor<64xf32>
    %134 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %135 = stablehlo.multiply %134, %arg126 : tensor<64xf32>
    %136 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %137 = stablehlo.multiply %136, %115 : tensor<64xf32>
    %138 = stablehlo.add %135, %137 : tensor<64xf32>
    %139 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %140 = stablehlo.multiply %139, %arg127 : tensor<64xf32>
    %141 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %142 = stablehlo.multiply %141, %123 : tensor<64xf32>
    %143 = stablehlo.add %140, %142 : tensor<64xf32>
    %144 = stablehlo.broadcast_in_dim %115, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %145 = stablehlo.broadcast_in_dim %123, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %146 = stablehlo.convert %97 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %147 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %148 = stablehlo.subtract %146, %147 : tensor<256x56x56x64xf32>
    %149 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %150 = stablehlo.add %145, %149 : tensor<1x1x1x64xf32>
    %151 = stablehlo.rsqrt %150 : tensor<1x1x1x64xf32>
    %152 = stablehlo.divide %151, %150 : tensor<1x1x1x64xf32>
    %153 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %154 = stablehlo.multiply %153, %152 : tensor<1x1x1x64xf32>
    %155 = stablehlo.reshape %arg4 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %156 = stablehlo.multiply %151, %155 : tensor<1x1x1x64xf32>
    %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %158 = stablehlo.multiply %148, %157 : tensor<256x56x56x64xf32>
    %159 = stablehlo.reshape %arg3 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %160 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %161 = stablehlo.add %158, %160 : tensor<256x56x56x64xf32>
    %162 = stablehlo.convert %161 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %163 = call @relu_41(%162) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %164 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %165 = stablehlo.compare  GT, %162, %164,  FLOAT : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xi1>
    %166 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %167 = stablehlo.convert %arg8 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %168 = stablehlo.convolution(%163, %167) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %169 = stablehlo.convert %168 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %170 = stablehlo.multiply %169, %169 : tensor<256x56x56x64xf32>
    %171 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
    %172 = stablehlo.multiply %171, %169 : tensor<256x56x56x64xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %173 = stablehlo.reduce(%169 init: %cst_23) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %174 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %175 = stablehlo.divide %173, %174 : tensor<64xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %176 = stablehlo.reduce(%170 init: %cst_24) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %177 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %178 = stablehlo.divide %176, %177 : tensor<64xf32>
    %179 = stablehlo.broadcast_in_dim %175, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %180 = stablehlo.broadcast_in_dim %178, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %181 = stablehlo.concatenate %179, %180, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
    %182 = "stablehlo.all_reduce"(%181) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %183 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %184 = stablehlo.divide %182, %183 : tensor<2x64xf32>
    %185 = stablehlo.slice %184 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %186 = stablehlo.reshape %185 : (tensor<1x64xf32>) -> tensor<64xf32>
    %187 = stablehlo.slice %184 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %188 = stablehlo.reshape %187 : (tensor<1x64xf32>) -> tensor<64xf32>
    %189 = stablehlo.multiply %186, %186 : tensor<64xf32>
    %190 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %191 = stablehlo.multiply %190, %186 : tensor<64xf32>
    %192 = stablehlo.subtract %188, %189 : tensor<64xf32>
    %193 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %194 = stablehlo.maximum %193, %192 : tensor<64xf32>
    %195 = stablehlo.compare  EQ, %192, %194,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %196 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %197 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %198 = stablehlo.select %195, %196, %197 : tensor<64xi1>, tensor<64xf32>
    %199 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %200 = stablehlo.compare  EQ, %199, %194,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %201 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %202 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %203 = stablehlo.select %200, %201, %202 : tensor<64xi1>, tensor<64xf32>
    %204 = stablehlo.divide %198, %203 : tensor<64xf32>
    %205 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %206 = stablehlo.multiply %205, %arg128 : tensor<64xf32>
    %207 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %208 = stablehlo.multiply %207, %186 : tensor<64xf32>
    %209 = stablehlo.add %206, %208 : tensor<64xf32>
    %210 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %211 = stablehlo.multiply %210, %arg129 : tensor<64xf32>
    %212 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %213 = stablehlo.multiply %212, %194 : tensor<64xf32>
    %214 = stablehlo.add %211, %213 : tensor<64xf32>
    %215 = stablehlo.broadcast_in_dim %186, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %216 = stablehlo.broadcast_in_dim %194, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %217 = stablehlo.convert %168 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %218 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %219 = stablehlo.subtract %217, %218 : tensor<256x56x56x64xf32>
    %220 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %221 = stablehlo.add %216, %220 : tensor<1x1x1x64xf32>
    %222 = stablehlo.rsqrt %221 : tensor<1x1x1x64xf32>
    %223 = stablehlo.divide %222, %221 : tensor<1x1x1x64xf32>
    %224 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %225 = stablehlo.multiply %224, %223 : tensor<1x1x1x64xf32>
    %226 = stablehlo.reshape %arg6 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %227 = stablehlo.multiply %222, %226 : tensor<1x1x1x64xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %229 = stablehlo.multiply %219, %228 : tensor<256x56x56x64xf32>
    %230 = stablehlo.reshape %arg5 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %231 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %232 = stablehlo.add %229, %231 : tensor<256x56x56x64xf32>
    %233 = stablehlo.convert %232 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %234 = stablehlo.add %95, %233 : tensor<256x56x56x64xbf16>
    %235 = call @relu_41(%234) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %236 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %237 = stablehlo.compare  GT, %234, %236,  FLOAT : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xi1>
    %238 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %239 = stablehlo.convert %arg13 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %240 = stablehlo.convolution(%235, %239) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %241 = stablehlo.convert %240 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %242 = stablehlo.multiply %241, %241 : tensor<256x56x56x64xf32>
    %243 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
    %244 = stablehlo.multiply %243, %241 : tensor<256x56x56x64xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %245 = stablehlo.reduce(%241 init: %cst_25) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %246 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %247 = stablehlo.divide %245, %246 : tensor<64xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %248 = stablehlo.reduce(%242 init: %cst_26) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %249 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %250 = stablehlo.divide %248, %249 : tensor<64xf32>
    %251 = stablehlo.broadcast_in_dim %247, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %252 = stablehlo.broadcast_in_dim %250, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %253 = stablehlo.concatenate %251, %252, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
    %254 = "stablehlo.all_reduce"(%253) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %255 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %256 = stablehlo.divide %254, %255 : tensor<2x64xf32>
    %257 = stablehlo.slice %256 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %258 = stablehlo.reshape %257 : (tensor<1x64xf32>) -> tensor<64xf32>
    %259 = stablehlo.slice %256 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %260 = stablehlo.reshape %259 : (tensor<1x64xf32>) -> tensor<64xf32>
    %261 = stablehlo.multiply %258, %258 : tensor<64xf32>
    %262 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %263 = stablehlo.multiply %262, %258 : tensor<64xf32>
    %264 = stablehlo.subtract %260, %261 : tensor<64xf32>
    %265 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %266 = stablehlo.maximum %265, %264 : tensor<64xf32>
    %267 = stablehlo.compare  EQ, %264, %266,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %268 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %269 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %270 = stablehlo.select %267, %268, %269 : tensor<64xi1>, tensor<64xf32>
    %271 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %272 = stablehlo.compare  EQ, %271, %266,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %273 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %274 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %275 = stablehlo.select %272, %273, %274 : tensor<64xi1>, tensor<64xf32>
    %276 = stablehlo.divide %270, %275 : tensor<64xf32>
    %277 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %278 = stablehlo.multiply %277, %arg130 : tensor<64xf32>
    %279 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %280 = stablehlo.multiply %279, %258 : tensor<64xf32>
    %281 = stablehlo.add %278, %280 : tensor<64xf32>
    %282 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %283 = stablehlo.multiply %282, %arg131 : tensor<64xf32>
    %284 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %285 = stablehlo.multiply %284, %266 : tensor<64xf32>
    %286 = stablehlo.add %283, %285 : tensor<64xf32>
    %287 = stablehlo.broadcast_in_dim %258, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %288 = stablehlo.broadcast_in_dim %266, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %289 = stablehlo.convert %240 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %290 = stablehlo.broadcast_in_dim %287, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %291 = stablehlo.subtract %289, %290 : tensor<256x56x56x64xf32>
    %292 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %293 = stablehlo.add %288, %292 : tensor<1x1x1x64xf32>
    %294 = stablehlo.rsqrt %293 : tensor<1x1x1x64xf32>
    %295 = stablehlo.divide %294, %293 : tensor<1x1x1x64xf32>
    %296 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %297 = stablehlo.multiply %296, %295 : tensor<1x1x1x64xf32>
    %298 = stablehlo.reshape %arg10 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %299 = stablehlo.multiply %294, %298 : tensor<1x1x1x64xf32>
    %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %301 = stablehlo.multiply %291, %300 : tensor<256x56x56x64xf32>
    %302 = stablehlo.reshape %arg9 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %304 = stablehlo.add %301, %303 : tensor<256x56x56x64xf32>
    %305 = stablehlo.convert %304 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %306 = call @relu_41(%305) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %307 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %308 = stablehlo.compare  GT, %305, %307,  FLOAT : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xi1>
    %309 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %310 = stablehlo.convert %arg14 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %311 = stablehlo.convolution(%306, %310) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %312 = stablehlo.convert %311 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %313 = stablehlo.multiply %312, %312 : tensor<256x56x56x64xf32>
    %314 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
    %315 = stablehlo.multiply %314, %312 : tensor<256x56x56x64xf32>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %316 = stablehlo.reduce(%312 init: %cst_27) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %317 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %318 = stablehlo.divide %316, %317 : tensor<64xf32>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %319 = stablehlo.reduce(%313 init: %cst_28) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %320 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %321 = stablehlo.divide %319, %320 : tensor<64xf32>
    %322 = stablehlo.broadcast_in_dim %318, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %323 = stablehlo.broadcast_in_dim %321, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %324 = stablehlo.concatenate %322, %323, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
    %325 = "stablehlo.all_reduce"(%324) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %326 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %327 = stablehlo.divide %325, %326 : tensor<2x64xf32>
    %328 = stablehlo.slice %327 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %329 = stablehlo.reshape %328 : (tensor<1x64xf32>) -> tensor<64xf32>
    %330 = stablehlo.slice %327 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %331 = stablehlo.reshape %330 : (tensor<1x64xf32>) -> tensor<64xf32>
    %332 = stablehlo.multiply %329, %329 : tensor<64xf32>
    %333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %334 = stablehlo.multiply %333, %329 : tensor<64xf32>
    %335 = stablehlo.subtract %331, %332 : tensor<64xf32>
    %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %337 = stablehlo.maximum %336, %335 : tensor<64xf32>
    %338 = stablehlo.compare  EQ, %335, %337,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %340 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %341 = stablehlo.select %338, %339, %340 : tensor<64xi1>, tensor<64xf32>
    %342 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %343 = stablehlo.compare  EQ, %342, %337,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %344 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %345 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %346 = stablehlo.select %343, %344, %345 : tensor<64xi1>, tensor<64xf32>
    %347 = stablehlo.divide %341, %346 : tensor<64xf32>
    %348 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %349 = stablehlo.multiply %348, %arg132 : tensor<64xf32>
    %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %351 = stablehlo.multiply %350, %329 : tensor<64xf32>
    %352 = stablehlo.add %349, %351 : tensor<64xf32>
    %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %354 = stablehlo.multiply %353, %arg133 : tensor<64xf32>
    %355 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %356 = stablehlo.multiply %355, %337 : tensor<64xf32>
    %357 = stablehlo.add %354, %356 : tensor<64xf32>
    %358 = stablehlo.broadcast_in_dim %329, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %359 = stablehlo.broadcast_in_dim %337, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %360 = stablehlo.convert %311 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %361 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %362 = stablehlo.subtract %360, %361 : tensor<256x56x56x64xf32>
    %363 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %364 = stablehlo.add %359, %363 : tensor<1x1x1x64xf32>
    %365 = stablehlo.rsqrt %364 : tensor<1x1x1x64xf32>
    %366 = stablehlo.divide %365, %364 : tensor<1x1x1x64xf32>
    %367 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %368 = stablehlo.multiply %367, %366 : tensor<1x1x1x64xf32>
    %369 = stablehlo.reshape %arg12 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %370 = stablehlo.multiply %365, %369 : tensor<1x1x1x64xf32>
    %371 = stablehlo.broadcast_in_dim %370, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %372 = stablehlo.multiply %362, %371 : tensor<256x56x56x64xf32>
    %373 = stablehlo.reshape %arg11 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %374 = stablehlo.broadcast_in_dim %373, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %375 = stablehlo.add %372, %374 : tensor<256x56x56x64xf32>
    %376 = stablehlo.convert %375 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %377 = stablehlo.add %235, %376 : tensor<256x56x56x64xbf16>
    %378 = call @relu_41(%377) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %379 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %380 = stablehlo.compare  GT, %377, %379,  FLOAT : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xi1>
    %381 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %382 = stablehlo.convert %arg19 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xbf16>
    %383 = stablehlo.convolution(%378, %382) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x128xbf16>) -> tensor<256x28x28x128xbf16>
    %384 = stablehlo.convert %383 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %385 = stablehlo.multiply %384, %384 : tensor<256x28x28x128xf32>
    %386 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %387 = stablehlo.multiply %386, %384 : tensor<256x28x28x128xf32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %388 = stablehlo.reduce(%384 init: %cst_29) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_30 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
    %389 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %390 = stablehlo.divide %388, %389 : tensor<128xf32>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %391 = stablehlo.reduce(%385 init: %cst_31) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %392 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %393 = stablehlo.divide %391, %392 : tensor<128xf32>
    %394 = stablehlo.broadcast_in_dim %390, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %395 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %396 = stablehlo.concatenate %394, %395, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %397 = "stablehlo.all_reduce"(%396) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %398 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %399 = stablehlo.divide %397, %398 : tensor<2x128xf32>
    %400 = stablehlo.slice %399 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %401 = stablehlo.reshape %400 : (tensor<1x128xf32>) -> tensor<128xf32>
    %402 = stablehlo.slice %399 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %403 = stablehlo.reshape %402 : (tensor<1x128xf32>) -> tensor<128xf32>
    %404 = stablehlo.multiply %401, %401 : tensor<128xf32>
    %405 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %406 = stablehlo.multiply %405, %401 : tensor<128xf32>
    %407 = stablehlo.subtract %403, %404 : tensor<128xf32>
    %408 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %409 = stablehlo.maximum %408, %407 : tensor<128xf32>
    %410 = stablehlo.compare  EQ, %407, %409,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %411 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %412 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %413 = stablehlo.select %410, %411, %412 : tensor<128xi1>, tensor<128xf32>
    %414 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %415 = stablehlo.compare  EQ, %414, %409,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %416 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %417 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %418 = stablehlo.select %415, %416, %417 : tensor<128xi1>, tensor<128xf32>
    %419 = stablehlo.divide %413, %418 : tensor<128xf32>
    %420 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %421 = stablehlo.multiply %420, %arg134 : tensor<128xf32>
    %422 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %423 = stablehlo.multiply %422, %401 : tensor<128xf32>
    %424 = stablehlo.add %421, %423 : tensor<128xf32>
    %425 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %426 = stablehlo.multiply %425, %arg135 : tensor<128xf32>
    %427 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %428 = stablehlo.multiply %427, %409 : tensor<128xf32>
    %429 = stablehlo.add %426, %428 : tensor<128xf32>
    %430 = stablehlo.broadcast_in_dim %401, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %431 = stablehlo.broadcast_in_dim %409, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %432 = stablehlo.convert %383 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %433 = stablehlo.broadcast_in_dim %430, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %434 = stablehlo.subtract %432, %433 : tensor<256x28x28x128xf32>
    %435 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %436 = stablehlo.add %431, %435 : tensor<1x1x1x128xf32>
    %437 = stablehlo.rsqrt %436 : tensor<1x1x1x128xf32>
    %438 = stablehlo.divide %437, %436 : tensor<1x1x1x128xf32>
    %439 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %440 = stablehlo.multiply %439, %438 : tensor<1x1x1x128xf32>
    %441 = stablehlo.reshape %arg16 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %442 = stablehlo.multiply %437, %441 : tensor<1x1x1x128xf32>
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %444 = stablehlo.multiply %434, %443 : tensor<256x28x28x128xf32>
    %445 = stablehlo.reshape %arg15 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %446 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %447 = stablehlo.add %444, %446 : tensor<256x28x28x128xf32>
    %448 = stablehlo.convert %447 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %449 = call @relu_85(%448) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %450 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %451 = stablehlo.compare  GT, %448, %450,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %453 = stablehlo.convert %arg20 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %454 = stablehlo.convolution(%449, %453) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %455 = stablehlo.convert %454 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %456 = stablehlo.multiply %455, %455 : tensor<256x28x28x128xf32>
    %457 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %458 = stablehlo.multiply %457, %455 : tensor<256x28x28x128xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %459 = stablehlo.reduce(%455 init: %cst_32) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %460 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %461 = stablehlo.divide %459, %460 : tensor<128xf32>
    %cst_33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %462 = stablehlo.reduce(%456 init: %cst_33) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %463 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %464 = stablehlo.divide %462, %463 : tensor<128xf32>
    %465 = stablehlo.broadcast_in_dim %461, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %466 = stablehlo.broadcast_in_dim %464, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %467 = stablehlo.concatenate %465, %466, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %468 = "stablehlo.all_reduce"(%467) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %469 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %470 = stablehlo.divide %468, %469 : tensor<2x128xf32>
    %471 = stablehlo.slice %470 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %472 = stablehlo.reshape %471 : (tensor<1x128xf32>) -> tensor<128xf32>
    %473 = stablehlo.slice %470 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %474 = stablehlo.reshape %473 : (tensor<1x128xf32>) -> tensor<128xf32>
    %475 = stablehlo.multiply %472, %472 : tensor<128xf32>
    %476 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %477 = stablehlo.multiply %476, %472 : tensor<128xf32>
    %478 = stablehlo.subtract %474, %475 : tensor<128xf32>
    %479 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %480 = stablehlo.maximum %479, %478 : tensor<128xf32>
    %481 = stablehlo.compare  EQ, %478, %480,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %482 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %483 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %484 = stablehlo.select %481, %482, %483 : tensor<128xi1>, tensor<128xf32>
    %485 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %486 = stablehlo.compare  EQ, %485, %480,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %487 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %488 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %489 = stablehlo.select %486, %487, %488 : tensor<128xi1>, tensor<128xf32>
    %490 = stablehlo.divide %484, %489 : tensor<128xf32>
    %491 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %492 = stablehlo.multiply %491, %arg136 : tensor<128xf32>
    %493 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %494 = stablehlo.multiply %493, %472 : tensor<128xf32>
    %495 = stablehlo.add %492, %494 : tensor<128xf32>
    %496 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %497 = stablehlo.multiply %496, %arg137 : tensor<128xf32>
    %498 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %499 = stablehlo.multiply %498, %480 : tensor<128xf32>
    %500 = stablehlo.add %497, %499 : tensor<128xf32>
    %501 = stablehlo.broadcast_in_dim %472, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %502 = stablehlo.broadcast_in_dim %480, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %503 = stablehlo.convert %454 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %504 = stablehlo.broadcast_in_dim %501, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %505 = stablehlo.subtract %503, %504 : tensor<256x28x28x128xf32>
    %506 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %507 = stablehlo.add %502, %506 : tensor<1x1x1x128xf32>
    %508 = stablehlo.rsqrt %507 : tensor<1x1x1x128xf32>
    %509 = stablehlo.divide %508, %507 : tensor<1x1x1x128xf32>
    %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %511 = stablehlo.multiply %510, %509 : tensor<1x1x1x128xf32>
    %512 = stablehlo.reshape %arg18 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %513 = stablehlo.multiply %508, %512 : tensor<1x1x1x128xf32>
    %514 = stablehlo.broadcast_in_dim %513, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %515 = stablehlo.multiply %505, %514 : tensor<256x28x28x128xf32>
    %516 = stablehlo.reshape %arg17 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %518 = stablehlo.add %515, %517 : tensor<256x28x28x128xf32>
    %519 = stablehlo.convert %518 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %520 = stablehlo.convert %arg21 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xbf16>
    %521 = stablehlo.convolution(%378, %520) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x128xbf16>) -> tensor<256x28x28x128xbf16>
    %522 = stablehlo.convert %521 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %523 = stablehlo.multiply %522, %522 : tensor<256x28x28x128xf32>
    %524 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %525 = stablehlo.multiply %524, %522 : tensor<256x28x28x128xf32>
    %cst_34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %526 = stablehlo.reduce(%522 init: %cst_34) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %527 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %528 = stablehlo.divide %526, %527 : tensor<128xf32>
    %cst_35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %529 = stablehlo.reduce(%523 init: %cst_35) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %530 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %531 = stablehlo.divide %529, %530 : tensor<128xf32>
    %532 = stablehlo.broadcast_in_dim %528, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %533 = stablehlo.broadcast_in_dim %531, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %534 = stablehlo.concatenate %532, %533, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %535 = "stablehlo.all_reduce"(%534) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %536 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %537 = stablehlo.divide %535, %536 : tensor<2x128xf32>
    %538 = stablehlo.slice %537 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %539 = stablehlo.reshape %538 : (tensor<1x128xf32>) -> tensor<128xf32>
    %540 = stablehlo.slice %537 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %541 = stablehlo.reshape %540 : (tensor<1x128xf32>) -> tensor<128xf32>
    %542 = stablehlo.multiply %539, %539 : tensor<128xf32>
    %543 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %544 = stablehlo.multiply %543, %539 : tensor<128xf32>
    %545 = stablehlo.subtract %541, %542 : tensor<128xf32>
    %546 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %547 = stablehlo.maximum %546, %545 : tensor<128xf32>
    %548 = stablehlo.compare  EQ, %545, %547,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %549 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %550 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %551 = stablehlo.select %548, %549, %550 : tensor<128xi1>, tensor<128xf32>
    %552 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %553 = stablehlo.compare  EQ, %552, %547,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %554 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %555 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %556 = stablehlo.select %553, %554, %555 : tensor<128xi1>, tensor<128xf32>
    %557 = stablehlo.divide %551, %556 : tensor<128xf32>
    %558 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %559 = stablehlo.multiply %558, %arg138 : tensor<128xf32>
    %560 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %561 = stablehlo.multiply %560, %539 : tensor<128xf32>
    %562 = stablehlo.add %559, %561 : tensor<128xf32>
    %563 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %564 = stablehlo.multiply %563, %arg139 : tensor<128xf32>
    %565 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %566 = stablehlo.multiply %565, %547 : tensor<128xf32>
    %567 = stablehlo.add %564, %566 : tensor<128xf32>
    %568 = stablehlo.broadcast_in_dim %539, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %569 = stablehlo.broadcast_in_dim %547, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %570 = stablehlo.convert %521 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %571 = stablehlo.broadcast_in_dim %568, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %572 = stablehlo.subtract %570, %571 : tensor<256x28x28x128xf32>
    %573 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %574 = stablehlo.add %569, %573 : tensor<1x1x1x128xf32>
    %575 = stablehlo.rsqrt %574 : tensor<1x1x1x128xf32>
    %576 = stablehlo.divide %575, %574 : tensor<1x1x1x128xf32>
    %577 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %578 = stablehlo.multiply %577, %576 : tensor<1x1x1x128xf32>
    %579 = stablehlo.reshape %arg23 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %580 = stablehlo.multiply %575, %579 : tensor<1x1x1x128xf32>
    %581 = stablehlo.broadcast_in_dim %580, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %582 = stablehlo.multiply %572, %581 : tensor<256x28x28x128xf32>
    %583 = stablehlo.reshape %arg22 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %584 = stablehlo.broadcast_in_dim %583, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %585 = stablehlo.add %582, %584 : tensor<256x28x28x128xf32>
    %586 = stablehlo.convert %585 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %587 = stablehlo.add %586, %519 : tensor<256x28x28x128xbf16>
    %588 = call @relu_85(%587) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %589 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %590 = stablehlo.compare  GT, %587, %589,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %591 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %592 = stablehlo.convert %arg28 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %593 = stablehlo.convolution(%588, %592) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %594 = stablehlo.convert %593 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %595 = stablehlo.multiply %594, %594 : tensor<256x28x28x128xf32>
    %596 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %597 = stablehlo.multiply %596, %594 : tensor<256x28x28x128xf32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %598 = stablehlo.reduce(%594 init: %cst_36) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %599 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %600 = stablehlo.divide %598, %599 : tensor<128xf32>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %601 = stablehlo.reduce(%595 init: %cst_37) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %602 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %603 = stablehlo.divide %601, %602 : tensor<128xf32>
    %604 = stablehlo.broadcast_in_dim %600, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %605 = stablehlo.broadcast_in_dim %603, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %606 = stablehlo.concatenate %604, %605, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %607 = "stablehlo.all_reduce"(%606) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %608 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %609 = stablehlo.divide %607, %608 : tensor<2x128xf32>
    %610 = stablehlo.slice %609 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %611 = stablehlo.reshape %610 : (tensor<1x128xf32>) -> tensor<128xf32>
    %612 = stablehlo.slice %609 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %613 = stablehlo.reshape %612 : (tensor<1x128xf32>) -> tensor<128xf32>
    %614 = stablehlo.multiply %611, %611 : tensor<128xf32>
    %615 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %616 = stablehlo.multiply %615, %611 : tensor<128xf32>
    %617 = stablehlo.subtract %613, %614 : tensor<128xf32>
    %618 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %619 = stablehlo.maximum %618, %617 : tensor<128xf32>
    %620 = stablehlo.compare  EQ, %617, %619,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %621 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %622 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %623 = stablehlo.select %620, %621, %622 : tensor<128xi1>, tensor<128xf32>
    %624 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %625 = stablehlo.compare  EQ, %624, %619,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %626 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %627 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %628 = stablehlo.select %625, %626, %627 : tensor<128xi1>, tensor<128xf32>
    %629 = stablehlo.divide %623, %628 : tensor<128xf32>
    %630 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %631 = stablehlo.multiply %630, %arg140 : tensor<128xf32>
    %632 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %633 = stablehlo.multiply %632, %611 : tensor<128xf32>
    %634 = stablehlo.add %631, %633 : tensor<128xf32>
    %635 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %636 = stablehlo.multiply %635, %arg141 : tensor<128xf32>
    %637 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %638 = stablehlo.multiply %637, %619 : tensor<128xf32>
    %639 = stablehlo.add %636, %638 : tensor<128xf32>
    %640 = stablehlo.broadcast_in_dim %611, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %641 = stablehlo.broadcast_in_dim %619, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %642 = stablehlo.convert %593 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %643 = stablehlo.broadcast_in_dim %640, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %644 = stablehlo.subtract %642, %643 : tensor<256x28x28x128xf32>
    %645 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %646 = stablehlo.add %641, %645 : tensor<1x1x1x128xf32>
    %647 = stablehlo.rsqrt %646 : tensor<1x1x1x128xf32>
    %648 = stablehlo.divide %647, %646 : tensor<1x1x1x128xf32>
    %649 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %650 = stablehlo.multiply %649, %648 : tensor<1x1x1x128xf32>
    %651 = stablehlo.reshape %arg25 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %652 = stablehlo.multiply %647, %651 : tensor<1x1x1x128xf32>
    %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %654 = stablehlo.multiply %644, %653 : tensor<256x28x28x128xf32>
    %655 = stablehlo.reshape %arg24 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %656 = stablehlo.broadcast_in_dim %655, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %657 = stablehlo.add %654, %656 : tensor<256x28x28x128xf32>
    %658 = stablehlo.convert %657 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %659 = call @relu_85(%658) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %660 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %661 = stablehlo.compare  GT, %658, %660,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %662 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %663 = stablehlo.convert %arg29 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %664 = stablehlo.convolution(%659, %663) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %665 = stablehlo.convert %664 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %666 = stablehlo.multiply %665, %665 : tensor<256x28x28x128xf32>
    %667 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %668 = stablehlo.multiply %667, %665 : tensor<256x28x28x128xf32>
    %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %669 = stablehlo.reduce(%665 init: %cst_38) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %670 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %671 = stablehlo.divide %669, %670 : tensor<128xf32>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %672 = stablehlo.reduce(%666 init: %cst_39) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %673 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %674 = stablehlo.divide %672, %673 : tensor<128xf32>
    %675 = stablehlo.broadcast_in_dim %671, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %676 = stablehlo.broadcast_in_dim %674, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %677 = stablehlo.concatenate %675, %676, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %678 = "stablehlo.all_reduce"(%677) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %679 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %680 = stablehlo.divide %678, %679 : tensor<2x128xf32>
    %681 = stablehlo.slice %680 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %682 = stablehlo.reshape %681 : (tensor<1x128xf32>) -> tensor<128xf32>
    %683 = stablehlo.slice %680 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %684 = stablehlo.reshape %683 : (tensor<1x128xf32>) -> tensor<128xf32>
    %685 = stablehlo.multiply %682, %682 : tensor<128xf32>
    %686 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %687 = stablehlo.multiply %686, %682 : tensor<128xf32>
    %688 = stablehlo.subtract %684, %685 : tensor<128xf32>
    %689 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %690 = stablehlo.maximum %689, %688 : tensor<128xf32>
    %691 = stablehlo.compare  EQ, %688, %690,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %692 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %693 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %694 = stablehlo.select %691, %692, %693 : tensor<128xi1>, tensor<128xf32>
    %695 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %696 = stablehlo.compare  EQ, %695, %690,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %697 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %698 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %699 = stablehlo.select %696, %697, %698 : tensor<128xi1>, tensor<128xf32>
    %700 = stablehlo.divide %694, %699 : tensor<128xf32>
    %701 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %702 = stablehlo.multiply %701, %arg142 : tensor<128xf32>
    %703 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %704 = stablehlo.multiply %703, %682 : tensor<128xf32>
    %705 = stablehlo.add %702, %704 : tensor<128xf32>
    %706 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %707 = stablehlo.multiply %706, %arg143 : tensor<128xf32>
    %708 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %709 = stablehlo.multiply %708, %690 : tensor<128xf32>
    %710 = stablehlo.add %707, %709 : tensor<128xf32>
    %711 = stablehlo.broadcast_in_dim %682, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %712 = stablehlo.broadcast_in_dim %690, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %713 = stablehlo.convert %664 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %714 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %715 = stablehlo.subtract %713, %714 : tensor<256x28x28x128xf32>
    %716 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %717 = stablehlo.add %712, %716 : tensor<1x1x1x128xf32>
    %718 = stablehlo.rsqrt %717 : tensor<1x1x1x128xf32>
    %719 = stablehlo.divide %718, %717 : tensor<1x1x1x128xf32>
    %720 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %721 = stablehlo.multiply %720, %719 : tensor<1x1x1x128xf32>
    %722 = stablehlo.reshape %arg27 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %723 = stablehlo.multiply %718, %722 : tensor<1x1x1x128xf32>
    %724 = stablehlo.broadcast_in_dim %723, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %725 = stablehlo.multiply %715, %724 : tensor<256x28x28x128xf32>
    %726 = stablehlo.reshape %arg26 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %727 = stablehlo.broadcast_in_dim %726, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %728 = stablehlo.add %725, %727 : tensor<256x28x28x128xf32>
    %729 = stablehlo.convert %728 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %730 = stablehlo.add %588, %729 : tensor<256x28x28x128xbf16>
    %731 = call @relu_85(%730) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %732 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %733 = stablehlo.compare  GT, %730, %732,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %734 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %735 = stablehlo.convert %arg34 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xbf16>
    %736 = stablehlo.convolution(%731, %735) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x256xbf16>) -> tensor<256x14x14x256xbf16>
    %737 = stablehlo.convert %736 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %738 = stablehlo.multiply %737, %737 : tensor<256x14x14x256xf32>
    %739 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %740 = stablehlo.multiply %739, %737 : tensor<256x14x14x256xf32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %741 = stablehlo.reduce(%737 init: %cst_40) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_41 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
    %742 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %743 = stablehlo.divide %741, %742 : tensor<256xf32>
    %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %744 = stablehlo.reduce(%738 init: %cst_42) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %745 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %746 = stablehlo.divide %744, %745 : tensor<256xf32>
    %747 = stablehlo.broadcast_in_dim %743, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %748 = stablehlo.broadcast_in_dim %746, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %749 = stablehlo.concatenate %747, %748, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %750 = "stablehlo.all_reduce"(%749) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %751 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %752 = stablehlo.divide %750, %751 : tensor<2x256xf32>
    %753 = stablehlo.slice %752 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %754 = stablehlo.reshape %753 : (tensor<1x256xf32>) -> tensor<256xf32>
    %755 = stablehlo.slice %752 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %756 = stablehlo.reshape %755 : (tensor<1x256xf32>) -> tensor<256xf32>
    %757 = stablehlo.multiply %754, %754 : tensor<256xf32>
    %758 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %759 = stablehlo.multiply %758, %754 : tensor<256xf32>
    %760 = stablehlo.subtract %756, %757 : tensor<256xf32>
    %761 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %762 = stablehlo.maximum %761, %760 : tensor<256xf32>
    %763 = stablehlo.compare  EQ, %760, %762,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %764 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %765 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %766 = stablehlo.select %763, %764, %765 : tensor<256xi1>, tensor<256xf32>
    %767 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %768 = stablehlo.compare  EQ, %767, %762,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %769 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %770 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %771 = stablehlo.select %768, %769, %770 : tensor<256xi1>, tensor<256xf32>
    %772 = stablehlo.divide %766, %771 : tensor<256xf32>
    %773 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %774 = stablehlo.multiply %773, %arg144 : tensor<256xf32>
    %775 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %776 = stablehlo.multiply %775, %754 : tensor<256xf32>
    %777 = stablehlo.add %774, %776 : tensor<256xf32>
    %778 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %779 = stablehlo.multiply %778, %arg145 : tensor<256xf32>
    %780 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %781 = stablehlo.multiply %780, %762 : tensor<256xf32>
    %782 = stablehlo.add %779, %781 : tensor<256xf32>
    %783 = stablehlo.broadcast_in_dim %754, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %784 = stablehlo.broadcast_in_dim %762, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %785 = stablehlo.convert %736 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %786 = stablehlo.broadcast_in_dim %783, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %787 = stablehlo.subtract %785, %786 : tensor<256x14x14x256xf32>
    %788 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %789 = stablehlo.add %784, %788 : tensor<1x1x1x256xf32>
    %790 = stablehlo.rsqrt %789 : tensor<1x1x1x256xf32>
    %791 = stablehlo.divide %790, %789 : tensor<1x1x1x256xf32>
    %792 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %793 = stablehlo.multiply %792, %791 : tensor<1x1x1x256xf32>
    %794 = stablehlo.reshape %arg31 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %795 = stablehlo.multiply %790, %794 : tensor<1x1x1x256xf32>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %797 = stablehlo.multiply %787, %796 : tensor<256x14x14x256xf32>
    %798 = stablehlo.reshape %arg30 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %799 = stablehlo.broadcast_in_dim %798, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %800 = stablehlo.add %797, %799 : tensor<256x14x14x256xf32>
    %801 = stablehlo.convert %800 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %802 = call @relu_133(%801) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %803 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %804 = stablehlo.compare  GT, %801, %803,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %805 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %806 = stablehlo.convert %arg35 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %807 = stablehlo.convolution(%802, %806) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %808 = stablehlo.convert %807 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %809 = stablehlo.multiply %808, %808 : tensor<256x14x14x256xf32>
    %810 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %811 = stablehlo.multiply %810, %808 : tensor<256x14x14x256xf32>
    %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %812 = stablehlo.reduce(%808 init: %cst_43) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %813 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %814 = stablehlo.divide %812, %813 : tensor<256xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %815 = stablehlo.reduce(%809 init: %cst_44) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %816 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %817 = stablehlo.divide %815, %816 : tensor<256xf32>
    %818 = stablehlo.broadcast_in_dim %814, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %819 = stablehlo.broadcast_in_dim %817, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %820 = stablehlo.concatenate %818, %819, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %821 = "stablehlo.all_reduce"(%820) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %822 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %823 = stablehlo.divide %821, %822 : tensor<2x256xf32>
    %824 = stablehlo.slice %823 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %825 = stablehlo.reshape %824 : (tensor<1x256xf32>) -> tensor<256xf32>
    %826 = stablehlo.slice %823 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %827 = stablehlo.reshape %826 : (tensor<1x256xf32>) -> tensor<256xf32>
    %828 = stablehlo.multiply %825, %825 : tensor<256xf32>
    %829 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %830 = stablehlo.multiply %829, %825 : tensor<256xf32>
    %831 = stablehlo.subtract %827, %828 : tensor<256xf32>
    %832 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %833 = stablehlo.maximum %832, %831 : tensor<256xf32>
    %834 = stablehlo.compare  EQ, %831, %833,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %835 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %836 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %837 = stablehlo.select %834, %835, %836 : tensor<256xi1>, tensor<256xf32>
    %838 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %839 = stablehlo.compare  EQ, %838, %833,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %840 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %841 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %842 = stablehlo.select %839, %840, %841 : tensor<256xi1>, tensor<256xf32>
    %843 = stablehlo.divide %837, %842 : tensor<256xf32>
    %844 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %845 = stablehlo.multiply %844, %arg146 : tensor<256xf32>
    %846 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %847 = stablehlo.multiply %846, %825 : tensor<256xf32>
    %848 = stablehlo.add %845, %847 : tensor<256xf32>
    %849 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %850 = stablehlo.multiply %849, %arg147 : tensor<256xf32>
    %851 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %852 = stablehlo.multiply %851, %833 : tensor<256xf32>
    %853 = stablehlo.add %850, %852 : tensor<256xf32>
    %854 = stablehlo.broadcast_in_dim %825, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %855 = stablehlo.broadcast_in_dim %833, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %856 = stablehlo.convert %807 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %857 = stablehlo.broadcast_in_dim %854, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %858 = stablehlo.subtract %856, %857 : tensor<256x14x14x256xf32>
    %859 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %860 = stablehlo.add %855, %859 : tensor<1x1x1x256xf32>
    %861 = stablehlo.rsqrt %860 : tensor<1x1x1x256xf32>
    %862 = stablehlo.divide %861, %860 : tensor<1x1x1x256xf32>
    %863 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %864 = stablehlo.multiply %863, %862 : tensor<1x1x1x256xf32>
    %865 = stablehlo.reshape %arg33 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %866 = stablehlo.multiply %861, %865 : tensor<1x1x1x256xf32>
    %867 = stablehlo.broadcast_in_dim %866, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %868 = stablehlo.multiply %858, %867 : tensor<256x14x14x256xf32>
    %869 = stablehlo.reshape %arg32 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %870 = stablehlo.broadcast_in_dim %869, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %871 = stablehlo.add %868, %870 : tensor<256x14x14x256xf32>
    %872 = stablehlo.convert %871 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %873 = stablehlo.convert %arg36 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xbf16>
    %874 = stablehlo.convolution(%731, %873) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x256xbf16>) -> tensor<256x14x14x256xbf16>
    %875 = stablehlo.convert %874 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %876 = stablehlo.multiply %875, %875 : tensor<256x14x14x256xf32>
    %877 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %878 = stablehlo.multiply %877, %875 : tensor<256x14x14x256xf32>
    %cst_45 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %879 = stablehlo.reduce(%875 init: %cst_45) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %880 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %881 = stablehlo.divide %879, %880 : tensor<256xf32>
    %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %882 = stablehlo.reduce(%876 init: %cst_46) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %883 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %884 = stablehlo.divide %882, %883 : tensor<256xf32>
    %885 = stablehlo.broadcast_in_dim %881, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %886 = stablehlo.broadcast_in_dim %884, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %887 = stablehlo.concatenate %885, %886, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %888 = "stablehlo.all_reduce"(%887) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %889 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %890 = stablehlo.divide %888, %889 : tensor<2x256xf32>
    %891 = stablehlo.slice %890 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %892 = stablehlo.reshape %891 : (tensor<1x256xf32>) -> tensor<256xf32>
    %893 = stablehlo.slice %890 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %894 = stablehlo.reshape %893 : (tensor<1x256xf32>) -> tensor<256xf32>
    %895 = stablehlo.multiply %892, %892 : tensor<256xf32>
    %896 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %897 = stablehlo.multiply %896, %892 : tensor<256xf32>
    %898 = stablehlo.subtract %894, %895 : tensor<256xf32>
    %899 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %900 = stablehlo.maximum %899, %898 : tensor<256xf32>
    %901 = stablehlo.compare  EQ, %898, %900,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %902 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %903 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %904 = stablehlo.select %901, %902, %903 : tensor<256xi1>, tensor<256xf32>
    %905 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %906 = stablehlo.compare  EQ, %905, %900,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %907 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %908 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %909 = stablehlo.select %906, %907, %908 : tensor<256xi1>, tensor<256xf32>
    %910 = stablehlo.divide %904, %909 : tensor<256xf32>
    %911 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %912 = stablehlo.multiply %911, %arg148 : tensor<256xf32>
    %913 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %914 = stablehlo.multiply %913, %892 : tensor<256xf32>
    %915 = stablehlo.add %912, %914 : tensor<256xf32>
    %916 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %917 = stablehlo.multiply %916, %arg149 : tensor<256xf32>
    %918 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %919 = stablehlo.multiply %918, %900 : tensor<256xf32>
    %920 = stablehlo.add %917, %919 : tensor<256xf32>
    %921 = stablehlo.broadcast_in_dim %892, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %922 = stablehlo.broadcast_in_dim %900, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %923 = stablehlo.convert %874 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %924 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %925 = stablehlo.subtract %923, %924 : tensor<256x14x14x256xf32>
    %926 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %927 = stablehlo.add %922, %926 : tensor<1x1x1x256xf32>
    %928 = stablehlo.rsqrt %927 : tensor<1x1x1x256xf32>
    %929 = stablehlo.divide %928, %927 : tensor<1x1x1x256xf32>
    %930 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %931 = stablehlo.multiply %930, %929 : tensor<1x1x1x256xf32>
    %932 = stablehlo.reshape %arg38 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %933 = stablehlo.multiply %928, %932 : tensor<1x1x1x256xf32>
    %934 = stablehlo.broadcast_in_dim %933, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %935 = stablehlo.multiply %925, %934 : tensor<256x14x14x256xf32>
    %936 = stablehlo.reshape %arg37 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %937 = stablehlo.broadcast_in_dim %936, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %938 = stablehlo.add %935, %937 : tensor<256x14x14x256xf32>
    %939 = stablehlo.convert %938 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %940 = stablehlo.add %939, %872 : tensor<256x14x14x256xbf16>
    %941 = call @relu_133(%940) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %942 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %943 = stablehlo.compare  GT, %940, %942,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %944 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %945 = stablehlo.convert %arg43 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %946 = stablehlo.convolution(%941, %945) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %947 = stablehlo.convert %946 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %948 = stablehlo.multiply %947, %947 : tensor<256x14x14x256xf32>
    %949 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %950 = stablehlo.multiply %949, %947 : tensor<256x14x14x256xf32>
    %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %951 = stablehlo.reduce(%947 init: %cst_47) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %952 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %953 = stablehlo.divide %951, %952 : tensor<256xf32>
    %cst_48 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %954 = stablehlo.reduce(%948 init: %cst_48) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %955 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %956 = stablehlo.divide %954, %955 : tensor<256xf32>
    %957 = stablehlo.broadcast_in_dim %953, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %958 = stablehlo.broadcast_in_dim %956, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %959 = stablehlo.concatenate %957, %958, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %960 = "stablehlo.all_reduce"(%959) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %961 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %962 = stablehlo.divide %960, %961 : tensor<2x256xf32>
    %963 = stablehlo.slice %962 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %964 = stablehlo.reshape %963 : (tensor<1x256xf32>) -> tensor<256xf32>
    %965 = stablehlo.slice %962 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %966 = stablehlo.reshape %965 : (tensor<1x256xf32>) -> tensor<256xf32>
    %967 = stablehlo.multiply %964, %964 : tensor<256xf32>
    %968 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %969 = stablehlo.multiply %968, %964 : tensor<256xf32>
    %970 = stablehlo.subtract %966, %967 : tensor<256xf32>
    %971 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %972 = stablehlo.maximum %971, %970 : tensor<256xf32>
    %973 = stablehlo.compare  EQ, %970, %972,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %974 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %975 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %976 = stablehlo.select %973, %974, %975 : tensor<256xi1>, tensor<256xf32>
    %977 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %978 = stablehlo.compare  EQ, %977, %972,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %979 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %980 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %981 = stablehlo.select %978, %979, %980 : tensor<256xi1>, tensor<256xf32>
    %982 = stablehlo.divide %976, %981 : tensor<256xf32>
    %983 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %984 = stablehlo.multiply %983, %arg150 : tensor<256xf32>
    %985 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %986 = stablehlo.multiply %985, %964 : tensor<256xf32>
    %987 = stablehlo.add %984, %986 : tensor<256xf32>
    %988 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %989 = stablehlo.multiply %988, %arg151 : tensor<256xf32>
    %990 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %991 = stablehlo.multiply %990, %972 : tensor<256xf32>
    %992 = stablehlo.add %989, %991 : tensor<256xf32>
    %993 = stablehlo.broadcast_in_dim %964, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %994 = stablehlo.broadcast_in_dim %972, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %995 = stablehlo.convert %946 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %996 = stablehlo.broadcast_in_dim %993, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %997 = stablehlo.subtract %995, %996 : tensor<256x14x14x256xf32>
    %998 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %999 = stablehlo.add %994, %998 : tensor<1x1x1x256xf32>
    %1000 = stablehlo.rsqrt %999 : tensor<1x1x1x256xf32>
    %1001 = stablehlo.divide %1000, %999 : tensor<1x1x1x256xf32>
    %1002 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1003 = stablehlo.multiply %1002, %1001 : tensor<1x1x1x256xf32>
    %1004 = stablehlo.reshape %arg40 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1005 = stablehlo.multiply %1000, %1004 : tensor<1x1x1x256xf32>
    %1006 = stablehlo.broadcast_in_dim %1005, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1007 = stablehlo.multiply %997, %1006 : tensor<256x14x14x256xf32>
    %1008 = stablehlo.reshape %arg39 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1009 = stablehlo.broadcast_in_dim %1008, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1010 = stablehlo.add %1007, %1009 : tensor<256x14x14x256xf32>
    %1011 = stablehlo.convert %1010 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1012 = call @relu_133(%1011) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1013 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1014 = stablehlo.compare  GT, %1011, %1013,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1015 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1016 = stablehlo.convert %arg44 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1017 = stablehlo.convolution(%1012, %1016) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1018 = stablehlo.convert %1017 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1019 = stablehlo.multiply %1018, %1018 : tensor<256x14x14x256xf32>
    %1020 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1021 = stablehlo.multiply %1020, %1018 : tensor<256x14x14x256xf32>
    %cst_49 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1022 = stablehlo.reduce(%1018 init: %cst_49) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1023 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1024 = stablehlo.divide %1022, %1023 : tensor<256xf32>
    %cst_50 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1025 = stablehlo.reduce(%1019 init: %cst_50) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1026 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1027 = stablehlo.divide %1025, %1026 : tensor<256xf32>
    %1028 = stablehlo.broadcast_in_dim %1024, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1029 = stablehlo.broadcast_in_dim %1027, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1030 = stablehlo.concatenate %1028, %1029, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1031 = "stablehlo.all_reduce"(%1030) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1032 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1033 = stablehlo.divide %1031, %1032 : tensor<2x256xf32>
    %1034 = stablehlo.slice %1033 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1035 = stablehlo.reshape %1034 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1036 = stablehlo.slice %1033 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1037 = stablehlo.reshape %1036 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1038 = stablehlo.multiply %1035, %1035 : tensor<256xf32>
    %1039 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1040 = stablehlo.multiply %1039, %1035 : tensor<256xf32>
    %1041 = stablehlo.subtract %1037, %1038 : tensor<256xf32>
    %1042 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1043 = stablehlo.maximum %1042, %1041 : tensor<256xf32>
    %1044 = stablehlo.compare  EQ, %1041, %1043,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1045 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1046 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1047 = stablehlo.select %1044, %1045, %1046 : tensor<256xi1>, tensor<256xf32>
    %1048 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1049 = stablehlo.compare  EQ, %1048, %1043,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1050 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1051 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1052 = stablehlo.select %1049, %1050, %1051 : tensor<256xi1>, tensor<256xf32>
    %1053 = stablehlo.divide %1047, %1052 : tensor<256xf32>
    %1054 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1055 = stablehlo.multiply %1054, %arg152 : tensor<256xf32>
    %1056 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1057 = stablehlo.multiply %1056, %1035 : tensor<256xf32>
    %1058 = stablehlo.add %1055, %1057 : tensor<256xf32>
    %1059 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1060 = stablehlo.multiply %1059, %arg153 : tensor<256xf32>
    %1061 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1062 = stablehlo.multiply %1061, %1043 : tensor<256xf32>
    %1063 = stablehlo.add %1060, %1062 : tensor<256xf32>
    %1064 = stablehlo.broadcast_in_dim %1035, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1065 = stablehlo.broadcast_in_dim %1043, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1066 = stablehlo.convert %1017 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1067 = stablehlo.broadcast_in_dim %1064, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1068 = stablehlo.subtract %1066, %1067 : tensor<256x14x14x256xf32>
    %1069 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1070 = stablehlo.add %1065, %1069 : tensor<1x1x1x256xf32>
    %1071 = stablehlo.rsqrt %1070 : tensor<1x1x1x256xf32>
    %1072 = stablehlo.divide %1071, %1070 : tensor<1x1x1x256xf32>
    %1073 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1074 = stablehlo.multiply %1073, %1072 : tensor<1x1x1x256xf32>
    %1075 = stablehlo.reshape %arg42 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1076 = stablehlo.multiply %1071, %1075 : tensor<1x1x1x256xf32>
    %1077 = stablehlo.broadcast_in_dim %1076, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1078 = stablehlo.multiply %1068, %1077 : tensor<256x14x14x256xf32>
    %1079 = stablehlo.reshape %arg41 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1080 = stablehlo.broadcast_in_dim %1079, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1081 = stablehlo.add %1078, %1080 : tensor<256x14x14x256xf32>
    %1082 = stablehlo.convert %1081 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1083 = stablehlo.add %941, %1082 : tensor<256x14x14x256xbf16>
    %1084 = call @relu_133(%1083) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1085 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1086 = stablehlo.compare  GT, %1083, %1085,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1087 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1088 = stablehlo.convert %arg49 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xbf16>
    %1089 = stablehlo.convolution(%1084, %1088) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1090 = stablehlo.convert %1089 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1091 = stablehlo.multiply %1090, %1090 : tensor<256x7x7x512xf32>
    %1092 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %1093 = stablehlo.multiply %1092, %1090 : tensor<256x7x7x512xf32>
    %cst_51 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1094 = stablehlo.reduce(%1090 init: %cst_51) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_52 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
    %1095 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1096 = stablehlo.divide %1094, %1095 : tensor<512xf32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1097 = stablehlo.reduce(%1091 init: %cst_53) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1098 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1099 = stablehlo.divide %1097, %1098 : tensor<512xf32>
    %1100 = stablehlo.broadcast_in_dim %1096, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1101 = stablehlo.broadcast_in_dim %1099, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1102 = stablehlo.concatenate %1100, %1101, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %1103 = "stablehlo.all_reduce"(%1102) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1104 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1105 = stablehlo.divide %1103, %1104 : tensor<2x512xf32>
    %1106 = stablehlo.slice %1105 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1107 = stablehlo.reshape %1106 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1108 = stablehlo.slice %1105 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1109 = stablehlo.reshape %1108 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1110 = stablehlo.multiply %1107, %1107 : tensor<512xf32>
    %1111 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1112 = stablehlo.multiply %1111, %1107 : tensor<512xf32>
    %1113 = stablehlo.subtract %1109, %1110 : tensor<512xf32>
    %1114 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1115 = stablehlo.maximum %1114, %1113 : tensor<512xf32>
    %1116 = stablehlo.compare  EQ, %1113, %1115,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1117 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1118 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1119 = stablehlo.select %1116, %1117, %1118 : tensor<512xi1>, tensor<512xf32>
    %1120 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1121 = stablehlo.compare  EQ, %1120, %1115,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1122 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1123 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1124 = stablehlo.select %1121, %1122, %1123 : tensor<512xi1>, tensor<512xf32>
    %1125 = stablehlo.divide %1119, %1124 : tensor<512xf32>
    %1126 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1127 = stablehlo.multiply %1126, %arg154 : tensor<512xf32>
    %1128 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1129 = stablehlo.multiply %1128, %1107 : tensor<512xf32>
    %1130 = stablehlo.add %1127, %1129 : tensor<512xf32>
    %1131 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1132 = stablehlo.multiply %1131, %arg155 : tensor<512xf32>
    %1133 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1134 = stablehlo.multiply %1133, %1115 : tensor<512xf32>
    %1135 = stablehlo.add %1132, %1134 : tensor<512xf32>
    %1136 = stablehlo.broadcast_in_dim %1107, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1137 = stablehlo.broadcast_in_dim %1115, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1138 = stablehlo.convert %1089 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1139 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1140 = stablehlo.subtract %1138, %1139 : tensor<256x7x7x512xf32>
    %1141 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1142 = stablehlo.add %1137, %1141 : tensor<1x1x1x512xf32>
    %1143 = stablehlo.rsqrt %1142 : tensor<1x1x1x512xf32>
    %1144 = stablehlo.divide %1143, %1142 : tensor<1x1x1x512xf32>
    %1145 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1146 = stablehlo.multiply %1145, %1144 : tensor<1x1x1x512xf32>
    %1147 = stablehlo.reshape %arg46 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1148 = stablehlo.multiply %1143, %1147 : tensor<1x1x1x512xf32>
    %1149 = stablehlo.broadcast_in_dim %1148, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1150 = stablehlo.multiply %1140, %1149 : tensor<256x7x7x512xf32>
    %1151 = stablehlo.reshape %arg45 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1152 = stablehlo.broadcast_in_dim %1151, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1153 = stablehlo.add %1150, %1152 : tensor<256x7x7x512xf32>
    %1154 = stablehlo.convert %1153 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1155 = call @relu_181(%1154) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1156 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1157 = stablehlo.compare  GT, %1154, %1156,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %1158 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1159 = stablehlo.convert %arg50 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %1160 = stablehlo.convolution(%1155, %1159) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1161 = stablehlo.convert %1160 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1162 = stablehlo.multiply %1161, %1161 : tensor<256x7x7x512xf32>
    %1163 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %1164 = stablehlo.multiply %1163, %1161 : tensor<256x7x7x512xf32>
    %cst_54 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1165 = stablehlo.reduce(%1161 init: %cst_54) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1166 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1167 = stablehlo.divide %1165, %1166 : tensor<512xf32>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1168 = stablehlo.reduce(%1162 init: %cst_55) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1169 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1170 = stablehlo.divide %1168, %1169 : tensor<512xf32>
    %1171 = stablehlo.broadcast_in_dim %1167, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1172 = stablehlo.broadcast_in_dim %1170, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1173 = stablehlo.concatenate %1171, %1172, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %1174 = "stablehlo.all_reduce"(%1173) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1175 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1176 = stablehlo.divide %1174, %1175 : tensor<2x512xf32>
    %1177 = stablehlo.slice %1176 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1178 = stablehlo.reshape %1177 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1179 = stablehlo.slice %1176 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1180 = stablehlo.reshape %1179 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1181 = stablehlo.multiply %1178, %1178 : tensor<512xf32>
    %1182 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1183 = stablehlo.multiply %1182, %1178 : tensor<512xf32>
    %1184 = stablehlo.subtract %1180, %1181 : tensor<512xf32>
    %1185 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1186 = stablehlo.maximum %1185, %1184 : tensor<512xf32>
    %1187 = stablehlo.compare  EQ, %1184, %1186,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1188 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1189 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1190 = stablehlo.select %1187, %1188, %1189 : tensor<512xi1>, tensor<512xf32>
    %1191 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1192 = stablehlo.compare  EQ, %1191, %1186,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1193 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1194 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1195 = stablehlo.select %1192, %1193, %1194 : tensor<512xi1>, tensor<512xf32>
    %1196 = stablehlo.divide %1190, %1195 : tensor<512xf32>
    %1197 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1198 = stablehlo.multiply %1197, %arg156 : tensor<512xf32>
    %1199 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1200 = stablehlo.multiply %1199, %1178 : tensor<512xf32>
    %1201 = stablehlo.add %1198, %1200 : tensor<512xf32>
    %1202 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1203 = stablehlo.multiply %1202, %arg157 : tensor<512xf32>
    %1204 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1205 = stablehlo.multiply %1204, %1186 : tensor<512xf32>
    %1206 = stablehlo.add %1203, %1205 : tensor<512xf32>
    %1207 = stablehlo.broadcast_in_dim %1178, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1208 = stablehlo.broadcast_in_dim %1186, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1209 = stablehlo.convert %1160 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1210 = stablehlo.broadcast_in_dim %1207, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1211 = stablehlo.subtract %1209, %1210 : tensor<256x7x7x512xf32>
    %1212 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1213 = stablehlo.add %1208, %1212 : tensor<1x1x1x512xf32>
    %1214 = stablehlo.rsqrt %1213 : tensor<1x1x1x512xf32>
    %1215 = stablehlo.divide %1214, %1213 : tensor<1x1x1x512xf32>
    %1216 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1217 = stablehlo.multiply %1216, %1215 : tensor<1x1x1x512xf32>
    %1218 = stablehlo.reshape %arg48 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1219 = stablehlo.multiply %1214, %1218 : tensor<1x1x1x512xf32>
    %1220 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1221 = stablehlo.multiply %1211, %1220 : tensor<256x7x7x512xf32>
    %1222 = stablehlo.reshape %arg47 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1223 = stablehlo.broadcast_in_dim %1222, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1224 = stablehlo.add %1221, %1223 : tensor<256x7x7x512xf32>
    %1225 = stablehlo.convert %1224 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1226 = stablehlo.convert %arg51 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xbf16>
    %1227 = stablehlo.convolution(%1084, %1226) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1228 = stablehlo.convert %1227 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1229 = stablehlo.multiply %1228, %1228 : tensor<256x7x7x512xf32>
    %1230 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %1231 = stablehlo.multiply %1230, %1228 : tensor<256x7x7x512xf32>
    %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1232 = stablehlo.reduce(%1228 init: %cst_56) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1233 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1234 = stablehlo.divide %1232, %1233 : tensor<512xf32>
    %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1235 = stablehlo.reduce(%1229 init: %cst_57) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1236 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1237 = stablehlo.divide %1235, %1236 : tensor<512xf32>
    %1238 = stablehlo.broadcast_in_dim %1234, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1239 = stablehlo.broadcast_in_dim %1237, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1240 = stablehlo.concatenate %1238, %1239, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %1241 = "stablehlo.all_reduce"(%1240) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1242 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1243 = stablehlo.divide %1241, %1242 : tensor<2x512xf32>
    %1244 = stablehlo.slice %1243 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1245 = stablehlo.reshape %1244 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1246 = stablehlo.slice %1243 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1247 = stablehlo.reshape %1246 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1248 = stablehlo.multiply %1245, %1245 : tensor<512xf32>
    %1249 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1250 = stablehlo.multiply %1249, %1245 : tensor<512xf32>
    %1251 = stablehlo.subtract %1247, %1248 : tensor<512xf32>
    %1252 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1253 = stablehlo.maximum %1252, %1251 : tensor<512xf32>
    %1254 = stablehlo.compare  EQ, %1251, %1253,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1255 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1256 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1257 = stablehlo.select %1254, %1255, %1256 : tensor<512xi1>, tensor<512xf32>
    %1258 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1259 = stablehlo.compare  EQ, %1258, %1253,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1260 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1261 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1262 = stablehlo.select %1259, %1260, %1261 : tensor<512xi1>, tensor<512xf32>
    %1263 = stablehlo.divide %1257, %1262 : tensor<512xf32>
    %1264 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1265 = stablehlo.multiply %1264, %arg158 : tensor<512xf32>
    %1266 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1267 = stablehlo.multiply %1266, %1245 : tensor<512xf32>
    %1268 = stablehlo.add %1265, %1267 : tensor<512xf32>
    %1269 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1270 = stablehlo.multiply %1269, %arg159 : tensor<512xf32>
    %1271 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1272 = stablehlo.multiply %1271, %1253 : tensor<512xf32>
    %1273 = stablehlo.add %1270, %1272 : tensor<512xf32>
    %1274 = stablehlo.broadcast_in_dim %1245, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1275 = stablehlo.broadcast_in_dim %1253, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1276 = stablehlo.convert %1227 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1277 = stablehlo.broadcast_in_dim %1274, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1278 = stablehlo.subtract %1276, %1277 : tensor<256x7x7x512xf32>
    %1279 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1280 = stablehlo.add %1275, %1279 : tensor<1x1x1x512xf32>
    %1281 = stablehlo.rsqrt %1280 : tensor<1x1x1x512xf32>
    %1282 = stablehlo.divide %1281, %1280 : tensor<1x1x1x512xf32>
    %1283 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1284 = stablehlo.multiply %1283, %1282 : tensor<1x1x1x512xf32>
    %1285 = stablehlo.reshape %arg53 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1286 = stablehlo.multiply %1281, %1285 : tensor<1x1x1x512xf32>
    %1287 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1288 = stablehlo.multiply %1278, %1287 : tensor<256x7x7x512xf32>
    %1289 = stablehlo.reshape %arg52 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1290 = stablehlo.broadcast_in_dim %1289, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1291 = stablehlo.add %1288, %1290 : tensor<256x7x7x512xf32>
    %1292 = stablehlo.convert %1291 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1293 = stablehlo.add %1292, %1225 : tensor<256x7x7x512xbf16>
    %1294 = call @relu_181(%1293) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1295 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1296 = stablehlo.compare  GT, %1293, %1295,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %1297 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1298 = stablehlo.convert %arg58 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %1299 = stablehlo.convolution(%1294, %1298) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1300 = stablehlo.convert %1299 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1301 = stablehlo.multiply %1300, %1300 : tensor<256x7x7x512xf32>
    %1302 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %1303 = stablehlo.multiply %1302, %1300 : tensor<256x7x7x512xf32>
    %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1304 = stablehlo.reduce(%1300 init: %cst_58) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1305 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1306 = stablehlo.divide %1304, %1305 : tensor<512xf32>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1307 = stablehlo.reduce(%1301 init: %cst_59) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1308 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1309 = stablehlo.divide %1307, %1308 : tensor<512xf32>
    %1310 = stablehlo.broadcast_in_dim %1306, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1311 = stablehlo.broadcast_in_dim %1309, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1312 = stablehlo.concatenate %1310, %1311, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %1313 = "stablehlo.all_reduce"(%1312) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1314 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1315 = stablehlo.divide %1313, %1314 : tensor<2x512xf32>
    %1316 = stablehlo.slice %1315 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1317 = stablehlo.reshape %1316 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1318 = stablehlo.slice %1315 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1319 = stablehlo.reshape %1318 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1320 = stablehlo.multiply %1317, %1317 : tensor<512xf32>
    %1321 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1322 = stablehlo.multiply %1321, %1317 : tensor<512xf32>
    %1323 = stablehlo.subtract %1319, %1320 : tensor<512xf32>
    %1324 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1325 = stablehlo.maximum %1324, %1323 : tensor<512xf32>
    %1326 = stablehlo.compare  EQ, %1323, %1325,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1327 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1328 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1329 = stablehlo.select %1326, %1327, %1328 : tensor<512xi1>, tensor<512xf32>
    %1330 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1331 = stablehlo.compare  EQ, %1330, %1325,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1332 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1333 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1334 = stablehlo.select %1331, %1332, %1333 : tensor<512xi1>, tensor<512xf32>
    %1335 = stablehlo.divide %1329, %1334 : tensor<512xf32>
    %1336 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1337 = stablehlo.multiply %1336, %arg160 : tensor<512xf32>
    %1338 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1339 = stablehlo.multiply %1338, %1317 : tensor<512xf32>
    %1340 = stablehlo.add %1337, %1339 : tensor<512xf32>
    %1341 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1342 = stablehlo.multiply %1341, %arg161 : tensor<512xf32>
    %1343 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1344 = stablehlo.multiply %1343, %1325 : tensor<512xf32>
    %1345 = stablehlo.add %1342, %1344 : tensor<512xf32>
    %1346 = stablehlo.broadcast_in_dim %1317, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1347 = stablehlo.broadcast_in_dim %1325, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1348 = stablehlo.convert %1299 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1349 = stablehlo.broadcast_in_dim %1346, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1350 = stablehlo.subtract %1348, %1349 : tensor<256x7x7x512xf32>
    %1351 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1352 = stablehlo.add %1347, %1351 : tensor<1x1x1x512xf32>
    %1353 = stablehlo.rsqrt %1352 : tensor<1x1x1x512xf32>
    %1354 = stablehlo.divide %1353, %1352 : tensor<1x1x1x512xf32>
    %1355 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1356 = stablehlo.multiply %1355, %1354 : tensor<1x1x1x512xf32>
    %1357 = stablehlo.reshape %arg55 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1358 = stablehlo.multiply %1353, %1357 : tensor<1x1x1x512xf32>
    %1359 = stablehlo.broadcast_in_dim %1358, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1360 = stablehlo.multiply %1350, %1359 : tensor<256x7x7x512xf32>
    %1361 = stablehlo.reshape %arg54 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1362 = stablehlo.broadcast_in_dim %1361, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1363 = stablehlo.add %1360, %1362 : tensor<256x7x7x512xf32>
    %1364 = stablehlo.convert %1363 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1365 = call @relu_181(%1364) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1366 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1367 = stablehlo.compare  GT, %1364, %1366,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %1368 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1369 = stablehlo.convert %arg59 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %1370 = stablehlo.convolution(%1365, %1369) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1371 = stablehlo.convert %1370 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1372 = stablehlo.multiply %1371, %1371 : tensor<256x7x7x512xf32>
    %1373 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %1374 = stablehlo.multiply %1373, %1371 : tensor<256x7x7x512xf32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1375 = stablehlo.reduce(%1371 init: %cst_60) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1376 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1377 = stablehlo.divide %1375, %1376 : tensor<512xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1378 = stablehlo.reduce(%1372 init: %cst_61) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1379 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1380 = stablehlo.divide %1378, %1379 : tensor<512xf32>
    %1381 = stablehlo.broadcast_in_dim %1377, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1382 = stablehlo.broadcast_in_dim %1380, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1383 = stablehlo.concatenate %1381, %1382, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %1384 = "stablehlo.all_reduce"(%1383) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1385 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1386 = stablehlo.divide %1384, %1385 : tensor<2x512xf32>
    %1387 = stablehlo.slice %1386 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1388 = stablehlo.reshape %1387 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1389 = stablehlo.slice %1386 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1390 = stablehlo.reshape %1389 : (tensor<1x512xf32>) -> tensor<512xf32>
    %1391 = stablehlo.multiply %1388, %1388 : tensor<512xf32>
    %1392 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1393 = stablehlo.multiply %1392, %1388 : tensor<512xf32>
    %1394 = stablehlo.subtract %1390, %1391 : tensor<512xf32>
    %1395 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1396 = stablehlo.maximum %1395, %1394 : tensor<512xf32>
    %1397 = stablehlo.compare  EQ, %1394, %1396,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1398 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1399 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1400 = stablehlo.select %1397, %1398, %1399 : tensor<512xi1>, tensor<512xf32>
    %1401 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1402 = stablehlo.compare  EQ, %1401, %1396,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %1403 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1404 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1405 = stablehlo.select %1402, %1403, %1404 : tensor<512xi1>, tensor<512xf32>
    %1406 = stablehlo.divide %1400, %1405 : tensor<512xf32>
    %1407 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1408 = stablehlo.multiply %1407, %arg162 : tensor<512xf32>
    %1409 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1410 = stablehlo.multiply %1409, %1388 : tensor<512xf32>
    %1411 = stablehlo.add %1408, %1410 : tensor<512xf32>
    %1412 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1413 = stablehlo.multiply %1412, %arg163 : tensor<512xf32>
    %1414 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1415 = stablehlo.multiply %1414, %1396 : tensor<512xf32>
    %1416 = stablehlo.add %1413, %1415 : tensor<512xf32>
    %1417 = stablehlo.broadcast_in_dim %1388, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1418 = stablehlo.broadcast_in_dim %1396, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1419 = stablehlo.convert %1370 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1420 = stablehlo.broadcast_in_dim %1417, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1421 = stablehlo.subtract %1419, %1420 : tensor<256x7x7x512xf32>
    %1422 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1423 = stablehlo.add %1418, %1422 : tensor<1x1x1x512xf32>
    %1424 = stablehlo.rsqrt %1423 : tensor<1x1x1x512xf32>
    %1425 = stablehlo.divide %1424, %1423 : tensor<1x1x1x512xf32>
    %1426 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1427 = stablehlo.multiply %1426, %1425 : tensor<1x1x1x512xf32>
    %1428 = stablehlo.reshape %arg57 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1429 = stablehlo.multiply %1424, %1428 : tensor<1x1x1x512xf32>
    %1430 = stablehlo.broadcast_in_dim %1429, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1431 = stablehlo.multiply %1421, %1430 : tensor<256x7x7x512xf32>
    %1432 = stablehlo.reshape %arg56 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1433 = stablehlo.broadcast_in_dim %1432, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1434 = stablehlo.add %1431, %1433 : tensor<256x7x7x512xf32>
    %1435 = stablehlo.convert %1434 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1436 = stablehlo.add %1294, %1435 : tensor<256x7x7x512xbf16>
    %1437 = call @relu_181(%1436) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1438 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1439 = stablehlo.compare  GT, %1436, %1438,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %1440 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1441 = stablehlo.convert %1437 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_62 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1442 = stablehlo.reduce(%1441 init: %cst_62) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<256x512xf32>
    %cst_63 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
    %1443 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
    %1444 = stablehlo.divide %1442, %1443 : tensor<256x512xf32>
    %1445 = stablehlo.convert %1444 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %1446 = stablehlo.convert %arg2 : (tensor<512x1000xf32>) -> tensor<512x1000xbf16>
    %1447 = stablehlo.convert %arg1 : (tensor<1000xf32>) -> tensor<1000xbf16>
    %1448 = stablehlo.dot_general %1445, %1446, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xbf16>, tensor<512x1000xbf16>) -> tensor<256x1000xbf16>
    %1449 = stablehlo.reshape %1447 : (tensor<1000xbf16>) -> tensor<1x1000xbf16>
    %1450 = stablehlo.broadcast_in_dim %1449, dims = [0, 1] : (tensor<1x1000xbf16>) -> tensor<256x1000xbf16>
    %1451 = stablehlo.add %1448, %1450 : tensor<256x1000xbf16>
    %1452 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %1453 = stablehlo.iota dim = 0 : tensor<1000xi32>
    %1454 = stablehlo.reshape %1453 : (tensor<1000xi32>) -> tensor<1x1000xi32>
    %1455 = stablehlo.broadcast_in_dim %1452, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
    %1456 = stablehlo.broadcast_in_dim %1454, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
    %1457 = stablehlo.compare  EQ, %1455, %1456,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
    %1458 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
    %1459 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
    %1460 = stablehlo.select %1457, %1458, %1459 : tensor<256x1000xi1>, tensor<256x1000xf32>
    %1461 = stablehlo.convert %1460 : tensor<256x1000xf32>
    %1462:3 = call @log_softmax(%1451) : (tensor<256x1000xbf16>) -> (tensor<256x1000xbf16>, tensor<256x1000xbf16>, tensor<256x1xbf16>)
    %1463 = stablehlo.convert %1462#0 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
    %1464 = stablehlo.multiply %1461, %1463 : tensor<256x1000xf32>
    %cst_64 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1465 = stablehlo.reduce(%1464 init: %cst_64) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %1466 = stablehlo.negate %1465 : tensor<256xf32>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1467 = stablehlo.reduce(%1466 init: %cst_65) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %cst_66 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
    %1468 = stablehlo.divide %1467, %cst_66 : tensor<f32>
    %1469 = stablehlo.multiply %arg2, %arg2 : tensor<512x1000xf32>
    %1470 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %1471 = stablehlo.multiply %1470, %arg2 : tensor<512x1000xf32>
    %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1472 = stablehlo.reduce(%1469 init: %cst_67) applies stablehlo.add across dimensions = [0, 1] : (tensor<512x1000xf32>, tensor<f32>) -> tensor<f32>
    %1473 = stablehlo.add %cst_12, %1472 : tensor<f32>
    %1474 = stablehlo.multiply %arg7, %arg7 : tensor<3x3x64x64xf32>
    %1475 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %1476 = stablehlo.multiply %1475, %arg7 : tensor<3x3x64x64xf32>
    %cst_68 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1477 = stablehlo.reduce(%1474 init: %cst_68) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %1478 = stablehlo.add %1473, %1477 : tensor<f32>
    %1479 = stablehlo.multiply %arg8, %arg8 : tensor<3x3x64x64xf32>
    %1480 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %1481 = stablehlo.multiply %1480, %arg8 : tensor<3x3x64x64xf32>
    %cst_69 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1482 = stablehlo.reduce(%1479 init: %cst_69) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %1483 = stablehlo.add %1478, %1482 : tensor<f32>
    %1484 = stablehlo.multiply %arg13, %arg13 : tensor<3x3x64x64xf32>
    %1485 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %1486 = stablehlo.multiply %1485, %arg13 : tensor<3x3x64x64xf32>
    %cst_70 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1487 = stablehlo.reduce(%1484 init: %cst_70) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %1488 = stablehlo.add %1483, %1487 : tensor<f32>
    %1489 = stablehlo.multiply %arg14, %arg14 : tensor<3x3x64x64xf32>
    %1490 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %1491 = stablehlo.multiply %1490, %arg14 : tensor<3x3x64x64xf32>
    %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1492 = stablehlo.reduce(%1489 init: %cst_71) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %1493 = stablehlo.add %1488, %1492 : tensor<f32>
    %1494 = stablehlo.multiply %arg19, %arg19 : tensor<3x3x64x128xf32>
    %1495 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %1496 = stablehlo.multiply %1495, %arg19 : tensor<3x3x64x128xf32>
    %cst_72 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1497 = stablehlo.reduce(%1494 init: %cst_72) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x128xf32>, tensor<f32>) -> tensor<f32>
    %1498 = stablehlo.add %1493, %1497 : tensor<f32>
    %1499 = stablehlo.multiply %arg20, %arg20 : tensor<3x3x128x128xf32>
    %1500 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %1501 = stablehlo.multiply %1500, %arg20 : tensor<3x3x128x128xf32>
    %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1502 = stablehlo.reduce(%1499 init: %cst_73) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %1503 = stablehlo.add %1498, %1502 : tensor<f32>
    %1504 = stablehlo.multiply %arg21, %arg21 : tensor<1x1x64x128xf32>
    %1505 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %1506 = stablehlo.multiply %1505, %arg21 : tensor<1x1x64x128xf32>
    %cst_74 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1507 = stablehlo.reduce(%1504 init: %cst_74) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<1x1x64x128xf32>, tensor<f32>) -> tensor<f32>
    %1508 = stablehlo.add %1503, %1507 : tensor<f32>
    %1509 = stablehlo.multiply %arg28, %arg28 : tensor<3x3x128x128xf32>
    %1510 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %1511 = stablehlo.multiply %1510, %arg28 : tensor<3x3x128x128xf32>
    %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1512 = stablehlo.reduce(%1509 init: %cst_75) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %1513 = stablehlo.add %1508, %1512 : tensor<f32>
    %1514 = stablehlo.multiply %arg29, %arg29 : tensor<3x3x128x128xf32>
    %1515 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %1516 = stablehlo.multiply %1515, %arg29 : tensor<3x3x128x128xf32>
    %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1517 = stablehlo.reduce(%1514 init: %cst_76) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %1518 = stablehlo.add %1513, %1517 : tensor<f32>
    %1519 = stablehlo.multiply %arg34, %arg34 : tensor<3x3x128x256xf32>
    %1520 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %1521 = stablehlo.multiply %1520, %arg34 : tensor<3x3x128x256xf32>
    %cst_77 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1522 = stablehlo.reduce(%1519 init: %cst_77) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x256xf32>, tensor<f32>) -> tensor<f32>
    %1523 = stablehlo.add %1518, %1522 : tensor<f32>
    %1524 = stablehlo.multiply %arg35, %arg35 : tensor<3x3x256x256xf32>
    %1525 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %1526 = stablehlo.multiply %1525, %arg35 : tensor<3x3x256x256xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1527 = stablehlo.reduce(%1524 init: %cst_78) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %1528 = stablehlo.add %1523, %1527 : tensor<f32>
    %1529 = stablehlo.multiply %arg36, %arg36 : tensor<1x1x128x256xf32>
    %1530 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %1531 = stablehlo.multiply %1530, %arg36 : tensor<1x1x128x256xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1532 = stablehlo.reduce(%1529 init: %cst_79) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<1x1x128x256xf32>, tensor<f32>) -> tensor<f32>
    %1533 = stablehlo.add %1528, %1532 : tensor<f32>
    %1534 = stablehlo.multiply %arg43, %arg43 : tensor<3x3x256x256xf32>
    %1535 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %1536 = stablehlo.multiply %1535, %arg43 : tensor<3x3x256x256xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1537 = stablehlo.reduce(%1534 init: %cst_80) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %1538 = stablehlo.add %1533, %1537 : tensor<f32>
    %1539 = stablehlo.multiply %arg44, %arg44 : tensor<3x3x256x256xf32>
    %1540 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %1541 = stablehlo.multiply %1540, %arg44 : tensor<3x3x256x256xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1542 = stablehlo.reduce(%1539 init: %cst_81) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %1543 = stablehlo.add %1538, %1542 : tensor<f32>
    %1544 = stablehlo.multiply %arg49, %arg49 : tensor<3x3x256x512xf32>
    %1545 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %1546 = stablehlo.multiply %1545, %arg49 : tensor<3x3x256x512xf32>
    %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1547 = stablehlo.reduce(%1544 init: %cst_82) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x512xf32>, tensor<f32>) -> tensor<f32>
    %1548 = stablehlo.add %1543, %1547 : tensor<f32>
    %1549 = stablehlo.multiply %arg50, %arg50 : tensor<3x3x512x512xf32>
    %1550 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %1551 = stablehlo.multiply %1550, %arg50 : tensor<3x3x512x512xf32>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1552 = stablehlo.reduce(%1549 init: %cst_83) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xf32>, tensor<f32>) -> tensor<f32>
    %1553 = stablehlo.add %1548, %1552 : tensor<f32>
    %1554 = stablehlo.multiply %arg51, %arg51 : tensor<1x1x256x512xf32>
    %1555 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %1556 = stablehlo.multiply %1555, %arg51 : tensor<1x1x256x512xf32>
    %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1557 = stablehlo.reduce(%1554 init: %cst_84) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<1x1x256x512xf32>, tensor<f32>) -> tensor<f32>
    %1558 = stablehlo.add %1553, %1557 : tensor<f32>
    %1559 = stablehlo.multiply %arg58, %arg58 : tensor<3x3x512x512xf32>
    %1560 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %1561 = stablehlo.multiply %1560, %arg58 : tensor<3x3x512x512xf32>
    %cst_85 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1562 = stablehlo.reduce(%1559 init: %cst_85) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xf32>, tensor<f32>) -> tensor<f32>
    %1563 = stablehlo.add %1558, %1562 : tensor<f32>
    %1564 = stablehlo.multiply %arg59, %arg59 : tensor<3x3x512x512xf32>
    %1565 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %1566 = stablehlo.multiply %1565, %arg59 : tensor<3x3x512x512xf32>
    %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1567 = stablehlo.reduce(%1564 init: %cst_86) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xf32>, tensor<f32>) -> tensor<f32>
    %1568 = stablehlo.add %1563, %1567 : tensor<f32>
    %1569 = stablehlo.multiply %arg62, %arg62 : tensor<7x7x3x64xf32>
    %1570 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %1571 = stablehlo.multiply %1570, %arg62 : tensor<7x7x3x64xf32>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1572 = stablehlo.reduce(%1569 init: %cst_87) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<7x7x3x64xf32>, tensor<f32>) -> tensor<f32>
    %1573 = stablehlo.add %1568, %1572 : tensor<f32>
    %cst_88 = stablehlo.constant dense<5.000000e-05> : tensor<f32>
    %1574 = stablehlo.multiply %cst_88, %1573 : tensor<f32>
    %1575 = stablehlo.add %1468, %1574 : tensor<f32>
    %1576 = stablehlo.convert %arg167 : tensor<f32>
    %1577 = stablehlo.multiply %1576, %1575 : tensor<f32>
    %1578 = stablehlo.multiply %1576, %cst_13 : tensor<f32>
    %1579 = stablehlo.multiply %cst_88, %1578 : tensor<f32>
    %1580 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %1581 = stablehlo.multiply %1580, %1571 : tensor<7x7x3x64xf32>
    %1582 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %1583 = stablehlo.multiply %1582, %1566 : tensor<3x3x512x512xf32>
    %1584 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %1585 = stablehlo.multiply %1584, %1561 : tensor<3x3x512x512xf32>
    %1586 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %1587 = stablehlo.multiply %1586, %1556 : tensor<1x1x256x512xf32>
    %1588 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %1589 = stablehlo.multiply %1588, %1551 : tensor<3x3x512x512xf32>
    %1590 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %1591 = stablehlo.multiply %1590, %1546 : tensor<3x3x256x512xf32>
    %1592 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %1593 = stablehlo.multiply %1592, %1541 : tensor<3x3x256x256xf32>
    %1594 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %1595 = stablehlo.multiply %1594, %1536 : tensor<3x3x256x256xf32>
    %1596 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %1597 = stablehlo.multiply %1596, %1531 : tensor<1x1x128x256xf32>
    %1598 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %1599 = stablehlo.multiply %1598, %1526 : tensor<3x3x256x256xf32>
    %1600 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %1601 = stablehlo.multiply %1600, %1521 : tensor<3x3x128x256xf32>
    %1602 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %1603 = stablehlo.multiply %1602, %1516 : tensor<3x3x128x128xf32>
    %1604 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %1605 = stablehlo.multiply %1604, %1511 : tensor<3x3x128x128xf32>
    %1606 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %1607 = stablehlo.multiply %1606, %1506 : tensor<1x1x64x128xf32>
    %1608 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %1609 = stablehlo.multiply %1608, %1501 : tensor<3x3x128x128xf32>
    %1610 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %1611 = stablehlo.multiply %1610, %1496 : tensor<3x3x64x128xf32>
    %1612 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %1613 = stablehlo.multiply %1612, %1491 : tensor<3x3x64x64xf32>
    %1614 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %1615 = stablehlo.multiply %1614, %1486 : tensor<3x3x64x64xf32>
    %1616 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %1617 = stablehlo.multiply %1616, %1481 : tensor<3x3x64x64xf32>
    %1618 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %1619 = stablehlo.multiply %1618, %1476 : tensor<3x3x64x64xf32>
    %1620 = stablehlo.broadcast_in_dim %1579, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %1621 = stablehlo.multiply %1620, %1471 : tensor<512x1000xf32>
    %1622 = stablehlo.divide %1578, %cst_66 : tensor<f32>
    %1623 = stablehlo.broadcast_in_dim %1622, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1624 = stablehlo.negate %1623 : tensor<256xf32>
    %1625 = stablehlo.broadcast_in_dim %1624, dims = [0] : (tensor<256xf32>) -> tensor<256x1000xf32>
    %1626 = stablehlo.multiply %1461, %1625 : tensor<256x1000xf32>
    %1627 = stablehlo.convert %1626 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
    %1628 = call @log_softmax_292(%1462#1, %1462#2, %1627) : (tensor<256x1000xbf16>, tensor<256x1xbf16>, tensor<256x1000xbf16>) -> tensor<256x1000xbf16>
    %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1629 = stablehlo.reduce(%1628 init: %cst_89) applies stablehlo.add across dimensions = [0] : (tensor<256x1000xbf16>, tensor<bf16>) -> tensor<1000xbf16>
    %1630 = stablehlo.reshape %1629 : (tensor<1000xbf16>) -> tensor<1x1000xbf16>
    %1631 = stablehlo.reshape %1630 : (tensor<1x1000xbf16>) -> tensor<1000xbf16>
    %1632 = stablehlo.dot_general %1628, %1445, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x1000xbf16>, tensor<256x512xbf16>) -> tensor<1000x512xbf16>
    %1633 = stablehlo.transpose %1632, dims = [1, 0] : (tensor<1000x512xbf16>) -> tensor<512x1000xbf16>
    %1634 = stablehlo.dot_general %1628, %1446, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1000xbf16>, tensor<512x1000xbf16>) -> tensor<256x512xbf16>
    %1635 = stablehlo.convert %1631 : (tensor<1000xbf16>) -> tensor<1000xf32>
    %1636 = stablehlo.convert %1633 : (tensor<512x1000xbf16>) -> tensor<512x1000xf32>
    %1637 = stablehlo.add %1621, %1636 : tensor<512x1000xf32>
    %1638 = stablehlo.convert %1634 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %1639 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
    %1640 = stablehlo.divide %1638, %1639 : tensor<256x512xf32>
    %1641 = stablehlo.broadcast_in_dim %1640, dims = [0, 3] : (tensor<256x512xf32>) -> tensor<256x7x7x512xf32>
    %1642 = stablehlo.convert %1641 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1643 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1644 = stablehlo.select %1439, %1642, %1643 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %1645 = stablehlo.convert %1644 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1646 = stablehlo.reduce(%1645 init: %cst_90) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1647 = stablehlo.reshape %1646 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1648 = stablehlo.reshape %1647 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1649 = stablehlo.multiply %1421, %1645 : tensor<256x7x7x512xf32>
    %cst_91 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1650 = stablehlo.reduce(%1649 init: %cst_91) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1651 = stablehlo.reshape %1650 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1652 = stablehlo.broadcast_in_dim %1429, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1653 = stablehlo.multiply %1645, %1652 : tensor<256x7x7x512xf32>
    %1654 = stablehlo.multiply %1424, %1651 : tensor<1x1x1x512xf32>
    %1655 = stablehlo.multiply %1651, %1428 : tensor<1x1x1x512xf32>
    %1656 = stablehlo.reshape %1654 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1657 = stablehlo.multiply %1655, %1427 : tensor<1x1x1x512xf32>
    %1658 = stablehlo.negate %1653 : tensor<256x7x7x512xf32>
    %cst_92 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1659 = stablehlo.reduce(%1658 init: %cst_92) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1660 = stablehlo.reshape %1659 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1661 = stablehlo.convert %1653 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1662 = stablehlo.reduce(%1657 init: %cst_93) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1663 = stablehlo.reduce(%1660 init: %cst_94) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1664 = stablehlo.multiply %1662, %1406 : tensor<512xf32>
    %1665 = stablehlo.negate %1664 : tensor<512xf32>
    %1666 = stablehlo.multiply %1665, %1393 : tensor<512xf32>
    %1667 = stablehlo.add %1663, %1666 : tensor<512xf32>
    %1668 = stablehlo.broadcast_in_dim %1664, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1669 = stablehlo.pad %1668, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1670 = stablehlo.broadcast_in_dim %1667, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1671 = stablehlo.pad %1670, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1672 = stablehlo.add %1669, %1671 : tensor<2x512xf32>
    %1673 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1674 = stablehlo.divide %1672, %1673 : tensor<2x512xf32>
    %1675 = "stablehlo.all_reduce"(%1674) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1676 = stablehlo.slice %1675 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1677 = stablehlo.slice %1675 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_95 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1678 = stablehlo.reduce(%1677 init: %cst_95) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1679 = stablehlo.reduce(%1676 init: %cst_96) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1680 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1681 = stablehlo.divide %1678, %1680 : tensor<512xf32>
    %1682 = stablehlo.broadcast_in_dim %1681, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1683 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1684 = stablehlo.divide %1679, %1683 : tensor<512xf32>
    %1685 = stablehlo.broadcast_in_dim %1684, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1686 = stablehlo.multiply %1682, %1374 : tensor<256x7x7x512xf32>
    %1687 = stablehlo.add %1685, %1686 : tensor<256x7x7x512xf32>
    %1688 = stablehlo.convert %1687 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1689 = stablehlo.add %1661, %1688 : tensor<256x7x7x512xbf16>
    %1690 = stablehlo.convolution(%1365, %1689) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x512x512xbf16>
    %1691 = stablehlo.reverse %1369, dims = [0, 1] : tensor<3x3x512x512xbf16>
    %1692 = stablehlo.convolution(%1689, %1691) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1693 = stablehlo.convert %1690 : (tensor<3x3x512x512xbf16>) -> tensor<3x3x512x512xf32>
    %1694 = stablehlo.add %1583, %1693 : tensor<3x3x512x512xf32>
    %1695 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1696 = stablehlo.select %1367, %1692, %1695 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %1697 = stablehlo.convert %1696 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_97 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1698 = stablehlo.reduce(%1697 init: %cst_97) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1699 = stablehlo.reshape %1698 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1700 = stablehlo.reshape %1699 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1701 = stablehlo.multiply %1350, %1697 : tensor<256x7x7x512xf32>
    %cst_98 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1702 = stablehlo.reduce(%1701 init: %cst_98) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1703 = stablehlo.reshape %1702 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1704 = stablehlo.broadcast_in_dim %1358, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1705 = stablehlo.multiply %1697, %1704 : tensor<256x7x7x512xf32>
    %1706 = stablehlo.multiply %1353, %1703 : tensor<1x1x1x512xf32>
    %1707 = stablehlo.multiply %1703, %1357 : tensor<1x1x1x512xf32>
    %1708 = stablehlo.reshape %1706 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1709 = stablehlo.multiply %1707, %1356 : tensor<1x1x1x512xf32>
    %1710 = stablehlo.negate %1705 : tensor<256x7x7x512xf32>
    %cst_99 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1711 = stablehlo.reduce(%1710 init: %cst_99) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1712 = stablehlo.reshape %1711 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1713 = stablehlo.convert %1705 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1714 = stablehlo.reduce(%1709 init: %cst_100) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1715 = stablehlo.reduce(%1712 init: %cst_101) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1716 = stablehlo.multiply %1714, %1335 : tensor<512xf32>
    %1717 = stablehlo.negate %1716 : tensor<512xf32>
    %1718 = stablehlo.multiply %1717, %1322 : tensor<512xf32>
    %1719 = stablehlo.add %1715, %1718 : tensor<512xf32>
    %1720 = stablehlo.broadcast_in_dim %1716, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1721 = stablehlo.pad %1720, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1722 = stablehlo.broadcast_in_dim %1719, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1723 = stablehlo.pad %1722, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1724 = stablehlo.add %1721, %1723 : tensor<2x512xf32>
    %1725 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1726 = stablehlo.divide %1724, %1725 : tensor<2x512xf32>
    %1727 = "stablehlo.all_reduce"(%1726) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1728 = stablehlo.slice %1727 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1729 = stablehlo.slice %1727 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1730 = stablehlo.reduce(%1729 init: %cst_102) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1731 = stablehlo.reduce(%1728 init: %cst_103) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1732 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1733 = stablehlo.divide %1730, %1732 : tensor<512xf32>
    %1734 = stablehlo.broadcast_in_dim %1733, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1735 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1736 = stablehlo.divide %1731, %1735 : tensor<512xf32>
    %1737 = stablehlo.broadcast_in_dim %1736, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1738 = stablehlo.multiply %1734, %1303 : tensor<256x7x7x512xf32>
    %1739 = stablehlo.add %1737, %1738 : tensor<256x7x7x512xf32>
    %1740 = stablehlo.convert %1739 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1741 = stablehlo.add %1713, %1740 : tensor<256x7x7x512xbf16>
    %1742 = stablehlo.convolution(%1294, %1741) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x512x512xbf16>
    %1743 = stablehlo.reverse %1298, dims = [0, 1] : tensor<3x3x512x512xbf16>
    %1744 = stablehlo.convolution(%1741, %1743) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1745 = stablehlo.add %1644, %1744 : tensor<256x7x7x512xbf16>
    %1746 = stablehlo.convert %1742 : (tensor<3x3x512x512xbf16>) -> tensor<3x3x512x512xf32>
    %1747 = stablehlo.add %1585, %1746 : tensor<3x3x512x512xf32>
    %1748 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1749 = stablehlo.select %1296, %1745, %1748 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %1750 = stablehlo.convert %1749 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1751 = stablehlo.reduce(%1750 init: %cst_104) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1752 = stablehlo.reshape %1751 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1753 = stablehlo.reshape %1752 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1754 = stablehlo.multiply %1278, %1750 : tensor<256x7x7x512xf32>
    %cst_105 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1755 = stablehlo.reduce(%1754 init: %cst_105) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1756 = stablehlo.reshape %1755 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1757 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1758 = stablehlo.multiply %1750, %1757 : tensor<256x7x7x512xf32>
    %1759 = stablehlo.multiply %1281, %1756 : tensor<1x1x1x512xf32>
    %1760 = stablehlo.multiply %1756, %1285 : tensor<1x1x1x512xf32>
    %1761 = stablehlo.reshape %1759 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1762 = stablehlo.multiply %1760, %1284 : tensor<1x1x1x512xf32>
    %1763 = stablehlo.negate %1758 : tensor<256x7x7x512xf32>
    %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1764 = stablehlo.reduce(%1763 init: %cst_106) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1765 = stablehlo.reshape %1764 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1766 = stablehlo.convert %1758 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1767 = stablehlo.reduce(%1762 init: %cst_107) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1768 = stablehlo.reduce(%1765 init: %cst_108) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1769 = stablehlo.multiply %1767, %1263 : tensor<512xf32>
    %1770 = stablehlo.negate %1769 : tensor<512xf32>
    %1771 = stablehlo.multiply %1770, %1250 : tensor<512xf32>
    %1772 = stablehlo.add %1768, %1771 : tensor<512xf32>
    %1773 = stablehlo.broadcast_in_dim %1769, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1774 = stablehlo.pad %1773, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1775 = stablehlo.broadcast_in_dim %1772, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1776 = stablehlo.pad %1775, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1777 = stablehlo.add %1774, %1776 : tensor<2x512xf32>
    %1778 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1779 = stablehlo.divide %1777, %1778 : tensor<2x512xf32>
    %1780 = "stablehlo.all_reduce"(%1779) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1781 = stablehlo.slice %1780 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1782 = stablehlo.slice %1780 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1783 = stablehlo.reduce(%1782 init: %cst_109) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1784 = stablehlo.reduce(%1781 init: %cst_110) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1785 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1786 = stablehlo.divide %1783, %1785 : tensor<512xf32>
    %1787 = stablehlo.broadcast_in_dim %1786, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1788 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1789 = stablehlo.divide %1784, %1788 : tensor<512xf32>
    %1790 = stablehlo.broadcast_in_dim %1789, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1791 = stablehlo.multiply %1787, %1231 : tensor<256x7x7x512xf32>
    %1792 = stablehlo.add %1790, %1791 : tensor<256x7x7x512xf32>
    %1793 = stablehlo.convert %1792 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1794 = stablehlo.add %1766, %1793 : tensor<256x7x7x512xbf16>
    %1795 = stablehlo.convolution(%1084, %1794) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x7x7x512xbf16>) -> tensor<1x1x256x512xbf16>
    %1796 = stablehlo.reverse %1226, dims = [0, 1] : tensor<1x1x256x512xbf16>
    %1797 = stablehlo.convolution(%1794, %1796) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<1x1x256x512xbf16>) -> tensor<256x14x14x256xbf16>
    %1798 = stablehlo.convert %1795 : (tensor<1x1x256x512xbf16>) -> tensor<1x1x256x512xf32>
    %1799 = stablehlo.add %1587, %1798 : tensor<1x1x256x512xf32>
    %1800 = stablehlo.convert %1749 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1801 = stablehlo.reduce(%1800 init: %cst_111) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1802 = stablehlo.reshape %1801 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1803 = stablehlo.reshape %1802 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1804 = stablehlo.multiply %1211, %1800 : tensor<256x7x7x512xf32>
    %cst_112 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1805 = stablehlo.reduce(%1804 init: %cst_112) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1806 = stablehlo.reshape %1805 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1807 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1808 = stablehlo.multiply %1800, %1807 : tensor<256x7x7x512xf32>
    %1809 = stablehlo.multiply %1214, %1806 : tensor<1x1x1x512xf32>
    %1810 = stablehlo.multiply %1806, %1218 : tensor<1x1x1x512xf32>
    %1811 = stablehlo.reshape %1809 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1812 = stablehlo.multiply %1810, %1217 : tensor<1x1x1x512xf32>
    %1813 = stablehlo.negate %1808 : tensor<256x7x7x512xf32>
    %cst_113 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1814 = stablehlo.reduce(%1813 init: %cst_113) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1815 = stablehlo.reshape %1814 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1816 = stablehlo.convert %1808 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1817 = stablehlo.reduce(%1812 init: %cst_114) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1818 = stablehlo.reduce(%1815 init: %cst_115) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1819 = stablehlo.multiply %1817, %1196 : tensor<512xf32>
    %1820 = stablehlo.negate %1819 : tensor<512xf32>
    %1821 = stablehlo.multiply %1820, %1183 : tensor<512xf32>
    %1822 = stablehlo.add %1818, %1821 : tensor<512xf32>
    %1823 = stablehlo.broadcast_in_dim %1819, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1824 = stablehlo.pad %1823, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1825 = stablehlo.broadcast_in_dim %1822, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1826 = stablehlo.pad %1825, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1827 = stablehlo.add %1824, %1826 : tensor<2x512xf32>
    %1828 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1829 = stablehlo.divide %1827, %1828 : tensor<2x512xf32>
    %1830 = "stablehlo.all_reduce"(%1829) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1831 = stablehlo.slice %1830 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1832 = stablehlo.slice %1830 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1833 = stablehlo.reduce(%1832 init: %cst_116) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1834 = stablehlo.reduce(%1831 init: %cst_117) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1835 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1836 = stablehlo.divide %1833, %1835 : tensor<512xf32>
    %1837 = stablehlo.broadcast_in_dim %1836, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1838 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1839 = stablehlo.divide %1834, %1838 : tensor<512xf32>
    %1840 = stablehlo.broadcast_in_dim %1839, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1841 = stablehlo.multiply %1837, %1164 : tensor<256x7x7x512xf32>
    %1842 = stablehlo.add %1840, %1841 : tensor<256x7x7x512xf32>
    %1843 = stablehlo.convert %1842 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1844 = stablehlo.add %1816, %1843 : tensor<256x7x7x512xbf16>
    %1845 = stablehlo.convolution(%1155, %1844) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x512x512xbf16>
    %1846 = stablehlo.reverse %1159, dims = [0, 1] : tensor<3x3x512x512xbf16>
    %1847 = stablehlo.convolution(%1844, %1846) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1848 = stablehlo.convert %1845 : (tensor<3x3x512x512xbf16>) -> tensor<3x3x512x512xf32>
    %1849 = stablehlo.add %1589, %1848 : tensor<3x3x512x512xf32>
    %1850 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1851 = stablehlo.select %1157, %1847, %1850 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %1852 = stablehlo.convert %1851 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1853 = stablehlo.reduce(%1852 init: %cst_118) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1854 = stablehlo.reshape %1853 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1855 = stablehlo.reshape %1854 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1856 = stablehlo.multiply %1140, %1852 : tensor<256x7x7x512xf32>
    %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1857 = stablehlo.reduce(%1856 init: %cst_119) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1858 = stablehlo.reshape %1857 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1859 = stablehlo.broadcast_in_dim %1148, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1860 = stablehlo.multiply %1852, %1859 : tensor<256x7x7x512xf32>
    %1861 = stablehlo.multiply %1143, %1858 : tensor<1x1x1x512xf32>
    %1862 = stablehlo.multiply %1858, %1147 : tensor<1x1x1x512xf32>
    %1863 = stablehlo.reshape %1861 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %1864 = stablehlo.multiply %1862, %1146 : tensor<1x1x1x512xf32>
    %1865 = stablehlo.negate %1860 : tensor<256x7x7x512xf32>
    %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1866 = stablehlo.reduce(%1865 init: %cst_120) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1867 = stablehlo.reshape %1866 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1868 = stablehlo.convert %1860 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1869 = stablehlo.reduce(%1864 init: %cst_121) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1870 = stablehlo.reduce(%1867 init: %cst_122) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1871 = stablehlo.multiply %1869, %1125 : tensor<512xf32>
    %1872 = stablehlo.negate %1871 : tensor<512xf32>
    %1873 = stablehlo.multiply %1872, %1112 : tensor<512xf32>
    %1874 = stablehlo.add %1870, %1873 : tensor<512xf32>
    %1875 = stablehlo.broadcast_in_dim %1871, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1876 = stablehlo.pad %1875, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1877 = stablehlo.broadcast_in_dim %1874, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %1878 = stablehlo.pad %1877, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %1879 = stablehlo.add %1876, %1878 : tensor<2x512xf32>
    %1880 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %1881 = stablehlo.divide %1879, %1880 : tensor<2x512xf32>
    %1882 = "stablehlo.all_reduce"(%1881) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %1883 = stablehlo.slice %1882 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %1884 = stablehlo.slice %1882 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1885 = stablehlo.reduce(%1884 init: %cst_123) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1886 = stablehlo.reduce(%1883 init: %cst_124) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %1887 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1888 = stablehlo.divide %1885, %1887 : tensor<512xf32>
    %1889 = stablehlo.broadcast_in_dim %1888, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1890 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1891 = stablehlo.divide %1886, %1890 : tensor<512xf32>
    %1892 = stablehlo.broadcast_in_dim %1891, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %1893 = stablehlo.multiply %1889, %1093 : tensor<256x7x7x512xf32>
    %1894 = stablehlo.add %1892, %1893 : tensor<256x7x7x512xf32>
    %1895 = stablehlo.convert %1894 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1896 = stablehlo.add %1868, %1895 : tensor<256x7x7x512xbf16>
    %1897 = stablehlo.convolution(%1084, %1896) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x256x512xbf16>
    %1898 = stablehlo.reverse %1088, dims = [0, 1] : tensor<3x3x256x512xbf16>
    %1899 = stablehlo.convolution(%1896, %1898) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x256x512xbf16>) -> tensor<256x14x14x256xbf16>
    %1900 = stablehlo.add %1797, %1899 : tensor<256x14x14x256xbf16>
    %1901 = stablehlo.convert %1897 : (tensor<3x3x256x512xbf16>) -> tensor<3x3x256x512xf32>
    %1902 = stablehlo.add %1591, %1901 : tensor<3x3x256x512xf32>
    %1903 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1904 = stablehlo.select %1086, %1900, %1903 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %1905 = stablehlo.convert %1904 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1906 = stablehlo.reduce(%1905 init: %cst_125) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1907 = stablehlo.reshape %1906 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1908 = stablehlo.reshape %1907 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %1909 = stablehlo.multiply %1068, %1905 : tensor<256x14x14x256xf32>
    %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1910 = stablehlo.reduce(%1909 init: %cst_126) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1911 = stablehlo.reshape %1910 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1912 = stablehlo.broadcast_in_dim %1076, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1913 = stablehlo.multiply %1905, %1912 : tensor<256x14x14x256xf32>
    %1914 = stablehlo.multiply %1071, %1911 : tensor<1x1x1x256xf32>
    %1915 = stablehlo.multiply %1911, %1075 : tensor<1x1x1x256xf32>
    %1916 = stablehlo.reshape %1914 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %1917 = stablehlo.multiply %1915, %1074 : tensor<1x1x1x256xf32>
    %1918 = stablehlo.negate %1913 : tensor<256x14x14x256xf32>
    %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1919 = stablehlo.reduce(%1918 init: %cst_127) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1920 = stablehlo.reshape %1919 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1921 = stablehlo.convert %1913 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1922 = stablehlo.reduce(%1917 init: %cst_128) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_129 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1923 = stablehlo.reduce(%1920 init: %cst_129) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1924 = stablehlo.multiply %1922, %1053 : tensor<256xf32>
    %1925 = stablehlo.negate %1924 : tensor<256xf32>
    %1926 = stablehlo.multiply %1925, %1040 : tensor<256xf32>
    %1927 = stablehlo.add %1923, %1926 : tensor<256xf32>
    %1928 = stablehlo.broadcast_in_dim %1924, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1929 = stablehlo.pad %1928, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %1930 = stablehlo.broadcast_in_dim %1927, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1931 = stablehlo.pad %1930, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %1932 = stablehlo.add %1929, %1931 : tensor<2x256xf32>
    %1933 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1934 = stablehlo.divide %1932, %1933 : tensor<2x256xf32>
    %1935 = "stablehlo.all_reduce"(%1934) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1936 = stablehlo.slice %1935 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1937 = stablehlo.slice %1935 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1938 = stablehlo.reduce(%1937 init: %cst_130) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1939 = stablehlo.reduce(%1936 init: %cst_131) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1940 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1941 = stablehlo.divide %1938, %1940 : tensor<256xf32>
    %1942 = stablehlo.broadcast_in_dim %1941, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1943 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1944 = stablehlo.divide %1939, %1943 : tensor<256xf32>
    %1945 = stablehlo.broadcast_in_dim %1944, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1946 = stablehlo.multiply %1942, %1021 : tensor<256x14x14x256xf32>
    %1947 = stablehlo.add %1945, %1946 : tensor<256x14x14x256xf32>
    %1948 = stablehlo.convert %1947 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1949 = stablehlo.add %1921, %1948 : tensor<256x14x14x256xbf16>
    %1950 = stablehlo.convolution(%1012, %1949) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %1951 = stablehlo.reverse %1016, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %1952 = stablehlo.convolution(%1949, %1951) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1953 = stablehlo.convert %1950 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %1954 = stablehlo.add %1593, %1953 : tensor<3x3x256x256xf32>
    %1955 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1956 = stablehlo.select %1014, %1952, %1955 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %1957 = stablehlo.convert %1956 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_132 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1958 = stablehlo.reduce(%1957 init: %cst_132) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1959 = stablehlo.reshape %1958 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1960 = stablehlo.reshape %1959 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %1961 = stablehlo.multiply %997, %1957 : tensor<256x14x14x256xf32>
    %cst_133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1962 = stablehlo.reduce(%1961 init: %cst_133) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1963 = stablehlo.reshape %1962 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1964 = stablehlo.broadcast_in_dim %1005, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1965 = stablehlo.multiply %1957, %1964 : tensor<256x14x14x256xf32>
    %1966 = stablehlo.multiply %1000, %1963 : tensor<1x1x1x256xf32>
    %1967 = stablehlo.multiply %1963, %1004 : tensor<1x1x1x256xf32>
    %1968 = stablehlo.reshape %1966 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %1969 = stablehlo.multiply %1967, %1003 : tensor<1x1x1x256xf32>
    %1970 = stablehlo.negate %1965 : tensor<256x14x14x256xf32>
    %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1971 = stablehlo.reduce(%1970 init: %cst_134) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1972 = stablehlo.reshape %1971 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1973 = stablehlo.convert %1965 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1974 = stablehlo.reduce(%1969 init: %cst_135) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1975 = stablehlo.reduce(%1972 init: %cst_136) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1976 = stablehlo.multiply %1974, %982 : tensor<256xf32>
    %1977 = stablehlo.negate %1976 : tensor<256xf32>
    %1978 = stablehlo.multiply %1977, %969 : tensor<256xf32>
    %1979 = stablehlo.add %1975, %1978 : tensor<256xf32>
    %1980 = stablehlo.broadcast_in_dim %1976, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1981 = stablehlo.pad %1980, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %1982 = stablehlo.broadcast_in_dim %1979, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1983 = stablehlo.pad %1982, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %1984 = stablehlo.add %1981, %1983 : tensor<2x256xf32>
    %1985 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1986 = stablehlo.divide %1984, %1985 : tensor<2x256xf32>
    %1987 = "stablehlo.all_reduce"(%1986) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1988 = stablehlo.slice %1987 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1989 = stablehlo.slice %1987 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_137 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1990 = stablehlo.reduce(%1989 init: %cst_137) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1991 = stablehlo.reduce(%1988 init: %cst_138) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1992 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1993 = stablehlo.divide %1990, %1992 : tensor<256xf32>
    %1994 = stablehlo.broadcast_in_dim %1993, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1995 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1996 = stablehlo.divide %1991, %1995 : tensor<256xf32>
    %1997 = stablehlo.broadcast_in_dim %1996, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %1998 = stablehlo.multiply %1994, %950 : tensor<256x14x14x256xf32>
    %1999 = stablehlo.add %1997, %1998 : tensor<256x14x14x256xf32>
    %2000 = stablehlo.convert %1999 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2001 = stablehlo.add %1973, %2000 : tensor<256x14x14x256xbf16>
    %2002 = stablehlo.convolution(%941, %2001) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %2003 = stablehlo.reverse %945, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %2004 = stablehlo.convolution(%2001, %2003) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2005 = stablehlo.add %1904, %2004 : tensor<256x14x14x256xbf16>
    %2006 = stablehlo.convert %2002 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %2007 = stablehlo.add %1595, %2006 : tensor<3x3x256x256xf32>
    %2008 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %2009 = stablehlo.select %943, %2005, %2008 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %2010 = stablehlo.convert %2009 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2011 = stablehlo.reduce(%2010 init: %cst_139) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2012 = stablehlo.reshape %2011 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2013 = stablehlo.reshape %2012 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %2014 = stablehlo.multiply %925, %2010 : tensor<256x14x14x256xf32>
    %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2015 = stablehlo.reduce(%2014 init: %cst_140) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2016 = stablehlo.reshape %2015 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2017 = stablehlo.broadcast_in_dim %933, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2018 = stablehlo.multiply %2010, %2017 : tensor<256x14x14x256xf32>
    %2019 = stablehlo.multiply %928, %2016 : tensor<1x1x1x256xf32>
    %2020 = stablehlo.multiply %2016, %932 : tensor<1x1x1x256xf32>
    %2021 = stablehlo.reshape %2019 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %2022 = stablehlo.multiply %2020, %931 : tensor<1x1x1x256xf32>
    %2023 = stablehlo.negate %2018 : tensor<256x14x14x256xf32>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2024 = stablehlo.reduce(%2023 init: %cst_141) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2025 = stablehlo.reshape %2024 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2026 = stablehlo.convert %2018 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2027 = stablehlo.reduce(%2022 init: %cst_142) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2028 = stablehlo.reduce(%2025 init: %cst_143) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2029 = stablehlo.multiply %2027, %910 : tensor<256xf32>
    %2030 = stablehlo.negate %2029 : tensor<256xf32>
    %2031 = stablehlo.multiply %2030, %897 : tensor<256xf32>
    %2032 = stablehlo.add %2028, %2031 : tensor<256xf32>
    %2033 = stablehlo.broadcast_in_dim %2029, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2034 = stablehlo.pad %2033, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %2035 = stablehlo.broadcast_in_dim %2032, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2036 = stablehlo.pad %2035, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %2037 = stablehlo.add %2034, %2036 : tensor<2x256xf32>
    %2038 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %2039 = stablehlo.divide %2037, %2038 : tensor<2x256xf32>
    %2040 = "stablehlo.all_reduce"(%2039) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %2041 = stablehlo.slice %2040 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %2042 = stablehlo.slice %2040 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2043 = stablehlo.reduce(%2042 init: %cst_144) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2044 = stablehlo.reduce(%2041 init: %cst_145) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2045 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2046 = stablehlo.divide %2043, %2045 : tensor<256xf32>
    %2047 = stablehlo.broadcast_in_dim %2046, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %2048 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2049 = stablehlo.divide %2044, %2048 : tensor<256xf32>
    %2050 = stablehlo.broadcast_in_dim %2049, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %2051 = stablehlo.multiply %2047, %878 : tensor<256x14x14x256xf32>
    %2052 = stablehlo.add %2050, %2051 : tensor<256x14x14x256xf32>
    %2053 = stablehlo.convert %2052 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2054 = stablehlo.add %2026, %2053 : tensor<256x14x14x256xbf16>
    %2055 = stablehlo.convolution(%731, %2054) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x14x14x256xbf16>) -> tensor<1x1x128x256xbf16>
    %2056 = stablehlo.reverse %873, dims = [0, 1] : tensor<1x1x128x256xbf16>
    %2057 = stablehlo.convolution(%2054, %2056) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x128x256xbf16>) -> tensor<256x28x28x128xbf16>
    %2058 = stablehlo.convert %2055 : (tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xf32>
    %2059 = stablehlo.add %1597, %2058 : tensor<1x1x128x256xf32>
    %2060 = stablehlo.convert %2009 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2061 = stablehlo.reduce(%2060 init: %cst_146) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2062 = stablehlo.reshape %2061 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2063 = stablehlo.reshape %2062 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %2064 = stablehlo.multiply %858, %2060 : tensor<256x14x14x256xf32>
    %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2065 = stablehlo.reduce(%2064 init: %cst_147) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2066 = stablehlo.reshape %2065 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2067 = stablehlo.broadcast_in_dim %866, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2068 = stablehlo.multiply %2060, %2067 : tensor<256x14x14x256xf32>
    %2069 = stablehlo.multiply %861, %2066 : tensor<1x1x1x256xf32>
    %2070 = stablehlo.multiply %2066, %865 : tensor<1x1x1x256xf32>
    %2071 = stablehlo.reshape %2069 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %2072 = stablehlo.multiply %2070, %864 : tensor<1x1x1x256xf32>
    %2073 = stablehlo.negate %2068 : tensor<256x14x14x256xf32>
    %cst_148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2074 = stablehlo.reduce(%2073 init: %cst_148) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2075 = stablehlo.reshape %2074 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2076 = stablehlo.convert %2068 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2077 = stablehlo.reduce(%2072 init: %cst_149) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_150 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2078 = stablehlo.reduce(%2075 init: %cst_150) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2079 = stablehlo.multiply %2077, %843 : tensor<256xf32>
    %2080 = stablehlo.negate %2079 : tensor<256xf32>
    %2081 = stablehlo.multiply %2080, %830 : tensor<256xf32>
    %2082 = stablehlo.add %2078, %2081 : tensor<256xf32>
    %2083 = stablehlo.broadcast_in_dim %2079, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2084 = stablehlo.pad %2083, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %2085 = stablehlo.broadcast_in_dim %2082, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2086 = stablehlo.pad %2085, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %2087 = stablehlo.add %2084, %2086 : tensor<2x256xf32>
    %2088 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %2089 = stablehlo.divide %2087, %2088 : tensor<2x256xf32>
    %2090 = "stablehlo.all_reduce"(%2089) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %2091 = stablehlo.slice %2090 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %2092 = stablehlo.slice %2090 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2093 = stablehlo.reduce(%2092 init: %cst_151) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2094 = stablehlo.reduce(%2091 init: %cst_152) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2095 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2096 = stablehlo.divide %2093, %2095 : tensor<256xf32>
    %2097 = stablehlo.broadcast_in_dim %2096, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %2098 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2099 = stablehlo.divide %2094, %2098 : tensor<256xf32>
    %2100 = stablehlo.broadcast_in_dim %2099, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %2101 = stablehlo.multiply %2097, %811 : tensor<256x14x14x256xf32>
    %2102 = stablehlo.add %2100, %2101 : tensor<256x14x14x256xf32>
    %2103 = stablehlo.convert %2102 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2104 = stablehlo.add %2076, %2103 : tensor<256x14x14x256xbf16>
    %2105 = stablehlo.convolution(%802, %2104) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %2106 = stablehlo.reverse %806, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %2107 = stablehlo.convolution(%2104, %2106) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2108 = stablehlo.convert %2105 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %2109 = stablehlo.add %1599, %2108 : tensor<3x3x256x256xf32>
    %2110 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %2111 = stablehlo.select %804, %2107, %2110 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %2112 = stablehlo.convert %2111 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2113 = stablehlo.reduce(%2112 init: %cst_153) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2114 = stablehlo.reshape %2113 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2115 = stablehlo.reshape %2114 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %2116 = stablehlo.multiply %787, %2112 : tensor<256x14x14x256xf32>
    %cst_154 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2117 = stablehlo.reduce(%2116 init: %cst_154) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2118 = stablehlo.reshape %2117 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2119 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2120 = stablehlo.multiply %2112, %2119 : tensor<256x14x14x256xf32>
    %2121 = stablehlo.multiply %790, %2118 : tensor<1x1x1x256xf32>
    %2122 = stablehlo.multiply %2118, %794 : tensor<1x1x1x256xf32>
    %2123 = stablehlo.reshape %2121 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %2124 = stablehlo.multiply %2122, %793 : tensor<1x1x1x256xf32>
    %2125 = stablehlo.negate %2120 : tensor<256x14x14x256xf32>
    %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2126 = stablehlo.reduce(%2125 init: %cst_155) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2127 = stablehlo.reshape %2126 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2128 = stablehlo.convert %2120 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2129 = stablehlo.reduce(%2124 init: %cst_156) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2130 = stablehlo.reduce(%2127 init: %cst_157) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2131 = stablehlo.multiply %2129, %772 : tensor<256xf32>
    %2132 = stablehlo.negate %2131 : tensor<256xf32>
    %2133 = stablehlo.multiply %2132, %759 : tensor<256xf32>
    %2134 = stablehlo.add %2130, %2133 : tensor<256xf32>
    %2135 = stablehlo.broadcast_in_dim %2131, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2136 = stablehlo.pad %2135, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %2137 = stablehlo.broadcast_in_dim %2134, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2138 = stablehlo.pad %2137, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %2139 = stablehlo.add %2136, %2138 : tensor<2x256xf32>
    %2140 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %2141 = stablehlo.divide %2139, %2140 : tensor<2x256xf32>
    %2142 = "stablehlo.all_reduce"(%2141) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %2143 = stablehlo.slice %2142 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %2144 = stablehlo.slice %2142 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_158 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2145 = stablehlo.reduce(%2144 init: %cst_158) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2146 = stablehlo.reduce(%2143 init: %cst_159) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2147 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2148 = stablehlo.divide %2145, %2147 : tensor<256xf32>
    %2149 = stablehlo.broadcast_in_dim %2148, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %2150 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2151 = stablehlo.divide %2146, %2150 : tensor<256xf32>
    %2152 = stablehlo.broadcast_in_dim %2151, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %2153 = stablehlo.multiply %2149, %740 : tensor<256x14x14x256xf32>
    %2154 = stablehlo.add %2152, %2153 : tensor<256x14x14x256xf32>
    %2155 = stablehlo.convert %2154 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2156 = stablehlo.add %2128, %2155 : tensor<256x14x14x256xbf16>
    %2157 = stablehlo.convolution(%731, %2156) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x128x256xbf16>
    %2158 = stablehlo.reverse %735, dims = [0, 1] : tensor<3x3x128x256xbf16>
    %2159 = stablehlo.convolution(%2156, %2158) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x128x256xbf16>) -> tensor<256x28x28x128xbf16>
    %2160 = stablehlo.add %2057, %2159 : tensor<256x28x28x128xbf16>
    %2161 = stablehlo.convert %2157 : (tensor<3x3x128x256xbf16>) -> tensor<3x3x128x256xf32>
    %2162 = stablehlo.add %1601, %2161 : tensor<3x3x128x256xf32>
    %2163 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %2164 = stablehlo.select %733, %2160, %2163 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %2165 = stablehlo.convert %2164 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2166 = stablehlo.reduce(%2165 init: %cst_160) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2167 = stablehlo.reshape %2166 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2168 = stablehlo.reshape %2167 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2169 = stablehlo.multiply %715, %2165 : tensor<256x28x28x128xf32>
    %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2170 = stablehlo.reduce(%2169 init: %cst_161) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2171 = stablehlo.reshape %2170 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2172 = stablehlo.broadcast_in_dim %723, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %2173 = stablehlo.multiply %2165, %2172 : tensor<256x28x28x128xf32>
    %2174 = stablehlo.multiply %718, %2171 : tensor<1x1x1x128xf32>
    %2175 = stablehlo.multiply %2171, %722 : tensor<1x1x1x128xf32>
    %2176 = stablehlo.reshape %2174 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2177 = stablehlo.multiply %2175, %721 : tensor<1x1x1x128xf32>
    %2178 = stablehlo.negate %2173 : tensor<256x28x28x128xf32>
    %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2179 = stablehlo.reduce(%2178 init: %cst_162) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2180 = stablehlo.reshape %2179 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2181 = stablehlo.convert %2173 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2182 = stablehlo.reduce(%2177 init: %cst_163) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2183 = stablehlo.reduce(%2180 init: %cst_164) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2184 = stablehlo.multiply %2182, %700 : tensor<128xf32>
    %2185 = stablehlo.negate %2184 : tensor<128xf32>
    %2186 = stablehlo.multiply %2185, %687 : tensor<128xf32>
    %2187 = stablehlo.add %2183, %2186 : tensor<128xf32>
    %2188 = stablehlo.broadcast_in_dim %2184, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2189 = stablehlo.pad %2188, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2190 = stablehlo.broadcast_in_dim %2187, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2191 = stablehlo.pad %2190, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2192 = stablehlo.add %2189, %2191 : tensor<2x128xf32>
    %2193 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %2194 = stablehlo.divide %2192, %2193 : tensor<2x128xf32>
    %2195 = "stablehlo.all_reduce"(%2194) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %2196 = stablehlo.slice %2195 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %2197 = stablehlo.slice %2195 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2198 = stablehlo.reduce(%2197 init: %cst_165) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2199 = stablehlo.reduce(%2196 init: %cst_166) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2200 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2201 = stablehlo.divide %2198, %2200 : tensor<128xf32>
    %2202 = stablehlo.broadcast_in_dim %2201, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2203 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2204 = stablehlo.divide %2199, %2203 : tensor<128xf32>
    %2205 = stablehlo.broadcast_in_dim %2204, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2206 = stablehlo.multiply %2202, %668 : tensor<256x28x28x128xf32>
    %2207 = stablehlo.add %2205, %2206 : tensor<256x28x28x128xf32>
    %2208 = stablehlo.convert %2207 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %2209 = stablehlo.add %2181, %2208 : tensor<256x28x28x128xbf16>
    %2210 = stablehlo.convolution(%659, %2209) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %2211 = stablehlo.reverse %663, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %2212 = stablehlo.convolution(%2209, %2211) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %2213 = stablehlo.convert %2210 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %2214 = stablehlo.add %1603, %2213 : tensor<3x3x128x128xf32>
    %2215 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %2216 = stablehlo.select %661, %2212, %2215 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %2217 = stablehlo.convert %2216 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_167 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2218 = stablehlo.reduce(%2217 init: %cst_167) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2219 = stablehlo.reshape %2218 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2220 = stablehlo.reshape %2219 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2221 = stablehlo.multiply %644, %2217 : tensor<256x28x28x128xf32>
    %cst_168 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2222 = stablehlo.reduce(%2221 init: %cst_168) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2223 = stablehlo.reshape %2222 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2224 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %2225 = stablehlo.multiply %2217, %2224 : tensor<256x28x28x128xf32>
    %2226 = stablehlo.multiply %647, %2223 : tensor<1x1x1x128xf32>
    %2227 = stablehlo.multiply %2223, %651 : tensor<1x1x1x128xf32>
    %2228 = stablehlo.reshape %2226 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2229 = stablehlo.multiply %2227, %650 : tensor<1x1x1x128xf32>
    %2230 = stablehlo.negate %2225 : tensor<256x28x28x128xf32>
    %cst_169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2231 = stablehlo.reduce(%2230 init: %cst_169) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2232 = stablehlo.reshape %2231 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2233 = stablehlo.convert %2225 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2234 = stablehlo.reduce(%2229 init: %cst_170) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2235 = stablehlo.reduce(%2232 init: %cst_171) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2236 = stablehlo.multiply %2234, %629 : tensor<128xf32>
    %2237 = stablehlo.negate %2236 : tensor<128xf32>
    %2238 = stablehlo.multiply %2237, %616 : tensor<128xf32>
    %2239 = stablehlo.add %2235, %2238 : tensor<128xf32>
    %2240 = stablehlo.broadcast_in_dim %2236, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2241 = stablehlo.pad %2240, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2242 = stablehlo.broadcast_in_dim %2239, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2243 = stablehlo.pad %2242, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2244 = stablehlo.add %2241, %2243 : tensor<2x128xf32>
    %2245 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %2246 = stablehlo.divide %2244, %2245 : tensor<2x128xf32>
    %2247 = "stablehlo.all_reduce"(%2246) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %2248 = stablehlo.slice %2247 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %2249 = stablehlo.slice %2247 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2250 = stablehlo.reduce(%2249 init: %cst_172) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2251 = stablehlo.reduce(%2248 init: %cst_173) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2252 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2253 = stablehlo.divide %2250, %2252 : tensor<128xf32>
    %2254 = stablehlo.broadcast_in_dim %2253, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2255 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2256 = stablehlo.divide %2251, %2255 : tensor<128xf32>
    %2257 = stablehlo.broadcast_in_dim %2256, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2258 = stablehlo.multiply %2254, %597 : tensor<256x28x28x128xf32>
    %2259 = stablehlo.add %2257, %2258 : tensor<256x28x28x128xf32>
    %2260 = stablehlo.convert %2259 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %2261 = stablehlo.add %2233, %2260 : tensor<256x28x28x128xbf16>
    %2262 = stablehlo.convolution(%588, %2261) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %2263 = stablehlo.reverse %592, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %2264 = stablehlo.convolution(%2261, %2263) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %2265 = stablehlo.add %2164, %2264 : tensor<256x28x28x128xbf16>
    %2266 = stablehlo.convert %2262 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %2267 = stablehlo.add %1605, %2266 : tensor<3x3x128x128xf32>
    %2268 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %2269 = stablehlo.select %590, %2265, %2268 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %2270 = stablehlo.convert %2269 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2271 = stablehlo.reduce(%2270 init: %cst_174) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2272 = stablehlo.reshape %2271 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2273 = stablehlo.reshape %2272 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2274 = stablehlo.multiply %572, %2270 : tensor<256x28x28x128xf32>
    %cst_175 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2275 = stablehlo.reduce(%2274 init: %cst_175) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2276 = stablehlo.reshape %2275 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2277 = stablehlo.broadcast_in_dim %580, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %2278 = stablehlo.multiply %2270, %2277 : tensor<256x28x28x128xf32>
    %2279 = stablehlo.multiply %575, %2276 : tensor<1x1x1x128xf32>
    %2280 = stablehlo.multiply %2276, %579 : tensor<1x1x1x128xf32>
    %2281 = stablehlo.reshape %2279 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2282 = stablehlo.multiply %2280, %578 : tensor<1x1x1x128xf32>
    %2283 = stablehlo.negate %2278 : tensor<256x28x28x128xf32>
    %cst_176 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2284 = stablehlo.reduce(%2283 init: %cst_176) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2285 = stablehlo.reshape %2284 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2286 = stablehlo.convert %2278 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2287 = stablehlo.reduce(%2282 init: %cst_177) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2288 = stablehlo.reduce(%2285 init: %cst_178) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2289 = stablehlo.multiply %2287, %557 : tensor<128xf32>
    %2290 = stablehlo.negate %2289 : tensor<128xf32>
    %2291 = stablehlo.multiply %2290, %544 : tensor<128xf32>
    %2292 = stablehlo.add %2288, %2291 : tensor<128xf32>
    %2293 = stablehlo.broadcast_in_dim %2289, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2294 = stablehlo.pad %2293, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2295 = stablehlo.broadcast_in_dim %2292, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2296 = stablehlo.pad %2295, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2297 = stablehlo.add %2294, %2296 : tensor<2x128xf32>
    %2298 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %2299 = stablehlo.divide %2297, %2298 : tensor<2x128xf32>
    %2300 = "stablehlo.all_reduce"(%2299) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %2301 = stablehlo.slice %2300 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %2302 = stablehlo.slice %2300 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2303 = stablehlo.reduce(%2302 init: %cst_179) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_180 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2304 = stablehlo.reduce(%2301 init: %cst_180) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2305 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2306 = stablehlo.divide %2303, %2305 : tensor<128xf32>
    %2307 = stablehlo.broadcast_in_dim %2306, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2308 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2309 = stablehlo.divide %2304, %2308 : tensor<128xf32>
    %2310 = stablehlo.broadcast_in_dim %2309, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2311 = stablehlo.multiply %2307, %525 : tensor<256x28x28x128xf32>
    %2312 = stablehlo.add %2310, %2311 : tensor<256x28x28x128xf32>
    %2313 = stablehlo.convert %2312 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %2314 = stablehlo.add %2286, %2313 : tensor<256x28x28x128xbf16>
    %2315 = stablehlo.convolution(%378, %2314) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x28x28x128xbf16>) -> tensor<1x1x64x128xbf16>
    %2316 = stablehlo.reverse %520, dims = [0, 1] : tensor<1x1x64x128xbf16>
    %2317 = stablehlo.convolution(%2314, %2316) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x64x128xbf16>) -> tensor<256x56x56x64xbf16>
    %2318 = stablehlo.convert %2315 : (tensor<1x1x64x128xbf16>) -> tensor<1x1x64x128xf32>
    %2319 = stablehlo.add %1607, %2318 : tensor<1x1x64x128xf32>
    %2320 = stablehlo.convert %2269 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_181 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2321 = stablehlo.reduce(%2320 init: %cst_181) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2322 = stablehlo.reshape %2321 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2323 = stablehlo.reshape %2322 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2324 = stablehlo.multiply %505, %2320 : tensor<256x28x28x128xf32>
    %cst_182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2325 = stablehlo.reduce(%2324 init: %cst_182) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2326 = stablehlo.reshape %2325 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2327 = stablehlo.broadcast_in_dim %513, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %2328 = stablehlo.multiply %2320, %2327 : tensor<256x28x28x128xf32>
    %2329 = stablehlo.multiply %508, %2326 : tensor<1x1x1x128xf32>
    %2330 = stablehlo.multiply %2326, %512 : tensor<1x1x1x128xf32>
    %2331 = stablehlo.reshape %2329 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2332 = stablehlo.multiply %2330, %511 : tensor<1x1x1x128xf32>
    %2333 = stablehlo.negate %2328 : tensor<256x28x28x128xf32>
    %cst_183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2334 = stablehlo.reduce(%2333 init: %cst_183) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2335 = stablehlo.reshape %2334 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2336 = stablehlo.convert %2328 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2337 = stablehlo.reduce(%2332 init: %cst_184) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2338 = stablehlo.reduce(%2335 init: %cst_185) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2339 = stablehlo.multiply %2337, %490 : tensor<128xf32>
    %2340 = stablehlo.negate %2339 : tensor<128xf32>
    %2341 = stablehlo.multiply %2340, %477 : tensor<128xf32>
    %2342 = stablehlo.add %2338, %2341 : tensor<128xf32>
    %2343 = stablehlo.broadcast_in_dim %2339, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2344 = stablehlo.pad %2343, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2345 = stablehlo.broadcast_in_dim %2342, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2346 = stablehlo.pad %2345, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2347 = stablehlo.add %2344, %2346 : tensor<2x128xf32>
    %2348 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %2349 = stablehlo.divide %2347, %2348 : tensor<2x128xf32>
    %2350 = "stablehlo.all_reduce"(%2349) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %2351 = stablehlo.slice %2350 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %2352 = stablehlo.slice %2350 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2353 = stablehlo.reduce(%2352 init: %cst_186) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2354 = stablehlo.reduce(%2351 init: %cst_187) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2355 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2356 = stablehlo.divide %2353, %2355 : tensor<128xf32>
    %2357 = stablehlo.broadcast_in_dim %2356, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2358 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2359 = stablehlo.divide %2354, %2358 : tensor<128xf32>
    %2360 = stablehlo.broadcast_in_dim %2359, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2361 = stablehlo.multiply %2357, %458 : tensor<256x28x28x128xf32>
    %2362 = stablehlo.add %2360, %2361 : tensor<256x28x28x128xf32>
    %2363 = stablehlo.convert %2362 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %2364 = stablehlo.add %2336, %2363 : tensor<256x28x28x128xbf16>
    %2365 = stablehlo.convolution(%449, %2364) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %2366 = stablehlo.reverse %453, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %2367 = stablehlo.convolution(%2364, %2366) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %2368 = stablehlo.convert %2365 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %2369 = stablehlo.add %1609, %2368 : tensor<3x3x128x128xf32>
    %2370 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %2371 = stablehlo.select %451, %2367, %2370 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %2372 = stablehlo.convert %2371 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_188 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2373 = stablehlo.reduce(%2372 init: %cst_188) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2374 = stablehlo.reshape %2373 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2375 = stablehlo.reshape %2374 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2376 = stablehlo.multiply %434, %2372 : tensor<256x28x28x128xf32>
    %cst_189 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2377 = stablehlo.reduce(%2376 init: %cst_189) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2378 = stablehlo.reshape %2377 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2379 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %2380 = stablehlo.multiply %2372, %2379 : tensor<256x28x28x128xf32>
    %2381 = stablehlo.multiply %437, %2378 : tensor<1x1x1x128xf32>
    %2382 = stablehlo.multiply %2378, %441 : tensor<1x1x1x128xf32>
    %2383 = stablehlo.reshape %2381 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %2384 = stablehlo.multiply %2382, %440 : tensor<1x1x1x128xf32>
    %2385 = stablehlo.negate %2380 : tensor<256x28x28x128xf32>
    %cst_190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2386 = stablehlo.reduce(%2385 init: %cst_190) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2387 = stablehlo.reshape %2386 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %2388 = stablehlo.convert %2380 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_191 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2389 = stablehlo.reduce(%2384 init: %cst_191) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_192 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2390 = stablehlo.reduce(%2387 init: %cst_192) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2391 = stablehlo.multiply %2389, %419 : tensor<128xf32>
    %2392 = stablehlo.negate %2391 : tensor<128xf32>
    %2393 = stablehlo.multiply %2392, %406 : tensor<128xf32>
    %2394 = stablehlo.add %2390, %2393 : tensor<128xf32>
    %2395 = stablehlo.broadcast_in_dim %2391, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2396 = stablehlo.pad %2395, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2397 = stablehlo.broadcast_in_dim %2394, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2398 = stablehlo.pad %2397, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %2399 = stablehlo.add %2396, %2398 : tensor<2x128xf32>
    %2400 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %2401 = stablehlo.divide %2399, %2400 : tensor<2x128xf32>
    %2402 = "stablehlo.all_reduce"(%2401) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %2403 = stablehlo.slice %2402 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %2404 = stablehlo.slice %2402 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_193 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2405 = stablehlo.reduce(%2404 init: %cst_193) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2406 = stablehlo.reduce(%2403 init: %cst_194) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %2407 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2408 = stablehlo.divide %2405, %2407 : tensor<128xf32>
    %2409 = stablehlo.broadcast_in_dim %2408, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2410 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2411 = stablehlo.divide %2406, %2410 : tensor<128xf32>
    %2412 = stablehlo.broadcast_in_dim %2411, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %2413 = stablehlo.multiply %2409, %387 : tensor<256x28x28x128xf32>
    %2414 = stablehlo.add %2412, %2413 : tensor<256x28x28x128xf32>
    %2415 = stablehlo.convert %2414 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %2416 = stablehlo.add %2388, %2415 : tensor<256x28x28x128xbf16>
    %2417 = stablehlo.convolution(%378, %2416) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x64x128xbf16>
    %2418 = stablehlo.reverse %382, dims = [0, 1] : tensor<3x3x64x128xbf16>
    %2419 = stablehlo.convolution(%2416, %2418) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x64x128xbf16>) -> tensor<256x56x56x64xbf16>
    %2420 = stablehlo.add %2317, %2419 : tensor<256x56x56x64xbf16>
    %2421 = stablehlo.convert %2417 : (tensor<3x3x64x128xbf16>) -> tensor<3x3x64x128xf32>
    %2422 = stablehlo.add %1611, %2421 : tensor<3x3x64x128xf32>
    %2423 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %2424 = stablehlo.select %380, %2420, %2423 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %2425 = stablehlo.convert %2424 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_195 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2426 = stablehlo.reduce(%2425 init: %cst_195) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2427 = stablehlo.reshape %2426 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2428 = stablehlo.reshape %2427 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2429 = stablehlo.multiply %362, %2425 : tensor<256x56x56x64xf32>
    %cst_196 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2430 = stablehlo.reduce(%2429 init: %cst_196) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2431 = stablehlo.reshape %2430 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2432 = stablehlo.broadcast_in_dim %370, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %2433 = stablehlo.multiply %2425, %2432 : tensor<256x56x56x64xf32>
    %2434 = stablehlo.multiply %365, %2431 : tensor<1x1x1x64xf32>
    %2435 = stablehlo.multiply %2431, %369 : tensor<1x1x1x64xf32>
    %2436 = stablehlo.reshape %2434 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2437 = stablehlo.multiply %2435, %368 : tensor<1x1x1x64xf32>
    %2438 = stablehlo.negate %2433 : tensor<256x56x56x64xf32>
    %cst_197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2439 = stablehlo.reduce(%2438 init: %cst_197) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2440 = stablehlo.reshape %2439 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2441 = stablehlo.convert %2433 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_198 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2442 = stablehlo.reduce(%2437 init: %cst_198) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2443 = stablehlo.reduce(%2440 init: %cst_199) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2444 = stablehlo.multiply %2442, %347 : tensor<64xf32>
    %2445 = stablehlo.negate %2444 : tensor<64xf32>
    %2446 = stablehlo.multiply %2445, %334 : tensor<64xf32>
    %2447 = stablehlo.add %2443, %2446 : tensor<64xf32>
    %2448 = stablehlo.broadcast_in_dim %2444, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2449 = stablehlo.pad %2448, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2450 = stablehlo.broadcast_in_dim %2447, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2451 = stablehlo.pad %2450, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2452 = stablehlo.add %2449, %2451 : tensor<2x64xf32>
    %2453 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %2454 = stablehlo.divide %2452, %2453 : tensor<2x64xf32>
    %2455 = "stablehlo.all_reduce"(%2454) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %2456 = stablehlo.slice %2455 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %2457 = stablehlo.slice %2455 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_200 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2458 = stablehlo.reduce(%2457 init: %cst_200) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2459 = stablehlo.reduce(%2456 init: %cst_201) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2460 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2461 = stablehlo.divide %2458, %2460 : tensor<64xf32>
    %2462 = stablehlo.broadcast_in_dim %2461, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %2463 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2464 = stablehlo.divide %2459, %2463 : tensor<64xf32>
    %2465 = stablehlo.broadcast_in_dim %2464, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %2466 = stablehlo.multiply %2462, %315 : tensor<256x56x56x64xf32>
    %2467 = stablehlo.add %2465, %2466 : tensor<256x56x56x64xf32>
    %2468 = stablehlo.convert %2467 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %2469 = stablehlo.add %2441, %2468 : tensor<256x56x56x64xbf16>
    %2470 = stablehlo.convolution(%306, %2469) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %2471 = stablehlo.reverse %310, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %2472 = stablehlo.convolution(%2469, %2471) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %2473 = stablehlo.convert %2470 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %2474 = stablehlo.add %1613, %2473 : tensor<3x3x64x64xf32>
    %2475 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %2476 = stablehlo.select %308, %2472, %2475 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %2477 = stablehlo.convert %2476 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_202 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2478 = stablehlo.reduce(%2477 init: %cst_202) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2479 = stablehlo.reshape %2478 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2480 = stablehlo.reshape %2479 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2481 = stablehlo.multiply %291, %2477 : tensor<256x56x56x64xf32>
    %cst_203 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2482 = stablehlo.reduce(%2481 init: %cst_203) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2483 = stablehlo.reshape %2482 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2484 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %2485 = stablehlo.multiply %2477, %2484 : tensor<256x56x56x64xf32>
    %2486 = stablehlo.multiply %294, %2483 : tensor<1x1x1x64xf32>
    %2487 = stablehlo.multiply %2483, %298 : tensor<1x1x1x64xf32>
    %2488 = stablehlo.reshape %2486 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2489 = stablehlo.multiply %2487, %297 : tensor<1x1x1x64xf32>
    %2490 = stablehlo.negate %2485 : tensor<256x56x56x64xf32>
    %cst_204 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2491 = stablehlo.reduce(%2490 init: %cst_204) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2492 = stablehlo.reshape %2491 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2493 = stablehlo.convert %2485 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_205 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2494 = stablehlo.reduce(%2489 init: %cst_205) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_206 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2495 = stablehlo.reduce(%2492 init: %cst_206) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2496 = stablehlo.multiply %2494, %276 : tensor<64xf32>
    %2497 = stablehlo.negate %2496 : tensor<64xf32>
    %2498 = stablehlo.multiply %2497, %263 : tensor<64xf32>
    %2499 = stablehlo.add %2495, %2498 : tensor<64xf32>
    %2500 = stablehlo.broadcast_in_dim %2496, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2501 = stablehlo.pad %2500, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2502 = stablehlo.broadcast_in_dim %2499, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2503 = stablehlo.pad %2502, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2504 = stablehlo.add %2501, %2503 : tensor<2x64xf32>
    %2505 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %2506 = stablehlo.divide %2504, %2505 : tensor<2x64xf32>
    %2507 = "stablehlo.all_reduce"(%2506) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %2508 = stablehlo.slice %2507 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %2509 = stablehlo.slice %2507 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_207 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2510 = stablehlo.reduce(%2509 init: %cst_207) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_208 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2511 = stablehlo.reduce(%2508 init: %cst_208) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2512 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2513 = stablehlo.divide %2510, %2512 : tensor<64xf32>
    %2514 = stablehlo.broadcast_in_dim %2513, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %2515 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2516 = stablehlo.divide %2511, %2515 : tensor<64xf32>
    %2517 = stablehlo.broadcast_in_dim %2516, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %2518 = stablehlo.multiply %2514, %244 : tensor<256x56x56x64xf32>
    %2519 = stablehlo.add %2517, %2518 : tensor<256x56x56x64xf32>
    %2520 = stablehlo.convert %2519 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %2521 = stablehlo.add %2493, %2520 : tensor<256x56x56x64xbf16>
    %2522 = stablehlo.convolution(%235, %2521) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %2523 = stablehlo.reverse %239, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %2524 = stablehlo.convolution(%2521, %2523) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %2525 = stablehlo.add %2424, %2524 : tensor<256x56x56x64xbf16>
    %2526 = stablehlo.convert %2522 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %2527 = stablehlo.add %1615, %2526 : tensor<3x3x64x64xf32>
    %2528 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %2529 = stablehlo.select %237, %2525, %2528 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %2530 = stablehlo.convert %2529 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_209 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2531 = stablehlo.reduce(%2530 init: %cst_209) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2532 = stablehlo.reshape %2531 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2533 = stablehlo.reshape %2532 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2534 = stablehlo.multiply %219, %2530 : tensor<256x56x56x64xf32>
    %cst_210 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2535 = stablehlo.reduce(%2534 init: %cst_210) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2536 = stablehlo.reshape %2535 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2537 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %2538 = stablehlo.multiply %2530, %2537 : tensor<256x56x56x64xf32>
    %2539 = stablehlo.multiply %222, %2536 : tensor<1x1x1x64xf32>
    %2540 = stablehlo.multiply %2536, %226 : tensor<1x1x1x64xf32>
    %2541 = stablehlo.reshape %2539 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2542 = stablehlo.multiply %2540, %225 : tensor<1x1x1x64xf32>
    %2543 = stablehlo.negate %2538 : tensor<256x56x56x64xf32>
    %cst_211 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2544 = stablehlo.reduce(%2543 init: %cst_211) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2545 = stablehlo.reshape %2544 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2546 = stablehlo.convert %2538 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_212 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2547 = stablehlo.reduce(%2542 init: %cst_212) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_213 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2548 = stablehlo.reduce(%2545 init: %cst_213) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2549 = stablehlo.multiply %2547, %204 : tensor<64xf32>
    %2550 = stablehlo.negate %2549 : tensor<64xf32>
    %2551 = stablehlo.multiply %2550, %191 : tensor<64xf32>
    %2552 = stablehlo.add %2548, %2551 : tensor<64xf32>
    %2553 = stablehlo.broadcast_in_dim %2549, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2554 = stablehlo.pad %2553, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2555 = stablehlo.broadcast_in_dim %2552, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2556 = stablehlo.pad %2555, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2557 = stablehlo.add %2554, %2556 : tensor<2x64xf32>
    %2558 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %2559 = stablehlo.divide %2557, %2558 : tensor<2x64xf32>
    %2560 = "stablehlo.all_reduce"(%2559) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %2561 = stablehlo.slice %2560 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %2562 = stablehlo.slice %2560 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_214 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2563 = stablehlo.reduce(%2562 init: %cst_214) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_215 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2564 = stablehlo.reduce(%2561 init: %cst_215) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2565 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2566 = stablehlo.divide %2563, %2565 : tensor<64xf32>
    %2567 = stablehlo.broadcast_in_dim %2566, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %2568 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2569 = stablehlo.divide %2564, %2568 : tensor<64xf32>
    %2570 = stablehlo.broadcast_in_dim %2569, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %2571 = stablehlo.multiply %2567, %172 : tensor<256x56x56x64xf32>
    %2572 = stablehlo.add %2570, %2571 : tensor<256x56x56x64xf32>
    %2573 = stablehlo.convert %2572 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %2574 = stablehlo.add %2546, %2573 : tensor<256x56x56x64xbf16>
    %2575 = stablehlo.convolution(%163, %2574) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %2576 = stablehlo.reverse %167, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %2577 = stablehlo.convolution(%2574, %2576) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %2578 = stablehlo.convert %2575 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %2579 = stablehlo.add %1617, %2578 : tensor<3x3x64x64xf32>
    %2580 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %2581 = stablehlo.select %165, %2577, %2580 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %2582 = stablehlo.convert %2581 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_216 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2583 = stablehlo.reduce(%2582 init: %cst_216) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2584 = stablehlo.reshape %2583 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2585 = stablehlo.reshape %2584 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2586 = stablehlo.multiply %148, %2582 : tensor<256x56x56x64xf32>
    %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2587 = stablehlo.reduce(%2586 init: %cst_217) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2588 = stablehlo.reshape %2587 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2589 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %2590 = stablehlo.multiply %2582, %2589 : tensor<256x56x56x64xf32>
    %2591 = stablehlo.multiply %151, %2588 : tensor<1x1x1x64xf32>
    %2592 = stablehlo.multiply %2588, %155 : tensor<1x1x1x64xf32>
    %2593 = stablehlo.reshape %2591 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2594 = stablehlo.multiply %2592, %154 : tensor<1x1x1x64xf32>
    %2595 = stablehlo.negate %2590 : tensor<256x56x56x64xf32>
    %cst_218 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2596 = stablehlo.reduce(%2595 init: %cst_218) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2597 = stablehlo.reshape %2596 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2598 = stablehlo.convert %2590 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_219 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2599 = stablehlo.reduce(%2594 init: %cst_219) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_220 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2600 = stablehlo.reduce(%2597 init: %cst_220) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2601 = stablehlo.multiply %2599, %133 : tensor<64xf32>
    %2602 = stablehlo.negate %2601 : tensor<64xf32>
    %2603 = stablehlo.multiply %2602, %120 : tensor<64xf32>
    %2604 = stablehlo.add %2600, %2603 : tensor<64xf32>
    %2605 = stablehlo.broadcast_in_dim %2601, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2606 = stablehlo.pad %2605, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2607 = stablehlo.broadcast_in_dim %2604, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2608 = stablehlo.pad %2607, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2609 = stablehlo.add %2606, %2608 : tensor<2x64xf32>
    %2610 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %2611 = stablehlo.divide %2609, %2610 : tensor<2x64xf32>
    %2612 = "stablehlo.all_reduce"(%2611) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %2613 = stablehlo.slice %2612 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %2614 = stablehlo.slice %2612 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_221 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2615 = stablehlo.reduce(%2614 init: %cst_221) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_222 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2616 = stablehlo.reduce(%2613 init: %cst_222) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2617 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2618 = stablehlo.divide %2615, %2617 : tensor<64xf32>
    %2619 = stablehlo.broadcast_in_dim %2618, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %2620 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2621 = stablehlo.divide %2616, %2620 : tensor<64xf32>
    %2622 = stablehlo.broadcast_in_dim %2621, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %2623 = stablehlo.multiply %2619, %101 : tensor<256x56x56x64xf32>
    %2624 = stablehlo.add %2622, %2623 : tensor<256x56x56x64xf32>
    %2625 = stablehlo.convert %2624 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %2626 = stablehlo.add %2598, %2625 : tensor<256x56x56x64xbf16>
    %2627 = stablehlo.convolution(%95, %2626) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %2628 = stablehlo.reverse %96, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %2629 = stablehlo.convolution(%2626, %2628) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %2630 = stablehlo.add %2529, %2629 : tensor<256x56x56x64xbf16>
    %2631 = stablehlo.convert %2627 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %2632 = stablehlo.add %1619, %2631 : tensor<3x3x64x64xf32>
    %cst_223 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %2633 = stablehlo.pad %90, %cst_223, low = [0, 0, 0, 0], high = [0, 1, 1, 0], interior = [0, 0, 0, 0] : (tensor<256x112x112x64xbf16>, tensor<bf16>) -> tensor<256x113x113x64xbf16>
    %cst_224 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2634 = "stablehlo.select_and_scatter"(%2633, %2630, %cst_224) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg170: tensor<bf16>, %arg171: tensor<bf16>):
      %4188 = stablehlo.compare  GE, %arg170, %arg171,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %4188 : tensor<i1>
    }, {
    ^bb0(%arg170: tensor<bf16>, %arg171: tensor<bf16>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<bf16>
      stablehlo.return %4188 : tensor<bf16>
    }) : (tensor<256x113x113x64xbf16>, tensor<256x56x56x64xbf16>, tensor<bf16>) -> tensor<256x113x113x64xbf16>
    %2635 = stablehlo.slice %2634 [0:256, 0:112, 0:112, 0:64] : (tensor<256x113x113x64xbf16>) -> tensor<256x112x112x64xbf16>
    %2636 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
    %2637 = stablehlo.select %92, %2635, %2636 : tensor<256x112x112x64xi1>, tensor<256x112x112x64xbf16>
    %2638 = stablehlo.convert %2637 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
    %cst_225 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2639 = stablehlo.reduce(%2638 init: %cst_225) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2640 = stablehlo.reshape %2639 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2641 = stablehlo.reshape %2640 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2642 = stablehlo.multiply %75, %2638 : tensor<256x112x112x64xf32>
    %cst_226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2643 = stablehlo.reduce(%2642 init: %cst_226) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2644 = stablehlo.reshape %2643 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2645 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %2646 = stablehlo.multiply %2638, %2645 : tensor<256x112x112x64xf32>
    %2647 = stablehlo.multiply %78, %2644 : tensor<1x1x1x64xf32>
    %2648 = stablehlo.multiply %2644, %82 : tensor<1x1x1x64xf32>
    %2649 = stablehlo.reshape %2647 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %2650 = stablehlo.multiply %2648, %81 : tensor<1x1x1x64xf32>
    %2651 = stablehlo.negate %2646 : tensor<256x112x112x64xf32>
    %cst_227 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2652 = stablehlo.reduce(%2651 init: %cst_227) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2653 = stablehlo.reshape %2652 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2654 = stablehlo.convert %2646 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %cst_228 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2655 = stablehlo.reduce(%2650 init: %cst_228) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_229 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2656 = stablehlo.reduce(%2653 init: %cst_229) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2657 = stablehlo.multiply %2655, %60 : tensor<64xf32>
    %2658 = stablehlo.negate %2657 : tensor<64xf32>
    %2659 = stablehlo.multiply %2658, %47 : tensor<64xf32>
    %2660 = stablehlo.add %2656, %2659 : tensor<64xf32>
    %2661 = stablehlo.broadcast_in_dim %2657, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2662 = stablehlo.pad %2661, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2663 = stablehlo.broadcast_in_dim %2660, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %2664 = stablehlo.pad %2663, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %2665 = stablehlo.add %2662, %2664 : tensor<2x64xf32>
    %2666 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %2667 = stablehlo.divide %2665, %2666 : tensor<2x64xf32>
    %2668 = "stablehlo.all_reduce"(%2667) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %2669 = stablehlo.slice %2668 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %2670 = stablehlo.slice %2668 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_230 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2671 = stablehlo.reduce(%2670 init: %cst_230) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_231 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2672 = stablehlo.reduce(%2669 init: %cst_231) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2673 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2674 = stablehlo.divide %2671, %2673 : tensor<64xf32>
    %2675 = stablehlo.broadcast_in_dim %2674, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
    %2676 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2677 = stablehlo.divide %2672, %2676 : tensor<64xf32>
    %2678 = stablehlo.broadcast_in_dim %2677, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
    %2679 = stablehlo.multiply %2675, %28 : tensor<256x112x112x64xf32>
    %2680 = stablehlo.add %2678, %2679 : tensor<256x112x112x64xf32>
    %2681 = stablehlo.convert %2680 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %2682 = stablehlo.add %2654, %2681 : tensor<256x112x112x64xbf16>
    %2683 = stablehlo.convolution(%0, %2682) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xbf16>, tensor<256x112x112x64xbf16>) -> tensor<7x7x3x64xbf16>
    %2684 = stablehlo.convert %2683 : (tensor<7x7x3x64xbf16>) -> tensor<7x7x3x64xf32>
    %2685 = stablehlo.add %1581, %2684 : tensor<7x7x3x64xf32>
    %2686 = stablehlo.convert %arg167 : tensor<f32>
    %2687 = stablehlo.divide %1577, %2686 : tensor<f32>
    %2688 = stablehlo.convert %arg167 : tensor<f32>
    %2689 = stablehlo.broadcast_in_dim %2688, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %2690 = stablehlo.divide %1635, %2689 : tensor<1000xf32>
    %2691 = stablehlo.convert %arg167 : tensor<f32>
    %2692 = stablehlo.broadcast_in_dim %2691, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %2693 = stablehlo.divide %1637, %2692 : tensor<512x1000xf32>
    %2694 = stablehlo.convert %arg167 : tensor<f32>
    %2695 = stablehlo.broadcast_in_dim %2694, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2696 = stablehlo.divide %2585, %2695 : tensor<64xf32>
    %2697 = stablehlo.convert %arg167 : tensor<f32>
    %2698 = stablehlo.broadcast_in_dim %2697, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2699 = stablehlo.divide %2593, %2698 : tensor<64xf32>
    %2700 = stablehlo.convert %arg167 : tensor<f32>
    %2701 = stablehlo.broadcast_in_dim %2700, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2702 = stablehlo.divide %2533, %2701 : tensor<64xf32>
    %2703 = stablehlo.convert %arg167 : tensor<f32>
    %2704 = stablehlo.broadcast_in_dim %2703, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2705 = stablehlo.divide %2541, %2704 : tensor<64xf32>
    %2706 = stablehlo.convert %arg167 : tensor<f32>
    %2707 = stablehlo.broadcast_in_dim %2706, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2708 = stablehlo.divide %2632, %2707 : tensor<3x3x64x64xf32>
    %2709 = stablehlo.convert %arg167 : tensor<f32>
    %2710 = stablehlo.broadcast_in_dim %2709, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2711 = stablehlo.divide %2579, %2710 : tensor<3x3x64x64xf32>
    %2712 = stablehlo.convert %arg167 : tensor<f32>
    %2713 = stablehlo.broadcast_in_dim %2712, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2714 = stablehlo.divide %2480, %2713 : tensor<64xf32>
    %2715 = stablehlo.convert %arg167 : tensor<f32>
    %2716 = stablehlo.broadcast_in_dim %2715, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2717 = stablehlo.divide %2488, %2716 : tensor<64xf32>
    %2718 = stablehlo.convert %arg167 : tensor<f32>
    %2719 = stablehlo.broadcast_in_dim %2718, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2720 = stablehlo.divide %2428, %2719 : tensor<64xf32>
    %2721 = stablehlo.convert %arg167 : tensor<f32>
    %2722 = stablehlo.broadcast_in_dim %2721, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2723 = stablehlo.divide %2436, %2722 : tensor<64xf32>
    %2724 = stablehlo.convert %arg167 : tensor<f32>
    %2725 = stablehlo.broadcast_in_dim %2724, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2726 = stablehlo.divide %2527, %2725 : tensor<3x3x64x64xf32>
    %2727 = stablehlo.convert %arg167 : tensor<f32>
    %2728 = stablehlo.broadcast_in_dim %2727, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2729 = stablehlo.divide %2474, %2728 : tensor<3x3x64x64xf32>
    %2730 = stablehlo.convert %arg167 : tensor<f32>
    %2731 = stablehlo.broadcast_in_dim %2730, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2732 = stablehlo.divide %2375, %2731 : tensor<128xf32>
    %2733 = stablehlo.convert %arg167 : tensor<f32>
    %2734 = stablehlo.broadcast_in_dim %2733, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2735 = stablehlo.divide %2383, %2734 : tensor<128xf32>
    %2736 = stablehlo.convert %arg167 : tensor<f32>
    %2737 = stablehlo.broadcast_in_dim %2736, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2738 = stablehlo.divide %2323, %2737 : tensor<128xf32>
    %2739 = stablehlo.convert %arg167 : tensor<f32>
    %2740 = stablehlo.broadcast_in_dim %2739, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2741 = stablehlo.divide %2331, %2740 : tensor<128xf32>
    %2742 = stablehlo.convert %arg167 : tensor<f32>
    %2743 = stablehlo.broadcast_in_dim %2742, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %2744 = stablehlo.divide %2422, %2743 : tensor<3x3x64x128xf32>
    %2745 = stablehlo.convert %arg167 : tensor<f32>
    %2746 = stablehlo.broadcast_in_dim %2745, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2747 = stablehlo.divide %2369, %2746 : tensor<3x3x128x128xf32>
    %2748 = stablehlo.convert %arg167 : tensor<f32>
    %2749 = stablehlo.broadcast_in_dim %2748, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %2750 = stablehlo.divide %2319, %2749 : tensor<1x1x64x128xf32>
    %2751 = stablehlo.convert %arg167 : tensor<f32>
    %2752 = stablehlo.broadcast_in_dim %2751, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2753 = stablehlo.divide %2273, %2752 : tensor<128xf32>
    %2754 = stablehlo.convert %arg167 : tensor<f32>
    %2755 = stablehlo.broadcast_in_dim %2754, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2756 = stablehlo.divide %2281, %2755 : tensor<128xf32>
    %2757 = stablehlo.convert %arg167 : tensor<f32>
    %2758 = stablehlo.broadcast_in_dim %2757, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2759 = stablehlo.divide %2220, %2758 : tensor<128xf32>
    %2760 = stablehlo.convert %arg167 : tensor<f32>
    %2761 = stablehlo.broadcast_in_dim %2760, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2762 = stablehlo.divide %2228, %2761 : tensor<128xf32>
    %2763 = stablehlo.convert %arg167 : tensor<f32>
    %2764 = stablehlo.broadcast_in_dim %2763, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2765 = stablehlo.divide %2168, %2764 : tensor<128xf32>
    %2766 = stablehlo.convert %arg167 : tensor<f32>
    %2767 = stablehlo.broadcast_in_dim %2766, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2768 = stablehlo.divide %2176, %2767 : tensor<128xf32>
    %2769 = stablehlo.convert %arg167 : tensor<f32>
    %2770 = stablehlo.broadcast_in_dim %2769, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2771 = stablehlo.divide %2267, %2770 : tensor<3x3x128x128xf32>
    %2772 = stablehlo.convert %arg167 : tensor<f32>
    %2773 = stablehlo.broadcast_in_dim %2772, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2774 = stablehlo.divide %2214, %2773 : tensor<3x3x128x128xf32>
    %2775 = stablehlo.convert %arg167 : tensor<f32>
    %2776 = stablehlo.broadcast_in_dim %2775, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2777 = stablehlo.divide %2115, %2776 : tensor<256xf32>
    %2778 = stablehlo.convert %arg167 : tensor<f32>
    %2779 = stablehlo.broadcast_in_dim %2778, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2780 = stablehlo.divide %2123, %2779 : tensor<256xf32>
    %2781 = stablehlo.convert %arg167 : tensor<f32>
    %2782 = stablehlo.broadcast_in_dim %2781, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2783 = stablehlo.divide %2063, %2782 : tensor<256xf32>
    %2784 = stablehlo.convert %arg167 : tensor<f32>
    %2785 = stablehlo.broadcast_in_dim %2784, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2786 = stablehlo.divide %2071, %2785 : tensor<256xf32>
    %2787 = stablehlo.convert %arg167 : tensor<f32>
    %2788 = stablehlo.broadcast_in_dim %2787, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %2789 = stablehlo.divide %2162, %2788 : tensor<3x3x128x256xf32>
    %2790 = stablehlo.convert %arg167 : tensor<f32>
    %2791 = stablehlo.broadcast_in_dim %2790, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2792 = stablehlo.divide %2109, %2791 : tensor<3x3x256x256xf32>
    %2793 = stablehlo.convert %arg167 : tensor<f32>
    %2794 = stablehlo.broadcast_in_dim %2793, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %2795 = stablehlo.divide %2059, %2794 : tensor<1x1x128x256xf32>
    %2796 = stablehlo.convert %arg167 : tensor<f32>
    %2797 = stablehlo.broadcast_in_dim %2796, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2798 = stablehlo.divide %2013, %2797 : tensor<256xf32>
    %2799 = stablehlo.convert %arg167 : tensor<f32>
    %2800 = stablehlo.broadcast_in_dim %2799, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2801 = stablehlo.divide %2021, %2800 : tensor<256xf32>
    %2802 = stablehlo.convert %arg167 : tensor<f32>
    %2803 = stablehlo.broadcast_in_dim %2802, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2804 = stablehlo.divide %1960, %2803 : tensor<256xf32>
    %2805 = stablehlo.convert %arg167 : tensor<f32>
    %2806 = stablehlo.broadcast_in_dim %2805, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2807 = stablehlo.divide %1968, %2806 : tensor<256xf32>
    %2808 = stablehlo.convert %arg167 : tensor<f32>
    %2809 = stablehlo.broadcast_in_dim %2808, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2810 = stablehlo.divide %1908, %2809 : tensor<256xf32>
    %2811 = stablehlo.convert %arg167 : tensor<f32>
    %2812 = stablehlo.broadcast_in_dim %2811, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2813 = stablehlo.divide %1916, %2812 : tensor<256xf32>
    %2814 = stablehlo.convert %arg167 : tensor<f32>
    %2815 = stablehlo.broadcast_in_dim %2814, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2816 = stablehlo.divide %2007, %2815 : tensor<3x3x256x256xf32>
    %2817 = stablehlo.convert %arg167 : tensor<f32>
    %2818 = stablehlo.broadcast_in_dim %2817, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2819 = stablehlo.divide %1954, %2818 : tensor<3x3x256x256xf32>
    %2820 = stablehlo.convert %arg167 : tensor<f32>
    %2821 = stablehlo.broadcast_in_dim %2820, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2822 = stablehlo.divide %1855, %2821 : tensor<512xf32>
    %2823 = stablehlo.convert %arg167 : tensor<f32>
    %2824 = stablehlo.broadcast_in_dim %2823, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2825 = stablehlo.divide %1863, %2824 : tensor<512xf32>
    %2826 = stablehlo.convert %arg167 : tensor<f32>
    %2827 = stablehlo.broadcast_in_dim %2826, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2828 = stablehlo.divide %1803, %2827 : tensor<512xf32>
    %2829 = stablehlo.convert %arg167 : tensor<f32>
    %2830 = stablehlo.broadcast_in_dim %2829, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2831 = stablehlo.divide %1811, %2830 : tensor<512xf32>
    %2832 = stablehlo.convert %arg167 : tensor<f32>
    %2833 = stablehlo.broadcast_in_dim %2832, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %2834 = stablehlo.divide %1902, %2833 : tensor<3x3x256x512xf32>
    %2835 = stablehlo.convert %arg167 : tensor<f32>
    %2836 = stablehlo.broadcast_in_dim %2835, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2837 = stablehlo.divide %1849, %2836 : tensor<3x3x512x512xf32>
    %2838 = stablehlo.convert %arg167 : tensor<f32>
    %2839 = stablehlo.broadcast_in_dim %2838, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %2840 = stablehlo.divide %1799, %2839 : tensor<1x1x256x512xf32>
    %2841 = stablehlo.convert %arg167 : tensor<f32>
    %2842 = stablehlo.broadcast_in_dim %2841, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2843 = stablehlo.divide %1753, %2842 : tensor<512xf32>
    %2844 = stablehlo.convert %arg167 : tensor<f32>
    %2845 = stablehlo.broadcast_in_dim %2844, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2846 = stablehlo.divide %1761, %2845 : tensor<512xf32>
    %2847 = stablehlo.convert %arg167 : tensor<f32>
    %2848 = stablehlo.broadcast_in_dim %2847, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2849 = stablehlo.divide %1700, %2848 : tensor<512xf32>
    %2850 = stablehlo.convert %arg167 : tensor<f32>
    %2851 = stablehlo.broadcast_in_dim %2850, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2852 = stablehlo.divide %1708, %2851 : tensor<512xf32>
    %2853 = stablehlo.convert %arg167 : tensor<f32>
    %2854 = stablehlo.broadcast_in_dim %2853, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2855 = stablehlo.divide %1648, %2854 : tensor<512xf32>
    %2856 = stablehlo.convert %arg167 : tensor<f32>
    %2857 = stablehlo.broadcast_in_dim %2856, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2858 = stablehlo.divide %1656, %2857 : tensor<512xf32>
    %2859 = stablehlo.convert %arg167 : tensor<f32>
    %2860 = stablehlo.broadcast_in_dim %2859, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2861 = stablehlo.divide %1747, %2860 : tensor<3x3x512x512xf32>
    %2862 = stablehlo.convert %arg167 : tensor<f32>
    %2863 = stablehlo.broadcast_in_dim %2862, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2864 = stablehlo.divide %1694, %2863 : tensor<3x3x512x512xf32>
    %2865 = stablehlo.convert %arg167 : tensor<f32>
    %2866 = stablehlo.broadcast_in_dim %2865, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2867 = stablehlo.divide %2641, %2866 : tensor<64xf32>
    %2868 = stablehlo.convert %arg167 : tensor<f32>
    %2869 = stablehlo.broadcast_in_dim %2868, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2870 = stablehlo.divide %2649, %2869 : tensor<64xf32>
    %2871 = stablehlo.convert %arg167 : tensor<f32>
    %2872 = stablehlo.broadcast_in_dim %2871, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %2873 = stablehlo.divide %2685, %2872 : tensor<7x7x3x64xf32>
    %2874 = "stablehlo.all_reduce"(%2690) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<1000xf32>) -> tensor<1000xf32>
    %2875 = "stablehlo.all_reduce"(%2693) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512x1000xf32>) -> tensor<512x1000xf32>
    %2876 = "stablehlo.all_reduce"(%2696) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2877 = "stablehlo.all_reduce"(%2699) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2878 = "stablehlo.all_reduce"(%2702) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2879 = "stablehlo.all_reduce"(%2705) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2880 = "stablehlo.all_reduce"(%2708) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %2881 = "stablehlo.all_reduce"(%2711) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %2882 = "stablehlo.all_reduce"(%2714) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2883 = "stablehlo.all_reduce"(%2717) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2884 = "stablehlo.all_reduce"(%2720) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2885 = "stablehlo.all_reduce"(%2723) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2886 = "stablehlo.all_reduce"(%2726) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %2887 = "stablehlo.all_reduce"(%2729) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %2888 = "stablehlo.all_reduce"(%2732) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2889 = "stablehlo.all_reduce"(%2735) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2890 = "stablehlo.all_reduce"(%2738) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2891 = "stablehlo.all_reduce"(%2741) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2892 = "stablehlo.all_reduce"(%2744) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
    %2893 = "stablehlo.all_reduce"(%2747) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %2894 = "stablehlo.all_reduce"(%2750) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
    %2895 = "stablehlo.all_reduce"(%2753) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2896 = "stablehlo.all_reduce"(%2756) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2897 = "stablehlo.all_reduce"(%2759) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2898 = "stablehlo.all_reduce"(%2762) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2899 = "stablehlo.all_reduce"(%2765) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2900 = "stablehlo.all_reduce"(%2768) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %2901 = "stablehlo.all_reduce"(%2771) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %2902 = "stablehlo.all_reduce"(%2774) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %2903 = "stablehlo.all_reduce"(%2777) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2904 = "stablehlo.all_reduce"(%2780) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2905 = "stablehlo.all_reduce"(%2783) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2906 = "stablehlo.all_reduce"(%2786) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2907 = "stablehlo.all_reduce"(%2789) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
    %2908 = "stablehlo.all_reduce"(%2792) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %2909 = "stablehlo.all_reduce"(%2795) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
    %2910 = "stablehlo.all_reduce"(%2798) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2911 = "stablehlo.all_reduce"(%2801) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2912 = "stablehlo.all_reduce"(%2804) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2913 = "stablehlo.all_reduce"(%2807) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2914 = "stablehlo.all_reduce"(%2810) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2915 = "stablehlo.all_reduce"(%2813) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %2916 = "stablehlo.all_reduce"(%2816) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %2917 = "stablehlo.all_reduce"(%2819) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %2918 = "stablehlo.all_reduce"(%2822) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2919 = "stablehlo.all_reduce"(%2825) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2920 = "stablehlo.all_reduce"(%2828) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2921 = "stablehlo.all_reduce"(%2831) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2922 = "stablehlo.all_reduce"(%2834) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
    %2923 = "stablehlo.all_reduce"(%2837) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %2924 = "stablehlo.all_reduce"(%2840) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    %2925 = "stablehlo.all_reduce"(%2843) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2926 = "stablehlo.all_reduce"(%2846) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2927 = "stablehlo.all_reduce"(%2849) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2928 = "stablehlo.all_reduce"(%2852) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2929 = "stablehlo.all_reduce"(%2855) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2930 = "stablehlo.all_reduce"(%2858) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %2931 = "stablehlo.all_reduce"(%2861) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %2932 = "stablehlo.all_reduce"(%2864) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %2933 = "stablehlo.all_reduce"(%2867) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2934 = "stablehlo.all_reduce"(%2870) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %2935 = "stablehlo.all_reduce"(%2873) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %2936 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %2937 = stablehlo.divide %2874, %2936 : tensor<1000xf32>
    %2938 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %2939 = stablehlo.divide %2875, %2938 : tensor<512x1000xf32>
    %2940 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2941 = stablehlo.divide %2876, %2940 : tensor<64xf32>
    %2942 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2943 = stablehlo.divide %2877, %2942 : tensor<64xf32>
    %2944 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2945 = stablehlo.divide %2878, %2944 : tensor<64xf32>
    %2946 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2947 = stablehlo.divide %2879, %2946 : tensor<64xf32>
    %2948 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2949 = stablehlo.divide %2880, %2948 : tensor<3x3x64x64xf32>
    %2950 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2951 = stablehlo.divide %2881, %2950 : tensor<3x3x64x64xf32>
    %2952 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2953 = stablehlo.divide %2882, %2952 : tensor<64xf32>
    %2954 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2955 = stablehlo.divide %2883, %2954 : tensor<64xf32>
    %2956 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2957 = stablehlo.divide %2884, %2956 : tensor<64xf32>
    %2958 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2959 = stablehlo.divide %2885, %2958 : tensor<64xf32>
    %2960 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2961 = stablehlo.divide %2886, %2960 : tensor<3x3x64x64xf32>
    %2962 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2963 = stablehlo.divide %2887, %2962 : tensor<3x3x64x64xf32>
    %2964 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2965 = stablehlo.divide %2888, %2964 : tensor<128xf32>
    %2966 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2967 = stablehlo.divide %2889, %2966 : tensor<128xf32>
    %2968 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2969 = stablehlo.divide %2890, %2968 : tensor<128xf32>
    %2970 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2971 = stablehlo.divide %2891, %2970 : tensor<128xf32>
    %2972 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %2973 = stablehlo.divide %2892, %2972 : tensor<3x3x64x128xf32>
    %2974 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2975 = stablehlo.divide %2893, %2974 : tensor<3x3x128x128xf32>
    %2976 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %2977 = stablehlo.divide %2894, %2976 : tensor<1x1x64x128xf32>
    %2978 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2979 = stablehlo.divide %2895, %2978 : tensor<128xf32>
    %2980 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2981 = stablehlo.divide %2896, %2980 : tensor<128xf32>
    %2982 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2983 = stablehlo.divide %2897, %2982 : tensor<128xf32>
    %2984 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2985 = stablehlo.divide %2898, %2984 : tensor<128xf32>
    %2986 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2987 = stablehlo.divide %2899, %2986 : tensor<128xf32>
    %2988 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2989 = stablehlo.divide %2900, %2988 : tensor<128xf32>
    %2990 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2991 = stablehlo.divide %2901, %2990 : tensor<3x3x128x128xf32>
    %2992 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2993 = stablehlo.divide %2902, %2992 : tensor<3x3x128x128xf32>
    %2994 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2995 = stablehlo.divide %2903, %2994 : tensor<256xf32>
    %2996 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2997 = stablehlo.divide %2904, %2996 : tensor<256xf32>
    %2998 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2999 = stablehlo.divide %2905, %2998 : tensor<256xf32>
    %3000 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3001 = stablehlo.divide %2906, %3000 : tensor<256xf32>
    %3002 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %3003 = stablehlo.divide %2907, %3002 : tensor<3x3x128x256xf32>
    %3004 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3005 = stablehlo.divide %2908, %3004 : tensor<3x3x256x256xf32>
    %3006 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %3007 = stablehlo.divide %2909, %3006 : tensor<1x1x128x256xf32>
    %3008 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3009 = stablehlo.divide %2910, %3008 : tensor<256xf32>
    %3010 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3011 = stablehlo.divide %2911, %3010 : tensor<256xf32>
    %3012 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3013 = stablehlo.divide %2912, %3012 : tensor<256xf32>
    %3014 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3015 = stablehlo.divide %2913, %3014 : tensor<256xf32>
    %3016 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3017 = stablehlo.divide %2914, %3016 : tensor<256xf32>
    %3018 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3019 = stablehlo.divide %2915, %3018 : tensor<256xf32>
    %3020 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3021 = stablehlo.divide %2916, %3020 : tensor<3x3x256x256xf32>
    %3022 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3023 = stablehlo.divide %2917, %3022 : tensor<3x3x256x256xf32>
    %3024 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3025 = stablehlo.divide %2918, %3024 : tensor<512xf32>
    %3026 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3027 = stablehlo.divide %2919, %3026 : tensor<512xf32>
    %3028 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3029 = stablehlo.divide %2920, %3028 : tensor<512xf32>
    %3030 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3031 = stablehlo.divide %2921, %3030 : tensor<512xf32>
    %3032 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %3033 = stablehlo.divide %2922, %3032 : tensor<3x3x256x512xf32>
    %3034 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3035 = stablehlo.divide %2923, %3034 : tensor<3x3x512x512xf32>
    %3036 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %3037 = stablehlo.divide %2924, %3036 : tensor<1x1x256x512xf32>
    %3038 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3039 = stablehlo.divide %2925, %3038 : tensor<512xf32>
    %3040 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3041 = stablehlo.divide %2926, %3040 : tensor<512xf32>
    %3042 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3043 = stablehlo.divide %2927, %3042 : tensor<512xf32>
    %3044 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3045 = stablehlo.divide %2928, %3044 : tensor<512xf32>
    %3046 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3047 = stablehlo.divide %2929, %3046 : tensor<512xf32>
    %3048 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3049 = stablehlo.divide %2930, %3048 : tensor<512xf32>
    %3050 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3051 = stablehlo.divide %2931, %3050 : tensor<3x3x512x512xf32>
    %3052 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3053 = stablehlo.divide %2932, %3052 : tensor<3x3x512x512xf32>
    %3054 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3055 = stablehlo.divide %2933, %3054 : tensor<64xf32>
    %3056 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3057 = stablehlo.divide %2934, %3056 : tensor<64xf32>
    %3058 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %3059 = stablehlo.divide %2935, %3058 : tensor<7x7x3x64xf32>
    %3060 = stablehlo.is_finite %2937 : (tensor<1000xf32>) -> tensor<1000xi1>
    %c_232 = stablehlo.constant dense<true> : tensor<i1>
    %3061 = stablehlo.reduce(%3060 init: %c_232) applies stablehlo.and across dimensions = [0] : (tensor<1000xi1>, tensor<i1>) -> tensor<i1>
    %c_233 = stablehlo.constant dense<true> : tensor<i1>
    %3062 = stablehlo.and %c_233, %3061 : tensor<i1>
    %3063 = stablehlo.is_finite %2939 : (tensor<512x1000xf32>) -> tensor<512x1000xi1>
    %c_234 = stablehlo.constant dense<true> : tensor<i1>
    %3064 = stablehlo.reduce(%3063 init: %c_234) applies stablehlo.and across dimensions = [0, 1] : (tensor<512x1000xi1>, tensor<i1>) -> tensor<i1>
    %3065 = stablehlo.and %3062, %3064 : tensor<i1>
    %3066 = stablehlo.is_finite %2941 : (tensor<64xf32>) -> tensor<64xi1>
    %c_235 = stablehlo.constant dense<true> : tensor<i1>
    %3067 = stablehlo.reduce(%3066 init: %c_235) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3068 = stablehlo.and %3065, %3067 : tensor<i1>
    %3069 = stablehlo.is_finite %2943 : (tensor<64xf32>) -> tensor<64xi1>
    %c_236 = stablehlo.constant dense<true> : tensor<i1>
    %3070 = stablehlo.reduce(%3069 init: %c_236) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3071 = stablehlo.and %3068, %3070 : tensor<i1>
    %3072 = stablehlo.is_finite %2945 : (tensor<64xf32>) -> tensor<64xi1>
    %c_237 = stablehlo.constant dense<true> : tensor<i1>
    %3073 = stablehlo.reduce(%3072 init: %c_237) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3074 = stablehlo.and %3071, %3073 : tensor<i1>
    %3075 = stablehlo.is_finite %2947 : (tensor<64xf32>) -> tensor<64xi1>
    %c_238 = stablehlo.constant dense<true> : tensor<i1>
    %3076 = stablehlo.reduce(%3075 init: %c_238) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3077 = stablehlo.and %3074, %3076 : tensor<i1>
    %3078 = stablehlo.is_finite %2949 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_239 = stablehlo.constant dense<true> : tensor<i1>
    %3079 = stablehlo.reduce(%3078 init: %c_239) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %3080 = stablehlo.and %3077, %3079 : tensor<i1>
    %3081 = stablehlo.is_finite %2951 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_240 = stablehlo.constant dense<true> : tensor<i1>
    %3082 = stablehlo.reduce(%3081 init: %c_240) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %3083 = stablehlo.and %3080, %3082 : tensor<i1>
    %3084 = stablehlo.is_finite %2953 : (tensor<64xf32>) -> tensor<64xi1>
    %c_241 = stablehlo.constant dense<true> : tensor<i1>
    %3085 = stablehlo.reduce(%3084 init: %c_241) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3086 = stablehlo.and %3083, %3085 : tensor<i1>
    %3087 = stablehlo.is_finite %2955 : (tensor<64xf32>) -> tensor<64xi1>
    %c_242 = stablehlo.constant dense<true> : tensor<i1>
    %3088 = stablehlo.reduce(%3087 init: %c_242) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3089 = stablehlo.and %3086, %3088 : tensor<i1>
    %3090 = stablehlo.is_finite %2957 : (tensor<64xf32>) -> tensor<64xi1>
    %c_243 = stablehlo.constant dense<true> : tensor<i1>
    %3091 = stablehlo.reduce(%3090 init: %c_243) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3092 = stablehlo.and %3089, %3091 : tensor<i1>
    %3093 = stablehlo.is_finite %2959 : (tensor<64xf32>) -> tensor<64xi1>
    %c_244 = stablehlo.constant dense<true> : tensor<i1>
    %3094 = stablehlo.reduce(%3093 init: %c_244) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3095 = stablehlo.and %3092, %3094 : tensor<i1>
    %3096 = stablehlo.is_finite %2961 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_245 = stablehlo.constant dense<true> : tensor<i1>
    %3097 = stablehlo.reduce(%3096 init: %c_245) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %3098 = stablehlo.and %3095, %3097 : tensor<i1>
    %3099 = stablehlo.is_finite %2963 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_246 = stablehlo.constant dense<true> : tensor<i1>
    %3100 = stablehlo.reduce(%3099 init: %c_246) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %3101 = stablehlo.and %3098, %3100 : tensor<i1>
    %3102 = stablehlo.is_finite %2965 : (tensor<128xf32>) -> tensor<128xi1>
    %c_247 = stablehlo.constant dense<true> : tensor<i1>
    %3103 = stablehlo.reduce(%3102 init: %c_247) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3104 = stablehlo.and %3101, %3103 : tensor<i1>
    %3105 = stablehlo.is_finite %2967 : (tensor<128xf32>) -> tensor<128xi1>
    %c_248 = stablehlo.constant dense<true> : tensor<i1>
    %3106 = stablehlo.reduce(%3105 init: %c_248) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3107 = stablehlo.and %3104, %3106 : tensor<i1>
    %3108 = stablehlo.is_finite %2969 : (tensor<128xf32>) -> tensor<128xi1>
    %c_249 = stablehlo.constant dense<true> : tensor<i1>
    %3109 = stablehlo.reduce(%3108 init: %c_249) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3110 = stablehlo.and %3107, %3109 : tensor<i1>
    %3111 = stablehlo.is_finite %2971 : (tensor<128xf32>) -> tensor<128xi1>
    %c_250 = stablehlo.constant dense<true> : tensor<i1>
    %3112 = stablehlo.reduce(%3111 init: %c_250) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3113 = stablehlo.and %3110, %3112 : tensor<i1>
    %3114 = stablehlo.is_finite %2973 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xi1>
    %c_251 = stablehlo.constant dense<true> : tensor<i1>
    %3115 = stablehlo.reduce(%3114 init: %c_251) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x128xi1>, tensor<i1>) -> tensor<i1>
    %3116 = stablehlo.and %3113, %3115 : tensor<i1>
    %3117 = stablehlo.is_finite %2975 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_252 = stablehlo.constant dense<true> : tensor<i1>
    %3118 = stablehlo.reduce(%3117 init: %c_252) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %3119 = stablehlo.and %3116, %3118 : tensor<i1>
    %3120 = stablehlo.is_finite %2977 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xi1>
    %c_253 = stablehlo.constant dense<true> : tensor<i1>
    %3121 = stablehlo.reduce(%3120 init: %c_253) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x64x128xi1>, tensor<i1>) -> tensor<i1>
    %3122 = stablehlo.and %3119, %3121 : tensor<i1>
    %3123 = stablehlo.is_finite %2979 : (tensor<128xf32>) -> tensor<128xi1>
    %c_254 = stablehlo.constant dense<true> : tensor<i1>
    %3124 = stablehlo.reduce(%3123 init: %c_254) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3125 = stablehlo.and %3122, %3124 : tensor<i1>
    %3126 = stablehlo.is_finite %2981 : (tensor<128xf32>) -> tensor<128xi1>
    %c_255 = stablehlo.constant dense<true> : tensor<i1>
    %3127 = stablehlo.reduce(%3126 init: %c_255) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3128 = stablehlo.and %3125, %3127 : tensor<i1>
    %3129 = stablehlo.is_finite %2983 : (tensor<128xf32>) -> tensor<128xi1>
    %c_256 = stablehlo.constant dense<true> : tensor<i1>
    %3130 = stablehlo.reduce(%3129 init: %c_256) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3131 = stablehlo.and %3128, %3130 : tensor<i1>
    %3132 = stablehlo.is_finite %2985 : (tensor<128xf32>) -> tensor<128xi1>
    %c_257 = stablehlo.constant dense<true> : tensor<i1>
    %3133 = stablehlo.reduce(%3132 init: %c_257) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3134 = stablehlo.and %3131, %3133 : tensor<i1>
    %3135 = stablehlo.is_finite %2987 : (tensor<128xf32>) -> tensor<128xi1>
    %c_258 = stablehlo.constant dense<true> : tensor<i1>
    %3136 = stablehlo.reduce(%3135 init: %c_258) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3137 = stablehlo.and %3134, %3136 : tensor<i1>
    %3138 = stablehlo.is_finite %2989 : (tensor<128xf32>) -> tensor<128xi1>
    %c_259 = stablehlo.constant dense<true> : tensor<i1>
    %3139 = stablehlo.reduce(%3138 init: %c_259) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %3140 = stablehlo.and %3137, %3139 : tensor<i1>
    %3141 = stablehlo.is_finite %2991 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_260 = stablehlo.constant dense<true> : tensor<i1>
    %3142 = stablehlo.reduce(%3141 init: %c_260) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %3143 = stablehlo.and %3140, %3142 : tensor<i1>
    %3144 = stablehlo.is_finite %2993 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_261 = stablehlo.constant dense<true> : tensor<i1>
    %3145 = stablehlo.reduce(%3144 init: %c_261) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %3146 = stablehlo.and %3143, %3145 : tensor<i1>
    %3147 = stablehlo.is_finite %2995 : (tensor<256xf32>) -> tensor<256xi1>
    %c_262 = stablehlo.constant dense<true> : tensor<i1>
    %3148 = stablehlo.reduce(%3147 init: %c_262) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3149 = stablehlo.and %3146, %3148 : tensor<i1>
    %3150 = stablehlo.is_finite %2997 : (tensor<256xf32>) -> tensor<256xi1>
    %c_263 = stablehlo.constant dense<true> : tensor<i1>
    %3151 = stablehlo.reduce(%3150 init: %c_263) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3152 = stablehlo.and %3149, %3151 : tensor<i1>
    %3153 = stablehlo.is_finite %2999 : (tensor<256xf32>) -> tensor<256xi1>
    %c_264 = stablehlo.constant dense<true> : tensor<i1>
    %3154 = stablehlo.reduce(%3153 init: %c_264) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3155 = stablehlo.and %3152, %3154 : tensor<i1>
    %3156 = stablehlo.is_finite %3001 : (tensor<256xf32>) -> tensor<256xi1>
    %c_265 = stablehlo.constant dense<true> : tensor<i1>
    %3157 = stablehlo.reduce(%3156 init: %c_265) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3158 = stablehlo.and %3155, %3157 : tensor<i1>
    %3159 = stablehlo.is_finite %3003 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xi1>
    %c_266 = stablehlo.constant dense<true> : tensor<i1>
    %3160 = stablehlo.reduce(%3159 init: %c_266) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x256xi1>, tensor<i1>) -> tensor<i1>
    %3161 = stablehlo.and %3158, %3160 : tensor<i1>
    %3162 = stablehlo.is_finite %3005 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_267 = stablehlo.constant dense<true> : tensor<i1>
    %3163 = stablehlo.reduce(%3162 init: %c_267) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %3164 = stablehlo.and %3161, %3163 : tensor<i1>
    %3165 = stablehlo.is_finite %3007 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xi1>
    %c_268 = stablehlo.constant dense<true> : tensor<i1>
    %3166 = stablehlo.reduce(%3165 init: %c_268) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x128x256xi1>, tensor<i1>) -> tensor<i1>
    %3167 = stablehlo.and %3164, %3166 : tensor<i1>
    %3168 = stablehlo.is_finite %3009 : (tensor<256xf32>) -> tensor<256xi1>
    %c_269 = stablehlo.constant dense<true> : tensor<i1>
    %3169 = stablehlo.reduce(%3168 init: %c_269) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3170 = stablehlo.and %3167, %3169 : tensor<i1>
    %3171 = stablehlo.is_finite %3011 : (tensor<256xf32>) -> tensor<256xi1>
    %c_270 = stablehlo.constant dense<true> : tensor<i1>
    %3172 = stablehlo.reduce(%3171 init: %c_270) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3173 = stablehlo.and %3170, %3172 : tensor<i1>
    %3174 = stablehlo.is_finite %3013 : (tensor<256xf32>) -> tensor<256xi1>
    %c_271 = stablehlo.constant dense<true> : tensor<i1>
    %3175 = stablehlo.reduce(%3174 init: %c_271) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3176 = stablehlo.and %3173, %3175 : tensor<i1>
    %3177 = stablehlo.is_finite %3015 : (tensor<256xf32>) -> tensor<256xi1>
    %c_272 = stablehlo.constant dense<true> : tensor<i1>
    %3178 = stablehlo.reduce(%3177 init: %c_272) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3179 = stablehlo.and %3176, %3178 : tensor<i1>
    %3180 = stablehlo.is_finite %3017 : (tensor<256xf32>) -> tensor<256xi1>
    %c_273 = stablehlo.constant dense<true> : tensor<i1>
    %3181 = stablehlo.reduce(%3180 init: %c_273) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3182 = stablehlo.and %3179, %3181 : tensor<i1>
    %3183 = stablehlo.is_finite %3019 : (tensor<256xf32>) -> tensor<256xi1>
    %c_274 = stablehlo.constant dense<true> : tensor<i1>
    %3184 = stablehlo.reduce(%3183 init: %c_274) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %3185 = stablehlo.and %3182, %3184 : tensor<i1>
    %3186 = stablehlo.is_finite %3021 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_275 = stablehlo.constant dense<true> : tensor<i1>
    %3187 = stablehlo.reduce(%3186 init: %c_275) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %3188 = stablehlo.and %3185, %3187 : tensor<i1>
    %3189 = stablehlo.is_finite %3023 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_276 = stablehlo.constant dense<true> : tensor<i1>
    %3190 = stablehlo.reduce(%3189 init: %c_276) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %3191 = stablehlo.and %3188, %3190 : tensor<i1>
    %3192 = stablehlo.is_finite %3025 : (tensor<512xf32>) -> tensor<512xi1>
    %c_277 = stablehlo.constant dense<true> : tensor<i1>
    %3193 = stablehlo.reduce(%3192 init: %c_277) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3194 = stablehlo.and %3191, %3193 : tensor<i1>
    %3195 = stablehlo.is_finite %3027 : (tensor<512xf32>) -> tensor<512xi1>
    %c_278 = stablehlo.constant dense<true> : tensor<i1>
    %3196 = stablehlo.reduce(%3195 init: %c_278) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3197 = stablehlo.and %3194, %3196 : tensor<i1>
    %3198 = stablehlo.is_finite %3029 : (tensor<512xf32>) -> tensor<512xi1>
    %c_279 = stablehlo.constant dense<true> : tensor<i1>
    %3199 = stablehlo.reduce(%3198 init: %c_279) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3200 = stablehlo.and %3197, %3199 : tensor<i1>
    %3201 = stablehlo.is_finite %3031 : (tensor<512xf32>) -> tensor<512xi1>
    %c_280 = stablehlo.constant dense<true> : tensor<i1>
    %3202 = stablehlo.reduce(%3201 init: %c_280) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3203 = stablehlo.and %3200, %3202 : tensor<i1>
    %3204 = stablehlo.is_finite %3033 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xi1>
    %c_281 = stablehlo.constant dense<true> : tensor<i1>
    %3205 = stablehlo.reduce(%3204 init: %c_281) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x512xi1>, tensor<i1>) -> tensor<i1>
    %3206 = stablehlo.and %3203, %3205 : tensor<i1>
    %3207 = stablehlo.is_finite %3035 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
    %c_282 = stablehlo.constant dense<true> : tensor<i1>
    %3208 = stablehlo.reduce(%3207 init: %c_282) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
    %3209 = stablehlo.and %3206, %3208 : tensor<i1>
    %3210 = stablehlo.is_finite %3037 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xi1>
    %c_283 = stablehlo.constant dense<true> : tensor<i1>
    %3211 = stablehlo.reduce(%3210 init: %c_283) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x256x512xi1>, tensor<i1>) -> tensor<i1>
    %3212 = stablehlo.and %3209, %3211 : tensor<i1>
    %3213 = stablehlo.is_finite %3039 : (tensor<512xf32>) -> tensor<512xi1>
    %c_284 = stablehlo.constant dense<true> : tensor<i1>
    %3214 = stablehlo.reduce(%3213 init: %c_284) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3215 = stablehlo.and %3212, %3214 : tensor<i1>
    %3216 = stablehlo.is_finite %3041 : (tensor<512xf32>) -> tensor<512xi1>
    %c_285 = stablehlo.constant dense<true> : tensor<i1>
    %3217 = stablehlo.reduce(%3216 init: %c_285) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3218 = stablehlo.and %3215, %3217 : tensor<i1>
    %3219 = stablehlo.is_finite %3043 : (tensor<512xf32>) -> tensor<512xi1>
    %c_286 = stablehlo.constant dense<true> : tensor<i1>
    %3220 = stablehlo.reduce(%3219 init: %c_286) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3221 = stablehlo.and %3218, %3220 : tensor<i1>
    %3222 = stablehlo.is_finite %3045 : (tensor<512xf32>) -> tensor<512xi1>
    %c_287 = stablehlo.constant dense<true> : tensor<i1>
    %3223 = stablehlo.reduce(%3222 init: %c_287) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3224 = stablehlo.and %3221, %3223 : tensor<i1>
    %3225 = stablehlo.is_finite %3047 : (tensor<512xf32>) -> tensor<512xi1>
    %c_288 = stablehlo.constant dense<true> : tensor<i1>
    %3226 = stablehlo.reduce(%3225 init: %c_288) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3227 = stablehlo.and %3224, %3226 : tensor<i1>
    %3228 = stablehlo.is_finite %3049 : (tensor<512xf32>) -> tensor<512xi1>
    %c_289 = stablehlo.constant dense<true> : tensor<i1>
    %3229 = stablehlo.reduce(%3228 init: %c_289) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %3230 = stablehlo.and %3227, %3229 : tensor<i1>
    %3231 = stablehlo.is_finite %3051 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
    %c_290 = stablehlo.constant dense<true> : tensor<i1>
    %3232 = stablehlo.reduce(%3231 init: %c_290) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
    %3233 = stablehlo.and %3230, %3232 : tensor<i1>
    %3234 = stablehlo.is_finite %3053 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
    %c_291 = stablehlo.constant dense<true> : tensor<i1>
    %3235 = stablehlo.reduce(%3234 init: %c_291) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
    %3236 = stablehlo.and %3233, %3235 : tensor<i1>
    %3237 = stablehlo.is_finite %3055 : (tensor<64xf32>) -> tensor<64xi1>
    %c_292 = stablehlo.constant dense<true> : tensor<i1>
    %3238 = stablehlo.reduce(%3237 init: %c_292) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3239 = stablehlo.and %3236, %3238 : tensor<i1>
    %3240 = stablehlo.is_finite %3057 : (tensor<64xf32>) -> tensor<64xi1>
    %c_293 = stablehlo.constant dense<true> : tensor<i1>
    %3241 = stablehlo.reduce(%3240 init: %c_293) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %3242 = stablehlo.and %3239, %3241 : tensor<i1>
    %3243 = stablehlo.is_finite %3059 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xi1>
    %c_294 = stablehlo.constant dense<true> : tensor<i1>
    %3244 = stablehlo.reduce(%3243 init: %c_294) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<7x7x3x64xi1>, tensor<i1>) -> tensor<i1>
    %3245 = stablehlo.and %3242, %3244 : tensor<i1>
    %c_295 = stablehlo.constant dense<2000> : tensor<i32>
    %3246 = stablehlo.compare  EQ, %arg166, %c_295,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_296 = stablehlo.constant dense<false> : tensor<i1>
    %3247 = stablehlo.broadcast_in_dim %c_296, dims = [] : (tensor<i1>) -> tensor<i1>
    %3248 = stablehlo.compare  NE, %3246, %3247,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3249 = stablehlo.convert %3248 : tensor<i1>
    %3250 = stablehlo.and %3249, %3245 : tensor<i1>
    %cst_297 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %3251 = stablehlo.multiply %arg167, %cst_297 : tensor<f32>
    %3252 = stablehlo.convert %3251 : tensor<f32>
    %cst_298 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %3253 = stablehlo.minimum %3252, %cst_298 : tensor<f32>
    %3254 = call @_where_490(%3250, %3253, %arg167) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %3255 = stablehlo.multiply %arg167, %cst_5 : tensor<f32>
    %3256 = stablehlo.convert %3255 : tensor<f32>
    %cst_299 = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
    %3257 = stablehlo.maximum %3256, %cst_299 : tensor<f32>
    %3258 = call @_where_493(%3245, %3254, %3257) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %3259 = stablehlo.not %3245 : tensor<i1>
    %c_300 = stablehlo.constant dense<false> : tensor<i1>
    %3260 = stablehlo.broadcast_in_dim %c_300, dims = [] : (tensor<i1>) -> tensor<i1>
    %3261 = stablehlo.compare  NE, %3246, %3260,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3262 = stablehlo.convert %3261 : tensor<i1>
    %3263 = stablehlo.or %3262, %3259 : tensor<i1>
    %c_301 = stablehlo.constant dense<1> : tensor<i32>
    %3264 = stablehlo.add %arg166, %c_301 : tensor<i32>
    %3265 = call @_where_495(%3263, %c, %3264) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %3266 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %3267 = stablehlo.iota dim = 0 : tensor<1000xi32>
    %3268 = stablehlo.reshape %3267 : (tensor<1000xi32>) -> tensor<1x1000xi32>
    %3269 = stablehlo.broadcast_in_dim %3266, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
    %3270 = stablehlo.broadcast_in_dim %3268, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
    %3271 = stablehlo.compare  EQ, %3269, %3270,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
    %3272 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
    %3273 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
    %3274 = stablehlo.select %3271, %3272, %3273 : tensor<256x1000xi1>, tensor<256x1000xf32>
    %3275 = stablehlo.convert %3274 : tensor<256x1000xf32>
    %3276 = call @log_softmax_497(%1451) : (tensor<256x1000xbf16>) -> tensor<256x1000xbf16>
    %3277 = stablehlo.convert %3276 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
    %3278 = stablehlo.multiply %3275, %3277 : tensor<256x1000xf32>
    %cst_302 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3279 = stablehlo.reduce(%3278 init: %cst_302) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %3280 = stablehlo.negate %3279 : tensor<256xf32>
    %cst_303 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3281 = stablehlo.reduce(%3280 init: %cst_303) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %3282 = stablehlo.divide %3281, %cst_66 : tensor<f32>
    %3283 = call @argmax(%1451) : (tensor<256x1000xbf16>) -> tensor<256xi32>
    %3284 = stablehlo.compare  EQ, %3283, %1,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %3285 = stablehlo.convert %3284 : (tensor<256xi1>) -> tensor<256xi32>
    %3286 = stablehlo.convert %3285 : (tensor<256xi32>) -> tensor<256xf32>
    %cst_304 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3287 = stablehlo.reduce(%3286 init: %cst_304) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %3288 = stablehlo.divide %3287, %cst_66 : tensor<f32>
    %3289 = "stablehlo.all_reduce"(%3288) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %3290 = "stablehlo.all_reduce"(%3282) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %3291 = stablehlo.divide %3289, %cst_11 : tensor<f32>
    %3292 = stablehlo.divide %3290, %cst_11 : tensor<f32>
    %3293 = "stablehlo.all_reduce"(%138) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3294 = "stablehlo.all_reduce"(%143) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3295 = "stablehlo.all_reduce"(%209) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3296 = "stablehlo.all_reduce"(%214) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3297 = "stablehlo.all_reduce"(%281) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3298 = "stablehlo.all_reduce"(%286) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3299 = "stablehlo.all_reduce"(%352) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3300 = "stablehlo.all_reduce"(%357) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3301 = "stablehlo.all_reduce"(%424) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3302 = "stablehlo.all_reduce"(%429) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3303 = "stablehlo.all_reduce"(%495) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3304 = "stablehlo.all_reduce"(%500) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3305 = "stablehlo.all_reduce"(%562) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3306 = "stablehlo.all_reduce"(%567) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3307 = "stablehlo.all_reduce"(%634) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3308 = "stablehlo.all_reduce"(%639) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3309 = "stablehlo.all_reduce"(%705) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3310 = "stablehlo.all_reduce"(%710) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %3311 = "stablehlo.all_reduce"(%777) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3312 = "stablehlo.all_reduce"(%782) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3313 = "stablehlo.all_reduce"(%848) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3314 = "stablehlo.all_reduce"(%853) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3315 = "stablehlo.all_reduce"(%915) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3316 = "stablehlo.all_reduce"(%920) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3317 = "stablehlo.all_reduce"(%987) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3318 = "stablehlo.all_reduce"(%992) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3319 = "stablehlo.all_reduce"(%1058) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3320 = "stablehlo.all_reduce"(%1063) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %3321 = "stablehlo.all_reduce"(%1130) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3322 = "stablehlo.all_reduce"(%1135) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3323 = "stablehlo.all_reduce"(%1201) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3324 = "stablehlo.all_reduce"(%1206) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3325 = "stablehlo.all_reduce"(%1268) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3326 = "stablehlo.all_reduce"(%1273) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3327 = "stablehlo.all_reduce"(%1340) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3328 = "stablehlo.all_reduce"(%1345) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3329 = "stablehlo.all_reduce"(%1411) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3330 = "stablehlo.all_reduce"(%1416) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %3331 = "stablehlo.all_reduce"(%65) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3332 = "stablehlo.all_reduce"(%70) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg170: tensor<f32>, %arg171: tensor<f32>):
      %4188 = stablehlo.add %arg170, %arg171 : tensor<f32>
      stablehlo.return %4188 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %3333 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3334 = stablehlo.divide %3293, %3333 : tensor<64xf32>
    %3335 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3336 = stablehlo.divide %3294, %3335 : tensor<64xf32>
    %3337 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3338 = stablehlo.divide %3295, %3337 : tensor<64xf32>
    %3339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3340 = stablehlo.divide %3296, %3339 : tensor<64xf32>
    %3341 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3342 = stablehlo.divide %3297, %3341 : tensor<64xf32>
    %3343 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3344 = stablehlo.divide %3298, %3343 : tensor<64xf32>
    %3345 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3346 = stablehlo.divide %3299, %3345 : tensor<64xf32>
    %3347 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3348 = stablehlo.divide %3300, %3347 : tensor<64xf32>
    %3349 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3350 = stablehlo.divide %3301, %3349 : tensor<128xf32>
    %3351 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3352 = stablehlo.divide %3302, %3351 : tensor<128xf32>
    %3353 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3354 = stablehlo.divide %3303, %3353 : tensor<128xf32>
    %3355 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3356 = stablehlo.divide %3304, %3355 : tensor<128xf32>
    %3357 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3358 = stablehlo.divide %3305, %3357 : tensor<128xf32>
    %3359 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3360 = stablehlo.divide %3306, %3359 : tensor<128xf32>
    %3361 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3362 = stablehlo.divide %3307, %3361 : tensor<128xf32>
    %3363 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3364 = stablehlo.divide %3308, %3363 : tensor<128xf32>
    %3365 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3366 = stablehlo.divide %3309, %3365 : tensor<128xf32>
    %3367 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3368 = stablehlo.divide %3310, %3367 : tensor<128xf32>
    %3369 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3370 = stablehlo.divide %3311, %3369 : tensor<256xf32>
    %3371 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3372 = stablehlo.divide %3312, %3371 : tensor<256xf32>
    %3373 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3374 = stablehlo.divide %3313, %3373 : tensor<256xf32>
    %3375 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3376 = stablehlo.divide %3314, %3375 : tensor<256xf32>
    %3377 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3378 = stablehlo.divide %3315, %3377 : tensor<256xf32>
    %3379 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3380 = stablehlo.divide %3316, %3379 : tensor<256xf32>
    %3381 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3382 = stablehlo.divide %3317, %3381 : tensor<256xf32>
    %3383 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3384 = stablehlo.divide %3318, %3383 : tensor<256xf32>
    %3385 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3386 = stablehlo.divide %3319, %3385 : tensor<256xf32>
    %3387 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3388 = stablehlo.divide %3320, %3387 : tensor<256xf32>
    %3389 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3390 = stablehlo.divide %3321, %3389 : tensor<512xf32>
    %3391 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3392 = stablehlo.divide %3322, %3391 : tensor<512xf32>
    %3393 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3394 = stablehlo.divide %3323, %3393 : tensor<512xf32>
    %3395 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3396 = stablehlo.divide %3324, %3395 : tensor<512xf32>
    %3397 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3398 = stablehlo.divide %3325, %3397 : tensor<512xf32>
    %3399 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3400 = stablehlo.divide %3326, %3399 : tensor<512xf32>
    %3401 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3402 = stablehlo.divide %3327, %3401 : tensor<512xf32>
    %3403 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3404 = stablehlo.divide %3328, %3403 : tensor<512xf32>
    %3405 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3406 = stablehlo.divide %3329, %3405 : tensor<512xf32>
    %3407 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3408 = stablehlo.divide %3330, %3407 : tensor<512xf32>
    %3409 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3410 = stablehlo.divide %3331, %3409 : tensor<64xf32>
    %3411 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3412 = stablehlo.divide %3332, %3411 : tensor<64xf32>
    %3413 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %3414 = stablehlo.multiply %3413, %arg63 : tensor<1000xf32>
    %3415 = stablehlo.add %2937, %3414 : tensor<1000xf32>
    %3416 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %3417 = stablehlo.multiply %3416, %arg64 : tensor<512x1000xf32>
    %3418 = stablehlo.add %2939, %3417 : tensor<512x1000xf32>
    %3419 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3420 = stablehlo.multiply %3419, %arg65 : tensor<64xf32>
    %3421 = stablehlo.add %2941, %3420 : tensor<64xf32>
    %3422 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3423 = stablehlo.multiply %3422, %arg66 : tensor<64xf32>
    %3424 = stablehlo.add %2943, %3423 : tensor<64xf32>
    %3425 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3426 = stablehlo.multiply %3425, %arg67 : tensor<64xf32>
    %3427 = stablehlo.add %2945, %3426 : tensor<64xf32>
    %3428 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3429 = stablehlo.multiply %3428, %arg68 : tensor<64xf32>
    %3430 = stablehlo.add %2947, %3429 : tensor<64xf32>
    %3431 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3432 = stablehlo.multiply %3431, %arg69 : tensor<3x3x64x64xf32>
    %3433 = stablehlo.add %2949, %3432 : tensor<3x3x64x64xf32>
    %3434 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3435 = stablehlo.multiply %3434, %arg70 : tensor<3x3x64x64xf32>
    %3436 = stablehlo.add %2951, %3435 : tensor<3x3x64x64xf32>
    %3437 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3438 = stablehlo.multiply %3437, %arg71 : tensor<64xf32>
    %3439 = stablehlo.add %2953, %3438 : tensor<64xf32>
    %3440 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3441 = stablehlo.multiply %3440, %arg72 : tensor<64xf32>
    %3442 = stablehlo.add %2955, %3441 : tensor<64xf32>
    %3443 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3444 = stablehlo.multiply %3443, %arg73 : tensor<64xf32>
    %3445 = stablehlo.add %2957, %3444 : tensor<64xf32>
    %3446 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3447 = stablehlo.multiply %3446, %arg74 : tensor<64xf32>
    %3448 = stablehlo.add %2959, %3447 : tensor<64xf32>
    %3449 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3450 = stablehlo.multiply %3449, %arg75 : tensor<3x3x64x64xf32>
    %3451 = stablehlo.add %2961, %3450 : tensor<3x3x64x64xf32>
    %3452 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3453 = stablehlo.multiply %3452, %arg76 : tensor<3x3x64x64xf32>
    %3454 = stablehlo.add %2963, %3453 : tensor<3x3x64x64xf32>
    %3455 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3456 = stablehlo.multiply %3455, %arg77 : tensor<128xf32>
    %3457 = stablehlo.add %2965, %3456 : tensor<128xf32>
    %3458 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3459 = stablehlo.multiply %3458, %arg78 : tensor<128xf32>
    %3460 = stablehlo.add %2967, %3459 : tensor<128xf32>
    %3461 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3462 = stablehlo.multiply %3461, %arg79 : tensor<128xf32>
    %3463 = stablehlo.add %2969, %3462 : tensor<128xf32>
    %3464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3465 = stablehlo.multiply %3464, %arg80 : tensor<128xf32>
    %3466 = stablehlo.add %2971, %3465 : tensor<128xf32>
    %3467 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %3468 = stablehlo.multiply %3467, %arg81 : tensor<3x3x64x128xf32>
    %3469 = stablehlo.add %2973, %3468 : tensor<3x3x64x128xf32>
    %3470 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %3471 = stablehlo.multiply %3470, %arg82 : tensor<3x3x128x128xf32>
    %3472 = stablehlo.add %2975, %3471 : tensor<3x3x128x128xf32>
    %3473 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %3474 = stablehlo.multiply %3473, %arg83 : tensor<1x1x64x128xf32>
    %3475 = stablehlo.add %2977, %3474 : tensor<1x1x64x128xf32>
    %3476 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3477 = stablehlo.multiply %3476, %arg84 : tensor<128xf32>
    %3478 = stablehlo.add %2979, %3477 : tensor<128xf32>
    %3479 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3480 = stablehlo.multiply %3479, %arg85 : tensor<128xf32>
    %3481 = stablehlo.add %2981, %3480 : tensor<128xf32>
    %3482 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3483 = stablehlo.multiply %3482, %arg86 : tensor<128xf32>
    %3484 = stablehlo.add %2983, %3483 : tensor<128xf32>
    %3485 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3486 = stablehlo.multiply %3485, %arg87 : tensor<128xf32>
    %3487 = stablehlo.add %2985, %3486 : tensor<128xf32>
    %3488 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3489 = stablehlo.multiply %3488, %arg88 : tensor<128xf32>
    %3490 = stablehlo.add %2987, %3489 : tensor<128xf32>
    %3491 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3492 = stablehlo.multiply %3491, %arg89 : tensor<128xf32>
    %3493 = stablehlo.add %2989, %3492 : tensor<128xf32>
    %3494 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %3495 = stablehlo.multiply %3494, %arg90 : tensor<3x3x128x128xf32>
    %3496 = stablehlo.add %2991, %3495 : tensor<3x3x128x128xf32>
    %3497 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %3498 = stablehlo.multiply %3497, %arg91 : tensor<3x3x128x128xf32>
    %3499 = stablehlo.add %2993, %3498 : tensor<3x3x128x128xf32>
    %3500 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3501 = stablehlo.multiply %3500, %arg92 : tensor<256xf32>
    %3502 = stablehlo.add %2995, %3501 : tensor<256xf32>
    %3503 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3504 = stablehlo.multiply %3503, %arg93 : tensor<256xf32>
    %3505 = stablehlo.add %2997, %3504 : tensor<256xf32>
    %3506 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3507 = stablehlo.multiply %3506, %arg94 : tensor<256xf32>
    %3508 = stablehlo.add %2999, %3507 : tensor<256xf32>
    %3509 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3510 = stablehlo.multiply %3509, %arg95 : tensor<256xf32>
    %3511 = stablehlo.add %3001, %3510 : tensor<256xf32>
    %3512 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %3513 = stablehlo.multiply %3512, %arg96 : tensor<3x3x128x256xf32>
    %3514 = stablehlo.add %3003, %3513 : tensor<3x3x128x256xf32>
    %3515 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3516 = stablehlo.multiply %3515, %arg97 : tensor<3x3x256x256xf32>
    %3517 = stablehlo.add %3005, %3516 : tensor<3x3x256x256xf32>
    %3518 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %3519 = stablehlo.multiply %3518, %arg98 : tensor<1x1x128x256xf32>
    %3520 = stablehlo.add %3007, %3519 : tensor<1x1x128x256xf32>
    %3521 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3522 = stablehlo.multiply %3521, %arg99 : tensor<256xf32>
    %3523 = stablehlo.add %3009, %3522 : tensor<256xf32>
    %3524 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3525 = stablehlo.multiply %3524, %arg100 : tensor<256xf32>
    %3526 = stablehlo.add %3011, %3525 : tensor<256xf32>
    %3527 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3528 = stablehlo.multiply %3527, %arg101 : tensor<256xf32>
    %3529 = stablehlo.add %3013, %3528 : tensor<256xf32>
    %3530 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3531 = stablehlo.multiply %3530, %arg102 : tensor<256xf32>
    %3532 = stablehlo.add %3015, %3531 : tensor<256xf32>
    %3533 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3534 = stablehlo.multiply %3533, %arg103 : tensor<256xf32>
    %3535 = stablehlo.add %3017, %3534 : tensor<256xf32>
    %3536 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3537 = stablehlo.multiply %3536, %arg104 : tensor<256xf32>
    %3538 = stablehlo.add %3019, %3537 : tensor<256xf32>
    %3539 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3540 = stablehlo.multiply %3539, %arg105 : tensor<3x3x256x256xf32>
    %3541 = stablehlo.add %3021, %3540 : tensor<3x3x256x256xf32>
    %3542 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3543 = stablehlo.multiply %3542, %arg106 : tensor<3x3x256x256xf32>
    %3544 = stablehlo.add %3023, %3543 : tensor<3x3x256x256xf32>
    %3545 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3546 = stablehlo.multiply %3545, %arg107 : tensor<512xf32>
    %3547 = stablehlo.add %3025, %3546 : tensor<512xf32>
    %3548 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3549 = stablehlo.multiply %3548, %arg108 : tensor<512xf32>
    %3550 = stablehlo.add %3027, %3549 : tensor<512xf32>
    %3551 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3552 = stablehlo.multiply %3551, %arg109 : tensor<512xf32>
    %3553 = stablehlo.add %3029, %3552 : tensor<512xf32>
    %3554 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3555 = stablehlo.multiply %3554, %arg110 : tensor<512xf32>
    %3556 = stablehlo.add %3031, %3555 : tensor<512xf32>
    %3557 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %3558 = stablehlo.multiply %3557, %arg111 : tensor<3x3x256x512xf32>
    %3559 = stablehlo.add %3033, %3558 : tensor<3x3x256x512xf32>
    %3560 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3561 = stablehlo.multiply %3560, %arg112 : tensor<3x3x512x512xf32>
    %3562 = stablehlo.add %3035, %3561 : tensor<3x3x512x512xf32>
    %3563 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %3564 = stablehlo.multiply %3563, %arg113 : tensor<1x1x256x512xf32>
    %3565 = stablehlo.add %3037, %3564 : tensor<1x1x256x512xf32>
    %3566 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3567 = stablehlo.multiply %3566, %arg114 : tensor<512xf32>
    %3568 = stablehlo.add %3039, %3567 : tensor<512xf32>
    %3569 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3570 = stablehlo.multiply %3569, %arg115 : tensor<512xf32>
    %3571 = stablehlo.add %3041, %3570 : tensor<512xf32>
    %3572 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3573 = stablehlo.multiply %3572, %arg116 : tensor<512xf32>
    %3574 = stablehlo.add %3043, %3573 : tensor<512xf32>
    %3575 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3576 = stablehlo.multiply %3575, %arg117 : tensor<512xf32>
    %3577 = stablehlo.add %3045, %3576 : tensor<512xf32>
    %3578 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3579 = stablehlo.multiply %3578, %arg118 : tensor<512xf32>
    %3580 = stablehlo.add %3047, %3579 : tensor<512xf32>
    %3581 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3582 = stablehlo.multiply %3581, %arg119 : tensor<512xf32>
    %3583 = stablehlo.add %3049, %3582 : tensor<512xf32>
    %3584 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3585 = stablehlo.multiply %3584, %arg120 : tensor<3x3x512x512xf32>
    %3586 = stablehlo.add %3051, %3585 : tensor<3x3x512x512xf32>
    %3587 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3588 = stablehlo.multiply %3587, %arg121 : tensor<3x3x512x512xf32>
    %3589 = stablehlo.add %3053, %3588 : tensor<3x3x512x512xf32>
    %3590 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3591 = stablehlo.multiply %3590, %arg122 : tensor<64xf32>
    %3592 = stablehlo.add %3055, %3591 : tensor<64xf32>
    %3593 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3594 = stablehlo.multiply %3593, %arg123 : tensor<64xf32>
    %3595 = stablehlo.add %3057, %3594 : tensor<64xf32>
    %3596 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %3597 = stablehlo.multiply %3596, %arg124 : tensor<7x7x3x64xf32>
    %3598 = stablehlo.add %3059, %3597 : tensor<7x7x3x64xf32>
    %3599 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %3600 = stablehlo.multiply %3599, %3415 : tensor<1000xf32>
    %3601 = stablehlo.add %2937, %3600 : tensor<1000xf32>
    %3602 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %3603 = stablehlo.multiply %3602, %3418 : tensor<512x1000xf32>
    %3604 = stablehlo.add %2939, %3603 : tensor<512x1000xf32>
    %3605 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3606 = stablehlo.multiply %3605, %3421 : tensor<64xf32>
    %3607 = stablehlo.add %2941, %3606 : tensor<64xf32>
    %3608 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3609 = stablehlo.multiply %3608, %3424 : tensor<64xf32>
    %3610 = stablehlo.add %2943, %3609 : tensor<64xf32>
    %3611 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3612 = stablehlo.multiply %3611, %3427 : tensor<64xf32>
    %3613 = stablehlo.add %2945, %3612 : tensor<64xf32>
    %3614 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3615 = stablehlo.multiply %3614, %3430 : tensor<64xf32>
    %3616 = stablehlo.add %2947, %3615 : tensor<64xf32>
    %3617 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3618 = stablehlo.multiply %3617, %3433 : tensor<3x3x64x64xf32>
    %3619 = stablehlo.add %2949, %3618 : tensor<3x3x64x64xf32>
    %3620 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3621 = stablehlo.multiply %3620, %3436 : tensor<3x3x64x64xf32>
    %3622 = stablehlo.add %2951, %3621 : tensor<3x3x64x64xf32>
    %3623 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3624 = stablehlo.multiply %3623, %3439 : tensor<64xf32>
    %3625 = stablehlo.add %2953, %3624 : tensor<64xf32>
    %3626 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3627 = stablehlo.multiply %3626, %3442 : tensor<64xf32>
    %3628 = stablehlo.add %2955, %3627 : tensor<64xf32>
    %3629 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3630 = stablehlo.multiply %3629, %3445 : tensor<64xf32>
    %3631 = stablehlo.add %2957, %3630 : tensor<64xf32>
    %3632 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3633 = stablehlo.multiply %3632, %3448 : tensor<64xf32>
    %3634 = stablehlo.add %2959, %3633 : tensor<64xf32>
    %3635 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3636 = stablehlo.multiply %3635, %3451 : tensor<3x3x64x64xf32>
    %3637 = stablehlo.add %2961, %3636 : tensor<3x3x64x64xf32>
    %3638 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3639 = stablehlo.multiply %3638, %3454 : tensor<3x3x64x64xf32>
    %3640 = stablehlo.add %2963, %3639 : tensor<3x3x64x64xf32>
    %3641 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3642 = stablehlo.multiply %3641, %3457 : tensor<128xf32>
    %3643 = stablehlo.add %2965, %3642 : tensor<128xf32>
    %3644 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3645 = stablehlo.multiply %3644, %3460 : tensor<128xf32>
    %3646 = stablehlo.add %2967, %3645 : tensor<128xf32>
    %3647 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3648 = stablehlo.multiply %3647, %3463 : tensor<128xf32>
    %3649 = stablehlo.add %2969, %3648 : tensor<128xf32>
    %3650 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3651 = stablehlo.multiply %3650, %3466 : tensor<128xf32>
    %3652 = stablehlo.add %2971, %3651 : tensor<128xf32>
    %3653 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %3654 = stablehlo.multiply %3653, %3469 : tensor<3x3x64x128xf32>
    %3655 = stablehlo.add %2973, %3654 : tensor<3x3x64x128xf32>
    %3656 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %3657 = stablehlo.multiply %3656, %3472 : tensor<3x3x128x128xf32>
    %3658 = stablehlo.add %2975, %3657 : tensor<3x3x128x128xf32>
    %3659 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %3660 = stablehlo.multiply %3659, %3475 : tensor<1x1x64x128xf32>
    %3661 = stablehlo.add %2977, %3660 : tensor<1x1x64x128xf32>
    %3662 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3663 = stablehlo.multiply %3662, %3478 : tensor<128xf32>
    %3664 = stablehlo.add %2979, %3663 : tensor<128xf32>
    %3665 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3666 = stablehlo.multiply %3665, %3481 : tensor<128xf32>
    %3667 = stablehlo.add %2981, %3666 : tensor<128xf32>
    %3668 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3669 = stablehlo.multiply %3668, %3484 : tensor<128xf32>
    %3670 = stablehlo.add %2983, %3669 : tensor<128xf32>
    %3671 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3672 = stablehlo.multiply %3671, %3487 : tensor<128xf32>
    %3673 = stablehlo.add %2985, %3672 : tensor<128xf32>
    %3674 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3675 = stablehlo.multiply %3674, %3490 : tensor<128xf32>
    %3676 = stablehlo.add %2987, %3675 : tensor<128xf32>
    %3677 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3678 = stablehlo.multiply %3677, %3493 : tensor<128xf32>
    %3679 = stablehlo.add %2989, %3678 : tensor<128xf32>
    %3680 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %3681 = stablehlo.multiply %3680, %3496 : tensor<3x3x128x128xf32>
    %3682 = stablehlo.add %2991, %3681 : tensor<3x3x128x128xf32>
    %3683 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %3684 = stablehlo.multiply %3683, %3499 : tensor<3x3x128x128xf32>
    %3685 = stablehlo.add %2993, %3684 : tensor<3x3x128x128xf32>
    %3686 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3687 = stablehlo.multiply %3686, %3502 : tensor<256xf32>
    %3688 = stablehlo.add %2995, %3687 : tensor<256xf32>
    %3689 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3690 = stablehlo.multiply %3689, %3505 : tensor<256xf32>
    %3691 = stablehlo.add %2997, %3690 : tensor<256xf32>
    %3692 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3693 = stablehlo.multiply %3692, %3508 : tensor<256xf32>
    %3694 = stablehlo.add %2999, %3693 : tensor<256xf32>
    %3695 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3696 = stablehlo.multiply %3695, %3511 : tensor<256xf32>
    %3697 = stablehlo.add %3001, %3696 : tensor<256xf32>
    %3698 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %3699 = stablehlo.multiply %3698, %3514 : tensor<3x3x128x256xf32>
    %3700 = stablehlo.add %3003, %3699 : tensor<3x3x128x256xf32>
    %3701 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3702 = stablehlo.multiply %3701, %3517 : tensor<3x3x256x256xf32>
    %3703 = stablehlo.add %3005, %3702 : tensor<3x3x256x256xf32>
    %3704 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %3705 = stablehlo.multiply %3704, %3520 : tensor<1x1x128x256xf32>
    %3706 = stablehlo.add %3007, %3705 : tensor<1x1x128x256xf32>
    %3707 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3708 = stablehlo.multiply %3707, %3523 : tensor<256xf32>
    %3709 = stablehlo.add %3009, %3708 : tensor<256xf32>
    %3710 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3711 = stablehlo.multiply %3710, %3526 : tensor<256xf32>
    %3712 = stablehlo.add %3011, %3711 : tensor<256xf32>
    %3713 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3714 = stablehlo.multiply %3713, %3529 : tensor<256xf32>
    %3715 = stablehlo.add %3013, %3714 : tensor<256xf32>
    %3716 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3717 = stablehlo.multiply %3716, %3532 : tensor<256xf32>
    %3718 = stablehlo.add %3015, %3717 : tensor<256xf32>
    %3719 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3720 = stablehlo.multiply %3719, %3535 : tensor<256xf32>
    %3721 = stablehlo.add %3017, %3720 : tensor<256xf32>
    %3722 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3723 = stablehlo.multiply %3722, %3538 : tensor<256xf32>
    %3724 = stablehlo.add %3019, %3723 : tensor<256xf32>
    %3725 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3726 = stablehlo.multiply %3725, %3541 : tensor<3x3x256x256xf32>
    %3727 = stablehlo.add %3021, %3726 : tensor<3x3x256x256xf32>
    %3728 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3729 = stablehlo.multiply %3728, %3544 : tensor<3x3x256x256xf32>
    %3730 = stablehlo.add %3023, %3729 : tensor<3x3x256x256xf32>
    %3731 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3732 = stablehlo.multiply %3731, %3547 : tensor<512xf32>
    %3733 = stablehlo.add %3025, %3732 : tensor<512xf32>
    %3734 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3735 = stablehlo.multiply %3734, %3550 : tensor<512xf32>
    %3736 = stablehlo.add %3027, %3735 : tensor<512xf32>
    %3737 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3738 = stablehlo.multiply %3737, %3553 : tensor<512xf32>
    %3739 = stablehlo.add %3029, %3738 : tensor<512xf32>
    %3740 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3741 = stablehlo.multiply %3740, %3556 : tensor<512xf32>
    %3742 = stablehlo.add %3031, %3741 : tensor<512xf32>
    %3743 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %3744 = stablehlo.multiply %3743, %3559 : tensor<3x3x256x512xf32>
    %3745 = stablehlo.add %3033, %3744 : tensor<3x3x256x512xf32>
    %3746 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3747 = stablehlo.multiply %3746, %3562 : tensor<3x3x512x512xf32>
    %3748 = stablehlo.add %3035, %3747 : tensor<3x3x512x512xf32>
    %3749 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %3750 = stablehlo.multiply %3749, %3565 : tensor<1x1x256x512xf32>
    %3751 = stablehlo.add %3037, %3750 : tensor<1x1x256x512xf32>
    %3752 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3753 = stablehlo.multiply %3752, %3568 : tensor<512xf32>
    %3754 = stablehlo.add %3039, %3753 : tensor<512xf32>
    %3755 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3756 = stablehlo.multiply %3755, %3571 : tensor<512xf32>
    %3757 = stablehlo.add %3041, %3756 : tensor<512xf32>
    %3758 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3759 = stablehlo.multiply %3758, %3574 : tensor<512xf32>
    %3760 = stablehlo.add %3043, %3759 : tensor<512xf32>
    %3761 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3762 = stablehlo.multiply %3761, %3577 : tensor<512xf32>
    %3763 = stablehlo.add %3045, %3762 : tensor<512xf32>
    %3764 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3765 = stablehlo.multiply %3764, %3580 : tensor<512xf32>
    %3766 = stablehlo.add %3047, %3765 : tensor<512xf32>
    %3767 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3768 = stablehlo.multiply %3767, %3583 : tensor<512xf32>
    %3769 = stablehlo.add %3049, %3768 : tensor<512xf32>
    %3770 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3771 = stablehlo.multiply %3770, %3586 : tensor<3x3x512x512xf32>
    %3772 = stablehlo.add %3051, %3771 : tensor<3x3x512x512xf32>
    %3773 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3774 = stablehlo.multiply %3773, %3589 : tensor<3x3x512x512xf32>
    %3775 = stablehlo.add %3053, %3774 : tensor<3x3x512x512xf32>
    %3776 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3777 = stablehlo.multiply %3776, %3592 : tensor<64xf32>
    %3778 = stablehlo.add %3055, %3777 : tensor<64xf32>
    %3779 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3780 = stablehlo.multiply %3779, %3595 : tensor<64xf32>
    %3781 = stablehlo.add %3057, %3780 : tensor<64xf32>
    %3782 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %3783 = stablehlo.multiply %3782, %3598 : tensor<7x7x3x64xf32>
    %3784 = stablehlo.add %3059, %3783 : tensor<7x7x3x64xf32>
    %c_305 = stablehlo.constant dense<0> : tensor<i32>
    %3785 = stablehlo.subtract %arg125, %c_305 : tensor<i32>
    %3786 = call @clip_524(%3785, %c, %cst) : (tensor<i32>, tensor<i32>, tensor<f32>) -> tensor<f32>
    %3787 = stablehlo.divide %3786, %cst : tensor<f32>
    %3788 = stablehlo.subtract %cst_0, %3787 : tensor<f32>
    %3789 = stablehlo.multiply %cst_1, %3788 : tensor<f32>
    %3790 = stablehlo.add %3789, %cst_2 : tensor<f32>
    %3791 = stablehlo.convert %arg125 : (tensor<i32>) -> tensor<f32>
    %3792 = stablehlo.compare  LT, %3791, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3793 = stablehlo.convert %arg125 : (tensor<i32>) -> tensor<f32>
    %3794 = stablehlo.subtract %3793, %cst : tensor<f32>
    %3795 = stablehlo.minimum %3794, %cst_3 : tensor<f32>
    %3796 = stablehlo.multiply %cst_4, %3795 : tensor<f32>
    %3797 = stablehlo.divide %3796, %cst_3 : tensor<f32>
    %3798 = stablehlo.cosine %3797 : tensor<f32>
    %3799 = stablehlo.add %cst_0, %3798 : tensor<f32>
    %3800 = stablehlo.multiply %cst_5, %3799 : tensor<f32>
    %3801 = stablehlo.power %3800, %cst_0 : tensor<f32>
    %3802 = stablehlo.multiply %cst_0, %3801 : tensor<f32>
    %3803 = stablehlo.add %3802, %cst_6 : tensor<f32>
    %3804 = stablehlo.multiply %cst_2, %3803 : tensor<f32>
    %3805 = call @_where(%3792, %3790, %3804) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %cst_306 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %3806 = stablehlo.multiply %cst_306, %3805 : tensor<f32>
    %3807 = stablehlo.convert %3806 : tensor<f32>
    %3808 = stablehlo.broadcast_in_dim %3807, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %3809 = stablehlo.multiply %3808, %3601 : tensor<1000xf32>
    %3810 = stablehlo.convert %3806 : tensor<f32>
    %3811 = stablehlo.broadcast_in_dim %3810, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %3812 = stablehlo.multiply %3811, %3604 : tensor<512x1000xf32>
    %3813 = stablehlo.convert %3806 : tensor<f32>
    %3814 = stablehlo.broadcast_in_dim %3813, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3815 = stablehlo.multiply %3814, %3607 : tensor<64xf32>
    %3816 = stablehlo.convert %3806 : tensor<f32>
    %3817 = stablehlo.broadcast_in_dim %3816, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3818 = stablehlo.multiply %3817, %3610 : tensor<64xf32>
    %3819 = stablehlo.convert %3806 : tensor<f32>
    %3820 = stablehlo.broadcast_in_dim %3819, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3821 = stablehlo.multiply %3820, %3613 : tensor<64xf32>
    %3822 = stablehlo.convert %3806 : tensor<f32>
    %3823 = stablehlo.broadcast_in_dim %3822, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3824 = stablehlo.multiply %3823, %3616 : tensor<64xf32>
    %3825 = stablehlo.convert %3806 : tensor<f32>
    %3826 = stablehlo.broadcast_in_dim %3825, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3827 = stablehlo.multiply %3826, %3619 : tensor<3x3x64x64xf32>
    %3828 = stablehlo.convert %3806 : tensor<f32>
    %3829 = stablehlo.broadcast_in_dim %3828, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3830 = stablehlo.multiply %3829, %3622 : tensor<3x3x64x64xf32>
    %3831 = stablehlo.convert %3806 : tensor<f32>
    %3832 = stablehlo.broadcast_in_dim %3831, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3833 = stablehlo.multiply %3832, %3625 : tensor<64xf32>
    %3834 = stablehlo.convert %3806 : tensor<f32>
    %3835 = stablehlo.broadcast_in_dim %3834, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3836 = stablehlo.multiply %3835, %3628 : tensor<64xf32>
    %3837 = stablehlo.convert %3806 : tensor<f32>
    %3838 = stablehlo.broadcast_in_dim %3837, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3839 = stablehlo.multiply %3838, %3631 : tensor<64xf32>
    %3840 = stablehlo.convert %3806 : tensor<f32>
    %3841 = stablehlo.broadcast_in_dim %3840, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3842 = stablehlo.multiply %3841, %3634 : tensor<64xf32>
    %3843 = stablehlo.convert %3806 : tensor<f32>
    %3844 = stablehlo.broadcast_in_dim %3843, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3845 = stablehlo.multiply %3844, %3637 : tensor<3x3x64x64xf32>
    %3846 = stablehlo.convert %3806 : tensor<f32>
    %3847 = stablehlo.broadcast_in_dim %3846, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %3848 = stablehlo.multiply %3847, %3640 : tensor<3x3x64x64xf32>
    %3849 = stablehlo.convert %3806 : tensor<f32>
    %3850 = stablehlo.broadcast_in_dim %3849, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3851 = stablehlo.multiply %3850, %3643 : tensor<128xf32>
    %3852 = stablehlo.convert %3806 : tensor<f32>
    %3853 = stablehlo.broadcast_in_dim %3852, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3854 = stablehlo.multiply %3853, %3646 : tensor<128xf32>
    %3855 = stablehlo.convert %3806 : tensor<f32>
    %3856 = stablehlo.broadcast_in_dim %3855, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3857 = stablehlo.multiply %3856, %3649 : tensor<128xf32>
    %3858 = stablehlo.convert %3806 : tensor<f32>
    %3859 = stablehlo.broadcast_in_dim %3858, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3860 = stablehlo.multiply %3859, %3652 : tensor<128xf32>
    %3861 = stablehlo.convert %3806 : tensor<f32>
    %3862 = stablehlo.broadcast_in_dim %3861, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %3863 = stablehlo.multiply %3862, %3655 : tensor<3x3x64x128xf32>
    %3864 = stablehlo.convert %3806 : tensor<f32>
    %3865 = stablehlo.broadcast_in_dim %3864, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %3866 = stablehlo.multiply %3865, %3658 : tensor<3x3x128x128xf32>
    %3867 = stablehlo.convert %3806 : tensor<f32>
    %3868 = stablehlo.broadcast_in_dim %3867, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %3869 = stablehlo.multiply %3868, %3661 : tensor<1x1x64x128xf32>
    %3870 = stablehlo.convert %3806 : tensor<f32>
    %3871 = stablehlo.broadcast_in_dim %3870, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3872 = stablehlo.multiply %3871, %3664 : tensor<128xf32>
    %3873 = stablehlo.convert %3806 : tensor<f32>
    %3874 = stablehlo.broadcast_in_dim %3873, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3875 = stablehlo.multiply %3874, %3667 : tensor<128xf32>
    %3876 = stablehlo.convert %3806 : tensor<f32>
    %3877 = stablehlo.broadcast_in_dim %3876, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3878 = stablehlo.multiply %3877, %3670 : tensor<128xf32>
    %3879 = stablehlo.convert %3806 : tensor<f32>
    %3880 = stablehlo.broadcast_in_dim %3879, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3881 = stablehlo.multiply %3880, %3673 : tensor<128xf32>
    %3882 = stablehlo.convert %3806 : tensor<f32>
    %3883 = stablehlo.broadcast_in_dim %3882, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3884 = stablehlo.multiply %3883, %3676 : tensor<128xf32>
    %3885 = stablehlo.convert %3806 : tensor<f32>
    %3886 = stablehlo.broadcast_in_dim %3885, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3887 = stablehlo.multiply %3886, %3679 : tensor<128xf32>
    %3888 = stablehlo.convert %3806 : tensor<f32>
    %3889 = stablehlo.broadcast_in_dim %3888, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %3890 = stablehlo.multiply %3889, %3682 : tensor<3x3x128x128xf32>
    %3891 = stablehlo.convert %3806 : tensor<f32>
    %3892 = stablehlo.broadcast_in_dim %3891, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %3893 = stablehlo.multiply %3892, %3685 : tensor<3x3x128x128xf32>
    %3894 = stablehlo.convert %3806 : tensor<f32>
    %3895 = stablehlo.broadcast_in_dim %3894, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3896 = stablehlo.multiply %3895, %3688 : tensor<256xf32>
    %3897 = stablehlo.convert %3806 : tensor<f32>
    %3898 = stablehlo.broadcast_in_dim %3897, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3899 = stablehlo.multiply %3898, %3691 : tensor<256xf32>
    %3900 = stablehlo.convert %3806 : tensor<f32>
    %3901 = stablehlo.broadcast_in_dim %3900, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3902 = stablehlo.multiply %3901, %3694 : tensor<256xf32>
    %3903 = stablehlo.convert %3806 : tensor<f32>
    %3904 = stablehlo.broadcast_in_dim %3903, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3905 = stablehlo.multiply %3904, %3697 : tensor<256xf32>
    %3906 = stablehlo.convert %3806 : tensor<f32>
    %3907 = stablehlo.broadcast_in_dim %3906, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %3908 = stablehlo.multiply %3907, %3700 : tensor<3x3x128x256xf32>
    %3909 = stablehlo.convert %3806 : tensor<f32>
    %3910 = stablehlo.broadcast_in_dim %3909, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3911 = stablehlo.multiply %3910, %3703 : tensor<3x3x256x256xf32>
    %3912 = stablehlo.convert %3806 : tensor<f32>
    %3913 = stablehlo.broadcast_in_dim %3912, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %3914 = stablehlo.multiply %3913, %3706 : tensor<1x1x128x256xf32>
    %3915 = stablehlo.convert %3806 : tensor<f32>
    %3916 = stablehlo.broadcast_in_dim %3915, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3917 = stablehlo.multiply %3916, %3709 : tensor<256xf32>
    %3918 = stablehlo.convert %3806 : tensor<f32>
    %3919 = stablehlo.broadcast_in_dim %3918, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3920 = stablehlo.multiply %3919, %3712 : tensor<256xf32>
    %3921 = stablehlo.convert %3806 : tensor<f32>
    %3922 = stablehlo.broadcast_in_dim %3921, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3923 = stablehlo.multiply %3922, %3715 : tensor<256xf32>
    %3924 = stablehlo.convert %3806 : tensor<f32>
    %3925 = stablehlo.broadcast_in_dim %3924, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3926 = stablehlo.multiply %3925, %3718 : tensor<256xf32>
    %3927 = stablehlo.convert %3806 : tensor<f32>
    %3928 = stablehlo.broadcast_in_dim %3927, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3929 = stablehlo.multiply %3928, %3721 : tensor<256xf32>
    %3930 = stablehlo.convert %3806 : tensor<f32>
    %3931 = stablehlo.broadcast_in_dim %3930, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3932 = stablehlo.multiply %3931, %3724 : tensor<256xf32>
    %3933 = stablehlo.convert %3806 : tensor<f32>
    %3934 = stablehlo.broadcast_in_dim %3933, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3935 = stablehlo.multiply %3934, %3727 : tensor<3x3x256x256xf32>
    %3936 = stablehlo.convert %3806 : tensor<f32>
    %3937 = stablehlo.broadcast_in_dim %3936, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %3938 = stablehlo.multiply %3937, %3730 : tensor<3x3x256x256xf32>
    %3939 = stablehlo.convert %3806 : tensor<f32>
    %3940 = stablehlo.broadcast_in_dim %3939, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3941 = stablehlo.multiply %3940, %3733 : tensor<512xf32>
    %3942 = stablehlo.convert %3806 : tensor<f32>
    %3943 = stablehlo.broadcast_in_dim %3942, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3944 = stablehlo.multiply %3943, %3736 : tensor<512xf32>
    %3945 = stablehlo.convert %3806 : tensor<f32>
    %3946 = stablehlo.broadcast_in_dim %3945, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3947 = stablehlo.multiply %3946, %3739 : tensor<512xf32>
    %3948 = stablehlo.convert %3806 : tensor<f32>
    %3949 = stablehlo.broadcast_in_dim %3948, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3950 = stablehlo.multiply %3949, %3742 : tensor<512xf32>
    %3951 = stablehlo.convert %3806 : tensor<f32>
    %3952 = stablehlo.broadcast_in_dim %3951, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %3953 = stablehlo.multiply %3952, %3745 : tensor<3x3x256x512xf32>
    %3954 = stablehlo.convert %3806 : tensor<f32>
    %3955 = stablehlo.broadcast_in_dim %3954, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3956 = stablehlo.multiply %3955, %3748 : tensor<3x3x512x512xf32>
    %3957 = stablehlo.convert %3806 : tensor<f32>
    %3958 = stablehlo.broadcast_in_dim %3957, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %3959 = stablehlo.multiply %3958, %3751 : tensor<1x1x256x512xf32>
    %3960 = stablehlo.convert %3806 : tensor<f32>
    %3961 = stablehlo.broadcast_in_dim %3960, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3962 = stablehlo.multiply %3961, %3754 : tensor<512xf32>
    %3963 = stablehlo.convert %3806 : tensor<f32>
    %3964 = stablehlo.broadcast_in_dim %3963, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3965 = stablehlo.multiply %3964, %3757 : tensor<512xf32>
    %3966 = stablehlo.convert %3806 : tensor<f32>
    %3967 = stablehlo.broadcast_in_dim %3966, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3968 = stablehlo.multiply %3967, %3760 : tensor<512xf32>
    %3969 = stablehlo.convert %3806 : tensor<f32>
    %3970 = stablehlo.broadcast_in_dim %3969, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3971 = stablehlo.multiply %3970, %3763 : tensor<512xf32>
    %3972 = stablehlo.convert %3806 : tensor<f32>
    %3973 = stablehlo.broadcast_in_dim %3972, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3974 = stablehlo.multiply %3973, %3766 : tensor<512xf32>
    %3975 = stablehlo.convert %3806 : tensor<f32>
    %3976 = stablehlo.broadcast_in_dim %3975, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3977 = stablehlo.multiply %3976, %3769 : tensor<512xf32>
    %3978 = stablehlo.convert %3806 : tensor<f32>
    %3979 = stablehlo.broadcast_in_dim %3978, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3980 = stablehlo.multiply %3979, %3772 : tensor<3x3x512x512xf32>
    %3981 = stablehlo.convert %3806 : tensor<f32>
    %3982 = stablehlo.broadcast_in_dim %3981, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %3983 = stablehlo.multiply %3982, %3775 : tensor<3x3x512x512xf32>
    %3984 = stablehlo.convert %3806 : tensor<f32>
    %3985 = stablehlo.broadcast_in_dim %3984, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3986 = stablehlo.multiply %3985, %3778 : tensor<64xf32>
    %3987 = stablehlo.convert %3806 : tensor<f32>
    %3988 = stablehlo.broadcast_in_dim %3987, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3989 = stablehlo.multiply %3988, %3781 : tensor<64xf32>
    %3990 = stablehlo.convert %3806 : tensor<f32>
    %3991 = stablehlo.broadcast_in_dim %3990, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %3992 = stablehlo.multiply %3991, %3784 : tensor<7x7x3x64xf32>
    %c_307 = stablehlo.constant dense<2147483647> : tensor<i32>
    %3993 = stablehlo.compare  LT, %arg125, %c_307,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_308 = stablehlo.constant dense<1> : tensor<i32>
    %3994 = stablehlo.add %arg125, %c_308 : tensor<i32>
    %3995 = call @_where_529(%3993, %3994, %c_307) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %3996 = stablehlo.add %arg1, %3809 : tensor<1000xf32>
    %3997 = stablehlo.add %arg2, %3812 : tensor<512x1000xf32>
    %3998 = stablehlo.add %arg3, %3815 : tensor<64xf32>
    %3999 = stablehlo.add %arg4, %3818 : tensor<64xf32>
    %4000 = stablehlo.add %arg5, %3821 : tensor<64xf32>
    %4001 = stablehlo.add %arg6, %3824 : tensor<64xf32>
    %4002 = stablehlo.add %arg7, %3827 : tensor<3x3x64x64xf32>
    %4003 = stablehlo.add %arg8, %3830 : tensor<3x3x64x64xf32>
    %4004 = stablehlo.add %arg9, %3833 : tensor<64xf32>
    %4005 = stablehlo.add %arg10, %3836 : tensor<64xf32>
    %4006 = stablehlo.add %arg11, %3839 : tensor<64xf32>
    %4007 = stablehlo.add %arg12, %3842 : tensor<64xf32>
    %4008 = stablehlo.add %arg13, %3845 : tensor<3x3x64x64xf32>
    %4009 = stablehlo.add %arg14, %3848 : tensor<3x3x64x64xf32>
    %4010 = stablehlo.add %arg15, %3851 : tensor<128xf32>
    %4011 = stablehlo.add %arg16, %3854 : tensor<128xf32>
    %4012 = stablehlo.add %arg17, %3857 : tensor<128xf32>
    %4013 = stablehlo.add %arg18, %3860 : tensor<128xf32>
    %4014 = stablehlo.add %arg19, %3863 : tensor<3x3x64x128xf32>
    %4015 = stablehlo.add %arg20, %3866 : tensor<3x3x128x128xf32>
    %4016 = stablehlo.add %arg21, %3869 : tensor<1x1x64x128xf32>
    %4017 = stablehlo.add %arg22, %3872 : tensor<128xf32>
    %4018 = stablehlo.add %arg23, %3875 : tensor<128xf32>
    %4019 = stablehlo.add %arg24, %3878 : tensor<128xf32>
    %4020 = stablehlo.add %arg25, %3881 : tensor<128xf32>
    %4021 = stablehlo.add %arg26, %3884 : tensor<128xf32>
    %4022 = stablehlo.add %arg27, %3887 : tensor<128xf32>
    %4023 = stablehlo.add %arg28, %3890 : tensor<3x3x128x128xf32>
    %4024 = stablehlo.add %arg29, %3893 : tensor<3x3x128x128xf32>
    %4025 = stablehlo.add %arg30, %3896 : tensor<256xf32>
    %4026 = stablehlo.add %arg31, %3899 : tensor<256xf32>
    %4027 = stablehlo.add %arg32, %3902 : tensor<256xf32>
    %4028 = stablehlo.add %arg33, %3905 : tensor<256xf32>
    %4029 = stablehlo.add %arg34, %3908 : tensor<3x3x128x256xf32>
    %4030 = stablehlo.add %arg35, %3911 : tensor<3x3x256x256xf32>
    %4031 = stablehlo.add %arg36, %3914 : tensor<1x1x128x256xf32>
    %4032 = stablehlo.add %arg37, %3917 : tensor<256xf32>
    %4033 = stablehlo.add %arg38, %3920 : tensor<256xf32>
    %4034 = stablehlo.add %arg39, %3923 : tensor<256xf32>
    %4035 = stablehlo.add %arg40, %3926 : tensor<256xf32>
    %4036 = stablehlo.add %arg41, %3929 : tensor<256xf32>
    %4037 = stablehlo.add %arg42, %3932 : tensor<256xf32>
    %4038 = stablehlo.add %arg43, %3935 : tensor<3x3x256x256xf32>
    %4039 = stablehlo.add %arg44, %3938 : tensor<3x3x256x256xf32>
    %4040 = stablehlo.add %arg45, %3941 : tensor<512xf32>
    %4041 = stablehlo.add %arg46, %3944 : tensor<512xf32>
    %4042 = stablehlo.add %arg47, %3947 : tensor<512xf32>
    %4043 = stablehlo.add %arg48, %3950 : tensor<512xf32>
    %4044 = stablehlo.add %arg49, %3953 : tensor<3x3x256x512xf32>
    %4045 = stablehlo.add %arg50, %3956 : tensor<3x3x512x512xf32>
    %4046 = stablehlo.add %arg51, %3959 : tensor<1x1x256x512xf32>
    %4047 = stablehlo.add %arg52, %3962 : tensor<512xf32>
    %4048 = stablehlo.add %arg53, %3965 : tensor<512xf32>
    %4049 = stablehlo.add %arg54, %3968 : tensor<512xf32>
    %4050 = stablehlo.add %arg55, %3971 : tensor<512xf32>
    %4051 = stablehlo.add %arg56, %3974 : tensor<512xf32>
    %4052 = stablehlo.add %arg57, %3977 : tensor<512xf32>
    %4053 = stablehlo.add %arg58, %3980 : tensor<3x3x512x512xf32>
    %4054 = stablehlo.add %arg59, %3983 : tensor<3x3x512x512xf32>
    %4055 = stablehlo.add %arg60, %3986 : tensor<64xf32>
    %4056 = stablehlo.add %arg61, %3989 : tensor<64xf32>
    %4057 = stablehlo.add %arg62, %3992 : tensor<7x7x3x64xf32>
    %4058 = stablehlo.add %arg0, %c_301 : tensor<i32>
    %4059 = call @_where_530(%3245, %3415, %arg63) : (tensor<i1>, tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
    %4060 = call @_where_532(%3245, %3418, %arg64) : (tensor<i1>, tensor<512x1000xf32>, tensor<512x1000xf32>) -> tensor<512x1000xf32>
    %4061 = call @_where_534(%3245, %3421, %arg65) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4062 = call @_where_534(%3245, %3424, %arg66) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4063 = call @_where_534(%3245, %3427, %arg67) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4064 = call @_where_534(%3245, %3430, %arg68) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4065 = call @_where_536(%3245, %3433, %arg69) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %4066 = call @_where_536(%3245, %3436, %arg70) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %4067 = call @_where_534(%3245, %3439, %arg71) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4068 = call @_where_534(%3245, %3442, %arg72) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4069 = call @_where_534(%3245, %3445, %arg73) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4070 = call @_where_534(%3245, %3448, %arg74) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4071 = call @_where_536(%3245, %3451, %arg75) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %4072 = call @_where_536(%3245, %3454, %arg76) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %4073 = call @_where_538(%3245, %3457, %arg77) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4074 = call @_where_538(%3245, %3460, %arg78) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4075 = call @_where_538(%3245, %3463, %arg79) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4076 = call @_where_538(%3245, %3466, %arg80) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4077 = call @_where_540(%3245, %3469, %arg81) : (tensor<i1>, tensor<3x3x64x128xf32>, tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
    %4078 = call @_where_542(%3245, %3472, %arg82) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %4079 = call @_where_544(%3245, %3475, %arg83) : (tensor<i1>, tensor<1x1x64x128xf32>, tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
    %4080 = call @_where_538(%3245, %3478, %arg84) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4081 = call @_where_538(%3245, %3481, %arg85) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4082 = call @_where_538(%3245, %3484, %arg86) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4083 = call @_where_538(%3245, %3487, %arg87) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4084 = call @_where_538(%3245, %3490, %arg88) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4085 = call @_where_538(%3245, %3493, %arg89) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4086 = call @_where_542(%3245, %3496, %arg90) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %4087 = call @_where_542(%3245, %3499, %arg91) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %4088 = call @_where_546(%3245, %3502, %arg92) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4089 = call @_where_546(%3245, %3505, %arg93) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4090 = call @_where_546(%3245, %3508, %arg94) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4091 = call @_where_546(%3245, %3511, %arg95) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4092 = call @_where_548(%3245, %3514, %arg96) : (tensor<i1>, tensor<3x3x128x256xf32>, tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
    %4093 = call @_where_550(%3245, %3517, %arg97) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %4094 = call @_where_552(%3245, %3520, %arg98) : (tensor<i1>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
    %4095 = call @_where_546(%3245, %3523, %arg99) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4096 = call @_where_546(%3245, %3526, %arg100) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4097 = call @_where_546(%3245, %3529, %arg101) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4098 = call @_where_546(%3245, %3532, %arg102) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4099 = call @_where_546(%3245, %3535, %arg103) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4100 = call @_where_546(%3245, %3538, %arg104) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4101 = call @_where_550(%3245, %3541, %arg105) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %4102 = call @_where_550(%3245, %3544, %arg106) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %4103 = call @_where_554(%3245, %3547, %arg107) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4104 = call @_where_554(%3245, %3550, %arg108) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4105 = call @_where_554(%3245, %3553, %arg109) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4106 = call @_where_554(%3245, %3556, %arg110) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4107 = call @_where_556(%3245, %3559, %arg111) : (tensor<i1>, tensor<3x3x256x512xf32>, tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
    %4108 = call @_where_558(%3245, %3562, %arg112) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %4109 = call @_where_560(%3245, %3565, %arg113) : (tensor<i1>, tensor<1x1x256x512xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    %4110 = call @_where_554(%3245, %3568, %arg114) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4111 = call @_where_554(%3245, %3571, %arg115) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4112 = call @_where_554(%3245, %3574, %arg116) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4113 = call @_where_554(%3245, %3577, %arg117) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4114 = call @_where_554(%3245, %3580, %arg118) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4115 = call @_where_554(%3245, %3583, %arg119) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4116 = call @_where_558(%3245, %3586, %arg120) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %4117 = call @_where_558(%3245, %3589, %arg121) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %4118 = call @_where_534(%3245, %3592, %arg122) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4119 = call @_where_534(%3245, %3595, %arg123) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4120 = call @_where_562(%3245, %3598, %arg124) : (tensor<i1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %4121 = call @_where_529(%3245, %3995, %arg125) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %4122 = call @_where_530(%3245, %3996, %arg1) : (tensor<i1>, tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
    %4123 = call @_where_532(%3245, %3997, %arg2) : (tensor<i1>, tensor<512x1000xf32>, tensor<512x1000xf32>) -> tensor<512x1000xf32>
    %4124 = call @_where_534(%3245, %3998, %arg3) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4125 = call @_where_534(%3245, %3999, %arg4) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4126 = call @_where_534(%3245, %4000, %arg5) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4127 = call @_where_534(%3245, %4001, %arg6) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4128 = call @_where_536(%3245, %4002, %arg7) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %4129 = call @_where_536(%3245, %4003, %arg8) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %4130 = call @_where_534(%3245, %4004, %arg9) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4131 = call @_where_534(%3245, %4005, %arg10) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4132 = call @_where_534(%3245, %4006, %arg11) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4133 = call @_where_534(%3245, %4007, %arg12) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4134 = call @_where_536(%3245, %4008, %arg13) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %4135 = call @_where_536(%3245, %4009, %arg14) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %4136 = call @_where_538(%3245, %4010, %arg15) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4137 = call @_where_538(%3245, %4011, %arg16) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4138 = call @_where_538(%3245, %4012, %arg17) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4139 = call @_where_538(%3245, %4013, %arg18) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4140 = call @_where_540(%3245, %4014, %arg19) : (tensor<i1>, tensor<3x3x64x128xf32>, tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
    %4141 = call @_where_542(%3245, %4015, %arg20) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %4142 = call @_where_544(%3245, %4016, %arg21) : (tensor<i1>, tensor<1x1x64x128xf32>, tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
    %4143 = call @_where_538(%3245, %4017, %arg22) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4144 = call @_where_538(%3245, %4018, %arg23) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4145 = call @_where_538(%3245, %4019, %arg24) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4146 = call @_where_538(%3245, %4020, %arg25) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4147 = call @_where_538(%3245, %4021, %arg26) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4148 = call @_where_538(%3245, %4022, %arg27) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %4149 = call @_where_542(%3245, %4023, %arg28) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %4150 = call @_where_542(%3245, %4024, %arg29) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %4151 = call @_where_546(%3245, %4025, %arg30) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4152 = call @_where_546(%3245, %4026, %arg31) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4153 = call @_where_546(%3245, %4027, %arg32) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4154 = call @_where_546(%3245, %4028, %arg33) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4155 = call @_where_548(%3245, %4029, %arg34) : (tensor<i1>, tensor<3x3x128x256xf32>, tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
    %4156 = call @_where_550(%3245, %4030, %arg35) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %4157 = call @_where_552(%3245, %4031, %arg36) : (tensor<i1>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
    %4158 = call @_where_546(%3245, %4032, %arg37) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4159 = call @_where_546(%3245, %4033, %arg38) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4160 = call @_where_546(%3245, %4034, %arg39) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4161 = call @_where_546(%3245, %4035, %arg40) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4162 = call @_where_546(%3245, %4036, %arg41) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4163 = call @_where_546(%3245, %4037, %arg42) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %4164 = call @_where_550(%3245, %4038, %arg43) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %4165 = call @_where_550(%3245, %4039, %arg44) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %4166 = call @_where_554(%3245, %4040, %arg45) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4167 = call @_where_554(%3245, %4041, %arg46) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4168 = call @_where_554(%3245, %4042, %arg47) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4169 = call @_where_554(%3245, %4043, %arg48) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4170 = call @_where_556(%3245, %4044, %arg49) : (tensor<i1>, tensor<3x3x256x512xf32>, tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
    %4171 = call @_where_558(%3245, %4045, %arg50) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %4172 = call @_where_560(%3245, %4046, %arg51) : (tensor<i1>, tensor<1x1x256x512xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    %4173 = call @_where_554(%3245, %4047, %arg52) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4174 = call @_where_554(%3245, %4048, %arg53) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4175 = call @_where_554(%3245, %4049, %arg54) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4176 = call @_where_554(%3245, %4050, %arg55) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4177 = call @_where_554(%3245, %4051, %arg56) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4178 = call @_where_554(%3245, %4052, %arg57) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %4179 = call @_where_558(%3245, %4053, %arg58) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %4180 = call @_where_558(%3245, %4054, %arg59) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %4181 = call @_where_534(%3245, %4055, %arg60) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4182 = call @_where_534(%3245, %4056, %arg61) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %4183 = call @_where_562(%3245, %4057, %arg62) : (tensor<i1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %4184 = stablehlo.broadcast_in_dim %3291, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %4185 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %4186 = stablehlo.broadcast_in_dim %3292, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %4187 = stablehlo.broadcast_in_dim %3258, dims = [] : (tensor<f32>) -> tensor<1xf32>
    return %4058, %4122, %4123, %4124, %4125, %4126, %4127, %4128, %4129, %4130, %4131, %4132, %4133, %4134, %4135, %4136, %4137, %4138, %4139, %4140, %4141, %4142, %4143, %4144, %4145, %4146, %4147, %4148, %4149, %4150, %4151, %4152, %4153, %4154, %4155, %4156, %4157, %4158, %4159, %4160, %4161, %4162, %4163, %4164, %4165, %4166, %4167, %4168, %4169, %4170, %4171, %4172, %4173, %4174, %4175, %4176, %4177, %4178, %4179, %4180, %4181, %4182, %4183, %4059, %4060, %4061, %4062, %4063, %4064, %4065, %4066, %4067, %4068, %4069, %4070, %4071, %4072, %4073, %4074, %4075, %4076, %4077, %4078, %4079, %4080, %4081, %4082, %4083, %4084, %4085, %4086, %4087, %4088, %4089, %4090, %4091, %4092, %4093, %4094, %4095, %4096, %4097, %4098, %4099, %4100, %4101, %4102, %4103, %4104, %4105, %4106, %4107, %4108, %4109, %4110, %4111, %4112, %4113, %4114, %4115, %4116, %4117, %4118, %4119, %4120, %4121, %3334, %3336, %3338, %3340, %3342, %3344, %3346, %3348, %3350, %3352, %3354, %3356, %3358, %3360, %3362, %3364, %3366, %3368, %3370, %3372, %3374, %3376, %3378, %3380, %3382, %3384, %3386, %3388, %3390, %3392, %3394, %3396, %3398, %3400, %3402, %3404, %3406, %3408, %3410, %3412, %3265, %3258, %4184, %4185, %4186, %4187 : tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<i32>
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<f32>
    %2 = stablehlo.minimum %arg2, %1 : tensor<f32>
    return %2 : tensor<f32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @relu(%arg0: tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x112x112x64xbf16>
    return %1 : tensor<256x112x112x64xbf16>
  }
  func.func private @relu_41(%arg0: tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x64xbf16>
    return %1 : tensor<256x56x56x64xbf16>
  }
  func.func private @relu_85(%arg0: tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xbf16>
    return %1 : tensor<256x28x28x128xbf16>
  }
  func.func private @relu_133(%arg0: tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xbf16>
    return %1 : tensor<256x14x14x256xbf16>
  }
  func.func private @relu_181(%arg0: tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xbf16>
    return %1 : tensor<256x7x7x512xbf16>
  }
  func.func private @log_softmax(%arg0: tensor<256x1000xbf16>) -> (tensor<256x1000xbf16>, tensor<256x1000xbf16>, tensor<256x1xbf16>) {
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<256x1000xbf16>, tensor<bf16>) -> tensor<256xbf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<256xbf16>
    %2 = stablehlo.maximum %1, %0 : tensor<256xbf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<256x1xbf16>) -> tensor<256x1000xbf16>
    %5 = stablehlo.subtract %arg0, %4 : tensor<256x1000xbf16>
    %6 = stablehlo.exponential %5 : tensor<256x1000xbf16>
    %7 = stablehlo.convert %6 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
    %10 = stablehlo.convert %9 : (tensor<256x1xf32>) -> tensor<256x1xbf16>
    %11 = stablehlo.log %10 : tensor<256x1xbf16>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<256x1xbf16>) -> tensor<256x1000xbf16>
    %13 = stablehlo.subtract %5, %12 : tensor<256x1000xbf16>
    return %13, %6, %10 : tensor<256x1000xbf16>, tensor<256x1000xbf16>, tensor<256x1xbf16>
  }
  func.func private @log_softmax_292(%arg0: tensor<256x1000xbf16>, %arg1: tensor<256x1xbf16>, %arg2: tensor<256x1000xbf16>) -> tensor<256x1000xbf16> {
    %0 = stablehlo.negate %arg2 : tensor<256x1000xbf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xbf16>, tensor<bf16>) -> tensor<256xbf16>
    %2 = stablehlo.reshape %1 : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %3 = stablehlo.divide %2, %arg1 : tensor<256x1xbf16>
    %4 = stablehlo.convert %3 : (tensor<256x1xbf16>) -> tensor<256x1xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x1xf32>, tensor<f32>) -> tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<256xf32>) -> tensor<256x1000xf32>
    %7 = stablehlo.convert %6 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
    %8 = stablehlo.multiply %7, %arg0 : tensor<256x1000xbf16>
    %9 = stablehlo.add %arg2, %8 : tensor<256x1000xbf16>
    return %9 : tensor<256x1000xbf16>
  }
  func.func private @_where_490(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_493(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_495(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @log_softmax_497(%arg0: tensor<256x1000xbf16>) -> tensor<256x1000xbf16> {
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<256x1000xbf16>, tensor<bf16>) -> tensor<256xbf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<256xbf16>
    %2 = stablehlo.maximum %1, %0 : tensor<256xbf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xbf16>) -> tensor<256x1xbf16>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<256x1xbf16>) -> tensor<256x1000xbf16>
    %5 = stablehlo.subtract %arg0, %4 : tensor<256x1000xbf16>
    %6 = stablehlo.exponential %5 : tensor<256x1000xbf16>
    %7 = stablehlo.convert %6 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
    %10 = stablehlo.convert %9 : (tensor<256x1xf32>) -> tensor<256x1xbf16>
    %11 = stablehlo.log %10 : tensor<256x1xbf16>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<256x1xbf16>) -> tensor<256x1000xbf16>
    %13 = stablehlo.subtract %5, %12 : tensor<256x1000xbf16>
    return %13 : tensor<256x1000xbf16>
  }
  func.func private @argmax(%arg0: tensor<256x1000xbf16>) -> tensor<256xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<256x1000xi32>
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<256x1000xbf16>, tensor<256x1000xi32>, tensor<bf16>, tensor<i32>) -> (tensor<256xbf16>, tensor<256xi32>)
     reducer(%arg1: tensor<bf16>, %arg3: tensor<bf16>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<bf16>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
      stablehlo.return %9, %10 : tensor<bf16>, tensor<i32>
    }
    return %1#1 : tensor<256xi32>
  }
  func.func private @clip_524(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %3 = stablehlo.minimum %arg2, %2 : tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @_where_529(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @_where_530(%arg0: tensor<i1>, %arg1: tensor<1000xf32>, %arg2: tensor<1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
  func.func private @_where_532(%arg0: tensor<i1>, %arg1: tensor<512x1000xf32>, %arg2: tensor<512x1000xf32>) -> tensor<512x1000xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<512x1000xf32>
    return %0 : tensor<512x1000xf32>
  }
  func.func private @_where_534(%arg0: tensor<i1>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @_where_536(%arg0: tensor<i1>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x64x64xf32>
    return %0 : tensor<3x3x64x64xf32>
  }
  func.func private @_where_538(%arg0: tensor<i1>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  func.func private @_where_540(%arg0: tensor<i1>, %arg1: tensor<3x3x64x128xf32>, %arg2: tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x64x128xf32>
    return %0 : tensor<3x3x64x128xf32>
  }
  func.func private @_where_542(%arg0: tensor<i1>, %arg1: tensor<3x3x128x128xf32>, %arg2: tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x128x128xf32>
    return %0 : tensor<3x3x128x128xf32>
  }
  func.func private @_where_544(%arg0: tensor<i1>, %arg1: tensor<1x1x64x128xf32>, %arg2: tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x64x128xf32>
    return %0 : tensor<1x1x64x128xf32>
  }
  func.func private @_where_546(%arg0: tensor<i1>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  func.func private @_where_548(%arg0: tensor<i1>, %arg1: tensor<3x3x128x256xf32>, %arg2: tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x128x256xf32>
    return %0 : tensor<3x3x128x256xf32>
  }
  func.func private @_where_550(%arg0: tensor<i1>, %arg1: tensor<3x3x256x256xf32>, %arg2: tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x256x256xf32>
    return %0 : tensor<3x3x256x256xf32>
  }
  func.func private @_where_552(%arg0: tensor<i1>, %arg1: tensor<1x1x128x256xf32>, %arg2: tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x128x256xf32>
    return %0 : tensor<1x1x128x256xf32>
  }
  func.func private @_where_554(%arg0: tensor<i1>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @_where_556(%arg0: tensor<i1>, %arg1: tensor<3x3x256x512xf32>, %arg2: tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x256x512xf32>
    return %0 : tensor<3x3x256x512xf32>
  }
  func.func private @_where_558(%arg0: tensor<i1>, %arg1: tensor<3x3x512x512xf32>, %arg2: tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x512x512xf32>
    return %0 : tensor<3x3x512x512xf32>
  }
  func.func private @_where_560(%arg0: tensor<i1>, %arg1: tensor<1x1x256x512xf32>, %arg2: tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x256x512xf32>
    return %0 : tensor<1x1x256x512xf32>
  }
  func.func private @_where_562(%arg0: tensor<i1>, %arg1: tensor<7x7x3x64xf32>, %arg2: tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<7x7x3x64xf32>
    return %0 : tensor<7x7x3x64xf32>
  }
}
