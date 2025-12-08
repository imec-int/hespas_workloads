module @pmap_inference_step attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg1: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg2: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg3: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg4: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg5: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg6: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg7: tensor<1x1x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg8: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg9: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg10: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg11: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg12: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg13: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg14: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg15: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg16: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg17: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg18: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg19: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg20: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg21: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg22: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg23: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg24: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg25: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg26: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg27: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg28: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg29: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg30: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg31: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg32: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg33: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg34: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg35: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg36: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg37: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg38: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg39: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg40: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg41: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg42: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg43: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg44: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg45: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg46: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg47: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg48: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg49: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg50: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg51: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg52: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg53: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg54: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg55: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg56: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg57: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg58: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg59: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg60: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg61: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg62: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg63: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg64: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg65: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg66: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg67: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg68: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg69: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg70: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg71: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg72: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg73: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg74: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg75: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg76: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg77: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg78: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg79: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg80: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg81: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg82: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg83: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg84: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg85: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg86: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg87: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg88: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg89: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg90: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg91: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg92: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg93: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg94: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg95: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg96: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg97: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg98: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg99: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg100: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg101: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg102: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg103: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg104: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg105: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg106: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg107: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg108: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg109: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg110: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg111: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg112: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg113: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg114: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg115: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg116: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg117: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg118: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg119: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg120: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg121: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg122: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg123: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg124: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg125: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg126: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg127: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg128: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg129: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg130: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg131: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg132: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg133: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg134: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg135: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg136: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg137: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg138: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg139: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg140: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg141: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg142: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg143: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg144: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg145: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg146: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg147: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg148: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg149: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg150: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg151: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg152: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg153: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg154: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg155: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg156: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg157: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg158: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg159: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg160: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg161: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg162: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg163: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg164: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg165: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg166: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg167: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg168: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg169: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg170: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg171: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg172: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg173: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg174: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg175: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg176: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg177: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg178: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg179: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg180: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg181: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg182: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg183: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg184: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg185: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg186: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg187: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg188: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg189: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg190: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg191: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg192: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg193: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg194: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg195: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg196: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg197: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg198: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg199: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg200: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg201: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg202: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg203: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg204: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg205: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg206: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg207: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg208: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg209: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg210: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg211: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg212: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg213: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg214: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg215: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg216: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg217: tensor<1x1x256x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg218: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg219: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg220: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg221: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg222: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg223: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg224: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg225: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg226: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg227: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg228: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg229: tensor<1x1x1024x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg230: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg231: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg232: tensor<1x1x1024x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg233: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg234: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg235: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg236: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg237: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg238: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg239: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg240: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg241: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg242: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg243: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg244: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg245: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg246: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg247: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg248: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg249: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg250: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg251: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg252: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg253: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg254: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg255: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg256: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg257: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg258: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg259: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg260: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg261: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg262: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg263: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg264: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg265: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg266: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg267: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg268: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg269: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg270: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg271: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg272: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg273: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg274: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg275: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg276: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg277: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg278: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg279: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg280: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg281: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg282: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg283: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg284: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg285: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg286: tensor<1x1x512x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg287: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg288: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg289: tensor<1x1x512x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg290: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg291: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg292: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg293: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg294: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg295: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg296: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg297: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg298: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg299: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg300: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg301: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg302: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg303: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg304: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg305: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg306: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg307: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg308: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg309: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg310: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg311: tensor<2048x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg312: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg313: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg314: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg315: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg316: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg317: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg318: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg319: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg320: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg321: tensor<1x1x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg322: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg323: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg324: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg325: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg326: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg327: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg328: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg329: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg330: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg331: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg332: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg333: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg334: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg335: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg336: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg337: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg338: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg339: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg340: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg341: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg342: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg343: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg344: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg345: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg346: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg347: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg348: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg349: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg350: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg351: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg352: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg353: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg354: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg355: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg356: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg357: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg358: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg359: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg360: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg361: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg362: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg363: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg364: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg365: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg366: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg367: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg368: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg369: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg370: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg371: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg372: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg373: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg374: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg375: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg376: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg377: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg378: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg379: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg380: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg381: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg382: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg383: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg384: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg385: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg386: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg387: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg388: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg389: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg390: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg391: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg392: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg393: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg394: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg395: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg396: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg397: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg398: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg399: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg400: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg401: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg402: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg403: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg404: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg405: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg406: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg407: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg408: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg409: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg410: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg411: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg412: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg413: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg414: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg415: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg416: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg417: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg418: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg419: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg420: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg421: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg422: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg423: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg424: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg425: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg426: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg427: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg428: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg429: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg430: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg431: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg432: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg433: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg434: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg435: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg436: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg437: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg438: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg439: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg440: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg441: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg442: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg443: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg444: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg445: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg446: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg447: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg448: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg449: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg450: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg451: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg452: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg453: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg454: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg455: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg456: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg457: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg458: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg459: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg460: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg461: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg462: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg463: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg464: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg465: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg466: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg467: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg468: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg469: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg470: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg471: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg472: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg473: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg474: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg475: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg476: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg477: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg478: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg479: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg480: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg481: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg482: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg483: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg484: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg485: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg486: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg487: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg488: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg489: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg490: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg491: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg492: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg493: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg494: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg495: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg496: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg497: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg498: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg499: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg500: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg501: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg502: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg503: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg504: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg505: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg506: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg507: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg508: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg509: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg510: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg511: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg512: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg513: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg514: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg515: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg516: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg517: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg518: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg519: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg520: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg521: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg522: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg523: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg524: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg525: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg526: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg527: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg528: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg529: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg530: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg531: tensor<1x1x256x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg532: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg533: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg534: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg535: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg536: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg537: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg538: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg539: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg540: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg541: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg542: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg543: tensor<1x1x1024x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg544: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg545: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg546: tensor<1x1x1024x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg547: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg548: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg549: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg550: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg551: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg552: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg553: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg554: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg555: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg556: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg557: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg558: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg559: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg560: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg561: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg562: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg563: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg564: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg565: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg566: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg567: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg568: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg569: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg570: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg571: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg572: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg573: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg574: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg575: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg576: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg577: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg578: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg579: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg580: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg581: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg582: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg583: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg584: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg585: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg586: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg587: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg588: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg589: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg590: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg591: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg592: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg593: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg594: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg595: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg596: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg597: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg598: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg599: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg600: tensor<1x1x512x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg601: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg602: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg603: tensor<1x1x512x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg604: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg605: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg606: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg607: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg608: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg609: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg610: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg611: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg612: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg613: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg614: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg615: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg616: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg617: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg618: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg619: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg620: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg621: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg622: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg623: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg624: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg625: tensor<2048x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg626: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg627: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg628: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg629: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg630: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg631: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg632: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg633: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg634: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg635: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg636: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg637: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg638: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg639: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg640: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg641: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg642: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg643: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg644: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg645: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg646: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg647: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg648: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg649: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg650: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg651: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg652: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg653: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg654: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg655: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg656: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg657: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg658: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg659: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg660: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg661: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg662: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg663: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg664: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg665: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg666: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg667: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg668: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg669: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg670: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg671: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg672: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg673: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg674: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg675: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg676: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg677: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg678: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg679: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg680: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg681: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg682: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg683: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg684: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg685: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg686: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg687: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg688: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg689: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg690: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg691: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg692: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg693: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg694: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg695: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg696: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg697: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg698: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg699: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg700: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg701: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg702: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg703: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg704: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg705: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg706: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg707: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg708: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg709: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg710: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg711: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg712: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg713: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg714: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg715: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg716: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg717: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg718: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg719: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg720: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg721: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg722: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg723: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg724: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg725: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg726: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg727: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg728: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg729: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg730: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg731: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg732: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg733: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg734: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg735: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg736: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg737: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg738: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg739: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg740: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg741: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg742: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg743: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg744: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg745: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg746: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg747: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg748: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg749: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg750: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg751: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg752: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg753: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg754: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg755: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg756: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg757: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg758: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg759: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg760: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg761: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg762: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg763: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg764: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg765: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg766: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg767: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg768: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg769: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg770: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg771: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg772: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg773: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg774: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg775: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg776: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg777: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg778: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg779: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg780: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg781: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg782: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg783: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg784: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg785: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg786: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg787: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg788: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg789: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg790: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg791: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg792: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg793: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg794: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg795: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg796: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg797: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg798: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg799: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg800: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg801: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg802: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg803: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg804: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg805: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg806: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg807: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg808: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg809: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg810: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg811: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg812: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg813: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg814: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg815: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg816: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg817: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg818: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg819: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg820: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg821: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg822: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg823: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg824: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg825: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg826: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg827: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg828: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg829: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg830: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg831: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg832: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg833: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg834: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg835: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg836: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg837: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg838: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg839: tensor<f32> {mhlo.is_same_data_across_replicas = true}, %arg840: tensor<1x256x224x224x3xbf16>, %arg841: tensor<1x256xi32>) -> (tensor<1x256x1000xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.reshape %arg840 : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
    %1 = stablehlo.reshape %arg841 : (tensor<1x256xi32>) -> tensor<256xi32>
    %2 = stablehlo.convert %arg314 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xbf16>
    %3 = stablehlo.convolution(%0, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xbf16>, tensor<7x7x3x64xbf16>) -> tensor<256x112x112x64xbf16>
    %4 = stablehlo.broadcast_in_dim %arg836, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg837, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %6 = stablehlo.convert %3 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
    %7 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %8 = stablehlo.subtract %6, %7 : tensor<256x112x112x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %10 = stablehlo.add %5, %9 : tensor<1x1x1x64xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x1x1x64xf32>
    %12 = stablehlo.reshape %arg313 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<1x1x1x64xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %15 = stablehlo.multiply %8, %14 : tensor<256x112x112x64xf32>
    %16 = stablehlo.reshape %arg312 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %18 = stablehlo.add %15, %17 : tensor<256x112x112x64xf32>
    %19 = stablehlo.convert %18 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %20 = call @relu(%19) : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xbf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %21 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %22 = "stablehlo.reduce_window"(%20, %21) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg842: tensor<bf16>, %arg843: tensor<bf16>):
      %2021 = stablehlo.maximum %arg842, %arg843 : tensor<bf16>
      stablehlo.return %2021 : tensor<bf16>
    }) : (tensor<256x112x112x64xbf16>, tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %23 = stablehlo.convert %arg7 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xbf16>
    %24 = stablehlo.convolution(%22, %23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %25 = stablehlo.broadcast_in_dim %arg630, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %26 = stablehlo.broadcast_in_dim %arg631, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %27 = stablehlo.convert %24 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %29 = stablehlo.subtract %27, %28 : tensor<256x56x56x64xf32>
    %30 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %31 = stablehlo.add %26, %30 : tensor<1x1x1x64xf32>
    %32 = stablehlo.rsqrt %31 : tensor<1x1x1x64xf32>
    %33 = stablehlo.reshape %arg2 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<1x1x1x64xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %36 = stablehlo.multiply %29, %35 : tensor<256x56x56x64xf32>
    %37 = stablehlo.reshape %arg1 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %39 = stablehlo.add %36, %38 : tensor<256x56x56x64xf32>
    %40 = stablehlo.convert %39 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %41 = call @relu_13(%40) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %42 = stablehlo.convert %arg8 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %43 = stablehlo.convolution(%41, %42) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %44 = stablehlo.broadcast_in_dim %arg632, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %45 = stablehlo.broadcast_in_dim %arg633, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %46 = stablehlo.convert %43 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %47 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %48 = stablehlo.subtract %46, %47 : tensor<256x56x56x64xf32>
    %49 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %50 = stablehlo.add %45, %49 : tensor<1x1x1x64xf32>
    %51 = stablehlo.rsqrt %50 : tensor<1x1x1x64xf32>
    %52 = stablehlo.reshape %arg4 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %53 = stablehlo.multiply %51, %52 : tensor<1x1x1x64xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %55 = stablehlo.multiply %48, %54 : tensor<256x56x56x64xf32>
    %56 = stablehlo.reshape %arg3 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %58 = stablehlo.add %55, %57 : tensor<256x56x56x64xf32>
    %59 = stablehlo.convert %58 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %60 = call @relu_13(%59) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %61 = stablehlo.convert %arg9 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xbf16>
    %62 = stablehlo.convolution(%60, %61) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x256xbf16>) -> tensor<256x56x56x256xbf16>
    %63 = stablehlo.broadcast_in_dim %arg634, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %64 = stablehlo.broadcast_in_dim %arg635, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %65 = stablehlo.convert %62 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %66 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %67 = stablehlo.subtract %65, %66 : tensor<256x56x56x256xf32>
    %68 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %69 = stablehlo.add %64, %68 : tensor<1x1x1x256xf32>
    %70 = stablehlo.rsqrt %69 : tensor<1x1x1x256xf32>
    %71 = stablehlo.reshape %arg6 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %72 = stablehlo.multiply %70, %71 : tensor<1x1x1x256xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %74 = stablehlo.multiply %67, %73 : tensor<256x56x56x256xf32>
    %75 = stablehlo.reshape %arg5 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %77 = stablehlo.add %74, %76 : tensor<256x56x56x256xf32>
    %78 = stablehlo.convert %77 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
    %79 = stablehlo.convert %arg10 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xbf16>
    %80 = stablehlo.convolution(%22, %79) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x256xbf16>) -> tensor<256x56x56x256xbf16>
    %81 = stablehlo.broadcast_in_dim %arg636, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %82 = stablehlo.broadcast_in_dim %arg637, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %83 = stablehlo.convert %80 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %84 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %85 = stablehlo.subtract %83, %84 : tensor<256x56x56x256xf32>
    %86 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %87 = stablehlo.add %82, %86 : tensor<1x1x1x256xf32>
    %88 = stablehlo.rsqrt %87 : tensor<1x1x1x256xf32>
    %89 = stablehlo.reshape %arg12 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %90 = stablehlo.multiply %88, %89 : tensor<1x1x1x256xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %92 = stablehlo.multiply %85, %91 : tensor<256x56x56x256xf32>
    %93 = stablehlo.reshape %arg11 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %94 = stablehlo.broadcast_in_dim %93, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %95 = stablehlo.add %92, %94 : tensor<256x56x56x256xf32>
    %96 = stablehlo.convert %95 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
    %97 = stablehlo.add %96, %78 : tensor<256x56x56x256xbf16>
    %98 = call @relu_32(%97) : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xbf16>
    %99 = stablehlo.convert %arg19 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xbf16>
    %100 = stablehlo.convolution(%98, %99) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xbf16>, tensor<1x1x256x64xbf16>) -> tensor<256x56x56x64xbf16>
    %101 = stablehlo.broadcast_in_dim %arg638, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %102 = stablehlo.broadcast_in_dim %arg639, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %103 = stablehlo.convert %100 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %104 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %105 = stablehlo.subtract %103, %104 : tensor<256x56x56x64xf32>
    %106 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %107 = stablehlo.add %102, %106 : tensor<1x1x1x64xf32>
    %108 = stablehlo.rsqrt %107 : tensor<1x1x1x64xf32>
    %109 = stablehlo.reshape %arg14 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %110 = stablehlo.multiply %108, %109 : tensor<1x1x1x64xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %112 = stablehlo.multiply %105, %111 : tensor<256x56x56x64xf32>
    %113 = stablehlo.reshape %arg13 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %115 = stablehlo.add %112, %114 : tensor<256x56x56x64xf32>
    %116 = stablehlo.convert %115 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %117 = call @relu_13(%116) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %118 = stablehlo.convert %arg20 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %119 = stablehlo.convolution(%117, %118) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %120 = stablehlo.broadcast_in_dim %arg640, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %121 = stablehlo.broadcast_in_dim %arg641, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %122 = stablehlo.convert %119 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %123 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %124 = stablehlo.subtract %122, %123 : tensor<256x56x56x64xf32>
    %125 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %126 = stablehlo.add %121, %125 : tensor<1x1x1x64xf32>
    %127 = stablehlo.rsqrt %126 : tensor<1x1x1x64xf32>
    %128 = stablehlo.reshape %arg16 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %129 = stablehlo.multiply %127, %128 : tensor<1x1x1x64xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %131 = stablehlo.multiply %124, %130 : tensor<256x56x56x64xf32>
    %132 = stablehlo.reshape %arg15 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %134 = stablehlo.add %131, %133 : tensor<256x56x56x64xf32>
    %135 = stablehlo.convert %134 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %136 = call @relu_13(%135) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %137 = stablehlo.convert %arg21 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xbf16>
    %138 = stablehlo.convolution(%136, %137) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x256xbf16>) -> tensor<256x56x56x256xbf16>
    %139 = stablehlo.broadcast_in_dim %arg642, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %140 = stablehlo.broadcast_in_dim %arg643, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %141 = stablehlo.convert %138 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %142 = stablehlo.broadcast_in_dim %139, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %143 = stablehlo.subtract %141, %142 : tensor<256x56x56x256xf32>
    %144 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %145 = stablehlo.add %140, %144 : tensor<1x1x1x256xf32>
    %146 = stablehlo.rsqrt %145 : tensor<1x1x1x256xf32>
    %147 = stablehlo.reshape %arg18 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %148 = stablehlo.multiply %146, %147 : tensor<1x1x1x256xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %150 = stablehlo.multiply %143, %149 : tensor<256x56x56x256xf32>
    %151 = stablehlo.reshape %arg17 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %153 = stablehlo.add %150, %152 : tensor<256x56x56x256xf32>
    %154 = stablehlo.convert %153 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
    %155 = stablehlo.add %98, %154 : tensor<256x56x56x256xbf16>
    %156 = call @relu_32(%155) : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xbf16>
    %157 = stablehlo.convert %arg118 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xbf16>
    %158 = stablehlo.convolution(%156, %157) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xbf16>, tensor<1x1x256x64xbf16>) -> tensor<256x56x56x64xbf16>
    %159 = stablehlo.broadcast_in_dim %arg704, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %160 = stablehlo.broadcast_in_dim %arg705, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %161 = stablehlo.convert %158 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %162 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %163 = stablehlo.subtract %161, %162 : tensor<256x56x56x64xf32>
    %164 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %165 = stablehlo.add %160, %164 : tensor<1x1x1x64xf32>
    %166 = stablehlo.rsqrt %165 : tensor<1x1x1x64xf32>
    %167 = stablehlo.reshape %arg113 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %168 = stablehlo.multiply %166, %167 : tensor<1x1x1x64xf32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %170 = stablehlo.multiply %163, %169 : tensor<256x56x56x64xf32>
    %171 = stablehlo.reshape %arg112 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %173 = stablehlo.add %170, %172 : tensor<256x56x56x64xf32>
    %174 = stablehlo.convert %173 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %175 = call @relu_13(%174) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %176 = stablehlo.convert %arg119 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %177 = stablehlo.convolution(%175, %176) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %178 = stablehlo.broadcast_in_dim %arg706, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %179 = stablehlo.broadcast_in_dim %arg707, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %180 = stablehlo.convert %177 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %181 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %182 = stablehlo.subtract %180, %181 : tensor<256x56x56x64xf32>
    %183 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %184 = stablehlo.add %179, %183 : tensor<1x1x1x64xf32>
    %185 = stablehlo.rsqrt %184 : tensor<1x1x1x64xf32>
    %186 = stablehlo.reshape %arg115 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %187 = stablehlo.multiply %185, %186 : tensor<1x1x1x64xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %189 = stablehlo.multiply %182, %188 : tensor<256x56x56x64xf32>
    %190 = stablehlo.reshape %arg114 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %192 = stablehlo.add %189, %191 : tensor<256x56x56x64xf32>
    %193 = stablehlo.convert %192 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %194 = call @relu_13(%193) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %195 = stablehlo.convert %arg120 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xbf16>
    %196 = stablehlo.convolution(%194, %195) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x256xbf16>) -> tensor<256x56x56x256xbf16>
    %197 = stablehlo.broadcast_in_dim %arg708, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %198 = stablehlo.broadcast_in_dim %arg709, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %199 = stablehlo.convert %196 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %200 = stablehlo.broadcast_in_dim %197, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %201 = stablehlo.subtract %199, %200 : tensor<256x56x56x256xf32>
    %202 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %203 = stablehlo.add %198, %202 : tensor<1x1x1x256xf32>
    %204 = stablehlo.rsqrt %203 : tensor<1x1x1x256xf32>
    %205 = stablehlo.reshape %arg117 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %206 = stablehlo.multiply %204, %205 : tensor<1x1x1x256xf32>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %208 = stablehlo.multiply %201, %207 : tensor<256x56x56x256xf32>
    %209 = stablehlo.reshape %arg116 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %211 = stablehlo.add %208, %210 : tensor<256x56x56x256xf32>
    %212 = stablehlo.convert %211 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
    %213 = stablehlo.add %156, %212 : tensor<256x56x56x256xbf16>
    %214 = call @relu_32(%213) : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xbf16>
    %215 = stablehlo.convert %arg217 : (tensor<1x1x256x128xf32>) -> tensor<1x1x256x128xbf16>
    %216 = stablehlo.convolution(%214, %215) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xbf16>, tensor<1x1x256x128xbf16>) -> tensor<256x56x56x128xbf16>
    %217 = stablehlo.broadcast_in_dim %arg770, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %218 = stablehlo.broadcast_in_dim %arg771, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %219 = stablehlo.convert %216 : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xf32>
    %220 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
    %221 = stablehlo.subtract %219, %220 : tensor<256x56x56x128xf32>
    %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %223 = stablehlo.add %218, %222 : tensor<1x1x1x128xf32>
    %224 = stablehlo.rsqrt %223 : tensor<1x1x1x128xf32>
    %225 = stablehlo.reshape %arg212 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %226 = stablehlo.multiply %224, %225 : tensor<1x1x1x128xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
    %228 = stablehlo.multiply %221, %227 : tensor<256x56x56x128xf32>
    %229 = stablehlo.reshape %arg211 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
    %231 = stablehlo.add %228, %230 : tensor<256x56x56x128xf32>
    %232 = stablehlo.convert %231 : (tensor<256x56x56x128xf32>) -> tensor<256x56x56x128xbf16>
    %233 = call @relu_55(%232) : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xbf16>
    %234 = stablehlo.convert %arg218 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %235 = stablehlo.convolution(%233, %234) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %236 = stablehlo.broadcast_in_dim %arg772, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %237 = stablehlo.broadcast_in_dim %arg773, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %238 = stablehlo.convert %235 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %239 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %240 = stablehlo.subtract %238, %239 : tensor<256x28x28x128xf32>
    %241 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %242 = stablehlo.add %237, %241 : tensor<1x1x1x128xf32>
    %243 = stablehlo.rsqrt %242 : tensor<1x1x1x128xf32>
    %244 = stablehlo.reshape %arg214 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %245 = stablehlo.multiply %243, %244 : tensor<1x1x1x128xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %247 = stablehlo.multiply %240, %246 : tensor<256x28x28x128xf32>
    %248 = stablehlo.reshape %arg213 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %249 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %250 = stablehlo.add %247, %249 : tensor<256x28x28x128xf32>
    %251 = stablehlo.convert %250 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %252 = call @relu_65(%251) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %253 = stablehlo.convert %arg219 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %254 = stablehlo.convolution(%252, %253) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %255 = stablehlo.broadcast_in_dim %arg774, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %256 = stablehlo.broadcast_in_dim %arg775, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %257 = stablehlo.convert %254 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %258 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %259 = stablehlo.subtract %257, %258 : tensor<256x28x28x512xf32>
    %260 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %261 = stablehlo.add %256, %260 : tensor<1x1x1x512xf32>
    %262 = stablehlo.rsqrt %261 : tensor<1x1x1x512xf32>
    %263 = stablehlo.reshape %arg216 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %264 = stablehlo.multiply %262, %263 : tensor<1x1x1x512xf32>
    %265 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %266 = stablehlo.multiply %259, %265 : tensor<256x28x28x512xf32>
    %267 = stablehlo.reshape %arg215 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %269 = stablehlo.add %266, %268 : tensor<256x28x28x512xf32>
    %270 = stablehlo.convert %269 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %271 = stablehlo.convert %arg220 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xbf16>
    %272 = stablehlo.convolution(%214, %271) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xbf16>, tensor<1x1x256x512xbf16>) -> tensor<256x28x28x512xbf16>
    %273 = stablehlo.broadcast_in_dim %arg776, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %274 = stablehlo.broadcast_in_dim %arg777, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %275 = stablehlo.convert %272 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %276 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %277 = stablehlo.subtract %275, %276 : tensor<256x28x28x512xf32>
    %278 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %279 = stablehlo.add %274, %278 : tensor<1x1x1x512xf32>
    %280 = stablehlo.rsqrt %279 : tensor<1x1x1x512xf32>
    %281 = stablehlo.reshape %arg222 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %282 = stablehlo.multiply %280, %281 : tensor<1x1x1x512xf32>
    %283 = stablehlo.broadcast_in_dim %282, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %284 = stablehlo.multiply %277, %283 : tensor<256x28x28x512xf32>
    %285 = stablehlo.reshape %arg221 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %287 = stablehlo.add %284, %286 : tensor<256x28x28x512xf32>
    %288 = stablehlo.convert %287 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %289 = stablehlo.add %288, %270 : tensor<256x28x28x512xbf16>
    %290 = call @relu_83(%289) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %291 = stablehlo.convert %arg259 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %292 = stablehlo.convolution(%290, %291) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %293 = stablehlo.broadcast_in_dim %arg798, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %294 = stablehlo.broadcast_in_dim %arg799, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %295 = stablehlo.convert %292 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %296 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %297 = stablehlo.subtract %295, %296 : tensor<256x28x28x128xf32>
    %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %299 = stablehlo.add %294, %298 : tensor<1x1x1x128xf32>
    %300 = stablehlo.rsqrt %299 : tensor<1x1x1x128xf32>
    %301 = stablehlo.reshape %arg254 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %302 = stablehlo.multiply %300, %301 : tensor<1x1x1x128xf32>
    %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %304 = stablehlo.multiply %297, %303 : tensor<256x28x28x128xf32>
    %305 = stablehlo.reshape %arg253 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %307 = stablehlo.add %304, %306 : tensor<256x28x28x128xf32>
    %308 = stablehlo.convert %307 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %309 = call @relu_65(%308) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %310 = stablehlo.convert %arg260 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %311 = stablehlo.convolution(%309, %310) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %312 = stablehlo.broadcast_in_dim %arg800, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %313 = stablehlo.broadcast_in_dim %arg801, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %314 = stablehlo.convert %311 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %315 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %316 = stablehlo.subtract %314, %315 : tensor<256x28x28x128xf32>
    %317 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %318 = stablehlo.add %313, %317 : tensor<1x1x1x128xf32>
    %319 = stablehlo.rsqrt %318 : tensor<1x1x1x128xf32>
    %320 = stablehlo.reshape %arg256 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %321 = stablehlo.multiply %319, %320 : tensor<1x1x1x128xf32>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %323 = stablehlo.multiply %316, %322 : tensor<256x28x28x128xf32>
    %324 = stablehlo.reshape %arg255 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %325 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %326 = stablehlo.add %323, %325 : tensor<256x28x28x128xf32>
    %327 = stablehlo.convert %326 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %328 = call @relu_65(%327) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %329 = stablehlo.convert %arg261 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %330 = stablehlo.convolution(%328, %329) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %331 = stablehlo.broadcast_in_dim %arg802, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %332 = stablehlo.broadcast_in_dim %arg803, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %333 = stablehlo.convert %330 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %334 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %335 = stablehlo.subtract %333, %334 : tensor<256x28x28x512xf32>
    %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %337 = stablehlo.add %332, %336 : tensor<1x1x1x512xf32>
    %338 = stablehlo.rsqrt %337 : tensor<1x1x1x512xf32>
    %339 = stablehlo.reshape %arg258 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %340 = stablehlo.multiply %338, %339 : tensor<1x1x1x512xf32>
    %341 = stablehlo.broadcast_in_dim %340, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %342 = stablehlo.multiply %335, %341 : tensor<256x28x28x512xf32>
    %343 = stablehlo.reshape %arg257 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %345 = stablehlo.add %342, %344 : tensor<256x28x28x512xf32>
    %346 = stablehlo.convert %345 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %347 = stablehlo.add %290, %346 : tensor<256x28x28x512xbf16>
    %348 = call @relu_83(%347) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %349 = stablehlo.convert %arg268 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %350 = stablehlo.convolution(%348, %349) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %351 = stablehlo.broadcast_in_dim %arg804, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %352 = stablehlo.broadcast_in_dim %arg805, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %353 = stablehlo.convert %350 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %354 = stablehlo.broadcast_in_dim %351, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %355 = stablehlo.subtract %353, %354 : tensor<256x28x28x128xf32>
    %356 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %357 = stablehlo.add %352, %356 : tensor<1x1x1x128xf32>
    %358 = stablehlo.rsqrt %357 : tensor<1x1x1x128xf32>
    %359 = stablehlo.reshape %arg263 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %360 = stablehlo.multiply %358, %359 : tensor<1x1x1x128xf32>
    %361 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %362 = stablehlo.multiply %355, %361 : tensor<256x28x28x128xf32>
    %363 = stablehlo.reshape %arg262 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %365 = stablehlo.add %362, %364 : tensor<256x28x28x128xf32>
    %366 = stablehlo.convert %365 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %367 = call @relu_65(%366) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %368 = stablehlo.convert %arg269 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %369 = stablehlo.convolution(%367, %368) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %370 = stablehlo.broadcast_in_dim %arg806, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %371 = stablehlo.broadcast_in_dim %arg807, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %372 = stablehlo.convert %369 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %373 = stablehlo.broadcast_in_dim %370, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %374 = stablehlo.subtract %372, %373 : tensor<256x28x28x128xf32>
    %375 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %376 = stablehlo.add %371, %375 : tensor<1x1x1x128xf32>
    %377 = stablehlo.rsqrt %376 : tensor<1x1x1x128xf32>
    %378 = stablehlo.reshape %arg265 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %379 = stablehlo.multiply %377, %378 : tensor<1x1x1x128xf32>
    %380 = stablehlo.broadcast_in_dim %379, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %381 = stablehlo.multiply %374, %380 : tensor<256x28x28x128xf32>
    %382 = stablehlo.reshape %arg264 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %383 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %384 = stablehlo.add %381, %383 : tensor<256x28x28x128xf32>
    %385 = stablehlo.convert %384 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %386 = call @relu_65(%385) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %387 = stablehlo.convert %arg270 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %388 = stablehlo.convolution(%386, %387) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %389 = stablehlo.broadcast_in_dim %arg808, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %390 = stablehlo.broadcast_in_dim %arg809, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %391 = stablehlo.convert %388 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %392 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %393 = stablehlo.subtract %391, %392 : tensor<256x28x28x512xf32>
    %394 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %395 = stablehlo.add %390, %394 : tensor<1x1x1x512xf32>
    %396 = stablehlo.rsqrt %395 : tensor<1x1x1x512xf32>
    %397 = stablehlo.reshape %arg267 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %398 = stablehlo.multiply %396, %397 : tensor<1x1x1x512xf32>
    %399 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %400 = stablehlo.multiply %393, %399 : tensor<256x28x28x512xf32>
    %401 = stablehlo.reshape %arg266 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %403 = stablehlo.add %400, %402 : tensor<256x28x28x512xf32>
    %404 = stablehlo.convert %403 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %405 = stablehlo.add %348, %404 : tensor<256x28x28x512xbf16>
    %406 = call @relu_83(%405) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %407 = stablehlo.convert %arg277 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %408 = stablehlo.convolution(%406, %407) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %409 = stablehlo.broadcast_in_dim %arg810, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %410 = stablehlo.broadcast_in_dim %arg811, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %411 = stablehlo.convert %408 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %412 = stablehlo.broadcast_in_dim %409, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %413 = stablehlo.subtract %411, %412 : tensor<256x28x28x128xf32>
    %414 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %415 = stablehlo.add %410, %414 : tensor<1x1x1x128xf32>
    %416 = stablehlo.rsqrt %415 : tensor<1x1x1x128xf32>
    %417 = stablehlo.reshape %arg272 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %418 = stablehlo.multiply %416, %417 : tensor<1x1x1x128xf32>
    %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %420 = stablehlo.multiply %413, %419 : tensor<256x28x28x128xf32>
    %421 = stablehlo.reshape %arg271 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %423 = stablehlo.add %420, %422 : tensor<256x28x28x128xf32>
    %424 = stablehlo.convert %423 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %425 = call @relu_65(%424) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %426 = stablehlo.convert %arg278 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %427 = stablehlo.convolution(%425, %426) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %428 = stablehlo.broadcast_in_dim %arg812, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %429 = stablehlo.broadcast_in_dim %arg813, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %430 = stablehlo.convert %427 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %431 = stablehlo.broadcast_in_dim %428, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %432 = stablehlo.subtract %430, %431 : tensor<256x28x28x128xf32>
    %433 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %434 = stablehlo.add %429, %433 : tensor<1x1x1x128xf32>
    %435 = stablehlo.rsqrt %434 : tensor<1x1x1x128xf32>
    %436 = stablehlo.reshape %arg274 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %437 = stablehlo.multiply %435, %436 : tensor<1x1x1x128xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %439 = stablehlo.multiply %432, %438 : tensor<256x28x28x128xf32>
    %440 = stablehlo.reshape %arg273 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %441 = stablehlo.broadcast_in_dim %440, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %442 = stablehlo.add %439, %441 : tensor<256x28x28x128xf32>
    %443 = stablehlo.convert %442 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %444 = call @relu_65(%443) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %445 = stablehlo.convert %arg279 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %446 = stablehlo.convolution(%444, %445) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %447 = stablehlo.broadcast_in_dim %arg814, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %448 = stablehlo.broadcast_in_dim %arg815, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %449 = stablehlo.convert %446 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %450 = stablehlo.broadcast_in_dim %447, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %451 = stablehlo.subtract %449, %450 : tensor<256x28x28x512xf32>
    %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %453 = stablehlo.add %448, %452 : tensor<1x1x1x512xf32>
    %454 = stablehlo.rsqrt %453 : tensor<1x1x1x512xf32>
    %455 = stablehlo.reshape %arg276 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %456 = stablehlo.multiply %454, %455 : tensor<1x1x1x512xf32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %458 = stablehlo.multiply %451, %457 : tensor<256x28x28x512xf32>
    %459 = stablehlo.reshape %arg275 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %461 = stablehlo.add %458, %460 : tensor<256x28x28x512xf32>
    %462 = stablehlo.convert %461 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %463 = stablehlo.add %406, %462 : tensor<256x28x28x512xbf16>
    %464 = call @relu_83(%463) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %465 = stablehlo.convert %arg286 : (tensor<1x1x512x256xf32>) -> tensor<1x1x512x256xbf16>
    %466 = stablehlo.convolution(%464, %465) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x256xbf16>) -> tensor<256x28x28x256xbf16>
    %467 = stablehlo.broadcast_in_dim %arg816, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %468 = stablehlo.broadcast_in_dim %arg817, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %469 = stablehlo.convert %466 : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xf32>
    %470 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
    %471 = stablehlo.subtract %469, %470 : tensor<256x28x28x256xf32>
    %472 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %473 = stablehlo.add %468, %472 : tensor<1x1x1x256xf32>
    %474 = stablehlo.rsqrt %473 : tensor<1x1x1x256xf32>
    %475 = stablehlo.reshape %arg281 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %476 = stablehlo.multiply %474, %475 : tensor<1x1x1x256xf32>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
    %478 = stablehlo.multiply %471, %477 : tensor<256x28x28x256xf32>
    %479 = stablehlo.reshape %arg280 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
    %481 = stablehlo.add %478, %480 : tensor<256x28x28x256xf32>
    %482 = stablehlo.convert %481 : (tensor<256x28x28x256xf32>) -> tensor<256x28x28x256xbf16>
    %483 = call @relu_105(%482) : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xbf16>
    %484 = stablehlo.convert %arg287 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %485 = stablehlo.convolution(%483, %484) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %486 = stablehlo.broadcast_in_dim %arg818, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %487 = stablehlo.broadcast_in_dim %arg819, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %488 = stablehlo.convert %485 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %489 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %490 = stablehlo.subtract %488, %489 : tensor<256x14x14x256xf32>
    %491 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %492 = stablehlo.add %487, %491 : tensor<1x1x1x256xf32>
    %493 = stablehlo.rsqrt %492 : tensor<1x1x1x256xf32>
    %494 = stablehlo.reshape %arg283 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %495 = stablehlo.multiply %493, %494 : tensor<1x1x1x256xf32>
    %496 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %497 = stablehlo.multiply %490, %496 : tensor<256x14x14x256xf32>
    %498 = stablehlo.reshape %arg282 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %499 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %500 = stablehlo.add %497, %499 : tensor<256x14x14x256xf32>
    %501 = stablehlo.convert %500 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %502 = call @relu_115(%501) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %503 = stablehlo.convert %arg288 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %504 = stablehlo.convolution(%502, %503) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %505 = stablehlo.broadcast_in_dim %arg820, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %506 = stablehlo.broadcast_in_dim %arg821, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %507 = stablehlo.convert %504 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %508 = stablehlo.broadcast_in_dim %505, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %509 = stablehlo.subtract %507, %508 : tensor<256x14x14x1024xf32>
    %510 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %511 = stablehlo.add %506, %510 : tensor<1x1x1x1024xf32>
    %512 = stablehlo.rsqrt %511 : tensor<1x1x1x1024xf32>
    %513 = stablehlo.reshape %arg285 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %514 = stablehlo.multiply %512, %513 : tensor<1x1x1x1024xf32>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %516 = stablehlo.multiply %509, %515 : tensor<256x14x14x1024xf32>
    %517 = stablehlo.reshape %arg284 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %519 = stablehlo.add %516, %518 : tensor<256x14x14x1024xf32>
    %520 = stablehlo.convert %519 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %521 = stablehlo.convert %arg289 : (tensor<1x1x512x1024xf32>) -> tensor<1x1x512x1024xbf16>
    %522 = stablehlo.convolution(%464, %521) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %523 = stablehlo.broadcast_in_dim %arg822, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %524 = stablehlo.broadcast_in_dim %arg823, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %525 = stablehlo.convert %522 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %526 = stablehlo.broadcast_in_dim %523, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %527 = stablehlo.subtract %525, %526 : tensor<256x14x14x1024xf32>
    %528 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %529 = stablehlo.add %524, %528 : tensor<1x1x1x1024xf32>
    %530 = stablehlo.rsqrt %529 : tensor<1x1x1x1024xf32>
    %531 = stablehlo.reshape %arg291 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %532 = stablehlo.multiply %530, %531 : tensor<1x1x1x1024xf32>
    %533 = stablehlo.broadcast_in_dim %532, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %534 = stablehlo.multiply %527, %533 : tensor<256x14x14x1024xf32>
    %535 = stablehlo.reshape %arg290 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %536 = stablehlo.broadcast_in_dim %535, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %537 = stablehlo.add %534, %536 : tensor<256x14x14x1024xf32>
    %538 = stablehlo.convert %537 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %539 = stablehlo.add %538, %520 : tensor<256x14x14x1024xbf16>
    %540 = call @relu_133(%539) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %541 = stablehlo.convert %arg298 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %542 = stablehlo.convolution(%540, %541) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %543 = stablehlo.broadcast_in_dim %arg824, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %544 = stablehlo.broadcast_in_dim %arg825, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %545 = stablehlo.convert %542 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %546 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %547 = stablehlo.subtract %545, %546 : tensor<256x14x14x256xf32>
    %548 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %549 = stablehlo.add %544, %548 : tensor<1x1x1x256xf32>
    %550 = stablehlo.rsqrt %549 : tensor<1x1x1x256xf32>
    %551 = stablehlo.reshape %arg293 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %552 = stablehlo.multiply %550, %551 : tensor<1x1x1x256xf32>
    %553 = stablehlo.broadcast_in_dim %552, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %554 = stablehlo.multiply %547, %553 : tensor<256x14x14x256xf32>
    %555 = stablehlo.reshape %arg292 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %556 = stablehlo.broadcast_in_dim %555, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %557 = stablehlo.add %554, %556 : tensor<256x14x14x256xf32>
    %558 = stablehlo.convert %557 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %559 = call @relu_115(%558) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %560 = stablehlo.convert %arg299 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %561 = stablehlo.convolution(%559, %560) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %562 = stablehlo.broadcast_in_dim %arg826, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %563 = stablehlo.broadcast_in_dim %arg827, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %564 = stablehlo.convert %561 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %565 = stablehlo.broadcast_in_dim %562, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %566 = stablehlo.subtract %564, %565 : tensor<256x14x14x256xf32>
    %567 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %568 = stablehlo.add %563, %567 : tensor<1x1x1x256xf32>
    %569 = stablehlo.rsqrt %568 : tensor<1x1x1x256xf32>
    %570 = stablehlo.reshape %arg295 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %571 = stablehlo.multiply %569, %570 : tensor<1x1x1x256xf32>
    %572 = stablehlo.broadcast_in_dim %571, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %573 = stablehlo.multiply %566, %572 : tensor<256x14x14x256xf32>
    %574 = stablehlo.reshape %arg294 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %575 = stablehlo.broadcast_in_dim %574, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %576 = stablehlo.add %573, %575 : tensor<256x14x14x256xf32>
    %577 = stablehlo.convert %576 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %578 = call @relu_115(%577) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %579 = stablehlo.convert %arg300 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %580 = stablehlo.convolution(%578, %579) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %581 = stablehlo.broadcast_in_dim %arg828, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %582 = stablehlo.broadcast_in_dim %arg829, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %583 = stablehlo.convert %580 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %584 = stablehlo.broadcast_in_dim %581, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %585 = stablehlo.subtract %583, %584 : tensor<256x14x14x1024xf32>
    %586 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %587 = stablehlo.add %582, %586 : tensor<1x1x1x1024xf32>
    %588 = stablehlo.rsqrt %587 : tensor<1x1x1x1024xf32>
    %589 = stablehlo.reshape %arg297 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %590 = stablehlo.multiply %588, %589 : tensor<1x1x1x1024xf32>
    %591 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %592 = stablehlo.multiply %585, %591 : tensor<256x14x14x1024xf32>
    %593 = stablehlo.reshape %arg296 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %594 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %595 = stablehlo.add %592, %594 : tensor<256x14x14x1024xf32>
    %596 = stablehlo.convert %595 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %597 = stablehlo.add %540, %596 : tensor<256x14x14x1024xbf16>
    %598 = call @relu_133(%597) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %599 = stablehlo.convert %arg307 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %600 = stablehlo.convolution(%598, %599) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %601 = stablehlo.broadcast_in_dim %arg830, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %602 = stablehlo.broadcast_in_dim %arg831, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %603 = stablehlo.convert %600 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %604 = stablehlo.broadcast_in_dim %601, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %605 = stablehlo.subtract %603, %604 : tensor<256x14x14x256xf32>
    %606 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %607 = stablehlo.add %602, %606 : tensor<1x1x1x256xf32>
    %608 = stablehlo.rsqrt %607 : tensor<1x1x1x256xf32>
    %609 = stablehlo.reshape %arg302 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %610 = stablehlo.multiply %608, %609 : tensor<1x1x1x256xf32>
    %611 = stablehlo.broadcast_in_dim %610, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %612 = stablehlo.multiply %605, %611 : tensor<256x14x14x256xf32>
    %613 = stablehlo.reshape %arg301 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %614 = stablehlo.broadcast_in_dim %613, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %615 = stablehlo.add %612, %614 : tensor<256x14x14x256xf32>
    %616 = stablehlo.convert %615 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %617 = call @relu_115(%616) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %618 = stablehlo.convert %arg308 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %619 = stablehlo.convolution(%617, %618) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %620 = stablehlo.broadcast_in_dim %arg832, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %621 = stablehlo.broadcast_in_dim %arg833, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %622 = stablehlo.convert %619 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %623 = stablehlo.broadcast_in_dim %620, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %624 = stablehlo.subtract %622, %623 : tensor<256x14x14x256xf32>
    %625 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %626 = stablehlo.add %621, %625 : tensor<1x1x1x256xf32>
    %627 = stablehlo.rsqrt %626 : tensor<1x1x1x256xf32>
    %628 = stablehlo.reshape %arg304 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %629 = stablehlo.multiply %627, %628 : tensor<1x1x1x256xf32>
    %630 = stablehlo.broadcast_in_dim %629, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %631 = stablehlo.multiply %624, %630 : tensor<256x14x14x256xf32>
    %632 = stablehlo.reshape %arg303 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %633 = stablehlo.broadcast_in_dim %632, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %634 = stablehlo.add %631, %633 : tensor<256x14x14x256xf32>
    %635 = stablehlo.convert %634 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %636 = call @relu_115(%635) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %637 = stablehlo.convert %arg309 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %638 = stablehlo.convolution(%636, %637) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %639 = stablehlo.broadcast_in_dim %arg834, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %640 = stablehlo.broadcast_in_dim %arg835, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %641 = stablehlo.convert %638 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %642 = stablehlo.broadcast_in_dim %639, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %643 = stablehlo.subtract %641, %642 : tensor<256x14x14x1024xf32>
    %644 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %645 = stablehlo.add %640, %644 : tensor<1x1x1x1024xf32>
    %646 = stablehlo.rsqrt %645 : tensor<1x1x1x1024xf32>
    %647 = stablehlo.reshape %arg306 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %648 = stablehlo.multiply %646, %647 : tensor<1x1x1x1024xf32>
    %649 = stablehlo.broadcast_in_dim %648, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %650 = stablehlo.multiply %643, %649 : tensor<256x14x14x1024xf32>
    %651 = stablehlo.reshape %arg305 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %652 = stablehlo.broadcast_in_dim %651, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %653 = stablehlo.add %650, %652 : tensor<256x14x14x1024xf32>
    %654 = stablehlo.convert %653 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %655 = stablehlo.add %598, %654 : tensor<256x14x14x1024xbf16>
    %656 = call @relu_133(%655) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %657 = stablehlo.convert %arg28 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %658 = stablehlo.convolution(%656, %657) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %659 = stablehlo.broadcast_in_dim %arg644, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %660 = stablehlo.broadcast_in_dim %arg645, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %661 = stablehlo.convert %658 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %662 = stablehlo.broadcast_in_dim %659, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %663 = stablehlo.subtract %661, %662 : tensor<256x14x14x256xf32>
    %664 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %665 = stablehlo.add %660, %664 : tensor<1x1x1x256xf32>
    %666 = stablehlo.rsqrt %665 : tensor<1x1x1x256xf32>
    %667 = stablehlo.reshape %arg23 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %668 = stablehlo.multiply %666, %667 : tensor<1x1x1x256xf32>
    %669 = stablehlo.broadcast_in_dim %668, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %670 = stablehlo.multiply %663, %669 : tensor<256x14x14x256xf32>
    %671 = stablehlo.reshape %arg22 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %672 = stablehlo.broadcast_in_dim %671, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %673 = stablehlo.add %670, %672 : tensor<256x14x14x256xf32>
    %674 = stablehlo.convert %673 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %675 = call @relu_115(%674) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %676 = stablehlo.convert %arg29 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %677 = stablehlo.convolution(%675, %676) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %678 = stablehlo.broadcast_in_dim %arg646, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %679 = stablehlo.broadcast_in_dim %arg647, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %680 = stablehlo.convert %677 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %681 = stablehlo.broadcast_in_dim %678, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %682 = stablehlo.subtract %680, %681 : tensor<256x14x14x256xf32>
    %683 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %684 = stablehlo.add %679, %683 : tensor<1x1x1x256xf32>
    %685 = stablehlo.rsqrt %684 : tensor<1x1x1x256xf32>
    %686 = stablehlo.reshape %arg25 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %687 = stablehlo.multiply %685, %686 : tensor<1x1x1x256xf32>
    %688 = stablehlo.broadcast_in_dim %687, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %689 = stablehlo.multiply %682, %688 : tensor<256x14x14x256xf32>
    %690 = stablehlo.reshape %arg24 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %691 = stablehlo.broadcast_in_dim %690, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %692 = stablehlo.add %689, %691 : tensor<256x14x14x256xf32>
    %693 = stablehlo.convert %692 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %694 = call @relu_115(%693) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %695 = stablehlo.convert %arg30 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %696 = stablehlo.convolution(%694, %695) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %697 = stablehlo.broadcast_in_dim %arg648, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %698 = stablehlo.broadcast_in_dim %arg649, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %699 = stablehlo.convert %696 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %700 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %701 = stablehlo.subtract %699, %700 : tensor<256x14x14x1024xf32>
    %702 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %703 = stablehlo.add %698, %702 : tensor<1x1x1x1024xf32>
    %704 = stablehlo.rsqrt %703 : tensor<1x1x1x1024xf32>
    %705 = stablehlo.reshape %arg27 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %706 = stablehlo.multiply %704, %705 : tensor<1x1x1x1024xf32>
    %707 = stablehlo.broadcast_in_dim %706, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %708 = stablehlo.multiply %701, %707 : tensor<256x14x14x1024xf32>
    %709 = stablehlo.reshape %arg26 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %710 = stablehlo.broadcast_in_dim %709, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %711 = stablehlo.add %708, %710 : tensor<256x14x14x1024xf32>
    %712 = stablehlo.convert %711 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %713 = stablehlo.add %656, %712 : tensor<256x14x14x1024xbf16>
    %714 = call @relu_133(%713) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %715 = stablehlo.convert %arg37 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %716 = stablehlo.convolution(%714, %715) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %717 = stablehlo.broadcast_in_dim %arg650, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %718 = stablehlo.broadcast_in_dim %arg651, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %719 = stablehlo.convert %716 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %720 = stablehlo.broadcast_in_dim %717, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %721 = stablehlo.subtract %719, %720 : tensor<256x14x14x256xf32>
    %722 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %723 = stablehlo.add %718, %722 : tensor<1x1x1x256xf32>
    %724 = stablehlo.rsqrt %723 : tensor<1x1x1x256xf32>
    %725 = stablehlo.reshape %arg32 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %726 = stablehlo.multiply %724, %725 : tensor<1x1x1x256xf32>
    %727 = stablehlo.broadcast_in_dim %726, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %728 = stablehlo.multiply %721, %727 : tensor<256x14x14x256xf32>
    %729 = stablehlo.reshape %arg31 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %730 = stablehlo.broadcast_in_dim %729, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %731 = stablehlo.add %728, %730 : tensor<256x14x14x256xf32>
    %732 = stablehlo.convert %731 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %733 = call @relu_115(%732) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %734 = stablehlo.convert %arg38 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %735 = stablehlo.convolution(%733, %734) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %736 = stablehlo.broadcast_in_dim %arg652, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %737 = stablehlo.broadcast_in_dim %arg653, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %738 = stablehlo.convert %735 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %739 = stablehlo.broadcast_in_dim %736, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %740 = stablehlo.subtract %738, %739 : tensor<256x14x14x256xf32>
    %741 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %742 = stablehlo.add %737, %741 : tensor<1x1x1x256xf32>
    %743 = stablehlo.rsqrt %742 : tensor<1x1x1x256xf32>
    %744 = stablehlo.reshape %arg34 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %745 = stablehlo.multiply %743, %744 : tensor<1x1x1x256xf32>
    %746 = stablehlo.broadcast_in_dim %745, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %747 = stablehlo.multiply %740, %746 : tensor<256x14x14x256xf32>
    %748 = stablehlo.reshape %arg33 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %749 = stablehlo.broadcast_in_dim %748, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %750 = stablehlo.add %747, %749 : tensor<256x14x14x256xf32>
    %751 = stablehlo.convert %750 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %752 = call @relu_115(%751) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %753 = stablehlo.convert %arg39 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %754 = stablehlo.convolution(%752, %753) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %755 = stablehlo.broadcast_in_dim %arg654, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %756 = stablehlo.broadcast_in_dim %arg655, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %757 = stablehlo.convert %754 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %758 = stablehlo.broadcast_in_dim %755, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %759 = stablehlo.subtract %757, %758 : tensor<256x14x14x1024xf32>
    %760 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %761 = stablehlo.add %756, %760 : tensor<1x1x1x1024xf32>
    %762 = stablehlo.rsqrt %761 : tensor<1x1x1x1024xf32>
    %763 = stablehlo.reshape %arg36 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %764 = stablehlo.multiply %762, %763 : tensor<1x1x1x1024xf32>
    %765 = stablehlo.broadcast_in_dim %764, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %766 = stablehlo.multiply %759, %765 : tensor<256x14x14x1024xf32>
    %767 = stablehlo.reshape %arg35 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %768 = stablehlo.broadcast_in_dim %767, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %769 = stablehlo.add %766, %768 : tensor<256x14x14x1024xf32>
    %770 = stablehlo.convert %769 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %771 = stablehlo.add %714, %770 : tensor<256x14x14x1024xbf16>
    %772 = call @relu_133(%771) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %773 = stablehlo.convert %arg46 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %774 = stablehlo.convolution(%772, %773) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %775 = stablehlo.broadcast_in_dim %arg656, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %776 = stablehlo.broadcast_in_dim %arg657, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %777 = stablehlo.convert %774 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %778 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %779 = stablehlo.subtract %777, %778 : tensor<256x14x14x256xf32>
    %780 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %781 = stablehlo.add %776, %780 : tensor<1x1x1x256xf32>
    %782 = stablehlo.rsqrt %781 : tensor<1x1x1x256xf32>
    %783 = stablehlo.reshape %arg41 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %784 = stablehlo.multiply %782, %783 : tensor<1x1x1x256xf32>
    %785 = stablehlo.broadcast_in_dim %784, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %786 = stablehlo.multiply %779, %785 : tensor<256x14x14x256xf32>
    %787 = stablehlo.reshape %arg40 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %789 = stablehlo.add %786, %788 : tensor<256x14x14x256xf32>
    %790 = stablehlo.convert %789 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %791 = call @relu_115(%790) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %792 = stablehlo.convert %arg47 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %793 = stablehlo.convolution(%791, %792) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %794 = stablehlo.broadcast_in_dim %arg658, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %795 = stablehlo.broadcast_in_dim %arg659, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %796 = stablehlo.convert %793 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %797 = stablehlo.broadcast_in_dim %794, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %798 = stablehlo.subtract %796, %797 : tensor<256x14x14x256xf32>
    %799 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %800 = stablehlo.add %795, %799 : tensor<1x1x1x256xf32>
    %801 = stablehlo.rsqrt %800 : tensor<1x1x1x256xf32>
    %802 = stablehlo.reshape %arg43 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %803 = stablehlo.multiply %801, %802 : tensor<1x1x1x256xf32>
    %804 = stablehlo.broadcast_in_dim %803, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %805 = stablehlo.multiply %798, %804 : tensor<256x14x14x256xf32>
    %806 = stablehlo.reshape %arg42 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %807 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %808 = stablehlo.add %805, %807 : tensor<256x14x14x256xf32>
    %809 = stablehlo.convert %808 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %810 = call @relu_115(%809) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %811 = stablehlo.convert %arg48 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %812 = stablehlo.convolution(%810, %811) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %813 = stablehlo.broadcast_in_dim %arg660, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %814 = stablehlo.broadcast_in_dim %arg661, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %815 = stablehlo.convert %812 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %816 = stablehlo.broadcast_in_dim %813, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %817 = stablehlo.subtract %815, %816 : tensor<256x14x14x1024xf32>
    %818 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %819 = stablehlo.add %814, %818 : tensor<1x1x1x1024xf32>
    %820 = stablehlo.rsqrt %819 : tensor<1x1x1x1024xf32>
    %821 = stablehlo.reshape %arg45 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %822 = stablehlo.multiply %820, %821 : tensor<1x1x1x1024xf32>
    %823 = stablehlo.broadcast_in_dim %822, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %824 = stablehlo.multiply %817, %823 : tensor<256x14x14x1024xf32>
    %825 = stablehlo.reshape %arg44 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %826 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %827 = stablehlo.add %824, %826 : tensor<256x14x14x1024xf32>
    %828 = stablehlo.convert %827 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %829 = stablehlo.add %772, %828 : tensor<256x14x14x1024xbf16>
    %830 = call @relu_133(%829) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %831 = stablehlo.convert %arg55 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %832 = stablehlo.convolution(%830, %831) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %833 = stablehlo.broadcast_in_dim %arg662, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %834 = stablehlo.broadcast_in_dim %arg663, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %835 = stablehlo.convert %832 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %836 = stablehlo.broadcast_in_dim %833, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %837 = stablehlo.subtract %835, %836 : tensor<256x14x14x256xf32>
    %838 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %839 = stablehlo.add %834, %838 : tensor<1x1x1x256xf32>
    %840 = stablehlo.rsqrt %839 : tensor<1x1x1x256xf32>
    %841 = stablehlo.reshape %arg50 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %842 = stablehlo.multiply %840, %841 : tensor<1x1x1x256xf32>
    %843 = stablehlo.broadcast_in_dim %842, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %844 = stablehlo.multiply %837, %843 : tensor<256x14x14x256xf32>
    %845 = stablehlo.reshape %arg49 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %846 = stablehlo.broadcast_in_dim %845, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %847 = stablehlo.add %844, %846 : tensor<256x14x14x256xf32>
    %848 = stablehlo.convert %847 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %849 = call @relu_115(%848) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %850 = stablehlo.convert %arg56 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %851 = stablehlo.convolution(%849, %850) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %852 = stablehlo.broadcast_in_dim %arg664, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %853 = stablehlo.broadcast_in_dim %arg665, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %854 = stablehlo.convert %851 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %855 = stablehlo.broadcast_in_dim %852, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %856 = stablehlo.subtract %854, %855 : tensor<256x14x14x256xf32>
    %857 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %858 = stablehlo.add %853, %857 : tensor<1x1x1x256xf32>
    %859 = stablehlo.rsqrt %858 : tensor<1x1x1x256xf32>
    %860 = stablehlo.reshape %arg52 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %861 = stablehlo.multiply %859, %860 : tensor<1x1x1x256xf32>
    %862 = stablehlo.broadcast_in_dim %861, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %863 = stablehlo.multiply %856, %862 : tensor<256x14x14x256xf32>
    %864 = stablehlo.reshape %arg51 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %865 = stablehlo.broadcast_in_dim %864, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %866 = stablehlo.add %863, %865 : tensor<256x14x14x256xf32>
    %867 = stablehlo.convert %866 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %868 = call @relu_115(%867) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %869 = stablehlo.convert %arg57 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %870 = stablehlo.convolution(%868, %869) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %871 = stablehlo.broadcast_in_dim %arg666, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %872 = stablehlo.broadcast_in_dim %arg667, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %873 = stablehlo.convert %870 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %874 = stablehlo.broadcast_in_dim %871, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %875 = stablehlo.subtract %873, %874 : tensor<256x14x14x1024xf32>
    %876 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %877 = stablehlo.add %872, %876 : tensor<1x1x1x1024xf32>
    %878 = stablehlo.rsqrt %877 : tensor<1x1x1x1024xf32>
    %879 = stablehlo.reshape %arg54 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %880 = stablehlo.multiply %878, %879 : tensor<1x1x1x1024xf32>
    %881 = stablehlo.broadcast_in_dim %880, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %882 = stablehlo.multiply %875, %881 : tensor<256x14x14x1024xf32>
    %883 = stablehlo.reshape %arg53 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %884 = stablehlo.broadcast_in_dim %883, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %885 = stablehlo.add %882, %884 : tensor<256x14x14x1024xf32>
    %886 = stablehlo.convert %885 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %887 = stablehlo.add %830, %886 : tensor<256x14x14x1024xbf16>
    %888 = call @relu_133(%887) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %889 = stablehlo.convert %arg64 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %890 = stablehlo.convolution(%888, %889) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %891 = stablehlo.broadcast_in_dim %arg668, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %892 = stablehlo.broadcast_in_dim %arg669, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %893 = stablehlo.convert %890 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %894 = stablehlo.broadcast_in_dim %891, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %895 = stablehlo.subtract %893, %894 : tensor<256x14x14x256xf32>
    %896 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %897 = stablehlo.add %892, %896 : tensor<1x1x1x256xf32>
    %898 = stablehlo.rsqrt %897 : tensor<1x1x1x256xf32>
    %899 = stablehlo.reshape %arg59 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %900 = stablehlo.multiply %898, %899 : tensor<1x1x1x256xf32>
    %901 = stablehlo.broadcast_in_dim %900, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %902 = stablehlo.multiply %895, %901 : tensor<256x14x14x256xf32>
    %903 = stablehlo.reshape %arg58 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %904 = stablehlo.broadcast_in_dim %903, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %905 = stablehlo.add %902, %904 : tensor<256x14x14x256xf32>
    %906 = stablehlo.convert %905 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %907 = call @relu_115(%906) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %908 = stablehlo.convert %arg65 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %909 = stablehlo.convolution(%907, %908) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %910 = stablehlo.broadcast_in_dim %arg670, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %911 = stablehlo.broadcast_in_dim %arg671, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %912 = stablehlo.convert %909 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %913 = stablehlo.broadcast_in_dim %910, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %914 = stablehlo.subtract %912, %913 : tensor<256x14x14x256xf32>
    %915 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %916 = stablehlo.add %911, %915 : tensor<1x1x1x256xf32>
    %917 = stablehlo.rsqrt %916 : tensor<1x1x1x256xf32>
    %918 = stablehlo.reshape %arg61 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %919 = stablehlo.multiply %917, %918 : tensor<1x1x1x256xf32>
    %920 = stablehlo.broadcast_in_dim %919, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %921 = stablehlo.multiply %914, %920 : tensor<256x14x14x256xf32>
    %922 = stablehlo.reshape %arg60 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %923 = stablehlo.broadcast_in_dim %922, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %924 = stablehlo.add %921, %923 : tensor<256x14x14x256xf32>
    %925 = stablehlo.convert %924 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %926 = call @relu_115(%925) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %927 = stablehlo.convert %arg66 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %928 = stablehlo.convolution(%926, %927) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %929 = stablehlo.broadcast_in_dim %arg672, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %930 = stablehlo.broadcast_in_dim %arg673, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %931 = stablehlo.convert %928 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %932 = stablehlo.broadcast_in_dim %929, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %933 = stablehlo.subtract %931, %932 : tensor<256x14x14x1024xf32>
    %934 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %935 = stablehlo.add %930, %934 : tensor<1x1x1x1024xf32>
    %936 = stablehlo.rsqrt %935 : tensor<1x1x1x1024xf32>
    %937 = stablehlo.reshape %arg63 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %938 = stablehlo.multiply %936, %937 : tensor<1x1x1x1024xf32>
    %939 = stablehlo.broadcast_in_dim %938, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %940 = stablehlo.multiply %933, %939 : tensor<256x14x14x1024xf32>
    %941 = stablehlo.reshape %arg62 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %942 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %943 = stablehlo.add %940, %942 : tensor<256x14x14x1024xf32>
    %944 = stablehlo.convert %943 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %945 = stablehlo.add %888, %944 : tensor<256x14x14x1024xbf16>
    %946 = call @relu_133(%945) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %947 = stablehlo.convert %arg73 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %948 = stablehlo.convolution(%946, %947) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %949 = stablehlo.broadcast_in_dim %arg674, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %950 = stablehlo.broadcast_in_dim %arg675, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %951 = stablehlo.convert %948 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %952 = stablehlo.broadcast_in_dim %949, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %953 = stablehlo.subtract %951, %952 : tensor<256x14x14x256xf32>
    %954 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %955 = stablehlo.add %950, %954 : tensor<1x1x1x256xf32>
    %956 = stablehlo.rsqrt %955 : tensor<1x1x1x256xf32>
    %957 = stablehlo.reshape %arg68 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %958 = stablehlo.multiply %956, %957 : tensor<1x1x1x256xf32>
    %959 = stablehlo.broadcast_in_dim %958, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %960 = stablehlo.multiply %953, %959 : tensor<256x14x14x256xf32>
    %961 = stablehlo.reshape %arg67 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %962 = stablehlo.broadcast_in_dim %961, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %963 = stablehlo.add %960, %962 : tensor<256x14x14x256xf32>
    %964 = stablehlo.convert %963 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %965 = call @relu_115(%964) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %966 = stablehlo.convert %arg74 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %967 = stablehlo.convolution(%965, %966) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %968 = stablehlo.broadcast_in_dim %arg676, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %969 = stablehlo.broadcast_in_dim %arg677, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %970 = stablehlo.convert %967 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %971 = stablehlo.broadcast_in_dim %968, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %972 = stablehlo.subtract %970, %971 : tensor<256x14x14x256xf32>
    %973 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %974 = stablehlo.add %969, %973 : tensor<1x1x1x256xf32>
    %975 = stablehlo.rsqrt %974 : tensor<1x1x1x256xf32>
    %976 = stablehlo.reshape %arg70 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %977 = stablehlo.multiply %975, %976 : tensor<1x1x1x256xf32>
    %978 = stablehlo.broadcast_in_dim %977, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %979 = stablehlo.multiply %972, %978 : tensor<256x14x14x256xf32>
    %980 = stablehlo.reshape %arg69 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %981 = stablehlo.broadcast_in_dim %980, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %982 = stablehlo.add %979, %981 : tensor<256x14x14x256xf32>
    %983 = stablehlo.convert %982 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %984 = call @relu_115(%983) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %985 = stablehlo.convert %arg75 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %986 = stablehlo.convolution(%984, %985) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %987 = stablehlo.broadcast_in_dim %arg678, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %988 = stablehlo.broadcast_in_dim %arg679, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %989 = stablehlo.convert %986 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %990 = stablehlo.broadcast_in_dim %987, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %991 = stablehlo.subtract %989, %990 : tensor<256x14x14x1024xf32>
    %992 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %993 = stablehlo.add %988, %992 : tensor<1x1x1x1024xf32>
    %994 = stablehlo.rsqrt %993 : tensor<1x1x1x1024xf32>
    %995 = stablehlo.reshape %arg72 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %996 = stablehlo.multiply %994, %995 : tensor<1x1x1x1024xf32>
    %997 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %998 = stablehlo.multiply %991, %997 : tensor<256x14x14x1024xf32>
    %999 = stablehlo.reshape %arg71 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1000 = stablehlo.broadcast_in_dim %999, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1001 = stablehlo.add %998, %1000 : tensor<256x14x14x1024xf32>
    %1002 = stablehlo.convert %1001 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1003 = stablehlo.add %946, %1002 : tensor<256x14x14x1024xbf16>
    %1004 = call @relu_133(%1003) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1005 = stablehlo.convert %arg82 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1006 = stablehlo.convolution(%1004, %1005) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1007 = stablehlo.broadcast_in_dim %arg680, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1008 = stablehlo.broadcast_in_dim %arg681, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1009 = stablehlo.convert %1006 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1010 = stablehlo.broadcast_in_dim %1007, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1011 = stablehlo.subtract %1009, %1010 : tensor<256x14x14x256xf32>
    %1012 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1013 = stablehlo.add %1008, %1012 : tensor<1x1x1x256xf32>
    %1014 = stablehlo.rsqrt %1013 : tensor<1x1x1x256xf32>
    %1015 = stablehlo.reshape %arg77 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1016 = stablehlo.multiply %1014, %1015 : tensor<1x1x1x256xf32>
    %1017 = stablehlo.broadcast_in_dim %1016, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1018 = stablehlo.multiply %1011, %1017 : tensor<256x14x14x256xf32>
    %1019 = stablehlo.reshape %arg76 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1020 = stablehlo.broadcast_in_dim %1019, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1021 = stablehlo.add %1018, %1020 : tensor<256x14x14x256xf32>
    %1022 = stablehlo.convert %1021 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1023 = call @relu_115(%1022) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1024 = stablehlo.convert %arg83 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1025 = stablehlo.convolution(%1023, %1024) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1026 = stablehlo.broadcast_in_dim %arg682, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1027 = stablehlo.broadcast_in_dim %arg683, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1028 = stablehlo.convert %1025 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1029 = stablehlo.broadcast_in_dim %1026, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1030 = stablehlo.subtract %1028, %1029 : tensor<256x14x14x256xf32>
    %1031 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1032 = stablehlo.add %1027, %1031 : tensor<1x1x1x256xf32>
    %1033 = stablehlo.rsqrt %1032 : tensor<1x1x1x256xf32>
    %1034 = stablehlo.reshape %arg79 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1035 = stablehlo.multiply %1033, %1034 : tensor<1x1x1x256xf32>
    %1036 = stablehlo.broadcast_in_dim %1035, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1037 = stablehlo.multiply %1030, %1036 : tensor<256x14x14x256xf32>
    %1038 = stablehlo.reshape %arg78 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1039 = stablehlo.broadcast_in_dim %1038, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1040 = stablehlo.add %1037, %1039 : tensor<256x14x14x256xf32>
    %1041 = stablehlo.convert %1040 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1042 = call @relu_115(%1041) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1043 = stablehlo.convert %arg84 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1044 = stablehlo.convolution(%1042, %1043) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1045 = stablehlo.broadcast_in_dim %arg684, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1046 = stablehlo.broadcast_in_dim %arg685, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1047 = stablehlo.convert %1044 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1048 = stablehlo.broadcast_in_dim %1045, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1049 = stablehlo.subtract %1047, %1048 : tensor<256x14x14x1024xf32>
    %1050 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1051 = stablehlo.add %1046, %1050 : tensor<1x1x1x1024xf32>
    %1052 = stablehlo.rsqrt %1051 : tensor<1x1x1x1024xf32>
    %1053 = stablehlo.reshape %arg81 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1054 = stablehlo.multiply %1052, %1053 : tensor<1x1x1x1024xf32>
    %1055 = stablehlo.broadcast_in_dim %1054, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1056 = stablehlo.multiply %1049, %1055 : tensor<256x14x14x1024xf32>
    %1057 = stablehlo.reshape %arg80 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1058 = stablehlo.broadcast_in_dim %1057, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1059 = stablehlo.add %1056, %1058 : tensor<256x14x14x1024xf32>
    %1060 = stablehlo.convert %1059 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1061 = stablehlo.add %1004, %1060 : tensor<256x14x14x1024xbf16>
    %1062 = call @relu_133(%1061) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1063 = stablehlo.convert %arg91 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1064 = stablehlo.convolution(%1062, %1063) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1065 = stablehlo.broadcast_in_dim %arg686, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1066 = stablehlo.broadcast_in_dim %arg687, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1067 = stablehlo.convert %1064 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1068 = stablehlo.broadcast_in_dim %1065, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1069 = stablehlo.subtract %1067, %1068 : tensor<256x14x14x256xf32>
    %1070 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1071 = stablehlo.add %1066, %1070 : tensor<1x1x1x256xf32>
    %1072 = stablehlo.rsqrt %1071 : tensor<1x1x1x256xf32>
    %1073 = stablehlo.reshape %arg86 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1074 = stablehlo.multiply %1072, %1073 : tensor<1x1x1x256xf32>
    %1075 = stablehlo.broadcast_in_dim %1074, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1076 = stablehlo.multiply %1069, %1075 : tensor<256x14x14x256xf32>
    %1077 = stablehlo.reshape %arg85 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1078 = stablehlo.broadcast_in_dim %1077, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1079 = stablehlo.add %1076, %1078 : tensor<256x14x14x256xf32>
    %1080 = stablehlo.convert %1079 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1081 = call @relu_115(%1080) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1082 = stablehlo.convert %arg92 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1083 = stablehlo.convolution(%1081, %1082) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1084 = stablehlo.broadcast_in_dim %arg688, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1085 = stablehlo.broadcast_in_dim %arg689, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1086 = stablehlo.convert %1083 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1087 = stablehlo.broadcast_in_dim %1084, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1088 = stablehlo.subtract %1086, %1087 : tensor<256x14x14x256xf32>
    %1089 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1090 = stablehlo.add %1085, %1089 : tensor<1x1x1x256xf32>
    %1091 = stablehlo.rsqrt %1090 : tensor<1x1x1x256xf32>
    %1092 = stablehlo.reshape %arg88 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1093 = stablehlo.multiply %1091, %1092 : tensor<1x1x1x256xf32>
    %1094 = stablehlo.broadcast_in_dim %1093, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1095 = stablehlo.multiply %1088, %1094 : tensor<256x14x14x256xf32>
    %1096 = stablehlo.reshape %arg87 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1097 = stablehlo.broadcast_in_dim %1096, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1098 = stablehlo.add %1095, %1097 : tensor<256x14x14x256xf32>
    %1099 = stablehlo.convert %1098 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1100 = call @relu_115(%1099) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1101 = stablehlo.convert %arg93 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1102 = stablehlo.convolution(%1100, %1101) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1103 = stablehlo.broadcast_in_dim %arg690, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1104 = stablehlo.broadcast_in_dim %arg691, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1105 = stablehlo.convert %1102 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1106 = stablehlo.broadcast_in_dim %1103, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1107 = stablehlo.subtract %1105, %1106 : tensor<256x14x14x1024xf32>
    %1108 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1109 = stablehlo.add %1104, %1108 : tensor<1x1x1x1024xf32>
    %1110 = stablehlo.rsqrt %1109 : tensor<1x1x1x1024xf32>
    %1111 = stablehlo.reshape %arg90 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1112 = stablehlo.multiply %1110, %1111 : tensor<1x1x1x1024xf32>
    %1113 = stablehlo.broadcast_in_dim %1112, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1114 = stablehlo.multiply %1107, %1113 : tensor<256x14x14x1024xf32>
    %1115 = stablehlo.reshape %arg89 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1116 = stablehlo.broadcast_in_dim %1115, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1117 = stablehlo.add %1114, %1116 : tensor<256x14x14x1024xf32>
    %1118 = stablehlo.convert %1117 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1119 = stablehlo.add %1062, %1118 : tensor<256x14x14x1024xbf16>
    %1120 = call @relu_133(%1119) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1121 = stablehlo.convert %arg100 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1122 = stablehlo.convolution(%1120, %1121) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1123 = stablehlo.broadcast_in_dim %arg692, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1124 = stablehlo.broadcast_in_dim %arg693, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1125 = stablehlo.convert %1122 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1126 = stablehlo.broadcast_in_dim %1123, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1127 = stablehlo.subtract %1125, %1126 : tensor<256x14x14x256xf32>
    %1128 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1129 = stablehlo.add %1124, %1128 : tensor<1x1x1x256xf32>
    %1130 = stablehlo.rsqrt %1129 : tensor<1x1x1x256xf32>
    %1131 = stablehlo.reshape %arg95 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1132 = stablehlo.multiply %1130, %1131 : tensor<1x1x1x256xf32>
    %1133 = stablehlo.broadcast_in_dim %1132, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1134 = stablehlo.multiply %1127, %1133 : tensor<256x14x14x256xf32>
    %1135 = stablehlo.reshape %arg94 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1136 = stablehlo.broadcast_in_dim %1135, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1137 = stablehlo.add %1134, %1136 : tensor<256x14x14x256xf32>
    %1138 = stablehlo.convert %1137 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1139 = call @relu_115(%1138) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1140 = stablehlo.convert %arg101 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1141 = stablehlo.convolution(%1139, %1140) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1142 = stablehlo.broadcast_in_dim %arg694, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1143 = stablehlo.broadcast_in_dim %arg695, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1144 = stablehlo.convert %1141 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1145 = stablehlo.broadcast_in_dim %1142, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1146 = stablehlo.subtract %1144, %1145 : tensor<256x14x14x256xf32>
    %1147 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1148 = stablehlo.add %1143, %1147 : tensor<1x1x1x256xf32>
    %1149 = stablehlo.rsqrt %1148 : tensor<1x1x1x256xf32>
    %1150 = stablehlo.reshape %arg97 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1151 = stablehlo.multiply %1149, %1150 : tensor<1x1x1x256xf32>
    %1152 = stablehlo.broadcast_in_dim %1151, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1153 = stablehlo.multiply %1146, %1152 : tensor<256x14x14x256xf32>
    %1154 = stablehlo.reshape %arg96 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1156 = stablehlo.add %1153, %1155 : tensor<256x14x14x256xf32>
    %1157 = stablehlo.convert %1156 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1158 = call @relu_115(%1157) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1159 = stablehlo.convert %arg102 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1160 = stablehlo.convolution(%1158, %1159) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1161 = stablehlo.broadcast_in_dim %arg696, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1162 = stablehlo.broadcast_in_dim %arg697, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1163 = stablehlo.convert %1160 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1164 = stablehlo.broadcast_in_dim %1161, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1165 = stablehlo.subtract %1163, %1164 : tensor<256x14x14x1024xf32>
    %1166 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1167 = stablehlo.add %1162, %1166 : tensor<1x1x1x1024xf32>
    %1168 = stablehlo.rsqrt %1167 : tensor<1x1x1x1024xf32>
    %1169 = stablehlo.reshape %arg99 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1170 = stablehlo.multiply %1168, %1169 : tensor<1x1x1x1024xf32>
    %1171 = stablehlo.broadcast_in_dim %1170, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1172 = stablehlo.multiply %1165, %1171 : tensor<256x14x14x1024xf32>
    %1173 = stablehlo.reshape %arg98 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1174 = stablehlo.broadcast_in_dim %1173, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1175 = stablehlo.add %1172, %1174 : tensor<256x14x14x1024xf32>
    %1176 = stablehlo.convert %1175 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1177 = stablehlo.add %1120, %1176 : tensor<256x14x14x1024xbf16>
    %1178 = call @relu_133(%1177) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1179 = stablehlo.convert %arg109 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1180 = stablehlo.convolution(%1178, %1179) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1181 = stablehlo.broadcast_in_dim %arg698, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1182 = stablehlo.broadcast_in_dim %arg699, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1183 = stablehlo.convert %1180 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1184 = stablehlo.broadcast_in_dim %1181, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1185 = stablehlo.subtract %1183, %1184 : tensor<256x14x14x256xf32>
    %1186 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1187 = stablehlo.add %1182, %1186 : tensor<1x1x1x256xf32>
    %1188 = stablehlo.rsqrt %1187 : tensor<1x1x1x256xf32>
    %1189 = stablehlo.reshape %arg104 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1190 = stablehlo.multiply %1188, %1189 : tensor<1x1x1x256xf32>
    %1191 = stablehlo.broadcast_in_dim %1190, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1192 = stablehlo.multiply %1185, %1191 : tensor<256x14x14x256xf32>
    %1193 = stablehlo.reshape %arg103 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1194 = stablehlo.broadcast_in_dim %1193, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1195 = stablehlo.add %1192, %1194 : tensor<256x14x14x256xf32>
    %1196 = stablehlo.convert %1195 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1197 = call @relu_115(%1196) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1198 = stablehlo.convert %arg110 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1199 = stablehlo.convolution(%1197, %1198) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1200 = stablehlo.broadcast_in_dim %arg700, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1201 = stablehlo.broadcast_in_dim %arg701, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1202 = stablehlo.convert %1199 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1203 = stablehlo.broadcast_in_dim %1200, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1204 = stablehlo.subtract %1202, %1203 : tensor<256x14x14x256xf32>
    %1205 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1206 = stablehlo.add %1201, %1205 : tensor<1x1x1x256xf32>
    %1207 = stablehlo.rsqrt %1206 : tensor<1x1x1x256xf32>
    %1208 = stablehlo.reshape %arg106 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1209 = stablehlo.multiply %1207, %1208 : tensor<1x1x1x256xf32>
    %1210 = stablehlo.broadcast_in_dim %1209, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1211 = stablehlo.multiply %1204, %1210 : tensor<256x14x14x256xf32>
    %1212 = stablehlo.reshape %arg105 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1213 = stablehlo.broadcast_in_dim %1212, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1214 = stablehlo.add %1211, %1213 : tensor<256x14x14x256xf32>
    %1215 = stablehlo.convert %1214 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1216 = call @relu_115(%1215) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1217 = stablehlo.convert %arg111 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1218 = stablehlo.convolution(%1216, %1217) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1219 = stablehlo.broadcast_in_dim %arg702, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1220 = stablehlo.broadcast_in_dim %arg703, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1221 = stablehlo.convert %1218 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1222 = stablehlo.broadcast_in_dim %1219, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1223 = stablehlo.subtract %1221, %1222 : tensor<256x14x14x1024xf32>
    %1224 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1225 = stablehlo.add %1220, %1224 : tensor<1x1x1x1024xf32>
    %1226 = stablehlo.rsqrt %1225 : tensor<1x1x1x1024xf32>
    %1227 = stablehlo.reshape %arg108 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1228 = stablehlo.multiply %1226, %1227 : tensor<1x1x1x1024xf32>
    %1229 = stablehlo.broadcast_in_dim %1228, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1230 = stablehlo.multiply %1223, %1229 : tensor<256x14x14x1024xf32>
    %1231 = stablehlo.reshape %arg107 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1232 = stablehlo.broadcast_in_dim %1231, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1233 = stablehlo.add %1230, %1232 : tensor<256x14x14x1024xf32>
    %1234 = stablehlo.convert %1233 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1235 = stablehlo.add %1178, %1234 : tensor<256x14x14x1024xbf16>
    %1236 = call @relu_133(%1235) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1237 = stablehlo.convert %arg127 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1238 = stablehlo.convolution(%1236, %1237) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1239 = stablehlo.broadcast_in_dim %arg710, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1240 = stablehlo.broadcast_in_dim %arg711, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1241 = stablehlo.convert %1238 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1242 = stablehlo.broadcast_in_dim %1239, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1243 = stablehlo.subtract %1241, %1242 : tensor<256x14x14x256xf32>
    %1244 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1245 = stablehlo.add %1240, %1244 : tensor<1x1x1x256xf32>
    %1246 = stablehlo.rsqrt %1245 : tensor<1x1x1x256xf32>
    %1247 = stablehlo.reshape %arg122 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1248 = stablehlo.multiply %1246, %1247 : tensor<1x1x1x256xf32>
    %1249 = stablehlo.broadcast_in_dim %1248, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1250 = stablehlo.multiply %1243, %1249 : tensor<256x14x14x256xf32>
    %1251 = stablehlo.reshape %arg121 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1252 = stablehlo.broadcast_in_dim %1251, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1253 = stablehlo.add %1250, %1252 : tensor<256x14x14x256xf32>
    %1254 = stablehlo.convert %1253 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1255 = call @relu_115(%1254) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1256 = stablehlo.convert %arg128 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1257 = stablehlo.convolution(%1255, %1256) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1258 = stablehlo.broadcast_in_dim %arg712, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1259 = stablehlo.broadcast_in_dim %arg713, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1260 = stablehlo.convert %1257 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1261 = stablehlo.broadcast_in_dim %1258, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1262 = stablehlo.subtract %1260, %1261 : tensor<256x14x14x256xf32>
    %1263 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1264 = stablehlo.add %1259, %1263 : tensor<1x1x1x256xf32>
    %1265 = stablehlo.rsqrt %1264 : tensor<1x1x1x256xf32>
    %1266 = stablehlo.reshape %arg124 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1267 = stablehlo.multiply %1265, %1266 : tensor<1x1x1x256xf32>
    %1268 = stablehlo.broadcast_in_dim %1267, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1269 = stablehlo.multiply %1262, %1268 : tensor<256x14x14x256xf32>
    %1270 = stablehlo.reshape %arg123 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1271 = stablehlo.broadcast_in_dim %1270, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1272 = stablehlo.add %1269, %1271 : tensor<256x14x14x256xf32>
    %1273 = stablehlo.convert %1272 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1274 = call @relu_115(%1273) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1275 = stablehlo.convert %arg129 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1276 = stablehlo.convolution(%1274, %1275) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1277 = stablehlo.broadcast_in_dim %arg714, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1278 = stablehlo.broadcast_in_dim %arg715, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1279 = stablehlo.convert %1276 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1280 = stablehlo.broadcast_in_dim %1277, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1281 = stablehlo.subtract %1279, %1280 : tensor<256x14x14x1024xf32>
    %1282 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1283 = stablehlo.add %1278, %1282 : tensor<1x1x1x1024xf32>
    %1284 = stablehlo.rsqrt %1283 : tensor<1x1x1x1024xf32>
    %1285 = stablehlo.reshape %arg126 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1286 = stablehlo.multiply %1284, %1285 : tensor<1x1x1x1024xf32>
    %1287 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1288 = stablehlo.multiply %1281, %1287 : tensor<256x14x14x1024xf32>
    %1289 = stablehlo.reshape %arg125 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1290 = stablehlo.broadcast_in_dim %1289, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1291 = stablehlo.add %1288, %1290 : tensor<256x14x14x1024xf32>
    %1292 = stablehlo.convert %1291 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1293 = stablehlo.add %1236, %1292 : tensor<256x14x14x1024xbf16>
    %1294 = call @relu_133(%1293) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1295 = stablehlo.convert %arg136 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1296 = stablehlo.convolution(%1294, %1295) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1297 = stablehlo.broadcast_in_dim %arg716, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1298 = stablehlo.broadcast_in_dim %arg717, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1299 = stablehlo.convert %1296 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1300 = stablehlo.broadcast_in_dim %1297, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1301 = stablehlo.subtract %1299, %1300 : tensor<256x14x14x256xf32>
    %1302 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1303 = stablehlo.add %1298, %1302 : tensor<1x1x1x256xf32>
    %1304 = stablehlo.rsqrt %1303 : tensor<1x1x1x256xf32>
    %1305 = stablehlo.reshape %arg131 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1306 = stablehlo.multiply %1304, %1305 : tensor<1x1x1x256xf32>
    %1307 = stablehlo.broadcast_in_dim %1306, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1308 = stablehlo.multiply %1301, %1307 : tensor<256x14x14x256xf32>
    %1309 = stablehlo.reshape %arg130 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1310 = stablehlo.broadcast_in_dim %1309, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1311 = stablehlo.add %1308, %1310 : tensor<256x14x14x256xf32>
    %1312 = stablehlo.convert %1311 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1313 = call @relu_115(%1312) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1314 = stablehlo.convert %arg137 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1315 = stablehlo.convolution(%1313, %1314) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1316 = stablehlo.broadcast_in_dim %arg718, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1317 = stablehlo.broadcast_in_dim %arg719, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1318 = stablehlo.convert %1315 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1319 = stablehlo.broadcast_in_dim %1316, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1320 = stablehlo.subtract %1318, %1319 : tensor<256x14x14x256xf32>
    %1321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1322 = stablehlo.add %1317, %1321 : tensor<1x1x1x256xf32>
    %1323 = stablehlo.rsqrt %1322 : tensor<1x1x1x256xf32>
    %1324 = stablehlo.reshape %arg133 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1325 = stablehlo.multiply %1323, %1324 : tensor<1x1x1x256xf32>
    %1326 = stablehlo.broadcast_in_dim %1325, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1327 = stablehlo.multiply %1320, %1326 : tensor<256x14x14x256xf32>
    %1328 = stablehlo.reshape %arg132 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1329 = stablehlo.broadcast_in_dim %1328, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1330 = stablehlo.add %1327, %1329 : tensor<256x14x14x256xf32>
    %1331 = stablehlo.convert %1330 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1332 = call @relu_115(%1331) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1333 = stablehlo.convert %arg138 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1334 = stablehlo.convolution(%1332, %1333) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1335 = stablehlo.broadcast_in_dim %arg720, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1336 = stablehlo.broadcast_in_dim %arg721, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1337 = stablehlo.convert %1334 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1338 = stablehlo.broadcast_in_dim %1335, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1339 = stablehlo.subtract %1337, %1338 : tensor<256x14x14x1024xf32>
    %1340 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1341 = stablehlo.add %1336, %1340 : tensor<1x1x1x1024xf32>
    %1342 = stablehlo.rsqrt %1341 : tensor<1x1x1x1024xf32>
    %1343 = stablehlo.reshape %arg135 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1344 = stablehlo.multiply %1342, %1343 : tensor<1x1x1x1024xf32>
    %1345 = stablehlo.broadcast_in_dim %1344, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1346 = stablehlo.multiply %1339, %1345 : tensor<256x14x14x1024xf32>
    %1347 = stablehlo.reshape %arg134 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1348 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1349 = stablehlo.add %1346, %1348 : tensor<256x14x14x1024xf32>
    %1350 = stablehlo.convert %1349 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1351 = stablehlo.add %1294, %1350 : tensor<256x14x14x1024xbf16>
    %1352 = call @relu_133(%1351) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1353 = stablehlo.convert %arg145 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1354 = stablehlo.convolution(%1352, %1353) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1355 = stablehlo.broadcast_in_dim %arg722, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1356 = stablehlo.broadcast_in_dim %arg723, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1357 = stablehlo.convert %1354 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1358 = stablehlo.broadcast_in_dim %1355, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1359 = stablehlo.subtract %1357, %1358 : tensor<256x14x14x256xf32>
    %1360 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1361 = stablehlo.add %1356, %1360 : tensor<1x1x1x256xf32>
    %1362 = stablehlo.rsqrt %1361 : tensor<1x1x1x256xf32>
    %1363 = stablehlo.reshape %arg140 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1364 = stablehlo.multiply %1362, %1363 : tensor<1x1x1x256xf32>
    %1365 = stablehlo.broadcast_in_dim %1364, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1366 = stablehlo.multiply %1359, %1365 : tensor<256x14x14x256xf32>
    %1367 = stablehlo.reshape %arg139 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1368 = stablehlo.broadcast_in_dim %1367, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1369 = stablehlo.add %1366, %1368 : tensor<256x14x14x256xf32>
    %1370 = stablehlo.convert %1369 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1371 = call @relu_115(%1370) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1372 = stablehlo.convert %arg146 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1373 = stablehlo.convolution(%1371, %1372) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1374 = stablehlo.broadcast_in_dim %arg724, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1375 = stablehlo.broadcast_in_dim %arg725, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1376 = stablehlo.convert %1373 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1377 = stablehlo.broadcast_in_dim %1374, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1378 = stablehlo.subtract %1376, %1377 : tensor<256x14x14x256xf32>
    %1379 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1380 = stablehlo.add %1375, %1379 : tensor<1x1x1x256xf32>
    %1381 = stablehlo.rsqrt %1380 : tensor<1x1x1x256xf32>
    %1382 = stablehlo.reshape %arg142 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1383 = stablehlo.multiply %1381, %1382 : tensor<1x1x1x256xf32>
    %1384 = stablehlo.broadcast_in_dim %1383, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1385 = stablehlo.multiply %1378, %1384 : tensor<256x14x14x256xf32>
    %1386 = stablehlo.reshape %arg141 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1387 = stablehlo.broadcast_in_dim %1386, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1388 = stablehlo.add %1385, %1387 : tensor<256x14x14x256xf32>
    %1389 = stablehlo.convert %1388 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1390 = call @relu_115(%1389) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1391 = stablehlo.convert %arg147 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1392 = stablehlo.convolution(%1390, %1391) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1393 = stablehlo.broadcast_in_dim %arg726, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1394 = stablehlo.broadcast_in_dim %arg727, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1395 = stablehlo.convert %1392 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1396 = stablehlo.broadcast_in_dim %1393, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1397 = stablehlo.subtract %1395, %1396 : tensor<256x14x14x1024xf32>
    %1398 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1399 = stablehlo.add %1394, %1398 : tensor<1x1x1x1024xf32>
    %1400 = stablehlo.rsqrt %1399 : tensor<1x1x1x1024xf32>
    %1401 = stablehlo.reshape %arg144 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1402 = stablehlo.multiply %1400, %1401 : tensor<1x1x1x1024xf32>
    %1403 = stablehlo.broadcast_in_dim %1402, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1404 = stablehlo.multiply %1397, %1403 : tensor<256x14x14x1024xf32>
    %1405 = stablehlo.reshape %arg143 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1406 = stablehlo.broadcast_in_dim %1405, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1407 = stablehlo.add %1404, %1406 : tensor<256x14x14x1024xf32>
    %1408 = stablehlo.convert %1407 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1409 = stablehlo.add %1352, %1408 : tensor<256x14x14x1024xbf16>
    %1410 = call @relu_133(%1409) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1411 = stablehlo.convert %arg154 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1412 = stablehlo.convolution(%1410, %1411) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1413 = stablehlo.broadcast_in_dim %arg728, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1414 = stablehlo.broadcast_in_dim %arg729, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1415 = stablehlo.convert %1412 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1416 = stablehlo.broadcast_in_dim %1413, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1417 = stablehlo.subtract %1415, %1416 : tensor<256x14x14x256xf32>
    %1418 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1419 = stablehlo.add %1414, %1418 : tensor<1x1x1x256xf32>
    %1420 = stablehlo.rsqrt %1419 : tensor<1x1x1x256xf32>
    %1421 = stablehlo.reshape %arg149 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1422 = stablehlo.multiply %1420, %1421 : tensor<1x1x1x256xf32>
    %1423 = stablehlo.broadcast_in_dim %1422, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1424 = stablehlo.multiply %1417, %1423 : tensor<256x14x14x256xf32>
    %1425 = stablehlo.reshape %arg148 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1426 = stablehlo.broadcast_in_dim %1425, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1427 = stablehlo.add %1424, %1426 : tensor<256x14x14x256xf32>
    %1428 = stablehlo.convert %1427 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1429 = call @relu_115(%1428) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1430 = stablehlo.convert %arg155 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1431 = stablehlo.convolution(%1429, %1430) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1432 = stablehlo.broadcast_in_dim %arg730, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1433 = stablehlo.broadcast_in_dim %arg731, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1434 = stablehlo.convert %1431 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1435 = stablehlo.broadcast_in_dim %1432, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1436 = stablehlo.subtract %1434, %1435 : tensor<256x14x14x256xf32>
    %1437 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1438 = stablehlo.add %1433, %1437 : tensor<1x1x1x256xf32>
    %1439 = stablehlo.rsqrt %1438 : tensor<1x1x1x256xf32>
    %1440 = stablehlo.reshape %arg151 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1441 = stablehlo.multiply %1439, %1440 : tensor<1x1x1x256xf32>
    %1442 = stablehlo.broadcast_in_dim %1441, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1443 = stablehlo.multiply %1436, %1442 : tensor<256x14x14x256xf32>
    %1444 = stablehlo.reshape %arg150 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1445 = stablehlo.broadcast_in_dim %1444, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1446 = stablehlo.add %1443, %1445 : tensor<256x14x14x256xf32>
    %1447 = stablehlo.convert %1446 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1448 = call @relu_115(%1447) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1449 = stablehlo.convert %arg156 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1450 = stablehlo.convolution(%1448, %1449) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1451 = stablehlo.broadcast_in_dim %arg732, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1452 = stablehlo.broadcast_in_dim %arg733, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1453 = stablehlo.convert %1450 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1454 = stablehlo.broadcast_in_dim %1451, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1455 = stablehlo.subtract %1453, %1454 : tensor<256x14x14x1024xf32>
    %1456 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1457 = stablehlo.add %1452, %1456 : tensor<1x1x1x1024xf32>
    %1458 = stablehlo.rsqrt %1457 : tensor<1x1x1x1024xf32>
    %1459 = stablehlo.reshape %arg153 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1460 = stablehlo.multiply %1458, %1459 : tensor<1x1x1x1024xf32>
    %1461 = stablehlo.broadcast_in_dim %1460, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1462 = stablehlo.multiply %1455, %1461 : tensor<256x14x14x1024xf32>
    %1463 = stablehlo.reshape %arg152 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1464 = stablehlo.broadcast_in_dim %1463, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1465 = stablehlo.add %1462, %1464 : tensor<256x14x14x1024xf32>
    %1466 = stablehlo.convert %1465 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1467 = stablehlo.add %1410, %1466 : tensor<256x14x14x1024xbf16>
    %1468 = call @relu_133(%1467) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1469 = stablehlo.convert %arg163 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1470 = stablehlo.convolution(%1468, %1469) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1471 = stablehlo.broadcast_in_dim %arg734, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1472 = stablehlo.broadcast_in_dim %arg735, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1473 = stablehlo.convert %1470 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1474 = stablehlo.broadcast_in_dim %1471, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1475 = stablehlo.subtract %1473, %1474 : tensor<256x14x14x256xf32>
    %1476 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1477 = stablehlo.add %1472, %1476 : tensor<1x1x1x256xf32>
    %1478 = stablehlo.rsqrt %1477 : tensor<1x1x1x256xf32>
    %1479 = stablehlo.reshape %arg158 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1480 = stablehlo.multiply %1478, %1479 : tensor<1x1x1x256xf32>
    %1481 = stablehlo.broadcast_in_dim %1480, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1482 = stablehlo.multiply %1475, %1481 : tensor<256x14x14x256xf32>
    %1483 = stablehlo.reshape %arg157 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1484 = stablehlo.broadcast_in_dim %1483, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1485 = stablehlo.add %1482, %1484 : tensor<256x14x14x256xf32>
    %1486 = stablehlo.convert %1485 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1487 = call @relu_115(%1486) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1488 = stablehlo.convert %arg164 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1489 = stablehlo.convolution(%1487, %1488) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1490 = stablehlo.broadcast_in_dim %arg736, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1491 = stablehlo.broadcast_in_dim %arg737, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1492 = stablehlo.convert %1489 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1493 = stablehlo.broadcast_in_dim %1490, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1494 = stablehlo.subtract %1492, %1493 : tensor<256x14x14x256xf32>
    %1495 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1496 = stablehlo.add %1491, %1495 : tensor<1x1x1x256xf32>
    %1497 = stablehlo.rsqrt %1496 : tensor<1x1x1x256xf32>
    %1498 = stablehlo.reshape %arg160 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1499 = stablehlo.multiply %1497, %1498 : tensor<1x1x1x256xf32>
    %1500 = stablehlo.broadcast_in_dim %1499, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1501 = stablehlo.multiply %1494, %1500 : tensor<256x14x14x256xf32>
    %1502 = stablehlo.reshape %arg159 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1503 = stablehlo.broadcast_in_dim %1502, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1504 = stablehlo.add %1501, %1503 : tensor<256x14x14x256xf32>
    %1505 = stablehlo.convert %1504 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1506 = call @relu_115(%1505) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1507 = stablehlo.convert %arg165 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1508 = stablehlo.convolution(%1506, %1507) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1509 = stablehlo.broadcast_in_dim %arg738, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1510 = stablehlo.broadcast_in_dim %arg739, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1511 = stablehlo.convert %1508 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1512 = stablehlo.broadcast_in_dim %1509, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1513 = stablehlo.subtract %1511, %1512 : tensor<256x14x14x1024xf32>
    %1514 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1515 = stablehlo.add %1510, %1514 : tensor<1x1x1x1024xf32>
    %1516 = stablehlo.rsqrt %1515 : tensor<1x1x1x1024xf32>
    %1517 = stablehlo.reshape %arg162 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1518 = stablehlo.multiply %1516, %1517 : tensor<1x1x1x1024xf32>
    %1519 = stablehlo.broadcast_in_dim %1518, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1520 = stablehlo.multiply %1513, %1519 : tensor<256x14x14x1024xf32>
    %1521 = stablehlo.reshape %arg161 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1522 = stablehlo.broadcast_in_dim %1521, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1523 = stablehlo.add %1520, %1522 : tensor<256x14x14x1024xf32>
    %1524 = stablehlo.convert %1523 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1525 = stablehlo.add %1468, %1524 : tensor<256x14x14x1024xbf16>
    %1526 = call @relu_133(%1525) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1527 = stablehlo.convert %arg172 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1528 = stablehlo.convolution(%1526, %1527) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1529 = stablehlo.broadcast_in_dim %arg740, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1530 = stablehlo.broadcast_in_dim %arg741, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1531 = stablehlo.convert %1528 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1532 = stablehlo.broadcast_in_dim %1529, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1533 = stablehlo.subtract %1531, %1532 : tensor<256x14x14x256xf32>
    %1534 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1535 = stablehlo.add %1530, %1534 : tensor<1x1x1x256xf32>
    %1536 = stablehlo.rsqrt %1535 : tensor<1x1x1x256xf32>
    %1537 = stablehlo.reshape %arg167 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1538 = stablehlo.multiply %1536, %1537 : tensor<1x1x1x256xf32>
    %1539 = stablehlo.broadcast_in_dim %1538, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1540 = stablehlo.multiply %1533, %1539 : tensor<256x14x14x256xf32>
    %1541 = stablehlo.reshape %arg166 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1542 = stablehlo.broadcast_in_dim %1541, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1543 = stablehlo.add %1540, %1542 : tensor<256x14x14x256xf32>
    %1544 = stablehlo.convert %1543 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1545 = call @relu_115(%1544) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1546 = stablehlo.convert %arg173 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1547 = stablehlo.convolution(%1545, %1546) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1548 = stablehlo.broadcast_in_dim %arg742, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1549 = stablehlo.broadcast_in_dim %arg743, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1550 = stablehlo.convert %1547 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1551 = stablehlo.broadcast_in_dim %1548, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1552 = stablehlo.subtract %1550, %1551 : tensor<256x14x14x256xf32>
    %1553 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1554 = stablehlo.add %1549, %1553 : tensor<1x1x1x256xf32>
    %1555 = stablehlo.rsqrt %1554 : tensor<1x1x1x256xf32>
    %1556 = stablehlo.reshape %arg169 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1557 = stablehlo.multiply %1555, %1556 : tensor<1x1x1x256xf32>
    %1558 = stablehlo.broadcast_in_dim %1557, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1559 = stablehlo.multiply %1552, %1558 : tensor<256x14x14x256xf32>
    %1560 = stablehlo.reshape %arg168 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1561 = stablehlo.broadcast_in_dim %1560, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1562 = stablehlo.add %1559, %1561 : tensor<256x14x14x256xf32>
    %1563 = stablehlo.convert %1562 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1564 = call @relu_115(%1563) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1565 = stablehlo.convert %arg174 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1566 = stablehlo.convolution(%1564, %1565) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1567 = stablehlo.broadcast_in_dim %arg744, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1568 = stablehlo.broadcast_in_dim %arg745, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1569 = stablehlo.convert %1566 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1570 = stablehlo.broadcast_in_dim %1567, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1571 = stablehlo.subtract %1569, %1570 : tensor<256x14x14x1024xf32>
    %1572 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1573 = stablehlo.add %1568, %1572 : tensor<1x1x1x1024xf32>
    %1574 = stablehlo.rsqrt %1573 : tensor<1x1x1x1024xf32>
    %1575 = stablehlo.reshape %arg171 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1576 = stablehlo.multiply %1574, %1575 : tensor<1x1x1x1024xf32>
    %1577 = stablehlo.broadcast_in_dim %1576, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1578 = stablehlo.multiply %1571, %1577 : tensor<256x14x14x1024xf32>
    %1579 = stablehlo.reshape %arg170 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1580 = stablehlo.broadcast_in_dim %1579, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1581 = stablehlo.add %1578, %1580 : tensor<256x14x14x1024xf32>
    %1582 = stablehlo.convert %1581 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1583 = stablehlo.add %1526, %1582 : tensor<256x14x14x1024xbf16>
    %1584 = call @relu_133(%1583) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1585 = stablehlo.convert %arg181 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1586 = stablehlo.convolution(%1584, %1585) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1587 = stablehlo.broadcast_in_dim %arg746, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1588 = stablehlo.broadcast_in_dim %arg747, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1589 = stablehlo.convert %1586 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1590 = stablehlo.broadcast_in_dim %1587, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1591 = stablehlo.subtract %1589, %1590 : tensor<256x14x14x256xf32>
    %1592 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1593 = stablehlo.add %1588, %1592 : tensor<1x1x1x256xf32>
    %1594 = stablehlo.rsqrt %1593 : tensor<1x1x1x256xf32>
    %1595 = stablehlo.reshape %arg176 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1596 = stablehlo.multiply %1594, %1595 : tensor<1x1x1x256xf32>
    %1597 = stablehlo.broadcast_in_dim %1596, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1598 = stablehlo.multiply %1591, %1597 : tensor<256x14x14x256xf32>
    %1599 = stablehlo.reshape %arg175 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1600 = stablehlo.broadcast_in_dim %1599, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1601 = stablehlo.add %1598, %1600 : tensor<256x14x14x256xf32>
    %1602 = stablehlo.convert %1601 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1603 = call @relu_115(%1602) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1604 = stablehlo.convert %arg182 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1605 = stablehlo.convolution(%1603, %1604) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1606 = stablehlo.broadcast_in_dim %arg748, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1607 = stablehlo.broadcast_in_dim %arg749, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1608 = stablehlo.convert %1605 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1609 = stablehlo.broadcast_in_dim %1606, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1610 = stablehlo.subtract %1608, %1609 : tensor<256x14x14x256xf32>
    %1611 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1612 = stablehlo.add %1607, %1611 : tensor<1x1x1x256xf32>
    %1613 = stablehlo.rsqrt %1612 : tensor<1x1x1x256xf32>
    %1614 = stablehlo.reshape %arg178 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1615 = stablehlo.multiply %1613, %1614 : tensor<1x1x1x256xf32>
    %1616 = stablehlo.broadcast_in_dim %1615, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1617 = stablehlo.multiply %1610, %1616 : tensor<256x14x14x256xf32>
    %1618 = stablehlo.reshape %arg177 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1619 = stablehlo.broadcast_in_dim %1618, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1620 = stablehlo.add %1617, %1619 : tensor<256x14x14x256xf32>
    %1621 = stablehlo.convert %1620 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1622 = call @relu_115(%1621) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1623 = stablehlo.convert %arg183 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1624 = stablehlo.convolution(%1622, %1623) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1625 = stablehlo.broadcast_in_dim %arg750, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1626 = stablehlo.broadcast_in_dim %arg751, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1627 = stablehlo.convert %1624 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1628 = stablehlo.broadcast_in_dim %1625, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1629 = stablehlo.subtract %1627, %1628 : tensor<256x14x14x1024xf32>
    %1630 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1631 = stablehlo.add %1626, %1630 : tensor<1x1x1x1024xf32>
    %1632 = stablehlo.rsqrt %1631 : tensor<1x1x1x1024xf32>
    %1633 = stablehlo.reshape %arg180 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1634 = stablehlo.multiply %1632, %1633 : tensor<1x1x1x1024xf32>
    %1635 = stablehlo.broadcast_in_dim %1634, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1636 = stablehlo.multiply %1629, %1635 : tensor<256x14x14x1024xf32>
    %1637 = stablehlo.reshape %arg179 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1638 = stablehlo.broadcast_in_dim %1637, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1639 = stablehlo.add %1636, %1638 : tensor<256x14x14x1024xf32>
    %1640 = stablehlo.convert %1639 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1641 = stablehlo.add %1584, %1640 : tensor<256x14x14x1024xbf16>
    %1642 = call @relu_133(%1641) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1643 = stablehlo.convert %arg190 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1644 = stablehlo.convolution(%1642, %1643) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1645 = stablehlo.broadcast_in_dim %arg752, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1646 = stablehlo.broadcast_in_dim %arg753, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1647 = stablehlo.convert %1644 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1648 = stablehlo.broadcast_in_dim %1645, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1649 = stablehlo.subtract %1647, %1648 : tensor<256x14x14x256xf32>
    %1650 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1651 = stablehlo.add %1646, %1650 : tensor<1x1x1x256xf32>
    %1652 = stablehlo.rsqrt %1651 : tensor<1x1x1x256xf32>
    %1653 = stablehlo.reshape %arg185 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1654 = stablehlo.multiply %1652, %1653 : tensor<1x1x1x256xf32>
    %1655 = stablehlo.broadcast_in_dim %1654, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1656 = stablehlo.multiply %1649, %1655 : tensor<256x14x14x256xf32>
    %1657 = stablehlo.reshape %arg184 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1658 = stablehlo.broadcast_in_dim %1657, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1659 = stablehlo.add %1656, %1658 : tensor<256x14x14x256xf32>
    %1660 = stablehlo.convert %1659 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1661 = call @relu_115(%1660) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1662 = stablehlo.convert %arg191 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1663 = stablehlo.convolution(%1661, %1662) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1664 = stablehlo.broadcast_in_dim %arg754, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1665 = stablehlo.broadcast_in_dim %arg755, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1666 = stablehlo.convert %1663 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1667 = stablehlo.broadcast_in_dim %1664, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1668 = stablehlo.subtract %1666, %1667 : tensor<256x14x14x256xf32>
    %1669 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1670 = stablehlo.add %1665, %1669 : tensor<1x1x1x256xf32>
    %1671 = stablehlo.rsqrt %1670 : tensor<1x1x1x256xf32>
    %1672 = stablehlo.reshape %arg187 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1673 = stablehlo.multiply %1671, %1672 : tensor<1x1x1x256xf32>
    %1674 = stablehlo.broadcast_in_dim %1673, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1675 = stablehlo.multiply %1668, %1674 : tensor<256x14x14x256xf32>
    %1676 = stablehlo.reshape %arg186 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1677 = stablehlo.broadcast_in_dim %1676, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1678 = stablehlo.add %1675, %1677 : tensor<256x14x14x256xf32>
    %1679 = stablehlo.convert %1678 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1680 = call @relu_115(%1679) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1681 = stablehlo.convert %arg192 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1682 = stablehlo.convolution(%1680, %1681) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1683 = stablehlo.broadcast_in_dim %arg756, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1684 = stablehlo.broadcast_in_dim %arg757, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1685 = stablehlo.convert %1682 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1686 = stablehlo.broadcast_in_dim %1683, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1687 = stablehlo.subtract %1685, %1686 : tensor<256x14x14x1024xf32>
    %1688 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1689 = stablehlo.add %1684, %1688 : tensor<1x1x1x1024xf32>
    %1690 = stablehlo.rsqrt %1689 : tensor<1x1x1x1024xf32>
    %1691 = stablehlo.reshape %arg189 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1692 = stablehlo.multiply %1690, %1691 : tensor<1x1x1x1024xf32>
    %1693 = stablehlo.broadcast_in_dim %1692, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1694 = stablehlo.multiply %1687, %1693 : tensor<256x14x14x1024xf32>
    %1695 = stablehlo.reshape %arg188 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1697 = stablehlo.add %1694, %1696 : tensor<256x14x14x1024xf32>
    %1698 = stablehlo.convert %1697 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1699 = stablehlo.add %1642, %1698 : tensor<256x14x14x1024xbf16>
    %1700 = call @relu_133(%1699) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1701 = stablehlo.convert %arg199 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1702 = stablehlo.convolution(%1700, %1701) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1703 = stablehlo.broadcast_in_dim %arg758, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1704 = stablehlo.broadcast_in_dim %arg759, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1705 = stablehlo.convert %1702 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1706 = stablehlo.broadcast_in_dim %1703, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1707 = stablehlo.subtract %1705, %1706 : tensor<256x14x14x256xf32>
    %1708 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1709 = stablehlo.add %1704, %1708 : tensor<1x1x1x256xf32>
    %1710 = stablehlo.rsqrt %1709 : tensor<1x1x1x256xf32>
    %1711 = stablehlo.reshape %arg194 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1712 = stablehlo.multiply %1710, %1711 : tensor<1x1x1x256xf32>
    %1713 = stablehlo.broadcast_in_dim %1712, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1714 = stablehlo.multiply %1707, %1713 : tensor<256x14x14x256xf32>
    %1715 = stablehlo.reshape %arg193 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1716 = stablehlo.broadcast_in_dim %1715, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1717 = stablehlo.add %1714, %1716 : tensor<256x14x14x256xf32>
    %1718 = stablehlo.convert %1717 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1719 = call @relu_115(%1718) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1720 = stablehlo.convert %arg200 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1721 = stablehlo.convolution(%1719, %1720) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1722 = stablehlo.broadcast_in_dim %arg760, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1723 = stablehlo.broadcast_in_dim %arg761, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1724 = stablehlo.convert %1721 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1725 = stablehlo.broadcast_in_dim %1722, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1726 = stablehlo.subtract %1724, %1725 : tensor<256x14x14x256xf32>
    %1727 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1728 = stablehlo.add %1723, %1727 : tensor<1x1x1x256xf32>
    %1729 = stablehlo.rsqrt %1728 : tensor<1x1x1x256xf32>
    %1730 = stablehlo.reshape %arg196 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1731 = stablehlo.multiply %1729, %1730 : tensor<1x1x1x256xf32>
    %1732 = stablehlo.broadcast_in_dim %1731, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1733 = stablehlo.multiply %1726, %1732 : tensor<256x14x14x256xf32>
    %1734 = stablehlo.reshape %arg195 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1735 = stablehlo.broadcast_in_dim %1734, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1736 = stablehlo.add %1733, %1735 : tensor<256x14x14x256xf32>
    %1737 = stablehlo.convert %1736 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1738 = call @relu_115(%1737) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1739 = stablehlo.convert %arg201 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1740 = stablehlo.convolution(%1738, %1739) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1741 = stablehlo.broadcast_in_dim %arg762, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1742 = stablehlo.broadcast_in_dim %arg763, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1743 = stablehlo.convert %1740 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1744 = stablehlo.broadcast_in_dim %1741, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1745 = stablehlo.subtract %1743, %1744 : tensor<256x14x14x1024xf32>
    %1746 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1747 = stablehlo.add %1742, %1746 : tensor<1x1x1x1024xf32>
    %1748 = stablehlo.rsqrt %1747 : tensor<1x1x1x1024xf32>
    %1749 = stablehlo.reshape %arg198 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1750 = stablehlo.multiply %1748, %1749 : tensor<1x1x1x1024xf32>
    %1751 = stablehlo.broadcast_in_dim %1750, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1752 = stablehlo.multiply %1745, %1751 : tensor<256x14x14x1024xf32>
    %1753 = stablehlo.reshape %arg197 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1754 = stablehlo.broadcast_in_dim %1753, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1755 = stablehlo.add %1752, %1754 : tensor<256x14x14x1024xf32>
    %1756 = stablehlo.convert %1755 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1757 = stablehlo.add %1700, %1756 : tensor<256x14x14x1024xbf16>
    %1758 = call @relu_133(%1757) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1759 = stablehlo.convert %arg208 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1760 = stablehlo.convolution(%1758, %1759) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1761 = stablehlo.broadcast_in_dim %arg764, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1762 = stablehlo.broadcast_in_dim %arg765, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1763 = stablehlo.convert %1760 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1764 = stablehlo.broadcast_in_dim %1761, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1765 = stablehlo.subtract %1763, %1764 : tensor<256x14x14x256xf32>
    %1766 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1767 = stablehlo.add %1762, %1766 : tensor<1x1x1x256xf32>
    %1768 = stablehlo.rsqrt %1767 : tensor<1x1x1x256xf32>
    %1769 = stablehlo.reshape %arg203 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1770 = stablehlo.multiply %1768, %1769 : tensor<1x1x1x256xf32>
    %1771 = stablehlo.broadcast_in_dim %1770, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1772 = stablehlo.multiply %1765, %1771 : tensor<256x14x14x256xf32>
    %1773 = stablehlo.reshape %arg202 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1774 = stablehlo.broadcast_in_dim %1773, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1775 = stablehlo.add %1772, %1774 : tensor<256x14x14x256xf32>
    %1776 = stablehlo.convert %1775 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1777 = call @relu_115(%1776) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1778 = stablehlo.convert %arg209 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1779 = stablehlo.convolution(%1777, %1778) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1780 = stablehlo.broadcast_in_dim %arg766, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1781 = stablehlo.broadcast_in_dim %arg767, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1782 = stablehlo.convert %1779 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1783 = stablehlo.broadcast_in_dim %1780, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1784 = stablehlo.subtract %1782, %1783 : tensor<256x14x14x256xf32>
    %1785 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1786 = stablehlo.add %1781, %1785 : tensor<1x1x1x256xf32>
    %1787 = stablehlo.rsqrt %1786 : tensor<1x1x1x256xf32>
    %1788 = stablehlo.reshape %arg205 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1789 = stablehlo.multiply %1787, %1788 : tensor<1x1x1x256xf32>
    %1790 = stablehlo.broadcast_in_dim %1789, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1791 = stablehlo.multiply %1784, %1790 : tensor<256x14x14x256xf32>
    %1792 = stablehlo.reshape %arg204 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1793 = stablehlo.broadcast_in_dim %1792, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1794 = stablehlo.add %1791, %1793 : tensor<256x14x14x256xf32>
    %1795 = stablehlo.convert %1794 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1796 = call @relu_115(%1795) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1797 = stablehlo.convert %arg210 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1798 = stablehlo.convolution(%1796, %1797) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1799 = stablehlo.broadcast_in_dim %arg768, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1800 = stablehlo.broadcast_in_dim %arg769, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1801 = stablehlo.convert %1798 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1802 = stablehlo.broadcast_in_dim %1799, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1803 = stablehlo.subtract %1801, %1802 : tensor<256x14x14x1024xf32>
    %1804 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1805 = stablehlo.add %1800, %1804 : tensor<1x1x1x1024xf32>
    %1806 = stablehlo.rsqrt %1805 : tensor<1x1x1x1024xf32>
    %1807 = stablehlo.reshape %arg207 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1808 = stablehlo.multiply %1806, %1807 : tensor<1x1x1x1024xf32>
    %1809 = stablehlo.broadcast_in_dim %1808, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1810 = stablehlo.multiply %1803, %1809 : tensor<256x14x14x1024xf32>
    %1811 = stablehlo.reshape %arg206 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1812 = stablehlo.broadcast_in_dim %1811, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1813 = stablehlo.add %1810, %1812 : tensor<256x14x14x1024xf32>
    %1814 = stablehlo.convert %1813 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1815 = stablehlo.add %1758, %1814 : tensor<256x14x14x1024xbf16>
    %1816 = call @relu_133(%1815) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1817 = stablehlo.convert %arg229 : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xbf16>
    %1818 = stablehlo.convolution(%1816, %1817) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x512xbf16>) -> tensor<256x14x14x512xbf16>
    %1819 = stablehlo.broadcast_in_dim %arg778, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1820 = stablehlo.broadcast_in_dim %arg779, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1821 = stablehlo.convert %1818 : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xf32>
    %1822 = stablehlo.broadcast_in_dim %1819, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
    %1823 = stablehlo.subtract %1821, %1822 : tensor<256x14x14x512xf32>
    %1824 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1825 = stablehlo.add %1820, %1824 : tensor<1x1x1x512xf32>
    %1826 = stablehlo.rsqrt %1825 : tensor<1x1x1x512xf32>
    %1827 = stablehlo.reshape %arg224 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1828 = stablehlo.multiply %1826, %1827 : tensor<1x1x1x512xf32>
    %1829 = stablehlo.broadcast_in_dim %1828, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
    %1830 = stablehlo.multiply %1823, %1829 : tensor<256x14x14x512xf32>
    %1831 = stablehlo.reshape %arg223 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1832 = stablehlo.broadcast_in_dim %1831, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
    %1833 = stablehlo.add %1830, %1832 : tensor<256x14x14x512xf32>
    %1834 = stablehlo.convert %1833 : (tensor<256x14x14x512xf32>) -> tensor<256x14x14x512xbf16>
    %1835 = call @relu_212(%1834) : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xbf16>
    %1836 = stablehlo.convert %arg230 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %1837 = stablehlo.convolution(%1835, %1836) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1838 = stablehlo.broadcast_in_dim %arg780, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1839 = stablehlo.broadcast_in_dim %arg781, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1840 = stablehlo.convert %1837 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1841 = stablehlo.broadcast_in_dim %1838, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1842 = stablehlo.subtract %1840, %1841 : tensor<256x7x7x512xf32>
    %1843 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1844 = stablehlo.add %1839, %1843 : tensor<1x1x1x512xf32>
    %1845 = stablehlo.rsqrt %1844 : tensor<1x1x1x512xf32>
    %1846 = stablehlo.reshape %arg226 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1847 = stablehlo.multiply %1845, %1846 : tensor<1x1x1x512xf32>
    %1848 = stablehlo.broadcast_in_dim %1847, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1849 = stablehlo.multiply %1842, %1848 : tensor<256x7x7x512xf32>
    %1850 = stablehlo.reshape %arg225 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1851 = stablehlo.broadcast_in_dim %1850, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1852 = stablehlo.add %1849, %1851 : tensor<256x7x7x512xf32>
    %1853 = stablehlo.convert %1852 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1854 = call @relu_222(%1853) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1855 = stablehlo.convert %arg231 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xbf16>
    %1856 = stablehlo.convolution(%1854, %1855) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<1x1x512x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %1857 = stablehlo.broadcast_in_dim %arg782, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1858 = stablehlo.broadcast_in_dim %arg783, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1859 = stablehlo.convert %1856 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %1860 = stablehlo.broadcast_in_dim %1857, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1861 = stablehlo.subtract %1859, %1860 : tensor<256x7x7x2048xf32>
    %1862 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %1863 = stablehlo.add %1858, %1862 : tensor<1x1x1x2048xf32>
    %1864 = stablehlo.rsqrt %1863 : tensor<1x1x1x2048xf32>
    %1865 = stablehlo.reshape %arg228 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1866 = stablehlo.multiply %1864, %1865 : tensor<1x1x1x2048xf32>
    %1867 = stablehlo.broadcast_in_dim %1866, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1868 = stablehlo.multiply %1861, %1867 : tensor<256x7x7x2048xf32>
    %1869 = stablehlo.reshape %arg227 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1870 = stablehlo.broadcast_in_dim %1869, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1871 = stablehlo.add %1868, %1870 : tensor<256x7x7x2048xf32>
    %1872 = stablehlo.convert %1871 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    %1873 = stablehlo.convert %arg232 : (tensor<1x1x1024x2048xf32>) -> tensor<1x1x1024x2048xbf16>
    %1874 = stablehlo.convolution(%1816, %1873) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %1875 = stablehlo.broadcast_in_dim %arg784, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1876 = stablehlo.broadcast_in_dim %arg785, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1877 = stablehlo.convert %1874 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %1878 = stablehlo.broadcast_in_dim %1875, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1879 = stablehlo.subtract %1877, %1878 : tensor<256x7x7x2048xf32>
    %1880 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %1881 = stablehlo.add %1876, %1880 : tensor<1x1x1x2048xf32>
    %1882 = stablehlo.rsqrt %1881 : tensor<1x1x1x2048xf32>
    %1883 = stablehlo.reshape %arg234 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1884 = stablehlo.multiply %1882, %1883 : tensor<1x1x1x2048xf32>
    %1885 = stablehlo.broadcast_in_dim %1884, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1886 = stablehlo.multiply %1879, %1885 : tensor<256x7x7x2048xf32>
    %1887 = stablehlo.reshape %arg233 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1888 = stablehlo.broadcast_in_dim %1887, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1889 = stablehlo.add %1886, %1888 : tensor<256x7x7x2048xf32>
    %1890 = stablehlo.convert %1889 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    %1891 = stablehlo.add %1890, %1872 : tensor<256x7x7x2048xbf16>
    %1892 = call @relu_240(%1891) : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %1893 = stablehlo.convert %arg241 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xbf16>
    %1894 = stablehlo.convolution(%1892, %1893) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xbf16>, tensor<1x1x2048x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1895 = stablehlo.broadcast_in_dim %arg786, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1896 = stablehlo.broadcast_in_dim %arg787, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1897 = stablehlo.convert %1894 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1898 = stablehlo.broadcast_in_dim %1895, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1899 = stablehlo.subtract %1897, %1898 : tensor<256x7x7x512xf32>
    %1900 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1901 = stablehlo.add %1896, %1900 : tensor<1x1x1x512xf32>
    %1902 = stablehlo.rsqrt %1901 : tensor<1x1x1x512xf32>
    %1903 = stablehlo.reshape %arg236 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1904 = stablehlo.multiply %1902, %1903 : tensor<1x1x1x512xf32>
    %1905 = stablehlo.broadcast_in_dim %1904, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1906 = stablehlo.multiply %1899, %1905 : tensor<256x7x7x512xf32>
    %1907 = stablehlo.reshape %arg235 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1908 = stablehlo.broadcast_in_dim %1907, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1909 = stablehlo.add %1906, %1908 : tensor<256x7x7x512xf32>
    %1910 = stablehlo.convert %1909 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1911 = call @relu_222(%1910) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1912 = stablehlo.convert %arg242 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %1913 = stablehlo.convolution(%1911, %1912) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1914 = stablehlo.broadcast_in_dim %arg788, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1915 = stablehlo.broadcast_in_dim %arg789, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1916 = stablehlo.convert %1913 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1917 = stablehlo.broadcast_in_dim %1914, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1918 = stablehlo.subtract %1916, %1917 : tensor<256x7x7x512xf32>
    %1919 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1920 = stablehlo.add %1915, %1919 : tensor<1x1x1x512xf32>
    %1921 = stablehlo.rsqrt %1920 : tensor<1x1x1x512xf32>
    %1922 = stablehlo.reshape %arg238 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1923 = stablehlo.multiply %1921, %1922 : tensor<1x1x1x512xf32>
    %1924 = stablehlo.broadcast_in_dim %1923, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1925 = stablehlo.multiply %1918, %1924 : tensor<256x7x7x512xf32>
    %1926 = stablehlo.reshape %arg237 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1927 = stablehlo.broadcast_in_dim %1926, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1928 = stablehlo.add %1925, %1927 : tensor<256x7x7x512xf32>
    %1929 = stablehlo.convert %1928 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1930 = call @relu_222(%1929) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1931 = stablehlo.convert %arg243 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xbf16>
    %1932 = stablehlo.convolution(%1930, %1931) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<1x1x512x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %1933 = stablehlo.broadcast_in_dim %arg790, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1934 = stablehlo.broadcast_in_dim %arg791, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1935 = stablehlo.convert %1932 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %1936 = stablehlo.broadcast_in_dim %1933, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1937 = stablehlo.subtract %1935, %1936 : tensor<256x7x7x2048xf32>
    %1938 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %1939 = stablehlo.add %1934, %1938 : tensor<1x1x1x2048xf32>
    %1940 = stablehlo.rsqrt %1939 : tensor<1x1x1x2048xf32>
    %1941 = stablehlo.reshape %arg240 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1942 = stablehlo.multiply %1940, %1941 : tensor<1x1x1x2048xf32>
    %1943 = stablehlo.broadcast_in_dim %1942, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1944 = stablehlo.multiply %1937, %1943 : tensor<256x7x7x2048xf32>
    %1945 = stablehlo.reshape %arg239 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1946 = stablehlo.broadcast_in_dim %1945, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1947 = stablehlo.add %1944, %1946 : tensor<256x7x7x2048xf32>
    %1948 = stablehlo.convert %1947 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    %1949 = stablehlo.add %1892, %1948 : tensor<256x7x7x2048xbf16>
    %1950 = call @relu_240(%1949) : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %1951 = stablehlo.convert %arg250 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xbf16>
    %1952 = stablehlo.convolution(%1950, %1951) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xbf16>, tensor<1x1x2048x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1953 = stablehlo.broadcast_in_dim %arg792, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1954 = stablehlo.broadcast_in_dim %arg793, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1955 = stablehlo.convert %1952 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1956 = stablehlo.broadcast_in_dim %1953, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1957 = stablehlo.subtract %1955, %1956 : tensor<256x7x7x512xf32>
    %1958 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1959 = stablehlo.add %1954, %1958 : tensor<1x1x1x512xf32>
    %1960 = stablehlo.rsqrt %1959 : tensor<1x1x1x512xf32>
    %1961 = stablehlo.reshape %arg245 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1962 = stablehlo.multiply %1960, %1961 : tensor<1x1x1x512xf32>
    %1963 = stablehlo.broadcast_in_dim %1962, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1964 = stablehlo.multiply %1957, %1963 : tensor<256x7x7x512xf32>
    %1965 = stablehlo.reshape %arg244 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1966 = stablehlo.broadcast_in_dim %1965, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1967 = stablehlo.add %1964, %1966 : tensor<256x7x7x512xf32>
    %1968 = stablehlo.convert %1967 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1969 = call @relu_222(%1968) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1970 = stablehlo.convert %arg251 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %1971 = stablehlo.convolution(%1969, %1970) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1972 = stablehlo.broadcast_in_dim %arg794, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1973 = stablehlo.broadcast_in_dim %arg795, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1974 = stablehlo.convert %1971 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %1975 = stablehlo.broadcast_in_dim %1972, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1976 = stablehlo.subtract %1974, %1975 : tensor<256x7x7x512xf32>
    %1977 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1978 = stablehlo.add %1973, %1977 : tensor<1x1x1x512xf32>
    %1979 = stablehlo.rsqrt %1978 : tensor<1x1x1x512xf32>
    %1980 = stablehlo.reshape %arg247 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1981 = stablehlo.multiply %1979, %1980 : tensor<1x1x1x512xf32>
    %1982 = stablehlo.broadcast_in_dim %1981, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1983 = stablehlo.multiply %1976, %1982 : tensor<256x7x7x512xf32>
    %1984 = stablehlo.reshape %arg246 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1985 = stablehlo.broadcast_in_dim %1984, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %1986 = stablehlo.add %1983, %1985 : tensor<256x7x7x512xf32>
    %1987 = stablehlo.convert %1986 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %1988 = call @relu_222(%1987) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %1989 = stablehlo.convert %arg252 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xbf16>
    %1990 = stablehlo.convolution(%1988, %1989) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<1x1x512x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %1991 = stablehlo.broadcast_in_dim %arg796, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1992 = stablehlo.broadcast_in_dim %arg797, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %1993 = stablehlo.convert %1990 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %1994 = stablehlo.broadcast_in_dim %1991, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %1995 = stablehlo.subtract %1993, %1994 : tensor<256x7x7x2048xf32>
    %1996 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %1997 = stablehlo.add %1992, %1996 : tensor<1x1x1x2048xf32>
    %1998 = stablehlo.rsqrt %1997 : tensor<1x1x1x2048xf32>
    %1999 = stablehlo.reshape %arg249 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %2000 = stablehlo.multiply %1998, %1999 : tensor<1x1x1x2048xf32>
    %2001 = stablehlo.broadcast_in_dim %2000, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %2002 = stablehlo.multiply %1995, %2001 : tensor<256x7x7x2048xf32>
    %2003 = stablehlo.reshape %arg248 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %2004 = stablehlo.broadcast_in_dim %2003, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %2005 = stablehlo.add %2002, %2004 : tensor<256x7x7x2048xf32>
    %2006 = stablehlo.convert %2005 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    %2007 = stablehlo.add %1950, %2006 : tensor<256x7x7x2048xbf16>
    %2008 = call @relu_240(%2007) : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %2009 = stablehlo.convert %2008 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2010 = stablehlo.reduce(%2009 init: %cst_1) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x2048xf32>, tensor<f32>) -> tensor<256x2048xf32>
    %cst_2 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
    %2011 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
    %2012 = stablehlo.divide %2010, %2011 : tensor<256x2048xf32>
    %2013 = stablehlo.convert %2012 : (tensor<256x2048xf32>) -> tensor<256x2048xbf16>
    %2014 = stablehlo.convert %arg311 : (tensor<2048x1000xf32>) -> tensor<2048x1000xbf16>
    %2015 = stablehlo.convert %arg310 : (tensor<1000xf32>) -> tensor<1000xbf16>
    %2016 = stablehlo.dot_general %2013, %2014, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xbf16>, tensor<2048x1000xbf16>) -> tensor<256x1000xbf16>
    %2017 = stablehlo.reshape %2015 : (tensor<1000xbf16>) -> tensor<1x1000xbf16>
    %2018 = stablehlo.broadcast_in_dim %2017, dims = [0, 1] : (tensor<1x1000xbf16>) -> tensor<256x1000xbf16>
    %2019 = stablehlo.add %2016, %2018 : tensor<256x1000xbf16>
    %2020 = stablehlo.broadcast_in_dim %2019, dims = [1, 2] : (tensor<256x1000xbf16>) -> tensor<1x256x1000xbf16>
    return %2020 : tensor<1x256x1000xbf16>
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
  func.func private @relu_32(%arg0: tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x256xbf16>
    return %1 : tensor<256x56x56x256xbf16>
  }
  func.func private @relu_55(%arg0: tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x56x56x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x128xbf16>
    return %1 : tensor<256x56x56x128xbf16>
  }
  func.func private @relu_65(%arg0: tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xbf16>
    return %1 : tensor<256x28x28x128xbf16>
  }
  func.func private @relu_83(%arg0: tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x512xbf16>
    return %1 : tensor<256x28x28x512xbf16>
  }
  func.func private @relu_105(%arg0: tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x256xbf16>
    return %1 : tensor<256x28x28x256xbf16>
  }
  func.func private @relu_115(%arg0: tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xbf16>
    return %1 : tensor<256x14x14x256xbf16>
  }
  func.func private @relu_133(%arg0: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x1024xbf16>
    return %1 : tensor<256x14x14x1024xbf16>
  }
  func.func private @relu_212(%arg0: tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x512xbf16>
    return %1 : tensor<256x14x14x512xbf16>
  }
  func.func private @relu_222(%arg0: tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xbf16>
    return %1 : tensor<256x7x7x512xbf16>
  }
  func.func private @relu_240(%arg0: tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x2048xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x2048xbf16>
    return %1 : tensor<256x7x7x2048xbf16>
  }
}
