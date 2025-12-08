module @pmap_inference_step attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg1: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg2: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg3: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg4: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg5: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg6: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg7: tensor<1x1x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg8: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg9: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg10: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg11: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg12: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg13: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg14: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg15: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg16: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg17: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg18: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg19: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg20: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg21: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg22: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg23: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg24: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg25: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg26: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg27: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg28: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg29: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg30: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg31: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg32: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg33: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg34: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg35: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg36: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg37: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg38: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg39: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg40: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg41: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg42: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg43: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg44: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg45: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg46: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg47: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg48: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg49: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg50: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg51: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg52: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg53: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg54: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg55: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg56: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg57: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg58: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg59: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg60: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg61: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg62: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg63: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg64: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg65: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg66: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg67: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg68: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg69: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg70: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg71: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg72: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg73: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg74: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg75: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg76: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg77: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg78: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg79: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg80: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg81: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg82: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg83: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg84: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg85: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg86: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg87: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg88: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg89: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg90: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg91: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg92: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg93: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg94: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg95: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg96: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg97: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg98: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg99: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg100: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg101: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg102: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg103: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg104: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg105: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg106: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg107: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg108: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg109: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg110: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg111: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg112: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg113: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg114: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg115: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg116: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg117: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg118: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg119: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg120: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg121: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg122: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg123: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg124: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg125: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg126: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg127: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg128: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg129: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg130: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg131: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg132: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg133: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg134: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg135: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg136: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg137: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg138: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg139: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg140: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg141: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg142: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg143: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg144: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg145: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg146: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg147: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg148: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg149: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg150: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg151: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg152: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg153: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg154: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg155: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg156: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg157: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg158: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg159: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg160: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg161: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg162: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg163: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg164: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg165: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg166: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg167: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg168: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg169: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg170: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg171: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg172: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg173: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg174: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg175: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg176: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg177: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg178: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg179: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg180: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg181: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg182: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg183: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg184: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg185: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg186: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg187: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg188: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg189: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg190: tensor<1x1x512x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg191: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg192: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg193: tensor<1x1x512x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg194: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg195: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg196: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg197: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg198: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg199: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg200: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg201: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg202: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg203: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg204: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg205: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg206: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg207: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg208: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg209: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg210: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg211: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg212: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg213: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg214: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg215: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg216: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg217: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg218: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg219: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg220: tensor<1x1x256x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg221: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg222: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg223: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg224: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg225: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg226: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg227: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg228: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg229: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg230: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg231: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg232: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg233: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg234: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg235: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg236: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg237: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg238: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg239: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg240: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg241: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg242: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg243: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg244: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg245: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg246: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg247: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg248: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg249: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg250: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg251: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg252: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg253: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg254: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg255: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg256: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg257: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg258: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg259: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg260: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg261: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg262: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg263: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg264: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg265: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg266: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg267: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg268: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg269: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg270: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg271: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg272: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg273: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg274: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg275: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg276: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg277: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg278: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg279: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg280: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg281: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg282: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg283: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg284: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg285: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg286: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg287: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg288: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg289: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg290: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg291: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg292: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg293: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg294: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg295: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg296: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg297: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg298: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg299: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg300: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg301: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg302: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg303: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg304: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg305: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg306: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg307: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg308: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg309: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg310: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg311: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg312: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg313: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg314: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg315: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg316: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg317: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg318: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg319: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg320: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg321: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg322: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg323: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg324: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg325: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg326: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg327: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg328: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg329: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg330: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg331: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg332: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg333: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg334: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg335: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg336: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg337: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg338: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg339: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg340: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg341: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg342: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg343: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg344: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg345: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg346: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg347: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg348: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg349: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg350: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg351: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg352: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg353: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg354: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg355: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg356: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg357: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg358: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg359: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg360: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg361: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg362: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg363: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg364: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg365: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg366: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg367: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg368: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg369: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg370: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg371: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg372: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg373: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg374: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg375: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg376: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg377: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg378: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg379: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg380: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg381: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg382: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg383: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg384: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg385: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg386: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg387: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg388: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg389: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg390: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg391: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg392: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg393: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg394: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg395: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg396: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg397: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg398: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg399: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg400: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg401: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg402: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg403: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg404: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg405: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg406: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg407: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg408: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg409: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg410: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg411: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg412: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg413: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg414: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg415: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg416: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg417: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg418: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg419: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg420: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg421: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg422: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg423: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg424: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg425: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg426: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg427: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg428: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg429: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg430: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg431: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg432: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg433: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg434: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg435: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg436: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg437: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg438: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg439: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg440: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg441: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg442: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg443: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg444: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg445: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg446: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg447: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg448: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg449: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg450: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg451: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg452: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg453: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg454: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg455: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg456: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg457: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg458: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg459: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg460: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg461: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg462: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg463: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg464: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg465: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg466: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg467: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg468: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg469: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg470: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg471: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg472: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg473: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg474: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg475: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg476: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg477: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg478: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg479: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg480: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg481: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg482: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg483: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg484: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg485: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg486: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg487: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg488: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg489: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg490: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg491: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg492: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg493: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg494: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg495: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg496: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg497: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg498: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg499: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg500: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg501: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg502: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg503: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg504: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg505: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg506: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg507: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg508: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg509: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg510: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg511: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg512: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg513: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg514: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg515: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg516: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg517: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg518: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg519: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg520: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg521: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg522: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg523: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg524: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg525: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg526: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg527: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg528: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg529: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg530: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg531: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg532: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg533: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg534: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg535: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg536: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg537: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg538: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg539: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg540: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg541: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg542: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg543: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg544: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg545: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg546: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg547: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg548: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg549: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg550: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg551: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg552: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg553: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg554: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg555: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg556: tensor<1x1x1024x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg557: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg558: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg559: tensor<1x1x1024x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg560: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg561: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg562: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg563: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg564: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg565: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg566: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg567: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg568: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg569: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg570: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg571: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg572: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg573: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg574: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg575: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg576: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg577: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg578: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg579: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg580: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg581: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg582: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg583: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg584: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg585: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg586: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg587: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg588: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg589: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg590: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg591: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg592: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg593: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg594: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg595: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg596: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg597: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg598: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg599: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg600: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg601: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg602: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg603: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg604: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg605: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg606: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg607: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg608: tensor<2048x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg609: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg610: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg611: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg612: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg613: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg614: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg615: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg616: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg617: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg618: tensor<1x1x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg619: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg620: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg621: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg622: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg623: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg624: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg625: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg626: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg627: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg628: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg629: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg630: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg631: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg632: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg633: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg634: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg635: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg636: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg637: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg638: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg639: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg640: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg641: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg642: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg643: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg644: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg645: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg646: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg647: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg648: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg649: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg650: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg651: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg652: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg653: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg654: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg655: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg656: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg657: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg658: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg659: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg660: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg661: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg662: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg663: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg664: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg665: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg666: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg667: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg668: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg669: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg670: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg671: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg672: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg673: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg674: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg675: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg676: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg677: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg678: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg679: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg680: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg681: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg682: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg683: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg684: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg685: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg686: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg687: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg688: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg689: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg690: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg691: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg692: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg693: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg694: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg695: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg696: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg697: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg698: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg699: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg700: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg701: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg702: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg703: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg704: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg705: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg706: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg707: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg708: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg709: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg710: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg711: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg712: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg713: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg714: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg715: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg716: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg717: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg718: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg719: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg720: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg721: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg722: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg723: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg724: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg725: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg726: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg727: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg728: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg729: tensor<1x1x256x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg730: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg731: tensor<1x1x64x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg732: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg733: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg734: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg735: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg736: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg737: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg738: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg739: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg740: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg741: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg742: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg743: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg744: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg745: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg746: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg747: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg748: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg749: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg750: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg751: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg752: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg753: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg754: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg755: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg756: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg757: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg758: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg759: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg760: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg761: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg762: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg763: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg764: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg765: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg766: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg767: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg768: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg769: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg770: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg771: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg772: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg773: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg774: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg775: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg776: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg777: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg778: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg779: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg780: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg781: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg782: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg783: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg784: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg785: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg786: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg787: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg788: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg789: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg790: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg791: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg792: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg793: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg794: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg795: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg796: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg797: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg798: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg799: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg800: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg801: tensor<1x1x512x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg802: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg803: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg804: tensor<1x1x512x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg805: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg806: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg807: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg808: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg809: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg810: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg811: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg812: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg813: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg814: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg815: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg816: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg817: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg818: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg819: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg820: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg821: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg822: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg823: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg824: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg825: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg826: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg827: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg828: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg829: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg830: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg831: tensor<1x1x256x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg832: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg833: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg834: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg835: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg836: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg837: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg838: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg839: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg840: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg841: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg842: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg843: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg844: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg845: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg846: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg847: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg848: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg849: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg850: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg851: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg852: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg853: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg854: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg855: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg856: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg857: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg858: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg859: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg860: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg861: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg862: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg863: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg864: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg865: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg866: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg867: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg868: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg869: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg870: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg871: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg872: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg873: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg874: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg875: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg876: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg877: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg878: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg879: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg880: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg881: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg882: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg883: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg884: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg885: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg886: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg887: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg888: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg889: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg890: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg891: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg892: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg893: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg894: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg895: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg896: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg897: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg898: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg899: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg900: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg901: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg902: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg903: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg904: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg905: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg906: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg907: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg908: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg909: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg910: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg911: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg912: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg913: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg914: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg915: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg916: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg917: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg918: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg919: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg920: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg921: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg922: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg923: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg924: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg925: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg926: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg927: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg928: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg929: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg930: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg931: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg932: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg933: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg934: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg935: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg936: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg937: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg938: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg939: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg940: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg941: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg942: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg943: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg944: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg945: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg946: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg947: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg948: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg949: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg950: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg951: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg952: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg953: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg954: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg955: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg956: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg957: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg958: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg959: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg960: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg961: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg962: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg963: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg964: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg965: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg966: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg967: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg968: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg969: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg970: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg971: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg972: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg973: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg974: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg975: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg976: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg977: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg978: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg979: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg980: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg981: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg982: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg983: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg984: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg985: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg986: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg987: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg988: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg989: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg990: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg991: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg992: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg993: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg994: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg995: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg996: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg997: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg998: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg999: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1000: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1001: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1002: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1003: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1004: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1005: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1006: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1007: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1008: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1009: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1010: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1011: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1012: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1013: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1014: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1015: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1016: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1017: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1018: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1019: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1020: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1021: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1022: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1023: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1024: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1025: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1026: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1027: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1028: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1029: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1030: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1031: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1032: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1033: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1034: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1035: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1036: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1037: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1038: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1039: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1040: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1041: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1042: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1043: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1044: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1045: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1046: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1047: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1048: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1049: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1050: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1051: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1052: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1053: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1054: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1055: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1056: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1057: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1058: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1059: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1060: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1061: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1062: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1063: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1064: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1065: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1066: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1067: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1068: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1069: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1070: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1071: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1072: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1073: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1074: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1075: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1076: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1077: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1078: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1079: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1080: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1081: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1082: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1083: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1084: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1085: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1086: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1087: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1088: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1089: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1090: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1091: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1092: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1093: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1094: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1095: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1096: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1097: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1098: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1099: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1100: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1101: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1102: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1103: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1104: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1105: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1106: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1107: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1108: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1109: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1110: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1111: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1112: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1113: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1114: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1115: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1116: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1117: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1118: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1119: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1120: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1121: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1122: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1123: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1124: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1125: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1126: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1127: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1128: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1129: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1130: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1131: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1132: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1133: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1134: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1135: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1136: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1137: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1138: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1139: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1140: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1141: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1142: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1143: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1144: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1145: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1146: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1147: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1148: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1149: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1150: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1151: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1152: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1153: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1154: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1155: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1156: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1157: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1158: tensor<1x1x1024x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1159: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1160: tensor<1x1x256x1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1161: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1162: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1163: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1164: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1165: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1166: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1167: tensor<1x1x1024x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1168: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1169: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1170: tensor<1x1x1024x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1171: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1172: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1173: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1174: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1175: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1176: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1177: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1178: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1179: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1180: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1181: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1182: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1183: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1184: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1185: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1186: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1187: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1188: tensor<1x1x2048x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1189: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1190: tensor<1x1x512x2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1191: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1192: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1193: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1194: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1195: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1196: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1197: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1198: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1199: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1200: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1201: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1202: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1203: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1204: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1205: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1206: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1207: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1208: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1209: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1210: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1211: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1212: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1213: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1214: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1215: tensor<1x1x512x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1216: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1217: tensor<1x1x128x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1218: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg1219: tensor<2048x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg1220: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1221: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1222: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1223: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg1224: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1225: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1226: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1227: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1228: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1229: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1230: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1231: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1232: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1233: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1234: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1235: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1236: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1237: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1238: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1239: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1240: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1241: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1242: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1243: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1244: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1245: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1246: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1247: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1248: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1249: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1250: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1251: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1252: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1253: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1254: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1255: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1256: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1257: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1258: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1259: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1260: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1261: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1262: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1263: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1264: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1265: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1266: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1267: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1268: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1269: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1270: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1271: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1272: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1273: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1274: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1275: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1276: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1277: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1278: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1279: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1280: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1281: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1282: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1283: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1284: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1285: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1286: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1287: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1288: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1289: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1290: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1291: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1292: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1293: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1294: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1295: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1296: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1297: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1298: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1299: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1300: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1301: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1302: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1303: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1304: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1305: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1306: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1307: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1308: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1309: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1310: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1311: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1312: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1313: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1314: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1315: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1316: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1317: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1318: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1319: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1320: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1321: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1322: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1323: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1324: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1325: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1326: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1327: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1328: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1329: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1330: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1331: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1332: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1333: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1334: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1335: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1336: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1337: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1338: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1339: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1340: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1341: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1342: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1343: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1344: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1345: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1346: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1347: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1348: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1349: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1350: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1351: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1352: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1353: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1354: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1355: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1356: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1357: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1358: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1359: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1360: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1361: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1362: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1363: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1364: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1365: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1366: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1367: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1368: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1369: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1370: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1371: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1372: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1373: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1374: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1375: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1376: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1377: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1378: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1379: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1380: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1381: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1382: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1383: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1384: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1385: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1386: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1387: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1388: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1389: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1390: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1391: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1392: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1393: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1394: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1395: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1396: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1397: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1398: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1399: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1400: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1401: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1402: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1403: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1404: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1405: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1406: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1407: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1408: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1409: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1410: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1411: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1412: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1413: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1414: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1415: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1416: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1417: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1418: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1419: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1420: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1421: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1422: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1423: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1424: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1425: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1426: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1427: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1428: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1429: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1430: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1431: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1432: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1433: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1434: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1435: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1436: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1437: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1438: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1439: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1440: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1441: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1442: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1443: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1444: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1445: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1446: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1447: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1448: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1449: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1450: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1451: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1452: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1453: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1454: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1455: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1456: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1457: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1458: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1459: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1460: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1461: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1462: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1463: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1464: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1465: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1466: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1467: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1468: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1469: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1470: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1471: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1472: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1473: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1474: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1475: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1476: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1477: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1478: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1479: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1480: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1481: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1482: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1483: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1484: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1485: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1486: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1487: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1488: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1489: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1490: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1491: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1492: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1493: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1494: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1495: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1496: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1497: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1498: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1499: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1500: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1501: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1502: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1503: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1504: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1505: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1506: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1507: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1508: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1509: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1510: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1511: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1512: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1513: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1514: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1515: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1516: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1517: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1518: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1519: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1520: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1521: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1522: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1523: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1524: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1525: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1526: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1527: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1528: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1529: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1530: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1531: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1532: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1533: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1534: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1535: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1536: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1537: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1538: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1539: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1540: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1541: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1542: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1543: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1544: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1545: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1546: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1547: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1548: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1549: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1550: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1551: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1552: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1553: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1554: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1555: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1556: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1557: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1558: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1559: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1560: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1561: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1562: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1563: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1564: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1565: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1566: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1567: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1568: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1569: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1570: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1571: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1572: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1573: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1574: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1575: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1576: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1577: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1578: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1579: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1580: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1581: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1582: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1583: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1584: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1585: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1586: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1587: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg1588: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1589: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true}, %arg1590: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1591: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1592: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1593: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1594: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1595: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1596: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1597: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1598: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1599: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1600: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1601: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1602: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1603: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1604: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1605: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1606: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1607: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1608: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1609: tensor<2048xf32> {mhlo.is_same_data_across_replicas = true}, %arg1610: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1611: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1612: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1613: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1614: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1615: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1616: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1617: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1618: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1619: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1620: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1621: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1622: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1623: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1624: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1625: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg1626: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1627: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg1628: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1629: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg1630: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg1631: tensor<f32> {mhlo.is_same_data_across_replicas = true}, %arg1632: tensor<1x256x224x224x3xbf16>, %arg1633: tensor<1x256xi32>) -> (tensor<1x256x1000xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.reshape %arg1632 : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
    %1 = stablehlo.reshape %arg1633 : (tensor<1x256xi32>) -> tensor<256xi32>
    %2 = stablehlo.convert %arg611 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xbf16>
    %3 = stablehlo.convolution(%0, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xbf16>, tensor<7x7x3x64xbf16>) -> tensor<256x112x112x64xbf16>
    %4 = stablehlo.broadcast_in_dim %arg1628, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1629, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %6 = stablehlo.convert %3 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
    %7 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %8 = stablehlo.subtract %6, %7 : tensor<256x112x112x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %10 = stablehlo.add %5, %9 : tensor<1x1x1x64xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x1x1x64xf32>
    %12 = stablehlo.reshape %arg610 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<1x1x1x64xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %15 = stablehlo.multiply %8, %14 : tensor<256x112x112x64xf32>
    %16 = stablehlo.reshape %arg609 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %18 = stablehlo.add %15, %17 : tensor<256x112x112x64xf32>
    %19 = stablehlo.convert %18 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %20 = call @relu(%19) : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xbf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %21 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %22 = "stablehlo.reduce_window"(%20, %21) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg1634: tensor<bf16>, %arg1635: tensor<bf16>):
      %3935 = stablehlo.maximum %arg1634, %arg1635 : tensor<bf16>
      stablehlo.return %3935 : tensor<bf16>
    }) : (tensor<256x112x112x64xbf16>, tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %23 = stablehlo.convert %arg7 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xbf16>
    %24 = stablehlo.convolution(%22, %23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %25 = stablehlo.broadcast_in_dim %arg1224, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %26 = stablehlo.broadcast_in_dim %arg1225, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %44 = stablehlo.broadcast_in_dim %arg1226, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %45 = stablehlo.broadcast_in_dim %arg1227, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %63 = stablehlo.broadcast_in_dim %arg1228, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %64 = stablehlo.broadcast_in_dim %arg1229, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
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
    %81 = stablehlo.broadcast_in_dim %arg1230, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %82 = stablehlo.broadcast_in_dim %arg1231, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
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
    %101 = stablehlo.broadcast_in_dim %arg1232, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %102 = stablehlo.broadcast_in_dim %arg1233, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %120 = stablehlo.broadcast_in_dim %arg1234, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %121 = stablehlo.broadcast_in_dim %arg1235, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %139 = stablehlo.broadcast_in_dim %arg1236, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %140 = stablehlo.broadcast_in_dim %arg1237, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
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
    %159 = stablehlo.broadcast_in_dim %arg1298, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %160 = stablehlo.broadcast_in_dim %arg1299, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %178 = stablehlo.broadcast_in_dim %arg1300, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %179 = stablehlo.broadcast_in_dim %arg1301, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %197 = stablehlo.broadcast_in_dim %arg1302, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %198 = stablehlo.broadcast_in_dim %arg1303, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
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
    %215 = stablehlo.convert %arg220 : (tensor<1x1x256x128xf32>) -> tensor<1x1x256x128xbf16>
    %216 = stablehlo.convolution(%214, %215) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xbf16>, tensor<1x1x256x128xbf16>) -> tensor<256x56x56x128xbf16>
    %217 = stablehlo.broadcast_in_dim %arg1366, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %218 = stablehlo.broadcast_in_dim %arg1367, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %219 = stablehlo.convert %216 : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xf32>
    %220 = stablehlo.broadcast_in_dim %217, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
    %221 = stablehlo.subtract %219, %220 : tensor<256x56x56x128xf32>
    %222 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %223 = stablehlo.add %218, %222 : tensor<1x1x1x128xf32>
    %224 = stablehlo.rsqrt %223 : tensor<1x1x1x128xf32>
    %225 = stablehlo.reshape %arg215 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %226 = stablehlo.multiply %224, %225 : tensor<1x1x1x128xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
    %228 = stablehlo.multiply %221, %227 : tensor<256x56x56x128xf32>
    %229 = stablehlo.reshape %arg214 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
    %231 = stablehlo.add %228, %230 : tensor<256x56x56x128xf32>
    %232 = stablehlo.convert %231 : (tensor<256x56x56x128xf32>) -> tensor<256x56x56x128xbf16>
    %233 = call @relu_55(%232) : (tensor<256x56x56x128xbf16>) -> tensor<256x56x56x128xbf16>
    %234 = stablehlo.convert %arg221 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %235 = stablehlo.convolution(%233, %234) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %236 = stablehlo.broadcast_in_dim %arg1368, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %237 = stablehlo.broadcast_in_dim %arg1369, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %238 = stablehlo.convert %235 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %239 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %240 = stablehlo.subtract %238, %239 : tensor<256x28x28x128xf32>
    %241 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %242 = stablehlo.add %237, %241 : tensor<1x1x1x128xf32>
    %243 = stablehlo.rsqrt %242 : tensor<1x1x1x128xf32>
    %244 = stablehlo.reshape %arg217 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %245 = stablehlo.multiply %243, %244 : tensor<1x1x1x128xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %247 = stablehlo.multiply %240, %246 : tensor<256x28x28x128xf32>
    %248 = stablehlo.reshape %arg216 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %249 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %250 = stablehlo.add %247, %249 : tensor<256x28x28x128xf32>
    %251 = stablehlo.convert %250 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %252 = call @relu_65(%251) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %253 = stablehlo.convert %arg222 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %254 = stablehlo.convolution(%252, %253) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %255 = stablehlo.broadcast_in_dim %arg1370, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %256 = stablehlo.broadcast_in_dim %arg1371, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %257 = stablehlo.convert %254 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %258 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %259 = stablehlo.subtract %257, %258 : tensor<256x28x28x512xf32>
    %260 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %261 = stablehlo.add %256, %260 : tensor<1x1x1x512xf32>
    %262 = stablehlo.rsqrt %261 : tensor<1x1x1x512xf32>
    %263 = stablehlo.reshape %arg219 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %264 = stablehlo.multiply %262, %263 : tensor<1x1x1x512xf32>
    %265 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %266 = stablehlo.multiply %259, %265 : tensor<256x28x28x512xf32>
    %267 = stablehlo.reshape %arg218 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %268 = stablehlo.broadcast_in_dim %267, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %269 = stablehlo.add %266, %268 : tensor<256x28x28x512xf32>
    %270 = stablehlo.convert %269 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %271 = stablehlo.convert %arg223 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xbf16>
    %272 = stablehlo.convolution(%214, %271) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xbf16>, tensor<1x1x256x512xbf16>) -> tensor<256x28x28x512xbf16>
    %273 = stablehlo.broadcast_in_dim %arg1372, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %274 = stablehlo.broadcast_in_dim %arg1373, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %275 = stablehlo.convert %272 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %276 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %277 = stablehlo.subtract %275, %276 : tensor<256x28x28x512xf32>
    %278 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %279 = stablehlo.add %274, %278 : tensor<1x1x1x512xf32>
    %280 = stablehlo.rsqrt %279 : tensor<1x1x1x512xf32>
    %281 = stablehlo.reshape %arg225 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %282 = stablehlo.multiply %280, %281 : tensor<1x1x1x512xf32>
    %283 = stablehlo.broadcast_in_dim %282, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %284 = stablehlo.multiply %277, %283 : tensor<256x28x28x512xf32>
    %285 = stablehlo.reshape %arg224 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %287 = stablehlo.add %284, %286 : tensor<256x28x28x512xf32>
    %288 = stablehlo.convert %287 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %289 = stablehlo.add %288, %270 : tensor<256x28x28x512xbf16>
    %290 = call @relu_83(%289) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %291 = stablehlo.convert %arg322 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %292 = stablehlo.convolution(%290, %291) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %293 = stablehlo.broadcast_in_dim %arg1434, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %294 = stablehlo.broadcast_in_dim %arg1435, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %295 = stablehlo.convert %292 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %296 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %297 = stablehlo.subtract %295, %296 : tensor<256x28x28x128xf32>
    %298 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %299 = stablehlo.add %294, %298 : tensor<1x1x1x128xf32>
    %300 = stablehlo.rsqrt %299 : tensor<1x1x1x128xf32>
    %301 = stablehlo.reshape %arg317 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %302 = stablehlo.multiply %300, %301 : tensor<1x1x1x128xf32>
    %303 = stablehlo.broadcast_in_dim %302, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %304 = stablehlo.multiply %297, %303 : tensor<256x28x28x128xf32>
    %305 = stablehlo.reshape %arg316 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %307 = stablehlo.add %304, %306 : tensor<256x28x28x128xf32>
    %308 = stablehlo.convert %307 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %309 = call @relu_65(%308) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %310 = stablehlo.convert %arg323 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %311 = stablehlo.convolution(%309, %310) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %312 = stablehlo.broadcast_in_dim %arg1436, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %313 = stablehlo.broadcast_in_dim %arg1437, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %314 = stablehlo.convert %311 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %315 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %316 = stablehlo.subtract %314, %315 : tensor<256x28x28x128xf32>
    %317 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %318 = stablehlo.add %313, %317 : tensor<1x1x1x128xf32>
    %319 = stablehlo.rsqrt %318 : tensor<1x1x1x128xf32>
    %320 = stablehlo.reshape %arg319 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %321 = stablehlo.multiply %319, %320 : tensor<1x1x1x128xf32>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %323 = stablehlo.multiply %316, %322 : tensor<256x28x28x128xf32>
    %324 = stablehlo.reshape %arg318 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %325 = stablehlo.broadcast_in_dim %324, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %326 = stablehlo.add %323, %325 : tensor<256x28x28x128xf32>
    %327 = stablehlo.convert %326 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %328 = call @relu_65(%327) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %329 = stablehlo.convert %arg324 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %330 = stablehlo.convolution(%328, %329) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %331 = stablehlo.broadcast_in_dim %arg1438, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %332 = stablehlo.broadcast_in_dim %arg1439, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %333 = stablehlo.convert %330 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %334 = stablehlo.broadcast_in_dim %331, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %335 = stablehlo.subtract %333, %334 : tensor<256x28x28x512xf32>
    %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %337 = stablehlo.add %332, %336 : tensor<1x1x1x512xf32>
    %338 = stablehlo.rsqrt %337 : tensor<1x1x1x512xf32>
    %339 = stablehlo.reshape %arg321 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %340 = stablehlo.multiply %338, %339 : tensor<1x1x1x512xf32>
    %341 = stablehlo.broadcast_in_dim %340, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %342 = stablehlo.multiply %335, %341 : tensor<256x28x28x512xf32>
    %343 = stablehlo.reshape %arg320 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %345 = stablehlo.add %342, %344 : tensor<256x28x28x512xf32>
    %346 = stablehlo.convert %345 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %347 = stablehlo.add %290, %346 : tensor<256x28x28x512xbf16>
    %348 = call @relu_83(%347) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %349 = stablehlo.convert %arg421 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %350 = stablehlo.convolution(%348, %349) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %351 = stablehlo.broadcast_in_dim %arg1500, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %352 = stablehlo.broadcast_in_dim %arg1501, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %353 = stablehlo.convert %350 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %354 = stablehlo.broadcast_in_dim %351, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %355 = stablehlo.subtract %353, %354 : tensor<256x28x28x128xf32>
    %356 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %357 = stablehlo.add %352, %356 : tensor<1x1x1x128xf32>
    %358 = stablehlo.rsqrt %357 : tensor<1x1x1x128xf32>
    %359 = stablehlo.reshape %arg416 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %360 = stablehlo.multiply %358, %359 : tensor<1x1x1x128xf32>
    %361 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %362 = stablehlo.multiply %355, %361 : tensor<256x28x28x128xf32>
    %363 = stablehlo.reshape %arg415 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %365 = stablehlo.add %362, %364 : tensor<256x28x28x128xf32>
    %366 = stablehlo.convert %365 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %367 = call @relu_65(%366) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %368 = stablehlo.convert %arg422 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %369 = stablehlo.convolution(%367, %368) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %370 = stablehlo.broadcast_in_dim %arg1502, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %371 = stablehlo.broadcast_in_dim %arg1503, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %372 = stablehlo.convert %369 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %373 = stablehlo.broadcast_in_dim %370, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %374 = stablehlo.subtract %372, %373 : tensor<256x28x28x128xf32>
    %375 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %376 = stablehlo.add %371, %375 : tensor<1x1x1x128xf32>
    %377 = stablehlo.rsqrt %376 : tensor<1x1x1x128xf32>
    %378 = stablehlo.reshape %arg418 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %379 = stablehlo.multiply %377, %378 : tensor<1x1x1x128xf32>
    %380 = stablehlo.broadcast_in_dim %379, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %381 = stablehlo.multiply %374, %380 : tensor<256x28x28x128xf32>
    %382 = stablehlo.reshape %arg417 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %383 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %384 = stablehlo.add %381, %383 : tensor<256x28x28x128xf32>
    %385 = stablehlo.convert %384 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %386 = call @relu_65(%385) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %387 = stablehlo.convert %arg423 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %388 = stablehlo.convolution(%386, %387) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %389 = stablehlo.broadcast_in_dim %arg1504, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %390 = stablehlo.broadcast_in_dim %arg1505, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %391 = stablehlo.convert %388 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %392 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %393 = stablehlo.subtract %391, %392 : tensor<256x28x28x512xf32>
    %394 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %395 = stablehlo.add %390, %394 : tensor<1x1x1x512xf32>
    %396 = stablehlo.rsqrt %395 : tensor<1x1x1x512xf32>
    %397 = stablehlo.reshape %arg420 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %398 = stablehlo.multiply %396, %397 : tensor<1x1x1x512xf32>
    %399 = stablehlo.broadcast_in_dim %398, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %400 = stablehlo.multiply %393, %399 : tensor<256x28x28x512xf32>
    %401 = stablehlo.reshape %arg419 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %403 = stablehlo.add %400, %402 : tensor<256x28x28x512xf32>
    %404 = stablehlo.convert %403 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %405 = stablehlo.add %348, %404 : tensor<256x28x28x512xbf16>
    %406 = call @relu_83(%405) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %407 = stablehlo.convert %arg520 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %408 = stablehlo.convolution(%406, %407) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %409 = stablehlo.broadcast_in_dim %arg1566, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %410 = stablehlo.broadcast_in_dim %arg1567, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %411 = stablehlo.convert %408 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %412 = stablehlo.broadcast_in_dim %409, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %413 = stablehlo.subtract %411, %412 : tensor<256x28x28x128xf32>
    %414 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %415 = stablehlo.add %410, %414 : tensor<1x1x1x128xf32>
    %416 = stablehlo.rsqrt %415 : tensor<1x1x1x128xf32>
    %417 = stablehlo.reshape %arg515 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %418 = stablehlo.multiply %416, %417 : tensor<1x1x1x128xf32>
    %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %420 = stablehlo.multiply %413, %419 : tensor<256x28x28x128xf32>
    %421 = stablehlo.reshape %arg514 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %423 = stablehlo.add %420, %422 : tensor<256x28x28x128xf32>
    %424 = stablehlo.convert %423 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %425 = call @relu_65(%424) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %426 = stablehlo.convert %arg521 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %427 = stablehlo.convolution(%425, %426) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %428 = stablehlo.broadcast_in_dim %arg1568, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %429 = stablehlo.broadcast_in_dim %arg1569, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %430 = stablehlo.convert %427 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %431 = stablehlo.broadcast_in_dim %428, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %432 = stablehlo.subtract %430, %431 : tensor<256x28x28x128xf32>
    %433 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %434 = stablehlo.add %429, %433 : tensor<1x1x1x128xf32>
    %435 = stablehlo.rsqrt %434 : tensor<1x1x1x128xf32>
    %436 = stablehlo.reshape %arg517 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %437 = stablehlo.multiply %435, %436 : tensor<1x1x1x128xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %439 = stablehlo.multiply %432, %438 : tensor<256x28x28x128xf32>
    %440 = stablehlo.reshape %arg516 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %441 = stablehlo.broadcast_in_dim %440, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %442 = stablehlo.add %439, %441 : tensor<256x28x28x128xf32>
    %443 = stablehlo.convert %442 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %444 = call @relu_65(%443) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %445 = stablehlo.convert %arg522 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %446 = stablehlo.convolution(%444, %445) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %447 = stablehlo.broadcast_in_dim %arg1570, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %448 = stablehlo.broadcast_in_dim %arg1571, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %449 = stablehlo.convert %446 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %450 = stablehlo.broadcast_in_dim %447, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %451 = stablehlo.subtract %449, %450 : tensor<256x28x28x512xf32>
    %452 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %453 = stablehlo.add %448, %452 : tensor<1x1x1x512xf32>
    %454 = stablehlo.rsqrt %453 : tensor<1x1x1x512xf32>
    %455 = stablehlo.reshape %arg519 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %456 = stablehlo.multiply %454, %455 : tensor<1x1x1x512xf32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %458 = stablehlo.multiply %451, %457 : tensor<256x28x28x512xf32>
    %459 = stablehlo.reshape %arg518 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %460 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %461 = stablehlo.add %458, %460 : tensor<256x28x28x512xf32>
    %462 = stablehlo.convert %461 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %463 = stablehlo.add %406, %462 : tensor<256x28x28x512xbf16>
    %464 = call @relu_83(%463) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %465 = stablehlo.convert %arg586 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %466 = stablehlo.convolution(%464, %465) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %467 = stablehlo.broadcast_in_dim %arg1610, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %468 = stablehlo.broadcast_in_dim %arg1611, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %469 = stablehlo.convert %466 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %470 = stablehlo.broadcast_in_dim %467, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %471 = stablehlo.subtract %469, %470 : tensor<256x28x28x128xf32>
    %472 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %473 = stablehlo.add %468, %472 : tensor<1x1x1x128xf32>
    %474 = stablehlo.rsqrt %473 : tensor<1x1x1x128xf32>
    %475 = stablehlo.reshape %arg581 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %476 = stablehlo.multiply %474, %475 : tensor<1x1x1x128xf32>
    %477 = stablehlo.broadcast_in_dim %476, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %478 = stablehlo.multiply %471, %477 : tensor<256x28x28x128xf32>
    %479 = stablehlo.reshape %arg580 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %481 = stablehlo.add %478, %480 : tensor<256x28x28x128xf32>
    %482 = stablehlo.convert %481 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %483 = call @relu_65(%482) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %484 = stablehlo.convert %arg587 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %485 = stablehlo.convolution(%483, %484) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %486 = stablehlo.broadcast_in_dim %arg1612, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %487 = stablehlo.broadcast_in_dim %arg1613, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %488 = stablehlo.convert %485 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %489 = stablehlo.broadcast_in_dim %486, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %490 = stablehlo.subtract %488, %489 : tensor<256x28x28x128xf32>
    %491 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %492 = stablehlo.add %487, %491 : tensor<1x1x1x128xf32>
    %493 = stablehlo.rsqrt %492 : tensor<1x1x1x128xf32>
    %494 = stablehlo.reshape %arg583 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %495 = stablehlo.multiply %493, %494 : tensor<1x1x1x128xf32>
    %496 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %497 = stablehlo.multiply %490, %496 : tensor<256x28x28x128xf32>
    %498 = stablehlo.reshape %arg582 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %499 = stablehlo.broadcast_in_dim %498, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %500 = stablehlo.add %497, %499 : tensor<256x28x28x128xf32>
    %501 = stablehlo.convert %500 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %502 = call @relu_65(%501) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %503 = stablehlo.convert %arg588 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %504 = stablehlo.convolution(%502, %503) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %505 = stablehlo.broadcast_in_dim %arg1614, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %506 = stablehlo.broadcast_in_dim %arg1615, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %507 = stablehlo.convert %504 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %508 = stablehlo.broadcast_in_dim %505, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %509 = stablehlo.subtract %507, %508 : tensor<256x28x28x512xf32>
    %510 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %511 = stablehlo.add %506, %510 : tensor<1x1x1x512xf32>
    %512 = stablehlo.rsqrt %511 : tensor<1x1x1x512xf32>
    %513 = stablehlo.reshape %arg585 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %514 = stablehlo.multiply %512, %513 : tensor<1x1x1x512xf32>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %516 = stablehlo.multiply %509, %515 : tensor<256x28x28x512xf32>
    %517 = stablehlo.reshape %arg584 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %519 = stablehlo.add %516, %518 : tensor<256x28x28x512xf32>
    %520 = stablehlo.convert %519 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %521 = stablehlo.add %464, %520 : tensor<256x28x28x512xbf16>
    %522 = call @relu_83(%521) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %523 = stablehlo.convert %arg595 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %524 = stablehlo.convolution(%522, %523) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %525 = stablehlo.broadcast_in_dim %arg1616, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %526 = stablehlo.broadcast_in_dim %arg1617, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %527 = stablehlo.convert %524 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %528 = stablehlo.broadcast_in_dim %525, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %529 = stablehlo.subtract %527, %528 : tensor<256x28x28x128xf32>
    %530 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %531 = stablehlo.add %526, %530 : tensor<1x1x1x128xf32>
    %532 = stablehlo.rsqrt %531 : tensor<1x1x1x128xf32>
    %533 = stablehlo.reshape %arg590 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %534 = stablehlo.multiply %532, %533 : tensor<1x1x1x128xf32>
    %535 = stablehlo.broadcast_in_dim %534, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %536 = stablehlo.multiply %529, %535 : tensor<256x28x28x128xf32>
    %537 = stablehlo.reshape %arg589 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %538 = stablehlo.broadcast_in_dim %537, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %539 = stablehlo.add %536, %538 : tensor<256x28x28x128xf32>
    %540 = stablehlo.convert %539 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %541 = call @relu_65(%540) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %542 = stablehlo.convert %arg596 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %543 = stablehlo.convolution(%541, %542) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %544 = stablehlo.broadcast_in_dim %arg1618, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %545 = stablehlo.broadcast_in_dim %arg1619, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %546 = stablehlo.convert %543 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %547 = stablehlo.broadcast_in_dim %544, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %548 = stablehlo.subtract %546, %547 : tensor<256x28x28x128xf32>
    %549 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %550 = stablehlo.add %545, %549 : tensor<1x1x1x128xf32>
    %551 = stablehlo.rsqrt %550 : tensor<1x1x1x128xf32>
    %552 = stablehlo.reshape %arg592 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %553 = stablehlo.multiply %551, %552 : tensor<1x1x1x128xf32>
    %554 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %555 = stablehlo.multiply %548, %554 : tensor<256x28x28x128xf32>
    %556 = stablehlo.reshape %arg591 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %557 = stablehlo.broadcast_in_dim %556, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %558 = stablehlo.add %555, %557 : tensor<256x28x28x128xf32>
    %559 = stablehlo.convert %558 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %560 = call @relu_65(%559) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %561 = stablehlo.convert %arg597 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %562 = stablehlo.convolution(%560, %561) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %563 = stablehlo.broadcast_in_dim %arg1620, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %564 = stablehlo.broadcast_in_dim %arg1621, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %565 = stablehlo.convert %562 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %566 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %567 = stablehlo.subtract %565, %566 : tensor<256x28x28x512xf32>
    %568 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %569 = stablehlo.add %564, %568 : tensor<1x1x1x512xf32>
    %570 = stablehlo.rsqrt %569 : tensor<1x1x1x512xf32>
    %571 = stablehlo.reshape %arg594 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %572 = stablehlo.multiply %570, %571 : tensor<1x1x1x512xf32>
    %573 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %574 = stablehlo.multiply %567, %573 : tensor<256x28x28x512xf32>
    %575 = stablehlo.reshape %arg593 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %577 = stablehlo.add %574, %576 : tensor<256x28x28x512xf32>
    %578 = stablehlo.convert %577 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %579 = stablehlo.add %522, %578 : tensor<256x28x28x512xbf16>
    %580 = call @relu_83(%579) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %581 = stablehlo.convert %arg604 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %582 = stablehlo.convolution(%580, %581) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %583 = stablehlo.broadcast_in_dim %arg1622, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %584 = stablehlo.broadcast_in_dim %arg1623, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %585 = stablehlo.convert %582 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %586 = stablehlo.broadcast_in_dim %583, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %587 = stablehlo.subtract %585, %586 : tensor<256x28x28x128xf32>
    %588 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %589 = stablehlo.add %584, %588 : tensor<1x1x1x128xf32>
    %590 = stablehlo.rsqrt %589 : tensor<1x1x1x128xf32>
    %591 = stablehlo.reshape %arg599 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %592 = stablehlo.multiply %590, %591 : tensor<1x1x1x128xf32>
    %593 = stablehlo.broadcast_in_dim %592, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %594 = stablehlo.multiply %587, %593 : tensor<256x28x28x128xf32>
    %595 = stablehlo.reshape %arg598 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %596 = stablehlo.broadcast_in_dim %595, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %597 = stablehlo.add %594, %596 : tensor<256x28x28x128xf32>
    %598 = stablehlo.convert %597 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %599 = call @relu_65(%598) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %600 = stablehlo.convert %arg605 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %601 = stablehlo.convolution(%599, %600) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %602 = stablehlo.broadcast_in_dim %arg1624, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %603 = stablehlo.broadcast_in_dim %arg1625, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %604 = stablehlo.convert %601 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %605 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %606 = stablehlo.subtract %604, %605 : tensor<256x28x28x128xf32>
    %607 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %608 = stablehlo.add %603, %607 : tensor<1x1x1x128xf32>
    %609 = stablehlo.rsqrt %608 : tensor<1x1x1x128xf32>
    %610 = stablehlo.reshape %arg601 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %611 = stablehlo.multiply %609, %610 : tensor<1x1x1x128xf32>
    %612 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %613 = stablehlo.multiply %606, %612 : tensor<256x28x28x128xf32>
    %614 = stablehlo.reshape %arg600 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %615 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %616 = stablehlo.add %613, %615 : tensor<256x28x28x128xf32>
    %617 = stablehlo.convert %616 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %618 = call @relu_65(%617) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %619 = stablehlo.convert %arg606 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %620 = stablehlo.convolution(%618, %619) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %621 = stablehlo.broadcast_in_dim %arg1626, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %622 = stablehlo.broadcast_in_dim %arg1627, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %623 = stablehlo.convert %620 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %624 = stablehlo.broadcast_in_dim %621, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %625 = stablehlo.subtract %623, %624 : tensor<256x28x28x512xf32>
    %626 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %627 = stablehlo.add %622, %626 : tensor<1x1x1x512xf32>
    %628 = stablehlo.rsqrt %627 : tensor<1x1x1x512xf32>
    %629 = stablehlo.reshape %arg603 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %630 = stablehlo.multiply %628, %629 : tensor<1x1x1x512xf32>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %632 = stablehlo.multiply %625, %631 : tensor<256x28x28x512xf32>
    %633 = stablehlo.reshape %arg602 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %634 = stablehlo.broadcast_in_dim %633, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %635 = stablehlo.add %632, %634 : tensor<256x28x28x512xf32>
    %636 = stablehlo.convert %635 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %637 = stablehlo.add %580, %636 : tensor<256x28x28x512xbf16>
    %638 = call @relu_83(%637) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %639 = stablehlo.convert %arg28 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %640 = stablehlo.convolution(%638, %639) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %641 = stablehlo.broadcast_in_dim %arg1238, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %642 = stablehlo.broadcast_in_dim %arg1239, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %643 = stablehlo.convert %640 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %644 = stablehlo.broadcast_in_dim %641, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %645 = stablehlo.subtract %643, %644 : tensor<256x28x28x128xf32>
    %646 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %647 = stablehlo.add %642, %646 : tensor<1x1x1x128xf32>
    %648 = stablehlo.rsqrt %647 : tensor<1x1x1x128xf32>
    %649 = stablehlo.reshape %arg23 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %650 = stablehlo.multiply %648, %649 : tensor<1x1x1x128xf32>
    %651 = stablehlo.broadcast_in_dim %650, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %652 = stablehlo.multiply %645, %651 : tensor<256x28x28x128xf32>
    %653 = stablehlo.reshape %arg22 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %654 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %655 = stablehlo.add %652, %654 : tensor<256x28x28x128xf32>
    %656 = stablehlo.convert %655 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %657 = call @relu_65(%656) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %658 = stablehlo.convert %arg29 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %659 = stablehlo.convolution(%657, %658) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %660 = stablehlo.broadcast_in_dim %arg1240, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %661 = stablehlo.broadcast_in_dim %arg1241, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %662 = stablehlo.convert %659 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %663 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %664 = stablehlo.subtract %662, %663 : tensor<256x28x28x128xf32>
    %665 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %666 = stablehlo.add %661, %665 : tensor<1x1x1x128xf32>
    %667 = stablehlo.rsqrt %666 : tensor<1x1x1x128xf32>
    %668 = stablehlo.reshape %arg25 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %669 = stablehlo.multiply %667, %668 : tensor<1x1x1x128xf32>
    %670 = stablehlo.broadcast_in_dim %669, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %671 = stablehlo.multiply %664, %670 : tensor<256x28x28x128xf32>
    %672 = stablehlo.reshape %arg24 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %674 = stablehlo.add %671, %673 : tensor<256x28x28x128xf32>
    %675 = stablehlo.convert %674 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %676 = call @relu_65(%675) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %677 = stablehlo.convert %arg30 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %678 = stablehlo.convolution(%676, %677) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %679 = stablehlo.broadcast_in_dim %arg1242, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %680 = stablehlo.broadcast_in_dim %arg1243, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %681 = stablehlo.convert %678 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %682 = stablehlo.broadcast_in_dim %679, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %683 = stablehlo.subtract %681, %682 : tensor<256x28x28x512xf32>
    %684 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %685 = stablehlo.add %680, %684 : tensor<1x1x1x512xf32>
    %686 = stablehlo.rsqrt %685 : tensor<1x1x1x512xf32>
    %687 = stablehlo.reshape %arg27 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %688 = stablehlo.multiply %686, %687 : tensor<1x1x1x512xf32>
    %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %690 = stablehlo.multiply %683, %689 : tensor<256x28x28x512xf32>
    %691 = stablehlo.reshape %arg26 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %692 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %693 = stablehlo.add %690, %692 : tensor<256x28x28x512xf32>
    %694 = stablehlo.convert %693 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %695 = stablehlo.add %638, %694 : tensor<256x28x28x512xbf16>
    %696 = call @relu_83(%695) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %697 = stablehlo.convert %arg37 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %698 = stablehlo.convolution(%696, %697) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %699 = stablehlo.broadcast_in_dim %arg1244, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %700 = stablehlo.broadcast_in_dim %arg1245, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %701 = stablehlo.convert %698 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %702 = stablehlo.broadcast_in_dim %699, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %703 = stablehlo.subtract %701, %702 : tensor<256x28x28x128xf32>
    %704 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %705 = stablehlo.add %700, %704 : tensor<1x1x1x128xf32>
    %706 = stablehlo.rsqrt %705 : tensor<1x1x1x128xf32>
    %707 = stablehlo.reshape %arg32 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %708 = stablehlo.multiply %706, %707 : tensor<1x1x1x128xf32>
    %709 = stablehlo.broadcast_in_dim %708, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %710 = stablehlo.multiply %703, %709 : tensor<256x28x28x128xf32>
    %711 = stablehlo.reshape %arg31 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %712 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %713 = stablehlo.add %710, %712 : tensor<256x28x28x128xf32>
    %714 = stablehlo.convert %713 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %715 = call @relu_65(%714) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %716 = stablehlo.convert %arg38 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %717 = stablehlo.convolution(%715, %716) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %718 = stablehlo.broadcast_in_dim %arg1246, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %719 = stablehlo.broadcast_in_dim %arg1247, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %720 = stablehlo.convert %717 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %721 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %722 = stablehlo.subtract %720, %721 : tensor<256x28x28x128xf32>
    %723 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %724 = stablehlo.add %719, %723 : tensor<1x1x1x128xf32>
    %725 = stablehlo.rsqrt %724 : tensor<1x1x1x128xf32>
    %726 = stablehlo.reshape %arg34 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %727 = stablehlo.multiply %725, %726 : tensor<1x1x1x128xf32>
    %728 = stablehlo.broadcast_in_dim %727, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %729 = stablehlo.multiply %722, %728 : tensor<256x28x28x128xf32>
    %730 = stablehlo.reshape %arg33 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %731 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %732 = stablehlo.add %729, %731 : tensor<256x28x28x128xf32>
    %733 = stablehlo.convert %732 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %734 = call @relu_65(%733) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %735 = stablehlo.convert %arg39 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %736 = stablehlo.convolution(%734, %735) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %737 = stablehlo.broadcast_in_dim %arg1248, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %738 = stablehlo.broadcast_in_dim %arg1249, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %739 = stablehlo.convert %736 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %740 = stablehlo.broadcast_in_dim %737, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %741 = stablehlo.subtract %739, %740 : tensor<256x28x28x512xf32>
    %742 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %743 = stablehlo.add %738, %742 : tensor<1x1x1x512xf32>
    %744 = stablehlo.rsqrt %743 : tensor<1x1x1x512xf32>
    %745 = stablehlo.reshape %arg36 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %746 = stablehlo.multiply %744, %745 : tensor<1x1x1x512xf32>
    %747 = stablehlo.broadcast_in_dim %746, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %748 = stablehlo.multiply %741, %747 : tensor<256x28x28x512xf32>
    %749 = stablehlo.reshape %arg35 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %750 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %751 = stablehlo.add %748, %750 : tensor<256x28x28x512xf32>
    %752 = stablehlo.convert %751 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %753 = stablehlo.add %696, %752 : tensor<256x28x28x512xbf16>
    %754 = call @relu_83(%753) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %755 = stablehlo.convert %arg46 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %756 = stablehlo.convolution(%754, %755) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %757 = stablehlo.broadcast_in_dim %arg1250, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %758 = stablehlo.broadcast_in_dim %arg1251, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %759 = stablehlo.convert %756 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %760 = stablehlo.broadcast_in_dim %757, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %761 = stablehlo.subtract %759, %760 : tensor<256x28x28x128xf32>
    %762 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %763 = stablehlo.add %758, %762 : tensor<1x1x1x128xf32>
    %764 = stablehlo.rsqrt %763 : tensor<1x1x1x128xf32>
    %765 = stablehlo.reshape %arg41 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %766 = stablehlo.multiply %764, %765 : tensor<1x1x1x128xf32>
    %767 = stablehlo.broadcast_in_dim %766, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %768 = stablehlo.multiply %761, %767 : tensor<256x28x28x128xf32>
    %769 = stablehlo.reshape %arg40 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %770 = stablehlo.broadcast_in_dim %769, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %771 = stablehlo.add %768, %770 : tensor<256x28x28x128xf32>
    %772 = stablehlo.convert %771 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %773 = call @relu_65(%772) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %774 = stablehlo.convert %arg47 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %775 = stablehlo.convolution(%773, %774) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %776 = stablehlo.broadcast_in_dim %arg1252, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %777 = stablehlo.broadcast_in_dim %arg1253, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %778 = stablehlo.convert %775 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %779 = stablehlo.broadcast_in_dim %776, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %780 = stablehlo.subtract %778, %779 : tensor<256x28x28x128xf32>
    %781 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %782 = stablehlo.add %777, %781 : tensor<1x1x1x128xf32>
    %783 = stablehlo.rsqrt %782 : tensor<1x1x1x128xf32>
    %784 = stablehlo.reshape %arg43 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %785 = stablehlo.multiply %783, %784 : tensor<1x1x1x128xf32>
    %786 = stablehlo.broadcast_in_dim %785, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %787 = stablehlo.multiply %780, %786 : tensor<256x28x28x128xf32>
    %788 = stablehlo.reshape %arg42 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %789 = stablehlo.broadcast_in_dim %788, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %790 = stablehlo.add %787, %789 : tensor<256x28x28x128xf32>
    %791 = stablehlo.convert %790 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %792 = call @relu_65(%791) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %793 = stablehlo.convert %arg48 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %794 = stablehlo.convolution(%792, %793) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %795 = stablehlo.broadcast_in_dim %arg1254, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %796 = stablehlo.broadcast_in_dim %arg1255, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %797 = stablehlo.convert %794 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %798 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %799 = stablehlo.subtract %797, %798 : tensor<256x28x28x512xf32>
    %800 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %801 = stablehlo.add %796, %800 : tensor<1x1x1x512xf32>
    %802 = stablehlo.rsqrt %801 : tensor<1x1x1x512xf32>
    %803 = stablehlo.reshape %arg45 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %804 = stablehlo.multiply %802, %803 : tensor<1x1x1x512xf32>
    %805 = stablehlo.broadcast_in_dim %804, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %806 = stablehlo.multiply %799, %805 : tensor<256x28x28x512xf32>
    %807 = stablehlo.reshape %arg44 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %808 = stablehlo.broadcast_in_dim %807, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %809 = stablehlo.add %806, %808 : tensor<256x28x28x512xf32>
    %810 = stablehlo.convert %809 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %811 = stablehlo.add %754, %810 : tensor<256x28x28x512xbf16>
    %812 = call @relu_83(%811) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %813 = stablehlo.convert %arg55 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %814 = stablehlo.convolution(%812, %813) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %815 = stablehlo.broadcast_in_dim %arg1256, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %816 = stablehlo.broadcast_in_dim %arg1257, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %817 = stablehlo.convert %814 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %818 = stablehlo.broadcast_in_dim %815, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %819 = stablehlo.subtract %817, %818 : tensor<256x28x28x128xf32>
    %820 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %821 = stablehlo.add %816, %820 : tensor<1x1x1x128xf32>
    %822 = stablehlo.rsqrt %821 : tensor<1x1x1x128xf32>
    %823 = stablehlo.reshape %arg50 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %824 = stablehlo.multiply %822, %823 : tensor<1x1x1x128xf32>
    %825 = stablehlo.broadcast_in_dim %824, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %826 = stablehlo.multiply %819, %825 : tensor<256x28x28x128xf32>
    %827 = stablehlo.reshape %arg49 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %828 = stablehlo.broadcast_in_dim %827, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %829 = stablehlo.add %826, %828 : tensor<256x28x28x128xf32>
    %830 = stablehlo.convert %829 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %831 = call @relu_65(%830) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %832 = stablehlo.convert %arg56 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %833 = stablehlo.convolution(%831, %832) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %834 = stablehlo.broadcast_in_dim %arg1258, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %835 = stablehlo.broadcast_in_dim %arg1259, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %836 = stablehlo.convert %833 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %837 = stablehlo.broadcast_in_dim %834, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %838 = stablehlo.subtract %836, %837 : tensor<256x28x28x128xf32>
    %839 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %840 = stablehlo.add %835, %839 : tensor<1x1x1x128xf32>
    %841 = stablehlo.rsqrt %840 : tensor<1x1x1x128xf32>
    %842 = stablehlo.reshape %arg52 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %843 = stablehlo.multiply %841, %842 : tensor<1x1x1x128xf32>
    %844 = stablehlo.broadcast_in_dim %843, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %845 = stablehlo.multiply %838, %844 : tensor<256x28x28x128xf32>
    %846 = stablehlo.reshape %arg51 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %847 = stablehlo.broadcast_in_dim %846, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %848 = stablehlo.add %845, %847 : tensor<256x28x28x128xf32>
    %849 = stablehlo.convert %848 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %850 = call @relu_65(%849) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %851 = stablehlo.convert %arg57 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %852 = stablehlo.convolution(%850, %851) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %853 = stablehlo.broadcast_in_dim %arg1260, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %854 = stablehlo.broadcast_in_dim %arg1261, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %855 = stablehlo.convert %852 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %856 = stablehlo.broadcast_in_dim %853, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %857 = stablehlo.subtract %855, %856 : tensor<256x28x28x512xf32>
    %858 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %859 = stablehlo.add %854, %858 : tensor<1x1x1x512xf32>
    %860 = stablehlo.rsqrt %859 : tensor<1x1x1x512xf32>
    %861 = stablehlo.reshape %arg54 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %862 = stablehlo.multiply %860, %861 : tensor<1x1x1x512xf32>
    %863 = stablehlo.broadcast_in_dim %862, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %864 = stablehlo.multiply %857, %863 : tensor<256x28x28x512xf32>
    %865 = stablehlo.reshape %arg53 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %866 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %867 = stablehlo.add %864, %866 : tensor<256x28x28x512xf32>
    %868 = stablehlo.convert %867 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %869 = stablehlo.add %812, %868 : tensor<256x28x28x512xbf16>
    %870 = call @relu_83(%869) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %871 = stablehlo.convert %arg64 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %872 = stablehlo.convolution(%870, %871) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %873 = stablehlo.broadcast_in_dim %arg1262, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %874 = stablehlo.broadcast_in_dim %arg1263, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %875 = stablehlo.convert %872 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %876 = stablehlo.broadcast_in_dim %873, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %877 = stablehlo.subtract %875, %876 : tensor<256x28x28x128xf32>
    %878 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %879 = stablehlo.add %874, %878 : tensor<1x1x1x128xf32>
    %880 = stablehlo.rsqrt %879 : tensor<1x1x1x128xf32>
    %881 = stablehlo.reshape %arg59 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %882 = stablehlo.multiply %880, %881 : tensor<1x1x1x128xf32>
    %883 = stablehlo.broadcast_in_dim %882, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %884 = stablehlo.multiply %877, %883 : tensor<256x28x28x128xf32>
    %885 = stablehlo.reshape %arg58 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %886 = stablehlo.broadcast_in_dim %885, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %887 = stablehlo.add %884, %886 : tensor<256x28x28x128xf32>
    %888 = stablehlo.convert %887 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %889 = call @relu_65(%888) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %890 = stablehlo.convert %arg65 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %891 = stablehlo.convolution(%889, %890) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %892 = stablehlo.broadcast_in_dim %arg1264, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %893 = stablehlo.broadcast_in_dim %arg1265, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %894 = stablehlo.convert %891 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %895 = stablehlo.broadcast_in_dim %892, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %896 = stablehlo.subtract %894, %895 : tensor<256x28x28x128xf32>
    %897 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %898 = stablehlo.add %893, %897 : tensor<1x1x1x128xf32>
    %899 = stablehlo.rsqrt %898 : tensor<1x1x1x128xf32>
    %900 = stablehlo.reshape %arg61 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %901 = stablehlo.multiply %899, %900 : tensor<1x1x1x128xf32>
    %902 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %903 = stablehlo.multiply %896, %902 : tensor<256x28x28x128xf32>
    %904 = stablehlo.reshape %arg60 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %905 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %906 = stablehlo.add %903, %905 : tensor<256x28x28x128xf32>
    %907 = stablehlo.convert %906 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %908 = call @relu_65(%907) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %909 = stablehlo.convert %arg66 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %910 = stablehlo.convolution(%908, %909) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %911 = stablehlo.broadcast_in_dim %arg1266, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %912 = stablehlo.broadcast_in_dim %arg1267, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %913 = stablehlo.convert %910 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %914 = stablehlo.broadcast_in_dim %911, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %915 = stablehlo.subtract %913, %914 : tensor<256x28x28x512xf32>
    %916 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %917 = stablehlo.add %912, %916 : tensor<1x1x1x512xf32>
    %918 = stablehlo.rsqrt %917 : tensor<1x1x1x512xf32>
    %919 = stablehlo.reshape %arg63 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %920 = stablehlo.multiply %918, %919 : tensor<1x1x1x512xf32>
    %921 = stablehlo.broadcast_in_dim %920, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %922 = stablehlo.multiply %915, %921 : tensor<256x28x28x512xf32>
    %923 = stablehlo.reshape %arg62 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %924 = stablehlo.broadcast_in_dim %923, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %925 = stablehlo.add %922, %924 : tensor<256x28x28x512xf32>
    %926 = stablehlo.convert %925 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %927 = stablehlo.add %870, %926 : tensor<256x28x28x512xbf16>
    %928 = call @relu_83(%927) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %929 = stablehlo.convert %arg73 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %930 = stablehlo.convolution(%928, %929) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %931 = stablehlo.broadcast_in_dim %arg1268, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %932 = stablehlo.broadcast_in_dim %arg1269, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %933 = stablehlo.convert %930 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %934 = stablehlo.broadcast_in_dim %931, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %935 = stablehlo.subtract %933, %934 : tensor<256x28x28x128xf32>
    %936 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %937 = stablehlo.add %932, %936 : tensor<1x1x1x128xf32>
    %938 = stablehlo.rsqrt %937 : tensor<1x1x1x128xf32>
    %939 = stablehlo.reshape %arg68 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %940 = stablehlo.multiply %938, %939 : tensor<1x1x1x128xf32>
    %941 = stablehlo.broadcast_in_dim %940, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %942 = stablehlo.multiply %935, %941 : tensor<256x28x28x128xf32>
    %943 = stablehlo.reshape %arg67 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %944 = stablehlo.broadcast_in_dim %943, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %945 = stablehlo.add %942, %944 : tensor<256x28x28x128xf32>
    %946 = stablehlo.convert %945 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %947 = call @relu_65(%946) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %948 = stablehlo.convert %arg74 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %949 = stablehlo.convolution(%947, %948) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %950 = stablehlo.broadcast_in_dim %arg1270, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %951 = stablehlo.broadcast_in_dim %arg1271, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %952 = stablehlo.convert %949 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %953 = stablehlo.broadcast_in_dim %950, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %954 = stablehlo.subtract %952, %953 : tensor<256x28x28x128xf32>
    %955 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %956 = stablehlo.add %951, %955 : tensor<1x1x1x128xf32>
    %957 = stablehlo.rsqrt %956 : tensor<1x1x1x128xf32>
    %958 = stablehlo.reshape %arg70 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %959 = stablehlo.multiply %957, %958 : tensor<1x1x1x128xf32>
    %960 = stablehlo.broadcast_in_dim %959, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %961 = stablehlo.multiply %954, %960 : tensor<256x28x28x128xf32>
    %962 = stablehlo.reshape %arg69 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %963 = stablehlo.broadcast_in_dim %962, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %964 = stablehlo.add %961, %963 : tensor<256x28x28x128xf32>
    %965 = stablehlo.convert %964 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %966 = call @relu_65(%965) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %967 = stablehlo.convert %arg75 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %968 = stablehlo.convolution(%966, %967) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %969 = stablehlo.broadcast_in_dim %arg1272, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %970 = stablehlo.broadcast_in_dim %arg1273, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %971 = stablehlo.convert %968 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %972 = stablehlo.broadcast_in_dim %969, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %973 = stablehlo.subtract %971, %972 : tensor<256x28x28x512xf32>
    %974 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %975 = stablehlo.add %970, %974 : tensor<1x1x1x512xf32>
    %976 = stablehlo.rsqrt %975 : tensor<1x1x1x512xf32>
    %977 = stablehlo.reshape %arg72 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %978 = stablehlo.multiply %976, %977 : tensor<1x1x1x512xf32>
    %979 = stablehlo.broadcast_in_dim %978, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %980 = stablehlo.multiply %973, %979 : tensor<256x28x28x512xf32>
    %981 = stablehlo.reshape %arg71 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %982 = stablehlo.broadcast_in_dim %981, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %983 = stablehlo.add %980, %982 : tensor<256x28x28x512xf32>
    %984 = stablehlo.convert %983 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %985 = stablehlo.add %928, %984 : tensor<256x28x28x512xbf16>
    %986 = call @relu_83(%985) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %987 = stablehlo.convert %arg82 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %988 = stablehlo.convolution(%986, %987) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %989 = stablehlo.broadcast_in_dim %arg1274, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %990 = stablehlo.broadcast_in_dim %arg1275, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %991 = stablehlo.convert %988 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %992 = stablehlo.broadcast_in_dim %989, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %993 = stablehlo.subtract %991, %992 : tensor<256x28x28x128xf32>
    %994 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %995 = stablehlo.add %990, %994 : tensor<1x1x1x128xf32>
    %996 = stablehlo.rsqrt %995 : tensor<1x1x1x128xf32>
    %997 = stablehlo.reshape %arg77 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %998 = stablehlo.multiply %996, %997 : tensor<1x1x1x128xf32>
    %999 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1000 = stablehlo.multiply %993, %999 : tensor<256x28x28x128xf32>
    %1001 = stablehlo.reshape %arg76 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1002 = stablehlo.broadcast_in_dim %1001, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1003 = stablehlo.add %1000, %1002 : tensor<256x28x28x128xf32>
    %1004 = stablehlo.convert %1003 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1005 = call @relu_65(%1004) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1006 = stablehlo.convert %arg83 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1007 = stablehlo.convolution(%1005, %1006) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1008 = stablehlo.broadcast_in_dim %arg1276, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1009 = stablehlo.broadcast_in_dim %arg1277, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1010 = stablehlo.convert %1007 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1011 = stablehlo.broadcast_in_dim %1008, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1012 = stablehlo.subtract %1010, %1011 : tensor<256x28x28x128xf32>
    %1013 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1014 = stablehlo.add %1009, %1013 : tensor<1x1x1x128xf32>
    %1015 = stablehlo.rsqrt %1014 : tensor<1x1x1x128xf32>
    %1016 = stablehlo.reshape %arg79 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1017 = stablehlo.multiply %1015, %1016 : tensor<1x1x1x128xf32>
    %1018 = stablehlo.broadcast_in_dim %1017, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1019 = stablehlo.multiply %1012, %1018 : tensor<256x28x28x128xf32>
    %1020 = stablehlo.reshape %arg78 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1021 = stablehlo.broadcast_in_dim %1020, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1022 = stablehlo.add %1019, %1021 : tensor<256x28x28x128xf32>
    %1023 = stablehlo.convert %1022 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1024 = call @relu_65(%1023) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1025 = stablehlo.convert %arg84 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1026 = stablehlo.convolution(%1024, %1025) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1027 = stablehlo.broadcast_in_dim %arg1278, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1028 = stablehlo.broadcast_in_dim %arg1279, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1029 = stablehlo.convert %1026 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1030 = stablehlo.broadcast_in_dim %1027, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1031 = stablehlo.subtract %1029, %1030 : tensor<256x28x28x512xf32>
    %1032 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1033 = stablehlo.add %1028, %1032 : tensor<1x1x1x512xf32>
    %1034 = stablehlo.rsqrt %1033 : tensor<1x1x1x512xf32>
    %1035 = stablehlo.reshape %arg81 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1036 = stablehlo.multiply %1034, %1035 : tensor<1x1x1x512xf32>
    %1037 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1038 = stablehlo.multiply %1031, %1037 : tensor<256x28x28x512xf32>
    %1039 = stablehlo.reshape %arg80 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1040 = stablehlo.broadcast_in_dim %1039, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1041 = stablehlo.add %1038, %1040 : tensor<256x28x28x512xf32>
    %1042 = stablehlo.convert %1041 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1043 = stablehlo.add %986, %1042 : tensor<256x28x28x512xbf16>
    %1044 = call @relu_83(%1043) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1045 = stablehlo.convert %arg91 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1046 = stablehlo.convolution(%1044, %1045) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1047 = stablehlo.broadcast_in_dim %arg1280, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1048 = stablehlo.broadcast_in_dim %arg1281, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1049 = stablehlo.convert %1046 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1050 = stablehlo.broadcast_in_dim %1047, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1051 = stablehlo.subtract %1049, %1050 : tensor<256x28x28x128xf32>
    %1052 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1053 = stablehlo.add %1048, %1052 : tensor<1x1x1x128xf32>
    %1054 = stablehlo.rsqrt %1053 : tensor<1x1x1x128xf32>
    %1055 = stablehlo.reshape %arg86 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1056 = stablehlo.multiply %1054, %1055 : tensor<1x1x1x128xf32>
    %1057 = stablehlo.broadcast_in_dim %1056, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1058 = stablehlo.multiply %1051, %1057 : tensor<256x28x28x128xf32>
    %1059 = stablehlo.reshape %arg85 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1060 = stablehlo.broadcast_in_dim %1059, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1061 = stablehlo.add %1058, %1060 : tensor<256x28x28x128xf32>
    %1062 = stablehlo.convert %1061 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1063 = call @relu_65(%1062) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1064 = stablehlo.convert %arg92 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1065 = stablehlo.convolution(%1063, %1064) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1066 = stablehlo.broadcast_in_dim %arg1282, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1067 = stablehlo.broadcast_in_dim %arg1283, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1068 = stablehlo.convert %1065 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1069 = stablehlo.broadcast_in_dim %1066, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1070 = stablehlo.subtract %1068, %1069 : tensor<256x28x28x128xf32>
    %1071 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1072 = stablehlo.add %1067, %1071 : tensor<1x1x1x128xf32>
    %1073 = stablehlo.rsqrt %1072 : tensor<1x1x1x128xf32>
    %1074 = stablehlo.reshape %arg88 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1075 = stablehlo.multiply %1073, %1074 : tensor<1x1x1x128xf32>
    %1076 = stablehlo.broadcast_in_dim %1075, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1077 = stablehlo.multiply %1070, %1076 : tensor<256x28x28x128xf32>
    %1078 = stablehlo.reshape %arg87 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1079 = stablehlo.broadcast_in_dim %1078, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1080 = stablehlo.add %1077, %1079 : tensor<256x28x28x128xf32>
    %1081 = stablehlo.convert %1080 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1082 = call @relu_65(%1081) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1083 = stablehlo.convert %arg93 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1084 = stablehlo.convolution(%1082, %1083) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1085 = stablehlo.broadcast_in_dim %arg1284, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1086 = stablehlo.broadcast_in_dim %arg1285, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1087 = stablehlo.convert %1084 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1088 = stablehlo.broadcast_in_dim %1085, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1089 = stablehlo.subtract %1087, %1088 : tensor<256x28x28x512xf32>
    %1090 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1091 = stablehlo.add %1086, %1090 : tensor<1x1x1x512xf32>
    %1092 = stablehlo.rsqrt %1091 : tensor<1x1x1x512xf32>
    %1093 = stablehlo.reshape %arg90 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1094 = stablehlo.multiply %1092, %1093 : tensor<1x1x1x512xf32>
    %1095 = stablehlo.broadcast_in_dim %1094, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1096 = stablehlo.multiply %1089, %1095 : tensor<256x28x28x512xf32>
    %1097 = stablehlo.reshape %arg89 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1098 = stablehlo.broadcast_in_dim %1097, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1099 = stablehlo.add %1096, %1098 : tensor<256x28x28x512xf32>
    %1100 = stablehlo.convert %1099 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1101 = stablehlo.add %1044, %1100 : tensor<256x28x28x512xbf16>
    %1102 = call @relu_83(%1101) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1103 = stablehlo.convert %arg100 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1104 = stablehlo.convolution(%1102, %1103) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1105 = stablehlo.broadcast_in_dim %arg1286, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1106 = stablehlo.broadcast_in_dim %arg1287, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1107 = stablehlo.convert %1104 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1108 = stablehlo.broadcast_in_dim %1105, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1109 = stablehlo.subtract %1107, %1108 : tensor<256x28x28x128xf32>
    %1110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1111 = stablehlo.add %1106, %1110 : tensor<1x1x1x128xf32>
    %1112 = stablehlo.rsqrt %1111 : tensor<1x1x1x128xf32>
    %1113 = stablehlo.reshape %arg95 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1114 = stablehlo.multiply %1112, %1113 : tensor<1x1x1x128xf32>
    %1115 = stablehlo.broadcast_in_dim %1114, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1116 = stablehlo.multiply %1109, %1115 : tensor<256x28x28x128xf32>
    %1117 = stablehlo.reshape %arg94 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1118 = stablehlo.broadcast_in_dim %1117, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1119 = stablehlo.add %1116, %1118 : tensor<256x28x28x128xf32>
    %1120 = stablehlo.convert %1119 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1121 = call @relu_65(%1120) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1122 = stablehlo.convert %arg101 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1123 = stablehlo.convolution(%1121, %1122) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1124 = stablehlo.broadcast_in_dim %arg1288, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1125 = stablehlo.broadcast_in_dim %arg1289, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1126 = stablehlo.convert %1123 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1127 = stablehlo.broadcast_in_dim %1124, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1128 = stablehlo.subtract %1126, %1127 : tensor<256x28x28x128xf32>
    %1129 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1130 = stablehlo.add %1125, %1129 : tensor<1x1x1x128xf32>
    %1131 = stablehlo.rsqrt %1130 : tensor<1x1x1x128xf32>
    %1132 = stablehlo.reshape %arg97 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1133 = stablehlo.multiply %1131, %1132 : tensor<1x1x1x128xf32>
    %1134 = stablehlo.broadcast_in_dim %1133, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1135 = stablehlo.multiply %1128, %1134 : tensor<256x28x28x128xf32>
    %1136 = stablehlo.reshape %arg96 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1138 = stablehlo.add %1135, %1137 : tensor<256x28x28x128xf32>
    %1139 = stablehlo.convert %1138 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1140 = call @relu_65(%1139) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1141 = stablehlo.convert %arg102 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1142 = stablehlo.convolution(%1140, %1141) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1143 = stablehlo.broadcast_in_dim %arg1290, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1144 = stablehlo.broadcast_in_dim %arg1291, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1145 = stablehlo.convert %1142 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1146 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1147 = stablehlo.subtract %1145, %1146 : tensor<256x28x28x512xf32>
    %1148 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1149 = stablehlo.add %1144, %1148 : tensor<1x1x1x512xf32>
    %1150 = stablehlo.rsqrt %1149 : tensor<1x1x1x512xf32>
    %1151 = stablehlo.reshape %arg99 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1152 = stablehlo.multiply %1150, %1151 : tensor<1x1x1x512xf32>
    %1153 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1154 = stablehlo.multiply %1147, %1153 : tensor<256x28x28x512xf32>
    %1155 = stablehlo.reshape %arg98 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1156 = stablehlo.broadcast_in_dim %1155, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1157 = stablehlo.add %1154, %1156 : tensor<256x28x28x512xf32>
    %1158 = stablehlo.convert %1157 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1159 = stablehlo.add %1102, %1158 : tensor<256x28x28x512xbf16>
    %1160 = call @relu_83(%1159) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1161 = stablehlo.convert %arg109 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1162 = stablehlo.convolution(%1160, %1161) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1163 = stablehlo.broadcast_in_dim %arg1292, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1164 = stablehlo.broadcast_in_dim %arg1293, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1165 = stablehlo.convert %1162 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1166 = stablehlo.broadcast_in_dim %1163, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1167 = stablehlo.subtract %1165, %1166 : tensor<256x28x28x128xf32>
    %1168 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1169 = stablehlo.add %1164, %1168 : tensor<1x1x1x128xf32>
    %1170 = stablehlo.rsqrt %1169 : tensor<1x1x1x128xf32>
    %1171 = stablehlo.reshape %arg104 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1172 = stablehlo.multiply %1170, %1171 : tensor<1x1x1x128xf32>
    %1173 = stablehlo.broadcast_in_dim %1172, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1174 = stablehlo.multiply %1167, %1173 : tensor<256x28x28x128xf32>
    %1175 = stablehlo.reshape %arg103 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1176 = stablehlo.broadcast_in_dim %1175, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1177 = stablehlo.add %1174, %1176 : tensor<256x28x28x128xf32>
    %1178 = stablehlo.convert %1177 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1179 = call @relu_65(%1178) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1180 = stablehlo.convert %arg110 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1181 = stablehlo.convolution(%1179, %1180) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1182 = stablehlo.broadcast_in_dim %arg1294, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1183 = stablehlo.broadcast_in_dim %arg1295, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1184 = stablehlo.convert %1181 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1185 = stablehlo.broadcast_in_dim %1182, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1186 = stablehlo.subtract %1184, %1185 : tensor<256x28x28x128xf32>
    %1187 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1188 = stablehlo.add %1183, %1187 : tensor<1x1x1x128xf32>
    %1189 = stablehlo.rsqrt %1188 : tensor<1x1x1x128xf32>
    %1190 = stablehlo.reshape %arg106 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1191 = stablehlo.multiply %1189, %1190 : tensor<1x1x1x128xf32>
    %1192 = stablehlo.broadcast_in_dim %1191, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1193 = stablehlo.multiply %1186, %1192 : tensor<256x28x28x128xf32>
    %1194 = stablehlo.reshape %arg105 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1195 = stablehlo.broadcast_in_dim %1194, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1196 = stablehlo.add %1193, %1195 : tensor<256x28x28x128xf32>
    %1197 = stablehlo.convert %1196 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1198 = call @relu_65(%1197) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1199 = stablehlo.convert %arg111 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1200 = stablehlo.convolution(%1198, %1199) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1201 = stablehlo.broadcast_in_dim %arg1296, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1202 = stablehlo.broadcast_in_dim %arg1297, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1203 = stablehlo.convert %1200 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1204 = stablehlo.broadcast_in_dim %1201, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1205 = stablehlo.subtract %1203, %1204 : tensor<256x28x28x512xf32>
    %1206 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1207 = stablehlo.add %1202, %1206 : tensor<1x1x1x512xf32>
    %1208 = stablehlo.rsqrt %1207 : tensor<1x1x1x512xf32>
    %1209 = stablehlo.reshape %arg108 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1210 = stablehlo.multiply %1208, %1209 : tensor<1x1x1x512xf32>
    %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1212 = stablehlo.multiply %1205, %1211 : tensor<256x28x28x512xf32>
    %1213 = stablehlo.reshape %arg107 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1214 = stablehlo.broadcast_in_dim %1213, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1215 = stablehlo.add %1212, %1214 : tensor<256x28x28x512xf32>
    %1216 = stablehlo.convert %1215 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1217 = stablehlo.add %1160, %1216 : tensor<256x28x28x512xbf16>
    %1218 = call @relu_83(%1217) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1219 = stablehlo.convert %arg127 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1220 = stablehlo.convolution(%1218, %1219) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1221 = stablehlo.broadcast_in_dim %arg1304, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1222 = stablehlo.broadcast_in_dim %arg1305, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1223 = stablehlo.convert %1220 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1224 = stablehlo.broadcast_in_dim %1221, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1225 = stablehlo.subtract %1223, %1224 : tensor<256x28x28x128xf32>
    %1226 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1227 = stablehlo.add %1222, %1226 : tensor<1x1x1x128xf32>
    %1228 = stablehlo.rsqrt %1227 : tensor<1x1x1x128xf32>
    %1229 = stablehlo.reshape %arg122 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1230 = stablehlo.multiply %1228, %1229 : tensor<1x1x1x128xf32>
    %1231 = stablehlo.broadcast_in_dim %1230, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1232 = stablehlo.multiply %1225, %1231 : tensor<256x28x28x128xf32>
    %1233 = stablehlo.reshape %arg121 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1234 = stablehlo.broadcast_in_dim %1233, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1235 = stablehlo.add %1232, %1234 : tensor<256x28x28x128xf32>
    %1236 = stablehlo.convert %1235 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1237 = call @relu_65(%1236) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1238 = stablehlo.convert %arg128 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1239 = stablehlo.convolution(%1237, %1238) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1240 = stablehlo.broadcast_in_dim %arg1306, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1241 = stablehlo.broadcast_in_dim %arg1307, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1242 = stablehlo.convert %1239 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1243 = stablehlo.broadcast_in_dim %1240, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1244 = stablehlo.subtract %1242, %1243 : tensor<256x28x28x128xf32>
    %1245 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1246 = stablehlo.add %1241, %1245 : tensor<1x1x1x128xf32>
    %1247 = stablehlo.rsqrt %1246 : tensor<1x1x1x128xf32>
    %1248 = stablehlo.reshape %arg124 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1249 = stablehlo.multiply %1247, %1248 : tensor<1x1x1x128xf32>
    %1250 = stablehlo.broadcast_in_dim %1249, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1251 = stablehlo.multiply %1244, %1250 : tensor<256x28x28x128xf32>
    %1252 = stablehlo.reshape %arg123 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1253 = stablehlo.broadcast_in_dim %1252, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1254 = stablehlo.add %1251, %1253 : tensor<256x28x28x128xf32>
    %1255 = stablehlo.convert %1254 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1256 = call @relu_65(%1255) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1257 = stablehlo.convert %arg129 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1258 = stablehlo.convolution(%1256, %1257) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1259 = stablehlo.broadcast_in_dim %arg1308, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1260 = stablehlo.broadcast_in_dim %arg1309, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1261 = stablehlo.convert %1258 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1262 = stablehlo.broadcast_in_dim %1259, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1263 = stablehlo.subtract %1261, %1262 : tensor<256x28x28x512xf32>
    %1264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1265 = stablehlo.add %1260, %1264 : tensor<1x1x1x512xf32>
    %1266 = stablehlo.rsqrt %1265 : tensor<1x1x1x512xf32>
    %1267 = stablehlo.reshape %arg126 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1268 = stablehlo.multiply %1266, %1267 : tensor<1x1x1x512xf32>
    %1269 = stablehlo.broadcast_in_dim %1268, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1270 = stablehlo.multiply %1263, %1269 : tensor<256x28x28x512xf32>
    %1271 = stablehlo.reshape %arg125 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1272 = stablehlo.broadcast_in_dim %1271, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1273 = stablehlo.add %1270, %1272 : tensor<256x28x28x512xf32>
    %1274 = stablehlo.convert %1273 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1275 = stablehlo.add %1218, %1274 : tensor<256x28x28x512xbf16>
    %1276 = call @relu_83(%1275) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1277 = stablehlo.convert %arg136 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1278 = stablehlo.convolution(%1276, %1277) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1279 = stablehlo.broadcast_in_dim %arg1310, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1280 = stablehlo.broadcast_in_dim %arg1311, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1281 = stablehlo.convert %1278 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1282 = stablehlo.broadcast_in_dim %1279, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1283 = stablehlo.subtract %1281, %1282 : tensor<256x28x28x128xf32>
    %1284 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1285 = stablehlo.add %1280, %1284 : tensor<1x1x1x128xf32>
    %1286 = stablehlo.rsqrt %1285 : tensor<1x1x1x128xf32>
    %1287 = stablehlo.reshape %arg131 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1288 = stablehlo.multiply %1286, %1287 : tensor<1x1x1x128xf32>
    %1289 = stablehlo.broadcast_in_dim %1288, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1290 = stablehlo.multiply %1283, %1289 : tensor<256x28x28x128xf32>
    %1291 = stablehlo.reshape %arg130 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1292 = stablehlo.broadcast_in_dim %1291, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1293 = stablehlo.add %1290, %1292 : tensor<256x28x28x128xf32>
    %1294 = stablehlo.convert %1293 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1295 = call @relu_65(%1294) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1296 = stablehlo.convert %arg137 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1297 = stablehlo.convolution(%1295, %1296) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1298 = stablehlo.broadcast_in_dim %arg1312, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1299 = stablehlo.broadcast_in_dim %arg1313, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1300 = stablehlo.convert %1297 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1301 = stablehlo.broadcast_in_dim %1298, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1302 = stablehlo.subtract %1300, %1301 : tensor<256x28x28x128xf32>
    %1303 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1304 = stablehlo.add %1299, %1303 : tensor<1x1x1x128xf32>
    %1305 = stablehlo.rsqrt %1304 : tensor<1x1x1x128xf32>
    %1306 = stablehlo.reshape %arg133 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1307 = stablehlo.multiply %1305, %1306 : tensor<1x1x1x128xf32>
    %1308 = stablehlo.broadcast_in_dim %1307, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1309 = stablehlo.multiply %1302, %1308 : tensor<256x28x28x128xf32>
    %1310 = stablehlo.reshape %arg132 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1311 = stablehlo.broadcast_in_dim %1310, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1312 = stablehlo.add %1309, %1311 : tensor<256x28x28x128xf32>
    %1313 = stablehlo.convert %1312 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1314 = call @relu_65(%1313) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1315 = stablehlo.convert %arg138 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1316 = stablehlo.convolution(%1314, %1315) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1317 = stablehlo.broadcast_in_dim %arg1314, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1318 = stablehlo.broadcast_in_dim %arg1315, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1319 = stablehlo.convert %1316 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1320 = stablehlo.broadcast_in_dim %1317, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1321 = stablehlo.subtract %1319, %1320 : tensor<256x28x28x512xf32>
    %1322 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1323 = stablehlo.add %1318, %1322 : tensor<1x1x1x512xf32>
    %1324 = stablehlo.rsqrt %1323 : tensor<1x1x1x512xf32>
    %1325 = stablehlo.reshape %arg135 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1326 = stablehlo.multiply %1324, %1325 : tensor<1x1x1x512xf32>
    %1327 = stablehlo.broadcast_in_dim %1326, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1328 = stablehlo.multiply %1321, %1327 : tensor<256x28x28x512xf32>
    %1329 = stablehlo.reshape %arg134 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1330 = stablehlo.broadcast_in_dim %1329, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1331 = stablehlo.add %1328, %1330 : tensor<256x28x28x512xf32>
    %1332 = stablehlo.convert %1331 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1333 = stablehlo.add %1276, %1332 : tensor<256x28x28x512xbf16>
    %1334 = call @relu_83(%1333) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1335 = stablehlo.convert %arg145 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1336 = stablehlo.convolution(%1334, %1335) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1337 = stablehlo.broadcast_in_dim %arg1316, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1338 = stablehlo.broadcast_in_dim %arg1317, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1339 = stablehlo.convert %1336 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1340 = stablehlo.broadcast_in_dim %1337, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1341 = stablehlo.subtract %1339, %1340 : tensor<256x28x28x128xf32>
    %1342 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1343 = stablehlo.add %1338, %1342 : tensor<1x1x1x128xf32>
    %1344 = stablehlo.rsqrt %1343 : tensor<1x1x1x128xf32>
    %1345 = stablehlo.reshape %arg140 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1346 = stablehlo.multiply %1344, %1345 : tensor<1x1x1x128xf32>
    %1347 = stablehlo.broadcast_in_dim %1346, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1348 = stablehlo.multiply %1341, %1347 : tensor<256x28x28x128xf32>
    %1349 = stablehlo.reshape %arg139 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1350 = stablehlo.broadcast_in_dim %1349, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1351 = stablehlo.add %1348, %1350 : tensor<256x28x28x128xf32>
    %1352 = stablehlo.convert %1351 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1353 = call @relu_65(%1352) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1354 = stablehlo.convert %arg146 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1355 = stablehlo.convolution(%1353, %1354) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1356 = stablehlo.broadcast_in_dim %arg1318, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1357 = stablehlo.broadcast_in_dim %arg1319, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1358 = stablehlo.convert %1355 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1359 = stablehlo.broadcast_in_dim %1356, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1360 = stablehlo.subtract %1358, %1359 : tensor<256x28x28x128xf32>
    %1361 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1362 = stablehlo.add %1357, %1361 : tensor<1x1x1x128xf32>
    %1363 = stablehlo.rsqrt %1362 : tensor<1x1x1x128xf32>
    %1364 = stablehlo.reshape %arg142 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1365 = stablehlo.multiply %1363, %1364 : tensor<1x1x1x128xf32>
    %1366 = stablehlo.broadcast_in_dim %1365, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1367 = stablehlo.multiply %1360, %1366 : tensor<256x28x28x128xf32>
    %1368 = stablehlo.reshape %arg141 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1369 = stablehlo.broadcast_in_dim %1368, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1370 = stablehlo.add %1367, %1369 : tensor<256x28x28x128xf32>
    %1371 = stablehlo.convert %1370 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1372 = call @relu_65(%1371) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1373 = stablehlo.convert %arg147 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1374 = stablehlo.convolution(%1372, %1373) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1375 = stablehlo.broadcast_in_dim %arg1320, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1376 = stablehlo.broadcast_in_dim %arg1321, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1377 = stablehlo.convert %1374 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1378 = stablehlo.broadcast_in_dim %1375, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1379 = stablehlo.subtract %1377, %1378 : tensor<256x28x28x512xf32>
    %1380 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1381 = stablehlo.add %1376, %1380 : tensor<1x1x1x512xf32>
    %1382 = stablehlo.rsqrt %1381 : tensor<1x1x1x512xf32>
    %1383 = stablehlo.reshape %arg144 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1384 = stablehlo.multiply %1382, %1383 : tensor<1x1x1x512xf32>
    %1385 = stablehlo.broadcast_in_dim %1384, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1386 = stablehlo.multiply %1379, %1385 : tensor<256x28x28x512xf32>
    %1387 = stablehlo.reshape %arg143 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1388 = stablehlo.broadcast_in_dim %1387, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1389 = stablehlo.add %1386, %1388 : tensor<256x28x28x512xf32>
    %1390 = stablehlo.convert %1389 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1391 = stablehlo.add %1334, %1390 : tensor<256x28x28x512xbf16>
    %1392 = call @relu_83(%1391) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1393 = stablehlo.convert %arg154 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1394 = stablehlo.convolution(%1392, %1393) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1395 = stablehlo.broadcast_in_dim %arg1322, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1396 = stablehlo.broadcast_in_dim %arg1323, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1397 = stablehlo.convert %1394 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1398 = stablehlo.broadcast_in_dim %1395, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1399 = stablehlo.subtract %1397, %1398 : tensor<256x28x28x128xf32>
    %1400 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1401 = stablehlo.add %1396, %1400 : tensor<1x1x1x128xf32>
    %1402 = stablehlo.rsqrt %1401 : tensor<1x1x1x128xf32>
    %1403 = stablehlo.reshape %arg149 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1404 = stablehlo.multiply %1402, %1403 : tensor<1x1x1x128xf32>
    %1405 = stablehlo.broadcast_in_dim %1404, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1406 = stablehlo.multiply %1399, %1405 : tensor<256x28x28x128xf32>
    %1407 = stablehlo.reshape %arg148 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1408 = stablehlo.broadcast_in_dim %1407, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1409 = stablehlo.add %1406, %1408 : tensor<256x28x28x128xf32>
    %1410 = stablehlo.convert %1409 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1411 = call @relu_65(%1410) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1412 = stablehlo.convert %arg155 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1413 = stablehlo.convolution(%1411, %1412) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1414 = stablehlo.broadcast_in_dim %arg1324, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1415 = stablehlo.broadcast_in_dim %arg1325, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1416 = stablehlo.convert %1413 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1417 = stablehlo.broadcast_in_dim %1414, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1418 = stablehlo.subtract %1416, %1417 : tensor<256x28x28x128xf32>
    %1419 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1420 = stablehlo.add %1415, %1419 : tensor<1x1x1x128xf32>
    %1421 = stablehlo.rsqrt %1420 : tensor<1x1x1x128xf32>
    %1422 = stablehlo.reshape %arg151 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1423 = stablehlo.multiply %1421, %1422 : tensor<1x1x1x128xf32>
    %1424 = stablehlo.broadcast_in_dim %1423, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1425 = stablehlo.multiply %1418, %1424 : tensor<256x28x28x128xf32>
    %1426 = stablehlo.reshape %arg150 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1427 = stablehlo.broadcast_in_dim %1426, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1428 = stablehlo.add %1425, %1427 : tensor<256x28x28x128xf32>
    %1429 = stablehlo.convert %1428 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1430 = call @relu_65(%1429) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1431 = stablehlo.convert %arg156 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1432 = stablehlo.convolution(%1430, %1431) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1433 = stablehlo.broadcast_in_dim %arg1326, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1434 = stablehlo.broadcast_in_dim %arg1327, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1435 = stablehlo.convert %1432 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1436 = stablehlo.broadcast_in_dim %1433, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1437 = stablehlo.subtract %1435, %1436 : tensor<256x28x28x512xf32>
    %1438 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1439 = stablehlo.add %1434, %1438 : tensor<1x1x1x512xf32>
    %1440 = stablehlo.rsqrt %1439 : tensor<1x1x1x512xf32>
    %1441 = stablehlo.reshape %arg153 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1442 = stablehlo.multiply %1440, %1441 : tensor<1x1x1x512xf32>
    %1443 = stablehlo.broadcast_in_dim %1442, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1444 = stablehlo.multiply %1437, %1443 : tensor<256x28x28x512xf32>
    %1445 = stablehlo.reshape %arg152 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1446 = stablehlo.broadcast_in_dim %1445, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1447 = stablehlo.add %1444, %1446 : tensor<256x28x28x512xf32>
    %1448 = stablehlo.convert %1447 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1449 = stablehlo.add %1392, %1448 : tensor<256x28x28x512xbf16>
    %1450 = call @relu_83(%1449) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1451 = stablehlo.convert %arg163 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1452 = stablehlo.convolution(%1450, %1451) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1453 = stablehlo.broadcast_in_dim %arg1328, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1454 = stablehlo.broadcast_in_dim %arg1329, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1455 = stablehlo.convert %1452 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1456 = stablehlo.broadcast_in_dim %1453, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1457 = stablehlo.subtract %1455, %1456 : tensor<256x28x28x128xf32>
    %1458 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1459 = stablehlo.add %1454, %1458 : tensor<1x1x1x128xf32>
    %1460 = stablehlo.rsqrt %1459 : tensor<1x1x1x128xf32>
    %1461 = stablehlo.reshape %arg158 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1462 = stablehlo.multiply %1460, %1461 : tensor<1x1x1x128xf32>
    %1463 = stablehlo.broadcast_in_dim %1462, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1464 = stablehlo.multiply %1457, %1463 : tensor<256x28x28x128xf32>
    %1465 = stablehlo.reshape %arg157 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1466 = stablehlo.broadcast_in_dim %1465, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1467 = stablehlo.add %1464, %1466 : tensor<256x28x28x128xf32>
    %1468 = stablehlo.convert %1467 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1469 = call @relu_65(%1468) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1470 = stablehlo.convert %arg164 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1471 = stablehlo.convolution(%1469, %1470) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1472 = stablehlo.broadcast_in_dim %arg1330, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1473 = stablehlo.broadcast_in_dim %arg1331, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1474 = stablehlo.convert %1471 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1475 = stablehlo.broadcast_in_dim %1472, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1476 = stablehlo.subtract %1474, %1475 : tensor<256x28x28x128xf32>
    %1477 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1478 = stablehlo.add %1473, %1477 : tensor<1x1x1x128xf32>
    %1479 = stablehlo.rsqrt %1478 : tensor<1x1x1x128xf32>
    %1480 = stablehlo.reshape %arg160 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1481 = stablehlo.multiply %1479, %1480 : tensor<1x1x1x128xf32>
    %1482 = stablehlo.broadcast_in_dim %1481, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1483 = stablehlo.multiply %1476, %1482 : tensor<256x28x28x128xf32>
    %1484 = stablehlo.reshape %arg159 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1485 = stablehlo.broadcast_in_dim %1484, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1486 = stablehlo.add %1483, %1485 : tensor<256x28x28x128xf32>
    %1487 = stablehlo.convert %1486 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1488 = call @relu_65(%1487) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1489 = stablehlo.convert %arg165 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1490 = stablehlo.convolution(%1488, %1489) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1491 = stablehlo.broadcast_in_dim %arg1332, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1492 = stablehlo.broadcast_in_dim %arg1333, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1493 = stablehlo.convert %1490 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1494 = stablehlo.broadcast_in_dim %1491, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1495 = stablehlo.subtract %1493, %1494 : tensor<256x28x28x512xf32>
    %1496 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1497 = stablehlo.add %1492, %1496 : tensor<1x1x1x512xf32>
    %1498 = stablehlo.rsqrt %1497 : tensor<1x1x1x512xf32>
    %1499 = stablehlo.reshape %arg162 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1500 = stablehlo.multiply %1498, %1499 : tensor<1x1x1x512xf32>
    %1501 = stablehlo.broadcast_in_dim %1500, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1502 = stablehlo.multiply %1495, %1501 : tensor<256x28x28x512xf32>
    %1503 = stablehlo.reshape %arg161 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1504 = stablehlo.broadcast_in_dim %1503, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1505 = stablehlo.add %1502, %1504 : tensor<256x28x28x512xf32>
    %1506 = stablehlo.convert %1505 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1507 = stablehlo.add %1450, %1506 : tensor<256x28x28x512xbf16>
    %1508 = call @relu_83(%1507) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1509 = stablehlo.convert %arg172 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1510 = stablehlo.convolution(%1508, %1509) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1511 = stablehlo.broadcast_in_dim %arg1334, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1512 = stablehlo.broadcast_in_dim %arg1335, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1513 = stablehlo.convert %1510 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1514 = stablehlo.broadcast_in_dim %1511, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1515 = stablehlo.subtract %1513, %1514 : tensor<256x28x28x128xf32>
    %1516 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1517 = stablehlo.add %1512, %1516 : tensor<1x1x1x128xf32>
    %1518 = stablehlo.rsqrt %1517 : tensor<1x1x1x128xf32>
    %1519 = stablehlo.reshape %arg167 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1520 = stablehlo.multiply %1518, %1519 : tensor<1x1x1x128xf32>
    %1521 = stablehlo.broadcast_in_dim %1520, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1522 = stablehlo.multiply %1515, %1521 : tensor<256x28x28x128xf32>
    %1523 = stablehlo.reshape %arg166 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1524 = stablehlo.broadcast_in_dim %1523, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1525 = stablehlo.add %1522, %1524 : tensor<256x28x28x128xf32>
    %1526 = stablehlo.convert %1525 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1527 = call @relu_65(%1526) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1528 = stablehlo.convert %arg173 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1529 = stablehlo.convolution(%1527, %1528) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1530 = stablehlo.broadcast_in_dim %arg1336, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1531 = stablehlo.broadcast_in_dim %arg1337, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1532 = stablehlo.convert %1529 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1533 = stablehlo.broadcast_in_dim %1530, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1534 = stablehlo.subtract %1532, %1533 : tensor<256x28x28x128xf32>
    %1535 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1536 = stablehlo.add %1531, %1535 : tensor<1x1x1x128xf32>
    %1537 = stablehlo.rsqrt %1536 : tensor<1x1x1x128xf32>
    %1538 = stablehlo.reshape %arg169 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1539 = stablehlo.multiply %1537, %1538 : tensor<1x1x1x128xf32>
    %1540 = stablehlo.broadcast_in_dim %1539, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1541 = stablehlo.multiply %1534, %1540 : tensor<256x28x28x128xf32>
    %1542 = stablehlo.reshape %arg168 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1543 = stablehlo.broadcast_in_dim %1542, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1544 = stablehlo.add %1541, %1543 : tensor<256x28x28x128xf32>
    %1545 = stablehlo.convert %1544 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1546 = call @relu_65(%1545) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1547 = stablehlo.convert %arg174 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1548 = stablehlo.convolution(%1546, %1547) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1549 = stablehlo.broadcast_in_dim %arg1338, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1550 = stablehlo.broadcast_in_dim %arg1339, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1551 = stablehlo.convert %1548 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1552 = stablehlo.broadcast_in_dim %1549, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1553 = stablehlo.subtract %1551, %1552 : tensor<256x28x28x512xf32>
    %1554 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1555 = stablehlo.add %1550, %1554 : tensor<1x1x1x512xf32>
    %1556 = stablehlo.rsqrt %1555 : tensor<1x1x1x512xf32>
    %1557 = stablehlo.reshape %arg171 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1558 = stablehlo.multiply %1556, %1557 : tensor<1x1x1x512xf32>
    %1559 = stablehlo.broadcast_in_dim %1558, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1560 = stablehlo.multiply %1553, %1559 : tensor<256x28x28x512xf32>
    %1561 = stablehlo.reshape %arg170 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1562 = stablehlo.broadcast_in_dim %1561, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1563 = stablehlo.add %1560, %1562 : tensor<256x28x28x512xf32>
    %1564 = stablehlo.convert %1563 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1565 = stablehlo.add %1508, %1564 : tensor<256x28x28x512xbf16>
    %1566 = call @relu_83(%1565) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1567 = stablehlo.convert %arg181 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xbf16>
    %1568 = stablehlo.convolution(%1566, %1567) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1569 = stablehlo.broadcast_in_dim %arg1340, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1570 = stablehlo.broadcast_in_dim %arg1341, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1571 = stablehlo.convert %1568 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1572 = stablehlo.broadcast_in_dim %1569, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1573 = stablehlo.subtract %1571, %1572 : tensor<256x28x28x128xf32>
    %1574 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1575 = stablehlo.add %1570, %1574 : tensor<1x1x1x128xf32>
    %1576 = stablehlo.rsqrt %1575 : tensor<1x1x1x128xf32>
    %1577 = stablehlo.reshape %arg176 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1578 = stablehlo.multiply %1576, %1577 : tensor<1x1x1x128xf32>
    %1579 = stablehlo.broadcast_in_dim %1578, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1580 = stablehlo.multiply %1573, %1579 : tensor<256x28x28x128xf32>
    %1581 = stablehlo.reshape %arg175 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1582 = stablehlo.broadcast_in_dim %1581, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1583 = stablehlo.add %1580, %1582 : tensor<256x28x28x128xf32>
    %1584 = stablehlo.convert %1583 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1585 = call @relu_65(%1584) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1586 = stablehlo.convert %arg182 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1587 = stablehlo.convolution(%1585, %1586) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1588 = stablehlo.broadcast_in_dim %arg1342, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1589 = stablehlo.broadcast_in_dim %arg1343, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1590 = stablehlo.convert %1587 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1591 = stablehlo.broadcast_in_dim %1588, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1592 = stablehlo.subtract %1590, %1591 : tensor<256x28x28x128xf32>
    %1593 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1594 = stablehlo.add %1589, %1593 : tensor<1x1x1x128xf32>
    %1595 = stablehlo.rsqrt %1594 : tensor<1x1x1x128xf32>
    %1596 = stablehlo.reshape %arg178 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1597 = stablehlo.multiply %1595, %1596 : tensor<1x1x1x128xf32>
    %1598 = stablehlo.broadcast_in_dim %1597, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1599 = stablehlo.multiply %1592, %1598 : tensor<256x28x28x128xf32>
    %1600 = stablehlo.reshape %arg177 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1601 = stablehlo.broadcast_in_dim %1600, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1602 = stablehlo.add %1599, %1601 : tensor<256x28x28x128xf32>
    %1603 = stablehlo.convert %1602 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1604 = call @relu_65(%1603) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1605 = stablehlo.convert %arg183 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xbf16>
    %1606 = stablehlo.convolution(%1604, %1605) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1607 = stablehlo.broadcast_in_dim %arg1344, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1608 = stablehlo.broadcast_in_dim %arg1345, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1609 = stablehlo.convert %1606 : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xf32>
    %1610 = stablehlo.broadcast_in_dim %1607, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1611 = stablehlo.subtract %1609, %1610 : tensor<256x28x28x512xf32>
    %1612 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %1613 = stablehlo.add %1608, %1612 : tensor<1x1x1x512xf32>
    %1614 = stablehlo.rsqrt %1613 : tensor<1x1x1x512xf32>
    %1615 = stablehlo.reshape %arg180 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1616 = stablehlo.multiply %1614, %1615 : tensor<1x1x1x512xf32>
    %1617 = stablehlo.broadcast_in_dim %1616, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1618 = stablehlo.multiply %1611, %1617 : tensor<256x28x28x512xf32>
    %1619 = stablehlo.reshape %arg179 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %1620 = stablehlo.broadcast_in_dim %1619, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
    %1621 = stablehlo.add %1618, %1620 : tensor<256x28x28x512xf32>
    %1622 = stablehlo.convert %1621 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xbf16>
    %1623 = stablehlo.add %1566, %1622 : tensor<256x28x28x512xbf16>
    %1624 = call @relu_83(%1623) : (tensor<256x28x28x512xbf16>) -> tensor<256x28x28x512xbf16>
    %1625 = stablehlo.convert %arg190 : (tensor<1x1x512x256xf32>) -> tensor<1x1x512x256xbf16>
    %1626 = stablehlo.convolution(%1624, %1625) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x256xbf16>) -> tensor<256x28x28x256xbf16>
    %1627 = stablehlo.broadcast_in_dim %arg1346, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1628 = stablehlo.broadcast_in_dim %arg1347, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1629 = stablehlo.convert %1626 : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xf32>
    %1630 = stablehlo.broadcast_in_dim %1627, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
    %1631 = stablehlo.subtract %1629, %1630 : tensor<256x28x28x256xf32>
    %1632 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1633 = stablehlo.add %1628, %1632 : tensor<1x1x1x256xf32>
    %1634 = stablehlo.rsqrt %1633 : tensor<1x1x1x256xf32>
    %1635 = stablehlo.reshape %arg185 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1636 = stablehlo.multiply %1634, %1635 : tensor<1x1x1x256xf32>
    %1637 = stablehlo.broadcast_in_dim %1636, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
    %1638 = stablehlo.multiply %1631, %1637 : tensor<256x28x28x256xf32>
    %1639 = stablehlo.reshape %arg184 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1640 = stablehlo.broadcast_in_dim %1639, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
    %1641 = stablehlo.add %1638, %1640 : tensor<256x28x28x256xf32>
    %1642 = stablehlo.convert %1641 : (tensor<256x28x28x256xf32>) -> tensor<256x28x28x256xbf16>
    %1643 = call @relu_165(%1642) : (tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xbf16>
    %1644 = stablehlo.convert %arg191 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1645 = stablehlo.convolution(%1643, %1644) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1646 = stablehlo.broadcast_in_dim %arg1348, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1647 = stablehlo.broadcast_in_dim %arg1349, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1648 = stablehlo.convert %1645 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1649 = stablehlo.broadcast_in_dim %1646, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1650 = stablehlo.subtract %1648, %1649 : tensor<256x14x14x256xf32>
    %1651 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1652 = stablehlo.add %1647, %1651 : tensor<1x1x1x256xf32>
    %1653 = stablehlo.rsqrt %1652 : tensor<1x1x1x256xf32>
    %1654 = stablehlo.reshape %arg187 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1655 = stablehlo.multiply %1653, %1654 : tensor<1x1x1x256xf32>
    %1656 = stablehlo.broadcast_in_dim %1655, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1657 = stablehlo.multiply %1650, %1656 : tensor<256x14x14x256xf32>
    %1658 = stablehlo.reshape %arg186 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1659 = stablehlo.broadcast_in_dim %1658, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1660 = stablehlo.add %1657, %1659 : tensor<256x14x14x256xf32>
    %1661 = stablehlo.convert %1660 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1662 = call @relu_175(%1661) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1663 = stablehlo.convert %arg192 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1664 = stablehlo.convolution(%1662, %1663) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1665 = stablehlo.broadcast_in_dim %arg1350, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1666 = stablehlo.broadcast_in_dim %arg1351, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1667 = stablehlo.convert %1664 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1668 = stablehlo.broadcast_in_dim %1665, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1669 = stablehlo.subtract %1667, %1668 : tensor<256x14x14x1024xf32>
    %1670 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1671 = stablehlo.add %1666, %1670 : tensor<1x1x1x1024xf32>
    %1672 = stablehlo.rsqrt %1671 : tensor<1x1x1x1024xf32>
    %1673 = stablehlo.reshape %arg189 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1674 = stablehlo.multiply %1672, %1673 : tensor<1x1x1x1024xf32>
    %1675 = stablehlo.broadcast_in_dim %1674, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1676 = stablehlo.multiply %1669, %1675 : tensor<256x14x14x1024xf32>
    %1677 = stablehlo.reshape %arg188 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1678 = stablehlo.broadcast_in_dim %1677, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1679 = stablehlo.add %1676, %1678 : tensor<256x14x14x1024xf32>
    %1680 = stablehlo.convert %1679 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1681 = stablehlo.convert %arg193 : (tensor<1x1x512x1024xf32>) -> tensor<1x1x512x1024xbf16>
    %1682 = stablehlo.convolution(%1624, %1681) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xbf16>, tensor<1x1x512x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1683 = stablehlo.broadcast_in_dim %arg1352, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1684 = stablehlo.broadcast_in_dim %arg1353, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1685 = stablehlo.convert %1682 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1686 = stablehlo.broadcast_in_dim %1683, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1687 = stablehlo.subtract %1685, %1686 : tensor<256x14x14x1024xf32>
    %1688 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1689 = stablehlo.add %1684, %1688 : tensor<1x1x1x1024xf32>
    %1690 = stablehlo.rsqrt %1689 : tensor<1x1x1x1024xf32>
    %1691 = stablehlo.reshape %arg195 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1692 = stablehlo.multiply %1690, %1691 : tensor<1x1x1x1024xf32>
    %1693 = stablehlo.broadcast_in_dim %1692, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1694 = stablehlo.multiply %1687, %1693 : tensor<256x14x14x1024xf32>
    %1695 = stablehlo.reshape %arg194 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1697 = stablehlo.add %1694, %1696 : tensor<256x14x14x1024xf32>
    %1698 = stablehlo.convert %1697 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1699 = stablehlo.add %1698, %1680 : tensor<256x14x14x1024xbf16>
    %1700 = call @relu_193(%1699) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1701 = stablehlo.convert %arg202 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1702 = stablehlo.convolution(%1700, %1701) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1703 = stablehlo.broadcast_in_dim %arg1354, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1704 = stablehlo.broadcast_in_dim %arg1355, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1705 = stablehlo.convert %1702 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1706 = stablehlo.broadcast_in_dim %1703, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1707 = stablehlo.subtract %1705, %1706 : tensor<256x14x14x256xf32>
    %1708 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1709 = stablehlo.add %1704, %1708 : tensor<1x1x1x256xf32>
    %1710 = stablehlo.rsqrt %1709 : tensor<1x1x1x256xf32>
    %1711 = stablehlo.reshape %arg197 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1712 = stablehlo.multiply %1710, %1711 : tensor<1x1x1x256xf32>
    %1713 = stablehlo.broadcast_in_dim %1712, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1714 = stablehlo.multiply %1707, %1713 : tensor<256x14x14x256xf32>
    %1715 = stablehlo.reshape %arg196 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1716 = stablehlo.broadcast_in_dim %1715, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1717 = stablehlo.add %1714, %1716 : tensor<256x14x14x256xf32>
    %1718 = stablehlo.convert %1717 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1719 = call @relu_175(%1718) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1720 = stablehlo.convert %arg203 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1721 = stablehlo.convolution(%1719, %1720) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1722 = stablehlo.broadcast_in_dim %arg1356, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1723 = stablehlo.broadcast_in_dim %arg1357, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1724 = stablehlo.convert %1721 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1725 = stablehlo.broadcast_in_dim %1722, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1726 = stablehlo.subtract %1724, %1725 : tensor<256x14x14x256xf32>
    %1727 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1728 = stablehlo.add %1723, %1727 : tensor<1x1x1x256xf32>
    %1729 = stablehlo.rsqrt %1728 : tensor<1x1x1x256xf32>
    %1730 = stablehlo.reshape %arg199 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1731 = stablehlo.multiply %1729, %1730 : tensor<1x1x1x256xf32>
    %1732 = stablehlo.broadcast_in_dim %1731, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1733 = stablehlo.multiply %1726, %1732 : tensor<256x14x14x256xf32>
    %1734 = stablehlo.reshape %arg198 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1735 = stablehlo.broadcast_in_dim %1734, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1736 = stablehlo.add %1733, %1735 : tensor<256x14x14x256xf32>
    %1737 = stablehlo.convert %1736 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1738 = call @relu_175(%1737) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1739 = stablehlo.convert %arg204 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1740 = stablehlo.convolution(%1738, %1739) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1741 = stablehlo.broadcast_in_dim %arg1358, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1742 = stablehlo.broadcast_in_dim %arg1359, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1743 = stablehlo.convert %1740 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1744 = stablehlo.broadcast_in_dim %1741, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1745 = stablehlo.subtract %1743, %1744 : tensor<256x14x14x1024xf32>
    %1746 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1747 = stablehlo.add %1742, %1746 : tensor<1x1x1x1024xf32>
    %1748 = stablehlo.rsqrt %1747 : tensor<1x1x1x1024xf32>
    %1749 = stablehlo.reshape %arg201 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1750 = stablehlo.multiply %1748, %1749 : tensor<1x1x1x1024xf32>
    %1751 = stablehlo.broadcast_in_dim %1750, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1752 = stablehlo.multiply %1745, %1751 : tensor<256x14x14x1024xf32>
    %1753 = stablehlo.reshape %arg200 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1754 = stablehlo.broadcast_in_dim %1753, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1755 = stablehlo.add %1752, %1754 : tensor<256x14x14x1024xf32>
    %1756 = stablehlo.convert %1755 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1757 = stablehlo.add %1700, %1756 : tensor<256x14x14x1024xbf16>
    %1758 = call @relu_193(%1757) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1759 = stablehlo.convert %arg211 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1760 = stablehlo.convolution(%1758, %1759) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1761 = stablehlo.broadcast_in_dim %arg1360, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1762 = stablehlo.broadcast_in_dim %arg1361, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1763 = stablehlo.convert %1760 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1764 = stablehlo.broadcast_in_dim %1761, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1765 = stablehlo.subtract %1763, %1764 : tensor<256x14x14x256xf32>
    %1766 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1767 = stablehlo.add %1762, %1766 : tensor<1x1x1x256xf32>
    %1768 = stablehlo.rsqrt %1767 : tensor<1x1x1x256xf32>
    %1769 = stablehlo.reshape %arg206 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1770 = stablehlo.multiply %1768, %1769 : tensor<1x1x1x256xf32>
    %1771 = stablehlo.broadcast_in_dim %1770, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1772 = stablehlo.multiply %1765, %1771 : tensor<256x14x14x256xf32>
    %1773 = stablehlo.reshape %arg205 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1774 = stablehlo.broadcast_in_dim %1773, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1775 = stablehlo.add %1772, %1774 : tensor<256x14x14x256xf32>
    %1776 = stablehlo.convert %1775 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1777 = call @relu_175(%1776) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1778 = stablehlo.convert %arg212 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1779 = stablehlo.convolution(%1777, %1778) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1780 = stablehlo.broadcast_in_dim %arg1362, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1781 = stablehlo.broadcast_in_dim %arg1363, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1782 = stablehlo.convert %1779 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1783 = stablehlo.broadcast_in_dim %1780, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1784 = stablehlo.subtract %1782, %1783 : tensor<256x14x14x256xf32>
    %1785 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1786 = stablehlo.add %1781, %1785 : tensor<1x1x1x256xf32>
    %1787 = stablehlo.rsqrt %1786 : tensor<1x1x1x256xf32>
    %1788 = stablehlo.reshape %arg208 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1789 = stablehlo.multiply %1787, %1788 : tensor<1x1x1x256xf32>
    %1790 = stablehlo.broadcast_in_dim %1789, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1791 = stablehlo.multiply %1784, %1790 : tensor<256x14x14x256xf32>
    %1792 = stablehlo.reshape %arg207 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1793 = stablehlo.broadcast_in_dim %1792, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1794 = stablehlo.add %1791, %1793 : tensor<256x14x14x256xf32>
    %1795 = stablehlo.convert %1794 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1796 = call @relu_175(%1795) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1797 = stablehlo.convert %arg213 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1798 = stablehlo.convolution(%1796, %1797) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1799 = stablehlo.broadcast_in_dim %arg1364, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1800 = stablehlo.broadcast_in_dim %arg1365, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1801 = stablehlo.convert %1798 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1802 = stablehlo.broadcast_in_dim %1799, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1803 = stablehlo.subtract %1801, %1802 : tensor<256x14x14x1024xf32>
    %1804 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1805 = stablehlo.add %1800, %1804 : tensor<1x1x1x1024xf32>
    %1806 = stablehlo.rsqrt %1805 : tensor<1x1x1x1024xf32>
    %1807 = stablehlo.reshape %arg210 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1808 = stablehlo.multiply %1806, %1807 : tensor<1x1x1x1024xf32>
    %1809 = stablehlo.broadcast_in_dim %1808, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1810 = stablehlo.multiply %1803, %1809 : tensor<256x14x14x1024xf32>
    %1811 = stablehlo.reshape %arg209 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1812 = stablehlo.broadcast_in_dim %1811, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1813 = stablehlo.add %1810, %1812 : tensor<256x14x14x1024xf32>
    %1814 = stablehlo.convert %1813 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1815 = stablehlo.add %1758, %1814 : tensor<256x14x14x1024xbf16>
    %1816 = call @relu_193(%1815) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1817 = stablehlo.convert %arg232 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1818 = stablehlo.convolution(%1816, %1817) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1819 = stablehlo.broadcast_in_dim %arg1374, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1820 = stablehlo.broadcast_in_dim %arg1375, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1821 = stablehlo.convert %1818 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1822 = stablehlo.broadcast_in_dim %1819, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1823 = stablehlo.subtract %1821, %1822 : tensor<256x14x14x256xf32>
    %1824 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1825 = stablehlo.add %1820, %1824 : tensor<1x1x1x256xf32>
    %1826 = stablehlo.rsqrt %1825 : tensor<1x1x1x256xf32>
    %1827 = stablehlo.reshape %arg227 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1828 = stablehlo.multiply %1826, %1827 : tensor<1x1x1x256xf32>
    %1829 = stablehlo.broadcast_in_dim %1828, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1830 = stablehlo.multiply %1823, %1829 : tensor<256x14x14x256xf32>
    %1831 = stablehlo.reshape %arg226 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1832 = stablehlo.broadcast_in_dim %1831, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1833 = stablehlo.add %1830, %1832 : tensor<256x14x14x256xf32>
    %1834 = stablehlo.convert %1833 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1835 = call @relu_175(%1834) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1836 = stablehlo.convert %arg233 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1837 = stablehlo.convolution(%1835, %1836) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1838 = stablehlo.broadcast_in_dim %arg1376, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1839 = stablehlo.broadcast_in_dim %arg1377, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1840 = stablehlo.convert %1837 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1841 = stablehlo.broadcast_in_dim %1838, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1842 = stablehlo.subtract %1840, %1841 : tensor<256x14x14x256xf32>
    %1843 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1844 = stablehlo.add %1839, %1843 : tensor<1x1x1x256xf32>
    %1845 = stablehlo.rsqrt %1844 : tensor<1x1x1x256xf32>
    %1846 = stablehlo.reshape %arg229 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1847 = stablehlo.multiply %1845, %1846 : tensor<1x1x1x256xf32>
    %1848 = stablehlo.broadcast_in_dim %1847, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1849 = stablehlo.multiply %1842, %1848 : tensor<256x14x14x256xf32>
    %1850 = stablehlo.reshape %arg228 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1851 = stablehlo.broadcast_in_dim %1850, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1852 = stablehlo.add %1849, %1851 : tensor<256x14x14x256xf32>
    %1853 = stablehlo.convert %1852 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1854 = call @relu_175(%1853) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1855 = stablehlo.convert %arg234 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1856 = stablehlo.convolution(%1854, %1855) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1857 = stablehlo.broadcast_in_dim %arg1378, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1858 = stablehlo.broadcast_in_dim %arg1379, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1859 = stablehlo.convert %1856 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1860 = stablehlo.broadcast_in_dim %1857, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1861 = stablehlo.subtract %1859, %1860 : tensor<256x14x14x1024xf32>
    %1862 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1863 = stablehlo.add %1858, %1862 : tensor<1x1x1x1024xf32>
    %1864 = stablehlo.rsqrt %1863 : tensor<1x1x1x1024xf32>
    %1865 = stablehlo.reshape %arg231 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1866 = stablehlo.multiply %1864, %1865 : tensor<1x1x1x1024xf32>
    %1867 = stablehlo.broadcast_in_dim %1866, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1868 = stablehlo.multiply %1861, %1867 : tensor<256x14x14x1024xf32>
    %1869 = stablehlo.reshape %arg230 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1870 = stablehlo.broadcast_in_dim %1869, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1871 = stablehlo.add %1868, %1870 : tensor<256x14x14x1024xf32>
    %1872 = stablehlo.convert %1871 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1873 = stablehlo.add %1816, %1872 : tensor<256x14x14x1024xbf16>
    %1874 = call @relu_193(%1873) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1875 = stablehlo.convert %arg241 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1876 = stablehlo.convolution(%1874, %1875) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1877 = stablehlo.broadcast_in_dim %arg1380, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1878 = stablehlo.broadcast_in_dim %arg1381, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1879 = stablehlo.convert %1876 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1880 = stablehlo.broadcast_in_dim %1877, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1881 = stablehlo.subtract %1879, %1880 : tensor<256x14x14x256xf32>
    %1882 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1883 = stablehlo.add %1878, %1882 : tensor<1x1x1x256xf32>
    %1884 = stablehlo.rsqrt %1883 : tensor<1x1x1x256xf32>
    %1885 = stablehlo.reshape %arg236 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1886 = stablehlo.multiply %1884, %1885 : tensor<1x1x1x256xf32>
    %1887 = stablehlo.broadcast_in_dim %1886, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1888 = stablehlo.multiply %1881, %1887 : tensor<256x14x14x256xf32>
    %1889 = stablehlo.reshape %arg235 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1890 = stablehlo.broadcast_in_dim %1889, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1891 = stablehlo.add %1888, %1890 : tensor<256x14x14x256xf32>
    %1892 = stablehlo.convert %1891 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1893 = call @relu_175(%1892) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1894 = stablehlo.convert %arg242 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1895 = stablehlo.convolution(%1893, %1894) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1896 = stablehlo.broadcast_in_dim %arg1382, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1897 = stablehlo.broadcast_in_dim %arg1383, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1898 = stablehlo.convert %1895 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1899 = stablehlo.broadcast_in_dim %1896, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1900 = stablehlo.subtract %1898, %1899 : tensor<256x14x14x256xf32>
    %1901 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1902 = stablehlo.add %1897, %1901 : tensor<1x1x1x256xf32>
    %1903 = stablehlo.rsqrt %1902 : tensor<1x1x1x256xf32>
    %1904 = stablehlo.reshape %arg238 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1905 = stablehlo.multiply %1903, %1904 : tensor<1x1x1x256xf32>
    %1906 = stablehlo.broadcast_in_dim %1905, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1907 = stablehlo.multiply %1900, %1906 : tensor<256x14x14x256xf32>
    %1908 = stablehlo.reshape %arg237 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1909 = stablehlo.broadcast_in_dim %1908, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1910 = stablehlo.add %1907, %1909 : tensor<256x14x14x256xf32>
    %1911 = stablehlo.convert %1910 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1912 = call @relu_175(%1911) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1913 = stablehlo.convert %arg243 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1914 = stablehlo.convolution(%1912, %1913) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1915 = stablehlo.broadcast_in_dim %arg1384, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1916 = stablehlo.broadcast_in_dim %arg1385, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1917 = stablehlo.convert %1914 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1918 = stablehlo.broadcast_in_dim %1915, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1919 = stablehlo.subtract %1917, %1918 : tensor<256x14x14x1024xf32>
    %1920 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1921 = stablehlo.add %1916, %1920 : tensor<1x1x1x1024xf32>
    %1922 = stablehlo.rsqrt %1921 : tensor<1x1x1x1024xf32>
    %1923 = stablehlo.reshape %arg240 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1924 = stablehlo.multiply %1922, %1923 : tensor<1x1x1x1024xf32>
    %1925 = stablehlo.broadcast_in_dim %1924, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1926 = stablehlo.multiply %1919, %1925 : tensor<256x14x14x1024xf32>
    %1927 = stablehlo.reshape %arg239 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1928 = stablehlo.broadcast_in_dim %1927, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1929 = stablehlo.add %1926, %1928 : tensor<256x14x14x1024xf32>
    %1930 = stablehlo.convert %1929 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1931 = stablehlo.add %1874, %1930 : tensor<256x14x14x1024xbf16>
    %1932 = call @relu_193(%1931) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1933 = stablehlo.convert %arg250 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1934 = stablehlo.convolution(%1932, %1933) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1935 = stablehlo.broadcast_in_dim %arg1386, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1936 = stablehlo.broadcast_in_dim %arg1387, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1937 = stablehlo.convert %1934 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1938 = stablehlo.broadcast_in_dim %1935, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1939 = stablehlo.subtract %1937, %1938 : tensor<256x14x14x256xf32>
    %1940 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1941 = stablehlo.add %1936, %1940 : tensor<1x1x1x256xf32>
    %1942 = stablehlo.rsqrt %1941 : tensor<1x1x1x256xf32>
    %1943 = stablehlo.reshape %arg245 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1944 = stablehlo.multiply %1942, %1943 : tensor<1x1x1x256xf32>
    %1945 = stablehlo.broadcast_in_dim %1944, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1946 = stablehlo.multiply %1939, %1945 : tensor<256x14x14x256xf32>
    %1947 = stablehlo.reshape %arg244 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1948 = stablehlo.broadcast_in_dim %1947, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1949 = stablehlo.add %1946, %1948 : tensor<256x14x14x256xf32>
    %1950 = stablehlo.convert %1949 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1951 = call @relu_175(%1950) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1952 = stablehlo.convert %arg251 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1953 = stablehlo.convolution(%1951, %1952) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1954 = stablehlo.broadcast_in_dim %arg1388, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1955 = stablehlo.broadcast_in_dim %arg1389, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1956 = stablehlo.convert %1953 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1957 = stablehlo.broadcast_in_dim %1954, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1958 = stablehlo.subtract %1956, %1957 : tensor<256x14x14x256xf32>
    %1959 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1960 = stablehlo.add %1955, %1959 : tensor<1x1x1x256xf32>
    %1961 = stablehlo.rsqrt %1960 : tensor<1x1x1x256xf32>
    %1962 = stablehlo.reshape %arg247 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1963 = stablehlo.multiply %1961, %1962 : tensor<1x1x1x256xf32>
    %1964 = stablehlo.broadcast_in_dim %1963, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1965 = stablehlo.multiply %1958, %1964 : tensor<256x14x14x256xf32>
    %1966 = stablehlo.reshape %arg246 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1967 = stablehlo.broadcast_in_dim %1966, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1968 = stablehlo.add %1965, %1967 : tensor<256x14x14x256xf32>
    %1969 = stablehlo.convert %1968 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1970 = call @relu_175(%1969) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1971 = stablehlo.convert %arg252 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %1972 = stablehlo.convolution(%1970, %1971) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1973 = stablehlo.broadcast_in_dim %arg1390, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1974 = stablehlo.broadcast_in_dim %arg1391, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1975 = stablehlo.convert %1972 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %1976 = stablehlo.broadcast_in_dim %1973, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1977 = stablehlo.subtract %1975, %1976 : tensor<256x14x14x1024xf32>
    %1978 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %1979 = stablehlo.add %1974, %1978 : tensor<1x1x1x1024xf32>
    %1980 = stablehlo.rsqrt %1979 : tensor<1x1x1x1024xf32>
    %1981 = stablehlo.reshape %arg249 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1982 = stablehlo.multiply %1980, %1981 : tensor<1x1x1x1024xf32>
    %1983 = stablehlo.broadcast_in_dim %1982, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1984 = stablehlo.multiply %1977, %1983 : tensor<256x14x14x1024xf32>
    %1985 = stablehlo.reshape %arg248 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %1986 = stablehlo.broadcast_in_dim %1985, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %1987 = stablehlo.add %1984, %1986 : tensor<256x14x14x1024xf32>
    %1988 = stablehlo.convert %1987 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %1989 = stablehlo.add %1932, %1988 : tensor<256x14x14x1024xbf16>
    %1990 = call @relu_193(%1989) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %1991 = stablehlo.convert %arg259 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %1992 = stablehlo.convolution(%1990, %1991) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1993 = stablehlo.broadcast_in_dim %arg1392, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1994 = stablehlo.broadcast_in_dim %arg1393, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1995 = stablehlo.convert %1992 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1996 = stablehlo.broadcast_in_dim %1993, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1997 = stablehlo.subtract %1995, %1996 : tensor<256x14x14x256xf32>
    %1998 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1999 = stablehlo.add %1994, %1998 : tensor<1x1x1x256xf32>
    %2000 = stablehlo.rsqrt %1999 : tensor<1x1x1x256xf32>
    %2001 = stablehlo.reshape %arg254 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2002 = stablehlo.multiply %2000, %2001 : tensor<1x1x1x256xf32>
    %2003 = stablehlo.broadcast_in_dim %2002, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2004 = stablehlo.multiply %1997, %2003 : tensor<256x14x14x256xf32>
    %2005 = stablehlo.reshape %arg253 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2006 = stablehlo.broadcast_in_dim %2005, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2007 = stablehlo.add %2004, %2006 : tensor<256x14x14x256xf32>
    %2008 = stablehlo.convert %2007 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2009 = call @relu_175(%2008) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2010 = stablehlo.convert %arg260 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2011 = stablehlo.convolution(%2009, %2010) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2012 = stablehlo.broadcast_in_dim %arg1394, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2013 = stablehlo.broadcast_in_dim %arg1395, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2014 = stablehlo.convert %2011 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2015 = stablehlo.broadcast_in_dim %2012, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2016 = stablehlo.subtract %2014, %2015 : tensor<256x14x14x256xf32>
    %2017 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2018 = stablehlo.add %2013, %2017 : tensor<1x1x1x256xf32>
    %2019 = stablehlo.rsqrt %2018 : tensor<1x1x1x256xf32>
    %2020 = stablehlo.reshape %arg256 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2021 = stablehlo.multiply %2019, %2020 : tensor<1x1x1x256xf32>
    %2022 = stablehlo.broadcast_in_dim %2021, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2023 = stablehlo.multiply %2016, %2022 : tensor<256x14x14x256xf32>
    %2024 = stablehlo.reshape %arg255 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2025 = stablehlo.broadcast_in_dim %2024, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2026 = stablehlo.add %2023, %2025 : tensor<256x14x14x256xf32>
    %2027 = stablehlo.convert %2026 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2028 = call @relu_175(%2027) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2029 = stablehlo.convert %arg261 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2030 = stablehlo.convolution(%2028, %2029) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2031 = stablehlo.broadcast_in_dim %arg1396, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2032 = stablehlo.broadcast_in_dim %arg1397, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2033 = stablehlo.convert %2030 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2034 = stablehlo.broadcast_in_dim %2031, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2035 = stablehlo.subtract %2033, %2034 : tensor<256x14x14x1024xf32>
    %2036 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2037 = stablehlo.add %2032, %2036 : tensor<1x1x1x1024xf32>
    %2038 = stablehlo.rsqrt %2037 : tensor<1x1x1x1024xf32>
    %2039 = stablehlo.reshape %arg258 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2040 = stablehlo.multiply %2038, %2039 : tensor<1x1x1x1024xf32>
    %2041 = stablehlo.broadcast_in_dim %2040, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2042 = stablehlo.multiply %2035, %2041 : tensor<256x14x14x1024xf32>
    %2043 = stablehlo.reshape %arg257 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2044 = stablehlo.broadcast_in_dim %2043, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2045 = stablehlo.add %2042, %2044 : tensor<256x14x14x1024xf32>
    %2046 = stablehlo.convert %2045 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2047 = stablehlo.add %1990, %2046 : tensor<256x14x14x1024xbf16>
    %2048 = call @relu_193(%2047) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2049 = stablehlo.convert %arg268 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2050 = stablehlo.convolution(%2048, %2049) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2051 = stablehlo.broadcast_in_dim %arg1398, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2052 = stablehlo.broadcast_in_dim %arg1399, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2053 = stablehlo.convert %2050 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2054 = stablehlo.broadcast_in_dim %2051, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2055 = stablehlo.subtract %2053, %2054 : tensor<256x14x14x256xf32>
    %2056 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2057 = stablehlo.add %2052, %2056 : tensor<1x1x1x256xf32>
    %2058 = stablehlo.rsqrt %2057 : tensor<1x1x1x256xf32>
    %2059 = stablehlo.reshape %arg263 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2060 = stablehlo.multiply %2058, %2059 : tensor<1x1x1x256xf32>
    %2061 = stablehlo.broadcast_in_dim %2060, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2062 = stablehlo.multiply %2055, %2061 : tensor<256x14x14x256xf32>
    %2063 = stablehlo.reshape %arg262 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2064 = stablehlo.broadcast_in_dim %2063, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2065 = stablehlo.add %2062, %2064 : tensor<256x14x14x256xf32>
    %2066 = stablehlo.convert %2065 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2067 = call @relu_175(%2066) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2068 = stablehlo.convert %arg269 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2069 = stablehlo.convolution(%2067, %2068) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2070 = stablehlo.broadcast_in_dim %arg1400, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2071 = stablehlo.broadcast_in_dim %arg1401, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2072 = stablehlo.convert %2069 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2073 = stablehlo.broadcast_in_dim %2070, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2074 = stablehlo.subtract %2072, %2073 : tensor<256x14x14x256xf32>
    %2075 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2076 = stablehlo.add %2071, %2075 : tensor<1x1x1x256xf32>
    %2077 = stablehlo.rsqrt %2076 : tensor<1x1x1x256xf32>
    %2078 = stablehlo.reshape %arg265 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2079 = stablehlo.multiply %2077, %2078 : tensor<1x1x1x256xf32>
    %2080 = stablehlo.broadcast_in_dim %2079, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2081 = stablehlo.multiply %2074, %2080 : tensor<256x14x14x256xf32>
    %2082 = stablehlo.reshape %arg264 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2083 = stablehlo.broadcast_in_dim %2082, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2084 = stablehlo.add %2081, %2083 : tensor<256x14x14x256xf32>
    %2085 = stablehlo.convert %2084 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2086 = call @relu_175(%2085) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2087 = stablehlo.convert %arg270 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2088 = stablehlo.convolution(%2086, %2087) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2089 = stablehlo.broadcast_in_dim %arg1402, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2090 = stablehlo.broadcast_in_dim %arg1403, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2091 = stablehlo.convert %2088 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2092 = stablehlo.broadcast_in_dim %2089, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2093 = stablehlo.subtract %2091, %2092 : tensor<256x14x14x1024xf32>
    %2094 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2095 = stablehlo.add %2090, %2094 : tensor<1x1x1x1024xf32>
    %2096 = stablehlo.rsqrt %2095 : tensor<1x1x1x1024xf32>
    %2097 = stablehlo.reshape %arg267 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2098 = stablehlo.multiply %2096, %2097 : tensor<1x1x1x1024xf32>
    %2099 = stablehlo.broadcast_in_dim %2098, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2100 = stablehlo.multiply %2093, %2099 : tensor<256x14x14x1024xf32>
    %2101 = stablehlo.reshape %arg266 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2102 = stablehlo.broadcast_in_dim %2101, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2103 = stablehlo.add %2100, %2102 : tensor<256x14x14x1024xf32>
    %2104 = stablehlo.convert %2103 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2105 = stablehlo.add %2048, %2104 : tensor<256x14x14x1024xbf16>
    %2106 = call @relu_193(%2105) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2107 = stablehlo.convert %arg277 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2108 = stablehlo.convolution(%2106, %2107) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2109 = stablehlo.broadcast_in_dim %arg1404, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2110 = stablehlo.broadcast_in_dim %arg1405, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2111 = stablehlo.convert %2108 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2112 = stablehlo.broadcast_in_dim %2109, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2113 = stablehlo.subtract %2111, %2112 : tensor<256x14x14x256xf32>
    %2114 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2115 = stablehlo.add %2110, %2114 : tensor<1x1x1x256xf32>
    %2116 = stablehlo.rsqrt %2115 : tensor<1x1x1x256xf32>
    %2117 = stablehlo.reshape %arg272 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2118 = stablehlo.multiply %2116, %2117 : tensor<1x1x1x256xf32>
    %2119 = stablehlo.broadcast_in_dim %2118, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2120 = stablehlo.multiply %2113, %2119 : tensor<256x14x14x256xf32>
    %2121 = stablehlo.reshape %arg271 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2122 = stablehlo.broadcast_in_dim %2121, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2123 = stablehlo.add %2120, %2122 : tensor<256x14x14x256xf32>
    %2124 = stablehlo.convert %2123 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2125 = call @relu_175(%2124) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2126 = stablehlo.convert %arg278 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2127 = stablehlo.convolution(%2125, %2126) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2128 = stablehlo.broadcast_in_dim %arg1406, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2129 = stablehlo.broadcast_in_dim %arg1407, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2130 = stablehlo.convert %2127 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2131 = stablehlo.broadcast_in_dim %2128, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2132 = stablehlo.subtract %2130, %2131 : tensor<256x14x14x256xf32>
    %2133 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2134 = stablehlo.add %2129, %2133 : tensor<1x1x1x256xf32>
    %2135 = stablehlo.rsqrt %2134 : tensor<1x1x1x256xf32>
    %2136 = stablehlo.reshape %arg274 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2137 = stablehlo.multiply %2135, %2136 : tensor<1x1x1x256xf32>
    %2138 = stablehlo.broadcast_in_dim %2137, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2139 = stablehlo.multiply %2132, %2138 : tensor<256x14x14x256xf32>
    %2140 = stablehlo.reshape %arg273 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2141 = stablehlo.broadcast_in_dim %2140, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2142 = stablehlo.add %2139, %2141 : tensor<256x14x14x256xf32>
    %2143 = stablehlo.convert %2142 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2144 = call @relu_175(%2143) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2145 = stablehlo.convert %arg279 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2146 = stablehlo.convolution(%2144, %2145) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2147 = stablehlo.broadcast_in_dim %arg1408, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2148 = stablehlo.broadcast_in_dim %arg1409, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2149 = stablehlo.convert %2146 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2150 = stablehlo.broadcast_in_dim %2147, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2151 = stablehlo.subtract %2149, %2150 : tensor<256x14x14x1024xf32>
    %2152 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2153 = stablehlo.add %2148, %2152 : tensor<1x1x1x1024xf32>
    %2154 = stablehlo.rsqrt %2153 : tensor<1x1x1x1024xf32>
    %2155 = stablehlo.reshape %arg276 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2156 = stablehlo.multiply %2154, %2155 : tensor<1x1x1x1024xf32>
    %2157 = stablehlo.broadcast_in_dim %2156, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2158 = stablehlo.multiply %2151, %2157 : tensor<256x14x14x1024xf32>
    %2159 = stablehlo.reshape %arg275 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2160 = stablehlo.broadcast_in_dim %2159, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2161 = stablehlo.add %2158, %2160 : tensor<256x14x14x1024xf32>
    %2162 = stablehlo.convert %2161 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2163 = stablehlo.add %2106, %2162 : tensor<256x14x14x1024xbf16>
    %2164 = call @relu_193(%2163) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2165 = stablehlo.convert %arg286 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2166 = stablehlo.convolution(%2164, %2165) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2167 = stablehlo.broadcast_in_dim %arg1410, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2168 = stablehlo.broadcast_in_dim %arg1411, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2169 = stablehlo.convert %2166 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2170 = stablehlo.broadcast_in_dim %2167, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2171 = stablehlo.subtract %2169, %2170 : tensor<256x14x14x256xf32>
    %2172 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2173 = stablehlo.add %2168, %2172 : tensor<1x1x1x256xf32>
    %2174 = stablehlo.rsqrt %2173 : tensor<1x1x1x256xf32>
    %2175 = stablehlo.reshape %arg281 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2176 = stablehlo.multiply %2174, %2175 : tensor<1x1x1x256xf32>
    %2177 = stablehlo.broadcast_in_dim %2176, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2178 = stablehlo.multiply %2171, %2177 : tensor<256x14x14x256xf32>
    %2179 = stablehlo.reshape %arg280 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2180 = stablehlo.broadcast_in_dim %2179, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2181 = stablehlo.add %2178, %2180 : tensor<256x14x14x256xf32>
    %2182 = stablehlo.convert %2181 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2183 = call @relu_175(%2182) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2184 = stablehlo.convert %arg287 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2185 = stablehlo.convolution(%2183, %2184) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2186 = stablehlo.broadcast_in_dim %arg1412, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2187 = stablehlo.broadcast_in_dim %arg1413, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2188 = stablehlo.convert %2185 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2189 = stablehlo.broadcast_in_dim %2186, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2190 = stablehlo.subtract %2188, %2189 : tensor<256x14x14x256xf32>
    %2191 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2192 = stablehlo.add %2187, %2191 : tensor<1x1x1x256xf32>
    %2193 = stablehlo.rsqrt %2192 : tensor<1x1x1x256xf32>
    %2194 = stablehlo.reshape %arg283 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2195 = stablehlo.multiply %2193, %2194 : tensor<1x1x1x256xf32>
    %2196 = stablehlo.broadcast_in_dim %2195, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2197 = stablehlo.multiply %2190, %2196 : tensor<256x14x14x256xf32>
    %2198 = stablehlo.reshape %arg282 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2199 = stablehlo.broadcast_in_dim %2198, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2200 = stablehlo.add %2197, %2199 : tensor<256x14x14x256xf32>
    %2201 = stablehlo.convert %2200 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2202 = call @relu_175(%2201) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2203 = stablehlo.convert %arg288 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2204 = stablehlo.convolution(%2202, %2203) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2205 = stablehlo.broadcast_in_dim %arg1414, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2206 = stablehlo.broadcast_in_dim %arg1415, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2207 = stablehlo.convert %2204 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2208 = stablehlo.broadcast_in_dim %2205, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2209 = stablehlo.subtract %2207, %2208 : tensor<256x14x14x1024xf32>
    %2210 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2211 = stablehlo.add %2206, %2210 : tensor<1x1x1x1024xf32>
    %2212 = stablehlo.rsqrt %2211 : tensor<1x1x1x1024xf32>
    %2213 = stablehlo.reshape %arg285 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2214 = stablehlo.multiply %2212, %2213 : tensor<1x1x1x1024xf32>
    %2215 = stablehlo.broadcast_in_dim %2214, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2216 = stablehlo.multiply %2209, %2215 : tensor<256x14x14x1024xf32>
    %2217 = stablehlo.reshape %arg284 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2218 = stablehlo.broadcast_in_dim %2217, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2219 = stablehlo.add %2216, %2218 : tensor<256x14x14x1024xf32>
    %2220 = stablehlo.convert %2219 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2221 = stablehlo.add %2164, %2220 : tensor<256x14x14x1024xbf16>
    %2222 = call @relu_193(%2221) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2223 = stablehlo.convert %arg295 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2224 = stablehlo.convolution(%2222, %2223) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2225 = stablehlo.broadcast_in_dim %arg1416, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2226 = stablehlo.broadcast_in_dim %arg1417, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2227 = stablehlo.convert %2224 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2228 = stablehlo.broadcast_in_dim %2225, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2229 = stablehlo.subtract %2227, %2228 : tensor<256x14x14x256xf32>
    %2230 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2231 = stablehlo.add %2226, %2230 : tensor<1x1x1x256xf32>
    %2232 = stablehlo.rsqrt %2231 : tensor<1x1x1x256xf32>
    %2233 = stablehlo.reshape %arg290 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2234 = stablehlo.multiply %2232, %2233 : tensor<1x1x1x256xf32>
    %2235 = stablehlo.broadcast_in_dim %2234, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2236 = stablehlo.multiply %2229, %2235 : tensor<256x14x14x256xf32>
    %2237 = stablehlo.reshape %arg289 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2238 = stablehlo.broadcast_in_dim %2237, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2239 = stablehlo.add %2236, %2238 : tensor<256x14x14x256xf32>
    %2240 = stablehlo.convert %2239 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2241 = call @relu_175(%2240) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2242 = stablehlo.convert %arg296 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2243 = stablehlo.convolution(%2241, %2242) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2244 = stablehlo.broadcast_in_dim %arg1418, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2245 = stablehlo.broadcast_in_dim %arg1419, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2246 = stablehlo.convert %2243 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2247 = stablehlo.broadcast_in_dim %2244, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2248 = stablehlo.subtract %2246, %2247 : tensor<256x14x14x256xf32>
    %2249 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2250 = stablehlo.add %2245, %2249 : tensor<1x1x1x256xf32>
    %2251 = stablehlo.rsqrt %2250 : tensor<1x1x1x256xf32>
    %2252 = stablehlo.reshape %arg292 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2253 = stablehlo.multiply %2251, %2252 : tensor<1x1x1x256xf32>
    %2254 = stablehlo.broadcast_in_dim %2253, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2255 = stablehlo.multiply %2248, %2254 : tensor<256x14x14x256xf32>
    %2256 = stablehlo.reshape %arg291 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2257 = stablehlo.broadcast_in_dim %2256, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2258 = stablehlo.add %2255, %2257 : tensor<256x14x14x256xf32>
    %2259 = stablehlo.convert %2258 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2260 = call @relu_175(%2259) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2261 = stablehlo.convert %arg297 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2262 = stablehlo.convolution(%2260, %2261) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2263 = stablehlo.broadcast_in_dim %arg1420, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2264 = stablehlo.broadcast_in_dim %arg1421, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2265 = stablehlo.convert %2262 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2266 = stablehlo.broadcast_in_dim %2263, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2267 = stablehlo.subtract %2265, %2266 : tensor<256x14x14x1024xf32>
    %2268 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2269 = stablehlo.add %2264, %2268 : tensor<1x1x1x1024xf32>
    %2270 = stablehlo.rsqrt %2269 : tensor<1x1x1x1024xf32>
    %2271 = stablehlo.reshape %arg294 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2272 = stablehlo.multiply %2270, %2271 : tensor<1x1x1x1024xf32>
    %2273 = stablehlo.broadcast_in_dim %2272, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2274 = stablehlo.multiply %2267, %2273 : tensor<256x14x14x1024xf32>
    %2275 = stablehlo.reshape %arg293 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2276 = stablehlo.broadcast_in_dim %2275, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2277 = stablehlo.add %2274, %2276 : tensor<256x14x14x1024xf32>
    %2278 = stablehlo.convert %2277 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2279 = stablehlo.add %2222, %2278 : tensor<256x14x14x1024xbf16>
    %2280 = call @relu_193(%2279) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2281 = stablehlo.convert %arg304 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2282 = stablehlo.convolution(%2280, %2281) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2283 = stablehlo.broadcast_in_dim %arg1422, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2284 = stablehlo.broadcast_in_dim %arg1423, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2285 = stablehlo.convert %2282 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2286 = stablehlo.broadcast_in_dim %2283, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2287 = stablehlo.subtract %2285, %2286 : tensor<256x14x14x256xf32>
    %2288 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2289 = stablehlo.add %2284, %2288 : tensor<1x1x1x256xf32>
    %2290 = stablehlo.rsqrt %2289 : tensor<1x1x1x256xf32>
    %2291 = stablehlo.reshape %arg299 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2292 = stablehlo.multiply %2290, %2291 : tensor<1x1x1x256xf32>
    %2293 = stablehlo.broadcast_in_dim %2292, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2294 = stablehlo.multiply %2287, %2293 : tensor<256x14x14x256xf32>
    %2295 = stablehlo.reshape %arg298 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2296 = stablehlo.broadcast_in_dim %2295, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2297 = stablehlo.add %2294, %2296 : tensor<256x14x14x256xf32>
    %2298 = stablehlo.convert %2297 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2299 = call @relu_175(%2298) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2300 = stablehlo.convert %arg305 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2301 = stablehlo.convolution(%2299, %2300) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2302 = stablehlo.broadcast_in_dim %arg1424, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2303 = stablehlo.broadcast_in_dim %arg1425, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2304 = stablehlo.convert %2301 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2305 = stablehlo.broadcast_in_dim %2302, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2306 = stablehlo.subtract %2304, %2305 : tensor<256x14x14x256xf32>
    %2307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2308 = stablehlo.add %2303, %2307 : tensor<1x1x1x256xf32>
    %2309 = stablehlo.rsqrt %2308 : tensor<1x1x1x256xf32>
    %2310 = stablehlo.reshape %arg301 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2311 = stablehlo.multiply %2309, %2310 : tensor<1x1x1x256xf32>
    %2312 = stablehlo.broadcast_in_dim %2311, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2313 = stablehlo.multiply %2306, %2312 : tensor<256x14x14x256xf32>
    %2314 = stablehlo.reshape %arg300 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2315 = stablehlo.broadcast_in_dim %2314, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2316 = stablehlo.add %2313, %2315 : tensor<256x14x14x256xf32>
    %2317 = stablehlo.convert %2316 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2318 = call @relu_175(%2317) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2319 = stablehlo.convert %arg306 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2320 = stablehlo.convolution(%2318, %2319) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2321 = stablehlo.broadcast_in_dim %arg1426, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2322 = stablehlo.broadcast_in_dim %arg1427, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2323 = stablehlo.convert %2320 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2324 = stablehlo.broadcast_in_dim %2321, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2325 = stablehlo.subtract %2323, %2324 : tensor<256x14x14x1024xf32>
    %2326 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2327 = stablehlo.add %2322, %2326 : tensor<1x1x1x1024xf32>
    %2328 = stablehlo.rsqrt %2327 : tensor<1x1x1x1024xf32>
    %2329 = stablehlo.reshape %arg303 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2330 = stablehlo.multiply %2328, %2329 : tensor<1x1x1x1024xf32>
    %2331 = stablehlo.broadcast_in_dim %2330, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2332 = stablehlo.multiply %2325, %2331 : tensor<256x14x14x1024xf32>
    %2333 = stablehlo.reshape %arg302 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2334 = stablehlo.broadcast_in_dim %2333, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2335 = stablehlo.add %2332, %2334 : tensor<256x14x14x1024xf32>
    %2336 = stablehlo.convert %2335 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2337 = stablehlo.add %2280, %2336 : tensor<256x14x14x1024xbf16>
    %2338 = call @relu_193(%2337) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2339 = stablehlo.convert %arg313 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2340 = stablehlo.convolution(%2338, %2339) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2341 = stablehlo.broadcast_in_dim %arg1428, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2342 = stablehlo.broadcast_in_dim %arg1429, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2343 = stablehlo.convert %2340 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2344 = stablehlo.broadcast_in_dim %2341, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2345 = stablehlo.subtract %2343, %2344 : tensor<256x14x14x256xf32>
    %2346 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2347 = stablehlo.add %2342, %2346 : tensor<1x1x1x256xf32>
    %2348 = stablehlo.rsqrt %2347 : tensor<1x1x1x256xf32>
    %2349 = stablehlo.reshape %arg308 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2350 = stablehlo.multiply %2348, %2349 : tensor<1x1x1x256xf32>
    %2351 = stablehlo.broadcast_in_dim %2350, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2352 = stablehlo.multiply %2345, %2351 : tensor<256x14x14x256xf32>
    %2353 = stablehlo.reshape %arg307 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2354 = stablehlo.broadcast_in_dim %2353, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2355 = stablehlo.add %2352, %2354 : tensor<256x14x14x256xf32>
    %2356 = stablehlo.convert %2355 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2357 = call @relu_175(%2356) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2358 = stablehlo.convert %arg314 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2359 = stablehlo.convolution(%2357, %2358) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2360 = stablehlo.broadcast_in_dim %arg1430, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2361 = stablehlo.broadcast_in_dim %arg1431, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2362 = stablehlo.convert %2359 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2363 = stablehlo.broadcast_in_dim %2360, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2364 = stablehlo.subtract %2362, %2363 : tensor<256x14x14x256xf32>
    %2365 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2366 = stablehlo.add %2361, %2365 : tensor<1x1x1x256xf32>
    %2367 = stablehlo.rsqrt %2366 : tensor<1x1x1x256xf32>
    %2368 = stablehlo.reshape %arg310 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2369 = stablehlo.multiply %2367, %2368 : tensor<1x1x1x256xf32>
    %2370 = stablehlo.broadcast_in_dim %2369, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2371 = stablehlo.multiply %2364, %2370 : tensor<256x14x14x256xf32>
    %2372 = stablehlo.reshape %arg309 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2373 = stablehlo.broadcast_in_dim %2372, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2374 = stablehlo.add %2371, %2373 : tensor<256x14x14x256xf32>
    %2375 = stablehlo.convert %2374 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2376 = call @relu_175(%2375) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2377 = stablehlo.convert %arg315 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2378 = stablehlo.convolution(%2376, %2377) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2379 = stablehlo.broadcast_in_dim %arg1432, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2380 = stablehlo.broadcast_in_dim %arg1433, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2381 = stablehlo.convert %2378 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2382 = stablehlo.broadcast_in_dim %2379, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2383 = stablehlo.subtract %2381, %2382 : tensor<256x14x14x1024xf32>
    %2384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2385 = stablehlo.add %2380, %2384 : tensor<1x1x1x1024xf32>
    %2386 = stablehlo.rsqrt %2385 : tensor<1x1x1x1024xf32>
    %2387 = stablehlo.reshape %arg312 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2388 = stablehlo.multiply %2386, %2387 : tensor<1x1x1x1024xf32>
    %2389 = stablehlo.broadcast_in_dim %2388, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2390 = stablehlo.multiply %2383, %2389 : tensor<256x14x14x1024xf32>
    %2391 = stablehlo.reshape %arg311 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2392 = stablehlo.broadcast_in_dim %2391, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2393 = stablehlo.add %2390, %2392 : tensor<256x14x14x1024xf32>
    %2394 = stablehlo.convert %2393 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2395 = stablehlo.add %2338, %2394 : tensor<256x14x14x1024xbf16>
    %2396 = call @relu_193(%2395) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2397 = stablehlo.convert %arg331 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2398 = stablehlo.convolution(%2396, %2397) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2399 = stablehlo.broadcast_in_dim %arg1440, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2400 = stablehlo.broadcast_in_dim %arg1441, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2401 = stablehlo.convert %2398 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2402 = stablehlo.broadcast_in_dim %2399, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2403 = stablehlo.subtract %2401, %2402 : tensor<256x14x14x256xf32>
    %2404 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2405 = stablehlo.add %2400, %2404 : tensor<1x1x1x256xf32>
    %2406 = stablehlo.rsqrt %2405 : tensor<1x1x1x256xf32>
    %2407 = stablehlo.reshape %arg326 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2408 = stablehlo.multiply %2406, %2407 : tensor<1x1x1x256xf32>
    %2409 = stablehlo.broadcast_in_dim %2408, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2410 = stablehlo.multiply %2403, %2409 : tensor<256x14x14x256xf32>
    %2411 = stablehlo.reshape %arg325 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2412 = stablehlo.broadcast_in_dim %2411, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2413 = stablehlo.add %2410, %2412 : tensor<256x14x14x256xf32>
    %2414 = stablehlo.convert %2413 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2415 = call @relu_175(%2414) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2416 = stablehlo.convert %arg332 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2417 = stablehlo.convolution(%2415, %2416) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2418 = stablehlo.broadcast_in_dim %arg1442, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2419 = stablehlo.broadcast_in_dim %arg1443, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2420 = stablehlo.convert %2417 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2421 = stablehlo.broadcast_in_dim %2418, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2422 = stablehlo.subtract %2420, %2421 : tensor<256x14x14x256xf32>
    %2423 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2424 = stablehlo.add %2419, %2423 : tensor<1x1x1x256xf32>
    %2425 = stablehlo.rsqrt %2424 : tensor<1x1x1x256xf32>
    %2426 = stablehlo.reshape %arg328 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2427 = stablehlo.multiply %2425, %2426 : tensor<1x1x1x256xf32>
    %2428 = stablehlo.broadcast_in_dim %2427, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2429 = stablehlo.multiply %2422, %2428 : tensor<256x14x14x256xf32>
    %2430 = stablehlo.reshape %arg327 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2431 = stablehlo.broadcast_in_dim %2430, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2432 = stablehlo.add %2429, %2431 : tensor<256x14x14x256xf32>
    %2433 = stablehlo.convert %2432 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2434 = call @relu_175(%2433) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2435 = stablehlo.convert %arg333 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2436 = stablehlo.convolution(%2434, %2435) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2437 = stablehlo.broadcast_in_dim %arg1444, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2438 = stablehlo.broadcast_in_dim %arg1445, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2439 = stablehlo.convert %2436 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2440 = stablehlo.broadcast_in_dim %2437, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2441 = stablehlo.subtract %2439, %2440 : tensor<256x14x14x1024xf32>
    %2442 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2443 = stablehlo.add %2438, %2442 : tensor<1x1x1x1024xf32>
    %2444 = stablehlo.rsqrt %2443 : tensor<1x1x1x1024xf32>
    %2445 = stablehlo.reshape %arg330 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2446 = stablehlo.multiply %2444, %2445 : tensor<1x1x1x1024xf32>
    %2447 = stablehlo.broadcast_in_dim %2446, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2448 = stablehlo.multiply %2441, %2447 : tensor<256x14x14x1024xf32>
    %2449 = stablehlo.reshape %arg329 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2450 = stablehlo.broadcast_in_dim %2449, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2451 = stablehlo.add %2448, %2450 : tensor<256x14x14x1024xf32>
    %2452 = stablehlo.convert %2451 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2453 = stablehlo.add %2396, %2452 : tensor<256x14x14x1024xbf16>
    %2454 = call @relu_193(%2453) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2455 = stablehlo.convert %arg340 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2456 = stablehlo.convolution(%2454, %2455) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2457 = stablehlo.broadcast_in_dim %arg1446, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2458 = stablehlo.broadcast_in_dim %arg1447, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2459 = stablehlo.convert %2456 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2460 = stablehlo.broadcast_in_dim %2457, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2461 = stablehlo.subtract %2459, %2460 : tensor<256x14x14x256xf32>
    %2462 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2463 = stablehlo.add %2458, %2462 : tensor<1x1x1x256xf32>
    %2464 = stablehlo.rsqrt %2463 : tensor<1x1x1x256xf32>
    %2465 = stablehlo.reshape %arg335 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2466 = stablehlo.multiply %2464, %2465 : tensor<1x1x1x256xf32>
    %2467 = stablehlo.broadcast_in_dim %2466, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2468 = stablehlo.multiply %2461, %2467 : tensor<256x14x14x256xf32>
    %2469 = stablehlo.reshape %arg334 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2470 = stablehlo.broadcast_in_dim %2469, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2471 = stablehlo.add %2468, %2470 : tensor<256x14x14x256xf32>
    %2472 = stablehlo.convert %2471 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2473 = call @relu_175(%2472) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2474 = stablehlo.convert %arg341 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2475 = stablehlo.convolution(%2473, %2474) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2476 = stablehlo.broadcast_in_dim %arg1448, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2477 = stablehlo.broadcast_in_dim %arg1449, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2478 = stablehlo.convert %2475 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2479 = stablehlo.broadcast_in_dim %2476, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2480 = stablehlo.subtract %2478, %2479 : tensor<256x14x14x256xf32>
    %2481 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2482 = stablehlo.add %2477, %2481 : tensor<1x1x1x256xf32>
    %2483 = stablehlo.rsqrt %2482 : tensor<1x1x1x256xf32>
    %2484 = stablehlo.reshape %arg337 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2485 = stablehlo.multiply %2483, %2484 : tensor<1x1x1x256xf32>
    %2486 = stablehlo.broadcast_in_dim %2485, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2487 = stablehlo.multiply %2480, %2486 : tensor<256x14x14x256xf32>
    %2488 = stablehlo.reshape %arg336 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2489 = stablehlo.broadcast_in_dim %2488, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2490 = stablehlo.add %2487, %2489 : tensor<256x14x14x256xf32>
    %2491 = stablehlo.convert %2490 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2492 = call @relu_175(%2491) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2493 = stablehlo.convert %arg342 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2494 = stablehlo.convolution(%2492, %2493) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2495 = stablehlo.broadcast_in_dim %arg1450, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2496 = stablehlo.broadcast_in_dim %arg1451, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2497 = stablehlo.convert %2494 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2498 = stablehlo.broadcast_in_dim %2495, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2499 = stablehlo.subtract %2497, %2498 : tensor<256x14x14x1024xf32>
    %2500 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2501 = stablehlo.add %2496, %2500 : tensor<1x1x1x1024xf32>
    %2502 = stablehlo.rsqrt %2501 : tensor<1x1x1x1024xf32>
    %2503 = stablehlo.reshape %arg339 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2504 = stablehlo.multiply %2502, %2503 : tensor<1x1x1x1024xf32>
    %2505 = stablehlo.broadcast_in_dim %2504, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2506 = stablehlo.multiply %2499, %2505 : tensor<256x14x14x1024xf32>
    %2507 = stablehlo.reshape %arg338 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2508 = stablehlo.broadcast_in_dim %2507, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2509 = stablehlo.add %2506, %2508 : tensor<256x14x14x1024xf32>
    %2510 = stablehlo.convert %2509 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2511 = stablehlo.add %2454, %2510 : tensor<256x14x14x1024xbf16>
    %2512 = call @relu_193(%2511) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2513 = stablehlo.convert %arg349 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2514 = stablehlo.convolution(%2512, %2513) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2515 = stablehlo.broadcast_in_dim %arg1452, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2516 = stablehlo.broadcast_in_dim %arg1453, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2517 = stablehlo.convert %2514 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2518 = stablehlo.broadcast_in_dim %2515, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2519 = stablehlo.subtract %2517, %2518 : tensor<256x14x14x256xf32>
    %2520 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2521 = stablehlo.add %2516, %2520 : tensor<1x1x1x256xf32>
    %2522 = stablehlo.rsqrt %2521 : tensor<1x1x1x256xf32>
    %2523 = stablehlo.reshape %arg344 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2524 = stablehlo.multiply %2522, %2523 : tensor<1x1x1x256xf32>
    %2525 = stablehlo.broadcast_in_dim %2524, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2526 = stablehlo.multiply %2519, %2525 : tensor<256x14x14x256xf32>
    %2527 = stablehlo.reshape %arg343 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2528 = stablehlo.broadcast_in_dim %2527, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2529 = stablehlo.add %2526, %2528 : tensor<256x14x14x256xf32>
    %2530 = stablehlo.convert %2529 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2531 = call @relu_175(%2530) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2532 = stablehlo.convert %arg350 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2533 = stablehlo.convolution(%2531, %2532) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2534 = stablehlo.broadcast_in_dim %arg1454, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2535 = stablehlo.broadcast_in_dim %arg1455, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2536 = stablehlo.convert %2533 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2537 = stablehlo.broadcast_in_dim %2534, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2538 = stablehlo.subtract %2536, %2537 : tensor<256x14x14x256xf32>
    %2539 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2540 = stablehlo.add %2535, %2539 : tensor<1x1x1x256xf32>
    %2541 = stablehlo.rsqrt %2540 : tensor<1x1x1x256xf32>
    %2542 = stablehlo.reshape %arg346 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2543 = stablehlo.multiply %2541, %2542 : tensor<1x1x1x256xf32>
    %2544 = stablehlo.broadcast_in_dim %2543, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2545 = stablehlo.multiply %2538, %2544 : tensor<256x14x14x256xf32>
    %2546 = stablehlo.reshape %arg345 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2547 = stablehlo.broadcast_in_dim %2546, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2548 = stablehlo.add %2545, %2547 : tensor<256x14x14x256xf32>
    %2549 = stablehlo.convert %2548 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2550 = call @relu_175(%2549) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2551 = stablehlo.convert %arg351 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2552 = stablehlo.convolution(%2550, %2551) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2553 = stablehlo.broadcast_in_dim %arg1456, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2554 = stablehlo.broadcast_in_dim %arg1457, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2555 = stablehlo.convert %2552 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2556 = stablehlo.broadcast_in_dim %2553, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2557 = stablehlo.subtract %2555, %2556 : tensor<256x14x14x1024xf32>
    %2558 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2559 = stablehlo.add %2554, %2558 : tensor<1x1x1x1024xf32>
    %2560 = stablehlo.rsqrt %2559 : tensor<1x1x1x1024xf32>
    %2561 = stablehlo.reshape %arg348 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2562 = stablehlo.multiply %2560, %2561 : tensor<1x1x1x1024xf32>
    %2563 = stablehlo.broadcast_in_dim %2562, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2564 = stablehlo.multiply %2557, %2563 : tensor<256x14x14x1024xf32>
    %2565 = stablehlo.reshape %arg347 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2566 = stablehlo.broadcast_in_dim %2565, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2567 = stablehlo.add %2564, %2566 : tensor<256x14x14x1024xf32>
    %2568 = stablehlo.convert %2567 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2569 = stablehlo.add %2512, %2568 : tensor<256x14x14x1024xbf16>
    %2570 = call @relu_193(%2569) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2571 = stablehlo.convert %arg358 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2572 = stablehlo.convolution(%2570, %2571) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2573 = stablehlo.broadcast_in_dim %arg1458, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2574 = stablehlo.broadcast_in_dim %arg1459, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2575 = stablehlo.convert %2572 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2576 = stablehlo.broadcast_in_dim %2573, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2577 = stablehlo.subtract %2575, %2576 : tensor<256x14x14x256xf32>
    %2578 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2579 = stablehlo.add %2574, %2578 : tensor<1x1x1x256xf32>
    %2580 = stablehlo.rsqrt %2579 : tensor<1x1x1x256xf32>
    %2581 = stablehlo.reshape %arg353 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2582 = stablehlo.multiply %2580, %2581 : tensor<1x1x1x256xf32>
    %2583 = stablehlo.broadcast_in_dim %2582, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2584 = stablehlo.multiply %2577, %2583 : tensor<256x14x14x256xf32>
    %2585 = stablehlo.reshape %arg352 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2586 = stablehlo.broadcast_in_dim %2585, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2587 = stablehlo.add %2584, %2586 : tensor<256x14x14x256xf32>
    %2588 = stablehlo.convert %2587 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2589 = call @relu_175(%2588) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2590 = stablehlo.convert %arg359 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2591 = stablehlo.convolution(%2589, %2590) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2592 = stablehlo.broadcast_in_dim %arg1460, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2593 = stablehlo.broadcast_in_dim %arg1461, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2594 = stablehlo.convert %2591 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2595 = stablehlo.broadcast_in_dim %2592, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2596 = stablehlo.subtract %2594, %2595 : tensor<256x14x14x256xf32>
    %2597 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2598 = stablehlo.add %2593, %2597 : tensor<1x1x1x256xf32>
    %2599 = stablehlo.rsqrt %2598 : tensor<1x1x1x256xf32>
    %2600 = stablehlo.reshape %arg355 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2601 = stablehlo.multiply %2599, %2600 : tensor<1x1x1x256xf32>
    %2602 = stablehlo.broadcast_in_dim %2601, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2603 = stablehlo.multiply %2596, %2602 : tensor<256x14x14x256xf32>
    %2604 = stablehlo.reshape %arg354 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2605 = stablehlo.broadcast_in_dim %2604, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2606 = stablehlo.add %2603, %2605 : tensor<256x14x14x256xf32>
    %2607 = stablehlo.convert %2606 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2608 = call @relu_175(%2607) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2609 = stablehlo.convert %arg360 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2610 = stablehlo.convolution(%2608, %2609) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2611 = stablehlo.broadcast_in_dim %arg1462, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2612 = stablehlo.broadcast_in_dim %arg1463, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2613 = stablehlo.convert %2610 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2614 = stablehlo.broadcast_in_dim %2611, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2615 = stablehlo.subtract %2613, %2614 : tensor<256x14x14x1024xf32>
    %2616 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2617 = stablehlo.add %2612, %2616 : tensor<1x1x1x1024xf32>
    %2618 = stablehlo.rsqrt %2617 : tensor<1x1x1x1024xf32>
    %2619 = stablehlo.reshape %arg357 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2620 = stablehlo.multiply %2618, %2619 : tensor<1x1x1x1024xf32>
    %2621 = stablehlo.broadcast_in_dim %2620, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2622 = stablehlo.multiply %2615, %2621 : tensor<256x14x14x1024xf32>
    %2623 = stablehlo.reshape %arg356 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2624 = stablehlo.broadcast_in_dim %2623, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2625 = stablehlo.add %2622, %2624 : tensor<256x14x14x1024xf32>
    %2626 = stablehlo.convert %2625 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2627 = stablehlo.add %2570, %2626 : tensor<256x14x14x1024xbf16>
    %2628 = call @relu_193(%2627) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2629 = stablehlo.convert %arg367 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2630 = stablehlo.convolution(%2628, %2629) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2631 = stablehlo.broadcast_in_dim %arg1464, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2632 = stablehlo.broadcast_in_dim %arg1465, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2633 = stablehlo.convert %2630 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2634 = stablehlo.broadcast_in_dim %2631, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2635 = stablehlo.subtract %2633, %2634 : tensor<256x14x14x256xf32>
    %2636 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2637 = stablehlo.add %2632, %2636 : tensor<1x1x1x256xf32>
    %2638 = stablehlo.rsqrt %2637 : tensor<1x1x1x256xf32>
    %2639 = stablehlo.reshape %arg362 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2640 = stablehlo.multiply %2638, %2639 : tensor<1x1x1x256xf32>
    %2641 = stablehlo.broadcast_in_dim %2640, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2642 = stablehlo.multiply %2635, %2641 : tensor<256x14x14x256xf32>
    %2643 = stablehlo.reshape %arg361 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2644 = stablehlo.broadcast_in_dim %2643, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2645 = stablehlo.add %2642, %2644 : tensor<256x14x14x256xf32>
    %2646 = stablehlo.convert %2645 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2647 = call @relu_175(%2646) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2648 = stablehlo.convert %arg368 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2649 = stablehlo.convolution(%2647, %2648) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2650 = stablehlo.broadcast_in_dim %arg1466, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2651 = stablehlo.broadcast_in_dim %arg1467, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2652 = stablehlo.convert %2649 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2653 = stablehlo.broadcast_in_dim %2650, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2654 = stablehlo.subtract %2652, %2653 : tensor<256x14x14x256xf32>
    %2655 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2656 = stablehlo.add %2651, %2655 : tensor<1x1x1x256xf32>
    %2657 = stablehlo.rsqrt %2656 : tensor<1x1x1x256xf32>
    %2658 = stablehlo.reshape %arg364 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2659 = stablehlo.multiply %2657, %2658 : tensor<1x1x1x256xf32>
    %2660 = stablehlo.broadcast_in_dim %2659, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2661 = stablehlo.multiply %2654, %2660 : tensor<256x14x14x256xf32>
    %2662 = stablehlo.reshape %arg363 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2663 = stablehlo.broadcast_in_dim %2662, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2664 = stablehlo.add %2661, %2663 : tensor<256x14x14x256xf32>
    %2665 = stablehlo.convert %2664 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2666 = call @relu_175(%2665) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2667 = stablehlo.convert %arg369 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2668 = stablehlo.convolution(%2666, %2667) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2669 = stablehlo.broadcast_in_dim %arg1468, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2670 = stablehlo.broadcast_in_dim %arg1469, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2671 = stablehlo.convert %2668 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2672 = stablehlo.broadcast_in_dim %2669, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2673 = stablehlo.subtract %2671, %2672 : tensor<256x14x14x1024xf32>
    %2674 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2675 = stablehlo.add %2670, %2674 : tensor<1x1x1x1024xf32>
    %2676 = stablehlo.rsqrt %2675 : tensor<1x1x1x1024xf32>
    %2677 = stablehlo.reshape %arg366 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2678 = stablehlo.multiply %2676, %2677 : tensor<1x1x1x1024xf32>
    %2679 = stablehlo.broadcast_in_dim %2678, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2680 = stablehlo.multiply %2673, %2679 : tensor<256x14x14x1024xf32>
    %2681 = stablehlo.reshape %arg365 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2682 = stablehlo.broadcast_in_dim %2681, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2683 = stablehlo.add %2680, %2682 : tensor<256x14x14x1024xf32>
    %2684 = stablehlo.convert %2683 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2685 = stablehlo.add %2628, %2684 : tensor<256x14x14x1024xbf16>
    %2686 = call @relu_193(%2685) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2687 = stablehlo.convert %arg376 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2688 = stablehlo.convolution(%2686, %2687) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2689 = stablehlo.broadcast_in_dim %arg1470, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2690 = stablehlo.broadcast_in_dim %arg1471, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2691 = stablehlo.convert %2688 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2692 = stablehlo.broadcast_in_dim %2689, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2693 = stablehlo.subtract %2691, %2692 : tensor<256x14x14x256xf32>
    %2694 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2695 = stablehlo.add %2690, %2694 : tensor<1x1x1x256xf32>
    %2696 = stablehlo.rsqrt %2695 : tensor<1x1x1x256xf32>
    %2697 = stablehlo.reshape %arg371 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2698 = stablehlo.multiply %2696, %2697 : tensor<1x1x1x256xf32>
    %2699 = stablehlo.broadcast_in_dim %2698, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2700 = stablehlo.multiply %2693, %2699 : tensor<256x14x14x256xf32>
    %2701 = stablehlo.reshape %arg370 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2702 = stablehlo.broadcast_in_dim %2701, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2703 = stablehlo.add %2700, %2702 : tensor<256x14x14x256xf32>
    %2704 = stablehlo.convert %2703 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2705 = call @relu_175(%2704) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2706 = stablehlo.convert %arg377 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2707 = stablehlo.convolution(%2705, %2706) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2708 = stablehlo.broadcast_in_dim %arg1472, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2709 = stablehlo.broadcast_in_dim %arg1473, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2710 = stablehlo.convert %2707 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2711 = stablehlo.broadcast_in_dim %2708, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2712 = stablehlo.subtract %2710, %2711 : tensor<256x14x14x256xf32>
    %2713 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2714 = stablehlo.add %2709, %2713 : tensor<1x1x1x256xf32>
    %2715 = stablehlo.rsqrt %2714 : tensor<1x1x1x256xf32>
    %2716 = stablehlo.reshape %arg373 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2717 = stablehlo.multiply %2715, %2716 : tensor<1x1x1x256xf32>
    %2718 = stablehlo.broadcast_in_dim %2717, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2719 = stablehlo.multiply %2712, %2718 : tensor<256x14x14x256xf32>
    %2720 = stablehlo.reshape %arg372 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2721 = stablehlo.broadcast_in_dim %2720, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2722 = stablehlo.add %2719, %2721 : tensor<256x14x14x256xf32>
    %2723 = stablehlo.convert %2722 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2724 = call @relu_175(%2723) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2725 = stablehlo.convert %arg378 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2726 = stablehlo.convolution(%2724, %2725) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2727 = stablehlo.broadcast_in_dim %arg1474, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2728 = stablehlo.broadcast_in_dim %arg1475, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2729 = stablehlo.convert %2726 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2730 = stablehlo.broadcast_in_dim %2727, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2731 = stablehlo.subtract %2729, %2730 : tensor<256x14x14x1024xf32>
    %2732 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2733 = stablehlo.add %2728, %2732 : tensor<1x1x1x1024xf32>
    %2734 = stablehlo.rsqrt %2733 : tensor<1x1x1x1024xf32>
    %2735 = stablehlo.reshape %arg375 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2736 = stablehlo.multiply %2734, %2735 : tensor<1x1x1x1024xf32>
    %2737 = stablehlo.broadcast_in_dim %2736, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2738 = stablehlo.multiply %2731, %2737 : tensor<256x14x14x1024xf32>
    %2739 = stablehlo.reshape %arg374 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2740 = stablehlo.broadcast_in_dim %2739, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2741 = stablehlo.add %2738, %2740 : tensor<256x14x14x1024xf32>
    %2742 = stablehlo.convert %2741 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2743 = stablehlo.add %2686, %2742 : tensor<256x14x14x1024xbf16>
    %2744 = call @relu_193(%2743) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2745 = stablehlo.convert %arg385 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2746 = stablehlo.convolution(%2744, %2745) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2747 = stablehlo.broadcast_in_dim %arg1476, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2748 = stablehlo.broadcast_in_dim %arg1477, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2749 = stablehlo.convert %2746 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2750 = stablehlo.broadcast_in_dim %2747, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2751 = stablehlo.subtract %2749, %2750 : tensor<256x14x14x256xf32>
    %2752 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2753 = stablehlo.add %2748, %2752 : tensor<1x1x1x256xf32>
    %2754 = stablehlo.rsqrt %2753 : tensor<1x1x1x256xf32>
    %2755 = stablehlo.reshape %arg380 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2756 = stablehlo.multiply %2754, %2755 : tensor<1x1x1x256xf32>
    %2757 = stablehlo.broadcast_in_dim %2756, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2758 = stablehlo.multiply %2751, %2757 : tensor<256x14x14x256xf32>
    %2759 = stablehlo.reshape %arg379 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2760 = stablehlo.broadcast_in_dim %2759, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2761 = stablehlo.add %2758, %2760 : tensor<256x14x14x256xf32>
    %2762 = stablehlo.convert %2761 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2763 = call @relu_175(%2762) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2764 = stablehlo.convert %arg386 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2765 = stablehlo.convolution(%2763, %2764) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2766 = stablehlo.broadcast_in_dim %arg1478, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2767 = stablehlo.broadcast_in_dim %arg1479, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2768 = stablehlo.convert %2765 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2769 = stablehlo.broadcast_in_dim %2766, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2770 = stablehlo.subtract %2768, %2769 : tensor<256x14x14x256xf32>
    %2771 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2772 = stablehlo.add %2767, %2771 : tensor<1x1x1x256xf32>
    %2773 = stablehlo.rsqrt %2772 : tensor<1x1x1x256xf32>
    %2774 = stablehlo.reshape %arg382 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2775 = stablehlo.multiply %2773, %2774 : tensor<1x1x1x256xf32>
    %2776 = stablehlo.broadcast_in_dim %2775, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2777 = stablehlo.multiply %2770, %2776 : tensor<256x14x14x256xf32>
    %2778 = stablehlo.reshape %arg381 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2779 = stablehlo.broadcast_in_dim %2778, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2780 = stablehlo.add %2777, %2779 : tensor<256x14x14x256xf32>
    %2781 = stablehlo.convert %2780 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2782 = call @relu_175(%2781) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2783 = stablehlo.convert %arg387 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2784 = stablehlo.convolution(%2782, %2783) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2785 = stablehlo.broadcast_in_dim %arg1480, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2786 = stablehlo.broadcast_in_dim %arg1481, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2787 = stablehlo.convert %2784 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2788 = stablehlo.broadcast_in_dim %2785, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2789 = stablehlo.subtract %2787, %2788 : tensor<256x14x14x1024xf32>
    %2790 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2791 = stablehlo.add %2786, %2790 : tensor<1x1x1x1024xf32>
    %2792 = stablehlo.rsqrt %2791 : tensor<1x1x1x1024xf32>
    %2793 = stablehlo.reshape %arg384 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2794 = stablehlo.multiply %2792, %2793 : tensor<1x1x1x1024xf32>
    %2795 = stablehlo.broadcast_in_dim %2794, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2796 = stablehlo.multiply %2789, %2795 : tensor<256x14x14x1024xf32>
    %2797 = stablehlo.reshape %arg383 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2798 = stablehlo.broadcast_in_dim %2797, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2799 = stablehlo.add %2796, %2798 : tensor<256x14x14x1024xf32>
    %2800 = stablehlo.convert %2799 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2801 = stablehlo.add %2744, %2800 : tensor<256x14x14x1024xbf16>
    %2802 = call @relu_193(%2801) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2803 = stablehlo.convert %arg394 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2804 = stablehlo.convolution(%2802, %2803) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2805 = stablehlo.broadcast_in_dim %arg1482, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2806 = stablehlo.broadcast_in_dim %arg1483, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2807 = stablehlo.convert %2804 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2808 = stablehlo.broadcast_in_dim %2805, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2809 = stablehlo.subtract %2807, %2808 : tensor<256x14x14x256xf32>
    %2810 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2811 = stablehlo.add %2806, %2810 : tensor<1x1x1x256xf32>
    %2812 = stablehlo.rsqrt %2811 : tensor<1x1x1x256xf32>
    %2813 = stablehlo.reshape %arg389 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2814 = stablehlo.multiply %2812, %2813 : tensor<1x1x1x256xf32>
    %2815 = stablehlo.broadcast_in_dim %2814, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2816 = stablehlo.multiply %2809, %2815 : tensor<256x14x14x256xf32>
    %2817 = stablehlo.reshape %arg388 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2818 = stablehlo.broadcast_in_dim %2817, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2819 = stablehlo.add %2816, %2818 : tensor<256x14x14x256xf32>
    %2820 = stablehlo.convert %2819 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2821 = call @relu_175(%2820) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2822 = stablehlo.convert %arg395 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2823 = stablehlo.convolution(%2821, %2822) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2824 = stablehlo.broadcast_in_dim %arg1484, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2825 = stablehlo.broadcast_in_dim %arg1485, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2826 = stablehlo.convert %2823 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2827 = stablehlo.broadcast_in_dim %2824, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2828 = stablehlo.subtract %2826, %2827 : tensor<256x14x14x256xf32>
    %2829 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2830 = stablehlo.add %2825, %2829 : tensor<1x1x1x256xf32>
    %2831 = stablehlo.rsqrt %2830 : tensor<1x1x1x256xf32>
    %2832 = stablehlo.reshape %arg391 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2833 = stablehlo.multiply %2831, %2832 : tensor<1x1x1x256xf32>
    %2834 = stablehlo.broadcast_in_dim %2833, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2835 = stablehlo.multiply %2828, %2834 : tensor<256x14x14x256xf32>
    %2836 = stablehlo.reshape %arg390 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2837 = stablehlo.broadcast_in_dim %2836, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2838 = stablehlo.add %2835, %2837 : tensor<256x14x14x256xf32>
    %2839 = stablehlo.convert %2838 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2840 = call @relu_175(%2839) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2841 = stablehlo.convert %arg396 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2842 = stablehlo.convolution(%2840, %2841) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2843 = stablehlo.broadcast_in_dim %arg1486, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2844 = stablehlo.broadcast_in_dim %arg1487, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2845 = stablehlo.convert %2842 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2846 = stablehlo.broadcast_in_dim %2843, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2847 = stablehlo.subtract %2845, %2846 : tensor<256x14x14x1024xf32>
    %2848 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2849 = stablehlo.add %2844, %2848 : tensor<1x1x1x1024xf32>
    %2850 = stablehlo.rsqrt %2849 : tensor<1x1x1x1024xf32>
    %2851 = stablehlo.reshape %arg393 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2852 = stablehlo.multiply %2850, %2851 : tensor<1x1x1x1024xf32>
    %2853 = stablehlo.broadcast_in_dim %2852, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2854 = stablehlo.multiply %2847, %2853 : tensor<256x14x14x1024xf32>
    %2855 = stablehlo.reshape %arg392 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2856 = stablehlo.broadcast_in_dim %2855, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2857 = stablehlo.add %2854, %2856 : tensor<256x14x14x1024xf32>
    %2858 = stablehlo.convert %2857 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2859 = stablehlo.add %2802, %2858 : tensor<256x14x14x1024xbf16>
    %2860 = call @relu_193(%2859) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2861 = stablehlo.convert %arg403 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2862 = stablehlo.convolution(%2860, %2861) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2863 = stablehlo.broadcast_in_dim %arg1488, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2864 = stablehlo.broadcast_in_dim %arg1489, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2865 = stablehlo.convert %2862 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2866 = stablehlo.broadcast_in_dim %2863, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2867 = stablehlo.subtract %2865, %2866 : tensor<256x14x14x256xf32>
    %2868 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2869 = stablehlo.add %2864, %2868 : tensor<1x1x1x256xf32>
    %2870 = stablehlo.rsqrt %2869 : tensor<1x1x1x256xf32>
    %2871 = stablehlo.reshape %arg398 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2872 = stablehlo.multiply %2870, %2871 : tensor<1x1x1x256xf32>
    %2873 = stablehlo.broadcast_in_dim %2872, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2874 = stablehlo.multiply %2867, %2873 : tensor<256x14x14x256xf32>
    %2875 = stablehlo.reshape %arg397 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2876 = stablehlo.broadcast_in_dim %2875, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2877 = stablehlo.add %2874, %2876 : tensor<256x14x14x256xf32>
    %2878 = stablehlo.convert %2877 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2879 = call @relu_175(%2878) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2880 = stablehlo.convert %arg404 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2881 = stablehlo.convolution(%2879, %2880) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2882 = stablehlo.broadcast_in_dim %arg1490, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2883 = stablehlo.broadcast_in_dim %arg1491, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2884 = stablehlo.convert %2881 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2885 = stablehlo.broadcast_in_dim %2882, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2886 = stablehlo.subtract %2884, %2885 : tensor<256x14x14x256xf32>
    %2887 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2888 = stablehlo.add %2883, %2887 : tensor<1x1x1x256xf32>
    %2889 = stablehlo.rsqrt %2888 : tensor<1x1x1x256xf32>
    %2890 = stablehlo.reshape %arg400 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2891 = stablehlo.multiply %2889, %2890 : tensor<1x1x1x256xf32>
    %2892 = stablehlo.broadcast_in_dim %2891, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2893 = stablehlo.multiply %2886, %2892 : tensor<256x14x14x256xf32>
    %2894 = stablehlo.reshape %arg399 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2895 = stablehlo.broadcast_in_dim %2894, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2896 = stablehlo.add %2893, %2895 : tensor<256x14x14x256xf32>
    %2897 = stablehlo.convert %2896 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2898 = call @relu_175(%2897) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2899 = stablehlo.convert %arg405 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2900 = stablehlo.convolution(%2898, %2899) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2901 = stablehlo.broadcast_in_dim %arg1492, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2902 = stablehlo.broadcast_in_dim %arg1493, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2903 = stablehlo.convert %2900 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2904 = stablehlo.broadcast_in_dim %2901, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2905 = stablehlo.subtract %2903, %2904 : tensor<256x14x14x1024xf32>
    %2906 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2907 = stablehlo.add %2902, %2906 : tensor<1x1x1x1024xf32>
    %2908 = stablehlo.rsqrt %2907 : tensor<1x1x1x1024xf32>
    %2909 = stablehlo.reshape %arg402 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2910 = stablehlo.multiply %2908, %2909 : tensor<1x1x1x1024xf32>
    %2911 = stablehlo.broadcast_in_dim %2910, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2912 = stablehlo.multiply %2905, %2911 : tensor<256x14x14x1024xf32>
    %2913 = stablehlo.reshape %arg401 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2914 = stablehlo.broadcast_in_dim %2913, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2915 = stablehlo.add %2912, %2914 : tensor<256x14x14x1024xf32>
    %2916 = stablehlo.convert %2915 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2917 = stablehlo.add %2860, %2916 : tensor<256x14x14x1024xbf16>
    %2918 = call @relu_193(%2917) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2919 = stablehlo.convert %arg412 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2920 = stablehlo.convolution(%2918, %2919) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2921 = stablehlo.broadcast_in_dim %arg1494, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2922 = stablehlo.broadcast_in_dim %arg1495, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2923 = stablehlo.convert %2920 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2924 = stablehlo.broadcast_in_dim %2921, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2925 = stablehlo.subtract %2923, %2924 : tensor<256x14x14x256xf32>
    %2926 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2927 = stablehlo.add %2922, %2926 : tensor<1x1x1x256xf32>
    %2928 = stablehlo.rsqrt %2927 : tensor<1x1x1x256xf32>
    %2929 = stablehlo.reshape %arg407 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2930 = stablehlo.multiply %2928, %2929 : tensor<1x1x1x256xf32>
    %2931 = stablehlo.broadcast_in_dim %2930, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2932 = stablehlo.multiply %2925, %2931 : tensor<256x14x14x256xf32>
    %2933 = stablehlo.reshape %arg406 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2934 = stablehlo.broadcast_in_dim %2933, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2935 = stablehlo.add %2932, %2934 : tensor<256x14x14x256xf32>
    %2936 = stablehlo.convert %2935 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2937 = call @relu_175(%2936) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2938 = stablehlo.convert %arg413 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2939 = stablehlo.convolution(%2937, %2938) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2940 = stablehlo.broadcast_in_dim %arg1496, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2941 = stablehlo.broadcast_in_dim %arg1497, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2942 = stablehlo.convert %2939 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2943 = stablehlo.broadcast_in_dim %2940, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2944 = stablehlo.subtract %2942, %2943 : tensor<256x14x14x256xf32>
    %2945 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2946 = stablehlo.add %2941, %2945 : tensor<1x1x1x256xf32>
    %2947 = stablehlo.rsqrt %2946 : tensor<1x1x1x256xf32>
    %2948 = stablehlo.reshape %arg409 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2949 = stablehlo.multiply %2947, %2948 : tensor<1x1x1x256xf32>
    %2950 = stablehlo.broadcast_in_dim %2949, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2951 = stablehlo.multiply %2944, %2950 : tensor<256x14x14x256xf32>
    %2952 = stablehlo.reshape %arg408 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2953 = stablehlo.broadcast_in_dim %2952, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2954 = stablehlo.add %2951, %2953 : tensor<256x14x14x256xf32>
    %2955 = stablehlo.convert %2954 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2956 = call @relu_175(%2955) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2957 = stablehlo.convert %arg414 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %2958 = stablehlo.convolution(%2956, %2957) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2959 = stablehlo.broadcast_in_dim %arg1498, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2960 = stablehlo.broadcast_in_dim %arg1499, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2961 = stablehlo.convert %2958 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %2962 = stablehlo.broadcast_in_dim %2959, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2963 = stablehlo.subtract %2961, %2962 : tensor<256x14x14x1024xf32>
    %2964 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %2965 = stablehlo.add %2960, %2964 : tensor<1x1x1x1024xf32>
    %2966 = stablehlo.rsqrt %2965 : tensor<1x1x1x1024xf32>
    %2967 = stablehlo.reshape %arg411 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2968 = stablehlo.multiply %2966, %2967 : tensor<1x1x1x1024xf32>
    %2969 = stablehlo.broadcast_in_dim %2968, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2970 = stablehlo.multiply %2963, %2969 : tensor<256x14x14x1024xf32>
    %2971 = stablehlo.reshape %arg410 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %2972 = stablehlo.broadcast_in_dim %2971, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %2973 = stablehlo.add %2970, %2972 : tensor<256x14x14x1024xf32>
    %2974 = stablehlo.convert %2973 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %2975 = stablehlo.add %2918, %2974 : tensor<256x14x14x1024xbf16>
    %2976 = call @relu_193(%2975) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %2977 = stablehlo.convert %arg430 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %2978 = stablehlo.convolution(%2976, %2977) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2979 = stablehlo.broadcast_in_dim %arg1506, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2980 = stablehlo.broadcast_in_dim %arg1507, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2981 = stablehlo.convert %2978 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2982 = stablehlo.broadcast_in_dim %2979, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2983 = stablehlo.subtract %2981, %2982 : tensor<256x14x14x256xf32>
    %2984 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2985 = stablehlo.add %2980, %2984 : tensor<1x1x1x256xf32>
    %2986 = stablehlo.rsqrt %2985 : tensor<1x1x1x256xf32>
    %2987 = stablehlo.reshape %arg425 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2988 = stablehlo.multiply %2986, %2987 : tensor<1x1x1x256xf32>
    %2989 = stablehlo.broadcast_in_dim %2988, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2990 = stablehlo.multiply %2983, %2989 : tensor<256x14x14x256xf32>
    %2991 = stablehlo.reshape %arg424 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2992 = stablehlo.broadcast_in_dim %2991, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2993 = stablehlo.add %2990, %2992 : tensor<256x14x14x256xf32>
    %2994 = stablehlo.convert %2993 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2995 = call @relu_175(%2994) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2996 = stablehlo.convert %arg431 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2997 = stablehlo.convolution(%2995, %2996) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2998 = stablehlo.broadcast_in_dim %arg1508, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2999 = stablehlo.broadcast_in_dim %arg1509, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3000 = stablehlo.convert %2997 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3001 = stablehlo.broadcast_in_dim %2998, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3002 = stablehlo.subtract %3000, %3001 : tensor<256x14x14x256xf32>
    %3003 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3004 = stablehlo.add %2999, %3003 : tensor<1x1x1x256xf32>
    %3005 = stablehlo.rsqrt %3004 : tensor<1x1x1x256xf32>
    %3006 = stablehlo.reshape %arg427 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3007 = stablehlo.multiply %3005, %3006 : tensor<1x1x1x256xf32>
    %3008 = stablehlo.broadcast_in_dim %3007, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3009 = stablehlo.multiply %3002, %3008 : tensor<256x14x14x256xf32>
    %3010 = stablehlo.reshape %arg426 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3011 = stablehlo.broadcast_in_dim %3010, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3012 = stablehlo.add %3009, %3011 : tensor<256x14x14x256xf32>
    %3013 = stablehlo.convert %3012 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3014 = call @relu_175(%3013) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3015 = stablehlo.convert %arg432 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3016 = stablehlo.convolution(%3014, %3015) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3017 = stablehlo.broadcast_in_dim %arg1510, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3018 = stablehlo.broadcast_in_dim %arg1511, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3019 = stablehlo.convert %3016 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3020 = stablehlo.broadcast_in_dim %3017, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3021 = stablehlo.subtract %3019, %3020 : tensor<256x14x14x1024xf32>
    %3022 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3023 = stablehlo.add %3018, %3022 : tensor<1x1x1x1024xf32>
    %3024 = stablehlo.rsqrt %3023 : tensor<1x1x1x1024xf32>
    %3025 = stablehlo.reshape %arg429 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3026 = stablehlo.multiply %3024, %3025 : tensor<1x1x1x1024xf32>
    %3027 = stablehlo.broadcast_in_dim %3026, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3028 = stablehlo.multiply %3021, %3027 : tensor<256x14x14x1024xf32>
    %3029 = stablehlo.reshape %arg428 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3030 = stablehlo.broadcast_in_dim %3029, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3031 = stablehlo.add %3028, %3030 : tensor<256x14x14x1024xf32>
    %3032 = stablehlo.convert %3031 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3033 = stablehlo.add %2976, %3032 : tensor<256x14x14x1024xbf16>
    %3034 = call @relu_193(%3033) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3035 = stablehlo.convert %arg439 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3036 = stablehlo.convolution(%3034, %3035) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3037 = stablehlo.broadcast_in_dim %arg1512, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3038 = stablehlo.broadcast_in_dim %arg1513, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3039 = stablehlo.convert %3036 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3040 = stablehlo.broadcast_in_dim %3037, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3041 = stablehlo.subtract %3039, %3040 : tensor<256x14x14x256xf32>
    %3042 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3043 = stablehlo.add %3038, %3042 : tensor<1x1x1x256xf32>
    %3044 = stablehlo.rsqrt %3043 : tensor<1x1x1x256xf32>
    %3045 = stablehlo.reshape %arg434 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3046 = stablehlo.multiply %3044, %3045 : tensor<1x1x1x256xf32>
    %3047 = stablehlo.broadcast_in_dim %3046, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3048 = stablehlo.multiply %3041, %3047 : tensor<256x14x14x256xf32>
    %3049 = stablehlo.reshape %arg433 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3050 = stablehlo.broadcast_in_dim %3049, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3051 = stablehlo.add %3048, %3050 : tensor<256x14x14x256xf32>
    %3052 = stablehlo.convert %3051 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3053 = call @relu_175(%3052) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3054 = stablehlo.convert %arg440 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3055 = stablehlo.convolution(%3053, %3054) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3056 = stablehlo.broadcast_in_dim %arg1514, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3057 = stablehlo.broadcast_in_dim %arg1515, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3058 = stablehlo.convert %3055 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3059 = stablehlo.broadcast_in_dim %3056, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3060 = stablehlo.subtract %3058, %3059 : tensor<256x14x14x256xf32>
    %3061 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3062 = stablehlo.add %3057, %3061 : tensor<1x1x1x256xf32>
    %3063 = stablehlo.rsqrt %3062 : tensor<1x1x1x256xf32>
    %3064 = stablehlo.reshape %arg436 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3065 = stablehlo.multiply %3063, %3064 : tensor<1x1x1x256xf32>
    %3066 = stablehlo.broadcast_in_dim %3065, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3067 = stablehlo.multiply %3060, %3066 : tensor<256x14x14x256xf32>
    %3068 = stablehlo.reshape %arg435 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3069 = stablehlo.broadcast_in_dim %3068, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3070 = stablehlo.add %3067, %3069 : tensor<256x14x14x256xf32>
    %3071 = stablehlo.convert %3070 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3072 = call @relu_175(%3071) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3073 = stablehlo.convert %arg441 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3074 = stablehlo.convolution(%3072, %3073) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3075 = stablehlo.broadcast_in_dim %arg1516, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3076 = stablehlo.broadcast_in_dim %arg1517, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3077 = stablehlo.convert %3074 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3078 = stablehlo.broadcast_in_dim %3075, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3079 = stablehlo.subtract %3077, %3078 : tensor<256x14x14x1024xf32>
    %3080 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3081 = stablehlo.add %3076, %3080 : tensor<1x1x1x1024xf32>
    %3082 = stablehlo.rsqrt %3081 : tensor<1x1x1x1024xf32>
    %3083 = stablehlo.reshape %arg438 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3084 = stablehlo.multiply %3082, %3083 : tensor<1x1x1x1024xf32>
    %3085 = stablehlo.broadcast_in_dim %3084, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3086 = stablehlo.multiply %3079, %3085 : tensor<256x14x14x1024xf32>
    %3087 = stablehlo.reshape %arg437 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3088 = stablehlo.broadcast_in_dim %3087, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3089 = stablehlo.add %3086, %3088 : tensor<256x14x14x1024xf32>
    %3090 = stablehlo.convert %3089 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3091 = stablehlo.add %3034, %3090 : tensor<256x14x14x1024xbf16>
    %3092 = call @relu_193(%3091) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3093 = stablehlo.convert %arg448 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3094 = stablehlo.convolution(%3092, %3093) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3095 = stablehlo.broadcast_in_dim %arg1518, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3096 = stablehlo.broadcast_in_dim %arg1519, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3097 = stablehlo.convert %3094 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3098 = stablehlo.broadcast_in_dim %3095, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3099 = stablehlo.subtract %3097, %3098 : tensor<256x14x14x256xf32>
    %3100 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3101 = stablehlo.add %3096, %3100 : tensor<1x1x1x256xf32>
    %3102 = stablehlo.rsqrt %3101 : tensor<1x1x1x256xf32>
    %3103 = stablehlo.reshape %arg443 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3104 = stablehlo.multiply %3102, %3103 : tensor<1x1x1x256xf32>
    %3105 = stablehlo.broadcast_in_dim %3104, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3106 = stablehlo.multiply %3099, %3105 : tensor<256x14x14x256xf32>
    %3107 = stablehlo.reshape %arg442 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3108 = stablehlo.broadcast_in_dim %3107, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3109 = stablehlo.add %3106, %3108 : tensor<256x14x14x256xf32>
    %3110 = stablehlo.convert %3109 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3111 = call @relu_175(%3110) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3112 = stablehlo.convert %arg449 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3113 = stablehlo.convolution(%3111, %3112) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3114 = stablehlo.broadcast_in_dim %arg1520, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3115 = stablehlo.broadcast_in_dim %arg1521, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3116 = stablehlo.convert %3113 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3117 = stablehlo.broadcast_in_dim %3114, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3118 = stablehlo.subtract %3116, %3117 : tensor<256x14x14x256xf32>
    %3119 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3120 = stablehlo.add %3115, %3119 : tensor<1x1x1x256xf32>
    %3121 = stablehlo.rsqrt %3120 : tensor<1x1x1x256xf32>
    %3122 = stablehlo.reshape %arg445 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3123 = stablehlo.multiply %3121, %3122 : tensor<1x1x1x256xf32>
    %3124 = stablehlo.broadcast_in_dim %3123, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3125 = stablehlo.multiply %3118, %3124 : tensor<256x14x14x256xf32>
    %3126 = stablehlo.reshape %arg444 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3127 = stablehlo.broadcast_in_dim %3126, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3128 = stablehlo.add %3125, %3127 : tensor<256x14x14x256xf32>
    %3129 = stablehlo.convert %3128 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3130 = call @relu_175(%3129) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3131 = stablehlo.convert %arg450 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3132 = stablehlo.convolution(%3130, %3131) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3133 = stablehlo.broadcast_in_dim %arg1522, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3134 = stablehlo.broadcast_in_dim %arg1523, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3135 = stablehlo.convert %3132 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3136 = stablehlo.broadcast_in_dim %3133, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3137 = stablehlo.subtract %3135, %3136 : tensor<256x14x14x1024xf32>
    %3138 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3139 = stablehlo.add %3134, %3138 : tensor<1x1x1x1024xf32>
    %3140 = stablehlo.rsqrt %3139 : tensor<1x1x1x1024xf32>
    %3141 = stablehlo.reshape %arg447 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3142 = stablehlo.multiply %3140, %3141 : tensor<1x1x1x1024xf32>
    %3143 = stablehlo.broadcast_in_dim %3142, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3144 = stablehlo.multiply %3137, %3143 : tensor<256x14x14x1024xf32>
    %3145 = stablehlo.reshape %arg446 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3146 = stablehlo.broadcast_in_dim %3145, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3147 = stablehlo.add %3144, %3146 : tensor<256x14x14x1024xf32>
    %3148 = stablehlo.convert %3147 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3149 = stablehlo.add %3092, %3148 : tensor<256x14x14x1024xbf16>
    %3150 = call @relu_193(%3149) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3151 = stablehlo.convert %arg457 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3152 = stablehlo.convolution(%3150, %3151) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3153 = stablehlo.broadcast_in_dim %arg1524, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3154 = stablehlo.broadcast_in_dim %arg1525, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3155 = stablehlo.convert %3152 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3156 = stablehlo.broadcast_in_dim %3153, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3157 = stablehlo.subtract %3155, %3156 : tensor<256x14x14x256xf32>
    %3158 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3159 = stablehlo.add %3154, %3158 : tensor<1x1x1x256xf32>
    %3160 = stablehlo.rsqrt %3159 : tensor<1x1x1x256xf32>
    %3161 = stablehlo.reshape %arg452 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3162 = stablehlo.multiply %3160, %3161 : tensor<1x1x1x256xf32>
    %3163 = stablehlo.broadcast_in_dim %3162, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3164 = stablehlo.multiply %3157, %3163 : tensor<256x14x14x256xf32>
    %3165 = stablehlo.reshape %arg451 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3166 = stablehlo.broadcast_in_dim %3165, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3167 = stablehlo.add %3164, %3166 : tensor<256x14x14x256xf32>
    %3168 = stablehlo.convert %3167 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3169 = call @relu_175(%3168) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3170 = stablehlo.convert %arg458 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3171 = stablehlo.convolution(%3169, %3170) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3172 = stablehlo.broadcast_in_dim %arg1526, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3173 = stablehlo.broadcast_in_dim %arg1527, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3174 = stablehlo.convert %3171 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3175 = stablehlo.broadcast_in_dim %3172, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3176 = stablehlo.subtract %3174, %3175 : tensor<256x14x14x256xf32>
    %3177 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3178 = stablehlo.add %3173, %3177 : tensor<1x1x1x256xf32>
    %3179 = stablehlo.rsqrt %3178 : tensor<1x1x1x256xf32>
    %3180 = stablehlo.reshape %arg454 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3181 = stablehlo.multiply %3179, %3180 : tensor<1x1x1x256xf32>
    %3182 = stablehlo.broadcast_in_dim %3181, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3183 = stablehlo.multiply %3176, %3182 : tensor<256x14x14x256xf32>
    %3184 = stablehlo.reshape %arg453 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3185 = stablehlo.broadcast_in_dim %3184, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3186 = stablehlo.add %3183, %3185 : tensor<256x14x14x256xf32>
    %3187 = stablehlo.convert %3186 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3188 = call @relu_175(%3187) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3189 = stablehlo.convert %arg459 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3190 = stablehlo.convolution(%3188, %3189) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3191 = stablehlo.broadcast_in_dim %arg1528, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3192 = stablehlo.broadcast_in_dim %arg1529, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3193 = stablehlo.convert %3190 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3194 = stablehlo.broadcast_in_dim %3191, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3195 = stablehlo.subtract %3193, %3194 : tensor<256x14x14x1024xf32>
    %3196 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3197 = stablehlo.add %3192, %3196 : tensor<1x1x1x1024xf32>
    %3198 = stablehlo.rsqrt %3197 : tensor<1x1x1x1024xf32>
    %3199 = stablehlo.reshape %arg456 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3200 = stablehlo.multiply %3198, %3199 : tensor<1x1x1x1024xf32>
    %3201 = stablehlo.broadcast_in_dim %3200, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3202 = stablehlo.multiply %3195, %3201 : tensor<256x14x14x1024xf32>
    %3203 = stablehlo.reshape %arg455 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3204 = stablehlo.broadcast_in_dim %3203, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3205 = stablehlo.add %3202, %3204 : tensor<256x14x14x1024xf32>
    %3206 = stablehlo.convert %3205 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3207 = stablehlo.add %3150, %3206 : tensor<256x14x14x1024xbf16>
    %3208 = call @relu_193(%3207) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3209 = stablehlo.convert %arg466 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3210 = stablehlo.convolution(%3208, %3209) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3211 = stablehlo.broadcast_in_dim %arg1530, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3212 = stablehlo.broadcast_in_dim %arg1531, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3213 = stablehlo.convert %3210 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3214 = stablehlo.broadcast_in_dim %3211, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3215 = stablehlo.subtract %3213, %3214 : tensor<256x14x14x256xf32>
    %3216 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3217 = stablehlo.add %3212, %3216 : tensor<1x1x1x256xf32>
    %3218 = stablehlo.rsqrt %3217 : tensor<1x1x1x256xf32>
    %3219 = stablehlo.reshape %arg461 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3220 = stablehlo.multiply %3218, %3219 : tensor<1x1x1x256xf32>
    %3221 = stablehlo.broadcast_in_dim %3220, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3222 = stablehlo.multiply %3215, %3221 : tensor<256x14x14x256xf32>
    %3223 = stablehlo.reshape %arg460 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3224 = stablehlo.broadcast_in_dim %3223, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3225 = stablehlo.add %3222, %3224 : tensor<256x14x14x256xf32>
    %3226 = stablehlo.convert %3225 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3227 = call @relu_175(%3226) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3228 = stablehlo.convert %arg467 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3229 = stablehlo.convolution(%3227, %3228) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3230 = stablehlo.broadcast_in_dim %arg1532, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3231 = stablehlo.broadcast_in_dim %arg1533, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3232 = stablehlo.convert %3229 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3233 = stablehlo.broadcast_in_dim %3230, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3234 = stablehlo.subtract %3232, %3233 : tensor<256x14x14x256xf32>
    %3235 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3236 = stablehlo.add %3231, %3235 : tensor<1x1x1x256xf32>
    %3237 = stablehlo.rsqrt %3236 : tensor<1x1x1x256xf32>
    %3238 = stablehlo.reshape %arg463 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3239 = stablehlo.multiply %3237, %3238 : tensor<1x1x1x256xf32>
    %3240 = stablehlo.broadcast_in_dim %3239, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3241 = stablehlo.multiply %3234, %3240 : tensor<256x14x14x256xf32>
    %3242 = stablehlo.reshape %arg462 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3243 = stablehlo.broadcast_in_dim %3242, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3244 = stablehlo.add %3241, %3243 : tensor<256x14x14x256xf32>
    %3245 = stablehlo.convert %3244 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3246 = call @relu_175(%3245) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3247 = stablehlo.convert %arg468 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3248 = stablehlo.convolution(%3246, %3247) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3249 = stablehlo.broadcast_in_dim %arg1534, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3250 = stablehlo.broadcast_in_dim %arg1535, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3251 = stablehlo.convert %3248 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3252 = stablehlo.broadcast_in_dim %3249, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3253 = stablehlo.subtract %3251, %3252 : tensor<256x14x14x1024xf32>
    %3254 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3255 = stablehlo.add %3250, %3254 : tensor<1x1x1x1024xf32>
    %3256 = stablehlo.rsqrt %3255 : tensor<1x1x1x1024xf32>
    %3257 = stablehlo.reshape %arg465 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3258 = stablehlo.multiply %3256, %3257 : tensor<1x1x1x1024xf32>
    %3259 = stablehlo.broadcast_in_dim %3258, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3260 = stablehlo.multiply %3253, %3259 : tensor<256x14x14x1024xf32>
    %3261 = stablehlo.reshape %arg464 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3262 = stablehlo.broadcast_in_dim %3261, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3263 = stablehlo.add %3260, %3262 : tensor<256x14x14x1024xf32>
    %3264 = stablehlo.convert %3263 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3265 = stablehlo.add %3208, %3264 : tensor<256x14x14x1024xbf16>
    %3266 = call @relu_193(%3265) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3267 = stablehlo.convert %arg475 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3268 = stablehlo.convolution(%3266, %3267) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3269 = stablehlo.broadcast_in_dim %arg1536, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3270 = stablehlo.broadcast_in_dim %arg1537, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3271 = stablehlo.convert %3268 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3272 = stablehlo.broadcast_in_dim %3269, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3273 = stablehlo.subtract %3271, %3272 : tensor<256x14x14x256xf32>
    %3274 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3275 = stablehlo.add %3270, %3274 : tensor<1x1x1x256xf32>
    %3276 = stablehlo.rsqrt %3275 : tensor<1x1x1x256xf32>
    %3277 = stablehlo.reshape %arg470 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3278 = stablehlo.multiply %3276, %3277 : tensor<1x1x1x256xf32>
    %3279 = stablehlo.broadcast_in_dim %3278, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3280 = stablehlo.multiply %3273, %3279 : tensor<256x14x14x256xf32>
    %3281 = stablehlo.reshape %arg469 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3282 = stablehlo.broadcast_in_dim %3281, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3283 = stablehlo.add %3280, %3282 : tensor<256x14x14x256xf32>
    %3284 = stablehlo.convert %3283 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3285 = call @relu_175(%3284) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3286 = stablehlo.convert %arg476 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3287 = stablehlo.convolution(%3285, %3286) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3288 = stablehlo.broadcast_in_dim %arg1538, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3289 = stablehlo.broadcast_in_dim %arg1539, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3290 = stablehlo.convert %3287 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3291 = stablehlo.broadcast_in_dim %3288, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3292 = stablehlo.subtract %3290, %3291 : tensor<256x14x14x256xf32>
    %3293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3294 = stablehlo.add %3289, %3293 : tensor<1x1x1x256xf32>
    %3295 = stablehlo.rsqrt %3294 : tensor<1x1x1x256xf32>
    %3296 = stablehlo.reshape %arg472 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3297 = stablehlo.multiply %3295, %3296 : tensor<1x1x1x256xf32>
    %3298 = stablehlo.broadcast_in_dim %3297, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3299 = stablehlo.multiply %3292, %3298 : tensor<256x14x14x256xf32>
    %3300 = stablehlo.reshape %arg471 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3301 = stablehlo.broadcast_in_dim %3300, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3302 = stablehlo.add %3299, %3301 : tensor<256x14x14x256xf32>
    %3303 = stablehlo.convert %3302 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3304 = call @relu_175(%3303) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3305 = stablehlo.convert %arg477 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3306 = stablehlo.convolution(%3304, %3305) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3307 = stablehlo.broadcast_in_dim %arg1540, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3308 = stablehlo.broadcast_in_dim %arg1541, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3309 = stablehlo.convert %3306 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3310 = stablehlo.broadcast_in_dim %3307, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3311 = stablehlo.subtract %3309, %3310 : tensor<256x14x14x1024xf32>
    %3312 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3313 = stablehlo.add %3308, %3312 : tensor<1x1x1x1024xf32>
    %3314 = stablehlo.rsqrt %3313 : tensor<1x1x1x1024xf32>
    %3315 = stablehlo.reshape %arg474 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3316 = stablehlo.multiply %3314, %3315 : tensor<1x1x1x1024xf32>
    %3317 = stablehlo.broadcast_in_dim %3316, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3318 = stablehlo.multiply %3311, %3317 : tensor<256x14x14x1024xf32>
    %3319 = stablehlo.reshape %arg473 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3320 = stablehlo.broadcast_in_dim %3319, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3321 = stablehlo.add %3318, %3320 : tensor<256x14x14x1024xf32>
    %3322 = stablehlo.convert %3321 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3323 = stablehlo.add %3266, %3322 : tensor<256x14x14x1024xbf16>
    %3324 = call @relu_193(%3323) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3325 = stablehlo.convert %arg484 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3326 = stablehlo.convolution(%3324, %3325) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3327 = stablehlo.broadcast_in_dim %arg1542, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3328 = stablehlo.broadcast_in_dim %arg1543, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3329 = stablehlo.convert %3326 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3330 = stablehlo.broadcast_in_dim %3327, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3331 = stablehlo.subtract %3329, %3330 : tensor<256x14x14x256xf32>
    %3332 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3333 = stablehlo.add %3328, %3332 : tensor<1x1x1x256xf32>
    %3334 = stablehlo.rsqrt %3333 : tensor<1x1x1x256xf32>
    %3335 = stablehlo.reshape %arg479 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3336 = stablehlo.multiply %3334, %3335 : tensor<1x1x1x256xf32>
    %3337 = stablehlo.broadcast_in_dim %3336, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3338 = stablehlo.multiply %3331, %3337 : tensor<256x14x14x256xf32>
    %3339 = stablehlo.reshape %arg478 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3340 = stablehlo.broadcast_in_dim %3339, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3341 = stablehlo.add %3338, %3340 : tensor<256x14x14x256xf32>
    %3342 = stablehlo.convert %3341 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3343 = call @relu_175(%3342) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3344 = stablehlo.convert %arg485 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3345 = stablehlo.convolution(%3343, %3344) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3346 = stablehlo.broadcast_in_dim %arg1544, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3347 = stablehlo.broadcast_in_dim %arg1545, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3348 = stablehlo.convert %3345 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3349 = stablehlo.broadcast_in_dim %3346, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3350 = stablehlo.subtract %3348, %3349 : tensor<256x14x14x256xf32>
    %3351 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3352 = stablehlo.add %3347, %3351 : tensor<1x1x1x256xf32>
    %3353 = stablehlo.rsqrt %3352 : tensor<1x1x1x256xf32>
    %3354 = stablehlo.reshape %arg481 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3355 = stablehlo.multiply %3353, %3354 : tensor<1x1x1x256xf32>
    %3356 = stablehlo.broadcast_in_dim %3355, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3357 = stablehlo.multiply %3350, %3356 : tensor<256x14x14x256xf32>
    %3358 = stablehlo.reshape %arg480 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3359 = stablehlo.broadcast_in_dim %3358, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3360 = stablehlo.add %3357, %3359 : tensor<256x14x14x256xf32>
    %3361 = stablehlo.convert %3360 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3362 = call @relu_175(%3361) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3363 = stablehlo.convert %arg486 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3364 = stablehlo.convolution(%3362, %3363) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3365 = stablehlo.broadcast_in_dim %arg1546, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3366 = stablehlo.broadcast_in_dim %arg1547, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3367 = stablehlo.convert %3364 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3368 = stablehlo.broadcast_in_dim %3365, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3369 = stablehlo.subtract %3367, %3368 : tensor<256x14x14x1024xf32>
    %3370 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3371 = stablehlo.add %3366, %3370 : tensor<1x1x1x1024xf32>
    %3372 = stablehlo.rsqrt %3371 : tensor<1x1x1x1024xf32>
    %3373 = stablehlo.reshape %arg483 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3374 = stablehlo.multiply %3372, %3373 : tensor<1x1x1x1024xf32>
    %3375 = stablehlo.broadcast_in_dim %3374, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3376 = stablehlo.multiply %3369, %3375 : tensor<256x14x14x1024xf32>
    %3377 = stablehlo.reshape %arg482 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3378 = stablehlo.broadcast_in_dim %3377, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3379 = stablehlo.add %3376, %3378 : tensor<256x14x14x1024xf32>
    %3380 = stablehlo.convert %3379 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3381 = stablehlo.add %3324, %3380 : tensor<256x14x14x1024xbf16>
    %3382 = call @relu_193(%3381) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3383 = stablehlo.convert %arg493 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3384 = stablehlo.convolution(%3382, %3383) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3385 = stablehlo.broadcast_in_dim %arg1548, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3386 = stablehlo.broadcast_in_dim %arg1549, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3387 = stablehlo.convert %3384 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3388 = stablehlo.broadcast_in_dim %3385, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3389 = stablehlo.subtract %3387, %3388 : tensor<256x14x14x256xf32>
    %3390 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3391 = stablehlo.add %3386, %3390 : tensor<1x1x1x256xf32>
    %3392 = stablehlo.rsqrt %3391 : tensor<1x1x1x256xf32>
    %3393 = stablehlo.reshape %arg488 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3394 = stablehlo.multiply %3392, %3393 : tensor<1x1x1x256xf32>
    %3395 = stablehlo.broadcast_in_dim %3394, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3396 = stablehlo.multiply %3389, %3395 : tensor<256x14x14x256xf32>
    %3397 = stablehlo.reshape %arg487 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3398 = stablehlo.broadcast_in_dim %3397, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3399 = stablehlo.add %3396, %3398 : tensor<256x14x14x256xf32>
    %3400 = stablehlo.convert %3399 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3401 = call @relu_175(%3400) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3402 = stablehlo.convert %arg494 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3403 = stablehlo.convolution(%3401, %3402) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3404 = stablehlo.broadcast_in_dim %arg1550, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3405 = stablehlo.broadcast_in_dim %arg1551, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3406 = stablehlo.convert %3403 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3407 = stablehlo.broadcast_in_dim %3404, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3408 = stablehlo.subtract %3406, %3407 : tensor<256x14x14x256xf32>
    %3409 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3410 = stablehlo.add %3405, %3409 : tensor<1x1x1x256xf32>
    %3411 = stablehlo.rsqrt %3410 : tensor<1x1x1x256xf32>
    %3412 = stablehlo.reshape %arg490 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3413 = stablehlo.multiply %3411, %3412 : tensor<1x1x1x256xf32>
    %3414 = stablehlo.broadcast_in_dim %3413, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3415 = stablehlo.multiply %3408, %3414 : tensor<256x14x14x256xf32>
    %3416 = stablehlo.reshape %arg489 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3417 = stablehlo.broadcast_in_dim %3416, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3418 = stablehlo.add %3415, %3417 : tensor<256x14x14x256xf32>
    %3419 = stablehlo.convert %3418 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3420 = call @relu_175(%3419) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3421 = stablehlo.convert %arg495 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3422 = stablehlo.convolution(%3420, %3421) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3423 = stablehlo.broadcast_in_dim %arg1552, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3424 = stablehlo.broadcast_in_dim %arg1553, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3425 = stablehlo.convert %3422 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3426 = stablehlo.broadcast_in_dim %3423, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3427 = stablehlo.subtract %3425, %3426 : tensor<256x14x14x1024xf32>
    %3428 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3429 = stablehlo.add %3424, %3428 : tensor<1x1x1x1024xf32>
    %3430 = stablehlo.rsqrt %3429 : tensor<1x1x1x1024xf32>
    %3431 = stablehlo.reshape %arg492 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3432 = stablehlo.multiply %3430, %3431 : tensor<1x1x1x1024xf32>
    %3433 = stablehlo.broadcast_in_dim %3432, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3434 = stablehlo.multiply %3427, %3433 : tensor<256x14x14x1024xf32>
    %3435 = stablehlo.reshape %arg491 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3436 = stablehlo.broadcast_in_dim %3435, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3437 = stablehlo.add %3434, %3436 : tensor<256x14x14x1024xf32>
    %3438 = stablehlo.convert %3437 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3439 = stablehlo.add %3382, %3438 : tensor<256x14x14x1024xbf16>
    %3440 = call @relu_193(%3439) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3441 = stablehlo.convert %arg502 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3442 = stablehlo.convolution(%3440, %3441) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3443 = stablehlo.broadcast_in_dim %arg1554, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3444 = stablehlo.broadcast_in_dim %arg1555, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3445 = stablehlo.convert %3442 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3446 = stablehlo.broadcast_in_dim %3443, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3447 = stablehlo.subtract %3445, %3446 : tensor<256x14x14x256xf32>
    %3448 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3449 = stablehlo.add %3444, %3448 : tensor<1x1x1x256xf32>
    %3450 = stablehlo.rsqrt %3449 : tensor<1x1x1x256xf32>
    %3451 = stablehlo.reshape %arg497 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3452 = stablehlo.multiply %3450, %3451 : tensor<1x1x1x256xf32>
    %3453 = stablehlo.broadcast_in_dim %3452, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3454 = stablehlo.multiply %3447, %3453 : tensor<256x14x14x256xf32>
    %3455 = stablehlo.reshape %arg496 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3456 = stablehlo.broadcast_in_dim %3455, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3457 = stablehlo.add %3454, %3456 : tensor<256x14x14x256xf32>
    %3458 = stablehlo.convert %3457 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3459 = call @relu_175(%3458) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3460 = stablehlo.convert %arg503 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3461 = stablehlo.convolution(%3459, %3460) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3462 = stablehlo.broadcast_in_dim %arg1556, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3463 = stablehlo.broadcast_in_dim %arg1557, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3464 = stablehlo.convert %3461 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3465 = stablehlo.broadcast_in_dim %3462, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3466 = stablehlo.subtract %3464, %3465 : tensor<256x14x14x256xf32>
    %3467 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3468 = stablehlo.add %3463, %3467 : tensor<1x1x1x256xf32>
    %3469 = stablehlo.rsqrt %3468 : tensor<1x1x1x256xf32>
    %3470 = stablehlo.reshape %arg499 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3471 = stablehlo.multiply %3469, %3470 : tensor<1x1x1x256xf32>
    %3472 = stablehlo.broadcast_in_dim %3471, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3473 = stablehlo.multiply %3466, %3472 : tensor<256x14x14x256xf32>
    %3474 = stablehlo.reshape %arg498 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3475 = stablehlo.broadcast_in_dim %3474, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3476 = stablehlo.add %3473, %3475 : tensor<256x14x14x256xf32>
    %3477 = stablehlo.convert %3476 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3478 = call @relu_175(%3477) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3479 = stablehlo.convert %arg504 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3480 = stablehlo.convolution(%3478, %3479) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3481 = stablehlo.broadcast_in_dim %arg1558, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3482 = stablehlo.broadcast_in_dim %arg1559, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3483 = stablehlo.convert %3480 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3484 = stablehlo.broadcast_in_dim %3481, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3485 = stablehlo.subtract %3483, %3484 : tensor<256x14x14x1024xf32>
    %3486 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3487 = stablehlo.add %3482, %3486 : tensor<1x1x1x1024xf32>
    %3488 = stablehlo.rsqrt %3487 : tensor<1x1x1x1024xf32>
    %3489 = stablehlo.reshape %arg501 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3490 = stablehlo.multiply %3488, %3489 : tensor<1x1x1x1024xf32>
    %3491 = stablehlo.broadcast_in_dim %3490, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3492 = stablehlo.multiply %3485, %3491 : tensor<256x14x14x1024xf32>
    %3493 = stablehlo.reshape %arg500 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3494 = stablehlo.broadcast_in_dim %3493, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3495 = stablehlo.add %3492, %3494 : tensor<256x14x14x1024xf32>
    %3496 = stablehlo.convert %3495 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3497 = stablehlo.add %3440, %3496 : tensor<256x14x14x1024xbf16>
    %3498 = call @relu_193(%3497) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3499 = stablehlo.convert %arg511 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3500 = stablehlo.convolution(%3498, %3499) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3501 = stablehlo.broadcast_in_dim %arg1560, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3502 = stablehlo.broadcast_in_dim %arg1561, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3503 = stablehlo.convert %3500 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3504 = stablehlo.broadcast_in_dim %3501, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3505 = stablehlo.subtract %3503, %3504 : tensor<256x14x14x256xf32>
    %3506 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3507 = stablehlo.add %3502, %3506 : tensor<1x1x1x256xf32>
    %3508 = stablehlo.rsqrt %3507 : tensor<1x1x1x256xf32>
    %3509 = stablehlo.reshape %arg506 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3510 = stablehlo.multiply %3508, %3509 : tensor<1x1x1x256xf32>
    %3511 = stablehlo.broadcast_in_dim %3510, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3512 = stablehlo.multiply %3505, %3511 : tensor<256x14x14x256xf32>
    %3513 = stablehlo.reshape %arg505 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3514 = stablehlo.broadcast_in_dim %3513, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3515 = stablehlo.add %3512, %3514 : tensor<256x14x14x256xf32>
    %3516 = stablehlo.convert %3515 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3517 = call @relu_175(%3516) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3518 = stablehlo.convert %arg512 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3519 = stablehlo.convolution(%3517, %3518) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3520 = stablehlo.broadcast_in_dim %arg1562, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3521 = stablehlo.broadcast_in_dim %arg1563, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3522 = stablehlo.convert %3519 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3523 = stablehlo.broadcast_in_dim %3520, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3524 = stablehlo.subtract %3522, %3523 : tensor<256x14x14x256xf32>
    %3525 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3526 = stablehlo.add %3521, %3525 : tensor<1x1x1x256xf32>
    %3527 = stablehlo.rsqrt %3526 : tensor<1x1x1x256xf32>
    %3528 = stablehlo.reshape %arg508 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3529 = stablehlo.multiply %3527, %3528 : tensor<1x1x1x256xf32>
    %3530 = stablehlo.broadcast_in_dim %3529, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3531 = stablehlo.multiply %3524, %3530 : tensor<256x14x14x256xf32>
    %3532 = stablehlo.reshape %arg507 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3533 = stablehlo.broadcast_in_dim %3532, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3534 = stablehlo.add %3531, %3533 : tensor<256x14x14x256xf32>
    %3535 = stablehlo.convert %3534 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3536 = call @relu_175(%3535) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3537 = stablehlo.convert %arg513 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3538 = stablehlo.convolution(%3536, %3537) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3539 = stablehlo.broadcast_in_dim %arg1564, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3540 = stablehlo.broadcast_in_dim %arg1565, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3541 = stablehlo.convert %3538 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3542 = stablehlo.broadcast_in_dim %3539, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3543 = stablehlo.subtract %3541, %3542 : tensor<256x14x14x1024xf32>
    %3544 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3545 = stablehlo.add %3540, %3544 : tensor<1x1x1x1024xf32>
    %3546 = stablehlo.rsqrt %3545 : tensor<1x1x1x1024xf32>
    %3547 = stablehlo.reshape %arg510 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3548 = stablehlo.multiply %3546, %3547 : tensor<1x1x1x1024xf32>
    %3549 = stablehlo.broadcast_in_dim %3548, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3550 = stablehlo.multiply %3543, %3549 : tensor<256x14x14x1024xf32>
    %3551 = stablehlo.reshape %arg509 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3552 = stablehlo.broadcast_in_dim %3551, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3553 = stablehlo.add %3550, %3552 : tensor<256x14x14x1024xf32>
    %3554 = stablehlo.convert %3553 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3555 = stablehlo.add %3498, %3554 : tensor<256x14x14x1024xbf16>
    %3556 = call @relu_193(%3555) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3557 = stablehlo.convert %arg529 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3558 = stablehlo.convolution(%3556, %3557) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3559 = stablehlo.broadcast_in_dim %arg1572, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3560 = stablehlo.broadcast_in_dim %arg1573, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3561 = stablehlo.convert %3558 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3562 = stablehlo.broadcast_in_dim %3559, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3563 = stablehlo.subtract %3561, %3562 : tensor<256x14x14x256xf32>
    %3564 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3565 = stablehlo.add %3560, %3564 : tensor<1x1x1x256xf32>
    %3566 = stablehlo.rsqrt %3565 : tensor<1x1x1x256xf32>
    %3567 = stablehlo.reshape %arg524 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3568 = stablehlo.multiply %3566, %3567 : tensor<1x1x1x256xf32>
    %3569 = stablehlo.broadcast_in_dim %3568, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3570 = stablehlo.multiply %3563, %3569 : tensor<256x14x14x256xf32>
    %3571 = stablehlo.reshape %arg523 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3572 = stablehlo.broadcast_in_dim %3571, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3573 = stablehlo.add %3570, %3572 : tensor<256x14x14x256xf32>
    %3574 = stablehlo.convert %3573 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3575 = call @relu_175(%3574) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3576 = stablehlo.convert %arg530 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3577 = stablehlo.convolution(%3575, %3576) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3578 = stablehlo.broadcast_in_dim %arg1574, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3579 = stablehlo.broadcast_in_dim %arg1575, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3580 = stablehlo.convert %3577 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3581 = stablehlo.broadcast_in_dim %3578, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3582 = stablehlo.subtract %3580, %3581 : tensor<256x14x14x256xf32>
    %3583 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3584 = stablehlo.add %3579, %3583 : tensor<1x1x1x256xf32>
    %3585 = stablehlo.rsqrt %3584 : tensor<1x1x1x256xf32>
    %3586 = stablehlo.reshape %arg526 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3587 = stablehlo.multiply %3585, %3586 : tensor<1x1x1x256xf32>
    %3588 = stablehlo.broadcast_in_dim %3587, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3589 = stablehlo.multiply %3582, %3588 : tensor<256x14x14x256xf32>
    %3590 = stablehlo.reshape %arg525 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3591 = stablehlo.broadcast_in_dim %3590, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3592 = stablehlo.add %3589, %3591 : tensor<256x14x14x256xf32>
    %3593 = stablehlo.convert %3592 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3594 = call @relu_175(%3593) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3595 = stablehlo.convert %arg531 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3596 = stablehlo.convolution(%3594, %3595) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3597 = stablehlo.broadcast_in_dim %arg1576, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3598 = stablehlo.broadcast_in_dim %arg1577, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3599 = stablehlo.convert %3596 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3600 = stablehlo.broadcast_in_dim %3597, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3601 = stablehlo.subtract %3599, %3600 : tensor<256x14x14x1024xf32>
    %3602 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3603 = stablehlo.add %3598, %3602 : tensor<1x1x1x1024xf32>
    %3604 = stablehlo.rsqrt %3603 : tensor<1x1x1x1024xf32>
    %3605 = stablehlo.reshape %arg528 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3606 = stablehlo.multiply %3604, %3605 : tensor<1x1x1x1024xf32>
    %3607 = stablehlo.broadcast_in_dim %3606, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3608 = stablehlo.multiply %3601, %3607 : tensor<256x14x14x1024xf32>
    %3609 = stablehlo.reshape %arg527 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3610 = stablehlo.broadcast_in_dim %3609, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3611 = stablehlo.add %3608, %3610 : tensor<256x14x14x1024xf32>
    %3612 = stablehlo.convert %3611 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3613 = stablehlo.add %3556, %3612 : tensor<256x14x14x1024xbf16>
    %3614 = call @relu_193(%3613) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3615 = stablehlo.convert %arg538 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3616 = stablehlo.convolution(%3614, %3615) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3617 = stablehlo.broadcast_in_dim %arg1578, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3618 = stablehlo.broadcast_in_dim %arg1579, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3619 = stablehlo.convert %3616 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3620 = stablehlo.broadcast_in_dim %3617, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3621 = stablehlo.subtract %3619, %3620 : tensor<256x14x14x256xf32>
    %3622 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3623 = stablehlo.add %3618, %3622 : tensor<1x1x1x256xf32>
    %3624 = stablehlo.rsqrt %3623 : tensor<1x1x1x256xf32>
    %3625 = stablehlo.reshape %arg533 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3626 = stablehlo.multiply %3624, %3625 : tensor<1x1x1x256xf32>
    %3627 = stablehlo.broadcast_in_dim %3626, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3628 = stablehlo.multiply %3621, %3627 : tensor<256x14x14x256xf32>
    %3629 = stablehlo.reshape %arg532 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3630 = stablehlo.broadcast_in_dim %3629, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3631 = stablehlo.add %3628, %3630 : tensor<256x14x14x256xf32>
    %3632 = stablehlo.convert %3631 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3633 = call @relu_175(%3632) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3634 = stablehlo.convert %arg539 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3635 = stablehlo.convolution(%3633, %3634) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3636 = stablehlo.broadcast_in_dim %arg1580, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3637 = stablehlo.broadcast_in_dim %arg1581, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3638 = stablehlo.convert %3635 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3639 = stablehlo.broadcast_in_dim %3636, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3640 = stablehlo.subtract %3638, %3639 : tensor<256x14x14x256xf32>
    %3641 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3642 = stablehlo.add %3637, %3641 : tensor<1x1x1x256xf32>
    %3643 = stablehlo.rsqrt %3642 : tensor<1x1x1x256xf32>
    %3644 = stablehlo.reshape %arg535 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3645 = stablehlo.multiply %3643, %3644 : tensor<1x1x1x256xf32>
    %3646 = stablehlo.broadcast_in_dim %3645, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3647 = stablehlo.multiply %3640, %3646 : tensor<256x14x14x256xf32>
    %3648 = stablehlo.reshape %arg534 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3649 = stablehlo.broadcast_in_dim %3648, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3650 = stablehlo.add %3647, %3649 : tensor<256x14x14x256xf32>
    %3651 = stablehlo.convert %3650 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3652 = call @relu_175(%3651) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3653 = stablehlo.convert %arg540 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3654 = stablehlo.convolution(%3652, %3653) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3655 = stablehlo.broadcast_in_dim %arg1582, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3656 = stablehlo.broadcast_in_dim %arg1583, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3657 = stablehlo.convert %3654 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3658 = stablehlo.broadcast_in_dim %3655, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3659 = stablehlo.subtract %3657, %3658 : tensor<256x14x14x1024xf32>
    %3660 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3661 = stablehlo.add %3656, %3660 : tensor<1x1x1x1024xf32>
    %3662 = stablehlo.rsqrt %3661 : tensor<1x1x1x1024xf32>
    %3663 = stablehlo.reshape %arg537 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3664 = stablehlo.multiply %3662, %3663 : tensor<1x1x1x1024xf32>
    %3665 = stablehlo.broadcast_in_dim %3664, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3666 = stablehlo.multiply %3659, %3665 : tensor<256x14x14x1024xf32>
    %3667 = stablehlo.reshape %arg536 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3668 = stablehlo.broadcast_in_dim %3667, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3669 = stablehlo.add %3666, %3668 : tensor<256x14x14x1024xf32>
    %3670 = stablehlo.convert %3669 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3671 = stablehlo.add %3614, %3670 : tensor<256x14x14x1024xbf16>
    %3672 = call @relu_193(%3671) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3673 = stablehlo.convert %arg547 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xbf16>
    %3674 = stablehlo.convolution(%3672, %3673) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3675 = stablehlo.broadcast_in_dim %arg1584, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3676 = stablehlo.broadcast_in_dim %arg1585, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3677 = stablehlo.convert %3674 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3678 = stablehlo.broadcast_in_dim %3675, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3679 = stablehlo.subtract %3677, %3678 : tensor<256x14x14x256xf32>
    %3680 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3681 = stablehlo.add %3676, %3680 : tensor<1x1x1x256xf32>
    %3682 = stablehlo.rsqrt %3681 : tensor<1x1x1x256xf32>
    %3683 = stablehlo.reshape %arg542 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3684 = stablehlo.multiply %3682, %3683 : tensor<1x1x1x256xf32>
    %3685 = stablehlo.broadcast_in_dim %3684, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3686 = stablehlo.multiply %3679, %3685 : tensor<256x14x14x256xf32>
    %3687 = stablehlo.reshape %arg541 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3688 = stablehlo.broadcast_in_dim %3687, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3689 = stablehlo.add %3686, %3688 : tensor<256x14x14x256xf32>
    %3690 = stablehlo.convert %3689 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3691 = call @relu_175(%3690) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3692 = stablehlo.convert %arg548 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %3693 = stablehlo.convolution(%3691, %3692) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3694 = stablehlo.broadcast_in_dim %arg1586, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3695 = stablehlo.broadcast_in_dim %arg1587, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3696 = stablehlo.convert %3693 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %3697 = stablehlo.broadcast_in_dim %3694, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3698 = stablehlo.subtract %3696, %3697 : tensor<256x14x14x256xf32>
    %3699 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %3700 = stablehlo.add %3695, %3699 : tensor<1x1x1x256xf32>
    %3701 = stablehlo.rsqrt %3700 : tensor<1x1x1x256xf32>
    %3702 = stablehlo.reshape %arg544 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3703 = stablehlo.multiply %3701, %3702 : tensor<1x1x1x256xf32>
    %3704 = stablehlo.broadcast_in_dim %3703, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3705 = stablehlo.multiply %3698, %3704 : tensor<256x14x14x256xf32>
    %3706 = stablehlo.reshape %arg543 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3707 = stablehlo.broadcast_in_dim %3706, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3708 = stablehlo.add %3705, %3707 : tensor<256x14x14x256xf32>
    %3709 = stablehlo.convert %3708 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3710 = call @relu_175(%3709) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3711 = stablehlo.convert %arg549 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xbf16>
    %3712 = stablehlo.convolution(%3710, %3711) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3713 = stablehlo.broadcast_in_dim %arg1588, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3714 = stablehlo.broadcast_in_dim %arg1589, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3715 = stablehlo.convert %3712 : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xf32>
    %3716 = stablehlo.broadcast_in_dim %3713, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3717 = stablehlo.subtract %3715, %3716 : tensor<256x14x14x1024xf32>
    %3718 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
    %3719 = stablehlo.add %3714, %3718 : tensor<1x1x1x1024xf32>
    %3720 = stablehlo.rsqrt %3719 : tensor<1x1x1x1024xf32>
    %3721 = stablehlo.reshape %arg546 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3722 = stablehlo.multiply %3720, %3721 : tensor<1x1x1x1024xf32>
    %3723 = stablehlo.broadcast_in_dim %3722, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3724 = stablehlo.multiply %3717, %3723 : tensor<256x14x14x1024xf32>
    %3725 = stablehlo.reshape %arg545 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %3726 = stablehlo.broadcast_in_dim %3725, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
    %3727 = stablehlo.add %3724, %3726 : tensor<256x14x14x1024xf32>
    %3728 = stablehlo.convert %3727 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xbf16>
    %3729 = stablehlo.add %3672, %3728 : tensor<256x14x14x1024xbf16>
    %3730 = call @relu_193(%3729) : (tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16>
    %3731 = stablehlo.convert %arg556 : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xbf16>
    %3732 = stablehlo.convolution(%3730, %3731) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x512xbf16>) -> tensor<256x14x14x512xbf16>
    %3733 = stablehlo.broadcast_in_dim %arg1590, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3734 = stablehlo.broadcast_in_dim %arg1591, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3735 = stablehlo.convert %3732 : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xf32>
    %3736 = stablehlo.broadcast_in_dim %3733, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
    %3737 = stablehlo.subtract %3735, %3736 : tensor<256x14x14x512xf32>
    %3738 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %3739 = stablehlo.add %3734, %3738 : tensor<1x1x1x512xf32>
    %3740 = stablehlo.rsqrt %3739 : tensor<1x1x1x512xf32>
    %3741 = stablehlo.reshape %arg551 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3742 = stablehlo.multiply %3740, %3741 : tensor<1x1x1x512xf32>
    %3743 = stablehlo.broadcast_in_dim %3742, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
    %3744 = stablehlo.multiply %3737, %3743 : tensor<256x14x14x512xf32>
    %3745 = stablehlo.reshape %arg550 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3746 = stablehlo.broadcast_in_dim %3745, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
    %3747 = stablehlo.add %3744, %3746 : tensor<256x14x14x512xf32>
    %3748 = stablehlo.convert %3747 : (tensor<256x14x14x512xf32>) -> tensor<256x14x14x512xbf16>
    %3749 = call @relu_311(%3748) : (tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xbf16>
    %3750 = stablehlo.convert %arg557 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %3751 = stablehlo.convolution(%3749, %3750) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3752 = stablehlo.broadcast_in_dim %arg1592, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3753 = stablehlo.broadcast_in_dim %arg1593, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3754 = stablehlo.convert %3751 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %3755 = stablehlo.broadcast_in_dim %3752, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3756 = stablehlo.subtract %3754, %3755 : tensor<256x7x7x512xf32>
    %3757 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %3758 = stablehlo.add %3753, %3757 : tensor<1x1x1x512xf32>
    %3759 = stablehlo.rsqrt %3758 : tensor<1x1x1x512xf32>
    %3760 = stablehlo.reshape %arg553 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3761 = stablehlo.multiply %3759, %3760 : tensor<1x1x1x512xf32>
    %3762 = stablehlo.broadcast_in_dim %3761, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3763 = stablehlo.multiply %3756, %3762 : tensor<256x7x7x512xf32>
    %3764 = stablehlo.reshape %arg552 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3765 = stablehlo.broadcast_in_dim %3764, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3766 = stablehlo.add %3763, %3765 : tensor<256x7x7x512xf32>
    %3767 = stablehlo.convert %3766 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3768 = call @relu_321(%3767) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3769 = stablehlo.convert %arg558 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xbf16>
    %3770 = stablehlo.convolution(%3768, %3769) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<1x1x512x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %3771 = stablehlo.broadcast_in_dim %arg1594, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3772 = stablehlo.broadcast_in_dim %arg1595, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3773 = stablehlo.convert %3770 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %3774 = stablehlo.broadcast_in_dim %3771, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3775 = stablehlo.subtract %3773, %3774 : tensor<256x7x7x2048xf32>
    %3776 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %3777 = stablehlo.add %3772, %3776 : tensor<1x1x1x2048xf32>
    %3778 = stablehlo.rsqrt %3777 : tensor<1x1x1x2048xf32>
    %3779 = stablehlo.reshape %arg555 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3780 = stablehlo.multiply %3778, %3779 : tensor<1x1x1x2048xf32>
    %3781 = stablehlo.broadcast_in_dim %3780, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3782 = stablehlo.multiply %3775, %3781 : tensor<256x7x7x2048xf32>
    %3783 = stablehlo.reshape %arg554 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3784 = stablehlo.broadcast_in_dim %3783, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3785 = stablehlo.add %3782, %3784 : tensor<256x7x7x2048xf32>
    %3786 = stablehlo.convert %3785 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    %3787 = stablehlo.convert %arg559 : (tensor<1x1x1024x2048xf32>) -> tensor<1x1x1024x2048xbf16>
    %3788 = stablehlo.convolution(%3730, %3787) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xbf16>, tensor<1x1x1024x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %3789 = stablehlo.broadcast_in_dim %arg1596, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3790 = stablehlo.broadcast_in_dim %arg1597, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3791 = stablehlo.convert %3788 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %3792 = stablehlo.broadcast_in_dim %3789, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3793 = stablehlo.subtract %3791, %3792 : tensor<256x7x7x2048xf32>
    %3794 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %3795 = stablehlo.add %3790, %3794 : tensor<1x1x1x2048xf32>
    %3796 = stablehlo.rsqrt %3795 : tensor<1x1x1x2048xf32>
    %3797 = stablehlo.reshape %arg561 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3798 = stablehlo.multiply %3796, %3797 : tensor<1x1x1x2048xf32>
    %3799 = stablehlo.broadcast_in_dim %3798, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3800 = stablehlo.multiply %3793, %3799 : tensor<256x7x7x2048xf32>
    %3801 = stablehlo.reshape %arg560 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3802 = stablehlo.broadcast_in_dim %3801, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3803 = stablehlo.add %3800, %3802 : tensor<256x7x7x2048xf32>
    %3804 = stablehlo.convert %3803 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    %3805 = stablehlo.add %3804, %3786 : tensor<256x7x7x2048xbf16>
    %3806 = call @relu_339(%3805) : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %3807 = stablehlo.convert %arg568 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xbf16>
    %3808 = stablehlo.convolution(%3806, %3807) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xbf16>, tensor<1x1x2048x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3809 = stablehlo.broadcast_in_dim %arg1598, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3810 = stablehlo.broadcast_in_dim %arg1599, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3811 = stablehlo.convert %3808 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %3812 = stablehlo.broadcast_in_dim %3809, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3813 = stablehlo.subtract %3811, %3812 : tensor<256x7x7x512xf32>
    %3814 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %3815 = stablehlo.add %3810, %3814 : tensor<1x1x1x512xf32>
    %3816 = stablehlo.rsqrt %3815 : tensor<1x1x1x512xf32>
    %3817 = stablehlo.reshape %arg563 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3818 = stablehlo.multiply %3816, %3817 : tensor<1x1x1x512xf32>
    %3819 = stablehlo.broadcast_in_dim %3818, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3820 = stablehlo.multiply %3813, %3819 : tensor<256x7x7x512xf32>
    %3821 = stablehlo.reshape %arg562 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3822 = stablehlo.broadcast_in_dim %3821, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3823 = stablehlo.add %3820, %3822 : tensor<256x7x7x512xf32>
    %3824 = stablehlo.convert %3823 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3825 = call @relu_321(%3824) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3826 = stablehlo.convert %arg569 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %3827 = stablehlo.convolution(%3825, %3826) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3828 = stablehlo.broadcast_in_dim %arg1600, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3829 = stablehlo.broadcast_in_dim %arg1601, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3830 = stablehlo.convert %3827 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %3831 = stablehlo.broadcast_in_dim %3828, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3832 = stablehlo.subtract %3830, %3831 : tensor<256x7x7x512xf32>
    %3833 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %3834 = stablehlo.add %3829, %3833 : tensor<1x1x1x512xf32>
    %3835 = stablehlo.rsqrt %3834 : tensor<1x1x1x512xf32>
    %3836 = stablehlo.reshape %arg565 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3837 = stablehlo.multiply %3835, %3836 : tensor<1x1x1x512xf32>
    %3838 = stablehlo.broadcast_in_dim %3837, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3839 = stablehlo.multiply %3832, %3838 : tensor<256x7x7x512xf32>
    %3840 = stablehlo.reshape %arg564 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3841 = stablehlo.broadcast_in_dim %3840, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3842 = stablehlo.add %3839, %3841 : tensor<256x7x7x512xf32>
    %3843 = stablehlo.convert %3842 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3844 = call @relu_321(%3843) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3845 = stablehlo.convert %arg570 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xbf16>
    %3846 = stablehlo.convolution(%3844, %3845) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<1x1x512x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %3847 = stablehlo.broadcast_in_dim %arg1602, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3848 = stablehlo.broadcast_in_dim %arg1603, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3849 = stablehlo.convert %3846 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %3850 = stablehlo.broadcast_in_dim %3847, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3851 = stablehlo.subtract %3849, %3850 : tensor<256x7x7x2048xf32>
    %3852 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %3853 = stablehlo.add %3848, %3852 : tensor<1x1x1x2048xf32>
    %3854 = stablehlo.rsqrt %3853 : tensor<1x1x1x2048xf32>
    %3855 = stablehlo.reshape %arg567 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3856 = stablehlo.multiply %3854, %3855 : tensor<1x1x1x2048xf32>
    %3857 = stablehlo.broadcast_in_dim %3856, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3858 = stablehlo.multiply %3851, %3857 : tensor<256x7x7x2048xf32>
    %3859 = stablehlo.reshape %arg566 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3860 = stablehlo.broadcast_in_dim %3859, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3861 = stablehlo.add %3858, %3860 : tensor<256x7x7x2048xf32>
    %3862 = stablehlo.convert %3861 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    %3863 = stablehlo.add %3806, %3862 : tensor<256x7x7x2048xbf16>
    %3864 = call @relu_339(%3863) : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %3865 = stablehlo.convert %arg577 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xbf16>
    %3866 = stablehlo.convolution(%3864, %3865) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xbf16>, tensor<1x1x2048x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3867 = stablehlo.broadcast_in_dim %arg1604, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3868 = stablehlo.broadcast_in_dim %arg1605, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3869 = stablehlo.convert %3866 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %3870 = stablehlo.broadcast_in_dim %3867, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3871 = stablehlo.subtract %3869, %3870 : tensor<256x7x7x512xf32>
    %3872 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %3873 = stablehlo.add %3868, %3872 : tensor<1x1x1x512xf32>
    %3874 = stablehlo.rsqrt %3873 : tensor<1x1x1x512xf32>
    %3875 = stablehlo.reshape %arg572 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3876 = stablehlo.multiply %3874, %3875 : tensor<1x1x1x512xf32>
    %3877 = stablehlo.broadcast_in_dim %3876, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3878 = stablehlo.multiply %3871, %3877 : tensor<256x7x7x512xf32>
    %3879 = stablehlo.reshape %arg571 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3880 = stablehlo.broadcast_in_dim %3879, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3881 = stablehlo.add %3878, %3880 : tensor<256x7x7x512xf32>
    %3882 = stablehlo.convert %3881 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3883 = call @relu_321(%3882) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3884 = stablehlo.convert %arg578 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %3885 = stablehlo.convolution(%3883, %3884) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3886 = stablehlo.broadcast_in_dim %arg1606, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3887 = stablehlo.broadcast_in_dim %arg1607, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3888 = stablehlo.convert %3885 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %3889 = stablehlo.broadcast_in_dim %3886, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3890 = stablehlo.subtract %3888, %3889 : tensor<256x7x7x512xf32>
    %3891 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %3892 = stablehlo.add %3887, %3891 : tensor<1x1x1x512xf32>
    %3893 = stablehlo.rsqrt %3892 : tensor<1x1x1x512xf32>
    %3894 = stablehlo.reshape %arg574 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3895 = stablehlo.multiply %3893, %3894 : tensor<1x1x1x512xf32>
    %3896 = stablehlo.broadcast_in_dim %3895, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3897 = stablehlo.multiply %3890, %3896 : tensor<256x7x7x512xf32>
    %3898 = stablehlo.reshape %arg573 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3899 = stablehlo.broadcast_in_dim %3898, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3900 = stablehlo.add %3897, %3899 : tensor<256x7x7x512xf32>
    %3901 = stablehlo.convert %3900 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3902 = call @relu_321(%3901) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3903 = stablehlo.convert %arg579 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xbf16>
    %3904 = stablehlo.convolution(%3902, %3903) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<1x1x512x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %3905 = stablehlo.broadcast_in_dim %arg1608, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3906 = stablehlo.broadcast_in_dim %arg1609, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3907 = stablehlo.convert %3904 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %3908 = stablehlo.broadcast_in_dim %3905, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3909 = stablehlo.subtract %3907, %3908 : tensor<256x7x7x2048xf32>
    %3910 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
    %3911 = stablehlo.add %3906, %3910 : tensor<1x1x1x2048xf32>
    %3912 = stablehlo.rsqrt %3911 : tensor<1x1x1x2048xf32>
    %3913 = stablehlo.reshape %arg576 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3914 = stablehlo.multiply %3912, %3913 : tensor<1x1x1x2048xf32>
    %3915 = stablehlo.broadcast_in_dim %3914, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3916 = stablehlo.multiply %3909, %3915 : tensor<256x7x7x2048xf32>
    %3917 = stablehlo.reshape %arg575 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
    %3918 = stablehlo.broadcast_in_dim %3917, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
    %3919 = stablehlo.add %3916, %3918 : tensor<256x7x7x2048xf32>
    %3920 = stablehlo.convert %3919 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xbf16>
    %3921 = stablehlo.add %3864, %3920 : tensor<256x7x7x2048xbf16>
    %3922 = call @relu_339(%3921) : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16>
    %3923 = stablehlo.convert %3922 : (tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3924 = stablehlo.reduce(%3923 init: %cst_1) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x2048xf32>, tensor<f32>) -> tensor<256x2048xf32>
    %cst_2 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
    %3925 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
    %3926 = stablehlo.divide %3924, %3925 : tensor<256x2048xf32>
    %3927 = stablehlo.convert %3926 : (tensor<256x2048xf32>) -> tensor<256x2048xbf16>
    %3928 = stablehlo.convert %arg608 : (tensor<2048x1000xf32>) -> tensor<2048x1000xbf16>
    %3929 = stablehlo.convert %arg607 : (tensor<1000xf32>) -> tensor<1000xbf16>
    %3930 = stablehlo.dot_general %3927, %3928, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xbf16>, tensor<2048x1000xbf16>) -> tensor<256x1000xbf16>
    %3931 = stablehlo.reshape %3929 : (tensor<1000xbf16>) -> tensor<1x1000xbf16>
    %3932 = stablehlo.broadcast_in_dim %3931, dims = [0, 1] : (tensor<1x1000xbf16>) -> tensor<256x1000xbf16>
    %3933 = stablehlo.add %3930, %3932 : tensor<256x1000xbf16>
    %3934 = stablehlo.broadcast_in_dim %3933, dims = [1, 2] : (tensor<256x1000xbf16>) -> tensor<1x256x1000xbf16>
    return %3934 : tensor<1x256x1000xbf16>
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
  func.func private @relu_165(%arg0: tensor<256x28x28x256xbf16>) -> tensor<256x28x28x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x256xbf16>
    return %1 : tensor<256x28x28x256xbf16>
  }
  func.func private @relu_175(%arg0: tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xbf16>
    return %1 : tensor<256x14x14x256xbf16>
  }
  func.func private @relu_193(%arg0: tensor<256x14x14x1024xbf16>) -> tensor<256x14x14x1024xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x1024xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x1024xbf16>
    return %1 : tensor<256x14x14x1024xbf16>
  }
  func.func private @relu_311(%arg0: tensor<256x14x14x512xbf16>) -> tensor<256x14x14x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x512xbf16>
    return %1 : tensor<256x14x14x512xbf16>
  }
  func.func private @relu_321(%arg0: tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xbf16>
    return %1 : tensor<256x7x7x512xbf16>
  }
  func.func private @relu_339(%arg0: tensor<256x7x7x2048xbf16>) -> tensor<256x7x7x2048xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x2048xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x2048xbf16>
    return %1 : tensor<256x7x7x2048xbf16>
  }
}
