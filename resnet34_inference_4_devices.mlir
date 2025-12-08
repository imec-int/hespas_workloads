module @pmap_inference_step attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg1: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg2: tensor<512x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg3: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg4: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg5: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg6: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg7: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg8: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg9: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg10: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg11: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg12: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg13: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg14: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg15: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg16: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg17: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg18: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg19: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg20: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg21: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg22: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg23: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg24: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg25: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg26: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg27: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg28: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg29: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg30: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg31: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg32: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg33: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg34: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg35: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg36: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg37: tensor<3x3x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg38: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg39: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg40: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg41: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg42: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg43: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg44: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg45: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg46: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg47: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg48: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg49: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg50: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg51: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg52: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg53: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg54: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg55: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg56: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg57: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg58: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg59: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg60: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg61: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg62: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg63: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg64: tensor<3x3x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg65: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg66: tensor<1x1x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg67: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg68: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg69: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg70: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg71: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg72: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg73: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg74: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg75: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg76: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg77: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg78: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg79: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg80: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg81: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg82: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg83: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg84: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg85: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg86: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg87: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg88: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg89: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg90: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg91: tensor<3x3x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg92: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg93: tensor<1x1x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg94: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg95: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg96: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg97: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg98: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg99: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg100: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg101: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg102: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg103: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg104: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg105: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg106: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg107: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg108: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg109: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg110: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg111: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg112: tensor<512x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg113: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg114: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg115: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg116: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg117: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg118: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg119: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg120: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg121: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg122: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg123: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg124: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg125: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg126: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg127: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg128: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg129: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg130: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg131: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg132: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg133: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg134: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg135: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg136: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg137: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg138: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg139: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg140: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg141: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg142: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg143: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg144: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg145: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg146: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg147: tensor<3x3x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg148: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg149: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg150: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg151: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg152: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg153: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg154: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg155: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg156: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg157: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg158: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg159: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg160: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg161: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg162: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg163: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg164: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg165: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg166: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg167: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg168: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg169: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg170: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg171: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg172: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg173: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg174: tensor<3x3x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg175: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg176: tensor<1x1x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg177: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg178: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg179: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg180: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg181: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg182: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg183: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg184: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg185: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg186: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg187: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg188: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg189: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg190: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg191: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg192: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg193: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg194: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg195: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg196: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg197: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg198: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg199: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg200: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg201: tensor<3x3x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg202: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg203: tensor<1x1x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg204: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg205: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg206: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg207: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg208: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg209: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg210: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg211: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg212: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg213: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg214: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg215: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg216: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg217: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg218: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg219: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg220: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg221: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg222: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg223: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg224: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg225: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg226: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg227: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg228: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg229: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg230: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg231: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg232: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg233: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg234: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg235: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg236: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg237: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg238: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg239: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg240: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg241: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg242: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg243: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg244: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg245: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg246: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg247: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg248: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg249: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg250: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg251: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg252: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg253: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg254: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg255: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg256: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg257: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg258: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg259: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg260: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg261: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg262: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg263: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg264: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg265: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg266: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg267: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg268: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg269: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg270: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg271: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg272: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg273: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg274: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg275: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg276: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg277: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg278: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg279: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg280: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg281: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg282: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg283: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg284: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg285: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg286: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg287: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg288: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg289: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg290: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg291: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg292: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg293: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg294: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg295: tensor<f32> {mhlo.is_same_data_across_replicas = true}, %arg296: tensor<1x256x224x224x3xbf16>, %arg297: tensor<1x256xi32>) -> (tensor<1x256x1000xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.reshape %arg296 : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
    %1 = stablehlo.reshape %arg297 : (tensor<1x256xi32>) -> tensor<256xi32>
    %2 = stablehlo.convert %arg110 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xbf16>
    %3 = stablehlo.convolution(%0, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xbf16>, tensor<7x7x3x64xbf16>) -> tensor<256x112x112x64xbf16>
    %4 = stablehlo.broadcast_in_dim %arg292, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg293, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %6 = stablehlo.convert %3 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
    %7 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %8 = stablehlo.subtract %6, %7 : tensor<256x112x112x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %10 = stablehlo.add %5, %9 : tensor<1x1x1x64xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x1x1x64xf32>
    %12 = stablehlo.reshape %arg109 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<1x1x1x64xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %15 = stablehlo.multiply %8, %14 : tensor<256x112x112x64xf32>
    %16 = stablehlo.reshape %arg108 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %18 = stablehlo.add %15, %17 : tensor<256x112x112x64xf32>
    %19 = stablehlo.convert %18 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %20 = call @relu(%19) : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xbf16>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %21 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %22 = "stablehlo.reduce_window"(%20, %21) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg298: tensor<bf16>, %arg299: tensor<bf16>):
      %713 = stablehlo.maximum %arg298, %arg299 : tensor<bf16>
      stablehlo.return %713 : tensor<bf16>
    }) : (tensor<256x112x112x64xbf16>, tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %23 = stablehlo.convert %arg7 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %24 = stablehlo.convolution(%22, %23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %25 = stablehlo.broadcast_in_dim %arg222, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %26 = stablehlo.broadcast_in_dim %arg223, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %44 = stablehlo.broadcast_in_dim %arg224, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %45 = stablehlo.broadcast_in_dim %arg225, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %64 = stablehlo.broadcast_in_dim %arg226, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %65 = stablehlo.broadcast_in_dim %arg227, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %83 = stablehlo.broadcast_in_dim %arg228, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %84 = stablehlo.broadcast_in_dim %arg229, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    %101 = stablehlo.convert %arg58 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %102 = stablehlo.convolution(%100, %101) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %103 = stablehlo.broadcast_in_dim %arg256, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %104 = stablehlo.broadcast_in_dim %arg257, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %105 = stablehlo.convert %102 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %106 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %107 = stablehlo.subtract %105, %106 : tensor<256x56x56x64xf32>
    %108 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %109 = stablehlo.add %104, %108 : tensor<1x1x1x64xf32>
    %110 = stablehlo.rsqrt %109 : tensor<1x1x1x64xf32>
    %111 = stablehlo.reshape %arg55 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %112 = stablehlo.multiply %110, %111 : tensor<1x1x1x64xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %114 = stablehlo.multiply %107, %113 : tensor<256x56x56x64xf32>
    %115 = stablehlo.reshape %arg54 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %117 = stablehlo.add %114, %116 : tensor<256x56x56x64xf32>
    %118 = stablehlo.convert %117 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %119 = call @relu_13(%118) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %120 = stablehlo.convert %arg59 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %121 = stablehlo.convolution(%119, %120) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %122 = stablehlo.broadcast_in_dim %arg258, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %123 = stablehlo.broadcast_in_dim %arg259, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %124 = stablehlo.convert %121 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %125 = stablehlo.broadcast_in_dim %122, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %126 = stablehlo.subtract %124, %125 : tensor<256x56x56x64xf32>
    %127 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %128 = stablehlo.add %123, %127 : tensor<1x1x1x64xf32>
    %129 = stablehlo.rsqrt %128 : tensor<1x1x1x64xf32>
    %130 = stablehlo.reshape %arg57 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %131 = stablehlo.multiply %129, %130 : tensor<1x1x1x64xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %133 = stablehlo.multiply %126, %132 : tensor<256x56x56x64xf32>
    %134 = stablehlo.reshape %arg56 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %135 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %136 = stablehlo.add %133, %135 : tensor<256x56x56x64xf32>
    %137 = stablehlo.convert %136 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %138 = stablehlo.add %100, %137 : tensor<256x56x56x64xbf16>
    %139 = call @relu_13(%138) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %140 = stablehlo.convert %arg64 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xbf16>
    %141 = stablehlo.convolution(%139, %140) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x128xbf16>) -> tensor<256x28x28x128xbf16>
    %142 = stablehlo.broadcast_in_dim %arg260, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %143 = stablehlo.broadcast_in_dim %arg261, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %144 = stablehlo.convert %141 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %145 = stablehlo.broadcast_in_dim %142, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %146 = stablehlo.subtract %144, %145 : tensor<256x28x28x128xf32>
    %147 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %148 = stablehlo.add %143, %147 : tensor<1x1x1x128xf32>
    %149 = stablehlo.rsqrt %148 : tensor<1x1x1x128xf32>
    %150 = stablehlo.reshape %arg61 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %151 = stablehlo.multiply %149, %150 : tensor<1x1x1x128xf32>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %153 = stablehlo.multiply %146, %152 : tensor<256x28x28x128xf32>
    %154 = stablehlo.reshape %arg60 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %156 = stablehlo.add %153, %155 : tensor<256x28x28x128xf32>
    %157 = stablehlo.convert %156 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %158 = call @relu_34(%157) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %159 = stablehlo.convert %arg65 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %160 = stablehlo.convolution(%158, %159) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %161 = stablehlo.broadcast_in_dim %arg262, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %162 = stablehlo.broadcast_in_dim %arg263, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %163 = stablehlo.convert %160 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %164 = stablehlo.broadcast_in_dim %161, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %165 = stablehlo.subtract %163, %164 : tensor<256x28x28x128xf32>
    %166 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %167 = stablehlo.add %162, %166 : tensor<1x1x1x128xf32>
    %168 = stablehlo.rsqrt %167 : tensor<1x1x1x128xf32>
    %169 = stablehlo.reshape %arg63 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %170 = stablehlo.multiply %168, %169 : tensor<1x1x1x128xf32>
    %171 = stablehlo.broadcast_in_dim %170, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %172 = stablehlo.multiply %165, %171 : tensor<256x28x28x128xf32>
    %173 = stablehlo.reshape %arg62 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %174 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %175 = stablehlo.add %172, %174 : tensor<256x28x28x128xf32>
    %176 = stablehlo.convert %175 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %177 = stablehlo.convert %arg66 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xbf16>
    %178 = stablehlo.convolution(%139, %177) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x128xbf16>) -> tensor<256x28x28x128xbf16>
    %179 = stablehlo.broadcast_in_dim %arg264, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %180 = stablehlo.broadcast_in_dim %arg265, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %181 = stablehlo.convert %178 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %182 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %183 = stablehlo.subtract %181, %182 : tensor<256x28x28x128xf32>
    %184 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %185 = stablehlo.add %180, %184 : tensor<1x1x1x128xf32>
    %186 = stablehlo.rsqrt %185 : tensor<1x1x1x128xf32>
    %187 = stablehlo.reshape %arg68 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %188 = stablehlo.multiply %186, %187 : tensor<1x1x1x128xf32>
    %189 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %190 = stablehlo.multiply %183, %189 : tensor<256x28x28x128xf32>
    %191 = stablehlo.reshape %arg67 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %193 = stablehlo.add %190, %192 : tensor<256x28x28x128xf32>
    %194 = stablehlo.convert %193 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %195 = stablehlo.add %194, %176 : tensor<256x28x28x128xbf16>
    %196 = call @relu_34(%195) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %197 = stablehlo.convert %arg73 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %198 = stablehlo.convolution(%196, %197) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %199 = stablehlo.broadcast_in_dim %arg266, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %200 = stablehlo.broadcast_in_dim %arg267, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %201 = stablehlo.convert %198 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %202 = stablehlo.broadcast_in_dim %199, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %203 = stablehlo.subtract %201, %202 : tensor<256x28x28x128xf32>
    %204 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %205 = stablehlo.add %200, %204 : tensor<1x1x1x128xf32>
    %206 = stablehlo.rsqrt %205 : tensor<1x1x1x128xf32>
    %207 = stablehlo.reshape %arg70 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %208 = stablehlo.multiply %206, %207 : tensor<1x1x1x128xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %210 = stablehlo.multiply %203, %209 : tensor<256x28x28x128xf32>
    %211 = stablehlo.reshape %arg69 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %212 = stablehlo.broadcast_in_dim %211, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %213 = stablehlo.add %210, %212 : tensor<256x28x28x128xf32>
    %214 = stablehlo.convert %213 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %215 = call @relu_34(%214) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %216 = stablehlo.convert %arg74 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %217 = stablehlo.convolution(%215, %216) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %218 = stablehlo.broadcast_in_dim %arg268, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %219 = stablehlo.broadcast_in_dim %arg269, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %220 = stablehlo.convert %217 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %221 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %222 = stablehlo.subtract %220, %221 : tensor<256x28x28x128xf32>
    %223 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %224 = stablehlo.add %219, %223 : tensor<1x1x1x128xf32>
    %225 = stablehlo.rsqrt %224 : tensor<1x1x1x128xf32>
    %226 = stablehlo.reshape %arg72 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %227 = stablehlo.multiply %225, %226 : tensor<1x1x1x128xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %229 = stablehlo.multiply %222, %228 : tensor<256x28x28x128xf32>
    %230 = stablehlo.reshape %arg71 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %231 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %232 = stablehlo.add %229, %231 : tensor<256x28x28x128xf32>
    %233 = stablehlo.convert %232 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %234 = stablehlo.add %196, %233 : tensor<256x28x28x128xbf16>
    %235 = call @relu_34(%234) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %236 = stablehlo.convert %arg79 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %237 = stablehlo.convolution(%235, %236) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %238 = stablehlo.broadcast_in_dim %arg270, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %239 = stablehlo.broadcast_in_dim %arg271, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %240 = stablehlo.convert %237 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %241 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %242 = stablehlo.subtract %240, %241 : tensor<256x28x28x128xf32>
    %243 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %244 = stablehlo.add %239, %243 : tensor<1x1x1x128xf32>
    %245 = stablehlo.rsqrt %244 : tensor<1x1x1x128xf32>
    %246 = stablehlo.reshape %arg76 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %247 = stablehlo.multiply %245, %246 : tensor<1x1x1x128xf32>
    %248 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %249 = stablehlo.multiply %242, %248 : tensor<256x28x28x128xf32>
    %250 = stablehlo.reshape %arg75 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %252 = stablehlo.add %249, %251 : tensor<256x28x28x128xf32>
    %253 = stablehlo.convert %252 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %254 = call @relu_34(%253) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %255 = stablehlo.convert %arg80 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %256 = stablehlo.convolution(%254, %255) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %257 = stablehlo.broadcast_in_dim %arg272, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %258 = stablehlo.broadcast_in_dim %arg273, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %259 = stablehlo.convert %256 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %260 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %261 = stablehlo.subtract %259, %260 : tensor<256x28x28x128xf32>
    %262 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %263 = stablehlo.add %258, %262 : tensor<1x1x1x128xf32>
    %264 = stablehlo.rsqrt %263 : tensor<1x1x1x128xf32>
    %265 = stablehlo.reshape %arg78 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %266 = stablehlo.multiply %264, %265 : tensor<1x1x1x128xf32>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %268 = stablehlo.multiply %261, %267 : tensor<256x28x28x128xf32>
    %269 = stablehlo.reshape %arg77 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %271 = stablehlo.add %268, %270 : tensor<256x28x28x128xf32>
    %272 = stablehlo.convert %271 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %273 = stablehlo.add %235, %272 : tensor<256x28x28x128xbf16>
    %274 = call @relu_34(%273) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %275 = stablehlo.convert %arg85 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %276 = stablehlo.convolution(%274, %275) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %277 = stablehlo.broadcast_in_dim %arg274, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %278 = stablehlo.broadcast_in_dim %arg275, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %279 = stablehlo.convert %276 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %280 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %281 = stablehlo.subtract %279, %280 : tensor<256x28x28x128xf32>
    %282 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %283 = stablehlo.add %278, %282 : tensor<1x1x1x128xf32>
    %284 = stablehlo.rsqrt %283 : tensor<1x1x1x128xf32>
    %285 = stablehlo.reshape %arg82 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %286 = stablehlo.multiply %284, %285 : tensor<1x1x1x128xf32>
    %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %288 = stablehlo.multiply %281, %287 : tensor<256x28x28x128xf32>
    %289 = stablehlo.reshape %arg81 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %291 = stablehlo.add %288, %290 : tensor<256x28x28x128xf32>
    %292 = stablehlo.convert %291 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %293 = call @relu_34(%292) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %294 = stablehlo.convert %arg86 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %295 = stablehlo.convolution(%293, %294) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %296 = stablehlo.broadcast_in_dim %arg276, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %297 = stablehlo.broadcast_in_dim %arg277, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %298 = stablehlo.convert %295 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %299 = stablehlo.broadcast_in_dim %296, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %300 = stablehlo.subtract %298, %299 : tensor<256x28x28x128xf32>
    %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %302 = stablehlo.add %297, %301 : tensor<1x1x1x128xf32>
    %303 = stablehlo.rsqrt %302 : tensor<1x1x1x128xf32>
    %304 = stablehlo.reshape %arg84 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %305 = stablehlo.multiply %303, %304 : tensor<1x1x1x128xf32>
    %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %307 = stablehlo.multiply %300, %306 : tensor<256x28x28x128xf32>
    %308 = stablehlo.reshape %arg83 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %310 = stablehlo.add %307, %309 : tensor<256x28x28x128xf32>
    %311 = stablehlo.convert %310 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %312 = stablehlo.add %274, %311 : tensor<256x28x28x128xbf16>
    %313 = call @relu_34(%312) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %314 = stablehlo.convert %arg91 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xbf16>
    %315 = stablehlo.convolution(%313, %314) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x256xbf16>) -> tensor<256x14x14x256xbf16>
    %316 = stablehlo.broadcast_in_dim %arg278, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %317 = stablehlo.broadcast_in_dim %arg279, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %318 = stablehlo.convert %315 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %319 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %320 = stablehlo.subtract %318, %319 : tensor<256x14x14x256xf32>
    %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %322 = stablehlo.add %317, %321 : tensor<1x1x1x256xf32>
    %323 = stablehlo.rsqrt %322 : tensor<1x1x1x256xf32>
    %324 = stablehlo.reshape %arg88 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %325 = stablehlo.multiply %323, %324 : tensor<1x1x1x256xf32>
    %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %327 = stablehlo.multiply %320, %326 : tensor<256x14x14x256xf32>
    %328 = stablehlo.reshape %arg87 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %330 = stablehlo.add %327, %329 : tensor<256x14x14x256xf32>
    %331 = stablehlo.convert %330 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %332 = call @relu_61(%331) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %333 = stablehlo.convert %arg92 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %334 = stablehlo.convolution(%332, %333) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %335 = stablehlo.broadcast_in_dim %arg280, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %336 = stablehlo.broadcast_in_dim %arg281, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %337 = stablehlo.convert %334 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %338 = stablehlo.broadcast_in_dim %335, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %339 = stablehlo.subtract %337, %338 : tensor<256x14x14x256xf32>
    %340 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %341 = stablehlo.add %336, %340 : tensor<1x1x1x256xf32>
    %342 = stablehlo.rsqrt %341 : tensor<1x1x1x256xf32>
    %343 = stablehlo.reshape %arg90 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %344 = stablehlo.multiply %342, %343 : tensor<1x1x1x256xf32>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %346 = stablehlo.multiply %339, %345 : tensor<256x14x14x256xf32>
    %347 = stablehlo.reshape %arg89 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %348 = stablehlo.broadcast_in_dim %347, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %349 = stablehlo.add %346, %348 : tensor<256x14x14x256xf32>
    %350 = stablehlo.convert %349 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %351 = stablehlo.convert %arg93 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xbf16>
    %352 = stablehlo.convolution(%313, %351) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x256xbf16>) -> tensor<256x14x14x256xbf16>
    %353 = stablehlo.broadcast_in_dim %arg282, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %354 = stablehlo.broadcast_in_dim %arg283, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %355 = stablehlo.convert %352 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %356 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %357 = stablehlo.subtract %355, %356 : tensor<256x14x14x256xf32>
    %358 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %359 = stablehlo.add %354, %358 : tensor<1x1x1x256xf32>
    %360 = stablehlo.rsqrt %359 : tensor<1x1x1x256xf32>
    %361 = stablehlo.reshape %arg95 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %362 = stablehlo.multiply %360, %361 : tensor<1x1x1x256xf32>
    %363 = stablehlo.broadcast_in_dim %362, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %364 = stablehlo.multiply %357, %363 : tensor<256x14x14x256xf32>
    %365 = stablehlo.reshape %arg94 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %366 = stablehlo.broadcast_in_dim %365, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %367 = stablehlo.add %364, %366 : tensor<256x14x14x256xf32>
    %368 = stablehlo.convert %367 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %369 = stablehlo.add %368, %350 : tensor<256x14x14x256xbf16>
    %370 = call @relu_61(%369) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %371 = stablehlo.convert %arg100 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %372 = stablehlo.convolution(%370, %371) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %373 = stablehlo.broadcast_in_dim %arg284, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %374 = stablehlo.broadcast_in_dim %arg285, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %375 = stablehlo.convert %372 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %376 = stablehlo.broadcast_in_dim %373, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %377 = stablehlo.subtract %375, %376 : tensor<256x14x14x256xf32>
    %378 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %379 = stablehlo.add %374, %378 : tensor<1x1x1x256xf32>
    %380 = stablehlo.rsqrt %379 : tensor<1x1x1x256xf32>
    %381 = stablehlo.reshape %arg97 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %382 = stablehlo.multiply %380, %381 : tensor<1x1x1x256xf32>
    %383 = stablehlo.broadcast_in_dim %382, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %384 = stablehlo.multiply %377, %383 : tensor<256x14x14x256xf32>
    %385 = stablehlo.reshape %arg96 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %386 = stablehlo.broadcast_in_dim %385, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %387 = stablehlo.add %384, %386 : tensor<256x14x14x256xf32>
    %388 = stablehlo.convert %387 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %389 = call @relu_61(%388) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %390 = stablehlo.convert %arg101 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %391 = stablehlo.convolution(%389, %390) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %392 = stablehlo.broadcast_in_dim %arg286, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %393 = stablehlo.broadcast_in_dim %arg287, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %394 = stablehlo.convert %391 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %395 = stablehlo.broadcast_in_dim %392, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %396 = stablehlo.subtract %394, %395 : tensor<256x14x14x256xf32>
    %397 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %398 = stablehlo.add %393, %397 : tensor<1x1x1x256xf32>
    %399 = stablehlo.rsqrt %398 : tensor<1x1x1x256xf32>
    %400 = stablehlo.reshape %arg99 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %401 = stablehlo.multiply %399, %400 : tensor<1x1x1x256xf32>
    %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %403 = stablehlo.multiply %396, %402 : tensor<256x14x14x256xf32>
    %404 = stablehlo.reshape %arg98 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %405 = stablehlo.broadcast_in_dim %404, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %406 = stablehlo.add %403, %405 : tensor<256x14x14x256xf32>
    %407 = stablehlo.convert %406 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %408 = stablehlo.add %370, %407 : tensor<256x14x14x256xbf16>
    %409 = call @relu_61(%408) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %410 = stablehlo.convert %arg106 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %411 = stablehlo.convolution(%409, %410) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %412 = stablehlo.broadcast_in_dim %arg288, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %413 = stablehlo.broadcast_in_dim %arg289, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %414 = stablehlo.convert %411 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %415 = stablehlo.broadcast_in_dim %412, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %416 = stablehlo.subtract %414, %415 : tensor<256x14x14x256xf32>
    %417 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %418 = stablehlo.add %413, %417 : tensor<1x1x1x256xf32>
    %419 = stablehlo.rsqrt %418 : tensor<1x1x1x256xf32>
    %420 = stablehlo.reshape %arg103 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %421 = stablehlo.multiply %419, %420 : tensor<1x1x1x256xf32>
    %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %423 = stablehlo.multiply %416, %422 : tensor<256x14x14x256xf32>
    %424 = stablehlo.reshape %arg102 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %425 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %426 = stablehlo.add %423, %425 : tensor<256x14x14x256xf32>
    %427 = stablehlo.convert %426 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %428 = call @relu_61(%427) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %429 = stablehlo.convert %arg107 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %430 = stablehlo.convolution(%428, %429) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %431 = stablehlo.broadcast_in_dim %arg290, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %432 = stablehlo.broadcast_in_dim %arg291, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %433 = stablehlo.convert %430 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %434 = stablehlo.broadcast_in_dim %431, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %435 = stablehlo.subtract %433, %434 : tensor<256x14x14x256xf32>
    %436 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %437 = stablehlo.add %432, %436 : tensor<1x1x1x256xf32>
    %438 = stablehlo.rsqrt %437 : tensor<1x1x1x256xf32>
    %439 = stablehlo.reshape %arg105 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %440 = stablehlo.multiply %438, %439 : tensor<1x1x1x256xf32>
    %441 = stablehlo.broadcast_in_dim %440, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %442 = stablehlo.multiply %435, %441 : tensor<256x14x14x256xf32>
    %443 = stablehlo.reshape %arg104 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %444 = stablehlo.broadcast_in_dim %443, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %445 = stablehlo.add %442, %444 : tensor<256x14x14x256xf32>
    %446 = stablehlo.convert %445 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %447 = stablehlo.add %409, %446 : tensor<256x14x14x256xbf16>
    %448 = call @relu_61(%447) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %449 = stablehlo.convert %arg19 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %450 = stablehlo.convolution(%448, %449) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %451 = stablehlo.broadcast_in_dim %arg230, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %452 = stablehlo.broadcast_in_dim %arg231, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %453 = stablehlo.convert %450 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %454 = stablehlo.broadcast_in_dim %451, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %455 = stablehlo.subtract %453, %454 : tensor<256x14x14x256xf32>
    %456 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %457 = stablehlo.add %452, %456 : tensor<1x1x1x256xf32>
    %458 = stablehlo.rsqrt %457 : tensor<1x1x1x256xf32>
    %459 = stablehlo.reshape %arg16 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %460 = stablehlo.multiply %458, %459 : tensor<1x1x1x256xf32>
    %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %462 = stablehlo.multiply %455, %461 : tensor<256x14x14x256xf32>
    %463 = stablehlo.reshape %arg15 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %465 = stablehlo.add %462, %464 : tensor<256x14x14x256xf32>
    %466 = stablehlo.convert %465 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %467 = call @relu_61(%466) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %468 = stablehlo.convert %arg20 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %469 = stablehlo.convolution(%467, %468) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %470 = stablehlo.broadcast_in_dim %arg232, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %471 = stablehlo.broadcast_in_dim %arg233, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %472 = stablehlo.convert %469 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %473 = stablehlo.broadcast_in_dim %470, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %474 = stablehlo.subtract %472, %473 : tensor<256x14x14x256xf32>
    %475 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %476 = stablehlo.add %471, %475 : tensor<1x1x1x256xf32>
    %477 = stablehlo.rsqrt %476 : tensor<1x1x1x256xf32>
    %478 = stablehlo.reshape %arg18 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %479 = stablehlo.multiply %477, %478 : tensor<1x1x1x256xf32>
    %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %481 = stablehlo.multiply %474, %480 : tensor<256x14x14x256xf32>
    %482 = stablehlo.reshape %arg17 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %484 = stablehlo.add %481, %483 : tensor<256x14x14x256xf32>
    %485 = stablehlo.convert %484 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %486 = stablehlo.add %448, %485 : tensor<256x14x14x256xbf16>
    %487 = call @relu_61(%486) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %488 = stablehlo.convert %arg25 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %489 = stablehlo.convolution(%487, %488) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %490 = stablehlo.broadcast_in_dim %arg234, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %491 = stablehlo.broadcast_in_dim %arg235, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %492 = stablehlo.convert %489 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %493 = stablehlo.broadcast_in_dim %490, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %494 = stablehlo.subtract %492, %493 : tensor<256x14x14x256xf32>
    %495 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %496 = stablehlo.add %491, %495 : tensor<1x1x1x256xf32>
    %497 = stablehlo.rsqrt %496 : tensor<1x1x1x256xf32>
    %498 = stablehlo.reshape %arg22 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %499 = stablehlo.multiply %497, %498 : tensor<1x1x1x256xf32>
    %500 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %501 = stablehlo.multiply %494, %500 : tensor<256x14x14x256xf32>
    %502 = stablehlo.reshape %arg21 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %503 = stablehlo.broadcast_in_dim %502, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %504 = stablehlo.add %501, %503 : tensor<256x14x14x256xf32>
    %505 = stablehlo.convert %504 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %506 = call @relu_61(%505) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %507 = stablehlo.convert %arg26 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %508 = stablehlo.convolution(%506, %507) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %509 = stablehlo.broadcast_in_dim %arg236, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %510 = stablehlo.broadcast_in_dim %arg237, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %511 = stablehlo.convert %508 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %512 = stablehlo.broadcast_in_dim %509, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %513 = stablehlo.subtract %511, %512 : tensor<256x14x14x256xf32>
    %514 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %515 = stablehlo.add %510, %514 : tensor<1x1x1x256xf32>
    %516 = stablehlo.rsqrt %515 : tensor<1x1x1x256xf32>
    %517 = stablehlo.reshape %arg24 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %518 = stablehlo.multiply %516, %517 : tensor<1x1x1x256xf32>
    %519 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %520 = stablehlo.multiply %513, %519 : tensor<256x14x14x256xf32>
    %521 = stablehlo.reshape %arg23 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %522 = stablehlo.broadcast_in_dim %521, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %523 = stablehlo.add %520, %522 : tensor<256x14x14x256xf32>
    %524 = stablehlo.convert %523 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %525 = stablehlo.add %487, %524 : tensor<256x14x14x256xbf16>
    %526 = call @relu_61(%525) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %527 = stablehlo.convert %arg31 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %528 = stablehlo.convolution(%526, %527) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %529 = stablehlo.broadcast_in_dim %arg238, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %530 = stablehlo.broadcast_in_dim %arg239, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %531 = stablehlo.convert %528 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %532 = stablehlo.broadcast_in_dim %529, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %533 = stablehlo.subtract %531, %532 : tensor<256x14x14x256xf32>
    %534 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %535 = stablehlo.add %530, %534 : tensor<1x1x1x256xf32>
    %536 = stablehlo.rsqrt %535 : tensor<1x1x1x256xf32>
    %537 = stablehlo.reshape %arg28 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %538 = stablehlo.multiply %536, %537 : tensor<1x1x1x256xf32>
    %539 = stablehlo.broadcast_in_dim %538, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %540 = stablehlo.multiply %533, %539 : tensor<256x14x14x256xf32>
    %541 = stablehlo.reshape %arg27 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %542 = stablehlo.broadcast_in_dim %541, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %543 = stablehlo.add %540, %542 : tensor<256x14x14x256xf32>
    %544 = stablehlo.convert %543 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %545 = call @relu_61(%544) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %546 = stablehlo.convert %arg32 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %547 = stablehlo.convolution(%545, %546) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %548 = stablehlo.broadcast_in_dim %arg240, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %549 = stablehlo.broadcast_in_dim %arg241, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %550 = stablehlo.convert %547 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %551 = stablehlo.broadcast_in_dim %548, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %552 = stablehlo.subtract %550, %551 : tensor<256x14x14x256xf32>
    %553 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %554 = stablehlo.add %549, %553 : tensor<1x1x1x256xf32>
    %555 = stablehlo.rsqrt %554 : tensor<1x1x1x256xf32>
    %556 = stablehlo.reshape %arg30 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %557 = stablehlo.multiply %555, %556 : tensor<1x1x1x256xf32>
    %558 = stablehlo.broadcast_in_dim %557, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %559 = stablehlo.multiply %552, %558 : tensor<256x14x14x256xf32>
    %560 = stablehlo.reshape %arg29 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %561 = stablehlo.broadcast_in_dim %560, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %562 = stablehlo.add %559, %561 : tensor<256x14x14x256xf32>
    %563 = stablehlo.convert %562 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %564 = stablehlo.add %526, %563 : tensor<256x14x14x256xbf16>
    %565 = call @relu_61(%564) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %566 = stablehlo.convert %arg37 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xbf16>
    %567 = stablehlo.convolution(%565, %566) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x512xbf16>) -> tensor<256x7x7x512xbf16>
    %568 = stablehlo.broadcast_in_dim %arg242, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %569 = stablehlo.broadcast_in_dim %arg243, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %570 = stablehlo.convert %567 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %571 = stablehlo.broadcast_in_dim %568, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %572 = stablehlo.subtract %570, %571 : tensor<256x7x7x512xf32>
    %573 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %574 = stablehlo.add %569, %573 : tensor<1x1x1x512xf32>
    %575 = stablehlo.rsqrt %574 : tensor<1x1x1x512xf32>
    %576 = stablehlo.reshape %arg34 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %577 = stablehlo.multiply %575, %576 : tensor<1x1x1x512xf32>
    %578 = stablehlo.broadcast_in_dim %577, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %579 = stablehlo.multiply %572, %578 : tensor<256x7x7x512xf32>
    %580 = stablehlo.reshape %arg33 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %581 = stablehlo.broadcast_in_dim %580, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %582 = stablehlo.add %579, %581 : tensor<256x7x7x512xf32>
    %583 = stablehlo.convert %582 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %584 = call @relu_92(%583) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %585 = stablehlo.convert %arg38 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %586 = stablehlo.convolution(%584, %585) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %587 = stablehlo.broadcast_in_dim %arg244, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %588 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %589 = stablehlo.convert %586 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %590 = stablehlo.broadcast_in_dim %587, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %591 = stablehlo.subtract %589, %590 : tensor<256x7x7x512xf32>
    %592 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %593 = stablehlo.add %588, %592 : tensor<1x1x1x512xf32>
    %594 = stablehlo.rsqrt %593 : tensor<1x1x1x512xf32>
    %595 = stablehlo.reshape %arg36 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %596 = stablehlo.multiply %594, %595 : tensor<1x1x1x512xf32>
    %597 = stablehlo.broadcast_in_dim %596, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %598 = stablehlo.multiply %591, %597 : tensor<256x7x7x512xf32>
    %599 = stablehlo.reshape %arg35 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %600 = stablehlo.broadcast_in_dim %599, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %601 = stablehlo.add %598, %600 : tensor<256x7x7x512xf32>
    %602 = stablehlo.convert %601 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %603 = stablehlo.convert %arg39 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xbf16>
    %604 = stablehlo.convolution(%565, %603) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x512xbf16>) -> tensor<256x7x7x512xbf16>
    %605 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %606 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %607 = stablehlo.convert %604 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %608 = stablehlo.broadcast_in_dim %605, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %609 = stablehlo.subtract %607, %608 : tensor<256x7x7x512xf32>
    %610 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %611 = stablehlo.add %606, %610 : tensor<1x1x1x512xf32>
    %612 = stablehlo.rsqrt %611 : tensor<1x1x1x512xf32>
    %613 = stablehlo.reshape %arg41 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %614 = stablehlo.multiply %612, %613 : tensor<1x1x1x512xf32>
    %615 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %616 = stablehlo.multiply %609, %615 : tensor<256x7x7x512xf32>
    %617 = stablehlo.reshape %arg40 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %618 = stablehlo.broadcast_in_dim %617, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %619 = stablehlo.add %616, %618 : tensor<256x7x7x512xf32>
    %620 = stablehlo.convert %619 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %621 = stablehlo.add %620, %602 : tensor<256x7x7x512xbf16>
    %622 = call @relu_92(%621) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %623 = stablehlo.convert %arg46 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %624 = stablehlo.convolution(%622, %623) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %625 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %626 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %627 = stablehlo.convert %624 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %628 = stablehlo.broadcast_in_dim %625, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %629 = stablehlo.subtract %627, %628 : tensor<256x7x7x512xf32>
    %630 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %631 = stablehlo.add %626, %630 : tensor<1x1x1x512xf32>
    %632 = stablehlo.rsqrt %631 : tensor<1x1x1x512xf32>
    %633 = stablehlo.reshape %arg43 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %634 = stablehlo.multiply %632, %633 : tensor<1x1x1x512xf32>
    %635 = stablehlo.broadcast_in_dim %634, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %636 = stablehlo.multiply %629, %635 : tensor<256x7x7x512xf32>
    %637 = stablehlo.reshape %arg42 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %638 = stablehlo.broadcast_in_dim %637, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %639 = stablehlo.add %636, %638 : tensor<256x7x7x512xf32>
    %640 = stablehlo.convert %639 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %641 = call @relu_92(%640) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %642 = stablehlo.convert %arg47 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %643 = stablehlo.convolution(%641, %642) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %644 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %645 = stablehlo.broadcast_in_dim %arg251, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %646 = stablehlo.convert %643 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %647 = stablehlo.broadcast_in_dim %644, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %648 = stablehlo.subtract %646, %647 : tensor<256x7x7x512xf32>
    %649 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %650 = stablehlo.add %645, %649 : tensor<1x1x1x512xf32>
    %651 = stablehlo.rsqrt %650 : tensor<1x1x1x512xf32>
    %652 = stablehlo.reshape %arg45 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %653 = stablehlo.multiply %651, %652 : tensor<1x1x1x512xf32>
    %654 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %655 = stablehlo.multiply %648, %654 : tensor<256x7x7x512xf32>
    %656 = stablehlo.reshape %arg44 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %657 = stablehlo.broadcast_in_dim %656, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %658 = stablehlo.add %655, %657 : tensor<256x7x7x512xf32>
    %659 = stablehlo.convert %658 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %660 = stablehlo.add %622, %659 : tensor<256x7x7x512xbf16>
    %661 = call @relu_92(%660) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %662 = stablehlo.convert %arg52 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %663 = stablehlo.convolution(%661, %662) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %664 = stablehlo.broadcast_in_dim %arg252, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %665 = stablehlo.broadcast_in_dim %arg253, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %666 = stablehlo.convert %663 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %667 = stablehlo.broadcast_in_dim %664, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %668 = stablehlo.subtract %666, %667 : tensor<256x7x7x512xf32>
    %669 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %670 = stablehlo.add %665, %669 : tensor<1x1x1x512xf32>
    %671 = stablehlo.rsqrt %670 : tensor<1x1x1x512xf32>
    %672 = stablehlo.reshape %arg49 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %673 = stablehlo.multiply %671, %672 : tensor<1x1x1x512xf32>
    %674 = stablehlo.broadcast_in_dim %673, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %675 = stablehlo.multiply %668, %674 : tensor<256x7x7x512xf32>
    %676 = stablehlo.reshape %arg48 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %677 = stablehlo.broadcast_in_dim %676, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %678 = stablehlo.add %675, %677 : tensor<256x7x7x512xf32>
    %679 = stablehlo.convert %678 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %680 = call @relu_92(%679) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %681 = stablehlo.convert %arg53 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %682 = stablehlo.convolution(%680, %681) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %683 = stablehlo.broadcast_in_dim %arg254, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %684 = stablehlo.broadcast_in_dim %arg255, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %685 = stablehlo.convert %682 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %686 = stablehlo.broadcast_in_dim %683, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %687 = stablehlo.subtract %685, %686 : tensor<256x7x7x512xf32>
    %688 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %689 = stablehlo.add %684, %688 : tensor<1x1x1x512xf32>
    %690 = stablehlo.rsqrt %689 : tensor<1x1x1x512xf32>
    %691 = stablehlo.reshape %arg51 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %692 = stablehlo.multiply %690, %691 : tensor<1x1x1x512xf32>
    %693 = stablehlo.broadcast_in_dim %692, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %694 = stablehlo.multiply %687, %693 : tensor<256x7x7x512xf32>
    %695 = stablehlo.reshape %arg50 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %696 = stablehlo.broadcast_in_dim %695, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %697 = stablehlo.add %694, %696 : tensor<256x7x7x512xf32>
    %698 = stablehlo.convert %697 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %699 = stablehlo.add %661, %698 : tensor<256x7x7x512xbf16>
    %700 = call @relu_92(%699) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %701 = stablehlo.convert %700 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %702 = stablehlo.reduce(%701 init: %cst_1) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<256x512xf32>
    %cst_2 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
    %703 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
    %704 = stablehlo.divide %702, %703 : tensor<256x512xf32>
    %705 = stablehlo.convert %704 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %706 = stablehlo.convert %arg2 : (tensor<512x1000xf32>) -> tensor<512x1000xbf16>
    %707 = stablehlo.convert %arg1 : (tensor<1000xf32>) -> tensor<1000xbf16>
    %708 = stablehlo.dot_general %705, %706, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xbf16>, tensor<512x1000xbf16>) -> tensor<256x1000xbf16>
    %709 = stablehlo.reshape %707 : (tensor<1000xbf16>) -> tensor<1x1000xbf16>
    %710 = stablehlo.broadcast_in_dim %709, dims = [0, 1] : (tensor<1x1000xbf16>) -> tensor<256x1000xbf16>
    %711 = stablehlo.add %708, %710 : tensor<256x1000xbf16>
    %712 = stablehlo.broadcast_in_dim %711, dims = [1, 2] : (tensor<256x1000xbf16>) -> tensor<1x256x1000xbf16>
    return %712 : tensor<1x256x1000xbf16>
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
  func.func private @relu_34(%arg0: tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xbf16>
    return %1 : tensor<256x28x28x128xbf16>
  }
  func.func private @relu_61(%arg0: tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xbf16>
    return %1 : tensor<256x14x14x256xbf16>
  }
  func.func private @relu_92(%arg0: tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xbf16>
    return %1 : tensor<256x7x7x512xbf16>
  }
}
