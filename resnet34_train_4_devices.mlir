module @pmap_train_step attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg1: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg2: tensor<512x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg3: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg4: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg5: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg6: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg7: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg8: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg9: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg10: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg11: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg12: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg13: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg14: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg15: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg16: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg17: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg18: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg19: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg20: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg21: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg22: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg23: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg24: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg25: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg26: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg27: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg28: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg29: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg30: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg31: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg32: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg33: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg34: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg35: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg36: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg37: tensor<3x3x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg38: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg39: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg40: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg41: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg42: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg43: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg44: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg45: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg46: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg47: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg48: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg49: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg50: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg51: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg52: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg53: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg54: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg55: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg56: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg57: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg58: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg59: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg60: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg61: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg62: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg63: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg64: tensor<3x3x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg65: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg66: tensor<1x1x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg67: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg68: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg69: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg70: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg71: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg72: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg73: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg74: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg75: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg76: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg77: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg78: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg79: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg80: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg81: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg82: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg83: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg84: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg85: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg86: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg87: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg88: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg89: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg90: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg91: tensor<3x3x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg92: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg93: tensor<1x1x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg94: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg95: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg96: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg97: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg98: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg99: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg100: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg101: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg102: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg103: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg104: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg105: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg106: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg107: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg108: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg109: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg110: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg111: tensor<1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg112: tensor<512x1000xf32> {mhlo.is_same_data_across_replicas = true}, %arg113: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg114: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg115: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg116: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg117: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg118: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg119: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg120: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg121: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg122: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg123: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg124: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg125: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg126: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg127: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg128: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg129: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg130: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg131: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg132: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg133: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg134: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg135: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg136: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg137: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg138: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg139: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg140: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg141: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg142: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg143: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg144: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg145: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg146: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg147: tensor<3x3x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg148: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg149: tensor<1x1x256x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg150: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg151: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg152: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg153: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg154: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg155: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg156: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg157: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg158: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg159: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg160: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg161: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg162: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg163: tensor<3x3x512x512xf32> {mhlo.is_same_data_across_replicas = true}, %arg164: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg165: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg166: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg167: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg168: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg169: tensor<3x3x64x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg170: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg171: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg172: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg173: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg174: tensor<3x3x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg175: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg176: tensor<1x1x64x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg177: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg178: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg179: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg180: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg181: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg182: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg183: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg184: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg185: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg186: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg187: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg188: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg189: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg190: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg191: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg192: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg193: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg194: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg195: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg196: tensor<3x3x128x128xf32> {mhlo.is_same_data_across_replicas = true}, %arg197: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg198: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg199: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg200: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg201: tensor<3x3x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg202: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg203: tensor<1x1x128x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg204: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg205: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg206: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg207: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg208: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg209: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg210: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg211: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg212: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg213: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg214: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg215: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg216: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg217: tensor<3x3x256x256xf32> {mhlo.is_same_data_across_replicas = true}, %arg218: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg219: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg220: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas = true}, %arg221: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg222: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg223: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg224: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg225: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg226: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg227: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg228: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg229: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg230: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg231: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg232: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg233: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg234: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg235: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg236: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg237: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg238: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg239: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg240: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg241: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg242: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg243: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg244: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg245: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg246: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg247: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg248: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg249: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg250: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg251: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg252: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg253: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg254: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg255: tensor<512xf32> {mhlo.is_same_data_across_replicas = true}, %arg256: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg257: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg258: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg259: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg260: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg261: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg262: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg263: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg264: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg265: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg266: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg267: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg268: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg269: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg270: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg271: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg272: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg273: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg274: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg275: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg276: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg277: tensor<128xf32> {mhlo.is_same_data_across_replicas = true}, %arg278: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg279: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg280: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg281: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg282: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg283: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg284: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg285: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg286: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg287: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg288: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg289: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg290: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg291: tensor<256xf32> {mhlo.is_same_data_across_replicas = true}, %arg292: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg293: tensor<64xf32> {mhlo.is_same_data_across_replicas = true}, %arg294: tensor<i32> {mhlo.is_same_data_across_replicas = true}, %arg295: tensor<f32> {mhlo.is_same_data_across_replicas = true}, %arg296: tensor<1x256x224x224x3xbf16>, %arg297: tensor<1x256xi32>) -> (tensor<i32> {jax.result_info = "result[0].step"}, tensor<1000xf32> {jax.result_info = "result[0].params['Dense_0']['bias']"}, tensor<512x1000xf32> {jax.result_info = "result[0].params['Dense_0']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_0']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_1']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_10']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_10']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_10']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_10']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_10']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_10']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_11']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_11']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_11']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_11']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_11']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_11']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_12']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_12']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_12']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_12']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_12']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_12']['Conv_1']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_13']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_13']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_13']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_13']['BatchNorm_1']['scale']"}, tensor<3x3x256x512xf32> {jax.result_info = "result[0].params['ResNetBlock_13']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].params['ResNetBlock_13']['Conv_1']['kernel']"}, tensor<1x1x256x512xf32> {jax.result_info = "result[0].params['ResNetBlock_13']['conv_proj']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_13']['norm_proj']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_13']['norm_proj']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_14']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_14']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_14']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_14']['BatchNorm_1']['scale']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].params['ResNetBlock_14']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].params['ResNetBlock_14']['Conv_1']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_15']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_15']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_15']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].params['ResNetBlock_15']['BatchNorm_1']['scale']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].params['ResNetBlock_15']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].params['ResNetBlock_15']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].params['ResNetBlock_2']['Conv_1']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['BatchNorm_1']['scale']"}, tensor<3x3x64x128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['Conv_1']['kernel']"}, tensor<1x1x64x128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['conv_proj']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['norm_proj']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_3']['norm_proj']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['BatchNorm_1']['scale']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_4']['Conv_1']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['BatchNorm_1']['scale']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_5']['Conv_1']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['BatchNorm_1']['scale']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].params['ResNetBlock_6']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['BatchNorm_1']['scale']"}, tensor<3x3x128x256xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['Conv_1']['kernel']"}, tensor<1x1x128x256xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['conv_proj']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['norm_proj']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_7']['norm_proj']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_8']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_8']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_8']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_8']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_8']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_8']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_9']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_9']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_9']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].params['ResNetBlock_9']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_9']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].params['ResNetBlock_9']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].params['bn_init']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].params['bn_init']['scale']"}, tensor<7x7x3x64xf32> {jax.result_info = "result[0].params['conv_init']['kernel']"}, tensor<1000xf32> {jax.result_info = "result[0].opt_state[0].trace['Dense_0']['bias']"}, tensor<512x1000xf32> {jax.result_info = "result[0].opt_state[0].trace['Dense_0']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_0']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_1']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_10']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_10']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_10']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_10']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_10']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_10']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_11']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_11']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_11']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_11']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_11']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_11']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_12']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_12']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_12']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_12']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_12']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_12']['Conv_1']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_13']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_13']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_13']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_13']['BatchNorm_1']['scale']"}, tensor<3x3x256x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_13']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_13']['Conv_1']['kernel']"}, tensor<1x1x256x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_13']['conv_proj']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_13']['norm_proj']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_13']['norm_proj']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_14']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_14']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_14']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_14']['BatchNorm_1']['scale']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_14']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_14']['Conv_1']['kernel']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_15']['BatchNorm_0']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_15']['BatchNorm_0']['scale']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_15']['BatchNorm_1']['bias']"}, tensor<512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_15']['BatchNorm_1']['scale']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_15']['Conv_0']['kernel']"}, tensor<3x3x512x512xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_15']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['BatchNorm_0']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['BatchNorm_0']['scale']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['BatchNorm_1']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['BatchNorm_1']['scale']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['Conv_0']['kernel']"}, tensor<3x3x64x64xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_2']['Conv_1']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['BatchNorm_1']['scale']"}, tensor<3x3x64x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['Conv_1']['kernel']"}, tensor<1x1x64x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['conv_proj']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['norm_proj']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_3']['norm_proj']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['BatchNorm_1']['scale']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_4']['Conv_1']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['BatchNorm_1']['scale']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_5']['Conv_1']['kernel']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['BatchNorm_0']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['BatchNorm_0']['scale']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['BatchNorm_1']['bias']"}, tensor<128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['BatchNorm_1']['scale']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['Conv_0']['kernel']"}, tensor<3x3x128x128xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_6']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['BatchNorm_1']['scale']"}, tensor<3x3x128x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['Conv_1']['kernel']"}, tensor<1x1x128x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['conv_proj']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['norm_proj']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_7']['norm_proj']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_8']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_8']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_8']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_8']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_8']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_8']['Conv_1']['kernel']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_9']['BatchNorm_0']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_9']['BatchNorm_0']['scale']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_9']['BatchNorm_1']['bias']"}, tensor<256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_9']['BatchNorm_1']['scale']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_9']['Conv_0']['kernel']"}, tensor<3x3x256x256xf32> {jax.result_info = "result[0].opt_state[0].trace['ResNetBlock_9']['Conv_1']['kernel']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['bn_init']['bias']"}, tensor<64xf32> {jax.result_info = "result[0].opt_state[0].trace['bn_init']['scale']"}, tensor<7x7x3x64xf32> {jax.result_info = "result[0].opt_state[0].trace['conv_init']['kernel']"}, tensor<i32> {jax.result_info = "result[0].opt_state[1].count"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_0']['BatchNorm_0']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_0']['BatchNorm_0']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_0']['BatchNorm_1']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_0']['BatchNorm_1']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_1']['BatchNorm_0']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_1']['BatchNorm_0']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_1']['BatchNorm_1']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_1']['BatchNorm_1']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_10']['BatchNorm_0']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_10']['BatchNorm_0']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_10']['BatchNorm_1']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_10']['BatchNorm_1']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_11']['BatchNorm_0']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_11']['BatchNorm_0']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_11']['BatchNorm_1']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_11']['BatchNorm_1']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_12']['BatchNorm_0']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_12']['BatchNorm_0']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_12']['BatchNorm_1']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_12']['BatchNorm_1']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_13']['BatchNorm_0']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_13']['BatchNorm_0']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_13']['BatchNorm_1']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_13']['BatchNorm_1']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_13']['norm_proj']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_13']['norm_proj']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_14']['BatchNorm_0']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_14']['BatchNorm_0']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_14']['BatchNorm_1']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_14']['BatchNorm_1']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_15']['BatchNorm_0']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_15']['BatchNorm_0']['var']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_15']['BatchNorm_1']['mean']"}, tensor<512xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_15']['BatchNorm_1']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['BatchNorm_0']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['BatchNorm_0']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['BatchNorm_1']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_2']['BatchNorm_1']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['BatchNorm_0']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['BatchNorm_0']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['BatchNorm_1']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['BatchNorm_1']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['norm_proj']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_3']['norm_proj']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['BatchNorm_0']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['BatchNorm_0']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['BatchNorm_1']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_4']['BatchNorm_1']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_5']['BatchNorm_0']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_5']['BatchNorm_0']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_5']['BatchNorm_1']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_5']['BatchNorm_1']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['BatchNorm_0']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['BatchNorm_0']['var']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['BatchNorm_1']['mean']"}, tensor<128xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_6']['BatchNorm_1']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['BatchNorm_0']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['BatchNorm_0']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['BatchNorm_1']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['BatchNorm_1']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['norm_proj']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_7']['norm_proj']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_8']['BatchNorm_0']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_8']['BatchNorm_0']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_8']['BatchNorm_1']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_8']['BatchNorm_1']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_9']['BatchNorm_0']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_9']['BatchNorm_0']['var']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_9']['BatchNorm_1']['mean']"}, tensor<256xf32> {jax.result_info = "result[0].batch_stats['ResNetBlock_9']['BatchNorm_1']['var']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['bn_init']['mean']"}, tensor<64xf32> {jax.result_info = "result[0].batch_stats['bn_init']['var']"}, tensor<i32> {jax.result_info = "result[0].dynamic_scale.fin_steps"}, tensor<f32> {jax.result_info = "result[0].dynamic_scale.scale"}, tensor<1xf32> {jax.result_info = "result[1]['accuracy']"}, tensor<1xf32> {jax.result_info = "result[1]['learning_rate']"}, tensor<1xf32> {jax.result_info = "result[1]['loss']"}, tensor<1xf32> {jax.result_info = "result[1]['scale']"}) {
    %0 = stablehlo.reshape %arg296 : (tensor<1x256x224x224x3xbf16>) -> tensor<256x224x224x3xbf16>
    %1 = stablehlo.reshape %arg297 : (tensor<1x256xi32>) -> tensor<256xi32>
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
    %23 = stablehlo.convert %arg110 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xbf16>
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
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
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
    %62 = stablehlo.multiply %61, %arg292 : tensor<64xf32>
    %cst_15 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %63 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %64 = stablehlo.multiply %63, %42 : tensor<64xf32>
    %65 = stablehlo.add %62, %64 : tensor<64xf32>
    %66 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %67 = stablehlo.multiply %66, %arg293 : tensor<64xf32>
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
    %82 = stablehlo.reshape %arg109 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %83 = stablehlo.multiply %78, %82 : tensor<1x1x1x64xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %85 = stablehlo.multiply %75, %84 : tensor<256x112x112x64xf32>
    %86 = stablehlo.reshape %arg108 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
    ^bb0(%arg298: tensor<bf16>, %arg299: tensor<bf16>):
      %7388 = stablehlo.maximum %arg298, %arg299 : tensor<bf16>
      stablehlo.return %7388 : tensor<bf16>
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
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
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
    %135 = stablehlo.multiply %134, %arg222 : tensor<64xf32>
    %136 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %137 = stablehlo.multiply %136, %115 : tensor<64xf32>
    %138 = stablehlo.add %135, %137 : tensor<64xf32>
    %139 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %140 = stablehlo.multiply %139, %arg223 : tensor<64xf32>
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
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
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
    %206 = stablehlo.multiply %205, %arg224 : tensor<64xf32>
    %207 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %208 = stablehlo.multiply %207, %186 : tensor<64xf32>
    %209 = stablehlo.add %206, %208 : tensor<64xf32>
    %210 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %211 = stablehlo.multiply %210, %arg225 : tensor<64xf32>
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
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
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
    %278 = stablehlo.multiply %277, %arg226 : tensor<64xf32>
    %279 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %280 = stablehlo.multiply %279, %258 : tensor<64xf32>
    %281 = stablehlo.add %278, %280 : tensor<64xf32>
    %282 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %283 = stablehlo.multiply %282, %arg227 : tensor<64xf32>
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
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
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
    %349 = stablehlo.multiply %348, %arg228 : tensor<64xf32>
    %350 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %351 = stablehlo.multiply %350, %329 : tensor<64xf32>
    %352 = stablehlo.add %349, %351 : tensor<64xf32>
    %353 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %354 = stablehlo.multiply %353, %arg229 : tensor<64xf32>
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
    %382 = stablehlo.convert %arg58 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %383 = stablehlo.convolution(%378, %382) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %384 = stablehlo.convert %383 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %385 = stablehlo.multiply %384, %384 : tensor<256x56x56x64xf32>
    %386 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
    %387 = stablehlo.multiply %386, %384 : tensor<256x56x56x64xf32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %388 = stablehlo.reduce(%384 init: %cst_29) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %389 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %390 = stablehlo.divide %388, %389 : tensor<64xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %391 = stablehlo.reduce(%385 init: %cst_30) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %392 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %393 = stablehlo.divide %391, %392 : tensor<64xf32>
    %394 = stablehlo.broadcast_in_dim %390, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %395 = stablehlo.broadcast_in_dim %393, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %396 = stablehlo.concatenate %394, %395, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
    %397 = "stablehlo.all_reduce"(%396) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %398 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %399 = stablehlo.divide %397, %398 : tensor<2x64xf32>
    %400 = stablehlo.slice %399 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %401 = stablehlo.reshape %400 : (tensor<1x64xf32>) -> tensor<64xf32>
    %402 = stablehlo.slice %399 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %403 = stablehlo.reshape %402 : (tensor<1x64xf32>) -> tensor<64xf32>
    %404 = stablehlo.multiply %401, %401 : tensor<64xf32>
    %405 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %406 = stablehlo.multiply %405, %401 : tensor<64xf32>
    %407 = stablehlo.subtract %403, %404 : tensor<64xf32>
    %408 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %409 = stablehlo.maximum %408, %407 : tensor<64xf32>
    %410 = stablehlo.compare  EQ, %407, %409,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %411 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %412 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %413 = stablehlo.select %410, %411, %412 : tensor<64xi1>, tensor<64xf32>
    %414 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %415 = stablehlo.compare  EQ, %414, %409,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %416 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %417 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %418 = stablehlo.select %415, %416, %417 : tensor<64xi1>, tensor<64xf32>
    %419 = stablehlo.divide %413, %418 : tensor<64xf32>
    %420 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %421 = stablehlo.multiply %420, %arg256 : tensor<64xf32>
    %422 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %423 = stablehlo.multiply %422, %401 : tensor<64xf32>
    %424 = stablehlo.add %421, %423 : tensor<64xf32>
    %425 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %426 = stablehlo.multiply %425, %arg257 : tensor<64xf32>
    %427 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %428 = stablehlo.multiply %427, %409 : tensor<64xf32>
    %429 = stablehlo.add %426, %428 : tensor<64xf32>
    %430 = stablehlo.broadcast_in_dim %401, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %431 = stablehlo.broadcast_in_dim %409, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %432 = stablehlo.convert %383 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %433 = stablehlo.broadcast_in_dim %430, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %434 = stablehlo.subtract %432, %433 : tensor<256x56x56x64xf32>
    %435 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %436 = stablehlo.add %431, %435 : tensor<1x1x1x64xf32>
    %437 = stablehlo.rsqrt %436 : tensor<1x1x1x64xf32>
    %438 = stablehlo.divide %437, %436 : tensor<1x1x1x64xf32>
    %439 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %440 = stablehlo.multiply %439, %438 : tensor<1x1x1x64xf32>
    %441 = stablehlo.reshape %arg55 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %442 = stablehlo.multiply %437, %441 : tensor<1x1x1x64xf32>
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %444 = stablehlo.multiply %434, %443 : tensor<256x56x56x64xf32>
    %445 = stablehlo.reshape %arg54 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %446 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %447 = stablehlo.add %444, %446 : tensor<256x56x56x64xf32>
    %448 = stablehlo.convert %447 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %449 = call @relu_41(%448) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %450 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %451 = stablehlo.compare  GT, %448, %450,  FLOAT : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xi1>
    %452 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %453 = stablehlo.convert %arg59 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xbf16>
    %454 = stablehlo.convolution(%449, %453) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %455 = stablehlo.convert %454 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %456 = stablehlo.multiply %455, %455 : tensor<256x56x56x64xf32>
    %457 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
    %458 = stablehlo.multiply %457, %455 : tensor<256x56x56x64xf32>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %459 = stablehlo.reduce(%455 init: %cst_31) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %460 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %461 = stablehlo.divide %459, %460 : tensor<64xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %462 = stablehlo.reduce(%456 init: %cst_32) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %463 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %464 = stablehlo.divide %462, %463 : tensor<64xf32>
    %465 = stablehlo.broadcast_in_dim %461, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %466 = stablehlo.broadcast_in_dim %464, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %467 = stablehlo.concatenate %465, %466, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
    %468 = "stablehlo.all_reduce"(%467) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %469 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %470 = stablehlo.divide %468, %469 : tensor<2x64xf32>
    %471 = stablehlo.slice %470 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %472 = stablehlo.reshape %471 : (tensor<1x64xf32>) -> tensor<64xf32>
    %473 = stablehlo.slice %470 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %474 = stablehlo.reshape %473 : (tensor<1x64xf32>) -> tensor<64xf32>
    %475 = stablehlo.multiply %472, %472 : tensor<64xf32>
    %476 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %477 = stablehlo.multiply %476, %472 : tensor<64xf32>
    %478 = stablehlo.subtract %474, %475 : tensor<64xf32>
    %479 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %480 = stablehlo.maximum %479, %478 : tensor<64xf32>
    %481 = stablehlo.compare  EQ, %478, %480,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %482 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %483 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %484 = stablehlo.select %481, %482, %483 : tensor<64xi1>, tensor<64xf32>
    %485 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %486 = stablehlo.compare  EQ, %485, %480,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    %487 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %488 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %489 = stablehlo.select %486, %487, %488 : tensor<64xi1>, tensor<64xf32>
    %490 = stablehlo.divide %484, %489 : tensor<64xf32>
    %491 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %492 = stablehlo.multiply %491, %arg258 : tensor<64xf32>
    %493 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %494 = stablehlo.multiply %493, %472 : tensor<64xf32>
    %495 = stablehlo.add %492, %494 : tensor<64xf32>
    %496 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %497 = stablehlo.multiply %496, %arg259 : tensor<64xf32>
    %498 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %499 = stablehlo.multiply %498, %480 : tensor<64xf32>
    %500 = stablehlo.add %497, %499 : tensor<64xf32>
    %501 = stablehlo.broadcast_in_dim %472, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %502 = stablehlo.broadcast_in_dim %480, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %503 = stablehlo.convert %454 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %504 = stablehlo.broadcast_in_dim %501, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %505 = stablehlo.subtract %503, %504 : tensor<256x56x56x64xf32>
    %506 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %507 = stablehlo.add %502, %506 : tensor<1x1x1x64xf32>
    %508 = stablehlo.rsqrt %507 : tensor<1x1x1x64xf32>
    %509 = stablehlo.divide %508, %507 : tensor<1x1x1x64xf32>
    %510 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %511 = stablehlo.multiply %510, %509 : tensor<1x1x1x64xf32>
    %512 = stablehlo.reshape %arg57 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %513 = stablehlo.multiply %508, %512 : tensor<1x1x1x64xf32>
    %514 = stablehlo.broadcast_in_dim %513, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %515 = stablehlo.multiply %505, %514 : tensor<256x56x56x64xf32>
    %516 = stablehlo.reshape %arg56 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %518 = stablehlo.add %515, %517 : tensor<256x56x56x64xf32>
    %519 = stablehlo.convert %518 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %520 = stablehlo.add %378, %519 : tensor<256x56x56x64xbf16>
    %521 = call @relu_41(%520) : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xbf16>
    %522 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %523 = stablehlo.compare  GT, %520, %522,  FLOAT : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xi1>
    %524 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %525 = stablehlo.convert %arg64 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xbf16>
    %526 = stablehlo.convolution(%521, %525) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x128xbf16>) -> tensor<256x28x28x128xbf16>
    %527 = stablehlo.convert %526 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %528 = stablehlo.multiply %527, %527 : tensor<256x28x28x128xf32>
    %529 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %530 = stablehlo.multiply %529, %527 : tensor<256x28x28x128xf32>
    %cst_33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %531 = stablehlo.reduce(%527 init: %cst_33) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_34 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
    %532 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %533 = stablehlo.divide %531, %532 : tensor<128xf32>
    %cst_35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %534 = stablehlo.reduce(%528 init: %cst_35) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %535 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %536 = stablehlo.divide %534, %535 : tensor<128xf32>
    %537 = stablehlo.broadcast_in_dim %533, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %538 = stablehlo.broadcast_in_dim %536, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %539 = stablehlo.concatenate %537, %538, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %540 = "stablehlo.all_reduce"(%539) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %541 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %542 = stablehlo.divide %540, %541 : tensor<2x128xf32>
    %543 = stablehlo.slice %542 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %544 = stablehlo.reshape %543 : (tensor<1x128xf32>) -> tensor<128xf32>
    %545 = stablehlo.slice %542 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %546 = stablehlo.reshape %545 : (tensor<1x128xf32>) -> tensor<128xf32>
    %547 = stablehlo.multiply %544, %544 : tensor<128xf32>
    %548 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %549 = stablehlo.multiply %548, %544 : tensor<128xf32>
    %550 = stablehlo.subtract %546, %547 : tensor<128xf32>
    %551 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %552 = stablehlo.maximum %551, %550 : tensor<128xf32>
    %553 = stablehlo.compare  EQ, %550, %552,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %554 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %555 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %556 = stablehlo.select %553, %554, %555 : tensor<128xi1>, tensor<128xf32>
    %557 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %558 = stablehlo.compare  EQ, %557, %552,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %559 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %560 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %561 = stablehlo.select %558, %559, %560 : tensor<128xi1>, tensor<128xf32>
    %562 = stablehlo.divide %556, %561 : tensor<128xf32>
    %563 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %564 = stablehlo.multiply %563, %arg260 : tensor<128xf32>
    %565 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %566 = stablehlo.multiply %565, %544 : tensor<128xf32>
    %567 = stablehlo.add %564, %566 : tensor<128xf32>
    %568 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %569 = stablehlo.multiply %568, %arg261 : tensor<128xf32>
    %570 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %571 = stablehlo.multiply %570, %552 : tensor<128xf32>
    %572 = stablehlo.add %569, %571 : tensor<128xf32>
    %573 = stablehlo.broadcast_in_dim %544, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %574 = stablehlo.broadcast_in_dim %552, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %575 = stablehlo.convert %526 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %576 = stablehlo.broadcast_in_dim %573, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %577 = stablehlo.subtract %575, %576 : tensor<256x28x28x128xf32>
    %578 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %579 = stablehlo.add %574, %578 : tensor<1x1x1x128xf32>
    %580 = stablehlo.rsqrt %579 : tensor<1x1x1x128xf32>
    %581 = stablehlo.divide %580, %579 : tensor<1x1x1x128xf32>
    %582 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %583 = stablehlo.multiply %582, %581 : tensor<1x1x1x128xf32>
    %584 = stablehlo.reshape %arg61 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %585 = stablehlo.multiply %580, %584 : tensor<1x1x1x128xf32>
    %586 = stablehlo.broadcast_in_dim %585, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %587 = stablehlo.multiply %577, %586 : tensor<256x28x28x128xf32>
    %588 = stablehlo.reshape %arg60 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %589 = stablehlo.broadcast_in_dim %588, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %590 = stablehlo.add %587, %589 : tensor<256x28x28x128xf32>
    %591 = stablehlo.convert %590 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %592 = call @relu_87(%591) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %593 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %594 = stablehlo.compare  GT, %591, %593,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %595 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %596 = stablehlo.convert %arg65 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %597 = stablehlo.convolution(%592, %596) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %598 = stablehlo.convert %597 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %599 = stablehlo.multiply %598, %598 : tensor<256x28x28x128xf32>
    %600 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %601 = stablehlo.multiply %600, %598 : tensor<256x28x28x128xf32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %602 = stablehlo.reduce(%598 init: %cst_36) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %603 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %604 = stablehlo.divide %602, %603 : tensor<128xf32>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %605 = stablehlo.reduce(%599 init: %cst_37) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %606 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %607 = stablehlo.divide %605, %606 : tensor<128xf32>
    %608 = stablehlo.broadcast_in_dim %604, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %609 = stablehlo.broadcast_in_dim %607, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %610 = stablehlo.concatenate %608, %609, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %611 = "stablehlo.all_reduce"(%610) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %612 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %613 = stablehlo.divide %611, %612 : tensor<2x128xf32>
    %614 = stablehlo.slice %613 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %615 = stablehlo.reshape %614 : (tensor<1x128xf32>) -> tensor<128xf32>
    %616 = stablehlo.slice %613 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %617 = stablehlo.reshape %616 : (tensor<1x128xf32>) -> tensor<128xf32>
    %618 = stablehlo.multiply %615, %615 : tensor<128xf32>
    %619 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %620 = stablehlo.multiply %619, %615 : tensor<128xf32>
    %621 = stablehlo.subtract %617, %618 : tensor<128xf32>
    %622 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %623 = stablehlo.maximum %622, %621 : tensor<128xf32>
    %624 = stablehlo.compare  EQ, %621, %623,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %625 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %626 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %627 = stablehlo.select %624, %625, %626 : tensor<128xi1>, tensor<128xf32>
    %628 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %629 = stablehlo.compare  EQ, %628, %623,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %630 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %631 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %632 = stablehlo.select %629, %630, %631 : tensor<128xi1>, tensor<128xf32>
    %633 = stablehlo.divide %627, %632 : tensor<128xf32>
    %634 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %635 = stablehlo.multiply %634, %arg262 : tensor<128xf32>
    %636 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %637 = stablehlo.multiply %636, %615 : tensor<128xf32>
    %638 = stablehlo.add %635, %637 : tensor<128xf32>
    %639 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %640 = stablehlo.multiply %639, %arg263 : tensor<128xf32>
    %641 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %642 = stablehlo.multiply %641, %623 : tensor<128xf32>
    %643 = stablehlo.add %640, %642 : tensor<128xf32>
    %644 = stablehlo.broadcast_in_dim %615, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %645 = stablehlo.broadcast_in_dim %623, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %646 = stablehlo.convert %597 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %647 = stablehlo.broadcast_in_dim %644, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %648 = stablehlo.subtract %646, %647 : tensor<256x28x28x128xf32>
    %649 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %650 = stablehlo.add %645, %649 : tensor<1x1x1x128xf32>
    %651 = stablehlo.rsqrt %650 : tensor<1x1x1x128xf32>
    %652 = stablehlo.divide %651, %650 : tensor<1x1x1x128xf32>
    %653 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %654 = stablehlo.multiply %653, %652 : tensor<1x1x1x128xf32>
    %655 = stablehlo.reshape %arg63 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %656 = stablehlo.multiply %651, %655 : tensor<1x1x1x128xf32>
    %657 = stablehlo.broadcast_in_dim %656, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %658 = stablehlo.multiply %648, %657 : tensor<256x28x28x128xf32>
    %659 = stablehlo.reshape %arg62 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %660 = stablehlo.broadcast_in_dim %659, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %661 = stablehlo.add %658, %660 : tensor<256x28x28x128xf32>
    %662 = stablehlo.convert %661 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %663 = stablehlo.convert %arg66 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xbf16>
    %664 = stablehlo.convolution(%521, %663) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x128xbf16>) -> tensor<256x28x28x128xbf16>
    %665 = stablehlo.convert %664 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %666 = stablehlo.multiply %665, %665 : tensor<256x28x28x128xf32>
    %667 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %668 = stablehlo.multiply %667, %665 : tensor<256x28x28x128xf32>
    %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %669 = stablehlo.reduce(%665 init: %cst_38) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %670 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %671 = stablehlo.divide %669, %670 : tensor<128xf32>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %672 = stablehlo.reduce(%666 init: %cst_39) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %673 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %674 = stablehlo.divide %672, %673 : tensor<128xf32>
    %675 = stablehlo.broadcast_in_dim %671, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %676 = stablehlo.broadcast_in_dim %674, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %677 = stablehlo.concatenate %675, %676, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %678 = "stablehlo.all_reduce"(%677) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
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
    %702 = stablehlo.multiply %701, %arg264 : tensor<128xf32>
    %703 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %704 = stablehlo.multiply %703, %682 : tensor<128xf32>
    %705 = stablehlo.add %702, %704 : tensor<128xf32>
    %706 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %707 = stablehlo.multiply %706, %arg265 : tensor<128xf32>
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
    %722 = stablehlo.reshape %arg68 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %723 = stablehlo.multiply %718, %722 : tensor<1x1x1x128xf32>
    %724 = stablehlo.broadcast_in_dim %723, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %725 = stablehlo.multiply %715, %724 : tensor<256x28x28x128xf32>
    %726 = stablehlo.reshape %arg67 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %727 = stablehlo.broadcast_in_dim %726, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %728 = stablehlo.add %725, %727 : tensor<256x28x28x128xf32>
    %729 = stablehlo.convert %728 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %730 = stablehlo.add %729, %662 : tensor<256x28x28x128xbf16>
    %731 = call @relu_87(%730) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %732 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %733 = stablehlo.compare  GT, %730, %732,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %734 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %735 = stablehlo.convert %arg73 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %736 = stablehlo.convolution(%731, %735) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %737 = stablehlo.convert %736 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %738 = stablehlo.multiply %737, %737 : tensor<256x28x28x128xf32>
    %739 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %740 = stablehlo.multiply %739, %737 : tensor<256x28x28x128xf32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %741 = stablehlo.reduce(%737 init: %cst_40) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %742 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %743 = stablehlo.divide %741, %742 : tensor<128xf32>
    %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %744 = stablehlo.reduce(%738 init: %cst_41) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %745 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %746 = stablehlo.divide %744, %745 : tensor<128xf32>
    %747 = stablehlo.broadcast_in_dim %743, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %748 = stablehlo.broadcast_in_dim %746, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %749 = stablehlo.concatenate %747, %748, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %750 = "stablehlo.all_reduce"(%749) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %751 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %752 = stablehlo.divide %750, %751 : tensor<2x128xf32>
    %753 = stablehlo.slice %752 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %754 = stablehlo.reshape %753 : (tensor<1x128xf32>) -> tensor<128xf32>
    %755 = stablehlo.slice %752 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %756 = stablehlo.reshape %755 : (tensor<1x128xf32>) -> tensor<128xf32>
    %757 = stablehlo.multiply %754, %754 : tensor<128xf32>
    %758 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %759 = stablehlo.multiply %758, %754 : tensor<128xf32>
    %760 = stablehlo.subtract %756, %757 : tensor<128xf32>
    %761 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %762 = stablehlo.maximum %761, %760 : tensor<128xf32>
    %763 = stablehlo.compare  EQ, %760, %762,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %764 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %765 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %766 = stablehlo.select %763, %764, %765 : tensor<128xi1>, tensor<128xf32>
    %767 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %768 = stablehlo.compare  EQ, %767, %762,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %769 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %770 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %771 = stablehlo.select %768, %769, %770 : tensor<128xi1>, tensor<128xf32>
    %772 = stablehlo.divide %766, %771 : tensor<128xf32>
    %773 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %774 = stablehlo.multiply %773, %arg266 : tensor<128xf32>
    %775 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %776 = stablehlo.multiply %775, %754 : tensor<128xf32>
    %777 = stablehlo.add %774, %776 : tensor<128xf32>
    %778 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %779 = stablehlo.multiply %778, %arg267 : tensor<128xf32>
    %780 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %781 = stablehlo.multiply %780, %762 : tensor<128xf32>
    %782 = stablehlo.add %779, %781 : tensor<128xf32>
    %783 = stablehlo.broadcast_in_dim %754, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %784 = stablehlo.broadcast_in_dim %762, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %785 = stablehlo.convert %736 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %786 = stablehlo.broadcast_in_dim %783, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %787 = stablehlo.subtract %785, %786 : tensor<256x28x28x128xf32>
    %788 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %789 = stablehlo.add %784, %788 : tensor<1x1x1x128xf32>
    %790 = stablehlo.rsqrt %789 : tensor<1x1x1x128xf32>
    %791 = stablehlo.divide %790, %789 : tensor<1x1x1x128xf32>
    %792 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %793 = stablehlo.multiply %792, %791 : tensor<1x1x1x128xf32>
    %794 = stablehlo.reshape %arg70 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %795 = stablehlo.multiply %790, %794 : tensor<1x1x1x128xf32>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %797 = stablehlo.multiply %787, %796 : tensor<256x28x28x128xf32>
    %798 = stablehlo.reshape %arg69 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %799 = stablehlo.broadcast_in_dim %798, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %800 = stablehlo.add %797, %799 : tensor<256x28x28x128xf32>
    %801 = stablehlo.convert %800 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %802 = call @relu_87(%801) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %803 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %804 = stablehlo.compare  GT, %801, %803,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %805 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %806 = stablehlo.convert %arg74 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %807 = stablehlo.convolution(%802, %806) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %808 = stablehlo.convert %807 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %809 = stablehlo.multiply %808, %808 : tensor<256x28x28x128xf32>
    %810 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %811 = stablehlo.multiply %810, %808 : tensor<256x28x28x128xf32>
    %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %812 = stablehlo.reduce(%808 init: %cst_42) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %813 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %814 = stablehlo.divide %812, %813 : tensor<128xf32>
    %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %815 = stablehlo.reduce(%809 init: %cst_43) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %816 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %817 = stablehlo.divide %815, %816 : tensor<128xf32>
    %818 = stablehlo.broadcast_in_dim %814, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %819 = stablehlo.broadcast_in_dim %817, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %820 = stablehlo.concatenate %818, %819, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %821 = "stablehlo.all_reduce"(%820) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %822 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %823 = stablehlo.divide %821, %822 : tensor<2x128xf32>
    %824 = stablehlo.slice %823 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %825 = stablehlo.reshape %824 : (tensor<1x128xf32>) -> tensor<128xf32>
    %826 = stablehlo.slice %823 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %827 = stablehlo.reshape %826 : (tensor<1x128xf32>) -> tensor<128xf32>
    %828 = stablehlo.multiply %825, %825 : tensor<128xf32>
    %829 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %830 = stablehlo.multiply %829, %825 : tensor<128xf32>
    %831 = stablehlo.subtract %827, %828 : tensor<128xf32>
    %832 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %833 = stablehlo.maximum %832, %831 : tensor<128xf32>
    %834 = stablehlo.compare  EQ, %831, %833,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %835 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %836 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %837 = stablehlo.select %834, %835, %836 : tensor<128xi1>, tensor<128xf32>
    %838 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %839 = stablehlo.compare  EQ, %838, %833,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %840 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %841 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %842 = stablehlo.select %839, %840, %841 : tensor<128xi1>, tensor<128xf32>
    %843 = stablehlo.divide %837, %842 : tensor<128xf32>
    %844 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %845 = stablehlo.multiply %844, %arg268 : tensor<128xf32>
    %846 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %847 = stablehlo.multiply %846, %825 : tensor<128xf32>
    %848 = stablehlo.add %845, %847 : tensor<128xf32>
    %849 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %850 = stablehlo.multiply %849, %arg269 : tensor<128xf32>
    %851 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %852 = stablehlo.multiply %851, %833 : tensor<128xf32>
    %853 = stablehlo.add %850, %852 : tensor<128xf32>
    %854 = stablehlo.broadcast_in_dim %825, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %855 = stablehlo.broadcast_in_dim %833, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %856 = stablehlo.convert %807 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %857 = stablehlo.broadcast_in_dim %854, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %858 = stablehlo.subtract %856, %857 : tensor<256x28x28x128xf32>
    %859 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %860 = stablehlo.add %855, %859 : tensor<1x1x1x128xf32>
    %861 = stablehlo.rsqrt %860 : tensor<1x1x1x128xf32>
    %862 = stablehlo.divide %861, %860 : tensor<1x1x1x128xf32>
    %863 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %864 = stablehlo.multiply %863, %862 : tensor<1x1x1x128xf32>
    %865 = stablehlo.reshape %arg72 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %866 = stablehlo.multiply %861, %865 : tensor<1x1x1x128xf32>
    %867 = stablehlo.broadcast_in_dim %866, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %868 = stablehlo.multiply %858, %867 : tensor<256x28x28x128xf32>
    %869 = stablehlo.reshape %arg71 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %870 = stablehlo.broadcast_in_dim %869, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %871 = stablehlo.add %868, %870 : tensor<256x28x28x128xf32>
    %872 = stablehlo.convert %871 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %873 = stablehlo.add %731, %872 : tensor<256x28x28x128xbf16>
    %874 = call @relu_87(%873) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %875 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %876 = stablehlo.compare  GT, %873, %875,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %877 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %878 = stablehlo.convert %arg79 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %879 = stablehlo.convolution(%874, %878) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %880 = stablehlo.convert %879 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %881 = stablehlo.multiply %880, %880 : tensor<256x28x28x128xf32>
    %882 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %883 = stablehlo.multiply %882, %880 : tensor<256x28x28x128xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %884 = stablehlo.reduce(%880 init: %cst_44) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %885 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %886 = stablehlo.divide %884, %885 : tensor<128xf32>
    %cst_45 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %887 = stablehlo.reduce(%881 init: %cst_45) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %888 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %889 = stablehlo.divide %887, %888 : tensor<128xf32>
    %890 = stablehlo.broadcast_in_dim %886, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %891 = stablehlo.broadcast_in_dim %889, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %892 = stablehlo.concatenate %890, %891, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %893 = "stablehlo.all_reduce"(%892) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %894 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %895 = stablehlo.divide %893, %894 : tensor<2x128xf32>
    %896 = stablehlo.slice %895 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %897 = stablehlo.reshape %896 : (tensor<1x128xf32>) -> tensor<128xf32>
    %898 = stablehlo.slice %895 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %899 = stablehlo.reshape %898 : (tensor<1x128xf32>) -> tensor<128xf32>
    %900 = stablehlo.multiply %897, %897 : tensor<128xf32>
    %901 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %902 = stablehlo.multiply %901, %897 : tensor<128xf32>
    %903 = stablehlo.subtract %899, %900 : tensor<128xf32>
    %904 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %905 = stablehlo.maximum %904, %903 : tensor<128xf32>
    %906 = stablehlo.compare  EQ, %903, %905,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %907 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %908 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %909 = stablehlo.select %906, %907, %908 : tensor<128xi1>, tensor<128xf32>
    %910 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %911 = stablehlo.compare  EQ, %910, %905,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %912 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %913 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %914 = stablehlo.select %911, %912, %913 : tensor<128xi1>, tensor<128xf32>
    %915 = stablehlo.divide %909, %914 : tensor<128xf32>
    %916 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %917 = stablehlo.multiply %916, %arg270 : tensor<128xf32>
    %918 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %919 = stablehlo.multiply %918, %897 : tensor<128xf32>
    %920 = stablehlo.add %917, %919 : tensor<128xf32>
    %921 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %922 = stablehlo.multiply %921, %arg271 : tensor<128xf32>
    %923 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %924 = stablehlo.multiply %923, %905 : tensor<128xf32>
    %925 = stablehlo.add %922, %924 : tensor<128xf32>
    %926 = stablehlo.broadcast_in_dim %897, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %927 = stablehlo.broadcast_in_dim %905, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %928 = stablehlo.convert %879 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %929 = stablehlo.broadcast_in_dim %926, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %930 = stablehlo.subtract %928, %929 : tensor<256x28x28x128xf32>
    %931 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %932 = stablehlo.add %927, %931 : tensor<1x1x1x128xf32>
    %933 = stablehlo.rsqrt %932 : tensor<1x1x1x128xf32>
    %934 = stablehlo.divide %933, %932 : tensor<1x1x1x128xf32>
    %935 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %936 = stablehlo.multiply %935, %934 : tensor<1x1x1x128xf32>
    %937 = stablehlo.reshape %arg76 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %938 = stablehlo.multiply %933, %937 : tensor<1x1x1x128xf32>
    %939 = stablehlo.broadcast_in_dim %938, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %940 = stablehlo.multiply %930, %939 : tensor<256x28x28x128xf32>
    %941 = stablehlo.reshape %arg75 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %942 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %943 = stablehlo.add %940, %942 : tensor<256x28x28x128xf32>
    %944 = stablehlo.convert %943 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %945 = call @relu_87(%944) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %946 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %947 = stablehlo.compare  GT, %944, %946,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %948 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %949 = stablehlo.convert %arg80 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %950 = stablehlo.convolution(%945, %949) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %951 = stablehlo.convert %950 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %952 = stablehlo.multiply %951, %951 : tensor<256x28x28x128xf32>
    %953 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %954 = stablehlo.multiply %953, %951 : tensor<256x28x28x128xf32>
    %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %955 = stablehlo.reduce(%951 init: %cst_46) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %956 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %957 = stablehlo.divide %955, %956 : tensor<128xf32>
    %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %958 = stablehlo.reduce(%952 init: %cst_47) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %959 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %960 = stablehlo.divide %958, %959 : tensor<128xf32>
    %961 = stablehlo.broadcast_in_dim %957, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %962 = stablehlo.broadcast_in_dim %960, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %963 = stablehlo.concatenate %961, %962, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %964 = "stablehlo.all_reduce"(%963) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %965 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %966 = stablehlo.divide %964, %965 : tensor<2x128xf32>
    %967 = stablehlo.slice %966 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %968 = stablehlo.reshape %967 : (tensor<1x128xf32>) -> tensor<128xf32>
    %969 = stablehlo.slice %966 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %970 = stablehlo.reshape %969 : (tensor<1x128xf32>) -> tensor<128xf32>
    %971 = stablehlo.multiply %968, %968 : tensor<128xf32>
    %972 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %973 = stablehlo.multiply %972, %968 : tensor<128xf32>
    %974 = stablehlo.subtract %970, %971 : tensor<128xf32>
    %975 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %976 = stablehlo.maximum %975, %974 : tensor<128xf32>
    %977 = stablehlo.compare  EQ, %974, %976,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %978 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %979 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %980 = stablehlo.select %977, %978, %979 : tensor<128xi1>, tensor<128xf32>
    %981 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %982 = stablehlo.compare  EQ, %981, %976,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %983 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %984 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %985 = stablehlo.select %982, %983, %984 : tensor<128xi1>, tensor<128xf32>
    %986 = stablehlo.divide %980, %985 : tensor<128xf32>
    %987 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %988 = stablehlo.multiply %987, %arg272 : tensor<128xf32>
    %989 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %990 = stablehlo.multiply %989, %968 : tensor<128xf32>
    %991 = stablehlo.add %988, %990 : tensor<128xf32>
    %992 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %993 = stablehlo.multiply %992, %arg273 : tensor<128xf32>
    %994 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %995 = stablehlo.multiply %994, %976 : tensor<128xf32>
    %996 = stablehlo.add %993, %995 : tensor<128xf32>
    %997 = stablehlo.broadcast_in_dim %968, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %998 = stablehlo.broadcast_in_dim %976, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %999 = stablehlo.convert %950 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1000 = stablehlo.broadcast_in_dim %997, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1001 = stablehlo.subtract %999, %1000 : tensor<256x28x28x128xf32>
    %1002 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1003 = stablehlo.add %998, %1002 : tensor<1x1x1x128xf32>
    %1004 = stablehlo.rsqrt %1003 : tensor<1x1x1x128xf32>
    %1005 = stablehlo.divide %1004, %1003 : tensor<1x1x1x128xf32>
    %1006 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1007 = stablehlo.multiply %1006, %1005 : tensor<1x1x1x128xf32>
    %1008 = stablehlo.reshape %arg78 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1009 = stablehlo.multiply %1004, %1008 : tensor<1x1x1x128xf32>
    %1010 = stablehlo.broadcast_in_dim %1009, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1011 = stablehlo.multiply %1001, %1010 : tensor<256x28x28x128xf32>
    %1012 = stablehlo.reshape %arg77 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1013 = stablehlo.broadcast_in_dim %1012, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1014 = stablehlo.add %1011, %1013 : tensor<256x28x28x128xf32>
    %1015 = stablehlo.convert %1014 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1016 = stablehlo.add %874, %1015 : tensor<256x28x28x128xbf16>
    %1017 = call @relu_87(%1016) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1018 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1019 = stablehlo.compare  GT, %1016, %1018,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %1020 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1021 = stablehlo.convert %arg85 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1022 = stablehlo.convolution(%1017, %1021) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1023 = stablehlo.convert %1022 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1024 = stablehlo.multiply %1023, %1023 : tensor<256x28x28x128xf32>
    %1025 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %1026 = stablehlo.multiply %1025, %1023 : tensor<256x28x28x128xf32>
    %cst_48 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1027 = stablehlo.reduce(%1023 init: %cst_48) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %1028 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1029 = stablehlo.divide %1027, %1028 : tensor<128xf32>
    %cst_49 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1030 = stablehlo.reduce(%1024 init: %cst_49) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %1031 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1032 = stablehlo.divide %1030, %1031 : tensor<128xf32>
    %1033 = stablehlo.broadcast_in_dim %1029, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %1034 = stablehlo.broadcast_in_dim %1032, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %1035 = stablehlo.concatenate %1033, %1034, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %1036 = "stablehlo.all_reduce"(%1035) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %1037 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %1038 = stablehlo.divide %1036, %1037 : tensor<2x128xf32>
    %1039 = stablehlo.slice %1038 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %1040 = stablehlo.reshape %1039 : (tensor<1x128xf32>) -> tensor<128xf32>
    %1041 = stablehlo.slice %1038 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %1042 = stablehlo.reshape %1041 : (tensor<1x128xf32>) -> tensor<128xf32>
    %1043 = stablehlo.multiply %1040, %1040 : tensor<128xf32>
    %1044 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1045 = stablehlo.multiply %1044, %1040 : tensor<128xf32>
    %1046 = stablehlo.subtract %1042, %1043 : tensor<128xf32>
    %1047 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1048 = stablehlo.maximum %1047, %1046 : tensor<128xf32>
    %1049 = stablehlo.compare  EQ, %1046, %1048,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %1050 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1051 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1052 = stablehlo.select %1049, %1050, %1051 : tensor<128xi1>, tensor<128xf32>
    %1053 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1054 = stablehlo.compare  EQ, %1053, %1048,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %1055 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1056 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1057 = stablehlo.select %1054, %1055, %1056 : tensor<128xi1>, tensor<128xf32>
    %1058 = stablehlo.divide %1052, %1057 : tensor<128xf32>
    %1059 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1060 = stablehlo.multiply %1059, %arg274 : tensor<128xf32>
    %1061 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1062 = stablehlo.multiply %1061, %1040 : tensor<128xf32>
    %1063 = stablehlo.add %1060, %1062 : tensor<128xf32>
    %1064 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1065 = stablehlo.multiply %1064, %arg275 : tensor<128xf32>
    %1066 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1067 = stablehlo.multiply %1066, %1048 : tensor<128xf32>
    %1068 = stablehlo.add %1065, %1067 : tensor<128xf32>
    %1069 = stablehlo.broadcast_in_dim %1040, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1070 = stablehlo.broadcast_in_dim %1048, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1071 = stablehlo.convert %1022 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1072 = stablehlo.broadcast_in_dim %1069, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1073 = stablehlo.subtract %1071, %1072 : tensor<256x28x28x128xf32>
    %1074 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1075 = stablehlo.add %1070, %1074 : tensor<1x1x1x128xf32>
    %1076 = stablehlo.rsqrt %1075 : tensor<1x1x1x128xf32>
    %1077 = stablehlo.divide %1076, %1075 : tensor<1x1x1x128xf32>
    %1078 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1079 = stablehlo.multiply %1078, %1077 : tensor<1x1x1x128xf32>
    %1080 = stablehlo.reshape %arg82 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1081 = stablehlo.multiply %1076, %1080 : tensor<1x1x1x128xf32>
    %1082 = stablehlo.broadcast_in_dim %1081, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1083 = stablehlo.multiply %1073, %1082 : tensor<256x28x28x128xf32>
    %1084 = stablehlo.reshape %arg81 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1085 = stablehlo.broadcast_in_dim %1084, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1086 = stablehlo.add %1083, %1085 : tensor<256x28x28x128xf32>
    %1087 = stablehlo.convert %1086 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1088 = call @relu_87(%1087) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1089 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1090 = stablehlo.compare  GT, %1087, %1089,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %1091 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1092 = stablehlo.convert %arg86 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xbf16>
    %1093 = stablehlo.convolution(%1088, %1092) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1094 = stablehlo.convert %1093 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1095 = stablehlo.multiply %1094, %1094 : tensor<256x28x28x128xf32>
    %1096 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
    %1097 = stablehlo.multiply %1096, %1094 : tensor<256x28x28x128xf32>
    %cst_50 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1098 = stablehlo.reduce(%1094 init: %cst_50) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %1099 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1100 = stablehlo.divide %1098, %1099 : tensor<128xf32>
    %cst_51 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1101 = stablehlo.reduce(%1095 init: %cst_51) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %1102 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1103 = stablehlo.divide %1101, %1102 : tensor<128xf32>
    %1104 = stablehlo.broadcast_in_dim %1100, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %1105 = stablehlo.broadcast_in_dim %1103, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %1106 = stablehlo.concatenate %1104, %1105, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
    %1107 = "stablehlo.all_reduce"(%1106) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %1108 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %1109 = stablehlo.divide %1107, %1108 : tensor<2x128xf32>
    %1110 = stablehlo.slice %1109 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %1111 = stablehlo.reshape %1110 : (tensor<1x128xf32>) -> tensor<128xf32>
    %1112 = stablehlo.slice %1109 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %1113 = stablehlo.reshape %1112 : (tensor<1x128xf32>) -> tensor<128xf32>
    %1114 = stablehlo.multiply %1111, %1111 : tensor<128xf32>
    %1115 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1116 = stablehlo.multiply %1115, %1111 : tensor<128xf32>
    %1117 = stablehlo.subtract %1113, %1114 : tensor<128xf32>
    %1118 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1119 = stablehlo.maximum %1118, %1117 : tensor<128xf32>
    %1120 = stablehlo.compare  EQ, %1117, %1119,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %1121 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1122 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1123 = stablehlo.select %1120, %1121, %1122 : tensor<128xi1>, tensor<128xf32>
    %1124 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1125 = stablehlo.compare  EQ, %1124, %1119,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %1126 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1127 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1128 = stablehlo.select %1125, %1126, %1127 : tensor<128xi1>, tensor<128xf32>
    %1129 = stablehlo.divide %1123, %1128 : tensor<128xf32>
    %1130 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1131 = stablehlo.multiply %1130, %arg276 : tensor<128xf32>
    %1132 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1133 = stablehlo.multiply %1132, %1111 : tensor<128xf32>
    %1134 = stablehlo.add %1131, %1133 : tensor<128xf32>
    %1135 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1136 = stablehlo.multiply %1135, %arg277 : tensor<128xf32>
    %1137 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1138 = stablehlo.multiply %1137, %1119 : tensor<128xf32>
    %1139 = stablehlo.add %1136, %1138 : tensor<128xf32>
    %1140 = stablehlo.broadcast_in_dim %1111, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1141 = stablehlo.broadcast_in_dim %1119, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1142 = stablehlo.convert %1093 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %1143 = stablehlo.broadcast_in_dim %1140, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1144 = stablehlo.subtract %1142, %1143 : tensor<256x28x28x128xf32>
    %1145 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1146 = stablehlo.add %1141, %1145 : tensor<1x1x1x128xf32>
    %1147 = stablehlo.rsqrt %1146 : tensor<1x1x1x128xf32>
    %1148 = stablehlo.divide %1147, %1146 : tensor<1x1x1x128xf32>
    %1149 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %1150 = stablehlo.multiply %1149, %1148 : tensor<1x1x1x128xf32>
    %1151 = stablehlo.reshape %arg84 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1152 = stablehlo.multiply %1147, %1151 : tensor<1x1x1x128xf32>
    %1153 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1154 = stablehlo.multiply %1144, %1153 : tensor<256x28x28x128xf32>
    %1155 = stablehlo.reshape %arg83 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %1156 = stablehlo.broadcast_in_dim %1155, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %1157 = stablehlo.add %1154, %1156 : tensor<256x28x28x128xf32>
    %1158 = stablehlo.convert %1157 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %1159 = stablehlo.add %1017, %1158 : tensor<256x28x28x128xbf16>
    %1160 = call @relu_87(%1159) : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16>
    %1161 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1162 = stablehlo.compare  GT, %1159, %1161,  FLOAT : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xi1>
    %1163 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1164 = stablehlo.convert %arg91 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xbf16>
    %1165 = stablehlo.convolution(%1160, %1164) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1166 = stablehlo.convert %1165 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1167 = stablehlo.multiply %1166, %1166 : tensor<256x14x14x256xf32>
    %1168 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1169 = stablehlo.multiply %1168, %1166 : tensor<256x14x14x256xf32>
    %cst_52 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1170 = stablehlo.reduce(%1166 init: %cst_52) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_53 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
    %1171 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1172 = stablehlo.divide %1170, %1171 : tensor<256xf32>
    %cst_54 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1173 = stablehlo.reduce(%1167 init: %cst_54) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1174 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1175 = stablehlo.divide %1173, %1174 : tensor<256xf32>
    %1176 = stablehlo.broadcast_in_dim %1172, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1177 = stablehlo.broadcast_in_dim %1175, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1178 = stablehlo.concatenate %1176, %1177, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1179 = "stablehlo.all_reduce"(%1178) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1180 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1181 = stablehlo.divide %1179, %1180 : tensor<2x256xf32>
    %1182 = stablehlo.slice %1181 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1183 = stablehlo.reshape %1182 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1184 = stablehlo.slice %1181 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1185 = stablehlo.reshape %1184 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1186 = stablehlo.multiply %1183, %1183 : tensor<256xf32>
    %1187 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1188 = stablehlo.multiply %1187, %1183 : tensor<256xf32>
    %1189 = stablehlo.subtract %1185, %1186 : tensor<256xf32>
    %1190 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1191 = stablehlo.maximum %1190, %1189 : tensor<256xf32>
    %1192 = stablehlo.compare  EQ, %1189, %1191,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1193 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1194 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1195 = stablehlo.select %1192, %1193, %1194 : tensor<256xi1>, tensor<256xf32>
    %1196 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1197 = stablehlo.compare  EQ, %1196, %1191,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1198 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1199 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1200 = stablehlo.select %1197, %1198, %1199 : tensor<256xi1>, tensor<256xf32>
    %1201 = stablehlo.divide %1195, %1200 : tensor<256xf32>
    %1202 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1203 = stablehlo.multiply %1202, %arg278 : tensor<256xf32>
    %1204 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1205 = stablehlo.multiply %1204, %1183 : tensor<256xf32>
    %1206 = stablehlo.add %1203, %1205 : tensor<256xf32>
    %1207 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1208 = stablehlo.multiply %1207, %arg279 : tensor<256xf32>
    %1209 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1210 = stablehlo.multiply %1209, %1191 : tensor<256xf32>
    %1211 = stablehlo.add %1208, %1210 : tensor<256xf32>
    %1212 = stablehlo.broadcast_in_dim %1183, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1213 = stablehlo.broadcast_in_dim %1191, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1214 = stablehlo.convert %1165 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1215 = stablehlo.broadcast_in_dim %1212, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1216 = stablehlo.subtract %1214, %1215 : tensor<256x14x14x256xf32>
    %1217 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1218 = stablehlo.add %1213, %1217 : tensor<1x1x1x256xf32>
    %1219 = stablehlo.rsqrt %1218 : tensor<1x1x1x256xf32>
    %1220 = stablehlo.divide %1219, %1218 : tensor<1x1x1x256xf32>
    %1221 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1222 = stablehlo.multiply %1221, %1220 : tensor<1x1x1x256xf32>
    %1223 = stablehlo.reshape %arg88 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1224 = stablehlo.multiply %1219, %1223 : tensor<1x1x1x256xf32>
    %1225 = stablehlo.broadcast_in_dim %1224, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1226 = stablehlo.multiply %1216, %1225 : tensor<256x14x14x256xf32>
    %1227 = stablehlo.reshape %arg87 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1228 = stablehlo.broadcast_in_dim %1227, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1229 = stablehlo.add %1226, %1228 : tensor<256x14x14x256xf32>
    %1230 = stablehlo.convert %1229 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1231 = call @relu_139(%1230) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1232 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1233 = stablehlo.compare  GT, %1230, %1232,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1234 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1235 = stablehlo.convert %arg92 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1236 = stablehlo.convolution(%1231, %1235) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1237 = stablehlo.convert %1236 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1238 = stablehlo.multiply %1237, %1237 : tensor<256x14x14x256xf32>
    %1239 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1240 = stablehlo.multiply %1239, %1237 : tensor<256x14x14x256xf32>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1241 = stablehlo.reduce(%1237 init: %cst_55) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1242 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1243 = stablehlo.divide %1241, %1242 : tensor<256xf32>
    %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1244 = stablehlo.reduce(%1238 init: %cst_56) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1245 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1246 = stablehlo.divide %1244, %1245 : tensor<256xf32>
    %1247 = stablehlo.broadcast_in_dim %1243, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1248 = stablehlo.broadcast_in_dim %1246, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1249 = stablehlo.concatenate %1247, %1248, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1250 = "stablehlo.all_reduce"(%1249) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1251 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1252 = stablehlo.divide %1250, %1251 : tensor<2x256xf32>
    %1253 = stablehlo.slice %1252 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1254 = stablehlo.reshape %1253 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1255 = stablehlo.slice %1252 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1256 = stablehlo.reshape %1255 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1257 = stablehlo.multiply %1254, %1254 : tensor<256xf32>
    %1258 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1259 = stablehlo.multiply %1258, %1254 : tensor<256xf32>
    %1260 = stablehlo.subtract %1256, %1257 : tensor<256xf32>
    %1261 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1262 = stablehlo.maximum %1261, %1260 : tensor<256xf32>
    %1263 = stablehlo.compare  EQ, %1260, %1262,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1264 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1265 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1266 = stablehlo.select %1263, %1264, %1265 : tensor<256xi1>, tensor<256xf32>
    %1267 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1268 = stablehlo.compare  EQ, %1267, %1262,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1269 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1270 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1271 = stablehlo.select %1268, %1269, %1270 : tensor<256xi1>, tensor<256xf32>
    %1272 = stablehlo.divide %1266, %1271 : tensor<256xf32>
    %1273 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1274 = stablehlo.multiply %1273, %arg280 : tensor<256xf32>
    %1275 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1276 = stablehlo.multiply %1275, %1254 : tensor<256xf32>
    %1277 = stablehlo.add %1274, %1276 : tensor<256xf32>
    %1278 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1279 = stablehlo.multiply %1278, %arg281 : tensor<256xf32>
    %1280 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1281 = stablehlo.multiply %1280, %1262 : tensor<256xf32>
    %1282 = stablehlo.add %1279, %1281 : tensor<256xf32>
    %1283 = stablehlo.broadcast_in_dim %1254, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1284 = stablehlo.broadcast_in_dim %1262, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1285 = stablehlo.convert %1236 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1286 = stablehlo.broadcast_in_dim %1283, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1287 = stablehlo.subtract %1285, %1286 : tensor<256x14x14x256xf32>
    %1288 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1289 = stablehlo.add %1284, %1288 : tensor<1x1x1x256xf32>
    %1290 = stablehlo.rsqrt %1289 : tensor<1x1x1x256xf32>
    %1291 = stablehlo.divide %1290, %1289 : tensor<1x1x1x256xf32>
    %1292 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1293 = stablehlo.multiply %1292, %1291 : tensor<1x1x1x256xf32>
    %1294 = stablehlo.reshape %arg90 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1295 = stablehlo.multiply %1290, %1294 : tensor<1x1x1x256xf32>
    %1296 = stablehlo.broadcast_in_dim %1295, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1297 = stablehlo.multiply %1287, %1296 : tensor<256x14x14x256xf32>
    %1298 = stablehlo.reshape %arg89 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1299 = stablehlo.broadcast_in_dim %1298, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1300 = stablehlo.add %1297, %1299 : tensor<256x14x14x256xf32>
    %1301 = stablehlo.convert %1300 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1302 = stablehlo.convert %arg93 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xbf16>
    %1303 = stablehlo.convolution(%1160, %1302) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x128x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1304 = stablehlo.convert %1303 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1305 = stablehlo.multiply %1304, %1304 : tensor<256x14x14x256xf32>
    %1306 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1307 = stablehlo.multiply %1306, %1304 : tensor<256x14x14x256xf32>
    %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1308 = stablehlo.reduce(%1304 init: %cst_57) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1309 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1310 = stablehlo.divide %1308, %1309 : tensor<256xf32>
    %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1311 = stablehlo.reduce(%1305 init: %cst_58) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1312 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1313 = stablehlo.divide %1311, %1312 : tensor<256xf32>
    %1314 = stablehlo.broadcast_in_dim %1310, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1315 = stablehlo.broadcast_in_dim %1313, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1316 = stablehlo.concatenate %1314, %1315, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1317 = "stablehlo.all_reduce"(%1316) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1318 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1319 = stablehlo.divide %1317, %1318 : tensor<2x256xf32>
    %1320 = stablehlo.slice %1319 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1321 = stablehlo.reshape %1320 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1322 = stablehlo.slice %1319 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1323 = stablehlo.reshape %1322 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1324 = stablehlo.multiply %1321, %1321 : tensor<256xf32>
    %1325 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1326 = stablehlo.multiply %1325, %1321 : tensor<256xf32>
    %1327 = stablehlo.subtract %1323, %1324 : tensor<256xf32>
    %1328 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1329 = stablehlo.maximum %1328, %1327 : tensor<256xf32>
    %1330 = stablehlo.compare  EQ, %1327, %1329,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1331 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1332 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1333 = stablehlo.select %1330, %1331, %1332 : tensor<256xi1>, tensor<256xf32>
    %1334 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1335 = stablehlo.compare  EQ, %1334, %1329,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1336 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1337 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1338 = stablehlo.select %1335, %1336, %1337 : tensor<256xi1>, tensor<256xf32>
    %1339 = stablehlo.divide %1333, %1338 : tensor<256xf32>
    %1340 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1341 = stablehlo.multiply %1340, %arg282 : tensor<256xf32>
    %1342 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1343 = stablehlo.multiply %1342, %1321 : tensor<256xf32>
    %1344 = stablehlo.add %1341, %1343 : tensor<256xf32>
    %1345 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1346 = stablehlo.multiply %1345, %arg283 : tensor<256xf32>
    %1347 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1348 = stablehlo.multiply %1347, %1329 : tensor<256xf32>
    %1349 = stablehlo.add %1346, %1348 : tensor<256xf32>
    %1350 = stablehlo.broadcast_in_dim %1321, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1351 = stablehlo.broadcast_in_dim %1329, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1352 = stablehlo.convert %1303 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1353 = stablehlo.broadcast_in_dim %1350, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1354 = stablehlo.subtract %1352, %1353 : tensor<256x14x14x256xf32>
    %1355 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1356 = stablehlo.add %1351, %1355 : tensor<1x1x1x256xf32>
    %1357 = stablehlo.rsqrt %1356 : tensor<1x1x1x256xf32>
    %1358 = stablehlo.divide %1357, %1356 : tensor<1x1x1x256xf32>
    %1359 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1360 = stablehlo.multiply %1359, %1358 : tensor<1x1x1x256xf32>
    %1361 = stablehlo.reshape %arg95 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1362 = stablehlo.multiply %1357, %1361 : tensor<1x1x1x256xf32>
    %1363 = stablehlo.broadcast_in_dim %1362, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1364 = stablehlo.multiply %1354, %1363 : tensor<256x14x14x256xf32>
    %1365 = stablehlo.reshape %arg94 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1366 = stablehlo.broadcast_in_dim %1365, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1367 = stablehlo.add %1364, %1366 : tensor<256x14x14x256xf32>
    %1368 = stablehlo.convert %1367 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1369 = stablehlo.add %1368, %1301 : tensor<256x14x14x256xbf16>
    %1370 = call @relu_139(%1369) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1371 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1372 = stablehlo.compare  GT, %1369, %1371,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1373 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1374 = stablehlo.convert %arg100 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1375 = stablehlo.convolution(%1370, %1374) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1376 = stablehlo.convert %1375 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1377 = stablehlo.multiply %1376, %1376 : tensor<256x14x14x256xf32>
    %1378 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1379 = stablehlo.multiply %1378, %1376 : tensor<256x14x14x256xf32>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1380 = stablehlo.reduce(%1376 init: %cst_59) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1381 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1382 = stablehlo.divide %1380, %1381 : tensor<256xf32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1383 = stablehlo.reduce(%1377 init: %cst_60) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1384 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1385 = stablehlo.divide %1383, %1384 : tensor<256xf32>
    %1386 = stablehlo.broadcast_in_dim %1382, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1387 = stablehlo.broadcast_in_dim %1385, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1388 = stablehlo.concatenate %1386, %1387, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1389 = "stablehlo.all_reduce"(%1388) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1390 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1391 = stablehlo.divide %1389, %1390 : tensor<2x256xf32>
    %1392 = stablehlo.slice %1391 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1393 = stablehlo.reshape %1392 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1394 = stablehlo.slice %1391 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1395 = stablehlo.reshape %1394 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1396 = stablehlo.multiply %1393, %1393 : tensor<256xf32>
    %1397 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1398 = stablehlo.multiply %1397, %1393 : tensor<256xf32>
    %1399 = stablehlo.subtract %1395, %1396 : tensor<256xf32>
    %1400 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1401 = stablehlo.maximum %1400, %1399 : tensor<256xf32>
    %1402 = stablehlo.compare  EQ, %1399, %1401,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1403 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1404 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1405 = stablehlo.select %1402, %1403, %1404 : tensor<256xi1>, tensor<256xf32>
    %1406 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1407 = stablehlo.compare  EQ, %1406, %1401,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1408 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1409 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1410 = stablehlo.select %1407, %1408, %1409 : tensor<256xi1>, tensor<256xf32>
    %1411 = stablehlo.divide %1405, %1410 : tensor<256xf32>
    %1412 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1413 = stablehlo.multiply %1412, %arg284 : tensor<256xf32>
    %1414 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1415 = stablehlo.multiply %1414, %1393 : tensor<256xf32>
    %1416 = stablehlo.add %1413, %1415 : tensor<256xf32>
    %1417 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1418 = stablehlo.multiply %1417, %arg285 : tensor<256xf32>
    %1419 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1420 = stablehlo.multiply %1419, %1401 : tensor<256xf32>
    %1421 = stablehlo.add %1418, %1420 : tensor<256xf32>
    %1422 = stablehlo.broadcast_in_dim %1393, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1423 = stablehlo.broadcast_in_dim %1401, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1424 = stablehlo.convert %1375 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1425 = stablehlo.broadcast_in_dim %1422, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1426 = stablehlo.subtract %1424, %1425 : tensor<256x14x14x256xf32>
    %1427 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1428 = stablehlo.add %1423, %1427 : tensor<1x1x1x256xf32>
    %1429 = stablehlo.rsqrt %1428 : tensor<1x1x1x256xf32>
    %1430 = stablehlo.divide %1429, %1428 : tensor<1x1x1x256xf32>
    %1431 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1432 = stablehlo.multiply %1431, %1430 : tensor<1x1x1x256xf32>
    %1433 = stablehlo.reshape %arg97 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1434 = stablehlo.multiply %1429, %1433 : tensor<1x1x1x256xf32>
    %1435 = stablehlo.broadcast_in_dim %1434, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1436 = stablehlo.multiply %1426, %1435 : tensor<256x14x14x256xf32>
    %1437 = stablehlo.reshape %arg96 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1438 = stablehlo.broadcast_in_dim %1437, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1439 = stablehlo.add %1436, %1438 : tensor<256x14x14x256xf32>
    %1440 = stablehlo.convert %1439 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1441 = call @relu_139(%1440) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1442 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1443 = stablehlo.compare  GT, %1440, %1442,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1444 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1445 = stablehlo.convert %arg101 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1446 = stablehlo.convolution(%1441, %1445) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1447 = stablehlo.convert %1446 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1448 = stablehlo.multiply %1447, %1447 : tensor<256x14x14x256xf32>
    %1449 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1450 = stablehlo.multiply %1449, %1447 : tensor<256x14x14x256xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1451 = stablehlo.reduce(%1447 init: %cst_61) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1452 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1453 = stablehlo.divide %1451, %1452 : tensor<256xf32>
    %cst_62 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1454 = stablehlo.reduce(%1448 init: %cst_62) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1455 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1456 = stablehlo.divide %1454, %1455 : tensor<256xf32>
    %1457 = stablehlo.broadcast_in_dim %1453, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1458 = stablehlo.broadcast_in_dim %1456, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1459 = stablehlo.concatenate %1457, %1458, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1460 = "stablehlo.all_reduce"(%1459) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1461 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1462 = stablehlo.divide %1460, %1461 : tensor<2x256xf32>
    %1463 = stablehlo.slice %1462 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1464 = stablehlo.reshape %1463 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1465 = stablehlo.slice %1462 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1466 = stablehlo.reshape %1465 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1467 = stablehlo.multiply %1464, %1464 : tensor<256xf32>
    %1468 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1469 = stablehlo.multiply %1468, %1464 : tensor<256xf32>
    %1470 = stablehlo.subtract %1466, %1467 : tensor<256xf32>
    %1471 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1472 = stablehlo.maximum %1471, %1470 : tensor<256xf32>
    %1473 = stablehlo.compare  EQ, %1470, %1472,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1474 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1475 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1476 = stablehlo.select %1473, %1474, %1475 : tensor<256xi1>, tensor<256xf32>
    %1477 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1478 = stablehlo.compare  EQ, %1477, %1472,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1479 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1480 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1481 = stablehlo.select %1478, %1479, %1480 : tensor<256xi1>, tensor<256xf32>
    %1482 = stablehlo.divide %1476, %1481 : tensor<256xf32>
    %1483 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1484 = stablehlo.multiply %1483, %arg286 : tensor<256xf32>
    %1485 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1486 = stablehlo.multiply %1485, %1464 : tensor<256xf32>
    %1487 = stablehlo.add %1484, %1486 : tensor<256xf32>
    %1488 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1489 = stablehlo.multiply %1488, %arg287 : tensor<256xf32>
    %1490 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1491 = stablehlo.multiply %1490, %1472 : tensor<256xf32>
    %1492 = stablehlo.add %1489, %1491 : tensor<256xf32>
    %1493 = stablehlo.broadcast_in_dim %1464, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1494 = stablehlo.broadcast_in_dim %1472, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1495 = stablehlo.convert %1446 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1496 = stablehlo.broadcast_in_dim %1493, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1497 = stablehlo.subtract %1495, %1496 : tensor<256x14x14x256xf32>
    %1498 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1499 = stablehlo.add %1494, %1498 : tensor<1x1x1x256xf32>
    %1500 = stablehlo.rsqrt %1499 : tensor<1x1x1x256xf32>
    %1501 = stablehlo.divide %1500, %1499 : tensor<1x1x1x256xf32>
    %1502 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1503 = stablehlo.multiply %1502, %1501 : tensor<1x1x1x256xf32>
    %1504 = stablehlo.reshape %arg99 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1505 = stablehlo.multiply %1500, %1504 : tensor<1x1x1x256xf32>
    %1506 = stablehlo.broadcast_in_dim %1505, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1507 = stablehlo.multiply %1497, %1506 : tensor<256x14x14x256xf32>
    %1508 = stablehlo.reshape %arg98 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1509 = stablehlo.broadcast_in_dim %1508, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1510 = stablehlo.add %1507, %1509 : tensor<256x14x14x256xf32>
    %1511 = stablehlo.convert %1510 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1512 = stablehlo.add %1370, %1511 : tensor<256x14x14x256xbf16>
    %1513 = call @relu_139(%1512) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1514 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1515 = stablehlo.compare  GT, %1512, %1514,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1516 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1517 = stablehlo.convert %arg106 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1518 = stablehlo.convolution(%1513, %1517) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1519 = stablehlo.convert %1518 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1520 = stablehlo.multiply %1519, %1519 : tensor<256x14x14x256xf32>
    %1521 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1522 = stablehlo.multiply %1521, %1519 : tensor<256x14x14x256xf32>
    %cst_63 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1523 = stablehlo.reduce(%1519 init: %cst_63) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1524 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1525 = stablehlo.divide %1523, %1524 : tensor<256xf32>
    %cst_64 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1526 = stablehlo.reduce(%1520 init: %cst_64) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1527 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1528 = stablehlo.divide %1526, %1527 : tensor<256xf32>
    %1529 = stablehlo.broadcast_in_dim %1525, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1530 = stablehlo.broadcast_in_dim %1528, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1531 = stablehlo.concatenate %1529, %1530, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1532 = "stablehlo.all_reduce"(%1531) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1533 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1534 = stablehlo.divide %1532, %1533 : tensor<2x256xf32>
    %1535 = stablehlo.slice %1534 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1536 = stablehlo.reshape %1535 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1537 = stablehlo.slice %1534 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1538 = stablehlo.reshape %1537 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1539 = stablehlo.multiply %1536, %1536 : tensor<256xf32>
    %1540 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1541 = stablehlo.multiply %1540, %1536 : tensor<256xf32>
    %1542 = stablehlo.subtract %1538, %1539 : tensor<256xf32>
    %1543 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1544 = stablehlo.maximum %1543, %1542 : tensor<256xf32>
    %1545 = stablehlo.compare  EQ, %1542, %1544,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1546 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1547 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1548 = stablehlo.select %1545, %1546, %1547 : tensor<256xi1>, tensor<256xf32>
    %1549 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1550 = stablehlo.compare  EQ, %1549, %1544,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1551 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1552 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1553 = stablehlo.select %1550, %1551, %1552 : tensor<256xi1>, tensor<256xf32>
    %1554 = stablehlo.divide %1548, %1553 : tensor<256xf32>
    %1555 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1556 = stablehlo.multiply %1555, %arg288 : tensor<256xf32>
    %1557 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1558 = stablehlo.multiply %1557, %1536 : tensor<256xf32>
    %1559 = stablehlo.add %1556, %1558 : tensor<256xf32>
    %1560 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1561 = stablehlo.multiply %1560, %arg289 : tensor<256xf32>
    %1562 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1563 = stablehlo.multiply %1562, %1544 : tensor<256xf32>
    %1564 = stablehlo.add %1561, %1563 : tensor<256xf32>
    %1565 = stablehlo.broadcast_in_dim %1536, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1566 = stablehlo.broadcast_in_dim %1544, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1567 = stablehlo.convert %1518 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1568 = stablehlo.broadcast_in_dim %1565, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1569 = stablehlo.subtract %1567, %1568 : tensor<256x14x14x256xf32>
    %1570 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1571 = stablehlo.add %1566, %1570 : tensor<1x1x1x256xf32>
    %1572 = stablehlo.rsqrt %1571 : tensor<1x1x1x256xf32>
    %1573 = stablehlo.divide %1572, %1571 : tensor<1x1x1x256xf32>
    %1574 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1575 = stablehlo.multiply %1574, %1573 : tensor<1x1x1x256xf32>
    %1576 = stablehlo.reshape %arg103 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1577 = stablehlo.multiply %1572, %1576 : tensor<1x1x1x256xf32>
    %1578 = stablehlo.broadcast_in_dim %1577, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1579 = stablehlo.multiply %1569, %1578 : tensor<256x14x14x256xf32>
    %1580 = stablehlo.reshape %arg102 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1581 = stablehlo.broadcast_in_dim %1580, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1582 = stablehlo.add %1579, %1581 : tensor<256x14x14x256xf32>
    %1583 = stablehlo.convert %1582 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1584 = call @relu_139(%1583) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1585 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1586 = stablehlo.compare  GT, %1583, %1585,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1587 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1588 = stablehlo.convert %arg107 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1589 = stablehlo.convolution(%1584, %1588) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1590 = stablehlo.convert %1589 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1591 = stablehlo.multiply %1590, %1590 : tensor<256x14x14x256xf32>
    %1592 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1593 = stablehlo.multiply %1592, %1590 : tensor<256x14x14x256xf32>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1594 = stablehlo.reduce(%1590 init: %cst_65) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1595 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1596 = stablehlo.divide %1594, %1595 : tensor<256xf32>
    %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1597 = stablehlo.reduce(%1591 init: %cst_66) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1598 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1599 = stablehlo.divide %1597, %1598 : tensor<256xf32>
    %1600 = stablehlo.broadcast_in_dim %1596, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1601 = stablehlo.broadcast_in_dim %1599, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1602 = stablehlo.concatenate %1600, %1601, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1603 = "stablehlo.all_reduce"(%1602) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1604 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1605 = stablehlo.divide %1603, %1604 : tensor<2x256xf32>
    %1606 = stablehlo.slice %1605 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1607 = stablehlo.reshape %1606 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1608 = stablehlo.slice %1605 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1609 = stablehlo.reshape %1608 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1610 = stablehlo.multiply %1607, %1607 : tensor<256xf32>
    %1611 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1612 = stablehlo.multiply %1611, %1607 : tensor<256xf32>
    %1613 = stablehlo.subtract %1609, %1610 : tensor<256xf32>
    %1614 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1615 = stablehlo.maximum %1614, %1613 : tensor<256xf32>
    %1616 = stablehlo.compare  EQ, %1613, %1615,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1617 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1618 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1619 = stablehlo.select %1616, %1617, %1618 : tensor<256xi1>, tensor<256xf32>
    %1620 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1621 = stablehlo.compare  EQ, %1620, %1615,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1622 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1623 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1624 = stablehlo.select %1621, %1622, %1623 : tensor<256xi1>, tensor<256xf32>
    %1625 = stablehlo.divide %1619, %1624 : tensor<256xf32>
    %1626 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1627 = stablehlo.multiply %1626, %arg290 : tensor<256xf32>
    %1628 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1629 = stablehlo.multiply %1628, %1607 : tensor<256xf32>
    %1630 = stablehlo.add %1627, %1629 : tensor<256xf32>
    %1631 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1632 = stablehlo.multiply %1631, %arg291 : tensor<256xf32>
    %1633 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1634 = stablehlo.multiply %1633, %1615 : tensor<256xf32>
    %1635 = stablehlo.add %1632, %1634 : tensor<256xf32>
    %1636 = stablehlo.broadcast_in_dim %1607, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1637 = stablehlo.broadcast_in_dim %1615, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1638 = stablehlo.convert %1589 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1639 = stablehlo.broadcast_in_dim %1636, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1640 = stablehlo.subtract %1638, %1639 : tensor<256x14x14x256xf32>
    %1641 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1642 = stablehlo.add %1637, %1641 : tensor<1x1x1x256xf32>
    %1643 = stablehlo.rsqrt %1642 : tensor<1x1x1x256xf32>
    %1644 = stablehlo.divide %1643, %1642 : tensor<1x1x1x256xf32>
    %1645 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1646 = stablehlo.multiply %1645, %1644 : tensor<1x1x1x256xf32>
    %1647 = stablehlo.reshape %arg105 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1648 = stablehlo.multiply %1643, %1647 : tensor<1x1x1x256xf32>
    %1649 = stablehlo.broadcast_in_dim %1648, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1650 = stablehlo.multiply %1640, %1649 : tensor<256x14x14x256xf32>
    %1651 = stablehlo.reshape %arg104 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1652 = stablehlo.broadcast_in_dim %1651, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1653 = stablehlo.add %1650, %1652 : tensor<256x14x14x256xf32>
    %1654 = stablehlo.convert %1653 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1655 = stablehlo.add %1513, %1654 : tensor<256x14x14x256xbf16>
    %1656 = call @relu_139(%1655) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1657 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1658 = stablehlo.compare  GT, %1655, %1657,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1659 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1660 = stablehlo.convert %arg19 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1661 = stablehlo.convolution(%1656, %1660) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1662 = stablehlo.convert %1661 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1663 = stablehlo.multiply %1662, %1662 : tensor<256x14x14x256xf32>
    %1664 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1665 = stablehlo.multiply %1664, %1662 : tensor<256x14x14x256xf32>
    %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1666 = stablehlo.reduce(%1662 init: %cst_67) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1667 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1668 = stablehlo.divide %1666, %1667 : tensor<256xf32>
    %cst_68 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1669 = stablehlo.reduce(%1663 init: %cst_68) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1670 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1671 = stablehlo.divide %1669, %1670 : tensor<256xf32>
    %1672 = stablehlo.broadcast_in_dim %1668, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1673 = stablehlo.broadcast_in_dim %1671, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1674 = stablehlo.concatenate %1672, %1673, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1675 = "stablehlo.all_reduce"(%1674) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1676 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1677 = stablehlo.divide %1675, %1676 : tensor<2x256xf32>
    %1678 = stablehlo.slice %1677 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1679 = stablehlo.reshape %1678 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1680 = stablehlo.slice %1677 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1681 = stablehlo.reshape %1680 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1682 = stablehlo.multiply %1679, %1679 : tensor<256xf32>
    %1683 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1684 = stablehlo.multiply %1683, %1679 : tensor<256xf32>
    %1685 = stablehlo.subtract %1681, %1682 : tensor<256xf32>
    %1686 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1687 = stablehlo.maximum %1686, %1685 : tensor<256xf32>
    %1688 = stablehlo.compare  EQ, %1685, %1687,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1689 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1690 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1691 = stablehlo.select %1688, %1689, %1690 : tensor<256xi1>, tensor<256xf32>
    %1692 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1693 = stablehlo.compare  EQ, %1692, %1687,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1694 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1695 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1696 = stablehlo.select %1693, %1694, %1695 : tensor<256xi1>, tensor<256xf32>
    %1697 = stablehlo.divide %1691, %1696 : tensor<256xf32>
    %1698 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1699 = stablehlo.multiply %1698, %arg230 : tensor<256xf32>
    %1700 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1701 = stablehlo.multiply %1700, %1679 : tensor<256xf32>
    %1702 = stablehlo.add %1699, %1701 : tensor<256xf32>
    %1703 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1704 = stablehlo.multiply %1703, %arg231 : tensor<256xf32>
    %1705 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1706 = stablehlo.multiply %1705, %1687 : tensor<256xf32>
    %1707 = stablehlo.add %1704, %1706 : tensor<256xf32>
    %1708 = stablehlo.broadcast_in_dim %1679, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1709 = stablehlo.broadcast_in_dim %1687, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1710 = stablehlo.convert %1661 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1711 = stablehlo.broadcast_in_dim %1708, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1712 = stablehlo.subtract %1710, %1711 : tensor<256x14x14x256xf32>
    %1713 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1714 = stablehlo.add %1709, %1713 : tensor<1x1x1x256xf32>
    %1715 = stablehlo.rsqrt %1714 : tensor<1x1x1x256xf32>
    %1716 = stablehlo.divide %1715, %1714 : tensor<1x1x1x256xf32>
    %1717 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1718 = stablehlo.multiply %1717, %1716 : tensor<1x1x1x256xf32>
    %1719 = stablehlo.reshape %arg16 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1720 = stablehlo.multiply %1715, %1719 : tensor<1x1x1x256xf32>
    %1721 = stablehlo.broadcast_in_dim %1720, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1722 = stablehlo.multiply %1712, %1721 : tensor<256x14x14x256xf32>
    %1723 = stablehlo.reshape %arg15 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1724 = stablehlo.broadcast_in_dim %1723, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1725 = stablehlo.add %1722, %1724 : tensor<256x14x14x256xf32>
    %1726 = stablehlo.convert %1725 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1727 = call @relu_139(%1726) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1728 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1729 = stablehlo.compare  GT, %1726, %1728,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1730 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1731 = stablehlo.convert %arg20 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1732 = stablehlo.convolution(%1727, %1731) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1733 = stablehlo.convert %1732 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1734 = stablehlo.multiply %1733, %1733 : tensor<256x14x14x256xf32>
    %1735 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1736 = stablehlo.multiply %1735, %1733 : tensor<256x14x14x256xf32>
    %cst_69 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1737 = stablehlo.reduce(%1733 init: %cst_69) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1738 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1739 = stablehlo.divide %1737, %1738 : tensor<256xf32>
    %cst_70 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1740 = stablehlo.reduce(%1734 init: %cst_70) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1741 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1742 = stablehlo.divide %1740, %1741 : tensor<256xf32>
    %1743 = stablehlo.broadcast_in_dim %1739, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1744 = stablehlo.broadcast_in_dim %1742, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1745 = stablehlo.concatenate %1743, %1744, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1746 = "stablehlo.all_reduce"(%1745) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1747 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1748 = stablehlo.divide %1746, %1747 : tensor<2x256xf32>
    %1749 = stablehlo.slice %1748 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1750 = stablehlo.reshape %1749 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1751 = stablehlo.slice %1748 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1752 = stablehlo.reshape %1751 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1753 = stablehlo.multiply %1750, %1750 : tensor<256xf32>
    %1754 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1755 = stablehlo.multiply %1754, %1750 : tensor<256xf32>
    %1756 = stablehlo.subtract %1752, %1753 : tensor<256xf32>
    %1757 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1758 = stablehlo.maximum %1757, %1756 : tensor<256xf32>
    %1759 = stablehlo.compare  EQ, %1756, %1758,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1760 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1761 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1762 = stablehlo.select %1759, %1760, %1761 : tensor<256xi1>, tensor<256xf32>
    %1763 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1764 = stablehlo.compare  EQ, %1763, %1758,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1765 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1766 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1767 = stablehlo.select %1764, %1765, %1766 : tensor<256xi1>, tensor<256xf32>
    %1768 = stablehlo.divide %1762, %1767 : tensor<256xf32>
    %1769 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1770 = stablehlo.multiply %1769, %arg232 : tensor<256xf32>
    %1771 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1772 = stablehlo.multiply %1771, %1750 : tensor<256xf32>
    %1773 = stablehlo.add %1770, %1772 : tensor<256xf32>
    %1774 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1775 = stablehlo.multiply %1774, %arg233 : tensor<256xf32>
    %1776 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1777 = stablehlo.multiply %1776, %1758 : tensor<256xf32>
    %1778 = stablehlo.add %1775, %1777 : tensor<256xf32>
    %1779 = stablehlo.broadcast_in_dim %1750, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1780 = stablehlo.broadcast_in_dim %1758, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1781 = stablehlo.convert %1732 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1782 = stablehlo.broadcast_in_dim %1779, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1783 = stablehlo.subtract %1781, %1782 : tensor<256x14x14x256xf32>
    %1784 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1785 = stablehlo.add %1780, %1784 : tensor<1x1x1x256xf32>
    %1786 = stablehlo.rsqrt %1785 : tensor<1x1x1x256xf32>
    %1787 = stablehlo.divide %1786, %1785 : tensor<1x1x1x256xf32>
    %1788 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1789 = stablehlo.multiply %1788, %1787 : tensor<1x1x1x256xf32>
    %1790 = stablehlo.reshape %arg18 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1791 = stablehlo.multiply %1786, %1790 : tensor<1x1x1x256xf32>
    %1792 = stablehlo.broadcast_in_dim %1791, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1793 = stablehlo.multiply %1783, %1792 : tensor<256x14x14x256xf32>
    %1794 = stablehlo.reshape %arg17 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1795 = stablehlo.broadcast_in_dim %1794, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1796 = stablehlo.add %1793, %1795 : tensor<256x14x14x256xf32>
    %1797 = stablehlo.convert %1796 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1798 = stablehlo.add %1656, %1797 : tensor<256x14x14x256xbf16>
    %1799 = call @relu_139(%1798) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1800 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1801 = stablehlo.compare  GT, %1798, %1800,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1802 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1803 = stablehlo.convert %arg25 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1804 = stablehlo.convolution(%1799, %1803) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1805 = stablehlo.convert %1804 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1806 = stablehlo.multiply %1805, %1805 : tensor<256x14x14x256xf32>
    %1807 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1808 = stablehlo.multiply %1807, %1805 : tensor<256x14x14x256xf32>
    %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1809 = stablehlo.reduce(%1805 init: %cst_71) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1810 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1811 = stablehlo.divide %1809, %1810 : tensor<256xf32>
    %cst_72 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1812 = stablehlo.reduce(%1806 init: %cst_72) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1813 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1814 = stablehlo.divide %1812, %1813 : tensor<256xf32>
    %1815 = stablehlo.broadcast_in_dim %1811, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1816 = stablehlo.broadcast_in_dim %1814, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1817 = stablehlo.concatenate %1815, %1816, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1818 = "stablehlo.all_reduce"(%1817) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1819 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1820 = stablehlo.divide %1818, %1819 : tensor<2x256xf32>
    %1821 = stablehlo.slice %1820 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1822 = stablehlo.reshape %1821 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1823 = stablehlo.slice %1820 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1824 = stablehlo.reshape %1823 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1825 = stablehlo.multiply %1822, %1822 : tensor<256xf32>
    %1826 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1827 = stablehlo.multiply %1826, %1822 : tensor<256xf32>
    %1828 = stablehlo.subtract %1824, %1825 : tensor<256xf32>
    %1829 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1830 = stablehlo.maximum %1829, %1828 : tensor<256xf32>
    %1831 = stablehlo.compare  EQ, %1828, %1830,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1832 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1833 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1834 = stablehlo.select %1831, %1832, %1833 : tensor<256xi1>, tensor<256xf32>
    %1835 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1836 = stablehlo.compare  EQ, %1835, %1830,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1837 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1838 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1839 = stablehlo.select %1836, %1837, %1838 : tensor<256xi1>, tensor<256xf32>
    %1840 = stablehlo.divide %1834, %1839 : tensor<256xf32>
    %1841 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1842 = stablehlo.multiply %1841, %arg234 : tensor<256xf32>
    %1843 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1844 = stablehlo.multiply %1843, %1822 : tensor<256xf32>
    %1845 = stablehlo.add %1842, %1844 : tensor<256xf32>
    %1846 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1847 = stablehlo.multiply %1846, %arg235 : tensor<256xf32>
    %1848 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1849 = stablehlo.multiply %1848, %1830 : tensor<256xf32>
    %1850 = stablehlo.add %1847, %1849 : tensor<256xf32>
    %1851 = stablehlo.broadcast_in_dim %1822, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1852 = stablehlo.broadcast_in_dim %1830, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1853 = stablehlo.convert %1804 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1854 = stablehlo.broadcast_in_dim %1851, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1855 = stablehlo.subtract %1853, %1854 : tensor<256x14x14x256xf32>
    %1856 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1857 = stablehlo.add %1852, %1856 : tensor<1x1x1x256xf32>
    %1858 = stablehlo.rsqrt %1857 : tensor<1x1x1x256xf32>
    %1859 = stablehlo.divide %1858, %1857 : tensor<1x1x1x256xf32>
    %1860 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1861 = stablehlo.multiply %1860, %1859 : tensor<1x1x1x256xf32>
    %1862 = stablehlo.reshape %arg22 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1863 = stablehlo.multiply %1858, %1862 : tensor<1x1x1x256xf32>
    %1864 = stablehlo.broadcast_in_dim %1863, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1865 = stablehlo.multiply %1855, %1864 : tensor<256x14x14x256xf32>
    %1866 = stablehlo.reshape %arg21 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1867 = stablehlo.broadcast_in_dim %1866, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1868 = stablehlo.add %1865, %1867 : tensor<256x14x14x256xf32>
    %1869 = stablehlo.convert %1868 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1870 = call @relu_139(%1869) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1871 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1872 = stablehlo.compare  GT, %1869, %1871,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1873 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1874 = stablehlo.convert %arg26 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1875 = stablehlo.convolution(%1870, %1874) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1876 = stablehlo.convert %1875 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1877 = stablehlo.multiply %1876, %1876 : tensor<256x14x14x256xf32>
    %1878 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1879 = stablehlo.multiply %1878, %1876 : tensor<256x14x14x256xf32>
    %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1880 = stablehlo.reduce(%1876 init: %cst_73) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1881 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1882 = stablehlo.divide %1880, %1881 : tensor<256xf32>
    %cst_74 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1883 = stablehlo.reduce(%1877 init: %cst_74) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1884 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1885 = stablehlo.divide %1883, %1884 : tensor<256xf32>
    %1886 = stablehlo.broadcast_in_dim %1882, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1887 = stablehlo.broadcast_in_dim %1885, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1888 = stablehlo.concatenate %1886, %1887, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1889 = "stablehlo.all_reduce"(%1888) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1890 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1891 = stablehlo.divide %1889, %1890 : tensor<2x256xf32>
    %1892 = stablehlo.slice %1891 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1893 = stablehlo.reshape %1892 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1894 = stablehlo.slice %1891 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1895 = stablehlo.reshape %1894 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1896 = stablehlo.multiply %1893, %1893 : tensor<256xf32>
    %1897 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1898 = stablehlo.multiply %1897, %1893 : tensor<256xf32>
    %1899 = stablehlo.subtract %1895, %1896 : tensor<256xf32>
    %1900 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1901 = stablehlo.maximum %1900, %1899 : tensor<256xf32>
    %1902 = stablehlo.compare  EQ, %1899, %1901,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1903 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1904 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1905 = stablehlo.select %1902, %1903, %1904 : tensor<256xi1>, tensor<256xf32>
    %1906 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1907 = stablehlo.compare  EQ, %1906, %1901,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1908 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1909 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1910 = stablehlo.select %1907, %1908, %1909 : tensor<256xi1>, tensor<256xf32>
    %1911 = stablehlo.divide %1905, %1910 : tensor<256xf32>
    %1912 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1913 = stablehlo.multiply %1912, %arg236 : tensor<256xf32>
    %1914 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1915 = stablehlo.multiply %1914, %1893 : tensor<256xf32>
    %1916 = stablehlo.add %1913, %1915 : tensor<256xf32>
    %1917 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1918 = stablehlo.multiply %1917, %arg237 : tensor<256xf32>
    %1919 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1920 = stablehlo.multiply %1919, %1901 : tensor<256xf32>
    %1921 = stablehlo.add %1918, %1920 : tensor<256xf32>
    %1922 = stablehlo.broadcast_in_dim %1893, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1923 = stablehlo.broadcast_in_dim %1901, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1924 = stablehlo.convert %1875 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1925 = stablehlo.broadcast_in_dim %1922, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1926 = stablehlo.subtract %1924, %1925 : tensor<256x14x14x256xf32>
    %1927 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1928 = stablehlo.add %1923, %1927 : tensor<1x1x1x256xf32>
    %1929 = stablehlo.rsqrt %1928 : tensor<1x1x1x256xf32>
    %1930 = stablehlo.divide %1929, %1928 : tensor<1x1x1x256xf32>
    %1931 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %1932 = stablehlo.multiply %1931, %1930 : tensor<1x1x1x256xf32>
    %1933 = stablehlo.reshape %arg24 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1934 = stablehlo.multiply %1929, %1933 : tensor<1x1x1x256xf32>
    %1935 = stablehlo.broadcast_in_dim %1934, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1936 = stablehlo.multiply %1926, %1935 : tensor<256x14x14x256xf32>
    %1937 = stablehlo.reshape %arg23 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1938 = stablehlo.broadcast_in_dim %1937, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1939 = stablehlo.add %1936, %1938 : tensor<256x14x14x256xf32>
    %1940 = stablehlo.convert %1939 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %1941 = stablehlo.add %1799, %1940 : tensor<256x14x14x256xbf16>
    %1942 = call @relu_139(%1941) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1943 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1944 = stablehlo.compare  GT, %1941, %1943,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %1945 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1946 = stablehlo.convert %arg31 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %1947 = stablehlo.convolution(%1942, %1946) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %1948 = stablehlo.convert %1947 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1949 = stablehlo.multiply %1948, %1948 : tensor<256x14x14x256xf32>
    %1950 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %1951 = stablehlo.multiply %1950, %1948 : tensor<256x14x14x256xf32>
    %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1952 = stablehlo.reduce(%1948 init: %cst_75) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1953 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1954 = stablehlo.divide %1952, %1953 : tensor<256xf32>
    %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1955 = stablehlo.reduce(%1949 init: %cst_76) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %1956 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1957 = stablehlo.divide %1955, %1956 : tensor<256xf32>
    %1958 = stablehlo.broadcast_in_dim %1954, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1959 = stablehlo.broadcast_in_dim %1957, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1960 = stablehlo.concatenate %1958, %1959, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %1961 = "stablehlo.all_reduce"(%1960) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %1962 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1963 = stablehlo.divide %1961, %1962 : tensor<2x256xf32>
    %1964 = stablehlo.slice %1963 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1965 = stablehlo.reshape %1964 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1966 = stablehlo.slice %1963 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %1967 = stablehlo.reshape %1966 : (tensor<1x256xf32>) -> tensor<256xf32>
    %1968 = stablehlo.multiply %1965, %1965 : tensor<256xf32>
    %1969 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1970 = stablehlo.multiply %1969, %1965 : tensor<256xf32>
    %1971 = stablehlo.subtract %1967, %1968 : tensor<256xf32>
    %1972 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1973 = stablehlo.maximum %1972, %1971 : tensor<256xf32>
    %1974 = stablehlo.compare  EQ, %1971, %1973,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1975 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1976 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1977 = stablehlo.select %1974, %1975, %1976 : tensor<256xi1>, tensor<256xf32>
    %1978 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1979 = stablehlo.compare  EQ, %1978, %1973,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %1980 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1981 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1982 = stablehlo.select %1979, %1980, %1981 : tensor<256xi1>, tensor<256xf32>
    %1983 = stablehlo.divide %1977, %1982 : tensor<256xf32>
    %1984 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1985 = stablehlo.multiply %1984, %arg238 : tensor<256xf32>
    %1986 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1987 = stablehlo.multiply %1986, %1965 : tensor<256xf32>
    %1988 = stablehlo.add %1985, %1987 : tensor<256xf32>
    %1989 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1990 = stablehlo.multiply %1989, %arg239 : tensor<256xf32>
    %1991 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1992 = stablehlo.multiply %1991, %1973 : tensor<256xf32>
    %1993 = stablehlo.add %1990, %1992 : tensor<256xf32>
    %1994 = stablehlo.broadcast_in_dim %1965, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1995 = stablehlo.broadcast_in_dim %1973, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %1996 = stablehlo.convert %1947 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %1997 = stablehlo.broadcast_in_dim %1994, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %1998 = stablehlo.subtract %1996, %1997 : tensor<256x14x14x256xf32>
    %1999 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2000 = stablehlo.add %1995, %1999 : tensor<1x1x1x256xf32>
    %2001 = stablehlo.rsqrt %2000 : tensor<1x1x1x256xf32>
    %2002 = stablehlo.divide %2001, %2000 : tensor<1x1x1x256xf32>
    %2003 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2004 = stablehlo.multiply %2003, %2002 : tensor<1x1x1x256xf32>
    %2005 = stablehlo.reshape %arg28 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2006 = stablehlo.multiply %2001, %2005 : tensor<1x1x1x256xf32>
    %2007 = stablehlo.broadcast_in_dim %2006, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2008 = stablehlo.multiply %1998, %2007 : tensor<256x14x14x256xf32>
    %2009 = stablehlo.reshape %arg27 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2010 = stablehlo.broadcast_in_dim %2009, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2011 = stablehlo.add %2008, %2010 : tensor<256x14x14x256xf32>
    %2012 = stablehlo.convert %2011 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2013 = call @relu_139(%2012) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2014 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %2015 = stablehlo.compare  GT, %2012, %2014,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %2016 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %2017 = stablehlo.convert %arg32 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xbf16>
    %2018 = stablehlo.convolution(%2013, %2017) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2019 = stablehlo.convert %2018 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2020 = stablehlo.multiply %2019, %2019 : tensor<256x14x14x256xf32>
    %2021 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
    %2022 = stablehlo.multiply %2021, %2019 : tensor<256x14x14x256xf32>
    %cst_77 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2023 = stablehlo.reduce(%2019 init: %cst_77) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2024 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2025 = stablehlo.divide %2023, %2024 : tensor<256xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2026 = stablehlo.reduce(%2020 init: %cst_78) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %2027 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2028 = stablehlo.divide %2026, %2027 : tensor<256xf32>
    %2029 = stablehlo.broadcast_in_dim %2025, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2030 = stablehlo.broadcast_in_dim %2028, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2031 = stablehlo.concatenate %2029, %2030, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    %2032 = "stablehlo.all_reduce"(%2031) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %2033 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %2034 = stablehlo.divide %2032, %2033 : tensor<2x256xf32>
    %2035 = stablehlo.slice %2034 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %2036 = stablehlo.reshape %2035 : (tensor<1x256xf32>) -> tensor<256xf32>
    %2037 = stablehlo.slice %2034 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %2038 = stablehlo.reshape %2037 : (tensor<1x256xf32>) -> tensor<256xf32>
    %2039 = stablehlo.multiply %2036, %2036 : tensor<256xf32>
    %2040 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2041 = stablehlo.multiply %2040, %2036 : tensor<256xf32>
    %2042 = stablehlo.subtract %2038, %2039 : tensor<256xf32>
    %2043 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2044 = stablehlo.maximum %2043, %2042 : tensor<256xf32>
    %2045 = stablehlo.compare  EQ, %2042, %2044,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %2046 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2047 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2048 = stablehlo.select %2045, %2046, %2047 : tensor<256xi1>, tensor<256xf32>
    %2049 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2050 = stablehlo.compare  EQ, %2049, %2044,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %2051 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2052 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2053 = stablehlo.select %2050, %2051, %2052 : tensor<256xi1>, tensor<256xf32>
    %2054 = stablehlo.divide %2048, %2053 : tensor<256xf32>
    %2055 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2056 = stablehlo.multiply %2055, %arg240 : tensor<256xf32>
    %2057 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2058 = stablehlo.multiply %2057, %2036 : tensor<256xf32>
    %2059 = stablehlo.add %2056, %2058 : tensor<256xf32>
    %2060 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2061 = stablehlo.multiply %2060, %arg241 : tensor<256xf32>
    %2062 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2063 = stablehlo.multiply %2062, %2044 : tensor<256xf32>
    %2064 = stablehlo.add %2061, %2063 : tensor<256xf32>
    %2065 = stablehlo.broadcast_in_dim %2036, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2066 = stablehlo.broadcast_in_dim %2044, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2067 = stablehlo.convert %2018 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %2068 = stablehlo.broadcast_in_dim %2065, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2069 = stablehlo.subtract %2067, %2068 : tensor<256x14x14x256xf32>
    %2070 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2071 = stablehlo.add %2066, %2070 : tensor<1x1x1x256xf32>
    %2072 = stablehlo.rsqrt %2071 : tensor<1x1x1x256xf32>
    %2073 = stablehlo.divide %2072, %2071 : tensor<1x1x1x256xf32>
    %2074 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %2075 = stablehlo.multiply %2074, %2073 : tensor<1x1x1x256xf32>
    %2076 = stablehlo.reshape %arg30 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2077 = stablehlo.multiply %2072, %2076 : tensor<1x1x1x256xf32>
    %2078 = stablehlo.broadcast_in_dim %2077, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2079 = stablehlo.multiply %2069, %2078 : tensor<256x14x14x256xf32>
    %2080 = stablehlo.reshape %arg29 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %2081 = stablehlo.broadcast_in_dim %2080, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %2082 = stablehlo.add %2079, %2081 : tensor<256x14x14x256xf32>
    %2083 = stablehlo.convert %2082 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %2084 = stablehlo.add %1942, %2083 : tensor<256x14x14x256xbf16>
    %2085 = call @relu_139(%2084) : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16>
    %2086 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %2087 = stablehlo.compare  GT, %2084, %2086,  FLOAT : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xi1>
    %2088 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %2089 = stablehlo.convert %arg37 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xbf16>
    %2090 = stablehlo.convolution(%2085, %2089) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2091 = stablehlo.convert %2090 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2092 = stablehlo.multiply %2091, %2091 : tensor<256x7x7x512xf32>
    %2093 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %2094 = stablehlo.multiply %2093, %2091 : tensor<256x7x7x512xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2095 = stablehlo.reduce(%2091 init: %cst_79) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_80 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
    %2096 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2097 = stablehlo.divide %2095, %2096 : tensor<512xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2098 = stablehlo.reduce(%2092 init: %cst_81) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2099 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2100 = stablehlo.divide %2098, %2099 : tensor<512xf32>
    %2101 = stablehlo.broadcast_in_dim %2097, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2102 = stablehlo.broadcast_in_dim %2100, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2103 = stablehlo.concatenate %2101, %2102, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %2104 = "stablehlo.all_reduce"(%2103) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %2105 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %2106 = stablehlo.divide %2104, %2105 : tensor<2x512xf32>
    %2107 = stablehlo.slice %2106 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2108 = stablehlo.reshape %2107 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2109 = stablehlo.slice %2106 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2110 = stablehlo.reshape %2109 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2111 = stablehlo.multiply %2108, %2108 : tensor<512xf32>
    %2112 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2113 = stablehlo.multiply %2112, %2108 : tensor<512xf32>
    %2114 = stablehlo.subtract %2110, %2111 : tensor<512xf32>
    %2115 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2116 = stablehlo.maximum %2115, %2114 : tensor<512xf32>
    %2117 = stablehlo.compare  EQ, %2114, %2116,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2118 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2119 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2120 = stablehlo.select %2117, %2118, %2119 : tensor<512xi1>, tensor<512xf32>
    %2121 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2122 = stablehlo.compare  EQ, %2121, %2116,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2123 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2124 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2125 = stablehlo.select %2122, %2123, %2124 : tensor<512xi1>, tensor<512xf32>
    %2126 = stablehlo.divide %2120, %2125 : tensor<512xf32>
    %2127 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2128 = stablehlo.multiply %2127, %arg242 : tensor<512xf32>
    %2129 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2130 = stablehlo.multiply %2129, %2108 : tensor<512xf32>
    %2131 = stablehlo.add %2128, %2130 : tensor<512xf32>
    %2132 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2133 = stablehlo.multiply %2132, %arg243 : tensor<512xf32>
    %2134 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2135 = stablehlo.multiply %2134, %2116 : tensor<512xf32>
    %2136 = stablehlo.add %2133, %2135 : tensor<512xf32>
    %2137 = stablehlo.broadcast_in_dim %2108, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2138 = stablehlo.broadcast_in_dim %2116, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2139 = stablehlo.convert %2090 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2140 = stablehlo.broadcast_in_dim %2137, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2141 = stablehlo.subtract %2139, %2140 : tensor<256x7x7x512xf32>
    %2142 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2143 = stablehlo.add %2138, %2142 : tensor<1x1x1x512xf32>
    %2144 = stablehlo.rsqrt %2143 : tensor<1x1x1x512xf32>
    %2145 = stablehlo.divide %2144, %2143 : tensor<1x1x1x512xf32>
    %2146 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2147 = stablehlo.multiply %2146, %2145 : tensor<1x1x1x512xf32>
    %2148 = stablehlo.reshape %arg34 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2149 = stablehlo.multiply %2144, %2148 : tensor<1x1x1x512xf32>
    %2150 = stablehlo.broadcast_in_dim %2149, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2151 = stablehlo.multiply %2141, %2150 : tensor<256x7x7x512xf32>
    %2152 = stablehlo.reshape %arg33 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2153 = stablehlo.broadcast_in_dim %2152, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2154 = stablehlo.add %2151, %2153 : tensor<256x7x7x512xf32>
    %2155 = stablehlo.convert %2154 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2156 = call @relu_195(%2155) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2157 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2158 = stablehlo.compare  GT, %2155, %2157,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %2159 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2160 = stablehlo.convert %arg38 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %2161 = stablehlo.convolution(%2156, %2160) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2162 = stablehlo.convert %2161 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2163 = stablehlo.multiply %2162, %2162 : tensor<256x7x7x512xf32>
    %2164 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %2165 = stablehlo.multiply %2164, %2162 : tensor<256x7x7x512xf32>
    %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2166 = stablehlo.reduce(%2162 init: %cst_82) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2167 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2168 = stablehlo.divide %2166, %2167 : tensor<512xf32>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2169 = stablehlo.reduce(%2163 init: %cst_83) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2170 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2171 = stablehlo.divide %2169, %2170 : tensor<512xf32>
    %2172 = stablehlo.broadcast_in_dim %2168, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2173 = stablehlo.broadcast_in_dim %2171, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2174 = stablehlo.concatenate %2172, %2173, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %2175 = "stablehlo.all_reduce"(%2174) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %2176 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %2177 = stablehlo.divide %2175, %2176 : tensor<2x512xf32>
    %2178 = stablehlo.slice %2177 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2179 = stablehlo.reshape %2178 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2180 = stablehlo.slice %2177 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2181 = stablehlo.reshape %2180 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2182 = stablehlo.multiply %2179, %2179 : tensor<512xf32>
    %2183 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2184 = stablehlo.multiply %2183, %2179 : tensor<512xf32>
    %2185 = stablehlo.subtract %2181, %2182 : tensor<512xf32>
    %2186 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2187 = stablehlo.maximum %2186, %2185 : tensor<512xf32>
    %2188 = stablehlo.compare  EQ, %2185, %2187,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2189 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2190 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2191 = stablehlo.select %2188, %2189, %2190 : tensor<512xi1>, tensor<512xf32>
    %2192 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2193 = stablehlo.compare  EQ, %2192, %2187,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2194 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2195 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2196 = stablehlo.select %2193, %2194, %2195 : tensor<512xi1>, tensor<512xf32>
    %2197 = stablehlo.divide %2191, %2196 : tensor<512xf32>
    %2198 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2199 = stablehlo.multiply %2198, %arg244 : tensor<512xf32>
    %2200 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2201 = stablehlo.multiply %2200, %2179 : tensor<512xf32>
    %2202 = stablehlo.add %2199, %2201 : tensor<512xf32>
    %2203 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2204 = stablehlo.multiply %2203, %arg245 : tensor<512xf32>
    %2205 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2206 = stablehlo.multiply %2205, %2187 : tensor<512xf32>
    %2207 = stablehlo.add %2204, %2206 : tensor<512xf32>
    %2208 = stablehlo.broadcast_in_dim %2179, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2209 = stablehlo.broadcast_in_dim %2187, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2210 = stablehlo.convert %2161 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2211 = stablehlo.broadcast_in_dim %2208, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2212 = stablehlo.subtract %2210, %2211 : tensor<256x7x7x512xf32>
    %2213 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2214 = stablehlo.add %2209, %2213 : tensor<1x1x1x512xf32>
    %2215 = stablehlo.rsqrt %2214 : tensor<1x1x1x512xf32>
    %2216 = stablehlo.divide %2215, %2214 : tensor<1x1x1x512xf32>
    %2217 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2218 = stablehlo.multiply %2217, %2216 : tensor<1x1x1x512xf32>
    %2219 = stablehlo.reshape %arg36 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2220 = stablehlo.multiply %2215, %2219 : tensor<1x1x1x512xf32>
    %2221 = stablehlo.broadcast_in_dim %2220, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2222 = stablehlo.multiply %2212, %2221 : tensor<256x7x7x512xf32>
    %2223 = stablehlo.reshape %arg35 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2224 = stablehlo.broadcast_in_dim %2223, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2225 = stablehlo.add %2222, %2224 : tensor<256x7x7x512xf32>
    %2226 = stablehlo.convert %2225 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2227 = stablehlo.convert %arg39 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xbf16>
    %2228 = stablehlo.convolution(%2085, %2227) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x256x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2229 = stablehlo.convert %2228 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2230 = stablehlo.multiply %2229, %2229 : tensor<256x7x7x512xf32>
    %2231 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %2232 = stablehlo.multiply %2231, %2229 : tensor<256x7x7x512xf32>
    %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2233 = stablehlo.reduce(%2229 init: %cst_84) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2234 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2235 = stablehlo.divide %2233, %2234 : tensor<512xf32>
    %cst_85 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2236 = stablehlo.reduce(%2230 init: %cst_85) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2237 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2238 = stablehlo.divide %2236, %2237 : tensor<512xf32>
    %2239 = stablehlo.broadcast_in_dim %2235, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2240 = stablehlo.broadcast_in_dim %2238, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2241 = stablehlo.concatenate %2239, %2240, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %2242 = "stablehlo.all_reduce"(%2241) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %2243 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %2244 = stablehlo.divide %2242, %2243 : tensor<2x512xf32>
    %2245 = stablehlo.slice %2244 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2246 = stablehlo.reshape %2245 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2247 = stablehlo.slice %2244 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2248 = stablehlo.reshape %2247 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2249 = stablehlo.multiply %2246, %2246 : tensor<512xf32>
    %2250 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2251 = stablehlo.multiply %2250, %2246 : tensor<512xf32>
    %2252 = stablehlo.subtract %2248, %2249 : tensor<512xf32>
    %2253 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2254 = stablehlo.maximum %2253, %2252 : tensor<512xf32>
    %2255 = stablehlo.compare  EQ, %2252, %2254,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2256 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2257 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2258 = stablehlo.select %2255, %2256, %2257 : tensor<512xi1>, tensor<512xf32>
    %2259 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2260 = stablehlo.compare  EQ, %2259, %2254,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2261 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2262 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2263 = stablehlo.select %2260, %2261, %2262 : tensor<512xi1>, tensor<512xf32>
    %2264 = stablehlo.divide %2258, %2263 : tensor<512xf32>
    %2265 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2266 = stablehlo.multiply %2265, %arg246 : tensor<512xf32>
    %2267 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2268 = stablehlo.multiply %2267, %2246 : tensor<512xf32>
    %2269 = stablehlo.add %2266, %2268 : tensor<512xf32>
    %2270 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2271 = stablehlo.multiply %2270, %arg247 : tensor<512xf32>
    %2272 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2273 = stablehlo.multiply %2272, %2254 : tensor<512xf32>
    %2274 = stablehlo.add %2271, %2273 : tensor<512xf32>
    %2275 = stablehlo.broadcast_in_dim %2246, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2276 = stablehlo.broadcast_in_dim %2254, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2277 = stablehlo.convert %2228 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2278 = stablehlo.broadcast_in_dim %2275, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2279 = stablehlo.subtract %2277, %2278 : tensor<256x7x7x512xf32>
    %2280 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2281 = stablehlo.add %2276, %2280 : tensor<1x1x1x512xf32>
    %2282 = stablehlo.rsqrt %2281 : tensor<1x1x1x512xf32>
    %2283 = stablehlo.divide %2282, %2281 : tensor<1x1x1x512xf32>
    %2284 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2285 = stablehlo.multiply %2284, %2283 : tensor<1x1x1x512xf32>
    %2286 = stablehlo.reshape %arg41 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2287 = stablehlo.multiply %2282, %2286 : tensor<1x1x1x512xf32>
    %2288 = stablehlo.broadcast_in_dim %2287, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2289 = stablehlo.multiply %2279, %2288 : tensor<256x7x7x512xf32>
    %2290 = stablehlo.reshape %arg40 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2291 = stablehlo.broadcast_in_dim %2290, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2292 = stablehlo.add %2289, %2291 : tensor<256x7x7x512xf32>
    %2293 = stablehlo.convert %2292 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2294 = stablehlo.add %2293, %2226 : tensor<256x7x7x512xbf16>
    %2295 = call @relu_195(%2294) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2296 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2297 = stablehlo.compare  GT, %2294, %2296,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %2298 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2299 = stablehlo.convert %arg46 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %2300 = stablehlo.convolution(%2295, %2299) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2301 = stablehlo.convert %2300 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2302 = stablehlo.multiply %2301, %2301 : tensor<256x7x7x512xf32>
    %2303 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %2304 = stablehlo.multiply %2303, %2301 : tensor<256x7x7x512xf32>
    %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2305 = stablehlo.reduce(%2301 init: %cst_86) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2306 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2307 = stablehlo.divide %2305, %2306 : tensor<512xf32>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2308 = stablehlo.reduce(%2302 init: %cst_87) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2309 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2310 = stablehlo.divide %2308, %2309 : tensor<512xf32>
    %2311 = stablehlo.broadcast_in_dim %2307, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2312 = stablehlo.broadcast_in_dim %2310, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2313 = stablehlo.concatenate %2311, %2312, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %2314 = "stablehlo.all_reduce"(%2313) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %2315 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %2316 = stablehlo.divide %2314, %2315 : tensor<2x512xf32>
    %2317 = stablehlo.slice %2316 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2318 = stablehlo.reshape %2317 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2319 = stablehlo.slice %2316 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2320 = stablehlo.reshape %2319 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2321 = stablehlo.multiply %2318, %2318 : tensor<512xf32>
    %2322 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2323 = stablehlo.multiply %2322, %2318 : tensor<512xf32>
    %2324 = stablehlo.subtract %2320, %2321 : tensor<512xf32>
    %2325 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2326 = stablehlo.maximum %2325, %2324 : tensor<512xf32>
    %2327 = stablehlo.compare  EQ, %2324, %2326,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2328 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2329 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2330 = stablehlo.select %2327, %2328, %2329 : tensor<512xi1>, tensor<512xf32>
    %2331 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2332 = stablehlo.compare  EQ, %2331, %2326,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2333 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2334 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2335 = stablehlo.select %2332, %2333, %2334 : tensor<512xi1>, tensor<512xf32>
    %2336 = stablehlo.divide %2330, %2335 : tensor<512xf32>
    %2337 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2338 = stablehlo.multiply %2337, %arg248 : tensor<512xf32>
    %2339 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2340 = stablehlo.multiply %2339, %2318 : tensor<512xf32>
    %2341 = stablehlo.add %2338, %2340 : tensor<512xf32>
    %2342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2343 = stablehlo.multiply %2342, %arg249 : tensor<512xf32>
    %2344 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2345 = stablehlo.multiply %2344, %2326 : tensor<512xf32>
    %2346 = stablehlo.add %2343, %2345 : tensor<512xf32>
    %2347 = stablehlo.broadcast_in_dim %2318, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2348 = stablehlo.broadcast_in_dim %2326, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2349 = stablehlo.convert %2300 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2350 = stablehlo.broadcast_in_dim %2347, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2351 = stablehlo.subtract %2349, %2350 : tensor<256x7x7x512xf32>
    %2352 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2353 = stablehlo.add %2348, %2352 : tensor<1x1x1x512xf32>
    %2354 = stablehlo.rsqrt %2353 : tensor<1x1x1x512xf32>
    %2355 = stablehlo.divide %2354, %2353 : tensor<1x1x1x512xf32>
    %2356 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2357 = stablehlo.multiply %2356, %2355 : tensor<1x1x1x512xf32>
    %2358 = stablehlo.reshape %arg43 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2359 = stablehlo.multiply %2354, %2358 : tensor<1x1x1x512xf32>
    %2360 = stablehlo.broadcast_in_dim %2359, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2361 = stablehlo.multiply %2351, %2360 : tensor<256x7x7x512xf32>
    %2362 = stablehlo.reshape %arg42 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2363 = stablehlo.broadcast_in_dim %2362, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2364 = stablehlo.add %2361, %2363 : tensor<256x7x7x512xf32>
    %2365 = stablehlo.convert %2364 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2366 = call @relu_195(%2365) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2367 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2368 = stablehlo.compare  GT, %2365, %2367,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %2369 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2370 = stablehlo.convert %arg47 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %2371 = stablehlo.convolution(%2366, %2370) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2372 = stablehlo.convert %2371 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2373 = stablehlo.multiply %2372, %2372 : tensor<256x7x7x512xf32>
    %2374 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %2375 = stablehlo.multiply %2374, %2372 : tensor<256x7x7x512xf32>
    %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2376 = stablehlo.reduce(%2372 init: %cst_88) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2377 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2378 = stablehlo.divide %2376, %2377 : tensor<512xf32>
    %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2379 = stablehlo.reduce(%2373 init: %cst_89) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2380 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2381 = stablehlo.divide %2379, %2380 : tensor<512xf32>
    %2382 = stablehlo.broadcast_in_dim %2378, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2383 = stablehlo.broadcast_in_dim %2381, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2384 = stablehlo.concatenate %2382, %2383, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %2385 = "stablehlo.all_reduce"(%2384) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %2386 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %2387 = stablehlo.divide %2385, %2386 : tensor<2x512xf32>
    %2388 = stablehlo.slice %2387 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2389 = stablehlo.reshape %2388 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2390 = stablehlo.slice %2387 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2391 = stablehlo.reshape %2390 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2392 = stablehlo.multiply %2389, %2389 : tensor<512xf32>
    %2393 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2394 = stablehlo.multiply %2393, %2389 : tensor<512xf32>
    %2395 = stablehlo.subtract %2391, %2392 : tensor<512xf32>
    %2396 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2397 = stablehlo.maximum %2396, %2395 : tensor<512xf32>
    %2398 = stablehlo.compare  EQ, %2395, %2397,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2399 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2400 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2401 = stablehlo.select %2398, %2399, %2400 : tensor<512xi1>, tensor<512xf32>
    %2402 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2403 = stablehlo.compare  EQ, %2402, %2397,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2404 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2405 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2406 = stablehlo.select %2403, %2404, %2405 : tensor<512xi1>, tensor<512xf32>
    %2407 = stablehlo.divide %2401, %2406 : tensor<512xf32>
    %2408 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2409 = stablehlo.multiply %2408, %arg250 : tensor<512xf32>
    %2410 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2411 = stablehlo.multiply %2410, %2389 : tensor<512xf32>
    %2412 = stablehlo.add %2409, %2411 : tensor<512xf32>
    %2413 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2414 = stablehlo.multiply %2413, %arg251 : tensor<512xf32>
    %2415 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2416 = stablehlo.multiply %2415, %2397 : tensor<512xf32>
    %2417 = stablehlo.add %2414, %2416 : tensor<512xf32>
    %2418 = stablehlo.broadcast_in_dim %2389, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2419 = stablehlo.broadcast_in_dim %2397, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2420 = stablehlo.convert %2371 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2421 = stablehlo.broadcast_in_dim %2418, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2422 = stablehlo.subtract %2420, %2421 : tensor<256x7x7x512xf32>
    %2423 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2424 = stablehlo.add %2419, %2423 : tensor<1x1x1x512xf32>
    %2425 = stablehlo.rsqrt %2424 : tensor<1x1x1x512xf32>
    %2426 = stablehlo.divide %2425, %2424 : tensor<1x1x1x512xf32>
    %2427 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2428 = stablehlo.multiply %2427, %2426 : tensor<1x1x1x512xf32>
    %2429 = stablehlo.reshape %arg45 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2430 = stablehlo.multiply %2425, %2429 : tensor<1x1x1x512xf32>
    %2431 = stablehlo.broadcast_in_dim %2430, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2432 = stablehlo.multiply %2422, %2431 : tensor<256x7x7x512xf32>
    %2433 = stablehlo.reshape %arg44 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2434 = stablehlo.broadcast_in_dim %2433, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2435 = stablehlo.add %2432, %2434 : tensor<256x7x7x512xf32>
    %2436 = stablehlo.convert %2435 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2437 = stablehlo.add %2295, %2436 : tensor<256x7x7x512xbf16>
    %2438 = call @relu_195(%2437) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2439 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2440 = stablehlo.compare  GT, %2437, %2439,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %2441 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2442 = stablehlo.convert %arg52 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %2443 = stablehlo.convolution(%2438, %2442) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2444 = stablehlo.convert %2443 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2445 = stablehlo.multiply %2444, %2444 : tensor<256x7x7x512xf32>
    %2446 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %2447 = stablehlo.multiply %2446, %2444 : tensor<256x7x7x512xf32>
    %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2448 = stablehlo.reduce(%2444 init: %cst_90) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2449 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2450 = stablehlo.divide %2448, %2449 : tensor<512xf32>
    %cst_91 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2451 = stablehlo.reduce(%2445 init: %cst_91) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2452 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2453 = stablehlo.divide %2451, %2452 : tensor<512xf32>
    %2454 = stablehlo.broadcast_in_dim %2450, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2455 = stablehlo.broadcast_in_dim %2453, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2456 = stablehlo.concatenate %2454, %2455, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %2457 = "stablehlo.all_reduce"(%2456) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %2458 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %2459 = stablehlo.divide %2457, %2458 : tensor<2x512xf32>
    %2460 = stablehlo.slice %2459 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2461 = stablehlo.reshape %2460 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2462 = stablehlo.slice %2459 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2463 = stablehlo.reshape %2462 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2464 = stablehlo.multiply %2461, %2461 : tensor<512xf32>
    %2465 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2466 = stablehlo.multiply %2465, %2461 : tensor<512xf32>
    %2467 = stablehlo.subtract %2463, %2464 : tensor<512xf32>
    %2468 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2469 = stablehlo.maximum %2468, %2467 : tensor<512xf32>
    %2470 = stablehlo.compare  EQ, %2467, %2469,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2471 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2472 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2473 = stablehlo.select %2470, %2471, %2472 : tensor<512xi1>, tensor<512xf32>
    %2474 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2475 = stablehlo.compare  EQ, %2474, %2469,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2476 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2477 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2478 = stablehlo.select %2475, %2476, %2477 : tensor<512xi1>, tensor<512xf32>
    %2479 = stablehlo.divide %2473, %2478 : tensor<512xf32>
    %2480 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2481 = stablehlo.multiply %2480, %arg252 : tensor<512xf32>
    %2482 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2483 = stablehlo.multiply %2482, %2461 : tensor<512xf32>
    %2484 = stablehlo.add %2481, %2483 : tensor<512xf32>
    %2485 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2486 = stablehlo.multiply %2485, %arg253 : tensor<512xf32>
    %2487 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2488 = stablehlo.multiply %2487, %2469 : tensor<512xf32>
    %2489 = stablehlo.add %2486, %2488 : tensor<512xf32>
    %2490 = stablehlo.broadcast_in_dim %2461, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2491 = stablehlo.broadcast_in_dim %2469, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2492 = stablehlo.convert %2443 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2493 = stablehlo.broadcast_in_dim %2490, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2494 = stablehlo.subtract %2492, %2493 : tensor<256x7x7x512xf32>
    %2495 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2496 = stablehlo.add %2491, %2495 : tensor<1x1x1x512xf32>
    %2497 = stablehlo.rsqrt %2496 : tensor<1x1x1x512xf32>
    %2498 = stablehlo.divide %2497, %2496 : tensor<1x1x1x512xf32>
    %2499 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2500 = stablehlo.multiply %2499, %2498 : tensor<1x1x1x512xf32>
    %2501 = stablehlo.reshape %arg49 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2502 = stablehlo.multiply %2497, %2501 : tensor<1x1x1x512xf32>
    %2503 = stablehlo.broadcast_in_dim %2502, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2504 = stablehlo.multiply %2494, %2503 : tensor<256x7x7x512xf32>
    %2505 = stablehlo.reshape %arg48 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2506 = stablehlo.broadcast_in_dim %2505, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2507 = stablehlo.add %2504, %2506 : tensor<256x7x7x512xf32>
    %2508 = stablehlo.convert %2507 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2509 = call @relu_195(%2508) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2510 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2511 = stablehlo.compare  GT, %2508, %2510,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %2512 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2513 = stablehlo.convert %arg53 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xbf16>
    %2514 = stablehlo.convolution(%2509, %2513) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2515 = stablehlo.convert %2514 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2516 = stablehlo.multiply %2515, %2515 : tensor<256x7x7x512xf32>
    %2517 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
    %2518 = stablehlo.multiply %2517, %2515 : tensor<256x7x7x512xf32>
    %cst_92 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2519 = stablehlo.reduce(%2515 init: %cst_92) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2520 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2521 = stablehlo.divide %2519, %2520 : tensor<512xf32>
    %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2522 = stablehlo.reduce(%2516 init: %cst_93) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2523 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2524 = stablehlo.divide %2522, %2523 : tensor<512xf32>
    %2525 = stablehlo.broadcast_in_dim %2521, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2526 = stablehlo.broadcast_in_dim %2524, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2527 = stablehlo.concatenate %2525, %2526, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
    %2528 = "stablehlo.all_reduce"(%2527) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %2529 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %2530 = stablehlo.divide %2528, %2529 : tensor<2x512xf32>
    %2531 = stablehlo.slice %2530 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2532 = stablehlo.reshape %2531 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2533 = stablehlo.slice %2530 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2534 = stablehlo.reshape %2533 : (tensor<1x512xf32>) -> tensor<512xf32>
    %2535 = stablehlo.multiply %2532, %2532 : tensor<512xf32>
    %2536 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2537 = stablehlo.multiply %2536, %2532 : tensor<512xf32>
    %2538 = stablehlo.subtract %2534, %2535 : tensor<512xf32>
    %2539 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2540 = stablehlo.maximum %2539, %2538 : tensor<512xf32>
    %2541 = stablehlo.compare  EQ, %2538, %2540,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2542 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2543 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2544 = stablehlo.select %2541, %2542, %2543 : tensor<512xi1>, tensor<512xf32>
    %2545 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2546 = stablehlo.compare  EQ, %2545, %2540,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
    %2547 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2548 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2549 = stablehlo.select %2546, %2547, %2548 : tensor<512xi1>, tensor<512xf32>
    %2550 = stablehlo.divide %2544, %2549 : tensor<512xf32>
    %2551 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2552 = stablehlo.multiply %2551, %arg254 : tensor<512xf32>
    %2553 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2554 = stablehlo.multiply %2553, %2532 : tensor<512xf32>
    %2555 = stablehlo.add %2552, %2554 : tensor<512xf32>
    %2556 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2557 = stablehlo.multiply %2556, %arg255 : tensor<512xf32>
    %2558 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2559 = stablehlo.multiply %2558, %2540 : tensor<512xf32>
    %2560 = stablehlo.add %2557, %2559 : tensor<512xf32>
    %2561 = stablehlo.broadcast_in_dim %2532, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2562 = stablehlo.broadcast_in_dim %2540, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2563 = stablehlo.convert %2514 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %2564 = stablehlo.broadcast_in_dim %2561, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2565 = stablehlo.subtract %2563, %2564 : tensor<256x7x7x512xf32>
    %2566 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2567 = stablehlo.add %2562, %2566 : tensor<1x1x1x512xf32>
    %2568 = stablehlo.rsqrt %2567 : tensor<1x1x1x512xf32>
    %2569 = stablehlo.divide %2568, %2567 : tensor<1x1x1x512xf32>
    %2570 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %2571 = stablehlo.multiply %2570, %2569 : tensor<1x1x1x512xf32>
    %2572 = stablehlo.reshape %arg51 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2573 = stablehlo.multiply %2568, %2572 : tensor<1x1x1x512xf32>
    %2574 = stablehlo.broadcast_in_dim %2573, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2575 = stablehlo.multiply %2565, %2574 : tensor<256x7x7x512xf32>
    %2576 = stablehlo.reshape %arg50 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2577 = stablehlo.broadcast_in_dim %2576, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2578 = stablehlo.add %2575, %2577 : tensor<256x7x7x512xf32>
    %2579 = stablehlo.convert %2578 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2580 = stablehlo.add %2438, %2579 : tensor<256x7x7x512xbf16>
    %2581 = call @relu_195(%2580) : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2582 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2583 = stablehlo.compare  GT, %2580, %2582,  FLOAT : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xi1>
    %2584 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2585 = stablehlo.convert %2581 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2586 = stablehlo.reduce(%2585 init: %cst_94) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<256x512xf32>
    %cst_95 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
    %2587 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
    %2588 = stablehlo.divide %2586, %2587 : tensor<256x512xf32>
    %2589 = stablehlo.convert %2588 : (tensor<256x512xf32>) -> tensor<256x512xbf16>
    %2590 = stablehlo.convert %arg2 : (tensor<512x1000xf32>) -> tensor<512x1000xbf16>
    %2591 = stablehlo.convert %arg1 : (tensor<1000xf32>) -> tensor<1000xbf16>
    %2592 = stablehlo.dot_general %2589, %2590, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xbf16>, tensor<512x1000xbf16>) -> tensor<256x1000xbf16>
    %2593 = stablehlo.reshape %2591 : (tensor<1000xbf16>) -> tensor<1x1000xbf16>
    %2594 = stablehlo.broadcast_in_dim %2593, dims = [0, 1] : (tensor<1x1000xbf16>) -> tensor<256x1000xbf16>
    %2595 = stablehlo.add %2592, %2594 : tensor<256x1000xbf16>
    %2596 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %2597 = stablehlo.iota dim = 0 : tensor<1000xi32>
    %2598 = stablehlo.reshape %2597 : (tensor<1000xi32>) -> tensor<1x1000xi32>
    %2599 = stablehlo.broadcast_in_dim %2596, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
    %2600 = stablehlo.broadcast_in_dim %2598, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
    %2601 = stablehlo.compare  EQ, %2599, %2600,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
    %2602 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
    %2603 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
    %2604 = stablehlo.select %2601, %2602, %2603 : tensor<256x1000xi1>, tensor<256x1000xf32>
    %2605 = stablehlo.convert %2604 : tensor<256x1000xf32>
    %2606:3 = call @log_softmax(%2595) : (tensor<256x1000xbf16>) -> (tensor<256x1000xbf16>, tensor<256x1000xbf16>, tensor<256x1xbf16>)
    %2607 = stablehlo.convert %2606#0 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
    %2608 = stablehlo.multiply %2605, %2607 : tensor<256x1000xf32>
    %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2609 = stablehlo.reduce(%2608 init: %cst_96) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %2610 = stablehlo.negate %2609 : tensor<256xf32>
    %cst_97 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2611 = stablehlo.reduce(%2610 init: %cst_97) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %cst_98 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
    %2612 = stablehlo.divide %2611, %cst_98 : tensor<f32>
    %2613 = stablehlo.multiply %arg2, %arg2 : tensor<512x1000xf32>
    %2614 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %2615 = stablehlo.multiply %2614, %arg2 : tensor<512x1000xf32>
    %cst_99 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2616 = stablehlo.reduce(%2613 init: %cst_99) applies stablehlo.add across dimensions = [0, 1] : (tensor<512x1000xf32>, tensor<f32>) -> tensor<f32>
    %2617 = stablehlo.add %cst_12, %2616 : tensor<f32>
    %2618 = stablehlo.multiply %arg7, %arg7 : tensor<3x3x64x64xf32>
    %2619 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2620 = stablehlo.multiply %2619, %arg7 : tensor<3x3x64x64xf32>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2621 = stablehlo.reduce(%2618 init: %cst_100) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %2622 = stablehlo.add %2617, %2621 : tensor<f32>
    %2623 = stablehlo.multiply %arg8, %arg8 : tensor<3x3x64x64xf32>
    %2624 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2625 = stablehlo.multiply %2624, %arg8 : tensor<3x3x64x64xf32>
    %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2626 = stablehlo.reduce(%2623 init: %cst_101) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %2627 = stablehlo.add %2622, %2626 : tensor<f32>
    %2628 = stablehlo.multiply %arg13, %arg13 : tensor<3x3x64x64xf32>
    %2629 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2630 = stablehlo.multiply %2629, %arg13 : tensor<3x3x64x64xf32>
    %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2631 = stablehlo.reduce(%2628 init: %cst_102) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %2632 = stablehlo.add %2627, %2631 : tensor<f32>
    %2633 = stablehlo.multiply %arg14, %arg14 : tensor<3x3x64x64xf32>
    %2634 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2635 = stablehlo.multiply %2634, %arg14 : tensor<3x3x64x64xf32>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2636 = stablehlo.reduce(%2633 init: %cst_103) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %2637 = stablehlo.add %2632, %2636 : tensor<f32>
    %2638 = stablehlo.multiply %arg19, %arg19 : tensor<3x3x256x256xf32>
    %2639 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2640 = stablehlo.multiply %2639, %arg19 : tensor<3x3x256x256xf32>
    %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2641 = stablehlo.reduce(%2638 init: %cst_104) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2642 = stablehlo.add %2637, %2641 : tensor<f32>
    %2643 = stablehlo.multiply %arg20, %arg20 : tensor<3x3x256x256xf32>
    %2644 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2645 = stablehlo.multiply %2644, %arg20 : tensor<3x3x256x256xf32>
    %cst_105 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2646 = stablehlo.reduce(%2643 init: %cst_105) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2647 = stablehlo.add %2642, %2646 : tensor<f32>
    %2648 = stablehlo.multiply %arg25, %arg25 : tensor<3x3x256x256xf32>
    %2649 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2650 = stablehlo.multiply %2649, %arg25 : tensor<3x3x256x256xf32>
    %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2651 = stablehlo.reduce(%2648 init: %cst_106) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2652 = stablehlo.add %2647, %2651 : tensor<f32>
    %2653 = stablehlo.multiply %arg26, %arg26 : tensor<3x3x256x256xf32>
    %2654 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2655 = stablehlo.multiply %2654, %arg26 : tensor<3x3x256x256xf32>
    %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2656 = stablehlo.reduce(%2653 init: %cst_107) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2657 = stablehlo.add %2652, %2656 : tensor<f32>
    %2658 = stablehlo.multiply %arg31, %arg31 : tensor<3x3x256x256xf32>
    %2659 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2660 = stablehlo.multiply %2659, %arg31 : tensor<3x3x256x256xf32>
    %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2661 = stablehlo.reduce(%2658 init: %cst_108) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2662 = stablehlo.add %2657, %2661 : tensor<f32>
    %2663 = stablehlo.multiply %arg32, %arg32 : tensor<3x3x256x256xf32>
    %2664 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2665 = stablehlo.multiply %2664, %arg32 : tensor<3x3x256x256xf32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2666 = stablehlo.reduce(%2663 init: %cst_109) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2667 = stablehlo.add %2662, %2666 : tensor<f32>
    %2668 = stablehlo.multiply %arg37, %arg37 : tensor<3x3x256x512xf32>
    %2669 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %2670 = stablehlo.multiply %2669, %arg37 : tensor<3x3x256x512xf32>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2671 = stablehlo.reduce(%2668 init: %cst_110) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x512xf32>, tensor<f32>) -> tensor<f32>
    %2672 = stablehlo.add %2667, %2671 : tensor<f32>
    %2673 = stablehlo.multiply %arg38, %arg38 : tensor<3x3x512x512xf32>
    %2674 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2675 = stablehlo.multiply %2674, %arg38 : tensor<3x3x512x512xf32>
    %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2676 = stablehlo.reduce(%2673 init: %cst_111) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xf32>, tensor<f32>) -> tensor<f32>
    %2677 = stablehlo.add %2672, %2676 : tensor<f32>
    %2678 = stablehlo.multiply %arg39, %arg39 : tensor<1x1x256x512xf32>
    %2679 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %2680 = stablehlo.multiply %2679, %arg39 : tensor<1x1x256x512xf32>
    %cst_112 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2681 = stablehlo.reduce(%2678 init: %cst_112) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<1x1x256x512xf32>, tensor<f32>) -> tensor<f32>
    %2682 = stablehlo.add %2677, %2681 : tensor<f32>
    %2683 = stablehlo.multiply %arg46, %arg46 : tensor<3x3x512x512xf32>
    %2684 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2685 = stablehlo.multiply %2684, %arg46 : tensor<3x3x512x512xf32>
    %cst_113 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2686 = stablehlo.reduce(%2683 init: %cst_113) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xf32>, tensor<f32>) -> tensor<f32>
    %2687 = stablehlo.add %2682, %2686 : tensor<f32>
    %2688 = stablehlo.multiply %arg47, %arg47 : tensor<3x3x512x512xf32>
    %2689 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2690 = stablehlo.multiply %2689, %arg47 : tensor<3x3x512x512xf32>
    %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2691 = stablehlo.reduce(%2688 init: %cst_114) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xf32>, tensor<f32>) -> tensor<f32>
    %2692 = stablehlo.add %2687, %2691 : tensor<f32>
    %2693 = stablehlo.multiply %arg52, %arg52 : tensor<3x3x512x512xf32>
    %2694 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2695 = stablehlo.multiply %2694, %arg52 : tensor<3x3x512x512xf32>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2696 = stablehlo.reduce(%2693 init: %cst_115) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xf32>, tensor<f32>) -> tensor<f32>
    %2697 = stablehlo.add %2692, %2696 : tensor<f32>
    %2698 = stablehlo.multiply %arg53, %arg53 : tensor<3x3x512x512xf32>
    %2699 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2700 = stablehlo.multiply %2699, %arg53 : tensor<3x3x512x512xf32>
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2701 = stablehlo.reduce(%2698 init: %cst_116) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xf32>, tensor<f32>) -> tensor<f32>
    %2702 = stablehlo.add %2697, %2701 : tensor<f32>
    %2703 = stablehlo.multiply %arg58, %arg58 : tensor<3x3x64x64xf32>
    %2704 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2705 = stablehlo.multiply %2704, %arg58 : tensor<3x3x64x64xf32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2706 = stablehlo.reduce(%2703 init: %cst_117) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %2707 = stablehlo.add %2702, %2706 : tensor<f32>
    %2708 = stablehlo.multiply %arg59, %arg59 : tensor<3x3x64x64xf32>
    %2709 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2710 = stablehlo.multiply %2709, %arg59 : tensor<3x3x64x64xf32>
    %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2711 = stablehlo.reduce(%2708 init: %cst_118) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xf32>, tensor<f32>) -> tensor<f32>
    %2712 = stablehlo.add %2707, %2711 : tensor<f32>
    %2713 = stablehlo.multiply %arg64, %arg64 : tensor<3x3x64x128xf32>
    %2714 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %2715 = stablehlo.multiply %2714, %arg64 : tensor<3x3x64x128xf32>
    %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2716 = stablehlo.reduce(%2713 init: %cst_119) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x128xf32>, tensor<f32>) -> tensor<f32>
    %2717 = stablehlo.add %2712, %2716 : tensor<f32>
    %2718 = stablehlo.multiply %arg65, %arg65 : tensor<3x3x128x128xf32>
    %2719 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2720 = stablehlo.multiply %2719, %arg65 : tensor<3x3x128x128xf32>
    %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2721 = stablehlo.reduce(%2718 init: %cst_120) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %2722 = stablehlo.add %2717, %2721 : tensor<f32>
    %2723 = stablehlo.multiply %arg66, %arg66 : tensor<1x1x64x128xf32>
    %2724 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %2725 = stablehlo.multiply %2724, %arg66 : tensor<1x1x64x128xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2726 = stablehlo.reduce(%2723 init: %cst_121) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<1x1x64x128xf32>, tensor<f32>) -> tensor<f32>
    %2727 = stablehlo.add %2722, %2726 : tensor<f32>
    %2728 = stablehlo.multiply %arg73, %arg73 : tensor<3x3x128x128xf32>
    %2729 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2730 = stablehlo.multiply %2729, %arg73 : tensor<3x3x128x128xf32>
    %cst_122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2731 = stablehlo.reduce(%2728 init: %cst_122) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %2732 = stablehlo.add %2727, %2731 : tensor<f32>
    %2733 = stablehlo.multiply %arg74, %arg74 : tensor<3x3x128x128xf32>
    %2734 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2735 = stablehlo.multiply %2734, %arg74 : tensor<3x3x128x128xf32>
    %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2736 = stablehlo.reduce(%2733 init: %cst_123) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %2737 = stablehlo.add %2732, %2736 : tensor<f32>
    %2738 = stablehlo.multiply %arg79, %arg79 : tensor<3x3x128x128xf32>
    %2739 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2740 = stablehlo.multiply %2739, %arg79 : tensor<3x3x128x128xf32>
    %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2741 = stablehlo.reduce(%2738 init: %cst_124) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %2742 = stablehlo.add %2737, %2741 : tensor<f32>
    %2743 = stablehlo.multiply %arg80, %arg80 : tensor<3x3x128x128xf32>
    %2744 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2745 = stablehlo.multiply %2744, %arg80 : tensor<3x3x128x128xf32>
    %cst_125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2746 = stablehlo.reduce(%2743 init: %cst_125) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %2747 = stablehlo.add %2742, %2746 : tensor<f32>
    %2748 = stablehlo.multiply %arg85, %arg85 : tensor<3x3x128x128xf32>
    %2749 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2750 = stablehlo.multiply %2749, %arg85 : tensor<3x3x128x128xf32>
    %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2751 = stablehlo.reduce(%2748 init: %cst_126) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %2752 = stablehlo.add %2747, %2751 : tensor<f32>
    %2753 = stablehlo.multiply %arg86, %arg86 : tensor<3x3x128x128xf32>
    %2754 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2755 = stablehlo.multiply %2754, %arg86 : tensor<3x3x128x128xf32>
    %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2756 = stablehlo.reduce(%2753 init: %cst_127) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xf32>, tensor<f32>) -> tensor<f32>
    %2757 = stablehlo.add %2752, %2756 : tensor<f32>
    %2758 = stablehlo.multiply %arg91, %arg91 : tensor<3x3x128x256xf32>
    %2759 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %2760 = stablehlo.multiply %2759, %arg91 : tensor<3x3x128x256xf32>
    %cst_128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2761 = stablehlo.reduce(%2758 init: %cst_128) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x256xf32>, tensor<f32>) -> tensor<f32>
    %2762 = stablehlo.add %2757, %2761 : tensor<f32>
    %2763 = stablehlo.multiply %arg92, %arg92 : tensor<3x3x256x256xf32>
    %2764 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2765 = stablehlo.multiply %2764, %arg92 : tensor<3x3x256x256xf32>
    %cst_129 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2766 = stablehlo.reduce(%2763 init: %cst_129) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2767 = stablehlo.add %2762, %2766 : tensor<f32>
    %2768 = stablehlo.multiply %arg93, %arg93 : tensor<1x1x128x256xf32>
    %2769 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %2770 = stablehlo.multiply %2769, %arg93 : tensor<1x1x128x256xf32>
    %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2771 = stablehlo.reduce(%2768 init: %cst_130) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<1x1x128x256xf32>, tensor<f32>) -> tensor<f32>
    %2772 = stablehlo.add %2767, %2771 : tensor<f32>
    %2773 = stablehlo.multiply %arg100, %arg100 : tensor<3x3x256x256xf32>
    %2774 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2775 = stablehlo.multiply %2774, %arg100 : tensor<3x3x256x256xf32>
    %cst_131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2776 = stablehlo.reduce(%2773 init: %cst_131) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2777 = stablehlo.add %2772, %2776 : tensor<f32>
    %2778 = stablehlo.multiply %arg101, %arg101 : tensor<3x3x256x256xf32>
    %2779 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2780 = stablehlo.multiply %2779, %arg101 : tensor<3x3x256x256xf32>
    %cst_132 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2781 = stablehlo.reduce(%2778 init: %cst_132) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2782 = stablehlo.add %2777, %2781 : tensor<f32>
    %2783 = stablehlo.multiply %arg106, %arg106 : tensor<3x3x256x256xf32>
    %2784 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2785 = stablehlo.multiply %2784, %arg106 : tensor<3x3x256x256xf32>
    %cst_133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2786 = stablehlo.reduce(%2783 init: %cst_133) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2787 = stablehlo.add %2782, %2786 : tensor<f32>
    %2788 = stablehlo.multiply %arg107, %arg107 : tensor<3x3x256x256xf32>
    %2789 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2790 = stablehlo.multiply %2789, %arg107 : tensor<3x3x256x256xf32>
    %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2791 = stablehlo.reduce(%2788 init: %cst_134) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xf32>, tensor<f32>) -> tensor<f32>
    %2792 = stablehlo.add %2787, %2791 : tensor<f32>
    %2793 = stablehlo.multiply %arg110, %arg110 : tensor<7x7x3x64xf32>
    %2794 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %2795 = stablehlo.multiply %2794, %arg110 : tensor<7x7x3x64xf32>
    %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2796 = stablehlo.reduce(%2793 init: %cst_135) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<7x7x3x64xf32>, tensor<f32>) -> tensor<f32>
    %2797 = stablehlo.add %2792, %2796 : tensor<f32>
    %cst_136 = stablehlo.constant dense<5.000000e-05> : tensor<f32>
    %2798 = stablehlo.multiply %cst_136, %2797 : tensor<f32>
    %2799 = stablehlo.add %2612, %2798 : tensor<f32>
    %2800 = stablehlo.convert %arg295 : tensor<f32>
    %2801 = stablehlo.multiply %2800, %2799 : tensor<f32>
    %2802 = stablehlo.multiply %2800, %cst_13 : tensor<f32>
    %2803 = stablehlo.multiply %cst_136, %2802 : tensor<f32>
    %2804 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %2805 = stablehlo.multiply %2804, %2795 : tensor<7x7x3x64xf32>
    %2806 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2807 = stablehlo.multiply %2806, %2790 : tensor<3x3x256x256xf32>
    %2808 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2809 = stablehlo.multiply %2808, %2785 : tensor<3x3x256x256xf32>
    %2810 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2811 = stablehlo.multiply %2810, %2780 : tensor<3x3x256x256xf32>
    %2812 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2813 = stablehlo.multiply %2812, %2775 : tensor<3x3x256x256xf32>
    %2814 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %2815 = stablehlo.multiply %2814, %2770 : tensor<1x1x128x256xf32>
    %2816 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2817 = stablehlo.multiply %2816, %2765 : tensor<3x3x256x256xf32>
    %2818 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %2819 = stablehlo.multiply %2818, %2760 : tensor<3x3x128x256xf32>
    %2820 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2821 = stablehlo.multiply %2820, %2755 : tensor<3x3x128x128xf32>
    %2822 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2823 = stablehlo.multiply %2822, %2750 : tensor<3x3x128x128xf32>
    %2824 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2825 = stablehlo.multiply %2824, %2745 : tensor<3x3x128x128xf32>
    %2826 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2827 = stablehlo.multiply %2826, %2740 : tensor<3x3x128x128xf32>
    %2828 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2829 = stablehlo.multiply %2828, %2735 : tensor<3x3x128x128xf32>
    %2830 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2831 = stablehlo.multiply %2830, %2730 : tensor<3x3x128x128xf32>
    %2832 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %2833 = stablehlo.multiply %2832, %2725 : tensor<1x1x64x128xf32>
    %2834 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %2835 = stablehlo.multiply %2834, %2720 : tensor<3x3x128x128xf32>
    %2836 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %2837 = stablehlo.multiply %2836, %2715 : tensor<3x3x64x128xf32>
    %2838 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2839 = stablehlo.multiply %2838, %2710 : tensor<3x3x64x64xf32>
    %2840 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2841 = stablehlo.multiply %2840, %2705 : tensor<3x3x64x64xf32>
    %2842 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2843 = stablehlo.multiply %2842, %2700 : tensor<3x3x512x512xf32>
    %2844 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2845 = stablehlo.multiply %2844, %2695 : tensor<3x3x512x512xf32>
    %2846 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2847 = stablehlo.multiply %2846, %2690 : tensor<3x3x512x512xf32>
    %2848 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2849 = stablehlo.multiply %2848, %2685 : tensor<3x3x512x512xf32>
    %2850 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %2851 = stablehlo.multiply %2850, %2680 : tensor<1x1x256x512xf32>
    %2852 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %2853 = stablehlo.multiply %2852, %2675 : tensor<3x3x512x512xf32>
    %2854 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %2855 = stablehlo.multiply %2854, %2670 : tensor<3x3x256x512xf32>
    %2856 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2857 = stablehlo.multiply %2856, %2665 : tensor<3x3x256x256xf32>
    %2858 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2859 = stablehlo.multiply %2858, %2660 : tensor<3x3x256x256xf32>
    %2860 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2861 = stablehlo.multiply %2860, %2655 : tensor<3x3x256x256xf32>
    %2862 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2863 = stablehlo.multiply %2862, %2650 : tensor<3x3x256x256xf32>
    %2864 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2865 = stablehlo.multiply %2864, %2645 : tensor<3x3x256x256xf32>
    %2866 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %2867 = stablehlo.multiply %2866, %2640 : tensor<3x3x256x256xf32>
    %2868 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2869 = stablehlo.multiply %2868, %2635 : tensor<3x3x64x64xf32>
    %2870 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2871 = stablehlo.multiply %2870, %2630 : tensor<3x3x64x64xf32>
    %2872 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2873 = stablehlo.multiply %2872, %2625 : tensor<3x3x64x64xf32>
    %2874 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %2875 = stablehlo.multiply %2874, %2620 : tensor<3x3x64x64xf32>
    %2876 = stablehlo.broadcast_in_dim %2803, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %2877 = stablehlo.multiply %2876, %2615 : tensor<512x1000xf32>
    %2878 = stablehlo.divide %2802, %cst_98 : tensor<f32>
    %2879 = stablehlo.broadcast_in_dim %2878, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2880 = stablehlo.negate %2879 : tensor<256xf32>
    %2881 = stablehlo.broadcast_in_dim %2880, dims = [0] : (tensor<256xf32>) -> tensor<256x1000xf32>
    %2882 = stablehlo.multiply %2605, %2881 : tensor<256x1000xf32>
    %2883 = stablehlo.convert %2882 : (tensor<256x1000xf32>) -> tensor<256x1000xbf16>
    %2884 = call @log_softmax_308(%2606#1, %2606#2, %2883) : (tensor<256x1000xbf16>, tensor<256x1xbf16>, tensor<256x1000xbf16>) -> tensor<256x1000xbf16>
    %cst_137 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %2885 = stablehlo.reduce(%2884 init: %cst_137) applies stablehlo.add across dimensions = [0] : (tensor<256x1000xbf16>, tensor<bf16>) -> tensor<1000xbf16>
    %2886 = stablehlo.reshape %2885 : (tensor<1000xbf16>) -> tensor<1x1000xbf16>
    %2887 = stablehlo.reshape %2886 : (tensor<1x1000xbf16>) -> tensor<1000xbf16>
    %2888 = stablehlo.dot_general %2884, %2589, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x1000xbf16>, tensor<256x512xbf16>) -> tensor<1000x512xbf16>
    %2889 = stablehlo.transpose %2888, dims = [1, 0] : (tensor<1000x512xbf16>) -> tensor<512x1000xbf16>
    %2890 = stablehlo.dot_general %2884, %2590, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1000xbf16>, tensor<512x1000xbf16>) -> tensor<256x512xbf16>
    %2891 = stablehlo.convert %2887 : (tensor<1000xbf16>) -> tensor<1000xf32>
    %2892 = stablehlo.convert %2889 : (tensor<512x1000xbf16>) -> tensor<512x1000xf32>
    %2893 = stablehlo.add %2877, %2892 : tensor<512x1000xf32>
    %2894 = stablehlo.convert %2890 : (tensor<256x512xbf16>) -> tensor<256x512xf32>
    %2895 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
    %2896 = stablehlo.divide %2894, %2895 : tensor<256x512xf32>
    %2897 = stablehlo.broadcast_in_dim %2896, dims = [0, 3] : (tensor<256x512xf32>) -> tensor<256x7x7x512xf32>
    %2898 = stablehlo.convert %2897 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2899 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2900 = stablehlo.select %2583, %2898, %2899 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %2901 = stablehlo.convert %2900 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2902 = stablehlo.reduce(%2901 init: %cst_138) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2903 = stablehlo.reshape %2902 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2904 = stablehlo.reshape %2903 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %2905 = stablehlo.multiply %2565, %2901 : tensor<256x7x7x512xf32>
    %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2906 = stablehlo.reduce(%2905 init: %cst_139) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2907 = stablehlo.reshape %2906 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2908 = stablehlo.broadcast_in_dim %2573, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2909 = stablehlo.multiply %2901, %2908 : tensor<256x7x7x512xf32>
    %2910 = stablehlo.multiply %2568, %2907 : tensor<1x1x1x512xf32>
    %2911 = stablehlo.multiply %2907, %2572 : tensor<1x1x1x512xf32>
    %2912 = stablehlo.reshape %2910 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %2913 = stablehlo.multiply %2911, %2571 : tensor<1x1x1x512xf32>
    %2914 = stablehlo.negate %2909 : tensor<256x7x7x512xf32>
    %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2915 = stablehlo.reduce(%2914 init: %cst_140) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2916 = stablehlo.reshape %2915 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2917 = stablehlo.convert %2909 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2918 = stablehlo.reduce(%2913 init: %cst_141) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2919 = stablehlo.reduce(%2916 init: %cst_142) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2920 = stablehlo.multiply %2918, %2550 : tensor<512xf32>
    %2921 = stablehlo.negate %2920 : tensor<512xf32>
    %2922 = stablehlo.multiply %2921, %2537 : tensor<512xf32>
    %2923 = stablehlo.add %2919, %2922 : tensor<512xf32>
    %2924 = stablehlo.broadcast_in_dim %2920, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2925 = stablehlo.pad %2924, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %2926 = stablehlo.broadcast_in_dim %2923, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2927 = stablehlo.pad %2926, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %2928 = stablehlo.add %2925, %2927 : tensor<2x512xf32>
    %2929 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %2930 = stablehlo.divide %2928, %2929 : tensor<2x512xf32>
    %2931 = "stablehlo.all_reduce"(%2930) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %2932 = stablehlo.slice %2931 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2933 = stablehlo.slice %2931 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2934 = stablehlo.reduce(%2933 init: %cst_143) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2935 = stablehlo.reduce(%2932 init: %cst_144) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2936 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2937 = stablehlo.divide %2934, %2936 : tensor<512xf32>
    %2938 = stablehlo.broadcast_in_dim %2937, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %2939 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2940 = stablehlo.divide %2935, %2939 : tensor<512xf32>
    %2941 = stablehlo.broadcast_in_dim %2940, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %2942 = stablehlo.multiply %2938, %2518 : tensor<256x7x7x512xf32>
    %2943 = stablehlo.add %2941, %2942 : tensor<256x7x7x512xf32>
    %2944 = stablehlo.convert %2943 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2945 = stablehlo.add %2917, %2944 : tensor<256x7x7x512xbf16>
    %2946 = stablehlo.convolution(%2509, %2945) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x512x512xbf16>
    %2947 = stablehlo.reverse %2513, dims = [0, 1] : tensor<3x3x512x512xbf16>
    %2948 = stablehlo.convolution(%2945, %2947) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %2949 = stablehlo.convert %2946 : (tensor<3x3x512x512xbf16>) -> tensor<3x3x512x512xf32>
    %2950 = stablehlo.add %2843, %2949 : tensor<3x3x512x512xf32>
    %2951 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %2952 = stablehlo.select %2511, %2948, %2951 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %2953 = stablehlo.convert %2952 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2954 = stablehlo.reduce(%2953 init: %cst_145) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2955 = stablehlo.reshape %2954 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2956 = stablehlo.reshape %2955 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %2957 = stablehlo.multiply %2494, %2953 : tensor<256x7x7x512xf32>
    %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2958 = stablehlo.reduce(%2957 init: %cst_146) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2959 = stablehlo.reshape %2958 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2960 = stablehlo.broadcast_in_dim %2502, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %2961 = stablehlo.multiply %2953, %2960 : tensor<256x7x7x512xf32>
    %2962 = stablehlo.multiply %2497, %2959 : tensor<1x1x1x512xf32>
    %2963 = stablehlo.multiply %2959, %2501 : tensor<1x1x1x512xf32>
    %2964 = stablehlo.reshape %2962 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %2965 = stablehlo.multiply %2963, %2500 : tensor<1x1x1x512xf32>
    %2966 = stablehlo.negate %2961 : tensor<256x7x7x512xf32>
    %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2967 = stablehlo.reduce(%2966 init: %cst_147) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2968 = stablehlo.reshape %2967 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %2969 = stablehlo.convert %2961 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2970 = stablehlo.reduce(%2965 init: %cst_148) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2971 = stablehlo.reduce(%2968 init: %cst_149) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2972 = stablehlo.multiply %2970, %2479 : tensor<512xf32>
    %2973 = stablehlo.negate %2972 : tensor<512xf32>
    %2974 = stablehlo.multiply %2973, %2466 : tensor<512xf32>
    %2975 = stablehlo.add %2971, %2974 : tensor<512xf32>
    %2976 = stablehlo.broadcast_in_dim %2972, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2977 = stablehlo.pad %2976, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %2978 = stablehlo.broadcast_in_dim %2975, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %2979 = stablehlo.pad %2978, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %2980 = stablehlo.add %2977, %2979 : tensor<2x512xf32>
    %2981 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %2982 = stablehlo.divide %2980, %2981 : tensor<2x512xf32>
    %2983 = "stablehlo.all_reduce"(%2982) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %2984 = stablehlo.slice %2983 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %2985 = stablehlo.slice %2983 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_150 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2986 = stablehlo.reduce(%2985 init: %cst_150) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2987 = stablehlo.reduce(%2984 init: %cst_151) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %2988 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2989 = stablehlo.divide %2986, %2988 : tensor<512xf32>
    %2990 = stablehlo.broadcast_in_dim %2989, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %2991 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2992 = stablehlo.divide %2987, %2991 : tensor<512xf32>
    %2993 = stablehlo.broadcast_in_dim %2992, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %2994 = stablehlo.multiply %2990, %2447 : tensor<256x7x7x512xf32>
    %2995 = stablehlo.add %2993, %2994 : tensor<256x7x7x512xf32>
    %2996 = stablehlo.convert %2995 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %2997 = stablehlo.add %2969, %2996 : tensor<256x7x7x512xbf16>
    %2998 = stablehlo.convolution(%2438, %2997) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x512x512xbf16>
    %2999 = stablehlo.reverse %2442, dims = [0, 1] : tensor<3x3x512x512xbf16>
    %3000 = stablehlo.convolution(%2997, %2999) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3001 = stablehlo.add %2900, %3000 : tensor<256x7x7x512xbf16>
    %3002 = stablehlo.convert %2998 : (tensor<3x3x512x512xbf16>) -> tensor<3x3x512x512xf32>
    %3003 = stablehlo.add %2845, %3002 : tensor<3x3x512x512xf32>
    %3004 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %3005 = stablehlo.select %2440, %3001, %3004 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %3006 = stablehlo.convert %3005 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3007 = stablehlo.reduce(%3006 init: %cst_152) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3008 = stablehlo.reshape %3007 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3009 = stablehlo.reshape %3008 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3010 = stablehlo.multiply %2422, %3006 : tensor<256x7x7x512xf32>
    %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3011 = stablehlo.reduce(%3010 init: %cst_153) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3012 = stablehlo.reshape %3011 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3013 = stablehlo.broadcast_in_dim %2430, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3014 = stablehlo.multiply %3006, %3013 : tensor<256x7x7x512xf32>
    %3015 = stablehlo.multiply %2425, %3012 : tensor<1x1x1x512xf32>
    %3016 = stablehlo.multiply %3012, %2429 : tensor<1x1x1x512xf32>
    %3017 = stablehlo.reshape %3015 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3018 = stablehlo.multiply %3016, %2428 : tensor<1x1x1x512xf32>
    %3019 = stablehlo.negate %3014 : tensor<256x7x7x512xf32>
    %cst_154 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3020 = stablehlo.reduce(%3019 init: %cst_154) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3021 = stablehlo.reshape %3020 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3022 = stablehlo.convert %3014 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3023 = stablehlo.reduce(%3018 init: %cst_155) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3024 = stablehlo.reduce(%3021 init: %cst_156) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3025 = stablehlo.multiply %3023, %2407 : tensor<512xf32>
    %3026 = stablehlo.negate %3025 : tensor<512xf32>
    %3027 = stablehlo.multiply %3026, %2394 : tensor<512xf32>
    %3028 = stablehlo.add %3024, %3027 : tensor<512xf32>
    %3029 = stablehlo.broadcast_in_dim %3025, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3030 = stablehlo.pad %3029, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3031 = stablehlo.broadcast_in_dim %3028, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3032 = stablehlo.pad %3031, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3033 = stablehlo.add %3030, %3032 : tensor<2x512xf32>
    %3034 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %3035 = stablehlo.divide %3033, %3034 : tensor<2x512xf32>
    %3036 = "stablehlo.all_reduce"(%3035) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %3037 = stablehlo.slice %3036 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %3038 = stablehlo.slice %3036 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3039 = stablehlo.reduce(%3038 init: %cst_157) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_158 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3040 = stablehlo.reduce(%3037 init: %cst_158) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3041 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3042 = stablehlo.divide %3039, %3041 : tensor<512xf32>
    %3043 = stablehlo.broadcast_in_dim %3042, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3044 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3045 = stablehlo.divide %3040, %3044 : tensor<512xf32>
    %3046 = stablehlo.broadcast_in_dim %3045, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3047 = stablehlo.multiply %3043, %2375 : tensor<256x7x7x512xf32>
    %3048 = stablehlo.add %3046, %3047 : tensor<256x7x7x512xf32>
    %3049 = stablehlo.convert %3048 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3050 = stablehlo.add %3022, %3049 : tensor<256x7x7x512xbf16>
    %3051 = stablehlo.convolution(%2366, %3050) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x512x512xbf16>
    %3052 = stablehlo.reverse %2370, dims = [0, 1] : tensor<3x3x512x512xbf16>
    %3053 = stablehlo.convolution(%3050, %3052) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3054 = stablehlo.convert %3051 : (tensor<3x3x512x512xbf16>) -> tensor<3x3x512x512xf32>
    %3055 = stablehlo.add %2847, %3054 : tensor<3x3x512x512xf32>
    %3056 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %3057 = stablehlo.select %2368, %3053, %3056 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %3058 = stablehlo.convert %3057 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3059 = stablehlo.reduce(%3058 init: %cst_159) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3060 = stablehlo.reshape %3059 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3061 = stablehlo.reshape %3060 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3062 = stablehlo.multiply %2351, %3058 : tensor<256x7x7x512xf32>
    %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3063 = stablehlo.reduce(%3062 init: %cst_160) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3064 = stablehlo.reshape %3063 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3065 = stablehlo.broadcast_in_dim %2359, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3066 = stablehlo.multiply %3058, %3065 : tensor<256x7x7x512xf32>
    %3067 = stablehlo.multiply %2354, %3064 : tensor<1x1x1x512xf32>
    %3068 = stablehlo.multiply %3064, %2358 : tensor<1x1x1x512xf32>
    %3069 = stablehlo.reshape %3067 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3070 = stablehlo.multiply %3068, %2357 : tensor<1x1x1x512xf32>
    %3071 = stablehlo.negate %3066 : tensor<256x7x7x512xf32>
    %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3072 = stablehlo.reduce(%3071 init: %cst_161) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3073 = stablehlo.reshape %3072 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3074 = stablehlo.convert %3066 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3075 = stablehlo.reduce(%3070 init: %cst_162) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3076 = stablehlo.reduce(%3073 init: %cst_163) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3077 = stablehlo.multiply %3075, %2336 : tensor<512xf32>
    %3078 = stablehlo.negate %3077 : tensor<512xf32>
    %3079 = stablehlo.multiply %3078, %2323 : tensor<512xf32>
    %3080 = stablehlo.add %3076, %3079 : tensor<512xf32>
    %3081 = stablehlo.broadcast_in_dim %3077, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3082 = stablehlo.pad %3081, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3083 = stablehlo.broadcast_in_dim %3080, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3084 = stablehlo.pad %3083, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3085 = stablehlo.add %3082, %3084 : tensor<2x512xf32>
    %3086 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %3087 = stablehlo.divide %3085, %3086 : tensor<2x512xf32>
    %3088 = "stablehlo.all_reduce"(%3087) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %3089 = stablehlo.slice %3088 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %3090 = stablehlo.slice %3088 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3091 = stablehlo.reduce(%3090 init: %cst_164) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3092 = stablehlo.reduce(%3089 init: %cst_165) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3093 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3094 = stablehlo.divide %3091, %3093 : tensor<512xf32>
    %3095 = stablehlo.broadcast_in_dim %3094, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3096 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3097 = stablehlo.divide %3092, %3096 : tensor<512xf32>
    %3098 = stablehlo.broadcast_in_dim %3097, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3099 = stablehlo.multiply %3095, %2304 : tensor<256x7x7x512xf32>
    %3100 = stablehlo.add %3098, %3099 : tensor<256x7x7x512xf32>
    %3101 = stablehlo.convert %3100 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3102 = stablehlo.add %3074, %3101 : tensor<256x7x7x512xbf16>
    %3103 = stablehlo.convolution(%2295, %3102) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x512x512xbf16>
    %3104 = stablehlo.reverse %2299, dims = [0, 1] : tensor<3x3x512x512xbf16>
    %3105 = stablehlo.convolution(%3102, %3104) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3106 = stablehlo.add %3005, %3105 : tensor<256x7x7x512xbf16>
    %3107 = stablehlo.convert %3103 : (tensor<3x3x512x512xbf16>) -> tensor<3x3x512x512xf32>
    %3108 = stablehlo.add %2849, %3107 : tensor<3x3x512x512xf32>
    %3109 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %3110 = stablehlo.select %2297, %3106, %3109 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %3111 = stablehlo.convert %3110 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3112 = stablehlo.reduce(%3111 init: %cst_166) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3113 = stablehlo.reshape %3112 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3114 = stablehlo.reshape %3113 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3115 = stablehlo.multiply %2279, %3111 : tensor<256x7x7x512xf32>
    %cst_167 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3116 = stablehlo.reduce(%3115 init: %cst_167) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3117 = stablehlo.reshape %3116 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3118 = stablehlo.broadcast_in_dim %2287, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3119 = stablehlo.multiply %3111, %3118 : tensor<256x7x7x512xf32>
    %3120 = stablehlo.multiply %2282, %3117 : tensor<1x1x1x512xf32>
    %3121 = stablehlo.multiply %3117, %2286 : tensor<1x1x1x512xf32>
    %3122 = stablehlo.reshape %3120 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3123 = stablehlo.multiply %3121, %2285 : tensor<1x1x1x512xf32>
    %3124 = stablehlo.negate %3119 : tensor<256x7x7x512xf32>
    %cst_168 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3125 = stablehlo.reduce(%3124 init: %cst_168) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3126 = stablehlo.reshape %3125 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3127 = stablehlo.convert %3119 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3128 = stablehlo.reduce(%3123 init: %cst_169) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3129 = stablehlo.reduce(%3126 init: %cst_170) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3130 = stablehlo.multiply %3128, %2264 : tensor<512xf32>
    %3131 = stablehlo.negate %3130 : tensor<512xf32>
    %3132 = stablehlo.multiply %3131, %2251 : tensor<512xf32>
    %3133 = stablehlo.add %3129, %3132 : tensor<512xf32>
    %3134 = stablehlo.broadcast_in_dim %3130, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3135 = stablehlo.pad %3134, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3136 = stablehlo.broadcast_in_dim %3133, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3137 = stablehlo.pad %3136, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3138 = stablehlo.add %3135, %3137 : tensor<2x512xf32>
    %3139 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %3140 = stablehlo.divide %3138, %3139 : tensor<2x512xf32>
    %3141 = "stablehlo.all_reduce"(%3140) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %3142 = stablehlo.slice %3141 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %3143 = stablehlo.slice %3141 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3144 = stablehlo.reduce(%3143 init: %cst_171) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3145 = stablehlo.reduce(%3142 init: %cst_172) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3146 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3147 = stablehlo.divide %3144, %3146 : tensor<512xf32>
    %3148 = stablehlo.broadcast_in_dim %3147, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3149 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3150 = stablehlo.divide %3145, %3149 : tensor<512xf32>
    %3151 = stablehlo.broadcast_in_dim %3150, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3152 = stablehlo.multiply %3148, %2232 : tensor<256x7x7x512xf32>
    %3153 = stablehlo.add %3151, %3152 : tensor<256x7x7x512xf32>
    %3154 = stablehlo.convert %3153 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3155 = stablehlo.add %3127, %3154 : tensor<256x7x7x512xbf16>
    %3156 = stablehlo.convolution(%2085, %3155) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x7x7x512xbf16>) -> tensor<1x1x256x512xbf16>
    %3157 = stablehlo.reverse %2227, dims = [0, 1] : tensor<1x1x256x512xbf16>
    %3158 = stablehlo.convolution(%3155, %3157) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<1x1x256x512xbf16>) -> tensor<256x14x14x256xbf16>
    %3159 = stablehlo.convert %3156 : (tensor<1x1x256x512xbf16>) -> tensor<1x1x256x512xf32>
    %3160 = stablehlo.add %2851, %3159 : tensor<1x1x256x512xf32>
    %3161 = stablehlo.convert %3110 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3162 = stablehlo.reduce(%3161 init: %cst_173) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3163 = stablehlo.reshape %3162 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3164 = stablehlo.reshape %3163 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3165 = stablehlo.multiply %2212, %3161 : tensor<256x7x7x512xf32>
    %cst_174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3166 = stablehlo.reduce(%3165 init: %cst_174) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3167 = stablehlo.reshape %3166 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3168 = stablehlo.broadcast_in_dim %2220, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3169 = stablehlo.multiply %3161, %3168 : tensor<256x7x7x512xf32>
    %3170 = stablehlo.multiply %2215, %3167 : tensor<1x1x1x512xf32>
    %3171 = stablehlo.multiply %3167, %2219 : tensor<1x1x1x512xf32>
    %3172 = stablehlo.reshape %3170 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3173 = stablehlo.multiply %3171, %2218 : tensor<1x1x1x512xf32>
    %3174 = stablehlo.negate %3169 : tensor<256x7x7x512xf32>
    %cst_175 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3175 = stablehlo.reduce(%3174 init: %cst_175) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3176 = stablehlo.reshape %3175 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3177 = stablehlo.convert %3169 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_176 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3178 = stablehlo.reduce(%3173 init: %cst_176) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3179 = stablehlo.reduce(%3176 init: %cst_177) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3180 = stablehlo.multiply %3178, %2197 : tensor<512xf32>
    %3181 = stablehlo.negate %3180 : tensor<512xf32>
    %3182 = stablehlo.multiply %3181, %2184 : tensor<512xf32>
    %3183 = stablehlo.add %3179, %3182 : tensor<512xf32>
    %3184 = stablehlo.broadcast_in_dim %3180, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3185 = stablehlo.pad %3184, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3186 = stablehlo.broadcast_in_dim %3183, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3187 = stablehlo.pad %3186, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3188 = stablehlo.add %3185, %3187 : tensor<2x512xf32>
    %3189 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %3190 = stablehlo.divide %3188, %3189 : tensor<2x512xf32>
    %3191 = "stablehlo.all_reduce"(%3190) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %3192 = stablehlo.slice %3191 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %3193 = stablehlo.slice %3191 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3194 = stablehlo.reduce(%3193 init: %cst_178) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3195 = stablehlo.reduce(%3192 init: %cst_179) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3196 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3197 = stablehlo.divide %3194, %3196 : tensor<512xf32>
    %3198 = stablehlo.broadcast_in_dim %3197, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3199 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3200 = stablehlo.divide %3195, %3199 : tensor<512xf32>
    %3201 = stablehlo.broadcast_in_dim %3200, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3202 = stablehlo.multiply %3198, %2165 : tensor<256x7x7x512xf32>
    %3203 = stablehlo.add %3201, %3202 : tensor<256x7x7x512xf32>
    %3204 = stablehlo.convert %3203 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3205 = stablehlo.add %3177, %3204 : tensor<256x7x7x512xbf16>
    %3206 = stablehlo.convolution(%2156, %3205) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x512x512xbf16>
    %3207 = stablehlo.reverse %2160, dims = [0, 1] : tensor<3x3x512x512xbf16>
    %3208 = stablehlo.convolution(%3205, %3207) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x512x512xbf16>) -> tensor<256x7x7x512xbf16>
    %3209 = stablehlo.convert %3206 : (tensor<3x3x512x512xbf16>) -> tensor<3x3x512x512xf32>
    %3210 = stablehlo.add %2853, %3209 : tensor<3x3x512x512xf32>
    %3211 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x7x7x512xbf16>
    %3212 = stablehlo.select %2158, %3208, %3211 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xbf16>
    %3213 = stablehlo.convert %3212 : (tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xf32>
    %cst_180 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3214 = stablehlo.reduce(%3213 init: %cst_180) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3215 = stablehlo.reshape %3214 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3216 = stablehlo.reshape %3215 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3217 = stablehlo.multiply %2141, %3213 : tensor<256x7x7x512xf32>
    %cst_181 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3218 = stablehlo.reduce(%3217 init: %cst_181) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3219 = stablehlo.reshape %3218 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3220 = stablehlo.broadcast_in_dim %2149, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
    %3221 = stablehlo.multiply %3213, %3220 : tensor<256x7x7x512xf32>
    %3222 = stablehlo.multiply %2144, %3219 : tensor<1x1x1x512xf32>
    %3223 = stablehlo.multiply %3219, %2148 : tensor<1x1x1x512xf32>
    %3224 = stablehlo.reshape %3222 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
    %3225 = stablehlo.multiply %3223, %2147 : tensor<1x1x1x512xf32>
    %3226 = stablehlo.negate %3221 : tensor<256x7x7x512xf32>
    %cst_182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3227 = stablehlo.reduce(%3226 init: %cst_182) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3228 = stablehlo.reshape %3227 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %3229 = stablehlo.convert %3221 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %cst_183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3230 = stablehlo.reduce(%3225 init: %cst_183) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3231 = stablehlo.reduce(%3228 init: %cst_184) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3232 = stablehlo.multiply %3230, %2126 : tensor<512xf32>
    %3233 = stablehlo.negate %3232 : tensor<512xf32>
    %3234 = stablehlo.multiply %3233, %2113 : tensor<512xf32>
    %3235 = stablehlo.add %3231, %3234 : tensor<512xf32>
    %3236 = stablehlo.broadcast_in_dim %3232, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3237 = stablehlo.pad %3236, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3238 = stablehlo.broadcast_in_dim %3235, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3239 = stablehlo.pad %3238, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
    %3240 = stablehlo.add %3237, %3239 : tensor<2x512xf32>
    %3241 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
    %3242 = stablehlo.divide %3240, %3241 : tensor<2x512xf32>
    %3243 = "stablehlo.all_reduce"(%3242) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
    %3244 = stablehlo.slice %3243 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %3245 = stablehlo.slice %3243 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
    %cst_185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3246 = stablehlo.reduce(%3245 init: %cst_185) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3247 = stablehlo.reduce(%3244 init: %cst_186) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %3248 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3249 = stablehlo.divide %3246, %3248 : tensor<512xf32>
    %3250 = stablehlo.broadcast_in_dim %3249, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3251 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3252 = stablehlo.divide %3247, %3251 : tensor<512xf32>
    %3253 = stablehlo.broadcast_in_dim %3252, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
    %3254 = stablehlo.multiply %3250, %2094 : tensor<256x7x7x512xf32>
    %3255 = stablehlo.add %3253, %3254 : tensor<256x7x7x512xf32>
    %3256 = stablehlo.convert %3255 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xbf16>
    %3257 = stablehlo.add %3229, %3256 : tensor<256x7x7x512xbf16>
    %3258 = stablehlo.convolution(%2085, %3257) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x7x7x512xbf16>) -> tensor<3x3x256x512xbf16>
    %3259 = stablehlo.reverse %2089, dims = [0, 1] : tensor<3x3x256x512xbf16>
    %3260 = stablehlo.convolution(%3257, %3259) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xbf16>, tensor<3x3x256x512xbf16>) -> tensor<256x14x14x256xbf16>
    %3261 = stablehlo.add %3158, %3260 : tensor<256x14x14x256xbf16>
    %3262 = stablehlo.convert %3258 : (tensor<3x3x256x512xbf16>) -> tensor<3x3x256x512xf32>
    %3263 = stablehlo.add %2855, %3262 : tensor<3x3x256x512xf32>
    %3264 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3265 = stablehlo.select %2087, %3261, %3264 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3266 = stablehlo.convert %3265 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3267 = stablehlo.reduce(%3266 init: %cst_187) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3268 = stablehlo.reshape %3267 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3269 = stablehlo.reshape %3268 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3270 = stablehlo.multiply %2069, %3266 : tensor<256x14x14x256xf32>
    %cst_188 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3271 = stablehlo.reduce(%3270 init: %cst_188) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3272 = stablehlo.reshape %3271 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3273 = stablehlo.broadcast_in_dim %2077, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3274 = stablehlo.multiply %3266, %3273 : tensor<256x14x14x256xf32>
    %3275 = stablehlo.multiply %2072, %3272 : tensor<1x1x1x256xf32>
    %3276 = stablehlo.multiply %3272, %2076 : tensor<1x1x1x256xf32>
    %3277 = stablehlo.reshape %3275 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3278 = stablehlo.multiply %3276, %2075 : tensor<1x1x1x256xf32>
    %3279 = stablehlo.negate %3274 : tensor<256x14x14x256xf32>
    %cst_189 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3280 = stablehlo.reduce(%3279 init: %cst_189) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3281 = stablehlo.reshape %3280 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3282 = stablehlo.convert %3274 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3283 = stablehlo.reduce(%3278 init: %cst_190) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_191 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3284 = stablehlo.reduce(%3281 init: %cst_191) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3285 = stablehlo.multiply %3283, %2054 : tensor<256xf32>
    %3286 = stablehlo.negate %3285 : tensor<256xf32>
    %3287 = stablehlo.multiply %3286, %2041 : tensor<256xf32>
    %3288 = stablehlo.add %3284, %3287 : tensor<256xf32>
    %3289 = stablehlo.broadcast_in_dim %3285, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3290 = stablehlo.pad %3289, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3291 = stablehlo.broadcast_in_dim %3288, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3292 = stablehlo.pad %3291, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3293 = stablehlo.add %3290, %3292 : tensor<2x256xf32>
    %3294 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3295 = stablehlo.divide %3293, %3294 : tensor<2x256xf32>
    %3296 = "stablehlo.all_reduce"(%3295) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3297 = stablehlo.slice %3296 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3298 = stablehlo.slice %3296 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_192 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3299 = stablehlo.reduce(%3298 init: %cst_192) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_193 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3300 = stablehlo.reduce(%3297 init: %cst_193) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3301 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3302 = stablehlo.divide %3299, %3301 : tensor<256xf32>
    %3303 = stablehlo.broadcast_in_dim %3302, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3304 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3305 = stablehlo.divide %3300, %3304 : tensor<256xf32>
    %3306 = stablehlo.broadcast_in_dim %3305, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3307 = stablehlo.multiply %3303, %2022 : tensor<256x14x14x256xf32>
    %3308 = stablehlo.add %3306, %3307 : tensor<256x14x14x256xf32>
    %3309 = stablehlo.convert %3308 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3310 = stablehlo.add %3282, %3309 : tensor<256x14x14x256xbf16>
    %3311 = stablehlo.convolution(%2013, %3310) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3312 = stablehlo.reverse %2017, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3313 = stablehlo.convolution(%3310, %3312) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3314 = stablehlo.convert %3311 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3315 = stablehlo.add %2857, %3314 : tensor<3x3x256x256xf32>
    %3316 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3317 = stablehlo.select %2015, %3313, %3316 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3318 = stablehlo.convert %3317 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3319 = stablehlo.reduce(%3318 init: %cst_194) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3320 = stablehlo.reshape %3319 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3321 = stablehlo.reshape %3320 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3322 = stablehlo.multiply %1998, %3318 : tensor<256x14x14x256xf32>
    %cst_195 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3323 = stablehlo.reduce(%3322 init: %cst_195) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3324 = stablehlo.reshape %3323 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3325 = stablehlo.broadcast_in_dim %2006, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3326 = stablehlo.multiply %3318, %3325 : tensor<256x14x14x256xf32>
    %3327 = stablehlo.multiply %2001, %3324 : tensor<1x1x1x256xf32>
    %3328 = stablehlo.multiply %3324, %2005 : tensor<1x1x1x256xf32>
    %3329 = stablehlo.reshape %3327 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3330 = stablehlo.multiply %3328, %2004 : tensor<1x1x1x256xf32>
    %3331 = stablehlo.negate %3326 : tensor<256x14x14x256xf32>
    %cst_196 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3332 = stablehlo.reduce(%3331 init: %cst_196) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3333 = stablehlo.reshape %3332 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3334 = stablehlo.convert %3326 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3335 = stablehlo.reduce(%3330 init: %cst_197) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_198 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3336 = stablehlo.reduce(%3333 init: %cst_198) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3337 = stablehlo.multiply %3335, %1983 : tensor<256xf32>
    %3338 = stablehlo.negate %3337 : tensor<256xf32>
    %3339 = stablehlo.multiply %3338, %1970 : tensor<256xf32>
    %3340 = stablehlo.add %3336, %3339 : tensor<256xf32>
    %3341 = stablehlo.broadcast_in_dim %3337, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3342 = stablehlo.pad %3341, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3343 = stablehlo.broadcast_in_dim %3340, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3344 = stablehlo.pad %3343, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3345 = stablehlo.add %3342, %3344 : tensor<2x256xf32>
    %3346 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3347 = stablehlo.divide %3345, %3346 : tensor<2x256xf32>
    %3348 = "stablehlo.all_reduce"(%3347) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3349 = stablehlo.slice %3348 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3350 = stablehlo.slice %3348 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3351 = stablehlo.reduce(%3350 init: %cst_199) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_200 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3352 = stablehlo.reduce(%3349 init: %cst_200) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3353 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3354 = stablehlo.divide %3351, %3353 : tensor<256xf32>
    %3355 = stablehlo.broadcast_in_dim %3354, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3356 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3357 = stablehlo.divide %3352, %3356 : tensor<256xf32>
    %3358 = stablehlo.broadcast_in_dim %3357, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3359 = stablehlo.multiply %3355, %1951 : tensor<256x14x14x256xf32>
    %3360 = stablehlo.add %3358, %3359 : tensor<256x14x14x256xf32>
    %3361 = stablehlo.convert %3360 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3362 = stablehlo.add %3334, %3361 : tensor<256x14x14x256xbf16>
    %3363 = stablehlo.convolution(%1942, %3362) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3364 = stablehlo.reverse %1946, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3365 = stablehlo.convolution(%3362, %3364) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3366 = stablehlo.add %3265, %3365 : tensor<256x14x14x256xbf16>
    %3367 = stablehlo.convert %3363 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3368 = stablehlo.add %2859, %3367 : tensor<3x3x256x256xf32>
    %3369 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3370 = stablehlo.select %1944, %3366, %3369 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3371 = stablehlo.convert %3370 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3372 = stablehlo.reduce(%3371 init: %cst_201) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3373 = stablehlo.reshape %3372 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3374 = stablehlo.reshape %3373 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3375 = stablehlo.multiply %1926, %3371 : tensor<256x14x14x256xf32>
    %cst_202 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3376 = stablehlo.reduce(%3375 init: %cst_202) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3377 = stablehlo.reshape %3376 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3378 = stablehlo.broadcast_in_dim %1934, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3379 = stablehlo.multiply %3371, %3378 : tensor<256x14x14x256xf32>
    %3380 = stablehlo.multiply %1929, %3377 : tensor<1x1x1x256xf32>
    %3381 = stablehlo.multiply %3377, %1933 : tensor<1x1x1x256xf32>
    %3382 = stablehlo.reshape %3380 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3383 = stablehlo.multiply %3381, %1932 : tensor<1x1x1x256xf32>
    %3384 = stablehlo.negate %3379 : tensor<256x14x14x256xf32>
    %cst_203 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3385 = stablehlo.reduce(%3384 init: %cst_203) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3386 = stablehlo.reshape %3385 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3387 = stablehlo.convert %3379 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_204 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3388 = stablehlo.reduce(%3383 init: %cst_204) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_205 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3389 = stablehlo.reduce(%3386 init: %cst_205) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3390 = stablehlo.multiply %3388, %1911 : tensor<256xf32>
    %3391 = stablehlo.negate %3390 : tensor<256xf32>
    %3392 = stablehlo.multiply %3391, %1898 : tensor<256xf32>
    %3393 = stablehlo.add %3389, %3392 : tensor<256xf32>
    %3394 = stablehlo.broadcast_in_dim %3390, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3395 = stablehlo.pad %3394, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3396 = stablehlo.broadcast_in_dim %3393, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3397 = stablehlo.pad %3396, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3398 = stablehlo.add %3395, %3397 : tensor<2x256xf32>
    %3399 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3400 = stablehlo.divide %3398, %3399 : tensor<2x256xf32>
    %3401 = "stablehlo.all_reduce"(%3400) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3402 = stablehlo.slice %3401 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3403 = stablehlo.slice %3401 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_206 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3404 = stablehlo.reduce(%3403 init: %cst_206) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_207 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3405 = stablehlo.reduce(%3402 init: %cst_207) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3406 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3407 = stablehlo.divide %3404, %3406 : tensor<256xf32>
    %3408 = stablehlo.broadcast_in_dim %3407, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3409 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3410 = stablehlo.divide %3405, %3409 : tensor<256xf32>
    %3411 = stablehlo.broadcast_in_dim %3410, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3412 = stablehlo.multiply %3408, %1879 : tensor<256x14x14x256xf32>
    %3413 = stablehlo.add %3411, %3412 : tensor<256x14x14x256xf32>
    %3414 = stablehlo.convert %3413 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3415 = stablehlo.add %3387, %3414 : tensor<256x14x14x256xbf16>
    %3416 = stablehlo.convolution(%1870, %3415) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3417 = stablehlo.reverse %1874, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3418 = stablehlo.convolution(%3415, %3417) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3419 = stablehlo.convert %3416 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3420 = stablehlo.add %2861, %3419 : tensor<3x3x256x256xf32>
    %3421 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3422 = stablehlo.select %1872, %3418, %3421 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3423 = stablehlo.convert %3422 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_208 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3424 = stablehlo.reduce(%3423 init: %cst_208) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3425 = stablehlo.reshape %3424 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3426 = stablehlo.reshape %3425 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3427 = stablehlo.multiply %1855, %3423 : tensor<256x14x14x256xf32>
    %cst_209 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3428 = stablehlo.reduce(%3427 init: %cst_209) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3429 = stablehlo.reshape %3428 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3430 = stablehlo.broadcast_in_dim %1863, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3431 = stablehlo.multiply %3423, %3430 : tensor<256x14x14x256xf32>
    %3432 = stablehlo.multiply %1858, %3429 : tensor<1x1x1x256xf32>
    %3433 = stablehlo.multiply %3429, %1862 : tensor<1x1x1x256xf32>
    %3434 = stablehlo.reshape %3432 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3435 = stablehlo.multiply %3433, %1861 : tensor<1x1x1x256xf32>
    %3436 = stablehlo.negate %3431 : tensor<256x14x14x256xf32>
    %cst_210 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3437 = stablehlo.reduce(%3436 init: %cst_210) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3438 = stablehlo.reshape %3437 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3439 = stablehlo.convert %3431 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_211 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3440 = stablehlo.reduce(%3435 init: %cst_211) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_212 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3441 = stablehlo.reduce(%3438 init: %cst_212) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3442 = stablehlo.multiply %3440, %1840 : tensor<256xf32>
    %3443 = stablehlo.negate %3442 : tensor<256xf32>
    %3444 = stablehlo.multiply %3443, %1827 : tensor<256xf32>
    %3445 = stablehlo.add %3441, %3444 : tensor<256xf32>
    %3446 = stablehlo.broadcast_in_dim %3442, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3447 = stablehlo.pad %3446, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3448 = stablehlo.broadcast_in_dim %3445, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3449 = stablehlo.pad %3448, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3450 = stablehlo.add %3447, %3449 : tensor<2x256xf32>
    %3451 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3452 = stablehlo.divide %3450, %3451 : tensor<2x256xf32>
    %3453 = "stablehlo.all_reduce"(%3452) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3454 = stablehlo.slice %3453 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3455 = stablehlo.slice %3453 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_213 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3456 = stablehlo.reduce(%3455 init: %cst_213) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_214 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3457 = stablehlo.reduce(%3454 init: %cst_214) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3458 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3459 = stablehlo.divide %3456, %3458 : tensor<256xf32>
    %3460 = stablehlo.broadcast_in_dim %3459, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3461 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3462 = stablehlo.divide %3457, %3461 : tensor<256xf32>
    %3463 = stablehlo.broadcast_in_dim %3462, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3464 = stablehlo.multiply %3460, %1808 : tensor<256x14x14x256xf32>
    %3465 = stablehlo.add %3463, %3464 : tensor<256x14x14x256xf32>
    %3466 = stablehlo.convert %3465 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3467 = stablehlo.add %3439, %3466 : tensor<256x14x14x256xbf16>
    %3468 = stablehlo.convolution(%1799, %3467) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3469 = stablehlo.reverse %1803, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3470 = stablehlo.convolution(%3467, %3469) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3471 = stablehlo.add %3370, %3470 : tensor<256x14x14x256xbf16>
    %3472 = stablehlo.convert %3468 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3473 = stablehlo.add %2863, %3472 : tensor<3x3x256x256xf32>
    %3474 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3475 = stablehlo.select %1801, %3471, %3474 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3476 = stablehlo.convert %3475 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_215 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3477 = stablehlo.reduce(%3476 init: %cst_215) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3478 = stablehlo.reshape %3477 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3479 = stablehlo.reshape %3478 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3480 = stablehlo.multiply %1783, %3476 : tensor<256x14x14x256xf32>
    %cst_216 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3481 = stablehlo.reduce(%3480 init: %cst_216) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3482 = stablehlo.reshape %3481 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3483 = stablehlo.broadcast_in_dim %1791, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3484 = stablehlo.multiply %3476, %3483 : tensor<256x14x14x256xf32>
    %3485 = stablehlo.multiply %1786, %3482 : tensor<1x1x1x256xf32>
    %3486 = stablehlo.multiply %3482, %1790 : tensor<1x1x1x256xf32>
    %3487 = stablehlo.reshape %3485 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3488 = stablehlo.multiply %3486, %1789 : tensor<1x1x1x256xf32>
    %3489 = stablehlo.negate %3484 : tensor<256x14x14x256xf32>
    %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3490 = stablehlo.reduce(%3489 init: %cst_217) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3491 = stablehlo.reshape %3490 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3492 = stablehlo.convert %3484 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_218 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3493 = stablehlo.reduce(%3488 init: %cst_218) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_219 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3494 = stablehlo.reduce(%3491 init: %cst_219) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3495 = stablehlo.multiply %3493, %1768 : tensor<256xf32>
    %3496 = stablehlo.negate %3495 : tensor<256xf32>
    %3497 = stablehlo.multiply %3496, %1755 : tensor<256xf32>
    %3498 = stablehlo.add %3494, %3497 : tensor<256xf32>
    %3499 = stablehlo.broadcast_in_dim %3495, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3500 = stablehlo.pad %3499, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3501 = stablehlo.broadcast_in_dim %3498, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3502 = stablehlo.pad %3501, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3503 = stablehlo.add %3500, %3502 : tensor<2x256xf32>
    %3504 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3505 = stablehlo.divide %3503, %3504 : tensor<2x256xf32>
    %3506 = "stablehlo.all_reduce"(%3505) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3507 = stablehlo.slice %3506 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3508 = stablehlo.slice %3506 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_220 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3509 = stablehlo.reduce(%3508 init: %cst_220) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_221 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3510 = stablehlo.reduce(%3507 init: %cst_221) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3511 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3512 = stablehlo.divide %3509, %3511 : tensor<256xf32>
    %3513 = stablehlo.broadcast_in_dim %3512, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3514 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3515 = stablehlo.divide %3510, %3514 : tensor<256xf32>
    %3516 = stablehlo.broadcast_in_dim %3515, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3517 = stablehlo.multiply %3513, %1736 : tensor<256x14x14x256xf32>
    %3518 = stablehlo.add %3516, %3517 : tensor<256x14x14x256xf32>
    %3519 = stablehlo.convert %3518 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3520 = stablehlo.add %3492, %3519 : tensor<256x14x14x256xbf16>
    %3521 = stablehlo.convolution(%1727, %3520) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3522 = stablehlo.reverse %1731, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3523 = stablehlo.convolution(%3520, %3522) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3524 = stablehlo.convert %3521 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3525 = stablehlo.add %2865, %3524 : tensor<3x3x256x256xf32>
    %3526 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3527 = stablehlo.select %1729, %3523, %3526 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3528 = stablehlo.convert %3527 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_222 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3529 = stablehlo.reduce(%3528 init: %cst_222) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3530 = stablehlo.reshape %3529 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3531 = stablehlo.reshape %3530 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3532 = stablehlo.multiply %1712, %3528 : tensor<256x14x14x256xf32>
    %cst_223 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3533 = stablehlo.reduce(%3532 init: %cst_223) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3534 = stablehlo.reshape %3533 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3535 = stablehlo.broadcast_in_dim %1720, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3536 = stablehlo.multiply %3528, %3535 : tensor<256x14x14x256xf32>
    %3537 = stablehlo.multiply %1715, %3534 : tensor<1x1x1x256xf32>
    %3538 = stablehlo.multiply %3534, %1719 : tensor<1x1x1x256xf32>
    %3539 = stablehlo.reshape %3537 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3540 = stablehlo.multiply %3538, %1718 : tensor<1x1x1x256xf32>
    %3541 = stablehlo.negate %3536 : tensor<256x14x14x256xf32>
    %cst_224 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3542 = stablehlo.reduce(%3541 init: %cst_224) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3543 = stablehlo.reshape %3542 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3544 = stablehlo.convert %3536 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_225 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3545 = stablehlo.reduce(%3540 init: %cst_225) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3546 = stablehlo.reduce(%3543 init: %cst_226) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3547 = stablehlo.multiply %3545, %1697 : tensor<256xf32>
    %3548 = stablehlo.negate %3547 : tensor<256xf32>
    %3549 = stablehlo.multiply %3548, %1684 : tensor<256xf32>
    %3550 = stablehlo.add %3546, %3549 : tensor<256xf32>
    %3551 = stablehlo.broadcast_in_dim %3547, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3552 = stablehlo.pad %3551, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3553 = stablehlo.broadcast_in_dim %3550, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3554 = stablehlo.pad %3553, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3555 = stablehlo.add %3552, %3554 : tensor<2x256xf32>
    %3556 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3557 = stablehlo.divide %3555, %3556 : tensor<2x256xf32>
    %3558 = "stablehlo.all_reduce"(%3557) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3559 = stablehlo.slice %3558 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3560 = stablehlo.slice %3558 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_227 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3561 = stablehlo.reduce(%3560 init: %cst_227) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_228 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3562 = stablehlo.reduce(%3559 init: %cst_228) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3563 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3564 = stablehlo.divide %3561, %3563 : tensor<256xf32>
    %3565 = stablehlo.broadcast_in_dim %3564, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3566 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3567 = stablehlo.divide %3562, %3566 : tensor<256xf32>
    %3568 = stablehlo.broadcast_in_dim %3567, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3569 = stablehlo.multiply %3565, %1665 : tensor<256x14x14x256xf32>
    %3570 = stablehlo.add %3568, %3569 : tensor<256x14x14x256xf32>
    %3571 = stablehlo.convert %3570 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3572 = stablehlo.add %3544, %3571 : tensor<256x14x14x256xbf16>
    %3573 = stablehlo.convolution(%1656, %3572) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3574 = stablehlo.reverse %1660, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3575 = stablehlo.convolution(%3572, %3574) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3576 = stablehlo.add %3475, %3575 : tensor<256x14x14x256xbf16>
    %3577 = stablehlo.convert %3573 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3578 = stablehlo.add %2867, %3577 : tensor<3x3x256x256xf32>
    %3579 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3580 = stablehlo.select %1658, %3576, %3579 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3581 = stablehlo.convert %3580 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_229 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3582 = stablehlo.reduce(%3581 init: %cst_229) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3583 = stablehlo.reshape %3582 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3584 = stablehlo.reshape %3583 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3585 = stablehlo.multiply %1640, %3581 : tensor<256x14x14x256xf32>
    %cst_230 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3586 = stablehlo.reduce(%3585 init: %cst_230) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3587 = stablehlo.reshape %3586 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3588 = stablehlo.broadcast_in_dim %1648, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3589 = stablehlo.multiply %3581, %3588 : tensor<256x14x14x256xf32>
    %3590 = stablehlo.multiply %1643, %3587 : tensor<1x1x1x256xf32>
    %3591 = stablehlo.multiply %3587, %1647 : tensor<1x1x1x256xf32>
    %3592 = stablehlo.reshape %3590 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3593 = stablehlo.multiply %3591, %1646 : tensor<1x1x1x256xf32>
    %3594 = stablehlo.negate %3589 : tensor<256x14x14x256xf32>
    %cst_231 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3595 = stablehlo.reduce(%3594 init: %cst_231) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3596 = stablehlo.reshape %3595 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3597 = stablehlo.convert %3589 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_232 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3598 = stablehlo.reduce(%3593 init: %cst_232) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_233 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3599 = stablehlo.reduce(%3596 init: %cst_233) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3600 = stablehlo.multiply %3598, %1625 : tensor<256xf32>
    %3601 = stablehlo.negate %3600 : tensor<256xf32>
    %3602 = stablehlo.multiply %3601, %1612 : tensor<256xf32>
    %3603 = stablehlo.add %3599, %3602 : tensor<256xf32>
    %3604 = stablehlo.broadcast_in_dim %3600, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3605 = stablehlo.pad %3604, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3606 = stablehlo.broadcast_in_dim %3603, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3607 = stablehlo.pad %3606, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3608 = stablehlo.add %3605, %3607 : tensor<2x256xf32>
    %3609 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3610 = stablehlo.divide %3608, %3609 : tensor<2x256xf32>
    %3611 = "stablehlo.all_reduce"(%3610) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3612 = stablehlo.slice %3611 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3613 = stablehlo.slice %3611 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_234 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3614 = stablehlo.reduce(%3613 init: %cst_234) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_235 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3615 = stablehlo.reduce(%3612 init: %cst_235) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3616 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3617 = stablehlo.divide %3614, %3616 : tensor<256xf32>
    %3618 = stablehlo.broadcast_in_dim %3617, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3619 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3620 = stablehlo.divide %3615, %3619 : tensor<256xf32>
    %3621 = stablehlo.broadcast_in_dim %3620, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3622 = stablehlo.multiply %3618, %1593 : tensor<256x14x14x256xf32>
    %3623 = stablehlo.add %3621, %3622 : tensor<256x14x14x256xf32>
    %3624 = stablehlo.convert %3623 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3625 = stablehlo.add %3597, %3624 : tensor<256x14x14x256xbf16>
    %3626 = stablehlo.convolution(%1584, %3625) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3627 = stablehlo.reverse %1588, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3628 = stablehlo.convolution(%3625, %3627) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3629 = stablehlo.convert %3626 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3630 = stablehlo.add %2807, %3629 : tensor<3x3x256x256xf32>
    %3631 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3632 = stablehlo.select %1586, %3628, %3631 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3633 = stablehlo.convert %3632 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_236 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3634 = stablehlo.reduce(%3633 init: %cst_236) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3635 = stablehlo.reshape %3634 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3636 = stablehlo.reshape %3635 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3637 = stablehlo.multiply %1569, %3633 : tensor<256x14x14x256xf32>
    %cst_237 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3638 = stablehlo.reduce(%3637 init: %cst_237) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3639 = stablehlo.reshape %3638 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3640 = stablehlo.broadcast_in_dim %1577, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3641 = stablehlo.multiply %3633, %3640 : tensor<256x14x14x256xf32>
    %3642 = stablehlo.multiply %1572, %3639 : tensor<1x1x1x256xf32>
    %3643 = stablehlo.multiply %3639, %1576 : tensor<1x1x1x256xf32>
    %3644 = stablehlo.reshape %3642 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3645 = stablehlo.multiply %3643, %1575 : tensor<1x1x1x256xf32>
    %3646 = stablehlo.negate %3641 : tensor<256x14x14x256xf32>
    %cst_238 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3647 = stablehlo.reduce(%3646 init: %cst_238) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3648 = stablehlo.reshape %3647 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3649 = stablehlo.convert %3641 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_239 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3650 = stablehlo.reduce(%3645 init: %cst_239) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_240 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3651 = stablehlo.reduce(%3648 init: %cst_240) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3652 = stablehlo.multiply %3650, %1554 : tensor<256xf32>
    %3653 = stablehlo.negate %3652 : tensor<256xf32>
    %3654 = stablehlo.multiply %3653, %1541 : tensor<256xf32>
    %3655 = stablehlo.add %3651, %3654 : tensor<256xf32>
    %3656 = stablehlo.broadcast_in_dim %3652, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3657 = stablehlo.pad %3656, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3658 = stablehlo.broadcast_in_dim %3655, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3659 = stablehlo.pad %3658, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3660 = stablehlo.add %3657, %3659 : tensor<2x256xf32>
    %3661 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3662 = stablehlo.divide %3660, %3661 : tensor<2x256xf32>
    %3663 = "stablehlo.all_reduce"(%3662) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3664 = stablehlo.slice %3663 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3665 = stablehlo.slice %3663 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_241 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3666 = stablehlo.reduce(%3665 init: %cst_241) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_242 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3667 = stablehlo.reduce(%3664 init: %cst_242) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3668 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3669 = stablehlo.divide %3666, %3668 : tensor<256xf32>
    %3670 = stablehlo.broadcast_in_dim %3669, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3671 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3672 = stablehlo.divide %3667, %3671 : tensor<256xf32>
    %3673 = stablehlo.broadcast_in_dim %3672, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3674 = stablehlo.multiply %3670, %1522 : tensor<256x14x14x256xf32>
    %3675 = stablehlo.add %3673, %3674 : tensor<256x14x14x256xf32>
    %3676 = stablehlo.convert %3675 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3677 = stablehlo.add %3649, %3676 : tensor<256x14x14x256xbf16>
    %3678 = stablehlo.convolution(%1513, %3677) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3679 = stablehlo.reverse %1517, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3680 = stablehlo.convolution(%3677, %3679) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3681 = stablehlo.add %3580, %3680 : tensor<256x14x14x256xbf16>
    %3682 = stablehlo.convert %3678 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3683 = stablehlo.add %2809, %3682 : tensor<3x3x256x256xf32>
    %3684 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3685 = stablehlo.select %1515, %3681, %3684 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3686 = stablehlo.convert %3685 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_243 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3687 = stablehlo.reduce(%3686 init: %cst_243) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3688 = stablehlo.reshape %3687 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3689 = stablehlo.reshape %3688 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3690 = stablehlo.multiply %1497, %3686 : tensor<256x14x14x256xf32>
    %cst_244 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3691 = stablehlo.reduce(%3690 init: %cst_244) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3692 = stablehlo.reshape %3691 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3693 = stablehlo.broadcast_in_dim %1505, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3694 = stablehlo.multiply %3686, %3693 : tensor<256x14x14x256xf32>
    %3695 = stablehlo.multiply %1500, %3692 : tensor<1x1x1x256xf32>
    %3696 = stablehlo.multiply %3692, %1504 : tensor<1x1x1x256xf32>
    %3697 = stablehlo.reshape %3695 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3698 = stablehlo.multiply %3696, %1503 : tensor<1x1x1x256xf32>
    %3699 = stablehlo.negate %3694 : tensor<256x14x14x256xf32>
    %cst_245 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3700 = stablehlo.reduce(%3699 init: %cst_245) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3701 = stablehlo.reshape %3700 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3702 = stablehlo.convert %3694 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_246 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3703 = stablehlo.reduce(%3698 init: %cst_246) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_247 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3704 = stablehlo.reduce(%3701 init: %cst_247) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3705 = stablehlo.multiply %3703, %1482 : tensor<256xf32>
    %3706 = stablehlo.negate %3705 : tensor<256xf32>
    %3707 = stablehlo.multiply %3706, %1469 : tensor<256xf32>
    %3708 = stablehlo.add %3704, %3707 : tensor<256xf32>
    %3709 = stablehlo.broadcast_in_dim %3705, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3710 = stablehlo.pad %3709, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3711 = stablehlo.broadcast_in_dim %3708, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3712 = stablehlo.pad %3711, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3713 = stablehlo.add %3710, %3712 : tensor<2x256xf32>
    %3714 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3715 = stablehlo.divide %3713, %3714 : tensor<2x256xf32>
    %3716 = "stablehlo.all_reduce"(%3715) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3717 = stablehlo.slice %3716 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3718 = stablehlo.slice %3716 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_248 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3719 = stablehlo.reduce(%3718 init: %cst_248) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_249 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3720 = stablehlo.reduce(%3717 init: %cst_249) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3721 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3722 = stablehlo.divide %3719, %3721 : tensor<256xf32>
    %3723 = stablehlo.broadcast_in_dim %3722, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3724 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3725 = stablehlo.divide %3720, %3724 : tensor<256xf32>
    %3726 = stablehlo.broadcast_in_dim %3725, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3727 = stablehlo.multiply %3723, %1450 : tensor<256x14x14x256xf32>
    %3728 = stablehlo.add %3726, %3727 : tensor<256x14x14x256xf32>
    %3729 = stablehlo.convert %3728 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3730 = stablehlo.add %3702, %3729 : tensor<256x14x14x256xbf16>
    %3731 = stablehlo.convolution(%1441, %3730) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3732 = stablehlo.reverse %1445, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3733 = stablehlo.convolution(%3730, %3732) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3734 = stablehlo.convert %3731 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3735 = stablehlo.add %2811, %3734 : tensor<3x3x256x256xf32>
    %3736 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3737 = stablehlo.select %1443, %3733, %3736 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3738 = stablehlo.convert %3737 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_250 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3739 = stablehlo.reduce(%3738 init: %cst_250) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3740 = stablehlo.reshape %3739 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3741 = stablehlo.reshape %3740 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3742 = stablehlo.multiply %1426, %3738 : tensor<256x14x14x256xf32>
    %cst_251 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3743 = stablehlo.reduce(%3742 init: %cst_251) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3744 = stablehlo.reshape %3743 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3745 = stablehlo.broadcast_in_dim %1434, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3746 = stablehlo.multiply %3738, %3745 : tensor<256x14x14x256xf32>
    %3747 = stablehlo.multiply %1429, %3744 : tensor<1x1x1x256xf32>
    %3748 = stablehlo.multiply %3744, %1433 : tensor<1x1x1x256xf32>
    %3749 = stablehlo.reshape %3747 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3750 = stablehlo.multiply %3748, %1432 : tensor<1x1x1x256xf32>
    %3751 = stablehlo.negate %3746 : tensor<256x14x14x256xf32>
    %cst_252 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3752 = stablehlo.reduce(%3751 init: %cst_252) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3753 = stablehlo.reshape %3752 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3754 = stablehlo.convert %3746 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_253 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3755 = stablehlo.reduce(%3750 init: %cst_253) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_254 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3756 = stablehlo.reduce(%3753 init: %cst_254) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3757 = stablehlo.multiply %3755, %1411 : tensor<256xf32>
    %3758 = stablehlo.negate %3757 : tensor<256xf32>
    %3759 = stablehlo.multiply %3758, %1398 : tensor<256xf32>
    %3760 = stablehlo.add %3756, %3759 : tensor<256xf32>
    %3761 = stablehlo.broadcast_in_dim %3757, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3762 = stablehlo.pad %3761, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3763 = stablehlo.broadcast_in_dim %3760, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3764 = stablehlo.pad %3763, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3765 = stablehlo.add %3762, %3764 : tensor<2x256xf32>
    %3766 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3767 = stablehlo.divide %3765, %3766 : tensor<2x256xf32>
    %3768 = "stablehlo.all_reduce"(%3767) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3769 = stablehlo.slice %3768 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3770 = stablehlo.slice %3768 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_255 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3771 = stablehlo.reduce(%3770 init: %cst_255) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_256 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3772 = stablehlo.reduce(%3769 init: %cst_256) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3773 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3774 = stablehlo.divide %3771, %3773 : tensor<256xf32>
    %3775 = stablehlo.broadcast_in_dim %3774, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3776 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3777 = stablehlo.divide %3772, %3776 : tensor<256xf32>
    %3778 = stablehlo.broadcast_in_dim %3777, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3779 = stablehlo.multiply %3775, %1379 : tensor<256x14x14x256xf32>
    %3780 = stablehlo.add %3778, %3779 : tensor<256x14x14x256xf32>
    %3781 = stablehlo.convert %3780 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3782 = stablehlo.add %3754, %3781 : tensor<256x14x14x256xbf16>
    %3783 = stablehlo.convolution(%1370, %3782) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3784 = stablehlo.reverse %1374, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3785 = stablehlo.convolution(%3782, %3784) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3786 = stablehlo.add %3685, %3785 : tensor<256x14x14x256xbf16>
    %3787 = stablehlo.convert %3783 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3788 = stablehlo.add %2813, %3787 : tensor<3x3x256x256xf32>
    %3789 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3790 = stablehlo.select %1372, %3786, %3789 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3791 = stablehlo.convert %3790 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_257 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3792 = stablehlo.reduce(%3791 init: %cst_257) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3793 = stablehlo.reshape %3792 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3794 = stablehlo.reshape %3793 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3795 = stablehlo.multiply %1354, %3791 : tensor<256x14x14x256xf32>
    %cst_258 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3796 = stablehlo.reduce(%3795 init: %cst_258) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3797 = stablehlo.reshape %3796 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3798 = stablehlo.broadcast_in_dim %1362, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3799 = stablehlo.multiply %3791, %3798 : tensor<256x14x14x256xf32>
    %3800 = stablehlo.multiply %1357, %3797 : tensor<1x1x1x256xf32>
    %3801 = stablehlo.multiply %3797, %1361 : tensor<1x1x1x256xf32>
    %3802 = stablehlo.reshape %3800 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3803 = stablehlo.multiply %3801, %1360 : tensor<1x1x1x256xf32>
    %3804 = stablehlo.negate %3799 : tensor<256x14x14x256xf32>
    %cst_259 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3805 = stablehlo.reduce(%3804 init: %cst_259) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3806 = stablehlo.reshape %3805 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3807 = stablehlo.convert %3799 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_260 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3808 = stablehlo.reduce(%3803 init: %cst_260) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_261 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3809 = stablehlo.reduce(%3806 init: %cst_261) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3810 = stablehlo.multiply %3808, %1339 : tensor<256xf32>
    %3811 = stablehlo.negate %3810 : tensor<256xf32>
    %3812 = stablehlo.multiply %3811, %1326 : tensor<256xf32>
    %3813 = stablehlo.add %3809, %3812 : tensor<256xf32>
    %3814 = stablehlo.broadcast_in_dim %3810, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3815 = stablehlo.pad %3814, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3816 = stablehlo.broadcast_in_dim %3813, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3817 = stablehlo.pad %3816, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3818 = stablehlo.add %3815, %3817 : tensor<2x256xf32>
    %3819 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3820 = stablehlo.divide %3818, %3819 : tensor<2x256xf32>
    %3821 = "stablehlo.all_reduce"(%3820) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3822 = stablehlo.slice %3821 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3823 = stablehlo.slice %3821 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_262 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3824 = stablehlo.reduce(%3823 init: %cst_262) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_263 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3825 = stablehlo.reduce(%3822 init: %cst_263) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3826 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3827 = stablehlo.divide %3824, %3826 : tensor<256xf32>
    %3828 = stablehlo.broadcast_in_dim %3827, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3829 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3830 = stablehlo.divide %3825, %3829 : tensor<256xf32>
    %3831 = stablehlo.broadcast_in_dim %3830, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3832 = stablehlo.multiply %3828, %1307 : tensor<256x14x14x256xf32>
    %3833 = stablehlo.add %3831, %3832 : tensor<256x14x14x256xf32>
    %3834 = stablehlo.convert %3833 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3835 = stablehlo.add %3807, %3834 : tensor<256x14x14x256xbf16>
    %3836 = stablehlo.convolution(%1160, %3835) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x14x14x256xbf16>) -> tensor<1x1x128x256xbf16>
    %3837 = stablehlo.reverse %1302, dims = [0, 1] : tensor<1x1x128x256xbf16>
    %3838 = stablehlo.convolution(%3835, %3837) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<1x1x128x256xbf16>) -> tensor<256x28x28x128xbf16>
    %3839 = stablehlo.convert %3836 : (tensor<1x1x128x256xbf16>) -> tensor<1x1x128x256xf32>
    %3840 = stablehlo.add %2815, %3839 : tensor<1x1x128x256xf32>
    %3841 = stablehlo.convert %3790 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_264 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3842 = stablehlo.reduce(%3841 init: %cst_264) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3843 = stablehlo.reshape %3842 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3844 = stablehlo.reshape %3843 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3845 = stablehlo.multiply %1287, %3841 : tensor<256x14x14x256xf32>
    %cst_265 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3846 = stablehlo.reduce(%3845 init: %cst_265) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3847 = stablehlo.reshape %3846 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3848 = stablehlo.broadcast_in_dim %1295, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3849 = stablehlo.multiply %3841, %3848 : tensor<256x14x14x256xf32>
    %3850 = stablehlo.multiply %1290, %3847 : tensor<1x1x1x256xf32>
    %3851 = stablehlo.multiply %3847, %1294 : tensor<1x1x1x256xf32>
    %3852 = stablehlo.reshape %3850 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3853 = stablehlo.multiply %3851, %1293 : tensor<1x1x1x256xf32>
    %3854 = stablehlo.negate %3849 : tensor<256x14x14x256xf32>
    %cst_266 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3855 = stablehlo.reduce(%3854 init: %cst_266) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3856 = stablehlo.reshape %3855 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3857 = stablehlo.convert %3849 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_267 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3858 = stablehlo.reduce(%3853 init: %cst_267) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_268 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3859 = stablehlo.reduce(%3856 init: %cst_268) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3860 = stablehlo.multiply %3858, %1272 : tensor<256xf32>
    %3861 = stablehlo.negate %3860 : tensor<256xf32>
    %3862 = stablehlo.multiply %3861, %1259 : tensor<256xf32>
    %3863 = stablehlo.add %3859, %3862 : tensor<256xf32>
    %3864 = stablehlo.broadcast_in_dim %3860, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3865 = stablehlo.pad %3864, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3866 = stablehlo.broadcast_in_dim %3863, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3867 = stablehlo.pad %3866, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3868 = stablehlo.add %3865, %3867 : tensor<2x256xf32>
    %3869 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3870 = stablehlo.divide %3868, %3869 : tensor<2x256xf32>
    %3871 = "stablehlo.all_reduce"(%3870) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3872 = stablehlo.slice %3871 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3873 = stablehlo.slice %3871 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_269 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3874 = stablehlo.reduce(%3873 init: %cst_269) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_270 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3875 = stablehlo.reduce(%3872 init: %cst_270) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3876 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3877 = stablehlo.divide %3874, %3876 : tensor<256xf32>
    %3878 = stablehlo.broadcast_in_dim %3877, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3879 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3880 = stablehlo.divide %3875, %3879 : tensor<256xf32>
    %3881 = stablehlo.broadcast_in_dim %3880, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3882 = stablehlo.multiply %3878, %1240 : tensor<256x14x14x256xf32>
    %3883 = stablehlo.add %3881, %3882 : tensor<256x14x14x256xf32>
    %3884 = stablehlo.convert %3883 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3885 = stablehlo.add %3857, %3884 : tensor<256x14x14x256xbf16>
    %3886 = stablehlo.convolution(%1231, %3885) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x256x256xbf16>
    %3887 = stablehlo.reverse %1235, dims = [0, 1] : tensor<3x3x256x256xbf16>
    %3888 = stablehlo.convolution(%3885, %3887) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x256x256xbf16>) -> tensor<256x14x14x256xbf16>
    %3889 = stablehlo.convert %3886 : (tensor<3x3x256x256xbf16>) -> tensor<3x3x256x256xf32>
    %3890 = stablehlo.add %2817, %3889 : tensor<3x3x256x256xf32>
    %3891 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %3892 = stablehlo.select %1233, %3888, %3891 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xbf16>
    %3893 = stablehlo.convert %3892 : (tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xf32>
    %cst_271 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3894 = stablehlo.reduce(%3893 init: %cst_271) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3895 = stablehlo.reshape %3894 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3896 = stablehlo.reshape %3895 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3897 = stablehlo.multiply %1216, %3893 : tensor<256x14x14x256xf32>
    %cst_272 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3898 = stablehlo.reduce(%3897 init: %cst_272) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3899 = stablehlo.reshape %3898 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3900 = stablehlo.broadcast_in_dim %1224, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
    %3901 = stablehlo.multiply %3893, %3900 : tensor<256x14x14x256xf32>
    %3902 = stablehlo.multiply %1219, %3899 : tensor<1x1x1x256xf32>
    %3903 = stablehlo.multiply %3899, %1223 : tensor<1x1x1x256xf32>
    %3904 = stablehlo.reshape %3902 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
    %3905 = stablehlo.multiply %3903, %1222 : tensor<1x1x1x256xf32>
    %3906 = stablehlo.negate %3901 : tensor<256x14x14x256xf32>
    %cst_273 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3907 = stablehlo.reduce(%3906 init: %cst_273) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3908 = stablehlo.reshape %3907 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %3909 = stablehlo.convert %3901 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %cst_274 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3910 = stablehlo.reduce(%3905 init: %cst_274) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_275 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3911 = stablehlo.reduce(%3908 init: %cst_275) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3912 = stablehlo.multiply %3910, %1201 : tensor<256xf32>
    %3913 = stablehlo.negate %3912 : tensor<256xf32>
    %3914 = stablehlo.multiply %3913, %1188 : tensor<256xf32>
    %3915 = stablehlo.add %3911, %3914 : tensor<256xf32>
    %3916 = stablehlo.broadcast_in_dim %3912, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3917 = stablehlo.pad %3916, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3918 = stablehlo.broadcast_in_dim %3915, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3919 = stablehlo.pad %3918, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
    %3920 = stablehlo.add %3917, %3919 : tensor<2x256xf32>
    %3921 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %3922 = stablehlo.divide %3920, %3921 : tensor<2x256xf32>
    %3923 = "stablehlo.all_reduce"(%3922) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
    %3924 = stablehlo.slice %3923 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3925 = stablehlo.slice %3923 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %cst_276 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3926 = stablehlo.reduce(%3925 init: %cst_276) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3927 = stablehlo.reduce(%3924 init: %cst_277) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
    %3928 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3929 = stablehlo.divide %3926, %3928 : tensor<256xf32>
    %3930 = stablehlo.broadcast_in_dim %3929, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3931 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3932 = stablehlo.divide %3927, %3931 : tensor<256xf32>
    %3933 = stablehlo.broadcast_in_dim %3932, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
    %3934 = stablehlo.multiply %3930, %1169 : tensor<256x14x14x256xf32>
    %3935 = stablehlo.add %3933, %3934 : tensor<256x14x14x256xf32>
    %3936 = stablehlo.convert %3935 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xbf16>
    %3937 = stablehlo.add %3909, %3936 : tensor<256x14x14x256xbf16>
    %3938 = stablehlo.convolution(%1160, %3937) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x14x14x256xbf16>) -> tensor<3x3x128x256xbf16>
    %3939 = stablehlo.reverse %1164, dims = [0, 1] : tensor<3x3x128x256xbf16>
    %3940 = stablehlo.convolution(%3937, %3939) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xbf16>, tensor<3x3x128x256xbf16>) -> tensor<256x28x28x128xbf16>
    %3941 = stablehlo.add %3838, %3940 : tensor<256x28x28x128xbf16>
    %3942 = stablehlo.convert %3938 : (tensor<3x3x128x256xbf16>) -> tensor<3x3x128x256xf32>
    %3943 = stablehlo.add %2819, %3942 : tensor<3x3x128x256xf32>
    %3944 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %3945 = stablehlo.select %1162, %3941, %3944 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %3946 = stablehlo.convert %3945 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_278 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3947 = stablehlo.reduce(%3946 init: %cst_278) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %3948 = stablehlo.reshape %3947 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %3949 = stablehlo.reshape %3948 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %3950 = stablehlo.multiply %1144, %3946 : tensor<256x28x28x128xf32>
    %cst_279 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3951 = stablehlo.reduce(%3950 init: %cst_279) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %3952 = stablehlo.reshape %3951 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %3953 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %3954 = stablehlo.multiply %3946, %3953 : tensor<256x28x28x128xf32>
    %3955 = stablehlo.multiply %1147, %3952 : tensor<1x1x1x128xf32>
    %3956 = stablehlo.multiply %3952, %1151 : tensor<1x1x1x128xf32>
    %3957 = stablehlo.reshape %3955 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %3958 = stablehlo.multiply %3956, %1150 : tensor<1x1x1x128xf32>
    %3959 = stablehlo.negate %3954 : tensor<256x28x28x128xf32>
    %cst_280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3960 = stablehlo.reduce(%3959 init: %cst_280) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %3961 = stablehlo.reshape %3960 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %3962 = stablehlo.convert %3954 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_281 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3963 = stablehlo.reduce(%3958 init: %cst_281) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_282 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3964 = stablehlo.reduce(%3961 init: %cst_282) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %3965 = stablehlo.multiply %3963, %1129 : tensor<128xf32>
    %3966 = stablehlo.negate %3965 : tensor<128xf32>
    %3967 = stablehlo.multiply %3966, %1116 : tensor<128xf32>
    %3968 = stablehlo.add %3964, %3967 : tensor<128xf32>
    %3969 = stablehlo.broadcast_in_dim %3965, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %3970 = stablehlo.pad %3969, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %3971 = stablehlo.broadcast_in_dim %3968, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %3972 = stablehlo.pad %3971, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %3973 = stablehlo.add %3970, %3972 : tensor<2x128xf32>
    %3974 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %3975 = stablehlo.divide %3973, %3974 : tensor<2x128xf32>
    %3976 = "stablehlo.all_reduce"(%3975) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %3977 = stablehlo.slice %3976 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %3978 = stablehlo.slice %3976 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_283 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3979 = stablehlo.reduce(%3978 init: %cst_283) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_284 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3980 = stablehlo.reduce(%3977 init: %cst_284) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %3981 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3982 = stablehlo.divide %3979, %3981 : tensor<128xf32>
    %3983 = stablehlo.broadcast_in_dim %3982, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3984 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3985 = stablehlo.divide %3980, %3984 : tensor<128xf32>
    %3986 = stablehlo.broadcast_in_dim %3985, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %3987 = stablehlo.multiply %3983, %1097 : tensor<256x28x28x128xf32>
    %3988 = stablehlo.add %3986, %3987 : tensor<256x28x28x128xf32>
    %3989 = stablehlo.convert %3988 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %3990 = stablehlo.add %3962, %3989 : tensor<256x28x28x128xbf16>
    %3991 = stablehlo.convolution(%1088, %3990) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %3992 = stablehlo.reverse %1092, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %3993 = stablehlo.convolution(%3990, %3992) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %3994 = stablehlo.convert %3991 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %3995 = stablehlo.add %2821, %3994 : tensor<3x3x128x128xf32>
    %3996 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %3997 = stablehlo.select %1090, %3993, %3996 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %3998 = stablehlo.convert %3997 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_285 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3999 = stablehlo.reduce(%3998 init: %cst_285) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4000 = stablehlo.reshape %3999 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4001 = stablehlo.reshape %4000 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4002 = stablehlo.multiply %1073, %3998 : tensor<256x28x28x128xf32>
    %cst_286 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4003 = stablehlo.reduce(%4002 init: %cst_286) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4004 = stablehlo.reshape %4003 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4005 = stablehlo.broadcast_in_dim %1081, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %4006 = stablehlo.multiply %3998, %4005 : tensor<256x28x28x128xf32>
    %4007 = stablehlo.multiply %1076, %4004 : tensor<1x1x1x128xf32>
    %4008 = stablehlo.multiply %4004, %1080 : tensor<1x1x1x128xf32>
    %4009 = stablehlo.reshape %4007 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4010 = stablehlo.multiply %4008, %1079 : tensor<1x1x1x128xf32>
    %4011 = stablehlo.negate %4006 : tensor<256x28x28x128xf32>
    %cst_287 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4012 = stablehlo.reduce(%4011 init: %cst_287) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4013 = stablehlo.reshape %4012 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4014 = stablehlo.convert %4006 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_288 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4015 = stablehlo.reduce(%4010 init: %cst_288) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_289 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4016 = stablehlo.reduce(%4013 init: %cst_289) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4017 = stablehlo.multiply %4015, %1058 : tensor<128xf32>
    %4018 = stablehlo.negate %4017 : tensor<128xf32>
    %4019 = stablehlo.multiply %4018, %1045 : tensor<128xf32>
    %4020 = stablehlo.add %4016, %4019 : tensor<128xf32>
    %4021 = stablehlo.broadcast_in_dim %4017, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4022 = stablehlo.pad %4021, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4023 = stablehlo.broadcast_in_dim %4020, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4024 = stablehlo.pad %4023, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4025 = stablehlo.add %4022, %4024 : tensor<2x128xf32>
    %4026 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %4027 = stablehlo.divide %4025, %4026 : tensor<2x128xf32>
    %4028 = "stablehlo.all_reduce"(%4027) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %4029 = stablehlo.slice %4028 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %4030 = stablehlo.slice %4028 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_290 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4031 = stablehlo.reduce(%4030 init: %cst_290) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_291 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4032 = stablehlo.reduce(%4029 init: %cst_291) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4033 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4034 = stablehlo.divide %4031, %4033 : tensor<128xf32>
    %4035 = stablehlo.broadcast_in_dim %4034, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4036 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4037 = stablehlo.divide %4032, %4036 : tensor<128xf32>
    %4038 = stablehlo.broadcast_in_dim %4037, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4039 = stablehlo.multiply %4035, %1026 : tensor<256x28x28x128xf32>
    %4040 = stablehlo.add %4038, %4039 : tensor<256x28x28x128xf32>
    %4041 = stablehlo.convert %4040 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %4042 = stablehlo.add %4014, %4041 : tensor<256x28x28x128xbf16>
    %4043 = stablehlo.convolution(%1017, %4042) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %4044 = stablehlo.reverse %1021, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %4045 = stablehlo.convolution(%4042, %4044) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %4046 = stablehlo.add %3945, %4045 : tensor<256x28x28x128xbf16>
    %4047 = stablehlo.convert %4043 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %4048 = stablehlo.add %2823, %4047 : tensor<3x3x128x128xf32>
    %4049 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %4050 = stablehlo.select %1019, %4046, %4049 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %4051 = stablehlo.convert %4050 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_292 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4052 = stablehlo.reduce(%4051 init: %cst_292) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4053 = stablehlo.reshape %4052 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4054 = stablehlo.reshape %4053 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4055 = stablehlo.multiply %1001, %4051 : tensor<256x28x28x128xf32>
    %cst_293 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4056 = stablehlo.reduce(%4055 init: %cst_293) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4057 = stablehlo.reshape %4056 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4058 = stablehlo.broadcast_in_dim %1009, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %4059 = stablehlo.multiply %4051, %4058 : tensor<256x28x28x128xf32>
    %4060 = stablehlo.multiply %1004, %4057 : tensor<1x1x1x128xf32>
    %4061 = stablehlo.multiply %4057, %1008 : tensor<1x1x1x128xf32>
    %4062 = stablehlo.reshape %4060 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4063 = stablehlo.multiply %4061, %1007 : tensor<1x1x1x128xf32>
    %4064 = stablehlo.negate %4059 : tensor<256x28x28x128xf32>
    %cst_294 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4065 = stablehlo.reduce(%4064 init: %cst_294) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4066 = stablehlo.reshape %4065 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4067 = stablehlo.convert %4059 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_295 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4068 = stablehlo.reduce(%4063 init: %cst_295) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_296 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4069 = stablehlo.reduce(%4066 init: %cst_296) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4070 = stablehlo.multiply %4068, %986 : tensor<128xf32>
    %4071 = stablehlo.negate %4070 : tensor<128xf32>
    %4072 = stablehlo.multiply %4071, %973 : tensor<128xf32>
    %4073 = stablehlo.add %4069, %4072 : tensor<128xf32>
    %4074 = stablehlo.broadcast_in_dim %4070, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4075 = stablehlo.pad %4074, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4076 = stablehlo.broadcast_in_dim %4073, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4077 = stablehlo.pad %4076, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4078 = stablehlo.add %4075, %4077 : tensor<2x128xf32>
    %4079 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %4080 = stablehlo.divide %4078, %4079 : tensor<2x128xf32>
    %4081 = "stablehlo.all_reduce"(%4080) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %4082 = stablehlo.slice %4081 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %4083 = stablehlo.slice %4081 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_297 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4084 = stablehlo.reduce(%4083 init: %cst_297) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_298 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4085 = stablehlo.reduce(%4082 init: %cst_298) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4086 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4087 = stablehlo.divide %4084, %4086 : tensor<128xf32>
    %4088 = stablehlo.broadcast_in_dim %4087, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4089 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4090 = stablehlo.divide %4085, %4089 : tensor<128xf32>
    %4091 = stablehlo.broadcast_in_dim %4090, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4092 = stablehlo.multiply %4088, %954 : tensor<256x28x28x128xf32>
    %4093 = stablehlo.add %4091, %4092 : tensor<256x28x28x128xf32>
    %4094 = stablehlo.convert %4093 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %4095 = stablehlo.add %4067, %4094 : tensor<256x28x28x128xbf16>
    %4096 = stablehlo.convolution(%945, %4095) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %4097 = stablehlo.reverse %949, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %4098 = stablehlo.convolution(%4095, %4097) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %4099 = stablehlo.convert %4096 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %4100 = stablehlo.add %2825, %4099 : tensor<3x3x128x128xf32>
    %4101 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %4102 = stablehlo.select %947, %4098, %4101 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %4103 = stablehlo.convert %4102 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_299 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4104 = stablehlo.reduce(%4103 init: %cst_299) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4105 = stablehlo.reshape %4104 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4106 = stablehlo.reshape %4105 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4107 = stablehlo.multiply %930, %4103 : tensor<256x28x28x128xf32>
    %cst_300 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4108 = stablehlo.reduce(%4107 init: %cst_300) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4109 = stablehlo.reshape %4108 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4110 = stablehlo.broadcast_in_dim %938, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %4111 = stablehlo.multiply %4103, %4110 : tensor<256x28x28x128xf32>
    %4112 = stablehlo.multiply %933, %4109 : tensor<1x1x1x128xf32>
    %4113 = stablehlo.multiply %4109, %937 : tensor<1x1x1x128xf32>
    %4114 = stablehlo.reshape %4112 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4115 = stablehlo.multiply %4113, %936 : tensor<1x1x1x128xf32>
    %4116 = stablehlo.negate %4111 : tensor<256x28x28x128xf32>
    %cst_301 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4117 = stablehlo.reduce(%4116 init: %cst_301) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4118 = stablehlo.reshape %4117 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4119 = stablehlo.convert %4111 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_302 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4120 = stablehlo.reduce(%4115 init: %cst_302) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_303 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4121 = stablehlo.reduce(%4118 init: %cst_303) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4122 = stablehlo.multiply %4120, %915 : tensor<128xf32>
    %4123 = stablehlo.negate %4122 : tensor<128xf32>
    %4124 = stablehlo.multiply %4123, %902 : tensor<128xf32>
    %4125 = stablehlo.add %4121, %4124 : tensor<128xf32>
    %4126 = stablehlo.broadcast_in_dim %4122, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4127 = stablehlo.pad %4126, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4128 = stablehlo.broadcast_in_dim %4125, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4129 = stablehlo.pad %4128, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4130 = stablehlo.add %4127, %4129 : tensor<2x128xf32>
    %4131 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %4132 = stablehlo.divide %4130, %4131 : tensor<2x128xf32>
    %4133 = "stablehlo.all_reduce"(%4132) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %4134 = stablehlo.slice %4133 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %4135 = stablehlo.slice %4133 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_304 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4136 = stablehlo.reduce(%4135 init: %cst_304) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_305 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4137 = stablehlo.reduce(%4134 init: %cst_305) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4138 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4139 = stablehlo.divide %4136, %4138 : tensor<128xf32>
    %4140 = stablehlo.broadcast_in_dim %4139, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4141 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4142 = stablehlo.divide %4137, %4141 : tensor<128xf32>
    %4143 = stablehlo.broadcast_in_dim %4142, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4144 = stablehlo.multiply %4140, %883 : tensor<256x28x28x128xf32>
    %4145 = stablehlo.add %4143, %4144 : tensor<256x28x28x128xf32>
    %4146 = stablehlo.convert %4145 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %4147 = stablehlo.add %4119, %4146 : tensor<256x28x28x128xbf16>
    %4148 = stablehlo.convolution(%874, %4147) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %4149 = stablehlo.reverse %878, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %4150 = stablehlo.convolution(%4147, %4149) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %4151 = stablehlo.add %4050, %4150 : tensor<256x28x28x128xbf16>
    %4152 = stablehlo.convert %4148 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %4153 = stablehlo.add %2827, %4152 : tensor<3x3x128x128xf32>
    %4154 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %4155 = stablehlo.select %876, %4151, %4154 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %4156 = stablehlo.convert %4155 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_306 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4157 = stablehlo.reduce(%4156 init: %cst_306) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4158 = stablehlo.reshape %4157 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4159 = stablehlo.reshape %4158 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4160 = stablehlo.multiply %858, %4156 : tensor<256x28x28x128xf32>
    %cst_307 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4161 = stablehlo.reduce(%4160 init: %cst_307) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4162 = stablehlo.reshape %4161 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4163 = stablehlo.broadcast_in_dim %866, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %4164 = stablehlo.multiply %4156, %4163 : tensor<256x28x28x128xf32>
    %4165 = stablehlo.multiply %861, %4162 : tensor<1x1x1x128xf32>
    %4166 = stablehlo.multiply %4162, %865 : tensor<1x1x1x128xf32>
    %4167 = stablehlo.reshape %4165 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4168 = stablehlo.multiply %4166, %864 : tensor<1x1x1x128xf32>
    %4169 = stablehlo.negate %4164 : tensor<256x28x28x128xf32>
    %cst_308 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4170 = stablehlo.reduce(%4169 init: %cst_308) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4171 = stablehlo.reshape %4170 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4172 = stablehlo.convert %4164 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_309 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4173 = stablehlo.reduce(%4168 init: %cst_309) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_310 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4174 = stablehlo.reduce(%4171 init: %cst_310) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4175 = stablehlo.multiply %4173, %843 : tensor<128xf32>
    %4176 = stablehlo.negate %4175 : tensor<128xf32>
    %4177 = stablehlo.multiply %4176, %830 : tensor<128xf32>
    %4178 = stablehlo.add %4174, %4177 : tensor<128xf32>
    %4179 = stablehlo.broadcast_in_dim %4175, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4180 = stablehlo.pad %4179, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4181 = stablehlo.broadcast_in_dim %4178, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4182 = stablehlo.pad %4181, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4183 = stablehlo.add %4180, %4182 : tensor<2x128xf32>
    %4184 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %4185 = stablehlo.divide %4183, %4184 : tensor<2x128xf32>
    %4186 = "stablehlo.all_reduce"(%4185) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %4187 = stablehlo.slice %4186 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %4188 = stablehlo.slice %4186 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_311 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4189 = stablehlo.reduce(%4188 init: %cst_311) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_312 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4190 = stablehlo.reduce(%4187 init: %cst_312) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4191 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4192 = stablehlo.divide %4189, %4191 : tensor<128xf32>
    %4193 = stablehlo.broadcast_in_dim %4192, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4194 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4195 = stablehlo.divide %4190, %4194 : tensor<128xf32>
    %4196 = stablehlo.broadcast_in_dim %4195, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4197 = stablehlo.multiply %4193, %811 : tensor<256x28x28x128xf32>
    %4198 = stablehlo.add %4196, %4197 : tensor<256x28x28x128xf32>
    %4199 = stablehlo.convert %4198 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %4200 = stablehlo.add %4172, %4199 : tensor<256x28x28x128xbf16>
    %4201 = stablehlo.convolution(%802, %4200) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %4202 = stablehlo.reverse %806, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %4203 = stablehlo.convolution(%4200, %4202) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %4204 = stablehlo.convert %4201 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %4205 = stablehlo.add %2829, %4204 : tensor<3x3x128x128xf32>
    %4206 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %4207 = stablehlo.select %804, %4203, %4206 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %4208 = stablehlo.convert %4207 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_313 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4209 = stablehlo.reduce(%4208 init: %cst_313) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4210 = stablehlo.reshape %4209 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4211 = stablehlo.reshape %4210 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4212 = stablehlo.multiply %787, %4208 : tensor<256x28x28x128xf32>
    %cst_314 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4213 = stablehlo.reduce(%4212 init: %cst_314) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4214 = stablehlo.reshape %4213 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4215 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %4216 = stablehlo.multiply %4208, %4215 : tensor<256x28x28x128xf32>
    %4217 = stablehlo.multiply %790, %4214 : tensor<1x1x1x128xf32>
    %4218 = stablehlo.multiply %4214, %794 : tensor<1x1x1x128xf32>
    %4219 = stablehlo.reshape %4217 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4220 = stablehlo.multiply %4218, %793 : tensor<1x1x1x128xf32>
    %4221 = stablehlo.negate %4216 : tensor<256x28x28x128xf32>
    %cst_315 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4222 = stablehlo.reduce(%4221 init: %cst_315) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4223 = stablehlo.reshape %4222 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4224 = stablehlo.convert %4216 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_316 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4225 = stablehlo.reduce(%4220 init: %cst_316) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_317 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4226 = stablehlo.reduce(%4223 init: %cst_317) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4227 = stablehlo.multiply %4225, %772 : tensor<128xf32>
    %4228 = stablehlo.negate %4227 : tensor<128xf32>
    %4229 = stablehlo.multiply %4228, %759 : tensor<128xf32>
    %4230 = stablehlo.add %4226, %4229 : tensor<128xf32>
    %4231 = stablehlo.broadcast_in_dim %4227, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4232 = stablehlo.pad %4231, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4233 = stablehlo.broadcast_in_dim %4230, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4234 = stablehlo.pad %4233, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4235 = stablehlo.add %4232, %4234 : tensor<2x128xf32>
    %4236 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %4237 = stablehlo.divide %4235, %4236 : tensor<2x128xf32>
    %4238 = "stablehlo.all_reduce"(%4237) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %4239 = stablehlo.slice %4238 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %4240 = stablehlo.slice %4238 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_318 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4241 = stablehlo.reduce(%4240 init: %cst_318) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_319 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4242 = stablehlo.reduce(%4239 init: %cst_319) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4243 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4244 = stablehlo.divide %4241, %4243 : tensor<128xf32>
    %4245 = stablehlo.broadcast_in_dim %4244, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4246 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4247 = stablehlo.divide %4242, %4246 : tensor<128xf32>
    %4248 = stablehlo.broadcast_in_dim %4247, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4249 = stablehlo.multiply %4245, %740 : tensor<256x28x28x128xf32>
    %4250 = stablehlo.add %4248, %4249 : tensor<256x28x28x128xf32>
    %4251 = stablehlo.convert %4250 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %4252 = stablehlo.add %4224, %4251 : tensor<256x28x28x128xbf16>
    %4253 = stablehlo.convolution(%731, %4252) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %4254 = stablehlo.reverse %735, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %4255 = stablehlo.convolution(%4252, %4254) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %4256 = stablehlo.add %4155, %4255 : tensor<256x28x28x128xbf16>
    %4257 = stablehlo.convert %4253 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %4258 = stablehlo.add %2831, %4257 : tensor<3x3x128x128xf32>
    %4259 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %4260 = stablehlo.select %733, %4256, %4259 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %4261 = stablehlo.convert %4260 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_320 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4262 = stablehlo.reduce(%4261 init: %cst_320) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4263 = stablehlo.reshape %4262 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4264 = stablehlo.reshape %4263 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4265 = stablehlo.multiply %715, %4261 : tensor<256x28x28x128xf32>
    %cst_321 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4266 = stablehlo.reduce(%4265 init: %cst_321) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4267 = stablehlo.reshape %4266 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4268 = stablehlo.broadcast_in_dim %723, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %4269 = stablehlo.multiply %4261, %4268 : tensor<256x28x28x128xf32>
    %4270 = stablehlo.multiply %718, %4267 : tensor<1x1x1x128xf32>
    %4271 = stablehlo.multiply %4267, %722 : tensor<1x1x1x128xf32>
    %4272 = stablehlo.reshape %4270 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4273 = stablehlo.multiply %4271, %721 : tensor<1x1x1x128xf32>
    %4274 = stablehlo.negate %4269 : tensor<256x28x28x128xf32>
    %cst_322 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4275 = stablehlo.reduce(%4274 init: %cst_322) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4276 = stablehlo.reshape %4275 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4277 = stablehlo.convert %4269 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_323 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4278 = stablehlo.reduce(%4273 init: %cst_323) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_324 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4279 = stablehlo.reduce(%4276 init: %cst_324) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4280 = stablehlo.multiply %4278, %700 : tensor<128xf32>
    %4281 = stablehlo.negate %4280 : tensor<128xf32>
    %4282 = stablehlo.multiply %4281, %687 : tensor<128xf32>
    %4283 = stablehlo.add %4279, %4282 : tensor<128xf32>
    %4284 = stablehlo.broadcast_in_dim %4280, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4285 = stablehlo.pad %4284, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4286 = stablehlo.broadcast_in_dim %4283, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4287 = stablehlo.pad %4286, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4288 = stablehlo.add %4285, %4287 : tensor<2x128xf32>
    %4289 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %4290 = stablehlo.divide %4288, %4289 : tensor<2x128xf32>
    %4291 = "stablehlo.all_reduce"(%4290) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %4292 = stablehlo.slice %4291 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %4293 = stablehlo.slice %4291 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_325 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4294 = stablehlo.reduce(%4293 init: %cst_325) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_326 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4295 = stablehlo.reduce(%4292 init: %cst_326) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4296 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4297 = stablehlo.divide %4294, %4296 : tensor<128xf32>
    %4298 = stablehlo.broadcast_in_dim %4297, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4299 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4300 = stablehlo.divide %4295, %4299 : tensor<128xf32>
    %4301 = stablehlo.broadcast_in_dim %4300, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4302 = stablehlo.multiply %4298, %668 : tensor<256x28x28x128xf32>
    %4303 = stablehlo.add %4301, %4302 : tensor<256x28x28x128xf32>
    %4304 = stablehlo.convert %4303 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %4305 = stablehlo.add %4277, %4304 : tensor<256x28x28x128xbf16>
    %4306 = stablehlo.convolution(%521, %4305) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x28x28x128xbf16>) -> tensor<1x1x64x128xbf16>
    %4307 = stablehlo.reverse %663, dims = [0, 1] : tensor<1x1x64x128xbf16>
    %4308 = stablehlo.convolution(%4305, %4307) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<1x1x64x128xbf16>) -> tensor<256x56x56x64xbf16>
    %4309 = stablehlo.convert %4306 : (tensor<1x1x64x128xbf16>) -> tensor<1x1x64x128xf32>
    %4310 = stablehlo.add %2833, %4309 : tensor<1x1x64x128xf32>
    %4311 = stablehlo.convert %4260 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_327 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4312 = stablehlo.reduce(%4311 init: %cst_327) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4313 = stablehlo.reshape %4312 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4314 = stablehlo.reshape %4313 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4315 = stablehlo.multiply %648, %4311 : tensor<256x28x28x128xf32>
    %cst_328 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4316 = stablehlo.reduce(%4315 init: %cst_328) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4317 = stablehlo.reshape %4316 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4318 = stablehlo.broadcast_in_dim %656, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %4319 = stablehlo.multiply %4311, %4318 : tensor<256x28x28x128xf32>
    %4320 = stablehlo.multiply %651, %4317 : tensor<1x1x1x128xf32>
    %4321 = stablehlo.multiply %4317, %655 : tensor<1x1x1x128xf32>
    %4322 = stablehlo.reshape %4320 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4323 = stablehlo.multiply %4321, %654 : tensor<1x1x1x128xf32>
    %4324 = stablehlo.negate %4319 : tensor<256x28x28x128xf32>
    %cst_329 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4325 = stablehlo.reduce(%4324 init: %cst_329) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4326 = stablehlo.reshape %4325 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4327 = stablehlo.convert %4319 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_330 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4328 = stablehlo.reduce(%4323 init: %cst_330) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_331 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4329 = stablehlo.reduce(%4326 init: %cst_331) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4330 = stablehlo.multiply %4328, %633 : tensor<128xf32>
    %4331 = stablehlo.negate %4330 : tensor<128xf32>
    %4332 = stablehlo.multiply %4331, %620 : tensor<128xf32>
    %4333 = stablehlo.add %4329, %4332 : tensor<128xf32>
    %4334 = stablehlo.broadcast_in_dim %4330, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4335 = stablehlo.pad %4334, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4336 = stablehlo.broadcast_in_dim %4333, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4337 = stablehlo.pad %4336, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4338 = stablehlo.add %4335, %4337 : tensor<2x128xf32>
    %4339 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %4340 = stablehlo.divide %4338, %4339 : tensor<2x128xf32>
    %4341 = "stablehlo.all_reduce"(%4340) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %4342 = stablehlo.slice %4341 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %4343 = stablehlo.slice %4341 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_332 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4344 = stablehlo.reduce(%4343 init: %cst_332) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_333 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4345 = stablehlo.reduce(%4342 init: %cst_333) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4346 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4347 = stablehlo.divide %4344, %4346 : tensor<128xf32>
    %4348 = stablehlo.broadcast_in_dim %4347, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4349 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4350 = stablehlo.divide %4345, %4349 : tensor<128xf32>
    %4351 = stablehlo.broadcast_in_dim %4350, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4352 = stablehlo.multiply %4348, %601 : tensor<256x28x28x128xf32>
    %4353 = stablehlo.add %4351, %4352 : tensor<256x28x28x128xf32>
    %4354 = stablehlo.convert %4353 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %4355 = stablehlo.add %4327, %4354 : tensor<256x28x28x128xbf16>
    %4356 = stablehlo.convolution(%592, %4355) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x128x128xbf16>
    %4357 = stablehlo.reverse %596, dims = [0, 1] : tensor<3x3x128x128xbf16>
    %4358 = stablehlo.convolution(%4355, %4357) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x128x128xbf16>) -> tensor<256x28x28x128xbf16>
    %4359 = stablehlo.convert %4356 : (tensor<3x3x128x128xbf16>) -> tensor<3x3x128x128xf32>
    %4360 = stablehlo.add %2835, %4359 : tensor<3x3x128x128xf32>
    %4361 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %4362 = stablehlo.select %594, %4358, %4361 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xbf16>
    %4363 = stablehlo.convert %4362 : (tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xf32>
    %cst_334 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4364 = stablehlo.reduce(%4363 init: %cst_334) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4365 = stablehlo.reshape %4364 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4366 = stablehlo.reshape %4365 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4367 = stablehlo.multiply %577, %4363 : tensor<256x28x28x128xf32>
    %cst_335 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4368 = stablehlo.reduce(%4367 init: %cst_335) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4369 = stablehlo.reshape %4368 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4370 = stablehlo.broadcast_in_dim %585, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
    %4371 = stablehlo.multiply %4363, %4370 : tensor<256x28x28x128xf32>
    %4372 = stablehlo.multiply %580, %4369 : tensor<1x1x1x128xf32>
    %4373 = stablehlo.multiply %4369, %584 : tensor<1x1x1x128xf32>
    %4374 = stablehlo.reshape %4372 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
    %4375 = stablehlo.multiply %4373, %583 : tensor<1x1x1x128xf32>
    %4376 = stablehlo.negate %4371 : tensor<256x28x28x128xf32>
    %cst_336 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4377 = stablehlo.reduce(%4376 init: %cst_336) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4378 = stablehlo.reshape %4377 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %4379 = stablehlo.convert %4371 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %cst_337 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4380 = stablehlo.reduce(%4375 init: %cst_337) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_338 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4381 = stablehlo.reduce(%4378 init: %cst_338) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4382 = stablehlo.multiply %4380, %562 : tensor<128xf32>
    %4383 = stablehlo.negate %4382 : tensor<128xf32>
    %4384 = stablehlo.multiply %4383, %549 : tensor<128xf32>
    %4385 = stablehlo.add %4381, %4384 : tensor<128xf32>
    %4386 = stablehlo.broadcast_in_dim %4382, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4387 = stablehlo.pad %4386, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4388 = stablehlo.broadcast_in_dim %4385, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4389 = stablehlo.pad %4388, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
    %4390 = stablehlo.add %4387, %4389 : tensor<2x128xf32>
    %4391 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
    %4392 = stablehlo.divide %4390, %4391 : tensor<2x128xf32>
    %4393 = "stablehlo.all_reduce"(%4392) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %4394 = stablehlo.slice %4393 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %4395 = stablehlo.slice %4393 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
    %cst_339 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4396 = stablehlo.reduce(%4395 init: %cst_339) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %cst_340 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4397 = stablehlo.reduce(%4394 init: %cst_340) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
    %4398 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4399 = stablehlo.divide %4396, %4398 : tensor<128xf32>
    %4400 = stablehlo.broadcast_in_dim %4399, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4401 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4402 = stablehlo.divide %4397, %4401 : tensor<128xf32>
    %4403 = stablehlo.broadcast_in_dim %4402, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
    %4404 = stablehlo.multiply %4400, %530 : tensor<256x28x28x128xf32>
    %4405 = stablehlo.add %4403, %4404 : tensor<256x28x28x128xf32>
    %4406 = stablehlo.convert %4405 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xbf16>
    %4407 = stablehlo.add %4379, %4406 : tensor<256x28x28x128xbf16>
    %4408 = stablehlo.convolution(%521, %4407) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x28x28x128xbf16>) -> tensor<3x3x64x128xbf16>
    %4409 = stablehlo.reverse %525, dims = [0, 1] : tensor<3x3x64x128xbf16>
    %4410 = stablehlo.convolution(%4407, %4409) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xbf16>, tensor<3x3x64x128xbf16>) -> tensor<256x56x56x64xbf16>
    %4411 = stablehlo.add %4308, %4410 : tensor<256x56x56x64xbf16>
    %4412 = stablehlo.convert %4408 : (tensor<3x3x64x128xbf16>) -> tensor<3x3x64x128xf32>
    %4413 = stablehlo.add %2837, %4412 : tensor<3x3x64x128xf32>
    %4414 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %4415 = stablehlo.select %523, %4411, %4414 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %4416 = stablehlo.convert %4415 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_341 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4417 = stablehlo.reduce(%4416 init: %cst_341) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4418 = stablehlo.reshape %4417 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4419 = stablehlo.reshape %4418 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4420 = stablehlo.multiply %505, %4416 : tensor<256x56x56x64xf32>
    %cst_342 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4421 = stablehlo.reduce(%4420 init: %cst_342) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4422 = stablehlo.reshape %4421 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4423 = stablehlo.broadcast_in_dim %513, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %4424 = stablehlo.multiply %4416, %4423 : tensor<256x56x56x64xf32>
    %4425 = stablehlo.multiply %508, %4422 : tensor<1x1x1x64xf32>
    %4426 = stablehlo.multiply %4422, %512 : tensor<1x1x1x64xf32>
    %4427 = stablehlo.reshape %4425 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4428 = stablehlo.multiply %4426, %511 : tensor<1x1x1x64xf32>
    %4429 = stablehlo.negate %4424 : tensor<256x56x56x64xf32>
    %cst_343 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4430 = stablehlo.reduce(%4429 init: %cst_343) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4431 = stablehlo.reshape %4430 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4432 = stablehlo.convert %4424 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_344 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4433 = stablehlo.reduce(%4428 init: %cst_344) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_345 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4434 = stablehlo.reduce(%4431 init: %cst_345) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4435 = stablehlo.multiply %4433, %490 : tensor<64xf32>
    %4436 = stablehlo.negate %4435 : tensor<64xf32>
    %4437 = stablehlo.multiply %4436, %477 : tensor<64xf32>
    %4438 = stablehlo.add %4434, %4437 : tensor<64xf32>
    %4439 = stablehlo.broadcast_in_dim %4435, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4440 = stablehlo.pad %4439, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4441 = stablehlo.broadcast_in_dim %4438, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4442 = stablehlo.pad %4441, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4443 = stablehlo.add %4440, %4442 : tensor<2x64xf32>
    %4444 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %4445 = stablehlo.divide %4443, %4444 : tensor<2x64xf32>
    %4446 = "stablehlo.all_reduce"(%4445) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %4447 = stablehlo.slice %4446 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %4448 = stablehlo.slice %4446 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_346 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4449 = stablehlo.reduce(%4448 init: %cst_346) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_347 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4450 = stablehlo.reduce(%4447 init: %cst_347) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4451 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4452 = stablehlo.divide %4449, %4451 : tensor<64xf32>
    %4453 = stablehlo.broadcast_in_dim %4452, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4455 = stablehlo.divide %4450, %4454 : tensor<64xf32>
    %4456 = stablehlo.broadcast_in_dim %4455, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4457 = stablehlo.multiply %4453, %458 : tensor<256x56x56x64xf32>
    %4458 = stablehlo.add %4456, %4457 : tensor<256x56x56x64xf32>
    %4459 = stablehlo.convert %4458 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %4460 = stablehlo.add %4432, %4459 : tensor<256x56x56x64xbf16>
    %4461 = stablehlo.convolution(%449, %4460) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %4462 = stablehlo.reverse %453, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %4463 = stablehlo.convolution(%4460, %4462) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %4464 = stablehlo.convert %4461 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %4465 = stablehlo.add %2839, %4464 : tensor<3x3x64x64xf32>
    %4466 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %4467 = stablehlo.select %451, %4463, %4466 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %4468 = stablehlo.convert %4467 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_348 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4469 = stablehlo.reduce(%4468 init: %cst_348) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4470 = stablehlo.reshape %4469 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4471 = stablehlo.reshape %4470 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4472 = stablehlo.multiply %434, %4468 : tensor<256x56x56x64xf32>
    %cst_349 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4473 = stablehlo.reduce(%4472 init: %cst_349) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4474 = stablehlo.reshape %4473 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4475 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %4476 = stablehlo.multiply %4468, %4475 : tensor<256x56x56x64xf32>
    %4477 = stablehlo.multiply %437, %4474 : tensor<1x1x1x64xf32>
    %4478 = stablehlo.multiply %4474, %441 : tensor<1x1x1x64xf32>
    %4479 = stablehlo.reshape %4477 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4480 = stablehlo.multiply %4478, %440 : tensor<1x1x1x64xf32>
    %4481 = stablehlo.negate %4476 : tensor<256x56x56x64xf32>
    %cst_350 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4482 = stablehlo.reduce(%4481 init: %cst_350) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4483 = stablehlo.reshape %4482 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4484 = stablehlo.convert %4476 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_351 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4485 = stablehlo.reduce(%4480 init: %cst_351) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_352 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4486 = stablehlo.reduce(%4483 init: %cst_352) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4487 = stablehlo.multiply %4485, %419 : tensor<64xf32>
    %4488 = stablehlo.negate %4487 : tensor<64xf32>
    %4489 = stablehlo.multiply %4488, %406 : tensor<64xf32>
    %4490 = stablehlo.add %4486, %4489 : tensor<64xf32>
    %4491 = stablehlo.broadcast_in_dim %4487, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4492 = stablehlo.pad %4491, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4493 = stablehlo.broadcast_in_dim %4490, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4494 = stablehlo.pad %4493, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4495 = stablehlo.add %4492, %4494 : tensor<2x64xf32>
    %4496 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %4497 = stablehlo.divide %4495, %4496 : tensor<2x64xf32>
    %4498 = "stablehlo.all_reduce"(%4497) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %4499 = stablehlo.slice %4498 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %4500 = stablehlo.slice %4498 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_353 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4501 = stablehlo.reduce(%4500 init: %cst_353) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_354 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4502 = stablehlo.reduce(%4499 init: %cst_354) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4503 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4504 = stablehlo.divide %4501, %4503 : tensor<64xf32>
    %4505 = stablehlo.broadcast_in_dim %4504, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4506 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4507 = stablehlo.divide %4502, %4506 : tensor<64xf32>
    %4508 = stablehlo.broadcast_in_dim %4507, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4509 = stablehlo.multiply %4505, %387 : tensor<256x56x56x64xf32>
    %4510 = stablehlo.add %4508, %4509 : tensor<256x56x56x64xf32>
    %4511 = stablehlo.convert %4510 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %4512 = stablehlo.add %4484, %4511 : tensor<256x56x56x64xbf16>
    %4513 = stablehlo.convolution(%378, %4512) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %4514 = stablehlo.reverse %382, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %4515 = stablehlo.convolution(%4512, %4514) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %4516 = stablehlo.add %4415, %4515 : tensor<256x56x56x64xbf16>
    %4517 = stablehlo.convert %4513 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %4518 = stablehlo.add %2841, %4517 : tensor<3x3x64x64xf32>
    %4519 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %4520 = stablehlo.select %380, %4516, %4519 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %4521 = stablehlo.convert %4520 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_355 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4522 = stablehlo.reduce(%4521 init: %cst_355) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4523 = stablehlo.reshape %4522 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4524 = stablehlo.reshape %4523 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4525 = stablehlo.multiply %362, %4521 : tensor<256x56x56x64xf32>
    %cst_356 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4526 = stablehlo.reduce(%4525 init: %cst_356) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4527 = stablehlo.reshape %4526 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4528 = stablehlo.broadcast_in_dim %370, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %4529 = stablehlo.multiply %4521, %4528 : tensor<256x56x56x64xf32>
    %4530 = stablehlo.multiply %365, %4527 : tensor<1x1x1x64xf32>
    %4531 = stablehlo.multiply %4527, %369 : tensor<1x1x1x64xf32>
    %4532 = stablehlo.reshape %4530 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4533 = stablehlo.multiply %4531, %368 : tensor<1x1x1x64xf32>
    %4534 = stablehlo.negate %4529 : tensor<256x56x56x64xf32>
    %cst_357 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4535 = stablehlo.reduce(%4534 init: %cst_357) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4536 = stablehlo.reshape %4535 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4537 = stablehlo.convert %4529 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_358 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4538 = stablehlo.reduce(%4533 init: %cst_358) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_359 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4539 = stablehlo.reduce(%4536 init: %cst_359) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4540 = stablehlo.multiply %4538, %347 : tensor<64xf32>
    %4541 = stablehlo.negate %4540 : tensor<64xf32>
    %4542 = stablehlo.multiply %4541, %334 : tensor<64xf32>
    %4543 = stablehlo.add %4539, %4542 : tensor<64xf32>
    %4544 = stablehlo.broadcast_in_dim %4540, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4545 = stablehlo.pad %4544, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4546 = stablehlo.broadcast_in_dim %4543, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4547 = stablehlo.pad %4546, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4548 = stablehlo.add %4545, %4547 : tensor<2x64xf32>
    %4549 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %4550 = stablehlo.divide %4548, %4549 : tensor<2x64xf32>
    %4551 = "stablehlo.all_reduce"(%4550) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %4552 = stablehlo.slice %4551 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %4553 = stablehlo.slice %4551 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_360 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4554 = stablehlo.reduce(%4553 init: %cst_360) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_361 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4555 = stablehlo.reduce(%4552 init: %cst_361) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4556 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4557 = stablehlo.divide %4554, %4556 : tensor<64xf32>
    %4558 = stablehlo.broadcast_in_dim %4557, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4559 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4560 = stablehlo.divide %4555, %4559 : tensor<64xf32>
    %4561 = stablehlo.broadcast_in_dim %4560, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4562 = stablehlo.multiply %4558, %315 : tensor<256x56x56x64xf32>
    %4563 = stablehlo.add %4561, %4562 : tensor<256x56x56x64xf32>
    %4564 = stablehlo.convert %4563 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %4565 = stablehlo.add %4537, %4564 : tensor<256x56x56x64xbf16>
    %4566 = stablehlo.convolution(%306, %4565) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %4567 = stablehlo.reverse %310, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %4568 = stablehlo.convolution(%4565, %4567) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %4569 = stablehlo.convert %4566 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %4570 = stablehlo.add %2869, %4569 : tensor<3x3x64x64xf32>
    %4571 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %4572 = stablehlo.select %308, %4568, %4571 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %4573 = stablehlo.convert %4572 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_362 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4574 = stablehlo.reduce(%4573 init: %cst_362) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4575 = stablehlo.reshape %4574 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4576 = stablehlo.reshape %4575 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4577 = stablehlo.multiply %291, %4573 : tensor<256x56x56x64xf32>
    %cst_363 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4578 = stablehlo.reduce(%4577 init: %cst_363) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4579 = stablehlo.reshape %4578 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4580 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %4581 = stablehlo.multiply %4573, %4580 : tensor<256x56x56x64xf32>
    %4582 = stablehlo.multiply %294, %4579 : tensor<1x1x1x64xf32>
    %4583 = stablehlo.multiply %4579, %298 : tensor<1x1x1x64xf32>
    %4584 = stablehlo.reshape %4582 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4585 = stablehlo.multiply %4583, %297 : tensor<1x1x1x64xf32>
    %4586 = stablehlo.negate %4581 : tensor<256x56x56x64xf32>
    %cst_364 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4587 = stablehlo.reduce(%4586 init: %cst_364) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4588 = stablehlo.reshape %4587 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4589 = stablehlo.convert %4581 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_365 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4590 = stablehlo.reduce(%4585 init: %cst_365) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_366 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4591 = stablehlo.reduce(%4588 init: %cst_366) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4592 = stablehlo.multiply %4590, %276 : tensor<64xf32>
    %4593 = stablehlo.negate %4592 : tensor<64xf32>
    %4594 = stablehlo.multiply %4593, %263 : tensor<64xf32>
    %4595 = stablehlo.add %4591, %4594 : tensor<64xf32>
    %4596 = stablehlo.broadcast_in_dim %4592, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4597 = stablehlo.pad %4596, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4598 = stablehlo.broadcast_in_dim %4595, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4599 = stablehlo.pad %4598, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4600 = stablehlo.add %4597, %4599 : tensor<2x64xf32>
    %4601 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %4602 = stablehlo.divide %4600, %4601 : tensor<2x64xf32>
    %4603 = "stablehlo.all_reduce"(%4602) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %4604 = stablehlo.slice %4603 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %4605 = stablehlo.slice %4603 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_367 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4606 = stablehlo.reduce(%4605 init: %cst_367) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_368 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4607 = stablehlo.reduce(%4604 init: %cst_368) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4608 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4609 = stablehlo.divide %4606, %4608 : tensor<64xf32>
    %4610 = stablehlo.broadcast_in_dim %4609, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4611 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4612 = stablehlo.divide %4607, %4611 : tensor<64xf32>
    %4613 = stablehlo.broadcast_in_dim %4612, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4614 = stablehlo.multiply %4610, %244 : tensor<256x56x56x64xf32>
    %4615 = stablehlo.add %4613, %4614 : tensor<256x56x56x64xf32>
    %4616 = stablehlo.convert %4615 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %4617 = stablehlo.add %4589, %4616 : tensor<256x56x56x64xbf16>
    %4618 = stablehlo.convolution(%235, %4617) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %4619 = stablehlo.reverse %239, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %4620 = stablehlo.convolution(%4617, %4619) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %4621 = stablehlo.add %4520, %4620 : tensor<256x56x56x64xbf16>
    %4622 = stablehlo.convert %4618 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %4623 = stablehlo.add %2871, %4622 : tensor<3x3x64x64xf32>
    %4624 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %4625 = stablehlo.select %237, %4621, %4624 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %4626 = stablehlo.convert %4625 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_369 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4627 = stablehlo.reduce(%4626 init: %cst_369) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4628 = stablehlo.reshape %4627 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4629 = stablehlo.reshape %4628 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4630 = stablehlo.multiply %219, %4626 : tensor<256x56x56x64xf32>
    %cst_370 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4631 = stablehlo.reduce(%4630 init: %cst_370) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4632 = stablehlo.reshape %4631 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4633 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %4634 = stablehlo.multiply %4626, %4633 : tensor<256x56x56x64xf32>
    %4635 = stablehlo.multiply %222, %4632 : tensor<1x1x1x64xf32>
    %4636 = stablehlo.multiply %4632, %226 : tensor<1x1x1x64xf32>
    %4637 = stablehlo.reshape %4635 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4638 = stablehlo.multiply %4636, %225 : tensor<1x1x1x64xf32>
    %4639 = stablehlo.negate %4634 : tensor<256x56x56x64xf32>
    %cst_371 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4640 = stablehlo.reduce(%4639 init: %cst_371) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4641 = stablehlo.reshape %4640 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4642 = stablehlo.convert %4634 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_372 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4643 = stablehlo.reduce(%4638 init: %cst_372) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_373 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4644 = stablehlo.reduce(%4641 init: %cst_373) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4645 = stablehlo.multiply %4643, %204 : tensor<64xf32>
    %4646 = stablehlo.negate %4645 : tensor<64xf32>
    %4647 = stablehlo.multiply %4646, %191 : tensor<64xf32>
    %4648 = stablehlo.add %4644, %4647 : tensor<64xf32>
    %4649 = stablehlo.broadcast_in_dim %4645, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4650 = stablehlo.pad %4649, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4651 = stablehlo.broadcast_in_dim %4648, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4652 = stablehlo.pad %4651, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4653 = stablehlo.add %4650, %4652 : tensor<2x64xf32>
    %4654 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %4655 = stablehlo.divide %4653, %4654 : tensor<2x64xf32>
    %4656 = "stablehlo.all_reduce"(%4655) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %4657 = stablehlo.slice %4656 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %4658 = stablehlo.slice %4656 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_374 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4659 = stablehlo.reduce(%4658 init: %cst_374) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_375 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4660 = stablehlo.reduce(%4657 init: %cst_375) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4661 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4662 = stablehlo.divide %4659, %4661 : tensor<64xf32>
    %4663 = stablehlo.broadcast_in_dim %4662, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4664 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4665 = stablehlo.divide %4660, %4664 : tensor<64xf32>
    %4666 = stablehlo.broadcast_in_dim %4665, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4667 = stablehlo.multiply %4663, %172 : tensor<256x56x56x64xf32>
    %4668 = stablehlo.add %4666, %4667 : tensor<256x56x56x64xf32>
    %4669 = stablehlo.convert %4668 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %4670 = stablehlo.add %4642, %4669 : tensor<256x56x56x64xbf16>
    %4671 = stablehlo.convolution(%163, %4670) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %4672 = stablehlo.reverse %167, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %4673 = stablehlo.convolution(%4670, %4672) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %4674 = stablehlo.convert %4671 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %4675 = stablehlo.add %2873, %4674 : tensor<3x3x64x64xf32>
    %4676 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x56x56x64xbf16>
    %4677 = stablehlo.select %165, %4673, %4676 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xbf16>
    %4678 = stablehlo.convert %4677 : (tensor<256x56x56x64xbf16>) -> tensor<256x56x56x64xf32>
    %cst_376 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4679 = stablehlo.reduce(%4678 init: %cst_376) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4680 = stablehlo.reshape %4679 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4681 = stablehlo.reshape %4680 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4682 = stablehlo.multiply %148, %4678 : tensor<256x56x56x64xf32>
    %cst_377 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4683 = stablehlo.reduce(%4682 init: %cst_377) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4684 = stablehlo.reshape %4683 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4685 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
    %4686 = stablehlo.multiply %4678, %4685 : tensor<256x56x56x64xf32>
    %4687 = stablehlo.multiply %151, %4684 : tensor<1x1x1x64xf32>
    %4688 = stablehlo.multiply %4684, %155 : tensor<1x1x1x64xf32>
    %4689 = stablehlo.reshape %4687 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4690 = stablehlo.multiply %4688, %154 : tensor<1x1x1x64xf32>
    %4691 = stablehlo.negate %4686 : tensor<256x56x56x64xf32>
    %cst_378 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4692 = stablehlo.reduce(%4691 init: %cst_378) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4693 = stablehlo.reshape %4692 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4694 = stablehlo.convert %4686 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %cst_379 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4695 = stablehlo.reduce(%4690 init: %cst_379) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_380 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4696 = stablehlo.reduce(%4693 init: %cst_380) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4697 = stablehlo.multiply %4695, %133 : tensor<64xf32>
    %4698 = stablehlo.negate %4697 : tensor<64xf32>
    %4699 = stablehlo.multiply %4698, %120 : tensor<64xf32>
    %4700 = stablehlo.add %4696, %4699 : tensor<64xf32>
    %4701 = stablehlo.broadcast_in_dim %4697, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4702 = stablehlo.pad %4701, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4703 = stablehlo.broadcast_in_dim %4700, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4704 = stablehlo.pad %4703, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4705 = stablehlo.add %4702, %4704 : tensor<2x64xf32>
    %4706 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %4707 = stablehlo.divide %4705, %4706 : tensor<2x64xf32>
    %4708 = "stablehlo.all_reduce"(%4707) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %4709 = stablehlo.slice %4708 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %4710 = stablehlo.slice %4708 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_381 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4711 = stablehlo.reduce(%4710 init: %cst_381) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_382 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4712 = stablehlo.reduce(%4709 init: %cst_382) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4713 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4714 = stablehlo.divide %4711, %4713 : tensor<64xf32>
    %4715 = stablehlo.broadcast_in_dim %4714, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4716 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4717 = stablehlo.divide %4712, %4716 : tensor<64xf32>
    %4718 = stablehlo.broadcast_in_dim %4717, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
    %4719 = stablehlo.multiply %4715, %101 : tensor<256x56x56x64xf32>
    %4720 = stablehlo.add %4718, %4719 : tensor<256x56x56x64xf32>
    %4721 = stablehlo.convert %4720 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xbf16>
    %4722 = stablehlo.add %4694, %4721 : tensor<256x56x56x64xbf16>
    %4723 = stablehlo.convolution(%95, %4722) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<256x56x56x64xbf16>) -> tensor<3x3x64x64xbf16>
    %4724 = stablehlo.reverse %96, dims = [0, 1] : tensor<3x3x64x64xbf16>
    %4725 = stablehlo.convolution(%4722, %4724) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<256x56x56x64xbf16>
    %4726 = stablehlo.add %4625, %4725 : tensor<256x56x56x64xbf16>
    %4727 = stablehlo.convert %4723 : (tensor<3x3x64x64xbf16>) -> tensor<3x3x64x64xf32>
    %4728 = stablehlo.add %2875, %4727 : tensor<3x3x64x64xf32>
    %cst_383 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %4729 = stablehlo.pad %90, %cst_383, low = [0, 0, 0, 0], high = [0, 1, 1, 0], interior = [0, 0, 0, 0] : (tensor<256x112x112x64xbf16>, tensor<bf16>) -> tensor<256x113x113x64xbf16>
    %cst_384 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %4730 = "stablehlo.select_and_scatter"(%4729, %4726, %cst_384) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg298: tensor<bf16>, %arg299: tensor<bf16>):
      %7388 = stablehlo.compare  GE, %arg298, %arg299,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %7388 : tensor<i1>
    }, {
    ^bb0(%arg298: tensor<bf16>, %arg299: tensor<bf16>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<bf16>
      stablehlo.return %7388 : tensor<bf16>
    }) : (tensor<256x113x113x64xbf16>, tensor<256x56x56x64xbf16>, tensor<bf16>) -> tensor<256x113x113x64xbf16>
    %4731 = stablehlo.slice %4730 [0:256, 0:112, 0:112, 0:64] : (tensor<256x113x113x64xbf16>) -> tensor<256x112x112x64xbf16>
    %4732 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<bf16>) -> tensor<256x112x112x64xbf16>
    %4733 = stablehlo.select %92, %4731, %4732 : tensor<256x112x112x64xi1>, tensor<256x112x112x64xbf16>
    %4734 = stablehlo.convert %4733 : (tensor<256x112x112x64xbf16>) -> tensor<256x112x112x64xf32>
    %cst_385 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4735 = stablehlo.reduce(%4734 init: %cst_385) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4736 = stablehlo.reshape %4735 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4737 = stablehlo.reshape %4736 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4738 = stablehlo.multiply %75, %4734 : tensor<256x112x112x64xf32>
    %cst_386 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4739 = stablehlo.reduce(%4738 init: %cst_386) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4740 = stablehlo.reshape %4739 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4741 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
    %4742 = stablehlo.multiply %4734, %4741 : tensor<256x112x112x64xf32>
    %4743 = stablehlo.multiply %78, %4740 : tensor<1x1x1x64xf32>
    %4744 = stablehlo.multiply %4740, %82 : tensor<1x1x1x64xf32>
    %4745 = stablehlo.reshape %4743 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
    %4746 = stablehlo.multiply %4744, %81 : tensor<1x1x1x64xf32>
    %4747 = stablehlo.negate %4742 : tensor<256x112x112x64xf32>
    %cst_387 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4748 = stablehlo.reduce(%4747 init: %cst_387) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4749 = stablehlo.reshape %4748 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %4750 = stablehlo.convert %4742 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %cst_388 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4751 = stablehlo.reduce(%4746 init: %cst_388) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_389 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4752 = stablehlo.reduce(%4749 init: %cst_389) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4753 = stablehlo.multiply %4751, %60 : tensor<64xf32>
    %4754 = stablehlo.negate %4753 : tensor<64xf32>
    %4755 = stablehlo.multiply %4754, %47 : tensor<64xf32>
    %4756 = stablehlo.add %4752, %4755 : tensor<64xf32>
    %4757 = stablehlo.broadcast_in_dim %4753, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4758 = stablehlo.pad %4757, %cst_12, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4759 = stablehlo.broadcast_in_dim %4756, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %4760 = stablehlo.pad %4759, %cst_12, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
    %4761 = stablehlo.add %4758, %4760 : tensor<2x64xf32>
    %4762 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %4763 = stablehlo.divide %4761, %4762 : tensor<2x64xf32>
    %4764 = "stablehlo.all_reduce"(%4763) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %4765 = stablehlo.slice %4764 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %4766 = stablehlo.slice %4764 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
    %cst_390 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4767 = stablehlo.reduce(%4766 init: %cst_390) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %cst_391 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4768 = stablehlo.reduce(%4765 init: %cst_391) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
    %4769 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4770 = stablehlo.divide %4767, %4769 : tensor<64xf32>
    %4771 = stablehlo.broadcast_in_dim %4770, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
    %4772 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4773 = stablehlo.divide %4768, %4772 : tensor<64xf32>
    %4774 = stablehlo.broadcast_in_dim %4773, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
    %4775 = stablehlo.multiply %4771, %28 : tensor<256x112x112x64xf32>
    %4776 = stablehlo.add %4774, %4775 : tensor<256x112x112x64xf32>
    %4777 = stablehlo.convert %4776 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xbf16>
    %4778 = stablehlo.add %4750, %4777 : tensor<256x112x112x64xbf16>
    %4779 = stablehlo.convolution(%0, %4778) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xbf16>, tensor<256x112x112x64xbf16>) -> tensor<7x7x3x64xbf16>
    %4780 = stablehlo.convert %4779 : (tensor<7x7x3x64xbf16>) -> tensor<7x7x3x64xf32>
    %4781 = stablehlo.add %2805, %4780 : tensor<7x7x3x64xf32>
    %4782 = stablehlo.convert %arg295 : tensor<f32>
    %4783 = stablehlo.divide %2801, %4782 : tensor<f32>
    %4784 = stablehlo.convert %arg295 : tensor<f32>
    %4785 = stablehlo.broadcast_in_dim %4784, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %4786 = stablehlo.divide %2891, %4785 : tensor<1000xf32>
    %4787 = stablehlo.convert %arg295 : tensor<f32>
    %4788 = stablehlo.broadcast_in_dim %4787, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %4789 = stablehlo.divide %2893, %4788 : tensor<512x1000xf32>
    %4790 = stablehlo.convert %arg295 : tensor<f32>
    %4791 = stablehlo.broadcast_in_dim %4790, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4792 = stablehlo.divide %4681, %4791 : tensor<64xf32>
    %4793 = stablehlo.convert %arg295 : tensor<f32>
    %4794 = stablehlo.broadcast_in_dim %4793, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4795 = stablehlo.divide %4689, %4794 : tensor<64xf32>
    %4796 = stablehlo.convert %arg295 : tensor<f32>
    %4797 = stablehlo.broadcast_in_dim %4796, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4798 = stablehlo.divide %4629, %4797 : tensor<64xf32>
    %4799 = stablehlo.convert %arg295 : tensor<f32>
    %4800 = stablehlo.broadcast_in_dim %4799, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4801 = stablehlo.divide %4637, %4800 : tensor<64xf32>
    %4802 = stablehlo.convert %arg295 : tensor<f32>
    %4803 = stablehlo.broadcast_in_dim %4802, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %4804 = stablehlo.divide %4728, %4803 : tensor<3x3x64x64xf32>
    %4805 = stablehlo.convert %arg295 : tensor<f32>
    %4806 = stablehlo.broadcast_in_dim %4805, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %4807 = stablehlo.divide %4675, %4806 : tensor<3x3x64x64xf32>
    %4808 = stablehlo.convert %arg295 : tensor<f32>
    %4809 = stablehlo.broadcast_in_dim %4808, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4810 = stablehlo.divide %4576, %4809 : tensor<64xf32>
    %4811 = stablehlo.convert %arg295 : tensor<f32>
    %4812 = stablehlo.broadcast_in_dim %4811, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4813 = stablehlo.divide %4584, %4812 : tensor<64xf32>
    %4814 = stablehlo.convert %arg295 : tensor<f32>
    %4815 = stablehlo.broadcast_in_dim %4814, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4816 = stablehlo.divide %4524, %4815 : tensor<64xf32>
    %4817 = stablehlo.convert %arg295 : tensor<f32>
    %4818 = stablehlo.broadcast_in_dim %4817, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4819 = stablehlo.divide %4532, %4818 : tensor<64xf32>
    %4820 = stablehlo.convert %arg295 : tensor<f32>
    %4821 = stablehlo.broadcast_in_dim %4820, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %4822 = stablehlo.divide %4623, %4821 : tensor<3x3x64x64xf32>
    %4823 = stablehlo.convert %arg295 : tensor<f32>
    %4824 = stablehlo.broadcast_in_dim %4823, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %4825 = stablehlo.divide %4570, %4824 : tensor<3x3x64x64xf32>
    %4826 = stablehlo.convert %arg295 : tensor<f32>
    %4827 = stablehlo.broadcast_in_dim %4826, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4828 = stablehlo.divide %3531, %4827 : tensor<256xf32>
    %4829 = stablehlo.convert %arg295 : tensor<f32>
    %4830 = stablehlo.broadcast_in_dim %4829, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4831 = stablehlo.divide %3539, %4830 : tensor<256xf32>
    %4832 = stablehlo.convert %arg295 : tensor<f32>
    %4833 = stablehlo.broadcast_in_dim %4832, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4834 = stablehlo.divide %3479, %4833 : tensor<256xf32>
    %4835 = stablehlo.convert %arg295 : tensor<f32>
    %4836 = stablehlo.broadcast_in_dim %4835, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4837 = stablehlo.divide %3487, %4836 : tensor<256xf32>
    %4838 = stablehlo.convert %arg295 : tensor<f32>
    %4839 = stablehlo.broadcast_in_dim %4838, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %4840 = stablehlo.divide %3578, %4839 : tensor<3x3x256x256xf32>
    %4841 = stablehlo.convert %arg295 : tensor<f32>
    %4842 = stablehlo.broadcast_in_dim %4841, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %4843 = stablehlo.divide %3525, %4842 : tensor<3x3x256x256xf32>
    %4844 = stablehlo.convert %arg295 : tensor<f32>
    %4845 = stablehlo.broadcast_in_dim %4844, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4846 = stablehlo.divide %3426, %4845 : tensor<256xf32>
    %4847 = stablehlo.convert %arg295 : tensor<f32>
    %4848 = stablehlo.broadcast_in_dim %4847, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4849 = stablehlo.divide %3434, %4848 : tensor<256xf32>
    %4850 = stablehlo.convert %arg295 : tensor<f32>
    %4851 = stablehlo.broadcast_in_dim %4850, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4852 = stablehlo.divide %3374, %4851 : tensor<256xf32>
    %4853 = stablehlo.convert %arg295 : tensor<f32>
    %4854 = stablehlo.broadcast_in_dim %4853, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4855 = stablehlo.divide %3382, %4854 : tensor<256xf32>
    %4856 = stablehlo.convert %arg295 : tensor<f32>
    %4857 = stablehlo.broadcast_in_dim %4856, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %4858 = stablehlo.divide %3473, %4857 : tensor<3x3x256x256xf32>
    %4859 = stablehlo.convert %arg295 : tensor<f32>
    %4860 = stablehlo.broadcast_in_dim %4859, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %4861 = stablehlo.divide %3420, %4860 : tensor<3x3x256x256xf32>
    %4862 = stablehlo.convert %arg295 : tensor<f32>
    %4863 = stablehlo.broadcast_in_dim %4862, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4864 = stablehlo.divide %3321, %4863 : tensor<256xf32>
    %4865 = stablehlo.convert %arg295 : tensor<f32>
    %4866 = stablehlo.broadcast_in_dim %4865, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4867 = stablehlo.divide %3329, %4866 : tensor<256xf32>
    %4868 = stablehlo.convert %arg295 : tensor<f32>
    %4869 = stablehlo.broadcast_in_dim %4868, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4870 = stablehlo.divide %3269, %4869 : tensor<256xf32>
    %4871 = stablehlo.convert %arg295 : tensor<f32>
    %4872 = stablehlo.broadcast_in_dim %4871, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %4873 = stablehlo.divide %3277, %4872 : tensor<256xf32>
    %4874 = stablehlo.convert %arg295 : tensor<f32>
    %4875 = stablehlo.broadcast_in_dim %4874, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %4876 = stablehlo.divide %3368, %4875 : tensor<3x3x256x256xf32>
    %4877 = stablehlo.convert %arg295 : tensor<f32>
    %4878 = stablehlo.broadcast_in_dim %4877, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %4879 = stablehlo.divide %3315, %4878 : tensor<3x3x256x256xf32>
    %4880 = stablehlo.convert %arg295 : tensor<f32>
    %4881 = stablehlo.broadcast_in_dim %4880, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4882 = stablehlo.divide %3216, %4881 : tensor<512xf32>
    %4883 = stablehlo.convert %arg295 : tensor<f32>
    %4884 = stablehlo.broadcast_in_dim %4883, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4885 = stablehlo.divide %3224, %4884 : tensor<512xf32>
    %4886 = stablehlo.convert %arg295 : tensor<f32>
    %4887 = stablehlo.broadcast_in_dim %4886, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4888 = stablehlo.divide %3164, %4887 : tensor<512xf32>
    %4889 = stablehlo.convert %arg295 : tensor<f32>
    %4890 = stablehlo.broadcast_in_dim %4889, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4891 = stablehlo.divide %3172, %4890 : tensor<512xf32>
    %4892 = stablehlo.convert %arg295 : tensor<f32>
    %4893 = stablehlo.broadcast_in_dim %4892, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %4894 = stablehlo.divide %3263, %4893 : tensor<3x3x256x512xf32>
    %4895 = stablehlo.convert %arg295 : tensor<f32>
    %4896 = stablehlo.broadcast_in_dim %4895, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %4897 = stablehlo.divide %3210, %4896 : tensor<3x3x512x512xf32>
    %4898 = stablehlo.convert %arg295 : tensor<f32>
    %4899 = stablehlo.broadcast_in_dim %4898, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %4900 = stablehlo.divide %3160, %4899 : tensor<1x1x256x512xf32>
    %4901 = stablehlo.convert %arg295 : tensor<f32>
    %4902 = stablehlo.broadcast_in_dim %4901, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4903 = stablehlo.divide %3114, %4902 : tensor<512xf32>
    %4904 = stablehlo.convert %arg295 : tensor<f32>
    %4905 = stablehlo.broadcast_in_dim %4904, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4906 = stablehlo.divide %3122, %4905 : tensor<512xf32>
    %4907 = stablehlo.convert %arg295 : tensor<f32>
    %4908 = stablehlo.broadcast_in_dim %4907, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4909 = stablehlo.divide %3061, %4908 : tensor<512xf32>
    %4910 = stablehlo.convert %arg295 : tensor<f32>
    %4911 = stablehlo.broadcast_in_dim %4910, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4912 = stablehlo.divide %3069, %4911 : tensor<512xf32>
    %4913 = stablehlo.convert %arg295 : tensor<f32>
    %4914 = stablehlo.broadcast_in_dim %4913, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4915 = stablehlo.divide %3009, %4914 : tensor<512xf32>
    %4916 = stablehlo.convert %arg295 : tensor<f32>
    %4917 = stablehlo.broadcast_in_dim %4916, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4918 = stablehlo.divide %3017, %4917 : tensor<512xf32>
    %4919 = stablehlo.convert %arg295 : tensor<f32>
    %4920 = stablehlo.broadcast_in_dim %4919, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %4921 = stablehlo.divide %3108, %4920 : tensor<3x3x512x512xf32>
    %4922 = stablehlo.convert %arg295 : tensor<f32>
    %4923 = stablehlo.broadcast_in_dim %4922, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %4924 = stablehlo.divide %3055, %4923 : tensor<3x3x512x512xf32>
    %4925 = stablehlo.convert %arg295 : tensor<f32>
    %4926 = stablehlo.broadcast_in_dim %4925, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4927 = stablehlo.divide %2956, %4926 : tensor<512xf32>
    %4928 = stablehlo.convert %arg295 : tensor<f32>
    %4929 = stablehlo.broadcast_in_dim %4928, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4930 = stablehlo.divide %2964, %4929 : tensor<512xf32>
    %4931 = stablehlo.convert %arg295 : tensor<f32>
    %4932 = stablehlo.broadcast_in_dim %4931, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4933 = stablehlo.divide %2904, %4932 : tensor<512xf32>
    %4934 = stablehlo.convert %arg295 : tensor<f32>
    %4935 = stablehlo.broadcast_in_dim %4934, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4936 = stablehlo.divide %2912, %4935 : tensor<512xf32>
    %4937 = stablehlo.convert %arg295 : tensor<f32>
    %4938 = stablehlo.broadcast_in_dim %4937, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %4939 = stablehlo.divide %3003, %4938 : tensor<3x3x512x512xf32>
    %4940 = stablehlo.convert %arg295 : tensor<f32>
    %4941 = stablehlo.broadcast_in_dim %4940, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %4942 = stablehlo.divide %2950, %4941 : tensor<3x3x512x512xf32>
    %4943 = stablehlo.convert %arg295 : tensor<f32>
    %4944 = stablehlo.broadcast_in_dim %4943, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4945 = stablehlo.divide %4471, %4944 : tensor<64xf32>
    %4946 = stablehlo.convert %arg295 : tensor<f32>
    %4947 = stablehlo.broadcast_in_dim %4946, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4948 = stablehlo.divide %4479, %4947 : tensor<64xf32>
    %4949 = stablehlo.convert %arg295 : tensor<f32>
    %4950 = stablehlo.broadcast_in_dim %4949, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4951 = stablehlo.divide %4419, %4950 : tensor<64xf32>
    %4952 = stablehlo.convert %arg295 : tensor<f32>
    %4953 = stablehlo.broadcast_in_dim %4952, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4954 = stablehlo.divide %4427, %4953 : tensor<64xf32>
    %4955 = stablehlo.convert %arg295 : tensor<f32>
    %4956 = stablehlo.broadcast_in_dim %4955, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %4957 = stablehlo.divide %4518, %4956 : tensor<3x3x64x64xf32>
    %4958 = stablehlo.convert %arg295 : tensor<f32>
    %4959 = stablehlo.broadcast_in_dim %4958, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %4960 = stablehlo.divide %4465, %4959 : tensor<3x3x64x64xf32>
    %4961 = stablehlo.convert %arg295 : tensor<f32>
    %4962 = stablehlo.broadcast_in_dim %4961, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4963 = stablehlo.divide %4366, %4962 : tensor<128xf32>
    %4964 = stablehlo.convert %arg295 : tensor<f32>
    %4965 = stablehlo.broadcast_in_dim %4964, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4966 = stablehlo.divide %4374, %4965 : tensor<128xf32>
    %4967 = stablehlo.convert %arg295 : tensor<f32>
    %4968 = stablehlo.broadcast_in_dim %4967, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4969 = stablehlo.divide %4314, %4968 : tensor<128xf32>
    %4970 = stablehlo.convert %arg295 : tensor<f32>
    %4971 = stablehlo.broadcast_in_dim %4970, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4972 = stablehlo.divide %4322, %4971 : tensor<128xf32>
    %4973 = stablehlo.convert %arg295 : tensor<f32>
    %4974 = stablehlo.broadcast_in_dim %4973, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %4975 = stablehlo.divide %4413, %4974 : tensor<3x3x64x128xf32>
    %4976 = stablehlo.convert %arg295 : tensor<f32>
    %4977 = stablehlo.broadcast_in_dim %4976, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %4978 = stablehlo.divide %4360, %4977 : tensor<3x3x128x128xf32>
    %4979 = stablehlo.convert %arg295 : tensor<f32>
    %4980 = stablehlo.broadcast_in_dim %4979, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %4981 = stablehlo.divide %4310, %4980 : tensor<1x1x64x128xf32>
    %4982 = stablehlo.convert %arg295 : tensor<f32>
    %4983 = stablehlo.broadcast_in_dim %4982, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4984 = stablehlo.divide %4264, %4983 : tensor<128xf32>
    %4985 = stablehlo.convert %arg295 : tensor<f32>
    %4986 = stablehlo.broadcast_in_dim %4985, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4987 = stablehlo.divide %4272, %4986 : tensor<128xf32>
    %4988 = stablehlo.convert %arg295 : tensor<f32>
    %4989 = stablehlo.broadcast_in_dim %4988, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4990 = stablehlo.divide %4211, %4989 : tensor<128xf32>
    %4991 = stablehlo.convert %arg295 : tensor<f32>
    %4992 = stablehlo.broadcast_in_dim %4991, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4993 = stablehlo.divide %4219, %4992 : tensor<128xf32>
    %4994 = stablehlo.convert %arg295 : tensor<f32>
    %4995 = stablehlo.broadcast_in_dim %4994, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4996 = stablehlo.divide %4159, %4995 : tensor<128xf32>
    %4997 = stablehlo.convert %arg295 : tensor<f32>
    %4998 = stablehlo.broadcast_in_dim %4997, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4999 = stablehlo.divide %4167, %4998 : tensor<128xf32>
    %5000 = stablehlo.convert %arg295 : tensor<f32>
    %5001 = stablehlo.broadcast_in_dim %5000, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5002 = stablehlo.divide %4258, %5001 : tensor<3x3x128x128xf32>
    %5003 = stablehlo.convert %arg295 : tensor<f32>
    %5004 = stablehlo.broadcast_in_dim %5003, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5005 = stablehlo.divide %4205, %5004 : tensor<3x3x128x128xf32>
    %5006 = stablehlo.convert %arg295 : tensor<f32>
    %5007 = stablehlo.broadcast_in_dim %5006, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5008 = stablehlo.divide %4106, %5007 : tensor<128xf32>
    %5009 = stablehlo.convert %arg295 : tensor<f32>
    %5010 = stablehlo.broadcast_in_dim %5009, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5011 = stablehlo.divide %4114, %5010 : tensor<128xf32>
    %5012 = stablehlo.convert %arg295 : tensor<f32>
    %5013 = stablehlo.broadcast_in_dim %5012, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5014 = stablehlo.divide %4054, %5013 : tensor<128xf32>
    %5015 = stablehlo.convert %arg295 : tensor<f32>
    %5016 = stablehlo.broadcast_in_dim %5015, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5017 = stablehlo.divide %4062, %5016 : tensor<128xf32>
    %5018 = stablehlo.convert %arg295 : tensor<f32>
    %5019 = stablehlo.broadcast_in_dim %5018, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5020 = stablehlo.divide %4153, %5019 : tensor<3x3x128x128xf32>
    %5021 = stablehlo.convert %arg295 : tensor<f32>
    %5022 = stablehlo.broadcast_in_dim %5021, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5023 = stablehlo.divide %4100, %5022 : tensor<3x3x128x128xf32>
    %5024 = stablehlo.convert %arg295 : tensor<f32>
    %5025 = stablehlo.broadcast_in_dim %5024, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5026 = stablehlo.divide %4001, %5025 : tensor<128xf32>
    %5027 = stablehlo.convert %arg295 : tensor<f32>
    %5028 = stablehlo.broadcast_in_dim %5027, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5029 = stablehlo.divide %4009, %5028 : tensor<128xf32>
    %5030 = stablehlo.convert %arg295 : tensor<f32>
    %5031 = stablehlo.broadcast_in_dim %5030, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5032 = stablehlo.divide %3949, %5031 : tensor<128xf32>
    %5033 = stablehlo.convert %arg295 : tensor<f32>
    %5034 = stablehlo.broadcast_in_dim %5033, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5035 = stablehlo.divide %3957, %5034 : tensor<128xf32>
    %5036 = stablehlo.convert %arg295 : tensor<f32>
    %5037 = stablehlo.broadcast_in_dim %5036, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5038 = stablehlo.divide %4048, %5037 : tensor<3x3x128x128xf32>
    %5039 = stablehlo.convert %arg295 : tensor<f32>
    %5040 = stablehlo.broadcast_in_dim %5039, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5041 = stablehlo.divide %3995, %5040 : tensor<3x3x128x128xf32>
    %5042 = stablehlo.convert %arg295 : tensor<f32>
    %5043 = stablehlo.broadcast_in_dim %5042, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5044 = stablehlo.divide %3896, %5043 : tensor<256xf32>
    %5045 = stablehlo.convert %arg295 : tensor<f32>
    %5046 = stablehlo.broadcast_in_dim %5045, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5047 = stablehlo.divide %3904, %5046 : tensor<256xf32>
    %5048 = stablehlo.convert %arg295 : tensor<f32>
    %5049 = stablehlo.broadcast_in_dim %5048, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5050 = stablehlo.divide %3844, %5049 : tensor<256xf32>
    %5051 = stablehlo.convert %arg295 : tensor<f32>
    %5052 = stablehlo.broadcast_in_dim %5051, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5053 = stablehlo.divide %3852, %5052 : tensor<256xf32>
    %5054 = stablehlo.convert %arg295 : tensor<f32>
    %5055 = stablehlo.broadcast_in_dim %5054, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %5056 = stablehlo.divide %3943, %5055 : tensor<3x3x128x256xf32>
    %5057 = stablehlo.convert %arg295 : tensor<f32>
    %5058 = stablehlo.broadcast_in_dim %5057, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5059 = stablehlo.divide %3890, %5058 : tensor<3x3x256x256xf32>
    %5060 = stablehlo.convert %arg295 : tensor<f32>
    %5061 = stablehlo.broadcast_in_dim %5060, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %5062 = stablehlo.divide %3840, %5061 : tensor<1x1x128x256xf32>
    %5063 = stablehlo.convert %arg295 : tensor<f32>
    %5064 = stablehlo.broadcast_in_dim %5063, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5065 = stablehlo.divide %3794, %5064 : tensor<256xf32>
    %5066 = stablehlo.convert %arg295 : tensor<f32>
    %5067 = stablehlo.broadcast_in_dim %5066, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5068 = stablehlo.divide %3802, %5067 : tensor<256xf32>
    %5069 = stablehlo.convert %arg295 : tensor<f32>
    %5070 = stablehlo.broadcast_in_dim %5069, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5071 = stablehlo.divide %3741, %5070 : tensor<256xf32>
    %5072 = stablehlo.convert %arg295 : tensor<f32>
    %5073 = stablehlo.broadcast_in_dim %5072, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5074 = stablehlo.divide %3749, %5073 : tensor<256xf32>
    %5075 = stablehlo.convert %arg295 : tensor<f32>
    %5076 = stablehlo.broadcast_in_dim %5075, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5077 = stablehlo.divide %3689, %5076 : tensor<256xf32>
    %5078 = stablehlo.convert %arg295 : tensor<f32>
    %5079 = stablehlo.broadcast_in_dim %5078, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5080 = stablehlo.divide %3697, %5079 : tensor<256xf32>
    %5081 = stablehlo.convert %arg295 : tensor<f32>
    %5082 = stablehlo.broadcast_in_dim %5081, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5083 = stablehlo.divide %3788, %5082 : tensor<3x3x256x256xf32>
    %5084 = stablehlo.convert %arg295 : tensor<f32>
    %5085 = stablehlo.broadcast_in_dim %5084, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5086 = stablehlo.divide %3735, %5085 : tensor<3x3x256x256xf32>
    %5087 = stablehlo.convert %arg295 : tensor<f32>
    %5088 = stablehlo.broadcast_in_dim %5087, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5089 = stablehlo.divide %3636, %5088 : tensor<256xf32>
    %5090 = stablehlo.convert %arg295 : tensor<f32>
    %5091 = stablehlo.broadcast_in_dim %5090, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5092 = stablehlo.divide %3644, %5091 : tensor<256xf32>
    %5093 = stablehlo.convert %arg295 : tensor<f32>
    %5094 = stablehlo.broadcast_in_dim %5093, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5095 = stablehlo.divide %3584, %5094 : tensor<256xf32>
    %5096 = stablehlo.convert %arg295 : tensor<f32>
    %5097 = stablehlo.broadcast_in_dim %5096, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5098 = stablehlo.divide %3592, %5097 : tensor<256xf32>
    %5099 = stablehlo.convert %arg295 : tensor<f32>
    %5100 = stablehlo.broadcast_in_dim %5099, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5101 = stablehlo.divide %3683, %5100 : tensor<3x3x256x256xf32>
    %5102 = stablehlo.convert %arg295 : tensor<f32>
    %5103 = stablehlo.broadcast_in_dim %5102, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5104 = stablehlo.divide %3630, %5103 : tensor<3x3x256x256xf32>
    %5105 = stablehlo.convert %arg295 : tensor<f32>
    %5106 = stablehlo.broadcast_in_dim %5105, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5107 = stablehlo.divide %4737, %5106 : tensor<64xf32>
    %5108 = stablehlo.convert %arg295 : tensor<f32>
    %5109 = stablehlo.broadcast_in_dim %5108, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5110 = stablehlo.divide %4745, %5109 : tensor<64xf32>
    %5111 = stablehlo.convert %arg295 : tensor<f32>
    %5112 = stablehlo.broadcast_in_dim %5111, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %5113 = stablehlo.divide %4781, %5112 : tensor<7x7x3x64xf32>
    %5114 = "stablehlo.all_reduce"(%4786) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<1000xf32>) -> tensor<1000xf32>
    %5115 = "stablehlo.all_reduce"(%4789) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512x1000xf32>) -> tensor<512x1000xf32>
    %5116 = "stablehlo.all_reduce"(%4792) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5117 = "stablehlo.all_reduce"(%4795) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5118 = "stablehlo.all_reduce"(%4798) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5119 = "stablehlo.all_reduce"(%4801) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5120 = "stablehlo.all_reduce"(%4804) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %5121 = "stablehlo.all_reduce"(%4807) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %5122 = "stablehlo.all_reduce"(%4810) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5123 = "stablehlo.all_reduce"(%4813) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5124 = "stablehlo.all_reduce"(%4816) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5125 = "stablehlo.all_reduce"(%4819) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5126 = "stablehlo.all_reduce"(%4822) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %5127 = "stablehlo.all_reduce"(%4825) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %5128 = "stablehlo.all_reduce"(%4828) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5129 = "stablehlo.all_reduce"(%4831) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5130 = "stablehlo.all_reduce"(%4834) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5131 = "stablehlo.all_reduce"(%4837) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5132 = "stablehlo.all_reduce"(%4840) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5133 = "stablehlo.all_reduce"(%4843) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5134 = "stablehlo.all_reduce"(%4846) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5135 = "stablehlo.all_reduce"(%4849) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5136 = "stablehlo.all_reduce"(%4852) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5137 = "stablehlo.all_reduce"(%4855) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5138 = "stablehlo.all_reduce"(%4858) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5139 = "stablehlo.all_reduce"(%4861) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5140 = "stablehlo.all_reduce"(%4864) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5141 = "stablehlo.all_reduce"(%4867) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5142 = "stablehlo.all_reduce"(%4870) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5143 = "stablehlo.all_reduce"(%4873) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5144 = "stablehlo.all_reduce"(%4876) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5145 = "stablehlo.all_reduce"(%4879) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5146 = "stablehlo.all_reduce"(%4882) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5147 = "stablehlo.all_reduce"(%4885) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5148 = "stablehlo.all_reduce"(%4888) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5149 = "stablehlo.all_reduce"(%4891) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5150 = "stablehlo.all_reduce"(%4894) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
    %5151 = "stablehlo.all_reduce"(%4897) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %5152 = "stablehlo.all_reduce"(%4900) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    %5153 = "stablehlo.all_reduce"(%4903) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5154 = "stablehlo.all_reduce"(%4906) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5155 = "stablehlo.all_reduce"(%4909) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5156 = "stablehlo.all_reduce"(%4912) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5157 = "stablehlo.all_reduce"(%4915) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5158 = "stablehlo.all_reduce"(%4918) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5159 = "stablehlo.all_reduce"(%4921) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %5160 = "stablehlo.all_reduce"(%4924) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %5161 = "stablehlo.all_reduce"(%4927) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5162 = "stablehlo.all_reduce"(%4930) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5163 = "stablehlo.all_reduce"(%4933) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5164 = "stablehlo.all_reduce"(%4936) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5165 = "stablehlo.all_reduce"(%4939) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %5166 = "stablehlo.all_reduce"(%4942) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %5167 = "stablehlo.all_reduce"(%4945) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5168 = "stablehlo.all_reduce"(%4948) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5169 = "stablehlo.all_reduce"(%4951) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5170 = "stablehlo.all_reduce"(%4954) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5171 = "stablehlo.all_reduce"(%4957) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %5172 = "stablehlo.all_reduce"(%4960) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %5173 = "stablehlo.all_reduce"(%4963) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5174 = "stablehlo.all_reduce"(%4966) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5175 = "stablehlo.all_reduce"(%4969) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5176 = "stablehlo.all_reduce"(%4972) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5177 = "stablehlo.all_reduce"(%4975) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
    %5178 = "stablehlo.all_reduce"(%4978) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %5179 = "stablehlo.all_reduce"(%4981) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
    %5180 = "stablehlo.all_reduce"(%4984) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5181 = "stablehlo.all_reduce"(%4987) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5182 = "stablehlo.all_reduce"(%4990) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5183 = "stablehlo.all_reduce"(%4993) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5184 = "stablehlo.all_reduce"(%4996) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5185 = "stablehlo.all_reduce"(%4999) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5186 = "stablehlo.all_reduce"(%5002) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %5187 = "stablehlo.all_reduce"(%5005) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %5188 = "stablehlo.all_reduce"(%5008) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5189 = "stablehlo.all_reduce"(%5011) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5190 = "stablehlo.all_reduce"(%5014) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5191 = "stablehlo.all_reduce"(%5017) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5192 = "stablehlo.all_reduce"(%5020) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %5193 = "stablehlo.all_reduce"(%5023) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %5194 = "stablehlo.all_reduce"(%5026) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5195 = "stablehlo.all_reduce"(%5029) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5196 = "stablehlo.all_reduce"(%5032) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5197 = "stablehlo.all_reduce"(%5035) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5198 = "stablehlo.all_reduce"(%5038) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %5199 = "stablehlo.all_reduce"(%5041) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %5200 = "stablehlo.all_reduce"(%5044) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5201 = "stablehlo.all_reduce"(%5047) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5202 = "stablehlo.all_reduce"(%5050) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5203 = "stablehlo.all_reduce"(%5053) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5204 = "stablehlo.all_reduce"(%5056) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
    %5205 = "stablehlo.all_reduce"(%5059) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5206 = "stablehlo.all_reduce"(%5062) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
    %5207 = "stablehlo.all_reduce"(%5065) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5208 = "stablehlo.all_reduce"(%5068) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5209 = "stablehlo.all_reduce"(%5071) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5210 = "stablehlo.all_reduce"(%5074) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5211 = "stablehlo.all_reduce"(%5077) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5212 = "stablehlo.all_reduce"(%5080) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5213 = "stablehlo.all_reduce"(%5083) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5214 = "stablehlo.all_reduce"(%5086) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5215 = "stablehlo.all_reduce"(%5089) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5216 = "stablehlo.all_reduce"(%5092) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5217 = "stablehlo.all_reduce"(%5095) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5218 = "stablehlo.all_reduce"(%5098) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5219 = "stablehlo.all_reduce"(%5101) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5220 = "stablehlo.all_reduce"(%5104) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5221 = "stablehlo.all_reduce"(%5107) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5222 = "stablehlo.all_reduce"(%5110) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5223 = "stablehlo.all_reduce"(%5113) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %5224 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %5225 = stablehlo.divide %5114, %5224 : tensor<1000xf32>
    %5226 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %5227 = stablehlo.divide %5115, %5226 : tensor<512x1000xf32>
    %5228 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5229 = stablehlo.divide %5116, %5228 : tensor<64xf32>
    %5230 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5231 = stablehlo.divide %5117, %5230 : tensor<64xf32>
    %5232 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5233 = stablehlo.divide %5118, %5232 : tensor<64xf32>
    %5234 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5235 = stablehlo.divide %5119, %5234 : tensor<64xf32>
    %5236 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %5237 = stablehlo.divide %5120, %5236 : tensor<3x3x64x64xf32>
    %5238 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %5239 = stablehlo.divide %5121, %5238 : tensor<3x3x64x64xf32>
    %5240 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5241 = stablehlo.divide %5122, %5240 : tensor<64xf32>
    %5242 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5243 = stablehlo.divide %5123, %5242 : tensor<64xf32>
    %5244 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5245 = stablehlo.divide %5124, %5244 : tensor<64xf32>
    %5246 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5247 = stablehlo.divide %5125, %5246 : tensor<64xf32>
    %5248 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %5249 = stablehlo.divide %5126, %5248 : tensor<3x3x64x64xf32>
    %5250 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %5251 = stablehlo.divide %5127, %5250 : tensor<3x3x64x64xf32>
    %5252 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5253 = stablehlo.divide %5128, %5252 : tensor<256xf32>
    %5254 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5255 = stablehlo.divide %5129, %5254 : tensor<256xf32>
    %5256 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5257 = stablehlo.divide %5130, %5256 : tensor<256xf32>
    %5258 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5259 = stablehlo.divide %5131, %5258 : tensor<256xf32>
    %5260 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5261 = stablehlo.divide %5132, %5260 : tensor<3x3x256x256xf32>
    %5262 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5263 = stablehlo.divide %5133, %5262 : tensor<3x3x256x256xf32>
    %5264 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5265 = stablehlo.divide %5134, %5264 : tensor<256xf32>
    %5266 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5267 = stablehlo.divide %5135, %5266 : tensor<256xf32>
    %5268 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5269 = stablehlo.divide %5136, %5268 : tensor<256xf32>
    %5270 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5271 = stablehlo.divide %5137, %5270 : tensor<256xf32>
    %5272 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5273 = stablehlo.divide %5138, %5272 : tensor<3x3x256x256xf32>
    %5274 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5275 = stablehlo.divide %5139, %5274 : tensor<3x3x256x256xf32>
    %5276 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5277 = stablehlo.divide %5140, %5276 : tensor<256xf32>
    %5278 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5279 = stablehlo.divide %5141, %5278 : tensor<256xf32>
    %5280 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5281 = stablehlo.divide %5142, %5280 : tensor<256xf32>
    %5282 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5283 = stablehlo.divide %5143, %5282 : tensor<256xf32>
    %5284 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5285 = stablehlo.divide %5144, %5284 : tensor<3x3x256x256xf32>
    %5286 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5287 = stablehlo.divide %5145, %5286 : tensor<3x3x256x256xf32>
    %5288 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5289 = stablehlo.divide %5146, %5288 : tensor<512xf32>
    %5290 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5291 = stablehlo.divide %5147, %5290 : tensor<512xf32>
    %5292 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5293 = stablehlo.divide %5148, %5292 : tensor<512xf32>
    %5294 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5295 = stablehlo.divide %5149, %5294 : tensor<512xf32>
    %5296 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %5297 = stablehlo.divide %5150, %5296 : tensor<3x3x256x512xf32>
    %5298 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %5299 = stablehlo.divide %5151, %5298 : tensor<3x3x512x512xf32>
    %5300 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %5301 = stablehlo.divide %5152, %5300 : tensor<1x1x256x512xf32>
    %5302 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5303 = stablehlo.divide %5153, %5302 : tensor<512xf32>
    %5304 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5305 = stablehlo.divide %5154, %5304 : tensor<512xf32>
    %5306 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5307 = stablehlo.divide %5155, %5306 : tensor<512xf32>
    %5308 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5309 = stablehlo.divide %5156, %5308 : tensor<512xf32>
    %5310 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5311 = stablehlo.divide %5157, %5310 : tensor<512xf32>
    %5312 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5313 = stablehlo.divide %5158, %5312 : tensor<512xf32>
    %5314 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %5315 = stablehlo.divide %5159, %5314 : tensor<3x3x512x512xf32>
    %5316 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %5317 = stablehlo.divide %5160, %5316 : tensor<3x3x512x512xf32>
    %5318 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5319 = stablehlo.divide %5161, %5318 : tensor<512xf32>
    %5320 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5321 = stablehlo.divide %5162, %5320 : tensor<512xf32>
    %5322 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5323 = stablehlo.divide %5163, %5322 : tensor<512xf32>
    %5324 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5325 = stablehlo.divide %5164, %5324 : tensor<512xf32>
    %5326 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %5327 = stablehlo.divide %5165, %5326 : tensor<3x3x512x512xf32>
    %5328 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %5329 = stablehlo.divide %5166, %5328 : tensor<3x3x512x512xf32>
    %5330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5331 = stablehlo.divide %5167, %5330 : tensor<64xf32>
    %5332 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5333 = stablehlo.divide %5168, %5332 : tensor<64xf32>
    %5334 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5335 = stablehlo.divide %5169, %5334 : tensor<64xf32>
    %5336 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5337 = stablehlo.divide %5170, %5336 : tensor<64xf32>
    %5338 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %5339 = stablehlo.divide %5171, %5338 : tensor<3x3x64x64xf32>
    %5340 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %5341 = stablehlo.divide %5172, %5340 : tensor<3x3x64x64xf32>
    %5342 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5343 = stablehlo.divide %5173, %5342 : tensor<128xf32>
    %5344 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5345 = stablehlo.divide %5174, %5344 : tensor<128xf32>
    %5346 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5347 = stablehlo.divide %5175, %5346 : tensor<128xf32>
    %5348 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5349 = stablehlo.divide %5176, %5348 : tensor<128xf32>
    %5350 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %5351 = stablehlo.divide %5177, %5350 : tensor<3x3x64x128xf32>
    %5352 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5353 = stablehlo.divide %5178, %5352 : tensor<3x3x128x128xf32>
    %5354 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %5355 = stablehlo.divide %5179, %5354 : tensor<1x1x64x128xf32>
    %5356 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5357 = stablehlo.divide %5180, %5356 : tensor<128xf32>
    %5358 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5359 = stablehlo.divide %5181, %5358 : tensor<128xf32>
    %5360 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5361 = stablehlo.divide %5182, %5360 : tensor<128xf32>
    %5362 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5363 = stablehlo.divide %5183, %5362 : tensor<128xf32>
    %5364 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5365 = stablehlo.divide %5184, %5364 : tensor<128xf32>
    %5366 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5367 = stablehlo.divide %5185, %5366 : tensor<128xf32>
    %5368 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5369 = stablehlo.divide %5186, %5368 : tensor<3x3x128x128xf32>
    %5370 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5371 = stablehlo.divide %5187, %5370 : tensor<3x3x128x128xf32>
    %5372 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5373 = stablehlo.divide %5188, %5372 : tensor<128xf32>
    %5374 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5375 = stablehlo.divide %5189, %5374 : tensor<128xf32>
    %5376 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5377 = stablehlo.divide %5190, %5376 : tensor<128xf32>
    %5378 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5379 = stablehlo.divide %5191, %5378 : tensor<128xf32>
    %5380 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5381 = stablehlo.divide %5192, %5380 : tensor<3x3x128x128xf32>
    %5382 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5383 = stablehlo.divide %5193, %5382 : tensor<3x3x128x128xf32>
    %5384 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5385 = stablehlo.divide %5194, %5384 : tensor<128xf32>
    %5386 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5387 = stablehlo.divide %5195, %5386 : tensor<128xf32>
    %5388 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5389 = stablehlo.divide %5196, %5388 : tensor<128xf32>
    %5390 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5391 = stablehlo.divide %5197, %5390 : tensor<128xf32>
    %5392 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5393 = stablehlo.divide %5198, %5392 : tensor<3x3x128x128xf32>
    %5394 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %5395 = stablehlo.divide %5199, %5394 : tensor<3x3x128x128xf32>
    %5396 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5397 = stablehlo.divide %5200, %5396 : tensor<256xf32>
    %5398 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5399 = stablehlo.divide %5201, %5398 : tensor<256xf32>
    %5400 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5401 = stablehlo.divide %5202, %5400 : tensor<256xf32>
    %5402 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5403 = stablehlo.divide %5203, %5402 : tensor<256xf32>
    %5404 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %5405 = stablehlo.divide %5204, %5404 : tensor<3x3x128x256xf32>
    %5406 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5407 = stablehlo.divide %5205, %5406 : tensor<3x3x256x256xf32>
    %5408 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %5409 = stablehlo.divide %5206, %5408 : tensor<1x1x128x256xf32>
    %5410 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5411 = stablehlo.divide %5207, %5410 : tensor<256xf32>
    %5412 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5413 = stablehlo.divide %5208, %5412 : tensor<256xf32>
    %5414 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5415 = stablehlo.divide %5209, %5414 : tensor<256xf32>
    %5416 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5417 = stablehlo.divide %5210, %5416 : tensor<256xf32>
    %5418 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5419 = stablehlo.divide %5211, %5418 : tensor<256xf32>
    %5420 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5421 = stablehlo.divide %5212, %5420 : tensor<256xf32>
    %5422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5423 = stablehlo.divide %5213, %5422 : tensor<3x3x256x256xf32>
    %5424 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5425 = stablehlo.divide %5214, %5424 : tensor<3x3x256x256xf32>
    %5426 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5427 = stablehlo.divide %5215, %5426 : tensor<256xf32>
    %5428 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5429 = stablehlo.divide %5216, %5428 : tensor<256xf32>
    %5430 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5431 = stablehlo.divide %5217, %5430 : tensor<256xf32>
    %5432 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5433 = stablehlo.divide %5218, %5432 : tensor<256xf32>
    %5434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5435 = stablehlo.divide %5219, %5434 : tensor<3x3x256x256xf32>
    %5436 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %5437 = stablehlo.divide %5220, %5436 : tensor<3x3x256x256xf32>
    %5438 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5439 = stablehlo.divide %5221, %5438 : tensor<64xf32>
    %5440 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5441 = stablehlo.divide %5222, %5440 : tensor<64xf32>
    %5442 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %5443 = stablehlo.divide %5223, %5442 : tensor<7x7x3x64xf32>
    %5444 = stablehlo.is_finite %5225 : (tensor<1000xf32>) -> tensor<1000xi1>
    %c_392 = stablehlo.constant dense<true> : tensor<i1>
    %5445 = stablehlo.reduce(%5444 init: %c_392) applies stablehlo.and across dimensions = [0] : (tensor<1000xi1>, tensor<i1>) -> tensor<i1>
    %c_393 = stablehlo.constant dense<true> : tensor<i1>
    %5446 = stablehlo.and %c_393, %5445 : tensor<i1>
    %5447 = stablehlo.is_finite %5227 : (tensor<512x1000xf32>) -> tensor<512x1000xi1>
    %c_394 = stablehlo.constant dense<true> : tensor<i1>
    %5448 = stablehlo.reduce(%5447 init: %c_394) applies stablehlo.and across dimensions = [0, 1] : (tensor<512x1000xi1>, tensor<i1>) -> tensor<i1>
    %5449 = stablehlo.and %5446, %5448 : tensor<i1>
    %5450 = stablehlo.is_finite %5229 : (tensor<64xf32>) -> tensor<64xi1>
    %c_395 = stablehlo.constant dense<true> : tensor<i1>
    %5451 = stablehlo.reduce(%5450 init: %c_395) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5452 = stablehlo.and %5449, %5451 : tensor<i1>
    %5453 = stablehlo.is_finite %5231 : (tensor<64xf32>) -> tensor<64xi1>
    %c_396 = stablehlo.constant dense<true> : tensor<i1>
    %5454 = stablehlo.reduce(%5453 init: %c_396) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5455 = stablehlo.and %5452, %5454 : tensor<i1>
    %5456 = stablehlo.is_finite %5233 : (tensor<64xf32>) -> tensor<64xi1>
    %c_397 = stablehlo.constant dense<true> : tensor<i1>
    %5457 = stablehlo.reduce(%5456 init: %c_397) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5458 = stablehlo.and %5455, %5457 : tensor<i1>
    %5459 = stablehlo.is_finite %5235 : (tensor<64xf32>) -> tensor<64xi1>
    %c_398 = stablehlo.constant dense<true> : tensor<i1>
    %5460 = stablehlo.reduce(%5459 init: %c_398) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5461 = stablehlo.and %5458, %5460 : tensor<i1>
    %5462 = stablehlo.is_finite %5237 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_399 = stablehlo.constant dense<true> : tensor<i1>
    %5463 = stablehlo.reduce(%5462 init: %c_399) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %5464 = stablehlo.and %5461, %5463 : tensor<i1>
    %5465 = stablehlo.is_finite %5239 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_400 = stablehlo.constant dense<true> : tensor<i1>
    %5466 = stablehlo.reduce(%5465 init: %c_400) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %5467 = stablehlo.and %5464, %5466 : tensor<i1>
    %5468 = stablehlo.is_finite %5241 : (tensor<64xf32>) -> tensor<64xi1>
    %c_401 = stablehlo.constant dense<true> : tensor<i1>
    %5469 = stablehlo.reduce(%5468 init: %c_401) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5470 = stablehlo.and %5467, %5469 : tensor<i1>
    %5471 = stablehlo.is_finite %5243 : (tensor<64xf32>) -> tensor<64xi1>
    %c_402 = stablehlo.constant dense<true> : tensor<i1>
    %5472 = stablehlo.reduce(%5471 init: %c_402) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5473 = stablehlo.and %5470, %5472 : tensor<i1>
    %5474 = stablehlo.is_finite %5245 : (tensor<64xf32>) -> tensor<64xi1>
    %c_403 = stablehlo.constant dense<true> : tensor<i1>
    %5475 = stablehlo.reduce(%5474 init: %c_403) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5476 = stablehlo.and %5473, %5475 : tensor<i1>
    %5477 = stablehlo.is_finite %5247 : (tensor<64xf32>) -> tensor<64xi1>
    %c_404 = stablehlo.constant dense<true> : tensor<i1>
    %5478 = stablehlo.reduce(%5477 init: %c_404) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5479 = stablehlo.and %5476, %5478 : tensor<i1>
    %5480 = stablehlo.is_finite %5249 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_405 = stablehlo.constant dense<true> : tensor<i1>
    %5481 = stablehlo.reduce(%5480 init: %c_405) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %5482 = stablehlo.and %5479, %5481 : tensor<i1>
    %5483 = stablehlo.is_finite %5251 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_406 = stablehlo.constant dense<true> : tensor<i1>
    %5484 = stablehlo.reduce(%5483 init: %c_406) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %5485 = stablehlo.and %5482, %5484 : tensor<i1>
    %5486 = stablehlo.is_finite %5253 : (tensor<256xf32>) -> tensor<256xi1>
    %c_407 = stablehlo.constant dense<true> : tensor<i1>
    %5487 = stablehlo.reduce(%5486 init: %c_407) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5488 = stablehlo.and %5485, %5487 : tensor<i1>
    %5489 = stablehlo.is_finite %5255 : (tensor<256xf32>) -> tensor<256xi1>
    %c_408 = stablehlo.constant dense<true> : tensor<i1>
    %5490 = stablehlo.reduce(%5489 init: %c_408) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5491 = stablehlo.and %5488, %5490 : tensor<i1>
    %5492 = stablehlo.is_finite %5257 : (tensor<256xf32>) -> tensor<256xi1>
    %c_409 = stablehlo.constant dense<true> : tensor<i1>
    %5493 = stablehlo.reduce(%5492 init: %c_409) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5494 = stablehlo.and %5491, %5493 : tensor<i1>
    %5495 = stablehlo.is_finite %5259 : (tensor<256xf32>) -> tensor<256xi1>
    %c_410 = stablehlo.constant dense<true> : tensor<i1>
    %5496 = stablehlo.reduce(%5495 init: %c_410) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5497 = stablehlo.and %5494, %5496 : tensor<i1>
    %5498 = stablehlo.is_finite %5261 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_411 = stablehlo.constant dense<true> : tensor<i1>
    %5499 = stablehlo.reduce(%5498 init: %c_411) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5500 = stablehlo.and %5497, %5499 : tensor<i1>
    %5501 = stablehlo.is_finite %5263 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_412 = stablehlo.constant dense<true> : tensor<i1>
    %5502 = stablehlo.reduce(%5501 init: %c_412) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5503 = stablehlo.and %5500, %5502 : tensor<i1>
    %5504 = stablehlo.is_finite %5265 : (tensor<256xf32>) -> tensor<256xi1>
    %c_413 = stablehlo.constant dense<true> : tensor<i1>
    %5505 = stablehlo.reduce(%5504 init: %c_413) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5506 = stablehlo.and %5503, %5505 : tensor<i1>
    %5507 = stablehlo.is_finite %5267 : (tensor<256xf32>) -> tensor<256xi1>
    %c_414 = stablehlo.constant dense<true> : tensor<i1>
    %5508 = stablehlo.reduce(%5507 init: %c_414) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5509 = stablehlo.and %5506, %5508 : tensor<i1>
    %5510 = stablehlo.is_finite %5269 : (tensor<256xf32>) -> tensor<256xi1>
    %c_415 = stablehlo.constant dense<true> : tensor<i1>
    %5511 = stablehlo.reduce(%5510 init: %c_415) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5512 = stablehlo.and %5509, %5511 : tensor<i1>
    %5513 = stablehlo.is_finite %5271 : (tensor<256xf32>) -> tensor<256xi1>
    %c_416 = stablehlo.constant dense<true> : tensor<i1>
    %5514 = stablehlo.reduce(%5513 init: %c_416) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5515 = stablehlo.and %5512, %5514 : tensor<i1>
    %5516 = stablehlo.is_finite %5273 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_417 = stablehlo.constant dense<true> : tensor<i1>
    %5517 = stablehlo.reduce(%5516 init: %c_417) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5518 = stablehlo.and %5515, %5517 : tensor<i1>
    %5519 = stablehlo.is_finite %5275 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_418 = stablehlo.constant dense<true> : tensor<i1>
    %5520 = stablehlo.reduce(%5519 init: %c_418) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5521 = stablehlo.and %5518, %5520 : tensor<i1>
    %5522 = stablehlo.is_finite %5277 : (tensor<256xf32>) -> tensor<256xi1>
    %c_419 = stablehlo.constant dense<true> : tensor<i1>
    %5523 = stablehlo.reduce(%5522 init: %c_419) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5524 = stablehlo.and %5521, %5523 : tensor<i1>
    %5525 = stablehlo.is_finite %5279 : (tensor<256xf32>) -> tensor<256xi1>
    %c_420 = stablehlo.constant dense<true> : tensor<i1>
    %5526 = stablehlo.reduce(%5525 init: %c_420) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5527 = stablehlo.and %5524, %5526 : tensor<i1>
    %5528 = stablehlo.is_finite %5281 : (tensor<256xf32>) -> tensor<256xi1>
    %c_421 = stablehlo.constant dense<true> : tensor<i1>
    %5529 = stablehlo.reduce(%5528 init: %c_421) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5530 = stablehlo.and %5527, %5529 : tensor<i1>
    %5531 = stablehlo.is_finite %5283 : (tensor<256xf32>) -> tensor<256xi1>
    %c_422 = stablehlo.constant dense<true> : tensor<i1>
    %5532 = stablehlo.reduce(%5531 init: %c_422) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5533 = stablehlo.and %5530, %5532 : tensor<i1>
    %5534 = stablehlo.is_finite %5285 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_423 = stablehlo.constant dense<true> : tensor<i1>
    %5535 = stablehlo.reduce(%5534 init: %c_423) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5536 = stablehlo.and %5533, %5535 : tensor<i1>
    %5537 = stablehlo.is_finite %5287 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_424 = stablehlo.constant dense<true> : tensor<i1>
    %5538 = stablehlo.reduce(%5537 init: %c_424) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5539 = stablehlo.and %5536, %5538 : tensor<i1>
    %5540 = stablehlo.is_finite %5289 : (tensor<512xf32>) -> tensor<512xi1>
    %c_425 = stablehlo.constant dense<true> : tensor<i1>
    %5541 = stablehlo.reduce(%5540 init: %c_425) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5542 = stablehlo.and %5539, %5541 : tensor<i1>
    %5543 = stablehlo.is_finite %5291 : (tensor<512xf32>) -> tensor<512xi1>
    %c_426 = stablehlo.constant dense<true> : tensor<i1>
    %5544 = stablehlo.reduce(%5543 init: %c_426) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5545 = stablehlo.and %5542, %5544 : tensor<i1>
    %5546 = stablehlo.is_finite %5293 : (tensor<512xf32>) -> tensor<512xi1>
    %c_427 = stablehlo.constant dense<true> : tensor<i1>
    %5547 = stablehlo.reduce(%5546 init: %c_427) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5548 = stablehlo.and %5545, %5547 : tensor<i1>
    %5549 = stablehlo.is_finite %5295 : (tensor<512xf32>) -> tensor<512xi1>
    %c_428 = stablehlo.constant dense<true> : tensor<i1>
    %5550 = stablehlo.reduce(%5549 init: %c_428) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5551 = stablehlo.and %5548, %5550 : tensor<i1>
    %5552 = stablehlo.is_finite %5297 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xi1>
    %c_429 = stablehlo.constant dense<true> : tensor<i1>
    %5553 = stablehlo.reduce(%5552 init: %c_429) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x512xi1>, tensor<i1>) -> tensor<i1>
    %5554 = stablehlo.and %5551, %5553 : tensor<i1>
    %5555 = stablehlo.is_finite %5299 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
    %c_430 = stablehlo.constant dense<true> : tensor<i1>
    %5556 = stablehlo.reduce(%5555 init: %c_430) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
    %5557 = stablehlo.and %5554, %5556 : tensor<i1>
    %5558 = stablehlo.is_finite %5301 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xi1>
    %c_431 = stablehlo.constant dense<true> : tensor<i1>
    %5559 = stablehlo.reduce(%5558 init: %c_431) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x256x512xi1>, tensor<i1>) -> tensor<i1>
    %5560 = stablehlo.and %5557, %5559 : tensor<i1>
    %5561 = stablehlo.is_finite %5303 : (tensor<512xf32>) -> tensor<512xi1>
    %c_432 = stablehlo.constant dense<true> : tensor<i1>
    %5562 = stablehlo.reduce(%5561 init: %c_432) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5563 = stablehlo.and %5560, %5562 : tensor<i1>
    %5564 = stablehlo.is_finite %5305 : (tensor<512xf32>) -> tensor<512xi1>
    %c_433 = stablehlo.constant dense<true> : tensor<i1>
    %5565 = stablehlo.reduce(%5564 init: %c_433) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5566 = stablehlo.and %5563, %5565 : tensor<i1>
    %5567 = stablehlo.is_finite %5307 : (tensor<512xf32>) -> tensor<512xi1>
    %c_434 = stablehlo.constant dense<true> : tensor<i1>
    %5568 = stablehlo.reduce(%5567 init: %c_434) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5569 = stablehlo.and %5566, %5568 : tensor<i1>
    %5570 = stablehlo.is_finite %5309 : (tensor<512xf32>) -> tensor<512xi1>
    %c_435 = stablehlo.constant dense<true> : tensor<i1>
    %5571 = stablehlo.reduce(%5570 init: %c_435) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5572 = stablehlo.and %5569, %5571 : tensor<i1>
    %5573 = stablehlo.is_finite %5311 : (tensor<512xf32>) -> tensor<512xi1>
    %c_436 = stablehlo.constant dense<true> : tensor<i1>
    %5574 = stablehlo.reduce(%5573 init: %c_436) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5575 = stablehlo.and %5572, %5574 : tensor<i1>
    %5576 = stablehlo.is_finite %5313 : (tensor<512xf32>) -> tensor<512xi1>
    %c_437 = stablehlo.constant dense<true> : tensor<i1>
    %5577 = stablehlo.reduce(%5576 init: %c_437) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5578 = stablehlo.and %5575, %5577 : tensor<i1>
    %5579 = stablehlo.is_finite %5315 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
    %c_438 = stablehlo.constant dense<true> : tensor<i1>
    %5580 = stablehlo.reduce(%5579 init: %c_438) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
    %5581 = stablehlo.and %5578, %5580 : tensor<i1>
    %5582 = stablehlo.is_finite %5317 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
    %c_439 = stablehlo.constant dense<true> : tensor<i1>
    %5583 = stablehlo.reduce(%5582 init: %c_439) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
    %5584 = stablehlo.and %5581, %5583 : tensor<i1>
    %5585 = stablehlo.is_finite %5319 : (tensor<512xf32>) -> tensor<512xi1>
    %c_440 = stablehlo.constant dense<true> : tensor<i1>
    %5586 = stablehlo.reduce(%5585 init: %c_440) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5587 = stablehlo.and %5584, %5586 : tensor<i1>
    %5588 = stablehlo.is_finite %5321 : (tensor<512xf32>) -> tensor<512xi1>
    %c_441 = stablehlo.constant dense<true> : tensor<i1>
    %5589 = stablehlo.reduce(%5588 init: %c_441) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5590 = stablehlo.and %5587, %5589 : tensor<i1>
    %5591 = stablehlo.is_finite %5323 : (tensor<512xf32>) -> tensor<512xi1>
    %c_442 = stablehlo.constant dense<true> : tensor<i1>
    %5592 = stablehlo.reduce(%5591 init: %c_442) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5593 = stablehlo.and %5590, %5592 : tensor<i1>
    %5594 = stablehlo.is_finite %5325 : (tensor<512xf32>) -> tensor<512xi1>
    %c_443 = stablehlo.constant dense<true> : tensor<i1>
    %5595 = stablehlo.reduce(%5594 init: %c_443) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
    %5596 = stablehlo.and %5593, %5595 : tensor<i1>
    %5597 = stablehlo.is_finite %5327 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
    %c_444 = stablehlo.constant dense<true> : tensor<i1>
    %5598 = stablehlo.reduce(%5597 init: %c_444) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
    %5599 = stablehlo.and %5596, %5598 : tensor<i1>
    %5600 = stablehlo.is_finite %5329 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
    %c_445 = stablehlo.constant dense<true> : tensor<i1>
    %5601 = stablehlo.reduce(%5600 init: %c_445) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
    %5602 = stablehlo.and %5599, %5601 : tensor<i1>
    %5603 = stablehlo.is_finite %5331 : (tensor<64xf32>) -> tensor<64xi1>
    %c_446 = stablehlo.constant dense<true> : tensor<i1>
    %5604 = stablehlo.reduce(%5603 init: %c_446) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5605 = stablehlo.and %5602, %5604 : tensor<i1>
    %5606 = stablehlo.is_finite %5333 : (tensor<64xf32>) -> tensor<64xi1>
    %c_447 = stablehlo.constant dense<true> : tensor<i1>
    %5607 = stablehlo.reduce(%5606 init: %c_447) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5608 = stablehlo.and %5605, %5607 : tensor<i1>
    %5609 = stablehlo.is_finite %5335 : (tensor<64xf32>) -> tensor<64xi1>
    %c_448 = stablehlo.constant dense<true> : tensor<i1>
    %5610 = stablehlo.reduce(%5609 init: %c_448) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5611 = stablehlo.and %5608, %5610 : tensor<i1>
    %5612 = stablehlo.is_finite %5337 : (tensor<64xf32>) -> tensor<64xi1>
    %c_449 = stablehlo.constant dense<true> : tensor<i1>
    %5613 = stablehlo.reduce(%5612 init: %c_449) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5614 = stablehlo.and %5611, %5613 : tensor<i1>
    %5615 = stablehlo.is_finite %5339 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_450 = stablehlo.constant dense<true> : tensor<i1>
    %5616 = stablehlo.reduce(%5615 init: %c_450) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %5617 = stablehlo.and %5614, %5616 : tensor<i1>
    %5618 = stablehlo.is_finite %5341 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
    %c_451 = stablehlo.constant dense<true> : tensor<i1>
    %5619 = stablehlo.reduce(%5618 init: %c_451) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
    %5620 = stablehlo.and %5617, %5619 : tensor<i1>
    %5621 = stablehlo.is_finite %5343 : (tensor<128xf32>) -> tensor<128xi1>
    %c_452 = stablehlo.constant dense<true> : tensor<i1>
    %5622 = stablehlo.reduce(%5621 init: %c_452) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5623 = stablehlo.and %5620, %5622 : tensor<i1>
    %5624 = stablehlo.is_finite %5345 : (tensor<128xf32>) -> tensor<128xi1>
    %c_453 = stablehlo.constant dense<true> : tensor<i1>
    %5625 = stablehlo.reduce(%5624 init: %c_453) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5626 = stablehlo.and %5623, %5625 : tensor<i1>
    %5627 = stablehlo.is_finite %5347 : (tensor<128xf32>) -> tensor<128xi1>
    %c_454 = stablehlo.constant dense<true> : tensor<i1>
    %5628 = stablehlo.reduce(%5627 init: %c_454) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5629 = stablehlo.and %5626, %5628 : tensor<i1>
    %5630 = stablehlo.is_finite %5349 : (tensor<128xf32>) -> tensor<128xi1>
    %c_455 = stablehlo.constant dense<true> : tensor<i1>
    %5631 = stablehlo.reduce(%5630 init: %c_455) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5632 = stablehlo.and %5629, %5631 : tensor<i1>
    %5633 = stablehlo.is_finite %5351 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xi1>
    %c_456 = stablehlo.constant dense<true> : tensor<i1>
    %5634 = stablehlo.reduce(%5633 init: %c_456) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x128xi1>, tensor<i1>) -> tensor<i1>
    %5635 = stablehlo.and %5632, %5634 : tensor<i1>
    %5636 = stablehlo.is_finite %5353 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_457 = stablehlo.constant dense<true> : tensor<i1>
    %5637 = stablehlo.reduce(%5636 init: %c_457) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %5638 = stablehlo.and %5635, %5637 : tensor<i1>
    %5639 = stablehlo.is_finite %5355 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xi1>
    %c_458 = stablehlo.constant dense<true> : tensor<i1>
    %5640 = stablehlo.reduce(%5639 init: %c_458) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x64x128xi1>, tensor<i1>) -> tensor<i1>
    %5641 = stablehlo.and %5638, %5640 : tensor<i1>
    %5642 = stablehlo.is_finite %5357 : (tensor<128xf32>) -> tensor<128xi1>
    %c_459 = stablehlo.constant dense<true> : tensor<i1>
    %5643 = stablehlo.reduce(%5642 init: %c_459) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5644 = stablehlo.and %5641, %5643 : tensor<i1>
    %5645 = stablehlo.is_finite %5359 : (tensor<128xf32>) -> tensor<128xi1>
    %c_460 = stablehlo.constant dense<true> : tensor<i1>
    %5646 = stablehlo.reduce(%5645 init: %c_460) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5647 = stablehlo.and %5644, %5646 : tensor<i1>
    %5648 = stablehlo.is_finite %5361 : (tensor<128xf32>) -> tensor<128xi1>
    %c_461 = stablehlo.constant dense<true> : tensor<i1>
    %5649 = stablehlo.reduce(%5648 init: %c_461) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5650 = stablehlo.and %5647, %5649 : tensor<i1>
    %5651 = stablehlo.is_finite %5363 : (tensor<128xf32>) -> tensor<128xi1>
    %c_462 = stablehlo.constant dense<true> : tensor<i1>
    %5652 = stablehlo.reduce(%5651 init: %c_462) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5653 = stablehlo.and %5650, %5652 : tensor<i1>
    %5654 = stablehlo.is_finite %5365 : (tensor<128xf32>) -> tensor<128xi1>
    %c_463 = stablehlo.constant dense<true> : tensor<i1>
    %5655 = stablehlo.reduce(%5654 init: %c_463) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5656 = stablehlo.and %5653, %5655 : tensor<i1>
    %5657 = stablehlo.is_finite %5367 : (tensor<128xf32>) -> tensor<128xi1>
    %c_464 = stablehlo.constant dense<true> : tensor<i1>
    %5658 = stablehlo.reduce(%5657 init: %c_464) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5659 = stablehlo.and %5656, %5658 : tensor<i1>
    %5660 = stablehlo.is_finite %5369 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_465 = stablehlo.constant dense<true> : tensor<i1>
    %5661 = stablehlo.reduce(%5660 init: %c_465) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %5662 = stablehlo.and %5659, %5661 : tensor<i1>
    %5663 = stablehlo.is_finite %5371 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_466 = stablehlo.constant dense<true> : tensor<i1>
    %5664 = stablehlo.reduce(%5663 init: %c_466) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %5665 = stablehlo.and %5662, %5664 : tensor<i1>
    %5666 = stablehlo.is_finite %5373 : (tensor<128xf32>) -> tensor<128xi1>
    %c_467 = stablehlo.constant dense<true> : tensor<i1>
    %5667 = stablehlo.reduce(%5666 init: %c_467) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5668 = stablehlo.and %5665, %5667 : tensor<i1>
    %5669 = stablehlo.is_finite %5375 : (tensor<128xf32>) -> tensor<128xi1>
    %c_468 = stablehlo.constant dense<true> : tensor<i1>
    %5670 = stablehlo.reduce(%5669 init: %c_468) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5671 = stablehlo.and %5668, %5670 : tensor<i1>
    %5672 = stablehlo.is_finite %5377 : (tensor<128xf32>) -> tensor<128xi1>
    %c_469 = stablehlo.constant dense<true> : tensor<i1>
    %5673 = stablehlo.reduce(%5672 init: %c_469) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5674 = stablehlo.and %5671, %5673 : tensor<i1>
    %5675 = stablehlo.is_finite %5379 : (tensor<128xf32>) -> tensor<128xi1>
    %c_470 = stablehlo.constant dense<true> : tensor<i1>
    %5676 = stablehlo.reduce(%5675 init: %c_470) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5677 = stablehlo.and %5674, %5676 : tensor<i1>
    %5678 = stablehlo.is_finite %5381 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_471 = stablehlo.constant dense<true> : tensor<i1>
    %5679 = stablehlo.reduce(%5678 init: %c_471) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %5680 = stablehlo.and %5677, %5679 : tensor<i1>
    %5681 = stablehlo.is_finite %5383 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_472 = stablehlo.constant dense<true> : tensor<i1>
    %5682 = stablehlo.reduce(%5681 init: %c_472) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %5683 = stablehlo.and %5680, %5682 : tensor<i1>
    %5684 = stablehlo.is_finite %5385 : (tensor<128xf32>) -> tensor<128xi1>
    %c_473 = stablehlo.constant dense<true> : tensor<i1>
    %5685 = stablehlo.reduce(%5684 init: %c_473) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5686 = stablehlo.and %5683, %5685 : tensor<i1>
    %5687 = stablehlo.is_finite %5387 : (tensor<128xf32>) -> tensor<128xi1>
    %c_474 = stablehlo.constant dense<true> : tensor<i1>
    %5688 = stablehlo.reduce(%5687 init: %c_474) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5689 = stablehlo.and %5686, %5688 : tensor<i1>
    %5690 = stablehlo.is_finite %5389 : (tensor<128xf32>) -> tensor<128xi1>
    %c_475 = stablehlo.constant dense<true> : tensor<i1>
    %5691 = stablehlo.reduce(%5690 init: %c_475) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5692 = stablehlo.and %5689, %5691 : tensor<i1>
    %5693 = stablehlo.is_finite %5391 : (tensor<128xf32>) -> tensor<128xi1>
    %c_476 = stablehlo.constant dense<true> : tensor<i1>
    %5694 = stablehlo.reduce(%5693 init: %c_476) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
    %5695 = stablehlo.and %5692, %5694 : tensor<i1>
    %5696 = stablehlo.is_finite %5393 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_477 = stablehlo.constant dense<true> : tensor<i1>
    %5697 = stablehlo.reduce(%5696 init: %c_477) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %5698 = stablehlo.and %5695, %5697 : tensor<i1>
    %5699 = stablehlo.is_finite %5395 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
    %c_478 = stablehlo.constant dense<true> : tensor<i1>
    %5700 = stablehlo.reduce(%5699 init: %c_478) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
    %5701 = stablehlo.and %5698, %5700 : tensor<i1>
    %5702 = stablehlo.is_finite %5397 : (tensor<256xf32>) -> tensor<256xi1>
    %c_479 = stablehlo.constant dense<true> : tensor<i1>
    %5703 = stablehlo.reduce(%5702 init: %c_479) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5704 = stablehlo.and %5701, %5703 : tensor<i1>
    %5705 = stablehlo.is_finite %5399 : (tensor<256xf32>) -> tensor<256xi1>
    %c_480 = stablehlo.constant dense<true> : tensor<i1>
    %5706 = stablehlo.reduce(%5705 init: %c_480) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5707 = stablehlo.and %5704, %5706 : tensor<i1>
    %5708 = stablehlo.is_finite %5401 : (tensor<256xf32>) -> tensor<256xi1>
    %c_481 = stablehlo.constant dense<true> : tensor<i1>
    %5709 = stablehlo.reduce(%5708 init: %c_481) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5710 = stablehlo.and %5707, %5709 : tensor<i1>
    %5711 = stablehlo.is_finite %5403 : (tensor<256xf32>) -> tensor<256xi1>
    %c_482 = stablehlo.constant dense<true> : tensor<i1>
    %5712 = stablehlo.reduce(%5711 init: %c_482) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5713 = stablehlo.and %5710, %5712 : tensor<i1>
    %5714 = stablehlo.is_finite %5405 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xi1>
    %c_483 = stablehlo.constant dense<true> : tensor<i1>
    %5715 = stablehlo.reduce(%5714 init: %c_483) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x256xi1>, tensor<i1>) -> tensor<i1>
    %5716 = stablehlo.and %5713, %5715 : tensor<i1>
    %5717 = stablehlo.is_finite %5407 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_484 = stablehlo.constant dense<true> : tensor<i1>
    %5718 = stablehlo.reduce(%5717 init: %c_484) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5719 = stablehlo.and %5716, %5718 : tensor<i1>
    %5720 = stablehlo.is_finite %5409 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xi1>
    %c_485 = stablehlo.constant dense<true> : tensor<i1>
    %5721 = stablehlo.reduce(%5720 init: %c_485) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x128x256xi1>, tensor<i1>) -> tensor<i1>
    %5722 = stablehlo.and %5719, %5721 : tensor<i1>
    %5723 = stablehlo.is_finite %5411 : (tensor<256xf32>) -> tensor<256xi1>
    %c_486 = stablehlo.constant dense<true> : tensor<i1>
    %5724 = stablehlo.reduce(%5723 init: %c_486) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5725 = stablehlo.and %5722, %5724 : tensor<i1>
    %5726 = stablehlo.is_finite %5413 : (tensor<256xf32>) -> tensor<256xi1>
    %c_487 = stablehlo.constant dense<true> : tensor<i1>
    %5727 = stablehlo.reduce(%5726 init: %c_487) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5728 = stablehlo.and %5725, %5727 : tensor<i1>
    %5729 = stablehlo.is_finite %5415 : (tensor<256xf32>) -> tensor<256xi1>
    %c_488 = stablehlo.constant dense<true> : tensor<i1>
    %5730 = stablehlo.reduce(%5729 init: %c_488) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5731 = stablehlo.and %5728, %5730 : tensor<i1>
    %5732 = stablehlo.is_finite %5417 : (tensor<256xf32>) -> tensor<256xi1>
    %c_489 = stablehlo.constant dense<true> : tensor<i1>
    %5733 = stablehlo.reduce(%5732 init: %c_489) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5734 = stablehlo.and %5731, %5733 : tensor<i1>
    %5735 = stablehlo.is_finite %5419 : (tensor<256xf32>) -> tensor<256xi1>
    %c_490 = stablehlo.constant dense<true> : tensor<i1>
    %5736 = stablehlo.reduce(%5735 init: %c_490) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5737 = stablehlo.and %5734, %5736 : tensor<i1>
    %5738 = stablehlo.is_finite %5421 : (tensor<256xf32>) -> tensor<256xi1>
    %c_491 = stablehlo.constant dense<true> : tensor<i1>
    %5739 = stablehlo.reduce(%5738 init: %c_491) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5740 = stablehlo.and %5737, %5739 : tensor<i1>
    %5741 = stablehlo.is_finite %5423 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_492 = stablehlo.constant dense<true> : tensor<i1>
    %5742 = stablehlo.reduce(%5741 init: %c_492) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5743 = stablehlo.and %5740, %5742 : tensor<i1>
    %5744 = stablehlo.is_finite %5425 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_493 = stablehlo.constant dense<true> : tensor<i1>
    %5745 = stablehlo.reduce(%5744 init: %c_493) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5746 = stablehlo.and %5743, %5745 : tensor<i1>
    %5747 = stablehlo.is_finite %5427 : (tensor<256xf32>) -> tensor<256xi1>
    %c_494 = stablehlo.constant dense<true> : tensor<i1>
    %5748 = stablehlo.reduce(%5747 init: %c_494) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5749 = stablehlo.and %5746, %5748 : tensor<i1>
    %5750 = stablehlo.is_finite %5429 : (tensor<256xf32>) -> tensor<256xi1>
    %c_495 = stablehlo.constant dense<true> : tensor<i1>
    %5751 = stablehlo.reduce(%5750 init: %c_495) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5752 = stablehlo.and %5749, %5751 : tensor<i1>
    %5753 = stablehlo.is_finite %5431 : (tensor<256xf32>) -> tensor<256xi1>
    %c_496 = stablehlo.constant dense<true> : tensor<i1>
    %5754 = stablehlo.reduce(%5753 init: %c_496) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5755 = stablehlo.and %5752, %5754 : tensor<i1>
    %5756 = stablehlo.is_finite %5433 : (tensor<256xf32>) -> tensor<256xi1>
    %c_497 = stablehlo.constant dense<true> : tensor<i1>
    %5757 = stablehlo.reduce(%5756 init: %c_497) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
    %5758 = stablehlo.and %5755, %5757 : tensor<i1>
    %5759 = stablehlo.is_finite %5435 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_498 = stablehlo.constant dense<true> : tensor<i1>
    %5760 = stablehlo.reduce(%5759 init: %c_498) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5761 = stablehlo.and %5758, %5760 : tensor<i1>
    %5762 = stablehlo.is_finite %5437 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
    %c_499 = stablehlo.constant dense<true> : tensor<i1>
    %5763 = stablehlo.reduce(%5762 init: %c_499) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
    %5764 = stablehlo.and %5761, %5763 : tensor<i1>
    %5765 = stablehlo.is_finite %5439 : (tensor<64xf32>) -> tensor<64xi1>
    %c_500 = stablehlo.constant dense<true> : tensor<i1>
    %5766 = stablehlo.reduce(%5765 init: %c_500) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5767 = stablehlo.and %5764, %5766 : tensor<i1>
    %5768 = stablehlo.is_finite %5441 : (tensor<64xf32>) -> tensor<64xi1>
    %c_501 = stablehlo.constant dense<true> : tensor<i1>
    %5769 = stablehlo.reduce(%5768 init: %c_501) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
    %5770 = stablehlo.and %5767, %5769 : tensor<i1>
    %5771 = stablehlo.is_finite %5443 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xi1>
    %c_502 = stablehlo.constant dense<true> : tensor<i1>
    %5772 = stablehlo.reduce(%5771 init: %c_502) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<7x7x3x64xi1>, tensor<i1>) -> tensor<i1>
    %5773 = stablehlo.and %5770, %5772 : tensor<i1>
    %c_503 = stablehlo.constant dense<2000> : tensor<i32>
    %5774 = stablehlo.compare  EQ, %arg294, %c_503,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_504 = stablehlo.constant dense<false> : tensor<i1>
    %5775 = stablehlo.broadcast_in_dim %c_504, dims = [] : (tensor<i1>) -> tensor<i1>
    %5776 = stablehlo.compare  NE, %5774, %5775,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5777 = stablehlo.convert %5776 : tensor<i1>
    %5778 = stablehlo.and %5777, %5773 : tensor<i1>
    %cst_505 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %5779 = stablehlo.multiply %arg295, %cst_505 : tensor<f32>
    %5780 = stablehlo.convert %5779 : tensor<f32>
    %cst_506 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %5781 = stablehlo.minimum %5780, %cst_506 : tensor<f32>
    %5782 = call @_where_506(%5778, %5781, %arg295) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5783 = stablehlo.multiply %arg295, %cst_5 : tensor<f32>
    %5784 = stablehlo.convert %5783 : tensor<f32>
    %cst_507 = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
    %5785 = stablehlo.maximum %5784, %cst_507 : tensor<f32>
    %5786 = call @_where_509(%5773, %5782, %5785) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %5787 = stablehlo.not %5773 : tensor<i1>
    %c_508 = stablehlo.constant dense<false> : tensor<i1>
    %5788 = stablehlo.broadcast_in_dim %c_508, dims = [] : (tensor<i1>) -> tensor<i1>
    %5789 = stablehlo.compare  NE, %5774, %5788,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5790 = stablehlo.convert %5789 : tensor<i1>
    %5791 = stablehlo.or %5790, %5787 : tensor<i1>
    %c_509 = stablehlo.constant dense<1> : tensor<i32>
    %5792 = stablehlo.add %arg294, %c_509 : tensor<i32>
    %5793 = call @_where_511(%5791, %c, %5792) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5794 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %5795 = stablehlo.iota dim = 0 : tensor<1000xi32>
    %5796 = stablehlo.reshape %5795 : (tensor<1000xi32>) -> tensor<1x1000xi32>
    %5797 = stablehlo.broadcast_in_dim %5794, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
    %5798 = stablehlo.broadcast_in_dim %5796, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
    %5799 = stablehlo.compare  EQ, %5797, %5798,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
    %5800 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
    %5801 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
    %5802 = stablehlo.select %5799, %5800, %5801 : tensor<256x1000xi1>, tensor<256x1000xf32>
    %5803 = stablehlo.convert %5802 : tensor<256x1000xf32>
    %5804 = call @log_softmax_513(%2595) : (tensor<256x1000xbf16>) -> tensor<256x1000xbf16>
    %5805 = stablehlo.convert %5804 : (tensor<256x1000xbf16>) -> tensor<256x1000xf32>
    %5806 = stablehlo.multiply %5803, %5805 : tensor<256x1000xf32>
    %cst_510 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5807 = stablehlo.reduce(%5806 init: %cst_510) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %5808 = stablehlo.negate %5807 : tensor<256xf32>
    %cst_511 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5809 = stablehlo.reduce(%5808 init: %cst_511) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %5810 = stablehlo.divide %5809, %cst_98 : tensor<f32>
    %5811 = call @argmax(%2595) : (tensor<256x1000xbf16>) -> tensor<256xi32>
    %5812 = stablehlo.compare  EQ, %5811, %1,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
    %5813 = stablehlo.convert %5812 : (tensor<256xi1>) -> tensor<256xi32>
    %5814 = stablehlo.convert %5813 : (tensor<256xi32>) -> tensor<256xf32>
    %cst_512 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5815 = stablehlo.reduce(%5814 init: %cst_512) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %5816 = stablehlo.divide %5815, %cst_98 : tensor<f32>
    %5817 = "stablehlo.all_reduce"(%5816) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5818 = "stablehlo.all_reduce"(%5810) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %5819 = stablehlo.divide %5817, %cst_11 : tensor<f32>
    %5820 = stablehlo.divide %5818, %cst_11 : tensor<f32>
    %5821 = "stablehlo.all_reduce"(%138) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5822 = "stablehlo.all_reduce"(%143) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5823 = "stablehlo.all_reduce"(%209) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5824 = "stablehlo.all_reduce"(%214) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5825 = "stablehlo.all_reduce"(%281) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5826 = "stablehlo.all_reduce"(%286) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5827 = "stablehlo.all_reduce"(%352) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5828 = "stablehlo.all_reduce"(%357) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5829 = "stablehlo.all_reduce"(%1702) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5830 = "stablehlo.all_reduce"(%1707) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5831 = "stablehlo.all_reduce"(%1773) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5832 = "stablehlo.all_reduce"(%1778) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5833 = "stablehlo.all_reduce"(%1845) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5834 = "stablehlo.all_reduce"(%1850) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5835 = "stablehlo.all_reduce"(%1916) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5836 = "stablehlo.all_reduce"(%1921) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5837 = "stablehlo.all_reduce"(%1988) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5838 = "stablehlo.all_reduce"(%1993) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5839 = "stablehlo.all_reduce"(%2059) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5840 = "stablehlo.all_reduce"(%2064) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5841 = "stablehlo.all_reduce"(%2131) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5842 = "stablehlo.all_reduce"(%2136) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5843 = "stablehlo.all_reduce"(%2202) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5844 = "stablehlo.all_reduce"(%2207) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5845 = "stablehlo.all_reduce"(%2269) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5846 = "stablehlo.all_reduce"(%2274) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5847 = "stablehlo.all_reduce"(%2341) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5848 = "stablehlo.all_reduce"(%2346) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5849 = "stablehlo.all_reduce"(%2412) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5850 = "stablehlo.all_reduce"(%2417) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5851 = "stablehlo.all_reduce"(%2484) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5852 = "stablehlo.all_reduce"(%2489) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5853 = "stablehlo.all_reduce"(%2555) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5854 = "stablehlo.all_reduce"(%2560) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<512xf32>) -> tensor<512xf32>
    %5855 = "stablehlo.all_reduce"(%424) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5856 = "stablehlo.all_reduce"(%429) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5857 = "stablehlo.all_reduce"(%495) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5858 = "stablehlo.all_reduce"(%500) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5859 = "stablehlo.all_reduce"(%567) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5860 = "stablehlo.all_reduce"(%572) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5861 = "stablehlo.all_reduce"(%638) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5862 = "stablehlo.all_reduce"(%643) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5863 = "stablehlo.all_reduce"(%705) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5864 = "stablehlo.all_reduce"(%710) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5865 = "stablehlo.all_reduce"(%777) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5866 = "stablehlo.all_reduce"(%782) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5867 = "stablehlo.all_reduce"(%848) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5868 = "stablehlo.all_reduce"(%853) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5869 = "stablehlo.all_reduce"(%920) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5870 = "stablehlo.all_reduce"(%925) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5871 = "stablehlo.all_reduce"(%991) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5872 = "stablehlo.all_reduce"(%996) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5873 = "stablehlo.all_reduce"(%1063) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5874 = "stablehlo.all_reduce"(%1068) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5875 = "stablehlo.all_reduce"(%1134) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5876 = "stablehlo.all_reduce"(%1139) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %5877 = "stablehlo.all_reduce"(%1206) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5878 = "stablehlo.all_reduce"(%1211) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5879 = "stablehlo.all_reduce"(%1277) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5880 = "stablehlo.all_reduce"(%1282) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5881 = "stablehlo.all_reduce"(%1344) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5882 = "stablehlo.all_reduce"(%1349) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5883 = "stablehlo.all_reduce"(%1416) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5884 = "stablehlo.all_reduce"(%1421) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5885 = "stablehlo.all_reduce"(%1487) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5886 = "stablehlo.all_reduce"(%1492) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5887 = "stablehlo.all_reduce"(%1559) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5888 = "stablehlo.all_reduce"(%1564) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5889 = "stablehlo.all_reduce"(%1630) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5890 = "stablehlo.all_reduce"(%1635) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<256xf32>) -> tensor<256xf32>
    %5891 = "stablehlo.all_reduce"(%65) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5892 = "stablehlo.all_reduce"(%70) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg298: tensor<f32>, %arg299: tensor<f32>):
      %7388 = stablehlo.add %arg298, %arg299 : tensor<f32>
      stablehlo.return %7388 : tensor<f32>
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %5893 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5894 = stablehlo.divide %5821, %5893 : tensor<64xf32>
    %5895 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5896 = stablehlo.divide %5822, %5895 : tensor<64xf32>
    %5897 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5898 = stablehlo.divide %5823, %5897 : tensor<64xf32>
    %5899 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5900 = stablehlo.divide %5824, %5899 : tensor<64xf32>
    %5901 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5902 = stablehlo.divide %5825, %5901 : tensor<64xf32>
    %5903 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5904 = stablehlo.divide %5826, %5903 : tensor<64xf32>
    %5905 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5906 = stablehlo.divide %5827, %5905 : tensor<64xf32>
    %5907 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5908 = stablehlo.divide %5828, %5907 : tensor<64xf32>
    %5909 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5910 = stablehlo.divide %5829, %5909 : tensor<256xf32>
    %5911 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5912 = stablehlo.divide %5830, %5911 : tensor<256xf32>
    %5913 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5914 = stablehlo.divide %5831, %5913 : tensor<256xf32>
    %5915 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5916 = stablehlo.divide %5832, %5915 : tensor<256xf32>
    %5917 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5918 = stablehlo.divide %5833, %5917 : tensor<256xf32>
    %5919 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5920 = stablehlo.divide %5834, %5919 : tensor<256xf32>
    %5921 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5922 = stablehlo.divide %5835, %5921 : tensor<256xf32>
    %5923 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5924 = stablehlo.divide %5836, %5923 : tensor<256xf32>
    %5925 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5926 = stablehlo.divide %5837, %5925 : tensor<256xf32>
    %5927 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5928 = stablehlo.divide %5838, %5927 : tensor<256xf32>
    %5929 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5930 = stablehlo.divide %5839, %5929 : tensor<256xf32>
    %5931 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %5932 = stablehlo.divide %5840, %5931 : tensor<256xf32>
    %5933 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5934 = stablehlo.divide %5841, %5933 : tensor<512xf32>
    %5935 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5936 = stablehlo.divide %5842, %5935 : tensor<512xf32>
    %5937 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5938 = stablehlo.divide %5843, %5937 : tensor<512xf32>
    %5939 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5940 = stablehlo.divide %5844, %5939 : tensor<512xf32>
    %5941 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5942 = stablehlo.divide %5845, %5941 : tensor<512xf32>
    %5943 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5944 = stablehlo.divide %5846, %5943 : tensor<512xf32>
    %5945 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5946 = stablehlo.divide %5847, %5945 : tensor<512xf32>
    %5947 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5948 = stablehlo.divide %5848, %5947 : tensor<512xf32>
    %5949 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5950 = stablehlo.divide %5849, %5949 : tensor<512xf32>
    %5951 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5952 = stablehlo.divide %5850, %5951 : tensor<512xf32>
    %5953 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5954 = stablehlo.divide %5851, %5953 : tensor<512xf32>
    %5955 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5956 = stablehlo.divide %5852, %5955 : tensor<512xf32>
    %5957 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5958 = stablehlo.divide %5853, %5957 : tensor<512xf32>
    %5959 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5960 = stablehlo.divide %5854, %5959 : tensor<512xf32>
    %5961 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5962 = stablehlo.divide %5855, %5961 : tensor<64xf32>
    %5963 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5964 = stablehlo.divide %5856, %5963 : tensor<64xf32>
    %5965 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5966 = stablehlo.divide %5857, %5965 : tensor<64xf32>
    %5967 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5968 = stablehlo.divide %5858, %5967 : tensor<64xf32>
    %5969 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5970 = stablehlo.divide %5859, %5969 : tensor<128xf32>
    %5971 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5972 = stablehlo.divide %5860, %5971 : tensor<128xf32>
    %5973 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5974 = stablehlo.divide %5861, %5973 : tensor<128xf32>
    %5975 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5976 = stablehlo.divide %5862, %5975 : tensor<128xf32>
    %5977 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5978 = stablehlo.divide %5863, %5977 : tensor<128xf32>
    %5979 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5980 = stablehlo.divide %5864, %5979 : tensor<128xf32>
    %5981 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5982 = stablehlo.divide %5865, %5981 : tensor<128xf32>
    %5983 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5984 = stablehlo.divide %5866, %5983 : tensor<128xf32>
    %5985 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5986 = stablehlo.divide %5867, %5985 : tensor<128xf32>
    %5987 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5988 = stablehlo.divide %5868, %5987 : tensor<128xf32>
    %5989 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5990 = stablehlo.divide %5869, %5989 : tensor<128xf32>
    %5991 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5992 = stablehlo.divide %5870, %5991 : tensor<128xf32>
    %5993 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5994 = stablehlo.divide %5871, %5993 : tensor<128xf32>
    %5995 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5996 = stablehlo.divide %5872, %5995 : tensor<128xf32>
    %5997 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5998 = stablehlo.divide %5873, %5997 : tensor<128xf32>
    %5999 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6000 = stablehlo.divide %5874, %5999 : tensor<128xf32>
    %6001 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6002 = stablehlo.divide %5875, %6001 : tensor<128xf32>
    %6003 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6004 = stablehlo.divide %5876, %6003 : tensor<128xf32>
    %6005 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6006 = stablehlo.divide %5877, %6005 : tensor<256xf32>
    %6007 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6008 = stablehlo.divide %5878, %6007 : tensor<256xf32>
    %6009 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6010 = stablehlo.divide %5879, %6009 : tensor<256xf32>
    %6011 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6012 = stablehlo.divide %5880, %6011 : tensor<256xf32>
    %6013 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6014 = stablehlo.divide %5881, %6013 : tensor<256xf32>
    %6015 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6016 = stablehlo.divide %5882, %6015 : tensor<256xf32>
    %6017 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6018 = stablehlo.divide %5883, %6017 : tensor<256xf32>
    %6019 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6020 = stablehlo.divide %5884, %6019 : tensor<256xf32>
    %6021 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6022 = stablehlo.divide %5885, %6021 : tensor<256xf32>
    %6023 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6024 = stablehlo.divide %5886, %6023 : tensor<256xf32>
    %6025 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6026 = stablehlo.divide %5887, %6025 : tensor<256xf32>
    %6027 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6028 = stablehlo.divide %5888, %6027 : tensor<256xf32>
    %6029 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6030 = stablehlo.divide %5889, %6029 : tensor<256xf32>
    %6031 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6032 = stablehlo.divide %5890, %6031 : tensor<256xf32>
    %6033 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6034 = stablehlo.divide %5891, %6033 : tensor<64xf32>
    %6035 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6036 = stablehlo.divide %5892, %6035 : tensor<64xf32>
    %6037 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %6038 = stablehlo.multiply %6037, %arg111 : tensor<1000xf32>
    %6039 = stablehlo.add %5225, %6038 : tensor<1000xf32>
    %6040 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %6041 = stablehlo.multiply %6040, %arg112 : tensor<512x1000xf32>
    %6042 = stablehlo.add %5227, %6041 : tensor<512x1000xf32>
    %6043 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6044 = stablehlo.multiply %6043, %arg113 : tensor<64xf32>
    %6045 = stablehlo.add %5229, %6044 : tensor<64xf32>
    %6046 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6047 = stablehlo.multiply %6046, %arg114 : tensor<64xf32>
    %6048 = stablehlo.add %5231, %6047 : tensor<64xf32>
    %6049 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6050 = stablehlo.multiply %6049, %arg115 : tensor<64xf32>
    %6051 = stablehlo.add %5233, %6050 : tensor<64xf32>
    %6052 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6053 = stablehlo.multiply %6052, %arg116 : tensor<64xf32>
    %6054 = stablehlo.add %5235, %6053 : tensor<64xf32>
    %6055 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6056 = stablehlo.multiply %6055, %arg117 : tensor<3x3x64x64xf32>
    %6057 = stablehlo.add %5237, %6056 : tensor<3x3x64x64xf32>
    %6058 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6059 = stablehlo.multiply %6058, %arg118 : tensor<3x3x64x64xf32>
    %6060 = stablehlo.add %5239, %6059 : tensor<3x3x64x64xf32>
    %6061 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6062 = stablehlo.multiply %6061, %arg119 : tensor<64xf32>
    %6063 = stablehlo.add %5241, %6062 : tensor<64xf32>
    %6064 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6065 = stablehlo.multiply %6064, %arg120 : tensor<64xf32>
    %6066 = stablehlo.add %5243, %6065 : tensor<64xf32>
    %6067 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6068 = stablehlo.multiply %6067, %arg121 : tensor<64xf32>
    %6069 = stablehlo.add %5245, %6068 : tensor<64xf32>
    %6070 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6071 = stablehlo.multiply %6070, %arg122 : tensor<64xf32>
    %6072 = stablehlo.add %5247, %6071 : tensor<64xf32>
    %6073 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6074 = stablehlo.multiply %6073, %arg123 : tensor<3x3x64x64xf32>
    %6075 = stablehlo.add %5249, %6074 : tensor<3x3x64x64xf32>
    %6076 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6077 = stablehlo.multiply %6076, %arg124 : tensor<3x3x64x64xf32>
    %6078 = stablehlo.add %5251, %6077 : tensor<3x3x64x64xf32>
    %6079 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6080 = stablehlo.multiply %6079, %arg125 : tensor<256xf32>
    %6081 = stablehlo.add %5253, %6080 : tensor<256xf32>
    %6082 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6083 = stablehlo.multiply %6082, %arg126 : tensor<256xf32>
    %6084 = stablehlo.add %5255, %6083 : tensor<256xf32>
    %6085 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6086 = stablehlo.multiply %6085, %arg127 : tensor<256xf32>
    %6087 = stablehlo.add %5257, %6086 : tensor<256xf32>
    %6088 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6089 = stablehlo.multiply %6088, %arg128 : tensor<256xf32>
    %6090 = stablehlo.add %5259, %6089 : tensor<256xf32>
    %6091 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6092 = stablehlo.multiply %6091, %arg129 : tensor<3x3x256x256xf32>
    %6093 = stablehlo.add %5261, %6092 : tensor<3x3x256x256xf32>
    %6094 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6095 = stablehlo.multiply %6094, %arg130 : tensor<3x3x256x256xf32>
    %6096 = stablehlo.add %5263, %6095 : tensor<3x3x256x256xf32>
    %6097 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6098 = stablehlo.multiply %6097, %arg131 : tensor<256xf32>
    %6099 = stablehlo.add %5265, %6098 : tensor<256xf32>
    %6100 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6101 = stablehlo.multiply %6100, %arg132 : tensor<256xf32>
    %6102 = stablehlo.add %5267, %6101 : tensor<256xf32>
    %6103 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6104 = stablehlo.multiply %6103, %arg133 : tensor<256xf32>
    %6105 = stablehlo.add %5269, %6104 : tensor<256xf32>
    %6106 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6107 = stablehlo.multiply %6106, %arg134 : tensor<256xf32>
    %6108 = stablehlo.add %5271, %6107 : tensor<256xf32>
    %6109 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6110 = stablehlo.multiply %6109, %arg135 : tensor<3x3x256x256xf32>
    %6111 = stablehlo.add %5273, %6110 : tensor<3x3x256x256xf32>
    %6112 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6113 = stablehlo.multiply %6112, %arg136 : tensor<3x3x256x256xf32>
    %6114 = stablehlo.add %5275, %6113 : tensor<3x3x256x256xf32>
    %6115 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6116 = stablehlo.multiply %6115, %arg137 : tensor<256xf32>
    %6117 = stablehlo.add %5277, %6116 : tensor<256xf32>
    %6118 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6119 = stablehlo.multiply %6118, %arg138 : tensor<256xf32>
    %6120 = stablehlo.add %5279, %6119 : tensor<256xf32>
    %6121 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6122 = stablehlo.multiply %6121, %arg139 : tensor<256xf32>
    %6123 = stablehlo.add %5281, %6122 : tensor<256xf32>
    %6124 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6125 = stablehlo.multiply %6124, %arg140 : tensor<256xf32>
    %6126 = stablehlo.add %5283, %6125 : tensor<256xf32>
    %6127 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6128 = stablehlo.multiply %6127, %arg141 : tensor<3x3x256x256xf32>
    %6129 = stablehlo.add %5285, %6128 : tensor<3x3x256x256xf32>
    %6130 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6131 = stablehlo.multiply %6130, %arg142 : tensor<3x3x256x256xf32>
    %6132 = stablehlo.add %5287, %6131 : tensor<3x3x256x256xf32>
    %6133 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6134 = stablehlo.multiply %6133, %arg143 : tensor<512xf32>
    %6135 = stablehlo.add %5289, %6134 : tensor<512xf32>
    %6136 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6137 = stablehlo.multiply %6136, %arg144 : tensor<512xf32>
    %6138 = stablehlo.add %5291, %6137 : tensor<512xf32>
    %6139 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6140 = stablehlo.multiply %6139, %arg145 : tensor<512xf32>
    %6141 = stablehlo.add %5293, %6140 : tensor<512xf32>
    %6142 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6143 = stablehlo.multiply %6142, %arg146 : tensor<512xf32>
    %6144 = stablehlo.add %5295, %6143 : tensor<512xf32>
    %6145 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %6146 = stablehlo.multiply %6145, %arg147 : tensor<3x3x256x512xf32>
    %6147 = stablehlo.add %5297, %6146 : tensor<3x3x256x512xf32>
    %6148 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6149 = stablehlo.multiply %6148, %arg148 : tensor<3x3x512x512xf32>
    %6150 = stablehlo.add %5299, %6149 : tensor<3x3x512x512xf32>
    %6151 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %6152 = stablehlo.multiply %6151, %arg149 : tensor<1x1x256x512xf32>
    %6153 = stablehlo.add %5301, %6152 : tensor<1x1x256x512xf32>
    %6154 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6155 = stablehlo.multiply %6154, %arg150 : tensor<512xf32>
    %6156 = stablehlo.add %5303, %6155 : tensor<512xf32>
    %6157 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6158 = stablehlo.multiply %6157, %arg151 : tensor<512xf32>
    %6159 = stablehlo.add %5305, %6158 : tensor<512xf32>
    %6160 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6161 = stablehlo.multiply %6160, %arg152 : tensor<512xf32>
    %6162 = stablehlo.add %5307, %6161 : tensor<512xf32>
    %6163 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6164 = stablehlo.multiply %6163, %arg153 : tensor<512xf32>
    %6165 = stablehlo.add %5309, %6164 : tensor<512xf32>
    %6166 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6167 = stablehlo.multiply %6166, %arg154 : tensor<512xf32>
    %6168 = stablehlo.add %5311, %6167 : tensor<512xf32>
    %6169 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6170 = stablehlo.multiply %6169, %arg155 : tensor<512xf32>
    %6171 = stablehlo.add %5313, %6170 : tensor<512xf32>
    %6172 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6173 = stablehlo.multiply %6172, %arg156 : tensor<3x3x512x512xf32>
    %6174 = stablehlo.add %5315, %6173 : tensor<3x3x512x512xf32>
    %6175 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6176 = stablehlo.multiply %6175, %arg157 : tensor<3x3x512x512xf32>
    %6177 = stablehlo.add %5317, %6176 : tensor<3x3x512x512xf32>
    %6178 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6179 = stablehlo.multiply %6178, %arg158 : tensor<512xf32>
    %6180 = stablehlo.add %5319, %6179 : tensor<512xf32>
    %6181 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6182 = stablehlo.multiply %6181, %arg159 : tensor<512xf32>
    %6183 = stablehlo.add %5321, %6182 : tensor<512xf32>
    %6184 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6185 = stablehlo.multiply %6184, %arg160 : tensor<512xf32>
    %6186 = stablehlo.add %5323, %6185 : tensor<512xf32>
    %6187 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6188 = stablehlo.multiply %6187, %arg161 : tensor<512xf32>
    %6189 = stablehlo.add %5325, %6188 : tensor<512xf32>
    %6190 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6191 = stablehlo.multiply %6190, %arg162 : tensor<3x3x512x512xf32>
    %6192 = stablehlo.add %5327, %6191 : tensor<3x3x512x512xf32>
    %6193 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6194 = stablehlo.multiply %6193, %arg163 : tensor<3x3x512x512xf32>
    %6195 = stablehlo.add %5329, %6194 : tensor<3x3x512x512xf32>
    %6196 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6197 = stablehlo.multiply %6196, %arg164 : tensor<64xf32>
    %6198 = stablehlo.add %5331, %6197 : tensor<64xf32>
    %6199 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6200 = stablehlo.multiply %6199, %arg165 : tensor<64xf32>
    %6201 = stablehlo.add %5333, %6200 : tensor<64xf32>
    %6202 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6203 = stablehlo.multiply %6202, %arg166 : tensor<64xf32>
    %6204 = stablehlo.add %5335, %6203 : tensor<64xf32>
    %6205 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6206 = stablehlo.multiply %6205, %arg167 : tensor<64xf32>
    %6207 = stablehlo.add %5337, %6206 : tensor<64xf32>
    %6208 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6209 = stablehlo.multiply %6208, %arg168 : tensor<3x3x64x64xf32>
    %6210 = stablehlo.add %5339, %6209 : tensor<3x3x64x64xf32>
    %6211 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6212 = stablehlo.multiply %6211, %arg169 : tensor<3x3x64x64xf32>
    %6213 = stablehlo.add %5341, %6212 : tensor<3x3x64x64xf32>
    %6214 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6215 = stablehlo.multiply %6214, %arg170 : tensor<128xf32>
    %6216 = stablehlo.add %5343, %6215 : tensor<128xf32>
    %6217 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6218 = stablehlo.multiply %6217, %arg171 : tensor<128xf32>
    %6219 = stablehlo.add %5345, %6218 : tensor<128xf32>
    %6220 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6221 = stablehlo.multiply %6220, %arg172 : tensor<128xf32>
    %6222 = stablehlo.add %5347, %6221 : tensor<128xf32>
    %6223 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6224 = stablehlo.multiply %6223, %arg173 : tensor<128xf32>
    %6225 = stablehlo.add %5349, %6224 : tensor<128xf32>
    %6226 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %6227 = stablehlo.multiply %6226, %arg174 : tensor<3x3x64x128xf32>
    %6228 = stablehlo.add %5351, %6227 : tensor<3x3x64x128xf32>
    %6229 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6230 = stablehlo.multiply %6229, %arg175 : tensor<3x3x128x128xf32>
    %6231 = stablehlo.add %5353, %6230 : tensor<3x3x128x128xf32>
    %6232 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %6233 = stablehlo.multiply %6232, %arg176 : tensor<1x1x64x128xf32>
    %6234 = stablehlo.add %5355, %6233 : tensor<1x1x64x128xf32>
    %6235 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6236 = stablehlo.multiply %6235, %arg177 : tensor<128xf32>
    %6237 = stablehlo.add %5357, %6236 : tensor<128xf32>
    %6238 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6239 = stablehlo.multiply %6238, %arg178 : tensor<128xf32>
    %6240 = stablehlo.add %5359, %6239 : tensor<128xf32>
    %6241 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6242 = stablehlo.multiply %6241, %arg179 : tensor<128xf32>
    %6243 = stablehlo.add %5361, %6242 : tensor<128xf32>
    %6244 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6245 = stablehlo.multiply %6244, %arg180 : tensor<128xf32>
    %6246 = stablehlo.add %5363, %6245 : tensor<128xf32>
    %6247 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6248 = stablehlo.multiply %6247, %arg181 : tensor<128xf32>
    %6249 = stablehlo.add %5365, %6248 : tensor<128xf32>
    %6250 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6251 = stablehlo.multiply %6250, %arg182 : tensor<128xf32>
    %6252 = stablehlo.add %5367, %6251 : tensor<128xf32>
    %6253 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6254 = stablehlo.multiply %6253, %arg183 : tensor<3x3x128x128xf32>
    %6255 = stablehlo.add %5369, %6254 : tensor<3x3x128x128xf32>
    %6256 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6257 = stablehlo.multiply %6256, %arg184 : tensor<3x3x128x128xf32>
    %6258 = stablehlo.add %5371, %6257 : tensor<3x3x128x128xf32>
    %6259 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6260 = stablehlo.multiply %6259, %arg185 : tensor<128xf32>
    %6261 = stablehlo.add %5373, %6260 : tensor<128xf32>
    %6262 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6263 = stablehlo.multiply %6262, %arg186 : tensor<128xf32>
    %6264 = stablehlo.add %5375, %6263 : tensor<128xf32>
    %6265 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6266 = stablehlo.multiply %6265, %arg187 : tensor<128xf32>
    %6267 = stablehlo.add %5377, %6266 : tensor<128xf32>
    %6268 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6269 = stablehlo.multiply %6268, %arg188 : tensor<128xf32>
    %6270 = stablehlo.add %5379, %6269 : tensor<128xf32>
    %6271 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6272 = stablehlo.multiply %6271, %arg189 : tensor<3x3x128x128xf32>
    %6273 = stablehlo.add %5381, %6272 : tensor<3x3x128x128xf32>
    %6274 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6275 = stablehlo.multiply %6274, %arg190 : tensor<3x3x128x128xf32>
    %6276 = stablehlo.add %5383, %6275 : tensor<3x3x128x128xf32>
    %6277 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6278 = stablehlo.multiply %6277, %arg191 : tensor<128xf32>
    %6279 = stablehlo.add %5385, %6278 : tensor<128xf32>
    %6280 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6281 = stablehlo.multiply %6280, %arg192 : tensor<128xf32>
    %6282 = stablehlo.add %5387, %6281 : tensor<128xf32>
    %6283 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6284 = stablehlo.multiply %6283, %arg193 : tensor<128xf32>
    %6285 = stablehlo.add %5389, %6284 : tensor<128xf32>
    %6286 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6287 = stablehlo.multiply %6286, %arg194 : tensor<128xf32>
    %6288 = stablehlo.add %5391, %6287 : tensor<128xf32>
    %6289 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6290 = stablehlo.multiply %6289, %arg195 : tensor<3x3x128x128xf32>
    %6291 = stablehlo.add %5393, %6290 : tensor<3x3x128x128xf32>
    %6292 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6293 = stablehlo.multiply %6292, %arg196 : tensor<3x3x128x128xf32>
    %6294 = stablehlo.add %5395, %6293 : tensor<3x3x128x128xf32>
    %6295 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6296 = stablehlo.multiply %6295, %arg197 : tensor<256xf32>
    %6297 = stablehlo.add %5397, %6296 : tensor<256xf32>
    %6298 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6299 = stablehlo.multiply %6298, %arg198 : tensor<256xf32>
    %6300 = stablehlo.add %5399, %6299 : tensor<256xf32>
    %6301 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6302 = stablehlo.multiply %6301, %arg199 : tensor<256xf32>
    %6303 = stablehlo.add %5401, %6302 : tensor<256xf32>
    %6304 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6305 = stablehlo.multiply %6304, %arg200 : tensor<256xf32>
    %6306 = stablehlo.add %5403, %6305 : tensor<256xf32>
    %6307 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %6308 = stablehlo.multiply %6307, %arg201 : tensor<3x3x128x256xf32>
    %6309 = stablehlo.add %5405, %6308 : tensor<3x3x128x256xf32>
    %6310 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6311 = stablehlo.multiply %6310, %arg202 : tensor<3x3x256x256xf32>
    %6312 = stablehlo.add %5407, %6311 : tensor<3x3x256x256xf32>
    %6313 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %6314 = stablehlo.multiply %6313, %arg203 : tensor<1x1x128x256xf32>
    %6315 = stablehlo.add %5409, %6314 : tensor<1x1x128x256xf32>
    %6316 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6317 = stablehlo.multiply %6316, %arg204 : tensor<256xf32>
    %6318 = stablehlo.add %5411, %6317 : tensor<256xf32>
    %6319 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6320 = stablehlo.multiply %6319, %arg205 : tensor<256xf32>
    %6321 = stablehlo.add %5413, %6320 : tensor<256xf32>
    %6322 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6323 = stablehlo.multiply %6322, %arg206 : tensor<256xf32>
    %6324 = stablehlo.add %5415, %6323 : tensor<256xf32>
    %6325 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6326 = stablehlo.multiply %6325, %arg207 : tensor<256xf32>
    %6327 = stablehlo.add %5417, %6326 : tensor<256xf32>
    %6328 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6329 = stablehlo.multiply %6328, %arg208 : tensor<256xf32>
    %6330 = stablehlo.add %5419, %6329 : tensor<256xf32>
    %6331 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6332 = stablehlo.multiply %6331, %arg209 : tensor<256xf32>
    %6333 = stablehlo.add %5421, %6332 : tensor<256xf32>
    %6334 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6335 = stablehlo.multiply %6334, %arg210 : tensor<3x3x256x256xf32>
    %6336 = stablehlo.add %5423, %6335 : tensor<3x3x256x256xf32>
    %6337 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6338 = stablehlo.multiply %6337, %arg211 : tensor<3x3x256x256xf32>
    %6339 = stablehlo.add %5425, %6338 : tensor<3x3x256x256xf32>
    %6340 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6341 = stablehlo.multiply %6340, %arg212 : tensor<256xf32>
    %6342 = stablehlo.add %5427, %6341 : tensor<256xf32>
    %6343 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6344 = stablehlo.multiply %6343, %arg213 : tensor<256xf32>
    %6345 = stablehlo.add %5429, %6344 : tensor<256xf32>
    %6346 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6347 = stablehlo.multiply %6346, %arg214 : tensor<256xf32>
    %6348 = stablehlo.add %5431, %6347 : tensor<256xf32>
    %6349 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6350 = stablehlo.multiply %6349, %arg215 : tensor<256xf32>
    %6351 = stablehlo.add %5433, %6350 : tensor<256xf32>
    %6352 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6353 = stablehlo.multiply %6352, %arg216 : tensor<3x3x256x256xf32>
    %6354 = stablehlo.add %5435, %6353 : tensor<3x3x256x256xf32>
    %6355 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6356 = stablehlo.multiply %6355, %arg217 : tensor<3x3x256x256xf32>
    %6357 = stablehlo.add %5437, %6356 : tensor<3x3x256x256xf32>
    %6358 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6359 = stablehlo.multiply %6358, %arg218 : tensor<64xf32>
    %6360 = stablehlo.add %5439, %6359 : tensor<64xf32>
    %6361 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6362 = stablehlo.multiply %6361, %arg219 : tensor<64xf32>
    %6363 = stablehlo.add %5441, %6362 : tensor<64xf32>
    %6364 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %6365 = stablehlo.multiply %6364, %arg220 : tensor<7x7x3x64xf32>
    %6366 = stablehlo.add %5443, %6365 : tensor<7x7x3x64xf32>
    %6367 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %6368 = stablehlo.multiply %6367, %6039 : tensor<1000xf32>
    %6369 = stablehlo.add %5225, %6368 : tensor<1000xf32>
    %6370 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %6371 = stablehlo.multiply %6370, %6042 : tensor<512x1000xf32>
    %6372 = stablehlo.add %5227, %6371 : tensor<512x1000xf32>
    %6373 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6374 = stablehlo.multiply %6373, %6045 : tensor<64xf32>
    %6375 = stablehlo.add %5229, %6374 : tensor<64xf32>
    %6376 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6377 = stablehlo.multiply %6376, %6048 : tensor<64xf32>
    %6378 = stablehlo.add %5231, %6377 : tensor<64xf32>
    %6379 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6380 = stablehlo.multiply %6379, %6051 : tensor<64xf32>
    %6381 = stablehlo.add %5233, %6380 : tensor<64xf32>
    %6382 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6383 = stablehlo.multiply %6382, %6054 : tensor<64xf32>
    %6384 = stablehlo.add %5235, %6383 : tensor<64xf32>
    %6385 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6386 = stablehlo.multiply %6385, %6057 : tensor<3x3x64x64xf32>
    %6387 = stablehlo.add %5237, %6386 : tensor<3x3x64x64xf32>
    %6388 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6389 = stablehlo.multiply %6388, %6060 : tensor<3x3x64x64xf32>
    %6390 = stablehlo.add %5239, %6389 : tensor<3x3x64x64xf32>
    %6391 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6392 = stablehlo.multiply %6391, %6063 : tensor<64xf32>
    %6393 = stablehlo.add %5241, %6392 : tensor<64xf32>
    %6394 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6395 = stablehlo.multiply %6394, %6066 : tensor<64xf32>
    %6396 = stablehlo.add %5243, %6395 : tensor<64xf32>
    %6397 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6398 = stablehlo.multiply %6397, %6069 : tensor<64xf32>
    %6399 = stablehlo.add %5245, %6398 : tensor<64xf32>
    %6400 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6401 = stablehlo.multiply %6400, %6072 : tensor<64xf32>
    %6402 = stablehlo.add %5247, %6401 : tensor<64xf32>
    %6403 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6404 = stablehlo.multiply %6403, %6075 : tensor<3x3x64x64xf32>
    %6405 = stablehlo.add %5249, %6404 : tensor<3x3x64x64xf32>
    %6406 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6407 = stablehlo.multiply %6406, %6078 : tensor<3x3x64x64xf32>
    %6408 = stablehlo.add %5251, %6407 : tensor<3x3x64x64xf32>
    %6409 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6410 = stablehlo.multiply %6409, %6081 : tensor<256xf32>
    %6411 = stablehlo.add %5253, %6410 : tensor<256xf32>
    %6412 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6413 = stablehlo.multiply %6412, %6084 : tensor<256xf32>
    %6414 = stablehlo.add %5255, %6413 : tensor<256xf32>
    %6415 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6416 = stablehlo.multiply %6415, %6087 : tensor<256xf32>
    %6417 = stablehlo.add %5257, %6416 : tensor<256xf32>
    %6418 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6419 = stablehlo.multiply %6418, %6090 : tensor<256xf32>
    %6420 = stablehlo.add %5259, %6419 : tensor<256xf32>
    %6421 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6422 = stablehlo.multiply %6421, %6093 : tensor<3x3x256x256xf32>
    %6423 = stablehlo.add %5261, %6422 : tensor<3x3x256x256xf32>
    %6424 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6425 = stablehlo.multiply %6424, %6096 : tensor<3x3x256x256xf32>
    %6426 = stablehlo.add %5263, %6425 : tensor<3x3x256x256xf32>
    %6427 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6428 = stablehlo.multiply %6427, %6099 : tensor<256xf32>
    %6429 = stablehlo.add %5265, %6428 : tensor<256xf32>
    %6430 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6431 = stablehlo.multiply %6430, %6102 : tensor<256xf32>
    %6432 = stablehlo.add %5267, %6431 : tensor<256xf32>
    %6433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6434 = stablehlo.multiply %6433, %6105 : tensor<256xf32>
    %6435 = stablehlo.add %5269, %6434 : tensor<256xf32>
    %6436 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6437 = stablehlo.multiply %6436, %6108 : tensor<256xf32>
    %6438 = stablehlo.add %5271, %6437 : tensor<256xf32>
    %6439 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6440 = stablehlo.multiply %6439, %6111 : tensor<3x3x256x256xf32>
    %6441 = stablehlo.add %5273, %6440 : tensor<3x3x256x256xf32>
    %6442 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6443 = stablehlo.multiply %6442, %6114 : tensor<3x3x256x256xf32>
    %6444 = stablehlo.add %5275, %6443 : tensor<3x3x256x256xf32>
    %6445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6446 = stablehlo.multiply %6445, %6117 : tensor<256xf32>
    %6447 = stablehlo.add %5277, %6446 : tensor<256xf32>
    %6448 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6449 = stablehlo.multiply %6448, %6120 : tensor<256xf32>
    %6450 = stablehlo.add %5279, %6449 : tensor<256xf32>
    %6451 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6452 = stablehlo.multiply %6451, %6123 : tensor<256xf32>
    %6453 = stablehlo.add %5281, %6452 : tensor<256xf32>
    %6454 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6455 = stablehlo.multiply %6454, %6126 : tensor<256xf32>
    %6456 = stablehlo.add %5283, %6455 : tensor<256xf32>
    %6457 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6458 = stablehlo.multiply %6457, %6129 : tensor<3x3x256x256xf32>
    %6459 = stablehlo.add %5285, %6458 : tensor<3x3x256x256xf32>
    %6460 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6461 = stablehlo.multiply %6460, %6132 : tensor<3x3x256x256xf32>
    %6462 = stablehlo.add %5287, %6461 : tensor<3x3x256x256xf32>
    %6463 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6464 = stablehlo.multiply %6463, %6135 : tensor<512xf32>
    %6465 = stablehlo.add %5289, %6464 : tensor<512xf32>
    %6466 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6467 = stablehlo.multiply %6466, %6138 : tensor<512xf32>
    %6468 = stablehlo.add %5291, %6467 : tensor<512xf32>
    %6469 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6470 = stablehlo.multiply %6469, %6141 : tensor<512xf32>
    %6471 = stablehlo.add %5293, %6470 : tensor<512xf32>
    %6472 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6473 = stablehlo.multiply %6472, %6144 : tensor<512xf32>
    %6474 = stablehlo.add %5295, %6473 : tensor<512xf32>
    %6475 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %6476 = stablehlo.multiply %6475, %6147 : tensor<3x3x256x512xf32>
    %6477 = stablehlo.add %5297, %6476 : tensor<3x3x256x512xf32>
    %6478 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6479 = stablehlo.multiply %6478, %6150 : tensor<3x3x512x512xf32>
    %6480 = stablehlo.add %5299, %6479 : tensor<3x3x512x512xf32>
    %6481 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %6482 = stablehlo.multiply %6481, %6153 : tensor<1x1x256x512xf32>
    %6483 = stablehlo.add %5301, %6482 : tensor<1x1x256x512xf32>
    %6484 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6485 = stablehlo.multiply %6484, %6156 : tensor<512xf32>
    %6486 = stablehlo.add %5303, %6485 : tensor<512xf32>
    %6487 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6488 = stablehlo.multiply %6487, %6159 : tensor<512xf32>
    %6489 = stablehlo.add %5305, %6488 : tensor<512xf32>
    %6490 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6491 = stablehlo.multiply %6490, %6162 : tensor<512xf32>
    %6492 = stablehlo.add %5307, %6491 : tensor<512xf32>
    %6493 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6494 = stablehlo.multiply %6493, %6165 : tensor<512xf32>
    %6495 = stablehlo.add %5309, %6494 : tensor<512xf32>
    %6496 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6497 = stablehlo.multiply %6496, %6168 : tensor<512xf32>
    %6498 = stablehlo.add %5311, %6497 : tensor<512xf32>
    %6499 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6500 = stablehlo.multiply %6499, %6171 : tensor<512xf32>
    %6501 = stablehlo.add %5313, %6500 : tensor<512xf32>
    %6502 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6503 = stablehlo.multiply %6502, %6174 : tensor<3x3x512x512xf32>
    %6504 = stablehlo.add %5315, %6503 : tensor<3x3x512x512xf32>
    %6505 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6506 = stablehlo.multiply %6505, %6177 : tensor<3x3x512x512xf32>
    %6507 = stablehlo.add %5317, %6506 : tensor<3x3x512x512xf32>
    %6508 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6509 = stablehlo.multiply %6508, %6180 : tensor<512xf32>
    %6510 = stablehlo.add %5319, %6509 : tensor<512xf32>
    %6511 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6512 = stablehlo.multiply %6511, %6183 : tensor<512xf32>
    %6513 = stablehlo.add %5321, %6512 : tensor<512xf32>
    %6514 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6515 = stablehlo.multiply %6514, %6186 : tensor<512xf32>
    %6516 = stablehlo.add %5323, %6515 : tensor<512xf32>
    %6517 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6518 = stablehlo.multiply %6517, %6189 : tensor<512xf32>
    %6519 = stablehlo.add %5325, %6518 : tensor<512xf32>
    %6520 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6521 = stablehlo.multiply %6520, %6192 : tensor<3x3x512x512xf32>
    %6522 = stablehlo.add %5327, %6521 : tensor<3x3x512x512xf32>
    %6523 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6524 = stablehlo.multiply %6523, %6195 : tensor<3x3x512x512xf32>
    %6525 = stablehlo.add %5329, %6524 : tensor<3x3x512x512xf32>
    %6526 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6527 = stablehlo.multiply %6526, %6198 : tensor<64xf32>
    %6528 = stablehlo.add %5331, %6527 : tensor<64xf32>
    %6529 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6530 = stablehlo.multiply %6529, %6201 : tensor<64xf32>
    %6531 = stablehlo.add %5333, %6530 : tensor<64xf32>
    %6532 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6533 = stablehlo.multiply %6532, %6204 : tensor<64xf32>
    %6534 = stablehlo.add %5335, %6533 : tensor<64xf32>
    %6535 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6536 = stablehlo.multiply %6535, %6207 : tensor<64xf32>
    %6537 = stablehlo.add %5337, %6536 : tensor<64xf32>
    %6538 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6539 = stablehlo.multiply %6538, %6210 : tensor<3x3x64x64xf32>
    %6540 = stablehlo.add %5339, %6539 : tensor<3x3x64x64xf32>
    %6541 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6542 = stablehlo.multiply %6541, %6213 : tensor<3x3x64x64xf32>
    %6543 = stablehlo.add %5341, %6542 : tensor<3x3x64x64xf32>
    %6544 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6545 = stablehlo.multiply %6544, %6216 : tensor<128xf32>
    %6546 = stablehlo.add %5343, %6545 : tensor<128xf32>
    %6547 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6548 = stablehlo.multiply %6547, %6219 : tensor<128xf32>
    %6549 = stablehlo.add %5345, %6548 : tensor<128xf32>
    %6550 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6551 = stablehlo.multiply %6550, %6222 : tensor<128xf32>
    %6552 = stablehlo.add %5347, %6551 : tensor<128xf32>
    %6553 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6554 = stablehlo.multiply %6553, %6225 : tensor<128xf32>
    %6555 = stablehlo.add %5349, %6554 : tensor<128xf32>
    %6556 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %6557 = stablehlo.multiply %6556, %6228 : tensor<3x3x64x128xf32>
    %6558 = stablehlo.add %5351, %6557 : tensor<3x3x64x128xf32>
    %6559 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6560 = stablehlo.multiply %6559, %6231 : tensor<3x3x128x128xf32>
    %6561 = stablehlo.add %5353, %6560 : tensor<3x3x128x128xf32>
    %6562 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %6563 = stablehlo.multiply %6562, %6234 : tensor<1x1x64x128xf32>
    %6564 = stablehlo.add %5355, %6563 : tensor<1x1x64x128xf32>
    %6565 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6566 = stablehlo.multiply %6565, %6237 : tensor<128xf32>
    %6567 = stablehlo.add %5357, %6566 : tensor<128xf32>
    %6568 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6569 = stablehlo.multiply %6568, %6240 : tensor<128xf32>
    %6570 = stablehlo.add %5359, %6569 : tensor<128xf32>
    %6571 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6572 = stablehlo.multiply %6571, %6243 : tensor<128xf32>
    %6573 = stablehlo.add %5361, %6572 : tensor<128xf32>
    %6574 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6575 = stablehlo.multiply %6574, %6246 : tensor<128xf32>
    %6576 = stablehlo.add %5363, %6575 : tensor<128xf32>
    %6577 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6578 = stablehlo.multiply %6577, %6249 : tensor<128xf32>
    %6579 = stablehlo.add %5365, %6578 : tensor<128xf32>
    %6580 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6581 = stablehlo.multiply %6580, %6252 : tensor<128xf32>
    %6582 = stablehlo.add %5367, %6581 : tensor<128xf32>
    %6583 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6584 = stablehlo.multiply %6583, %6255 : tensor<3x3x128x128xf32>
    %6585 = stablehlo.add %5369, %6584 : tensor<3x3x128x128xf32>
    %6586 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6587 = stablehlo.multiply %6586, %6258 : tensor<3x3x128x128xf32>
    %6588 = stablehlo.add %5371, %6587 : tensor<3x3x128x128xf32>
    %6589 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6590 = stablehlo.multiply %6589, %6261 : tensor<128xf32>
    %6591 = stablehlo.add %5373, %6590 : tensor<128xf32>
    %6592 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6593 = stablehlo.multiply %6592, %6264 : tensor<128xf32>
    %6594 = stablehlo.add %5375, %6593 : tensor<128xf32>
    %6595 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6596 = stablehlo.multiply %6595, %6267 : tensor<128xf32>
    %6597 = stablehlo.add %5377, %6596 : tensor<128xf32>
    %6598 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6599 = stablehlo.multiply %6598, %6270 : tensor<128xf32>
    %6600 = stablehlo.add %5379, %6599 : tensor<128xf32>
    %6601 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6602 = stablehlo.multiply %6601, %6273 : tensor<3x3x128x128xf32>
    %6603 = stablehlo.add %5381, %6602 : tensor<3x3x128x128xf32>
    %6604 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6605 = stablehlo.multiply %6604, %6276 : tensor<3x3x128x128xf32>
    %6606 = stablehlo.add %5383, %6605 : tensor<3x3x128x128xf32>
    %6607 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6608 = stablehlo.multiply %6607, %6279 : tensor<128xf32>
    %6609 = stablehlo.add %5385, %6608 : tensor<128xf32>
    %6610 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6611 = stablehlo.multiply %6610, %6282 : tensor<128xf32>
    %6612 = stablehlo.add %5387, %6611 : tensor<128xf32>
    %6613 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6614 = stablehlo.multiply %6613, %6285 : tensor<128xf32>
    %6615 = stablehlo.add %5389, %6614 : tensor<128xf32>
    %6616 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6617 = stablehlo.multiply %6616, %6288 : tensor<128xf32>
    %6618 = stablehlo.add %5391, %6617 : tensor<128xf32>
    %6619 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6620 = stablehlo.multiply %6619, %6291 : tensor<3x3x128x128xf32>
    %6621 = stablehlo.add %5393, %6620 : tensor<3x3x128x128xf32>
    %6622 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6623 = stablehlo.multiply %6622, %6294 : tensor<3x3x128x128xf32>
    %6624 = stablehlo.add %5395, %6623 : tensor<3x3x128x128xf32>
    %6625 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6626 = stablehlo.multiply %6625, %6297 : tensor<256xf32>
    %6627 = stablehlo.add %5397, %6626 : tensor<256xf32>
    %6628 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6629 = stablehlo.multiply %6628, %6300 : tensor<256xf32>
    %6630 = stablehlo.add %5399, %6629 : tensor<256xf32>
    %6631 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6632 = stablehlo.multiply %6631, %6303 : tensor<256xf32>
    %6633 = stablehlo.add %5401, %6632 : tensor<256xf32>
    %6634 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6635 = stablehlo.multiply %6634, %6306 : tensor<256xf32>
    %6636 = stablehlo.add %5403, %6635 : tensor<256xf32>
    %6637 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %6638 = stablehlo.multiply %6637, %6309 : tensor<3x3x128x256xf32>
    %6639 = stablehlo.add %5405, %6638 : tensor<3x3x128x256xf32>
    %6640 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6641 = stablehlo.multiply %6640, %6312 : tensor<3x3x256x256xf32>
    %6642 = stablehlo.add %5407, %6641 : tensor<3x3x256x256xf32>
    %6643 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %6644 = stablehlo.multiply %6643, %6315 : tensor<1x1x128x256xf32>
    %6645 = stablehlo.add %5409, %6644 : tensor<1x1x128x256xf32>
    %6646 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6647 = stablehlo.multiply %6646, %6318 : tensor<256xf32>
    %6648 = stablehlo.add %5411, %6647 : tensor<256xf32>
    %6649 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6650 = stablehlo.multiply %6649, %6321 : tensor<256xf32>
    %6651 = stablehlo.add %5413, %6650 : tensor<256xf32>
    %6652 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6653 = stablehlo.multiply %6652, %6324 : tensor<256xf32>
    %6654 = stablehlo.add %5415, %6653 : tensor<256xf32>
    %6655 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6656 = stablehlo.multiply %6655, %6327 : tensor<256xf32>
    %6657 = stablehlo.add %5417, %6656 : tensor<256xf32>
    %6658 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6659 = stablehlo.multiply %6658, %6330 : tensor<256xf32>
    %6660 = stablehlo.add %5419, %6659 : tensor<256xf32>
    %6661 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6662 = stablehlo.multiply %6661, %6333 : tensor<256xf32>
    %6663 = stablehlo.add %5421, %6662 : tensor<256xf32>
    %6664 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6665 = stablehlo.multiply %6664, %6336 : tensor<3x3x256x256xf32>
    %6666 = stablehlo.add %5423, %6665 : tensor<3x3x256x256xf32>
    %6667 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6668 = stablehlo.multiply %6667, %6339 : tensor<3x3x256x256xf32>
    %6669 = stablehlo.add %5425, %6668 : tensor<3x3x256x256xf32>
    %6670 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6671 = stablehlo.multiply %6670, %6342 : tensor<256xf32>
    %6672 = stablehlo.add %5427, %6671 : tensor<256xf32>
    %6673 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6674 = stablehlo.multiply %6673, %6345 : tensor<256xf32>
    %6675 = stablehlo.add %5429, %6674 : tensor<256xf32>
    %6676 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6677 = stablehlo.multiply %6676, %6348 : tensor<256xf32>
    %6678 = stablehlo.add %5431, %6677 : tensor<256xf32>
    %6679 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6680 = stablehlo.multiply %6679, %6351 : tensor<256xf32>
    %6681 = stablehlo.add %5433, %6680 : tensor<256xf32>
    %6682 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6683 = stablehlo.multiply %6682, %6354 : tensor<3x3x256x256xf32>
    %6684 = stablehlo.add %5435, %6683 : tensor<3x3x256x256xf32>
    %6685 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6686 = stablehlo.multiply %6685, %6357 : tensor<3x3x256x256xf32>
    %6687 = stablehlo.add %5437, %6686 : tensor<3x3x256x256xf32>
    %6688 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6689 = stablehlo.multiply %6688, %6360 : tensor<64xf32>
    %6690 = stablehlo.add %5439, %6689 : tensor<64xf32>
    %6691 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6692 = stablehlo.multiply %6691, %6363 : tensor<64xf32>
    %6693 = stablehlo.add %5441, %6692 : tensor<64xf32>
    %6694 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %6695 = stablehlo.multiply %6694, %6366 : tensor<7x7x3x64xf32>
    %6696 = stablehlo.add %5443, %6695 : tensor<7x7x3x64xf32>
    %c_513 = stablehlo.constant dense<0> : tensor<i32>
    %6697 = stablehlo.subtract %arg221, %c_513 : tensor<i32>
    %6698 = call @clip_540(%6697, %c, %cst) : (tensor<i32>, tensor<i32>, tensor<f32>) -> tensor<f32>
    %6699 = stablehlo.divide %6698, %cst : tensor<f32>
    %6700 = stablehlo.subtract %cst_0, %6699 : tensor<f32>
    %6701 = stablehlo.multiply %cst_1, %6700 : tensor<f32>
    %6702 = stablehlo.add %6701, %cst_2 : tensor<f32>
    %6703 = stablehlo.convert %arg221 : (tensor<i32>) -> tensor<f32>
    %6704 = stablehlo.compare  LT, %6703, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6705 = stablehlo.convert %arg221 : (tensor<i32>) -> tensor<f32>
    %6706 = stablehlo.subtract %6705, %cst : tensor<f32>
    %6707 = stablehlo.minimum %6706, %cst_3 : tensor<f32>
    %6708 = stablehlo.multiply %cst_4, %6707 : tensor<f32>
    %6709 = stablehlo.divide %6708, %cst_3 : tensor<f32>
    %6710 = stablehlo.cosine %6709 : tensor<f32>
    %6711 = stablehlo.add %cst_0, %6710 : tensor<f32>
    %6712 = stablehlo.multiply %cst_5, %6711 : tensor<f32>
    %6713 = stablehlo.power %6712, %cst_0 : tensor<f32>
    %6714 = stablehlo.multiply %cst_0, %6713 : tensor<f32>
    %6715 = stablehlo.add %6714, %cst_6 : tensor<f32>
    %6716 = stablehlo.multiply %cst_2, %6715 : tensor<f32>
    %6717 = call @_where(%6704, %6702, %6716) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %cst_514 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %6718 = stablehlo.multiply %cst_514, %6717 : tensor<f32>
    %6719 = stablehlo.convert %6718 : tensor<f32>
    %6720 = stablehlo.broadcast_in_dim %6719, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %6721 = stablehlo.multiply %6720, %6369 : tensor<1000xf32>
    %6722 = stablehlo.convert %6718 : tensor<f32>
    %6723 = stablehlo.broadcast_in_dim %6722, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %6724 = stablehlo.multiply %6723, %6372 : tensor<512x1000xf32>
    %6725 = stablehlo.convert %6718 : tensor<f32>
    %6726 = stablehlo.broadcast_in_dim %6725, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6727 = stablehlo.multiply %6726, %6375 : tensor<64xf32>
    %6728 = stablehlo.convert %6718 : tensor<f32>
    %6729 = stablehlo.broadcast_in_dim %6728, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6730 = stablehlo.multiply %6729, %6378 : tensor<64xf32>
    %6731 = stablehlo.convert %6718 : tensor<f32>
    %6732 = stablehlo.broadcast_in_dim %6731, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6733 = stablehlo.multiply %6732, %6381 : tensor<64xf32>
    %6734 = stablehlo.convert %6718 : tensor<f32>
    %6735 = stablehlo.broadcast_in_dim %6734, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6736 = stablehlo.multiply %6735, %6384 : tensor<64xf32>
    %6737 = stablehlo.convert %6718 : tensor<f32>
    %6738 = stablehlo.broadcast_in_dim %6737, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6739 = stablehlo.multiply %6738, %6387 : tensor<3x3x64x64xf32>
    %6740 = stablehlo.convert %6718 : tensor<f32>
    %6741 = stablehlo.broadcast_in_dim %6740, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6742 = stablehlo.multiply %6741, %6390 : tensor<3x3x64x64xf32>
    %6743 = stablehlo.convert %6718 : tensor<f32>
    %6744 = stablehlo.broadcast_in_dim %6743, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6745 = stablehlo.multiply %6744, %6393 : tensor<64xf32>
    %6746 = stablehlo.convert %6718 : tensor<f32>
    %6747 = stablehlo.broadcast_in_dim %6746, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6748 = stablehlo.multiply %6747, %6396 : tensor<64xf32>
    %6749 = stablehlo.convert %6718 : tensor<f32>
    %6750 = stablehlo.broadcast_in_dim %6749, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6751 = stablehlo.multiply %6750, %6399 : tensor<64xf32>
    %6752 = stablehlo.convert %6718 : tensor<f32>
    %6753 = stablehlo.broadcast_in_dim %6752, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6754 = stablehlo.multiply %6753, %6402 : tensor<64xf32>
    %6755 = stablehlo.convert %6718 : tensor<f32>
    %6756 = stablehlo.broadcast_in_dim %6755, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6757 = stablehlo.multiply %6756, %6405 : tensor<3x3x64x64xf32>
    %6758 = stablehlo.convert %6718 : tensor<f32>
    %6759 = stablehlo.broadcast_in_dim %6758, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6760 = stablehlo.multiply %6759, %6408 : tensor<3x3x64x64xf32>
    %6761 = stablehlo.convert %6718 : tensor<f32>
    %6762 = stablehlo.broadcast_in_dim %6761, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6763 = stablehlo.multiply %6762, %6411 : tensor<256xf32>
    %6764 = stablehlo.convert %6718 : tensor<f32>
    %6765 = stablehlo.broadcast_in_dim %6764, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6766 = stablehlo.multiply %6765, %6414 : tensor<256xf32>
    %6767 = stablehlo.convert %6718 : tensor<f32>
    %6768 = stablehlo.broadcast_in_dim %6767, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6769 = stablehlo.multiply %6768, %6417 : tensor<256xf32>
    %6770 = stablehlo.convert %6718 : tensor<f32>
    %6771 = stablehlo.broadcast_in_dim %6770, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6772 = stablehlo.multiply %6771, %6420 : tensor<256xf32>
    %6773 = stablehlo.convert %6718 : tensor<f32>
    %6774 = stablehlo.broadcast_in_dim %6773, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6775 = stablehlo.multiply %6774, %6423 : tensor<3x3x256x256xf32>
    %6776 = stablehlo.convert %6718 : tensor<f32>
    %6777 = stablehlo.broadcast_in_dim %6776, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6778 = stablehlo.multiply %6777, %6426 : tensor<3x3x256x256xf32>
    %6779 = stablehlo.convert %6718 : tensor<f32>
    %6780 = stablehlo.broadcast_in_dim %6779, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6781 = stablehlo.multiply %6780, %6429 : tensor<256xf32>
    %6782 = stablehlo.convert %6718 : tensor<f32>
    %6783 = stablehlo.broadcast_in_dim %6782, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6784 = stablehlo.multiply %6783, %6432 : tensor<256xf32>
    %6785 = stablehlo.convert %6718 : tensor<f32>
    %6786 = stablehlo.broadcast_in_dim %6785, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6787 = stablehlo.multiply %6786, %6435 : tensor<256xf32>
    %6788 = stablehlo.convert %6718 : tensor<f32>
    %6789 = stablehlo.broadcast_in_dim %6788, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6790 = stablehlo.multiply %6789, %6438 : tensor<256xf32>
    %6791 = stablehlo.convert %6718 : tensor<f32>
    %6792 = stablehlo.broadcast_in_dim %6791, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6793 = stablehlo.multiply %6792, %6441 : tensor<3x3x256x256xf32>
    %6794 = stablehlo.convert %6718 : tensor<f32>
    %6795 = stablehlo.broadcast_in_dim %6794, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6796 = stablehlo.multiply %6795, %6444 : tensor<3x3x256x256xf32>
    %6797 = stablehlo.convert %6718 : tensor<f32>
    %6798 = stablehlo.broadcast_in_dim %6797, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6799 = stablehlo.multiply %6798, %6447 : tensor<256xf32>
    %6800 = stablehlo.convert %6718 : tensor<f32>
    %6801 = stablehlo.broadcast_in_dim %6800, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6802 = stablehlo.multiply %6801, %6450 : tensor<256xf32>
    %6803 = stablehlo.convert %6718 : tensor<f32>
    %6804 = stablehlo.broadcast_in_dim %6803, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6805 = stablehlo.multiply %6804, %6453 : tensor<256xf32>
    %6806 = stablehlo.convert %6718 : tensor<f32>
    %6807 = stablehlo.broadcast_in_dim %6806, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6808 = stablehlo.multiply %6807, %6456 : tensor<256xf32>
    %6809 = stablehlo.convert %6718 : tensor<f32>
    %6810 = stablehlo.broadcast_in_dim %6809, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6811 = stablehlo.multiply %6810, %6459 : tensor<3x3x256x256xf32>
    %6812 = stablehlo.convert %6718 : tensor<f32>
    %6813 = stablehlo.broadcast_in_dim %6812, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6814 = stablehlo.multiply %6813, %6462 : tensor<3x3x256x256xf32>
    %6815 = stablehlo.convert %6718 : tensor<f32>
    %6816 = stablehlo.broadcast_in_dim %6815, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6817 = stablehlo.multiply %6816, %6465 : tensor<512xf32>
    %6818 = stablehlo.convert %6718 : tensor<f32>
    %6819 = stablehlo.broadcast_in_dim %6818, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6820 = stablehlo.multiply %6819, %6468 : tensor<512xf32>
    %6821 = stablehlo.convert %6718 : tensor<f32>
    %6822 = stablehlo.broadcast_in_dim %6821, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6823 = stablehlo.multiply %6822, %6471 : tensor<512xf32>
    %6824 = stablehlo.convert %6718 : tensor<f32>
    %6825 = stablehlo.broadcast_in_dim %6824, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6826 = stablehlo.multiply %6825, %6474 : tensor<512xf32>
    %6827 = stablehlo.convert %6718 : tensor<f32>
    %6828 = stablehlo.broadcast_in_dim %6827, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
    %6829 = stablehlo.multiply %6828, %6477 : tensor<3x3x256x512xf32>
    %6830 = stablehlo.convert %6718 : tensor<f32>
    %6831 = stablehlo.broadcast_in_dim %6830, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6832 = stablehlo.multiply %6831, %6480 : tensor<3x3x512x512xf32>
    %6833 = stablehlo.convert %6718 : tensor<f32>
    %6834 = stablehlo.broadcast_in_dim %6833, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
    %6835 = stablehlo.multiply %6834, %6483 : tensor<1x1x256x512xf32>
    %6836 = stablehlo.convert %6718 : tensor<f32>
    %6837 = stablehlo.broadcast_in_dim %6836, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6838 = stablehlo.multiply %6837, %6486 : tensor<512xf32>
    %6839 = stablehlo.convert %6718 : tensor<f32>
    %6840 = stablehlo.broadcast_in_dim %6839, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6841 = stablehlo.multiply %6840, %6489 : tensor<512xf32>
    %6842 = stablehlo.convert %6718 : tensor<f32>
    %6843 = stablehlo.broadcast_in_dim %6842, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6844 = stablehlo.multiply %6843, %6492 : tensor<512xf32>
    %6845 = stablehlo.convert %6718 : tensor<f32>
    %6846 = stablehlo.broadcast_in_dim %6845, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6847 = stablehlo.multiply %6846, %6495 : tensor<512xf32>
    %6848 = stablehlo.convert %6718 : tensor<f32>
    %6849 = stablehlo.broadcast_in_dim %6848, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6850 = stablehlo.multiply %6849, %6498 : tensor<512xf32>
    %6851 = stablehlo.convert %6718 : tensor<f32>
    %6852 = stablehlo.broadcast_in_dim %6851, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6853 = stablehlo.multiply %6852, %6501 : tensor<512xf32>
    %6854 = stablehlo.convert %6718 : tensor<f32>
    %6855 = stablehlo.broadcast_in_dim %6854, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6856 = stablehlo.multiply %6855, %6504 : tensor<3x3x512x512xf32>
    %6857 = stablehlo.convert %6718 : tensor<f32>
    %6858 = stablehlo.broadcast_in_dim %6857, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6859 = stablehlo.multiply %6858, %6507 : tensor<3x3x512x512xf32>
    %6860 = stablehlo.convert %6718 : tensor<f32>
    %6861 = stablehlo.broadcast_in_dim %6860, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6862 = stablehlo.multiply %6861, %6510 : tensor<512xf32>
    %6863 = stablehlo.convert %6718 : tensor<f32>
    %6864 = stablehlo.broadcast_in_dim %6863, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6865 = stablehlo.multiply %6864, %6513 : tensor<512xf32>
    %6866 = stablehlo.convert %6718 : tensor<f32>
    %6867 = stablehlo.broadcast_in_dim %6866, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6868 = stablehlo.multiply %6867, %6516 : tensor<512xf32>
    %6869 = stablehlo.convert %6718 : tensor<f32>
    %6870 = stablehlo.broadcast_in_dim %6869, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6871 = stablehlo.multiply %6870, %6519 : tensor<512xf32>
    %6872 = stablehlo.convert %6718 : tensor<f32>
    %6873 = stablehlo.broadcast_in_dim %6872, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6874 = stablehlo.multiply %6873, %6522 : tensor<3x3x512x512xf32>
    %6875 = stablehlo.convert %6718 : tensor<f32>
    %6876 = stablehlo.broadcast_in_dim %6875, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
    %6877 = stablehlo.multiply %6876, %6525 : tensor<3x3x512x512xf32>
    %6878 = stablehlo.convert %6718 : tensor<f32>
    %6879 = stablehlo.broadcast_in_dim %6878, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6880 = stablehlo.multiply %6879, %6528 : tensor<64xf32>
    %6881 = stablehlo.convert %6718 : tensor<f32>
    %6882 = stablehlo.broadcast_in_dim %6881, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6883 = stablehlo.multiply %6882, %6531 : tensor<64xf32>
    %6884 = stablehlo.convert %6718 : tensor<f32>
    %6885 = stablehlo.broadcast_in_dim %6884, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6886 = stablehlo.multiply %6885, %6534 : tensor<64xf32>
    %6887 = stablehlo.convert %6718 : tensor<f32>
    %6888 = stablehlo.broadcast_in_dim %6887, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6889 = stablehlo.multiply %6888, %6537 : tensor<64xf32>
    %6890 = stablehlo.convert %6718 : tensor<f32>
    %6891 = stablehlo.broadcast_in_dim %6890, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6892 = stablehlo.multiply %6891, %6540 : tensor<3x3x64x64xf32>
    %6893 = stablehlo.convert %6718 : tensor<f32>
    %6894 = stablehlo.broadcast_in_dim %6893, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
    %6895 = stablehlo.multiply %6894, %6543 : tensor<3x3x64x64xf32>
    %6896 = stablehlo.convert %6718 : tensor<f32>
    %6897 = stablehlo.broadcast_in_dim %6896, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6898 = stablehlo.multiply %6897, %6546 : tensor<128xf32>
    %6899 = stablehlo.convert %6718 : tensor<f32>
    %6900 = stablehlo.broadcast_in_dim %6899, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6901 = stablehlo.multiply %6900, %6549 : tensor<128xf32>
    %6902 = stablehlo.convert %6718 : tensor<f32>
    %6903 = stablehlo.broadcast_in_dim %6902, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6904 = stablehlo.multiply %6903, %6552 : tensor<128xf32>
    %6905 = stablehlo.convert %6718 : tensor<f32>
    %6906 = stablehlo.broadcast_in_dim %6905, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6907 = stablehlo.multiply %6906, %6555 : tensor<128xf32>
    %6908 = stablehlo.convert %6718 : tensor<f32>
    %6909 = stablehlo.broadcast_in_dim %6908, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
    %6910 = stablehlo.multiply %6909, %6558 : tensor<3x3x64x128xf32>
    %6911 = stablehlo.convert %6718 : tensor<f32>
    %6912 = stablehlo.broadcast_in_dim %6911, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6913 = stablehlo.multiply %6912, %6561 : tensor<3x3x128x128xf32>
    %6914 = stablehlo.convert %6718 : tensor<f32>
    %6915 = stablehlo.broadcast_in_dim %6914, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
    %6916 = stablehlo.multiply %6915, %6564 : tensor<1x1x64x128xf32>
    %6917 = stablehlo.convert %6718 : tensor<f32>
    %6918 = stablehlo.broadcast_in_dim %6917, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6919 = stablehlo.multiply %6918, %6567 : tensor<128xf32>
    %6920 = stablehlo.convert %6718 : tensor<f32>
    %6921 = stablehlo.broadcast_in_dim %6920, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6922 = stablehlo.multiply %6921, %6570 : tensor<128xf32>
    %6923 = stablehlo.convert %6718 : tensor<f32>
    %6924 = stablehlo.broadcast_in_dim %6923, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6925 = stablehlo.multiply %6924, %6573 : tensor<128xf32>
    %6926 = stablehlo.convert %6718 : tensor<f32>
    %6927 = stablehlo.broadcast_in_dim %6926, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6928 = stablehlo.multiply %6927, %6576 : tensor<128xf32>
    %6929 = stablehlo.convert %6718 : tensor<f32>
    %6930 = stablehlo.broadcast_in_dim %6929, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6931 = stablehlo.multiply %6930, %6579 : tensor<128xf32>
    %6932 = stablehlo.convert %6718 : tensor<f32>
    %6933 = stablehlo.broadcast_in_dim %6932, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6934 = stablehlo.multiply %6933, %6582 : tensor<128xf32>
    %6935 = stablehlo.convert %6718 : tensor<f32>
    %6936 = stablehlo.broadcast_in_dim %6935, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6937 = stablehlo.multiply %6936, %6585 : tensor<3x3x128x128xf32>
    %6938 = stablehlo.convert %6718 : tensor<f32>
    %6939 = stablehlo.broadcast_in_dim %6938, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6940 = stablehlo.multiply %6939, %6588 : tensor<3x3x128x128xf32>
    %6941 = stablehlo.convert %6718 : tensor<f32>
    %6942 = stablehlo.broadcast_in_dim %6941, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6943 = stablehlo.multiply %6942, %6591 : tensor<128xf32>
    %6944 = stablehlo.convert %6718 : tensor<f32>
    %6945 = stablehlo.broadcast_in_dim %6944, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6946 = stablehlo.multiply %6945, %6594 : tensor<128xf32>
    %6947 = stablehlo.convert %6718 : tensor<f32>
    %6948 = stablehlo.broadcast_in_dim %6947, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6949 = stablehlo.multiply %6948, %6597 : tensor<128xf32>
    %6950 = stablehlo.convert %6718 : tensor<f32>
    %6951 = stablehlo.broadcast_in_dim %6950, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6952 = stablehlo.multiply %6951, %6600 : tensor<128xf32>
    %6953 = stablehlo.convert %6718 : tensor<f32>
    %6954 = stablehlo.broadcast_in_dim %6953, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6955 = stablehlo.multiply %6954, %6603 : tensor<3x3x128x128xf32>
    %6956 = stablehlo.convert %6718 : tensor<f32>
    %6957 = stablehlo.broadcast_in_dim %6956, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6958 = stablehlo.multiply %6957, %6606 : tensor<3x3x128x128xf32>
    %6959 = stablehlo.convert %6718 : tensor<f32>
    %6960 = stablehlo.broadcast_in_dim %6959, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6961 = stablehlo.multiply %6960, %6609 : tensor<128xf32>
    %6962 = stablehlo.convert %6718 : tensor<f32>
    %6963 = stablehlo.broadcast_in_dim %6962, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6964 = stablehlo.multiply %6963, %6612 : tensor<128xf32>
    %6965 = stablehlo.convert %6718 : tensor<f32>
    %6966 = stablehlo.broadcast_in_dim %6965, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6967 = stablehlo.multiply %6966, %6615 : tensor<128xf32>
    %6968 = stablehlo.convert %6718 : tensor<f32>
    %6969 = stablehlo.broadcast_in_dim %6968, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6970 = stablehlo.multiply %6969, %6618 : tensor<128xf32>
    %6971 = stablehlo.convert %6718 : tensor<f32>
    %6972 = stablehlo.broadcast_in_dim %6971, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6973 = stablehlo.multiply %6972, %6621 : tensor<3x3x128x128xf32>
    %6974 = stablehlo.convert %6718 : tensor<f32>
    %6975 = stablehlo.broadcast_in_dim %6974, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
    %6976 = stablehlo.multiply %6975, %6624 : tensor<3x3x128x128xf32>
    %6977 = stablehlo.convert %6718 : tensor<f32>
    %6978 = stablehlo.broadcast_in_dim %6977, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6979 = stablehlo.multiply %6978, %6627 : tensor<256xf32>
    %6980 = stablehlo.convert %6718 : tensor<f32>
    %6981 = stablehlo.broadcast_in_dim %6980, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6982 = stablehlo.multiply %6981, %6630 : tensor<256xf32>
    %6983 = stablehlo.convert %6718 : tensor<f32>
    %6984 = stablehlo.broadcast_in_dim %6983, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6985 = stablehlo.multiply %6984, %6633 : tensor<256xf32>
    %6986 = stablehlo.convert %6718 : tensor<f32>
    %6987 = stablehlo.broadcast_in_dim %6986, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %6988 = stablehlo.multiply %6987, %6636 : tensor<256xf32>
    %6989 = stablehlo.convert %6718 : tensor<f32>
    %6990 = stablehlo.broadcast_in_dim %6989, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
    %6991 = stablehlo.multiply %6990, %6639 : tensor<3x3x128x256xf32>
    %6992 = stablehlo.convert %6718 : tensor<f32>
    %6993 = stablehlo.broadcast_in_dim %6992, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %6994 = stablehlo.multiply %6993, %6642 : tensor<3x3x256x256xf32>
    %6995 = stablehlo.convert %6718 : tensor<f32>
    %6996 = stablehlo.broadcast_in_dim %6995, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
    %6997 = stablehlo.multiply %6996, %6645 : tensor<1x1x128x256xf32>
    %6998 = stablehlo.convert %6718 : tensor<f32>
    %6999 = stablehlo.broadcast_in_dim %6998, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7000 = stablehlo.multiply %6999, %6648 : tensor<256xf32>
    %7001 = stablehlo.convert %6718 : tensor<f32>
    %7002 = stablehlo.broadcast_in_dim %7001, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7003 = stablehlo.multiply %7002, %6651 : tensor<256xf32>
    %7004 = stablehlo.convert %6718 : tensor<f32>
    %7005 = stablehlo.broadcast_in_dim %7004, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7006 = stablehlo.multiply %7005, %6654 : tensor<256xf32>
    %7007 = stablehlo.convert %6718 : tensor<f32>
    %7008 = stablehlo.broadcast_in_dim %7007, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7009 = stablehlo.multiply %7008, %6657 : tensor<256xf32>
    %7010 = stablehlo.convert %6718 : tensor<f32>
    %7011 = stablehlo.broadcast_in_dim %7010, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7012 = stablehlo.multiply %7011, %6660 : tensor<256xf32>
    %7013 = stablehlo.convert %6718 : tensor<f32>
    %7014 = stablehlo.broadcast_in_dim %7013, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7015 = stablehlo.multiply %7014, %6663 : tensor<256xf32>
    %7016 = stablehlo.convert %6718 : tensor<f32>
    %7017 = stablehlo.broadcast_in_dim %7016, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %7018 = stablehlo.multiply %7017, %6666 : tensor<3x3x256x256xf32>
    %7019 = stablehlo.convert %6718 : tensor<f32>
    %7020 = stablehlo.broadcast_in_dim %7019, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %7021 = stablehlo.multiply %7020, %6669 : tensor<3x3x256x256xf32>
    %7022 = stablehlo.convert %6718 : tensor<f32>
    %7023 = stablehlo.broadcast_in_dim %7022, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7024 = stablehlo.multiply %7023, %6672 : tensor<256xf32>
    %7025 = stablehlo.convert %6718 : tensor<f32>
    %7026 = stablehlo.broadcast_in_dim %7025, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7027 = stablehlo.multiply %7026, %6675 : tensor<256xf32>
    %7028 = stablehlo.convert %6718 : tensor<f32>
    %7029 = stablehlo.broadcast_in_dim %7028, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7030 = stablehlo.multiply %7029, %6678 : tensor<256xf32>
    %7031 = stablehlo.convert %6718 : tensor<f32>
    %7032 = stablehlo.broadcast_in_dim %7031, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %7033 = stablehlo.multiply %7032, %6681 : tensor<256xf32>
    %7034 = stablehlo.convert %6718 : tensor<f32>
    %7035 = stablehlo.broadcast_in_dim %7034, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %7036 = stablehlo.multiply %7035, %6684 : tensor<3x3x256x256xf32>
    %7037 = stablehlo.convert %6718 : tensor<f32>
    %7038 = stablehlo.broadcast_in_dim %7037, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
    %7039 = stablehlo.multiply %7038, %6687 : tensor<3x3x256x256xf32>
    %7040 = stablehlo.convert %6718 : tensor<f32>
    %7041 = stablehlo.broadcast_in_dim %7040, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %7042 = stablehlo.multiply %7041, %6690 : tensor<64xf32>
    %7043 = stablehlo.convert %6718 : tensor<f32>
    %7044 = stablehlo.broadcast_in_dim %7043, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %7045 = stablehlo.multiply %7044, %6693 : tensor<64xf32>
    %7046 = stablehlo.convert %6718 : tensor<f32>
    %7047 = stablehlo.broadcast_in_dim %7046, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
    %7048 = stablehlo.multiply %7047, %6696 : tensor<7x7x3x64xf32>
    %c_515 = stablehlo.constant dense<2147483647> : tensor<i32>
    %7049 = stablehlo.compare  LT, %arg221, %c_515,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_516 = stablehlo.constant dense<1> : tensor<i32>
    %7050 = stablehlo.add %arg221, %c_516 : tensor<i32>
    %7051 = call @_where_545(%7049, %7050, %c_515) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %7052 = stablehlo.add %arg1, %6721 : tensor<1000xf32>
    %7053 = stablehlo.add %arg2, %6724 : tensor<512x1000xf32>
    %7054 = stablehlo.add %arg3, %6727 : tensor<64xf32>
    %7055 = stablehlo.add %arg4, %6730 : tensor<64xf32>
    %7056 = stablehlo.add %arg5, %6733 : tensor<64xf32>
    %7057 = stablehlo.add %arg6, %6736 : tensor<64xf32>
    %7058 = stablehlo.add %arg7, %6739 : tensor<3x3x64x64xf32>
    %7059 = stablehlo.add %arg8, %6742 : tensor<3x3x64x64xf32>
    %7060 = stablehlo.add %arg9, %6745 : tensor<64xf32>
    %7061 = stablehlo.add %arg10, %6748 : tensor<64xf32>
    %7062 = stablehlo.add %arg11, %6751 : tensor<64xf32>
    %7063 = stablehlo.add %arg12, %6754 : tensor<64xf32>
    %7064 = stablehlo.add %arg13, %6757 : tensor<3x3x64x64xf32>
    %7065 = stablehlo.add %arg14, %6760 : tensor<3x3x64x64xf32>
    %7066 = stablehlo.add %arg15, %6763 : tensor<256xf32>
    %7067 = stablehlo.add %arg16, %6766 : tensor<256xf32>
    %7068 = stablehlo.add %arg17, %6769 : tensor<256xf32>
    %7069 = stablehlo.add %arg18, %6772 : tensor<256xf32>
    %7070 = stablehlo.add %arg19, %6775 : tensor<3x3x256x256xf32>
    %7071 = stablehlo.add %arg20, %6778 : tensor<3x3x256x256xf32>
    %7072 = stablehlo.add %arg21, %6781 : tensor<256xf32>
    %7073 = stablehlo.add %arg22, %6784 : tensor<256xf32>
    %7074 = stablehlo.add %arg23, %6787 : tensor<256xf32>
    %7075 = stablehlo.add %arg24, %6790 : tensor<256xf32>
    %7076 = stablehlo.add %arg25, %6793 : tensor<3x3x256x256xf32>
    %7077 = stablehlo.add %arg26, %6796 : tensor<3x3x256x256xf32>
    %7078 = stablehlo.add %arg27, %6799 : tensor<256xf32>
    %7079 = stablehlo.add %arg28, %6802 : tensor<256xf32>
    %7080 = stablehlo.add %arg29, %6805 : tensor<256xf32>
    %7081 = stablehlo.add %arg30, %6808 : tensor<256xf32>
    %7082 = stablehlo.add %arg31, %6811 : tensor<3x3x256x256xf32>
    %7083 = stablehlo.add %arg32, %6814 : tensor<3x3x256x256xf32>
    %7084 = stablehlo.add %arg33, %6817 : tensor<512xf32>
    %7085 = stablehlo.add %arg34, %6820 : tensor<512xf32>
    %7086 = stablehlo.add %arg35, %6823 : tensor<512xf32>
    %7087 = stablehlo.add %arg36, %6826 : tensor<512xf32>
    %7088 = stablehlo.add %arg37, %6829 : tensor<3x3x256x512xf32>
    %7089 = stablehlo.add %arg38, %6832 : tensor<3x3x512x512xf32>
    %7090 = stablehlo.add %arg39, %6835 : tensor<1x1x256x512xf32>
    %7091 = stablehlo.add %arg40, %6838 : tensor<512xf32>
    %7092 = stablehlo.add %arg41, %6841 : tensor<512xf32>
    %7093 = stablehlo.add %arg42, %6844 : tensor<512xf32>
    %7094 = stablehlo.add %arg43, %6847 : tensor<512xf32>
    %7095 = stablehlo.add %arg44, %6850 : tensor<512xf32>
    %7096 = stablehlo.add %arg45, %6853 : tensor<512xf32>
    %7097 = stablehlo.add %arg46, %6856 : tensor<3x3x512x512xf32>
    %7098 = stablehlo.add %arg47, %6859 : tensor<3x3x512x512xf32>
    %7099 = stablehlo.add %arg48, %6862 : tensor<512xf32>
    %7100 = stablehlo.add %arg49, %6865 : tensor<512xf32>
    %7101 = stablehlo.add %arg50, %6868 : tensor<512xf32>
    %7102 = stablehlo.add %arg51, %6871 : tensor<512xf32>
    %7103 = stablehlo.add %arg52, %6874 : tensor<3x3x512x512xf32>
    %7104 = stablehlo.add %arg53, %6877 : tensor<3x3x512x512xf32>
    %7105 = stablehlo.add %arg54, %6880 : tensor<64xf32>
    %7106 = stablehlo.add %arg55, %6883 : tensor<64xf32>
    %7107 = stablehlo.add %arg56, %6886 : tensor<64xf32>
    %7108 = stablehlo.add %arg57, %6889 : tensor<64xf32>
    %7109 = stablehlo.add %arg58, %6892 : tensor<3x3x64x64xf32>
    %7110 = stablehlo.add %arg59, %6895 : tensor<3x3x64x64xf32>
    %7111 = stablehlo.add %arg60, %6898 : tensor<128xf32>
    %7112 = stablehlo.add %arg61, %6901 : tensor<128xf32>
    %7113 = stablehlo.add %arg62, %6904 : tensor<128xf32>
    %7114 = stablehlo.add %arg63, %6907 : tensor<128xf32>
    %7115 = stablehlo.add %arg64, %6910 : tensor<3x3x64x128xf32>
    %7116 = stablehlo.add %arg65, %6913 : tensor<3x3x128x128xf32>
    %7117 = stablehlo.add %arg66, %6916 : tensor<1x1x64x128xf32>
    %7118 = stablehlo.add %arg67, %6919 : tensor<128xf32>
    %7119 = stablehlo.add %arg68, %6922 : tensor<128xf32>
    %7120 = stablehlo.add %arg69, %6925 : tensor<128xf32>
    %7121 = stablehlo.add %arg70, %6928 : tensor<128xf32>
    %7122 = stablehlo.add %arg71, %6931 : tensor<128xf32>
    %7123 = stablehlo.add %arg72, %6934 : tensor<128xf32>
    %7124 = stablehlo.add %arg73, %6937 : tensor<3x3x128x128xf32>
    %7125 = stablehlo.add %arg74, %6940 : tensor<3x3x128x128xf32>
    %7126 = stablehlo.add %arg75, %6943 : tensor<128xf32>
    %7127 = stablehlo.add %arg76, %6946 : tensor<128xf32>
    %7128 = stablehlo.add %arg77, %6949 : tensor<128xf32>
    %7129 = stablehlo.add %arg78, %6952 : tensor<128xf32>
    %7130 = stablehlo.add %arg79, %6955 : tensor<3x3x128x128xf32>
    %7131 = stablehlo.add %arg80, %6958 : tensor<3x3x128x128xf32>
    %7132 = stablehlo.add %arg81, %6961 : tensor<128xf32>
    %7133 = stablehlo.add %arg82, %6964 : tensor<128xf32>
    %7134 = stablehlo.add %arg83, %6967 : tensor<128xf32>
    %7135 = stablehlo.add %arg84, %6970 : tensor<128xf32>
    %7136 = stablehlo.add %arg85, %6973 : tensor<3x3x128x128xf32>
    %7137 = stablehlo.add %arg86, %6976 : tensor<3x3x128x128xf32>
    %7138 = stablehlo.add %arg87, %6979 : tensor<256xf32>
    %7139 = stablehlo.add %arg88, %6982 : tensor<256xf32>
    %7140 = stablehlo.add %arg89, %6985 : tensor<256xf32>
    %7141 = stablehlo.add %arg90, %6988 : tensor<256xf32>
    %7142 = stablehlo.add %arg91, %6991 : tensor<3x3x128x256xf32>
    %7143 = stablehlo.add %arg92, %6994 : tensor<3x3x256x256xf32>
    %7144 = stablehlo.add %arg93, %6997 : tensor<1x1x128x256xf32>
    %7145 = stablehlo.add %arg94, %7000 : tensor<256xf32>
    %7146 = stablehlo.add %arg95, %7003 : tensor<256xf32>
    %7147 = stablehlo.add %arg96, %7006 : tensor<256xf32>
    %7148 = stablehlo.add %arg97, %7009 : tensor<256xf32>
    %7149 = stablehlo.add %arg98, %7012 : tensor<256xf32>
    %7150 = stablehlo.add %arg99, %7015 : tensor<256xf32>
    %7151 = stablehlo.add %arg100, %7018 : tensor<3x3x256x256xf32>
    %7152 = stablehlo.add %arg101, %7021 : tensor<3x3x256x256xf32>
    %7153 = stablehlo.add %arg102, %7024 : tensor<256xf32>
    %7154 = stablehlo.add %arg103, %7027 : tensor<256xf32>
    %7155 = stablehlo.add %arg104, %7030 : tensor<256xf32>
    %7156 = stablehlo.add %arg105, %7033 : tensor<256xf32>
    %7157 = stablehlo.add %arg106, %7036 : tensor<3x3x256x256xf32>
    %7158 = stablehlo.add %arg107, %7039 : tensor<3x3x256x256xf32>
    %7159 = stablehlo.add %arg108, %7042 : tensor<64xf32>
    %7160 = stablehlo.add %arg109, %7045 : tensor<64xf32>
    %7161 = stablehlo.add %arg110, %7048 : tensor<7x7x3x64xf32>
    %7162 = stablehlo.add %arg0, %c_509 : tensor<i32>
    %7163 = call @_where_546(%5773, %6039, %arg111) : (tensor<i1>, tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
    %7164 = call @_where_548(%5773, %6042, %arg112) : (tensor<i1>, tensor<512x1000xf32>, tensor<512x1000xf32>) -> tensor<512x1000xf32>
    %7165 = call @_where_550(%5773, %6045, %arg113) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7166 = call @_where_550(%5773, %6048, %arg114) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7167 = call @_where_550(%5773, %6051, %arg115) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7168 = call @_where_550(%5773, %6054, %arg116) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7169 = call @_where_552(%5773, %6057, %arg117) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7170 = call @_where_552(%5773, %6060, %arg118) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7171 = call @_where_550(%5773, %6063, %arg119) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7172 = call @_where_550(%5773, %6066, %arg120) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7173 = call @_where_550(%5773, %6069, %arg121) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7174 = call @_where_550(%5773, %6072, %arg122) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7175 = call @_where_552(%5773, %6075, %arg123) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7176 = call @_where_552(%5773, %6078, %arg124) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7177 = call @_where_554(%5773, %6081, %arg125) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7178 = call @_where_554(%5773, %6084, %arg126) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7179 = call @_where_554(%5773, %6087, %arg127) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7180 = call @_where_554(%5773, %6090, %arg128) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7181 = call @_where_556(%5773, %6093, %arg129) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7182 = call @_where_556(%5773, %6096, %arg130) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7183 = call @_where_554(%5773, %6099, %arg131) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7184 = call @_where_554(%5773, %6102, %arg132) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7185 = call @_where_554(%5773, %6105, %arg133) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7186 = call @_where_554(%5773, %6108, %arg134) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7187 = call @_where_556(%5773, %6111, %arg135) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7188 = call @_where_556(%5773, %6114, %arg136) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7189 = call @_where_554(%5773, %6117, %arg137) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7190 = call @_where_554(%5773, %6120, %arg138) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7191 = call @_where_554(%5773, %6123, %arg139) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7192 = call @_where_554(%5773, %6126, %arg140) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7193 = call @_where_556(%5773, %6129, %arg141) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7194 = call @_where_556(%5773, %6132, %arg142) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7195 = call @_where_558(%5773, %6135, %arg143) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7196 = call @_where_558(%5773, %6138, %arg144) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7197 = call @_where_558(%5773, %6141, %arg145) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7198 = call @_where_558(%5773, %6144, %arg146) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7199 = call @_where_560(%5773, %6147, %arg147) : (tensor<i1>, tensor<3x3x256x512xf32>, tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
    %7200 = call @_where_562(%5773, %6150, %arg148) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7201 = call @_where_564(%5773, %6153, %arg149) : (tensor<i1>, tensor<1x1x256x512xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    %7202 = call @_where_558(%5773, %6156, %arg150) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7203 = call @_where_558(%5773, %6159, %arg151) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7204 = call @_where_558(%5773, %6162, %arg152) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7205 = call @_where_558(%5773, %6165, %arg153) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7206 = call @_where_558(%5773, %6168, %arg154) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7207 = call @_where_558(%5773, %6171, %arg155) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7208 = call @_where_562(%5773, %6174, %arg156) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7209 = call @_where_562(%5773, %6177, %arg157) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7210 = call @_where_558(%5773, %6180, %arg158) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7211 = call @_where_558(%5773, %6183, %arg159) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7212 = call @_where_558(%5773, %6186, %arg160) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7213 = call @_where_558(%5773, %6189, %arg161) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7214 = call @_where_562(%5773, %6192, %arg162) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7215 = call @_where_562(%5773, %6195, %arg163) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7216 = call @_where_550(%5773, %6198, %arg164) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7217 = call @_where_550(%5773, %6201, %arg165) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7218 = call @_where_550(%5773, %6204, %arg166) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7219 = call @_where_550(%5773, %6207, %arg167) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7220 = call @_where_552(%5773, %6210, %arg168) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7221 = call @_where_552(%5773, %6213, %arg169) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7222 = call @_where_566(%5773, %6216, %arg170) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7223 = call @_where_566(%5773, %6219, %arg171) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7224 = call @_where_566(%5773, %6222, %arg172) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7225 = call @_where_566(%5773, %6225, %arg173) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7226 = call @_where_568(%5773, %6228, %arg174) : (tensor<i1>, tensor<3x3x64x128xf32>, tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
    %7227 = call @_where_570(%5773, %6231, %arg175) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7228 = call @_where_572(%5773, %6234, %arg176) : (tensor<i1>, tensor<1x1x64x128xf32>, tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
    %7229 = call @_where_566(%5773, %6237, %arg177) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7230 = call @_where_566(%5773, %6240, %arg178) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7231 = call @_where_566(%5773, %6243, %arg179) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7232 = call @_where_566(%5773, %6246, %arg180) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7233 = call @_where_566(%5773, %6249, %arg181) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7234 = call @_where_566(%5773, %6252, %arg182) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7235 = call @_where_570(%5773, %6255, %arg183) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7236 = call @_where_570(%5773, %6258, %arg184) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7237 = call @_where_566(%5773, %6261, %arg185) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7238 = call @_where_566(%5773, %6264, %arg186) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7239 = call @_where_566(%5773, %6267, %arg187) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7240 = call @_where_566(%5773, %6270, %arg188) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7241 = call @_where_570(%5773, %6273, %arg189) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7242 = call @_where_570(%5773, %6276, %arg190) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7243 = call @_where_566(%5773, %6279, %arg191) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7244 = call @_where_566(%5773, %6282, %arg192) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7245 = call @_where_566(%5773, %6285, %arg193) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7246 = call @_where_566(%5773, %6288, %arg194) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7247 = call @_where_570(%5773, %6291, %arg195) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7248 = call @_where_570(%5773, %6294, %arg196) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7249 = call @_where_554(%5773, %6297, %arg197) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7250 = call @_where_554(%5773, %6300, %arg198) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7251 = call @_where_554(%5773, %6303, %arg199) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7252 = call @_where_554(%5773, %6306, %arg200) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7253 = call @_where_574(%5773, %6309, %arg201) : (tensor<i1>, tensor<3x3x128x256xf32>, tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
    %7254 = call @_where_556(%5773, %6312, %arg202) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7255 = call @_where_576(%5773, %6315, %arg203) : (tensor<i1>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
    %7256 = call @_where_554(%5773, %6318, %arg204) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7257 = call @_where_554(%5773, %6321, %arg205) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7258 = call @_where_554(%5773, %6324, %arg206) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7259 = call @_where_554(%5773, %6327, %arg207) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7260 = call @_where_554(%5773, %6330, %arg208) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7261 = call @_where_554(%5773, %6333, %arg209) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7262 = call @_where_556(%5773, %6336, %arg210) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7263 = call @_where_556(%5773, %6339, %arg211) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7264 = call @_where_554(%5773, %6342, %arg212) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7265 = call @_where_554(%5773, %6345, %arg213) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7266 = call @_where_554(%5773, %6348, %arg214) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7267 = call @_where_554(%5773, %6351, %arg215) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7268 = call @_where_556(%5773, %6354, %arg216) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7269 = call @_where_556(%5773, %6357, %arg217) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7270 = call @_where_550(%5773, %6360, %arg218) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7271 = call @_where_550(%5773, %6363, %arg219) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7272 = call @_where_578(%5773, %6366, %arg220) : (tensor<i1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %7273 = call @_where_545(%5773, %7051, %arg221) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %7274 = call @_where_546(%5773, %7052, %arg1) : (tensor<i1>, tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
    %7275 = call @_where_548(%5773, %7053, %arg2) : (tensor<i1>, tensor<512x1000xf32>, tensor<512x1000xf32>) -> tensor<512x1000xf32>
    %7276 = call @_where_550(%5773, %7054, %arg3) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7277 = call @_where_550(%5773, %7055, %arg4) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7278 = call @_where_550(%5773, %7056, %arg5) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7279 = call @_where_550(%5773, %7057, %arg6) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7280 = call @_where_552(%5773, %7058, %arg7) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7281 = call @_where_552(%5773, %7059, %arg8) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7282 = call @_where_550(%5773, %7060, %arg9) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7283 = call @_where_550(%5773, %7061, %arg10) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7284 = call @_where_550(%5773, %7062, %arg11) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7285 = call @_where_550(%5773, %7063, %arg12) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7286 = call @_where_552(%5773, %7064, %arg13) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7287 = call @_where_552(%5773, %7065, %arg14) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7288 = call @_where_554(%5773, %7066, %arg15) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7289 = call @_where_554(%5773, %7067, %arg16) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7290 = call @_where_554(%5773, %7068, %arg17) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7291 = call @_where_554(%5773, %7069, %arg18) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7292 = call @_where_556(%5773, %7070, %arg19) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7293 = call @_where_556(%5773, %7071, %arg20) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7294 = call @_where_554(%5773, %7072, %arg21) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7295 = call @_where_554(%5773, %7073, %arg22) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7296 = call @_where_554(%5773, %7074, %arg23) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7297 = call @_where_554(%5773, %7075, %arg24) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7298 = call @_where_556(%5773, %7076, %arg25) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7299 = call @_where_556(%5773, %7077, %arg26) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7300 = call @_where_554(%5773, %7078, %arg27) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7301 = call @_where_554(%5773, %7079, %arg28) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7302 = call @_where_554(%5773, %7080, %arg29) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7303 = call @_where_554(%5773, %7081, %arg30) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7304 = call @_where_556(%5773, %7082, %arg31) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7305 = call @_where_556(%5773, %7083, %arg32) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7306 = call @_where_558(%5773, %7084, %arg33) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7307 = call @_where_558(%5773, %7085, %arg34) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7308 = call @_where_558(%5773, %7086, %arg35) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7309 = call @_where_558(%5773, %7087, %arg36) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7310 = call @_where_560(%5773, %7088, %arg37) : (tensor<i1>, tensor<3x3x256x512xf32>, tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
    %7311 = call @_where_562(%5773, %7089, %arg38) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7312 = call @_where_564(%5773, %7090, %arg39) : (tensor<i1>, tensor<1x1x256x512xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    %7313 = call @_where_558(%5773, %7091, %arg40) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7314 = call @_where_558(%5773, %7092, %arg41) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7315 = call @_where_558(%5773, %7093, %arg42) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7316 = call @_where_558(%5773, %7094, %arg43) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7317 = call @_where_558(%5773, %7095, %arg44) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7318 = call @_where_558(%5773, %7096, %arg45) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7319 = call @_where_562(%5773, %7097, %arg46) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7320 = call @_where_562(%5773, %7098, %arg47) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7321 = call @_where_558(%5773, %7099, %arg48) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7322 = call @_where_558(%5773, %7100, %arg49) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7323 = call @_where_558(%5773, %7101, %arg50) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7324 = call @_where_558(%5773, %7102, %arg51) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %7325 = call @_where_562(%5773, %7103, %arg52) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7326 = call @_where_562(%5773, %7104, %arg53) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
    %7327 = call @_where_550(%5773, %7105, %arg54) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7328 = call @_where_550(%5773, %7106, %arg55) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7329 = call @_where_550(%5773, %7107, %arg56) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7330 = call @_where_550(%5773, %7108, %arg57) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7331 = call @_where_552(%5773, %7109, %arg58) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7332 = call @_where_552(%5773, %7110, %arg59) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
    %7333 = call @_where_566(%5773, %7111, %arg60) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7334 = call @_where_566(%5773, %7112, %arg61) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7335 = call @_where_566(%5773, %7113, %arg62) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7336 = call @_where_566(%5773, %7114, %arg63) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7337 = call @_where_568(%5773, %7115, %arg64) : (tensor<i1>, tensor<3x3x64x128xf32>, tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
    %7338 = call @_where_570(%5773, %7116, %arg65) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7339 = call @_where_572(%5773, %7117, %arg66) : (tensor<i1>, tensor<1x1x64x128xf32>, tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
    %7340 = call @_where_566(%5773, %7118, %arg67) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7341 = call @_where_566(%5773, %7119, %arg68) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7342 = call @_where_566(%5773, %7120, %arg69) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7343 = call @_where_566(%5773, %7121, %arg70) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7344 = call @_where_566(%5773, %7122, %arg71) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7345 = call @_where_566(%5773, %7123, %arg72) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7346 = call @_where_570(%5773, %7124, %arg73) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7347 = call @_where_570(%5773, %7125, %arg74) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7348 = call @_where_566(%5773, %7126, %arg75) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7349 = call @_where_566(%5773, %7127, %arg76) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7350 = call @_where_566(%5773, %7128, %arg77) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7351 = call @_where_566(%5773, %7129, %arg78) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7352 = call @_where_570(%5773, %7130, %arg79) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7353 = call @_where_570(%5773, %7131, %arg80) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7354 = call @_where_566(%5773, %7132, %arg81) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7355 = call @_where_566(%5773, %7133, %arg82) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7356 = call @_where_566(%5773, %7134, %arg83) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7357 = call @_where_566(%5773, %7135, %arg84) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %7358 = call @_where_570(%5773, %7136, %arg85) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7359 = call @_where_570(%5773, %7137, %arg86) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
    %7360 = call @_where_554(%5773, %7138, %arg87) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7361 = call @_where_554(%5773, %7139, %arg88) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7362 = call @_where_554(%5773, %7140, %arg89) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7363 = call @_where_554(%5773, %7141, %arg90) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7364 = call @_where_574(%5773, %7142, %arg91) : (tensor<i1>, tensor<3x3x128x256xf32>, tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
    %7365 = call @_where_556(%5773, %7143, %arg92) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7366 = call @_where_576(%5773, %7144, %arg93) : (tensor<i1>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
    %7367 = call @_where_554(%5773, %7145, %arg94) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7368 = call @_where_554(%5773, %7146, %arg95) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7369 = call @_where_554(%5773, %7147, %arg96) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7370 = call @_where_554(%5773, %7148, %arg97) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7371 = call @_where_554(%5773, %7149, %arg98) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7372 = call @_where_554(%5773, %7150, %arg99) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7373 = call @_where_556(%5773, %7151, %arg100) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7374 = call @_where_556(%5773, %7152, %arg101) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7375 = call @_where_554(%5773, %7153, %arg102) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7376 = call @_where_554(%5773, %7154, %arg103) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7377 = call @_where_554(%5773, %7155, %arg104) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7378 = call @_where_554(%5773, %7156, %arg105) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %7379 = call @_where_556(%5773, %7157, %arg106) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7380 = call @_where_556(%5773, %7158, %arg107) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %7381 = call @_where_550(%5773, %7159, %arg108) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7382 = call @_where_550(%5773, %7160, %arg109) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %7383 = call @_where_578(%5773, %7161, %arg110) : (tensor<i1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %7384 = stablehlo.broadcast_in_dim %5819, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %7385 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %7386 = stablehlo.broadcast_in_dim %5820, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %7387 = stablehlo.broadcast_in_dim %5786, dims = [] : (tensor<f32>) -> tensor<1xf32>
    return %7162, %7274, %7275, %7276, %7277, %7278, %7279, %7280, %7281, %7282, %7283, %7284, %7285, %7286, %7287, %7288, %7289, %7290, %7291, %7292, %7293, %7294, %7295, %7296, %7297, %7298, %7299, %7300, %7301, %7302, %7303, %7304, %7305, %7306, %7307, %7308, %7309, %7310, %7311, %7312, %7313, %7314, %7315, %7316, %7317, %7318, %7319, %7320, %7321, %7322, %7323, %7324, %7325, %7326, %7327, %7328, %7329, %7330, %7331, %7332, %7333, %7334, %7335, %7336, %7337, %7338, %7339, %7340, %7341, %7342, %7343, %7344, %7345, %7346, %7347, %7348, %7349, %7350, %7351, %7352, %7353, %7354, %7355, %7356, %7357, %7358, %7359, %7360, %7361, %7362, %7363, %7364, %7365, %7366, %7367, %7368, %7369, %7370, %7371, %7372, %7373, %7374, %7375, %7376, %7377, %7378, %7379, %7380, %7381, %7382, %7383, %7163, %7164, %7165, %7166, %7167, %7168, %7169, %7170, %7171, %7172, %7173, %7174, %7175, %7176, %7177, %7178, %7179, %7180, %7181, %7182, %7183, %7184, %7185, %7186, %7187, %7188, %7189, %7190, %7191, %7192, %7193, %7194, %7195, %7196, %7197, %7198, %7199, %7200, %7201, %7202, %7203, %7204, %7205, %7206, %7207, %7208, %7209, %7210, %7211, %7212, %7213, %7214, %7215, %7216, %7217, %7218, %7219, %7220, %7221, %7222, %7223, %7224, %7225, %7226, %7227, %7228, %7229, %7230, %7231, %7232, %7233, %7234, %7235, %7236, %7237, %7238, %7239, %7240, %7241, %7242, %7243, %7244, %7245, %7246, %7247, %7248, %7249, %7250, %7251, %7252, %7253, %7254, %7255, %7256, %7257, %7258, %7259, %7260, %7261, %7262, %7263, %7264, %7265, %7266, %7267, %7268, %7269, %7270, %7271, %7272, %7273, %5894, %5896, %5898, %5900, %5902, %5904, %5906, %5908, %5910, %5912, %5914, %5916, %5918, %5920, %5922, %5924, %5926, %5928, %5930, %5932, %5934, %5936, %5938, %5940, %5942, %5944, %5946, %5948, %5950, %5952, %5954, %5956, %5958, %5960, %5962, %5964, %5966, %5968, %5970, %5972, %5974, %5976, %5978, %5980, %5982, %5984, %5986, %5988, %5990, %5992, %5994, %5996, %5998, %6000, %6002, %6004, %6006, %6008, %6010, %6012, %6014, %6016, %6018, %6020, %6022, %6024, %6026, %6028, %6030, %6032, %6034, %6036, %5793, %5786, %7384, %7385, %7386, %7387 : tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
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
  func.func private @relu_87(%arg0: tensor<256x28x28x128xbf16>) -> tensor<256x28x28x128xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x28x28x128xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xbf16>
    return %1 : tensor<256x28x28x128xbf16>
  }
  func.func private @relu_139(%arg0: tensor<256x14x14x256xbf16>) -> tensor<256x14x14x256xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<256x14x14x256xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xbf16>
    return %1 : tensor<256x14x14x256xbf16>
  }
  func.func private @relu_195(%arg0: tensor<256x7x7x512xbf16>) -> tensor<256x7x7x512xbf16> {
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
  func.func private @log_softmax_308(%arg0: tensor<256x1000xbf16>, %arg1: tensor<256x1xbf16>, %arg2: tensor<256x1000xbf16>) -> tensor<256x1000xbf16> {
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
  func.func private @_where_506(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_509(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_511(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @log_softmax_513(%arg0: tensor<256x1000xbf16>) -> tensor<256x1000xbf16> {
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
  func.func private @clip_540(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %3 = stablehlo.minimum %arg2, %2 : tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @_where_545(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @_where_546(%arg0: tensor<i1>, %arg1: tensor<1000xf32>, %arg2: tensor<1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
  func.func private @_where_548(%arg0: tensor<i1>, %arg1: tensor<512x1000xf32>, %arg2: tensor<512x1000xf32>) -> tensor<512x1000xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<512x1000xf32>
    return %0 : tensor<512x1000xf32>
  }
  func.func private @_where_550(%arg0: tensor<i1>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @_where_552(%arg0: tensor<i1>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x64x64xf32>
    return %0 : tensor<3x3x64x64xf32>
  }
  func.func private @_where_554(%arg0: tensor<i1>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  func.func private @_where_556(%arg0: tensor<i1>, %arg1: tensor<3x3x256x256xf32>, %arg2: tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x256x256xf32>
    return %0 : tensor<3x3x256x256xf32>
  }
  func.func private @_where_558(%arg0: tensor<i1>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @_where_560(%arg0: tensor<i1>, %arg1: tensor<3x3x256x512xf32>, %arg2: tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x256x512xf32>
    return %0 : tensor<3x3x256x512xf32>
  }
  func.func private @_where_562(%arg0: tensor<i1>, %arg1: tensor<3x3x512x512xf32>, %arg2: tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x512x512xf32>
    return %0 : tensor<3x3x512x512xf32>
  }
  func.func private @_where_564(%arg0: tensor<i1>, %arg1: tensor<1x1x256x512xf32>, %arg2: tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x256x512xf32>
    return %0 : tensor<1x1x256x512xf32>
  }
  func.func private @_where_566(%arg0: tensor<i1>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  func.func private @_where_568(%arg0: tensor<i1>, %arg1: tensor<3x3x64x128xf32>, %arg2: tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x64x128xf32>
    return %0 : tensor<3x3x64x128xf32>
  }
  func.func private @_where_570(%arg0: tensor<i1>, %arg1: tensor<3x3x128x128xf32>, %arg2: tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x128x128xf32>
    return %0 : tensor<3x3x128x128xf32>
  }
  func.func private @_where_572(%arg0: tensor<i1>, %arg1: tensor<1x1x64x128xf32>, %arg2: tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x64x128xf32>
    return %0 : tensor<1x1x64x128xf32>
  }
  func.func private @_where_574(%arg0: tensor<i1>, %arg1: tensor<3x3x128x256xf32>, %arg2: tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x128x256xf32>
    return %0 : tensor<3x3x128x256xf32>
  }
  func.func private @_where_576(%arg0: tensor<i1>, %arg1: tensor<1x1x128x256xf32>, %arg2: tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x128x256xf32>
    return %0 : tensor<1x1x128x256xf32>
  }
  func.func private @_where_578(%arg0: tensor<i1>, %arg1: tensor<7x7x3x64xf32>, %arg2: tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<7x7x3x64xf32>
    return %0 : tensor<7x7x3x64xf32>
  }
}
