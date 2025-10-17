module @jit_call_wrapped attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  sdy.mesh @mesh = <["batch"=4]>
  func.func public @main(%arg0: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg1: tensor<1000xf32>, %arg2: tensor<512x1000xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<3x3x64x64xf32>, %arg8: tensor<3x3x64x64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64xf32>, %arg13: tensor<3x3x64x64xf32>, %arg14: tensor<3x3x64x64xf32>, %arg15: tensor<256xf32>, %arg16: tensor<256xf32>, %arg17: tensor<256xf32>, %arg18: tensor<256xf32>, %arg19: tensor<3x3x256x256xf32>, %arg20: tensor<3x3x256x256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256xf32>, %arg25: tensor<3x3x256x256xf32>, %arg26: tensor<3x3x256x256xf32>, %arg27: tensor<256xf32>, %arg28: tensor<256xf32>, %arg29: tensor<256xf32>, %arg30: tensor<256xf32>, %arg31: tensor<3x3x256x256xf32>, %arg32: tensor<3x3x256x256xf32>, %arg33: tensor<512xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<512xf32>, %arg37: tensor<3x3x256x512xf32>, %arg38: tensor<3x3x512x512xf32>, %arg39: tensor<1x1x256x512xf32>, %arg40: tensor<512xf32>, %arg41: tensor<512xf32>, %arg42: tensor<512xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<512xf32>, %arg46: tensor<3x3x512x512xf32>, %arg47: tensor<3x3x512x512xf32>, %arg48: tensor<512xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<512xf32>, %arg52: tensor<3x3x512x512xf32>, %arg53: tensor<3x3x512x512xf32>, %arg54: tensor<64xf32>, %arg55: tensor<64xf32>, %arg56: tensor<64xf32>, %arg57: tensor<64xf32>, %arg58: tensor<3x3x64x64xf32>, %arg59: tensor<3x3x64x64xf32>, %arg60: tensor<128xf32>, %arg61: tensor<128xf32>, %arg62: tensor<128xf32>, %arg63: tensor<128xf32>, %arg64: tensor<3x3x64x128xf32>, %arg65: tensor<3x3x128x128xf32>, %arg66: tensor<1x1x64x128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<128xf32>, %arg70: tensor<128xf32>, %arg71: tensor<128xf32>, %arg72: tensor<128xf32>, %arg73: tensor<3x3x128x128xf32>, %arg74: tensor<3x3x128x128xf32>, %arg75: tensor<128xf32>, %arg76: tensor<128xf32>, %arg77: tensor<128xf32>, %arg78: tensor<128xf32>, %arg79: tensor<3x3x128x128xf32>, %arg80: tensor<3x3x128x128xf32>, %arg81: tensor<128xf32>, %arg82: tensor<128xf32>, %arg83: tensor<128xf32>, %arg84: tensor<128xf32>, %arg85: tensor<3x3x128x128xf32>, %arg86: tensor<3x3x128x128xf32>, %arg87: tensor<256xf32>, %arg88: tensor<256xf32>, %arg89: tensor<256xf32>, %arg90: tensor<256xf32>, %arg91: tensor<3x3x128x256xf32>, %arg92: tensor<3x3x256x256xf32>, %arg93: tensor<1x1x128x256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256xf32>, %arg97: tensor<256xf32>, %arg98: tensor<256xf32>, %arg99: tensor<256xf32>, %arg100: tensor<3x3x256x256xf32>, %arg101: tensor<3x3x256x256xf32>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256xf32>, %arg105: tensor<256xf32>, %arg106: tensor<3x3x256x256xf32>, %arg107: tensor<3x3x256x256xf32>, %arg108: tensor<64xf32>, %arg109: tensor<64xf32>, %arg110: tensor<7x7x3x64xf32>, %arg111: tensor<1000xf32>, %arg112: tensor<512x1000xf32>, %arg113: tensor<64xf32>, %arg114: tensor<64xf32>, %arg115: tensor<64xf32>, %arg116: tensor<64xf32>, %arg117: tensor<3x3x64x64xf32>, %arg118: tensor<3x3x64x64xf32>, %arg119: tensor<64xf32>, %arg120: tensor<64xf32>, %arg121: tensor<64xf32>, %arg122: tensor<64xf32>, %arg123: tensor<3x3x64x64xf32>, %arg124: tensor<3x3x64x64xf32>, %arg125: tensor<256xf32>, %arg126: tensor<256xf32>, %arg127: tensor<256xf32>, %arg128: tensor<256xf32>, %arg129: tensor<3x3x256x256xf32>, %arg130: tensor<3x3x256x256xf32>, %arg131: tensor<256xf32>, %arg132: tensor<256xf32>, %arg133: tensor<256xf32>, %arg134: tensor<256xf32>, %arg135: tensor<3x3x256x256xf32>, %arg136: tensor<3x3x256x256xf32>, %arg137: tensor<256xf32>, %arg138: tensor<256xf32>, %arg139: tensor<256xf32>, %arg140: tensor<256xf32>, %arg141: tensor<3x3x256x256xf32>, %arg142: tensor<3x3x256x256xf32>, %arg143: tensor<512xf32>, %arg144: tensor<512xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<3x3x256x512xf32>, %arg148: tensor<3x3x512x512xf32>, %arg149: tensor<1x1x256x512xf32>, %arg150: tensor<512xf32>, %arg151: tensor<512xf32>, %arg152: tensor<512xf32>, %arg153: tensor<512xf32>, %arg154: tensor<512xf32>, %arg155: tensor<512xf32>, %arg156: tensor<3x3x512x512xf32>, %arg157: tensor<3x3x512x512xf32>, %arg158: tensor<512xf32>, %arg159: tensor<512xf32>, %arg160: tensor<512xf32>, %arg161: tensor<512xf32>, %arg162: tensor<3x3x512x512xf32>, %arg163: tensor<3x3x512x512xf32>, %arg164: tensor<64xf32>, %arg165: tensor<64xf32>, %arg166: tensor<64xf32>, %arg167: tensor<64xf32>, %arg168: tensor<3x3x64x64xf32>, %arg169: tensor<3x3x64x64xf32>, %arg170: tensor<128xf32>, %arg171: tensor<128xf32>, %arg172: tensor<128xf32>, %arg173: tensor<128xf32>, %arg174: tensor<3x3x64x128xf32>, %arg175: tensor<3x3x128x128xf32>, %arg176: tensor<1x1x64x128xf32>, %arg177: tensor<128xf32>, %arg178: tensor<128xf32>, %arg179: tensor<128xf32>, %arg180: tensor<128xf32>, %arg181: tensor<128xf32>, %arg182: tensor<128xf32>, %arg183: tensor<3x3x128x128xf32>, %arg184: tensor<3x3x128x128xf32>, %arg185: tensor<128xf32>, %arg186: tensor<128xf32>, %arg187: tensor<128xf32>, %arg188: tensor<128xf32>, %arg189: tensor<3x3x128x128xf32>, %arg190: tensor<3x3x128x128xf32>, %arg191: tensor<128xf32>, %arg192: tensor<128xf32>, %arg193: tensor<128xf32>, %arg194: tensor<128xf32>, %arg195: tensor<3x3x128x128xf32>, %arg196: tensor<3x3x128x128xf32>, %arg197: tensor<256xf32>, %arg198: tensor<256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256xf32>, %arg201: tensor<3x3x128x256xf32>, %arg202: tensor<3x3x256x256xf32>, %arg203: tensor<1x1x128x256xf32>, %arg204: tensor<256xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<256xf32>, %arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<3x3x256x256xf32>, %arg211: tensor<3x3x256x256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<3x3x256x256xf32>, %arg217: tensor<3x3x256x256xf32>, %arg218: tensor<64xf32>, %arg219: tensor<64xf32>, %arg220: tensor<7x7x3x64xf32>, %arg221: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg222: tensor<64xf32>, %arg223: tensor<64xf32>, %arg224: tensor<64xf32>, %arg225: tensor<64xf32>, %arg226: tensor<64xf32>, %arg227: tensor<64xf32>, %arg228: tensor<64xf32>, %arg229: tensor<64xf32>, %arg230: tensor<256xf32>, %arg231: tensor<256xf32>, %arg232: tensor<256xf32>, %arg233: tensor<256xf32>, %arg234: tensor<256xf32>, %arg235: tensor<256xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<256xf32>, %arg239: tensor<256xf32>, %arg240: tensor<256xf32>, %arg241: tensor<256xf32>, %arg242: tensor<512xf32>, %arg243: tensor<512xf32>, %arg244: tensor<512xf32>, %arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>, %arg250: tensor<512xf32>, %arg251: tensor<512xf32>, %arg252: tensor<512xf32>, %arg253: tensor<512xf32>, %arg254: tensor<512xf32>, %arg255: tensor<512xf32>, %arg256: tensor<64xf32>, %arg257: tensor<64xf32>, %arg258: tensor<64xf32>, %arg259: tensor<64xf32>, %arg260: tensor<128xf32>, %arg261: tensor<128xf32>, %arg262: tensor<128xf32>, %arg263: tensor<128xf32>, %arg264: tensor<128xf32>, %arg265: tensor<128xf32>, %arg266: tensor<128xf32>, %arg267: tensor<128xf32>, %arg268: tensor<128xf32>, %arg269: tensor<128xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>, %arg273: tensor<128xf32>, %arg274: tensor<128xf32>, %arg275: tensor<128xf32>, %arg276: tensor<128xf32>, %arg277: tensor<128xf32>, %arg278: tensor<256xf32>, %arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<256xf32>, %arg284: tensor<256xf32>, %arg285: tensor<256xf32>, %arg286: tensor<256xf32>, %arg287: tensor<256xf32>, %arg288: tensor<256xf32>, %arg289: tensor<256xf32>, %arg290: tensor<256xf32>, %arg291: tensor<256xf32>, %arg292: tensor<64xf32>, %arg293: tensor<64xf32>, %arg294: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg295: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg296: tensor<4x256x224x224x3xf16>, %arg297: tensor<4x256xi32>) -> (tensor<i32> {jax.result_info = "result[0]"}, tensor<1000xf32> {jax.result_info = "result[1]"}, tensor<512x1000xf32> {jax.result_info = "result[2]"}, tensor<64xf32> {jax.result_info = "result[3]"}, tensor<64xf32> {jax.result_info = "result[4]"}, tensor<64xf32> {jax.result_info = "result[5]"}, tensor<64xf32> {jax.result_info = "result[6]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[7]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[8]"}, tensor<64xf32> {jax.result_info = "result[9]"}, tensor<64xf32> {jax.result_info = "result[10]"}, tensor<64xf32> {jax.result_info = "result[11]"}, tensor<64xf32> {jax.result_info = "result[12]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[13]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[14]"}, tensor<256xf32> {jax.result_info = "result[15]"}, tensor<256xf32> {jax.result_info = "result[16]"}, tensor<256xf32> {jax.result_info = "result[17]"}, tensor<256xf32> {jax.result_info = "result[18]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[19]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[20]"}, tensor<256xf32> {jax.result_info = "result[21]"}, tensor<256xf32> {jax.result_info = "result[22]"}, tensor<256xf32> {jax.result_info = "result[23]"}, tensor<256xf32> {jax.result_info = "result[24]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[25]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[26]"}, tensor<256xf32> {jax.result_info = "result[27]"}, tensor<256xf32> {jax.result_info = "result[28]"}, tensor<256xf32> {jax.result_info = "result[29]"}, tensor<256xf32> {jax.result_info = "result[30]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[31]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[32]"}, tensor<512xf32> {jax.result_info = "result[33]"}, tensor<512xf32> {jax.result_info = "result[34]"}, tensor<512xf32> {jax.result_info = "result[35]"}, tensor<512xf32> {jax.result_info = "result[36]"}, tensor<3x3x256x512xf32> {jax.result_info = "result[37]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[38]"}, tensor<1x1x256x512xf32> {jax.result_info = "result[39]"}, tensor<512xf32> {jax.result_info = "result[40]"}, tensor<512xf32> {jax.result_info = "result[41]"}, tensor<512xf32> {jax.result_info = "result[42]"}, tensor<512xf32> {jax.result_info = "result[43]"}, tensor<512xf32> {jax.result_info = "result[44]"}, tensor<512xf32> {jax.result_info = "result[45]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[46]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[47]"}, tensor<512xf32> {jax.result_info = "result[48]"}, tensor<512xf32> {jax.result_info = "result[49]"}, tensor<512xf32> {jax.result_info = "result[50]"}, tensor<512xf32> {jax.result_info = "result[51]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[52]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[53]"}, tensor<64xf32> {jax.result_info = "result[54]"}, tensor<64xf32> {jax.result_info = "result[55]"}, tensor<64xf32> {jax.result_info = "result[56]"}, tensor<64xf32> {jax.result_info = "result[57]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[58]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[59]"}, tensor<128xf32> {jax.result_info = "result[60]"}, tensor<128xf32> {jax.result_info = "result[61]"}, tensor<128xf32> {jax.result_info = "result[62]"}, tensor<128xf32> {jax.result_info = "result[63]"}, tensor<3x3x64x128xf32> {jax.result_info = "result[64]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[65]"}, tensor<1x1x64x128xf32> {jax.result_info = "result[66]"}, tensor<128xf32> {jax.result_info = "result[67]"}, tensor<128xf32> {jax.result_info = "result[68]"}, tensor<128xf32> {jax.result_info = "result[69]"}, tensor<128xf32> {jax.result_info = "result[70]"}, tensor<128xf32> {jax.result_info = "result[71]"}, tensor<128xf32> {jax.result_info = "result[72]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[73]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[74]"}, tensor<128xf32> {jax.result_info = "result[75]"}, tensor<128xf32> {jax.result_info = "result[76]"}, tensor<128xf32> {jax.result_info = "result[77]"}, tensor<128xf32> {jax.result_info = "result[78]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[79]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[80]"}, tensor<128xf32> {jax.result_info = "result[81]"}, tensor<128xf32> {jax.result_info = "result[82]"}, tensor<128xf32> {jax.result_info = "result[83]"}, tensor<128xf32> {jax.result_info = "result[84]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[85]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[86]"}, tensor<256xf32> {jax.result_info = "result[87]"}, tensor<256xf32> {jax.result_info = "result[88]"}, tensor<256xf32> {jax.result_info = "result[89]"}, tensor<256xf32> {jax.result_info = "result[90]"}, tensor<3x3x128x256xf32> {jax.result_info = "result[91]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[92]"}, tensor<1x1x128x256xf32> {jax.result_info = "result[93]"}, tensor<256xf32> {jax.result_info = "result[94]"}, tensor<256xf32> {jax.result_info = "result[95]"}, tensor<256xf32> {jax.result_info = "result[96]"}, tensor<256xf32> {jax.result_info = "result[97]"}, tensor<256xf32> {jax.result_info = "result[98]"}, tensor<256xf32> {jax.result_info = "result[99]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[100]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[101]"}, tensor<256xf32> {jax.result_info = "result[102]"}, tensor<256xf32> {jax.result_info = "result[103]"}, tensor<256xf32> {jax.result_info = "result[104]"}, tensor<256xf32> {jax.result_info = "result[105]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[106]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[107]"}, tensor<64xf32> {jax.result_info = "result[108]"}, tensor<64xf32> {jax.result_info = "result[109]"}, tensor<7x7x3x64xf32> {jax.result_info = "result[110]"}, tensor<1000xf32> {jax.result_info = "result[111]"}, tensor<512x1000xf32> {jax.result_info = "result[112]"}, tensor<64xf32> {jax.result_info = "result[113]"}, tensor<64xf32> {jax.result_info = "result[114]"}, tensor<64xf32> {jax.result_info = "result[115]"}, tensor<64xf32> {jax.result_info = "result[116]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[117]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[118]"}, tensor<64xf32> {jax.result_info = "result[119]"}, tensor<64xf32> {jax.result_info = "result[120]"}, tensor<64xf32> {jax.result_info = "result[121]"}, tensor<64xf32> {jax.result_info = "result[122]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[123]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[124]"}, tensor<256xf32> {jax.result_info = "result[125]"}, tensor<256xf32> {jax.result_info = "result[126]"}, tensor<256xf32> {jax.result_info = "result[127]"}, tensor<256xf32> {jax.result_info = "result[128]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[129]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[130]"}, tensor<256xf32> {jax.result_info = "result[131]"}, tensor<256xf32> {jax.result_info = "result[132]"}, tensor<256xf32> {jax.result_info = "result[133]"}, tensor<256xf32> {jax.result_info = "result[134]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[135]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[136]"}, tensor<256xf32> {jax.result_info = "result[137]"}, tensor<256xf32> {jax.result_info = "result[138]"}, tensor<256xf32> {jax.result_info = "result[139]"}, tensor<256xf32> {jax.result_info = "result[140]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[141]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[142]"}, tensor<512xf32> {jax.result_info = "result[143]"}, tensor<512xf32> {jax.result_info = "result[144]"}, tensor<512xf32> {jax.result_info = "result[145]"}, tensor<512xf32> {jax.result_info = "result[146]"}, tensor<3x3x256x512xf32> {jax.result_info = "result[147]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[148]"}, tensor<1x1x256x512xf32> {jax.result_info = "result[149]"}, tensor<512xf32> {jax.result_info = "result[150]"}, tensor<512xf32> {jax.result_info = "result[151]"}, tensor<512xf32> {jax.result_info = "result[152]"}, tensor<512xf32> {jax.result_info = "result[153]"}, tensor<512xf32> {jax.result_info = "result[154]"}, tensor<512xf32> {jax.result_info = "result[155]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[156]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[157]"}, tensor<512xf32> {jax.result_info = "result[158]"}, tensor<512xf32> {jax.result_info = "result[159]"}, tensor<512xf32> {jax.result_info = "result[160]"}, tensor<512xf32> {jax.result_info = "result[161]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[162]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[163]"}, tensor<64xf32> {jax.result_info = "result[164]"}, tensor<64xf32> {jax.result_info = "result[165]"}, tensor<64xf32> {jax.result_info = "result[166]"}, tensor<64xf32> {jax.result_info = "result[167]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[168]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[169]"}, tensor<128xf32> {jax.result_info = "result[170]"}, tensor<128xf32> {jax.result_info = "result[171]"}, tensor<128xf32> {jax.result_info = "result[172]"}, tensor<128xf32> {jax.result_info = "result[173]"}, tensor<3x3x64x128xf32> {jax.result_info = "result[174]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[175]"}, tensor<1x1x64x128xf32> {jax.result_info = "result[176]"}, tensor<128xf32> {jax.result_info = "result[177]"}, tensor<128xf32> {jax.result_info = "result[178]"}, tensor<128xf32> {jax.result_info = "result[179]"}, tensor<128xf32> {jax.result_info = "result[180]"}, tensor<128xf32> {jax.result_info = "result[181]"}, tensor<128xf32> {jax.result_info = "result[182]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[183]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[184]"}, tensor<128xf32> {jax.result_info = "result[185]"}, tensor<128xf32> {jax.result_info = "result[186]"}, tensor<128xf32> {jax.result_info = "result[187]"}, tensor<128xf32> {jax.result_info = "result[188]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[189]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[190]"}, tensor<128xf32> {jax.result_info = "result[191]"}, tensor<128xf32> {jax.result_info = "result[192]"}, tensor<128xf32> {jax.result_info = "result[193]"}, tensor<128xf32> {jax.result_info = "result[194]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[195]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[196]"}, tensor<256xf32> {jax.result_info = "result[197]"}, tensor<256xf32> {jax.result_info = "result[198]"}, tensor<256xf32> {jax.result_info = "result[199]"}, tensor<256xf32> {jax.result_info = "result[200]"}, tensor<3x3x128x256xf32> {jax.result_info = "result[201]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[202]"}, tensor<1x1x128x256xf32> {jax.result_info = "result[203]"}, tensor<256xf32> {jax.result_info = "result[204]"}, tensor<256xf32> {jax.result_info = "result[205]"}, tensor<256xf32> {jax.result_info = "result[206]"}, tensor<256xf32> {jax.result_info = "result[207]"}, tensor<256xf32> {jax.result_info = "result[208]"}, tensor<256xf32> {jax.result_info = "result[209]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[210]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[211]"}, tensor<256xf32> {jax.result_info = "result[212]"}, tensor<256xf32> {jax.result_info = "result[213]"}, tensor<256xf32> {jax.result_info = "result[214]"}, tensor<256xf32> {jax.result_info = "result[215]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[216]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[217]"}, tensor<64xf32> {jax.result_info = "result[218]"}, tensor<64xf32> {jax.result_info = "result[219]"}, tensor<7x7x3x64xf32> {jax.result_info = "result[220]"}, tensor<i32> {jax.result_info = "result[221]"}, tensor<64xf32> {jax.result_info = "result[222]"}, tensor<64xf32> {jax.result_info = "result[223]"}, tensor<64xf32> {jax.result_info = "result[224]"}, tensor<64xf32> {jax.result_info = "result[225]"}, tensor<64xf32> {jax.result_info = "result[226]"}, tensor<64xf32> {jax.result_info = "result[227]"}, tensor<64xf32> {jax.result_info = "result[228]"}, tensor<64xf32> {jax.result_info = "result[229]"}, tensor<256xf32> {jax.result_info = "result[230]"}, tensor<256xf32> {jax.result_info = "result[231]"}, tensor<256xf32> {jax.result_info = "result[232]"}, tensor<256xf32> {jax.result_info = "result[233]"}, tensor<256xf32> {jax.result_info = "result[234]"}, tensor<256xf32> {jax.result_info = "result[235]"}, tensor<256xf32> {jax.result_info = "result[236]"}, tensor<256xf32> {jax.result_info = "result[237]"}, tensor<256xf32> {jax.result_info = "result[238]"}, tensor<256xf32> {jax.result_info = "result[239]"}, tensor<256xf32> {jax.result_info = "result[240]"}, tensor<256xf32> {jax.result_info = "result[241]"}, tensor<512xf32> {jax.result_info = "result[242]"}, tensor<512xf32> {jax.result_info = "result[243]"}, tensor<512xf32> {jax.result_info = "result[244]"}, tensor<512xf32> {jax.result_info = "result[245]"}, tensor<512xf32> {jax.result_info = "result[246]"}, tensor<512xf32> {jax.result_info = "result[247]"}, tensor<512xf32> {jax.result_info = "result[248]"}, tensor<512xf32> {jax.result_info = "result[249]"}, tensor<512xf32> {jax.result_info = "result[250]"}, tensor<512xf32> {jax.result_info = "result[251]"}, tensor<512xf32> {jax.result_info = "result[252]"}, tensor<512xf32> {jax.result_info = "result[253]"}, tensor<512xf32> {jax.result_info = "result[254]"}, tensor<512xf32> {jax.result_info = "result[255]"}, tensor<64xf32> {jax.result_info = "result[256]"}, tensor<64xf32> {jax.result_info = "result[257]"}, tensor<64xf32> {jax.result_info = "result[258]"}, tensor<64xf32> {jax.result_info = "result[259]"}, tensor<128xf32> {jax.result_info = "result[260]"}, tensor<128xf32> {jax.result_info = "result[261]"}, tensor<128xf32> {jax.result_info = "result[262]"}, tensor<128xf32> {jax.result_info = "result[263]"}, tensor<128xf32> {jax.result_info = "result[264]"}, tensor<128xf32> {jax.result_info = "result[265]"}, tensor<128xf32> {jax.result_info = "result[266]"}, tensor<128xf32> {jax.result_info = "result[267]"}, tensor<128xf32> {jax.result_info = "result[268]"}, tensor<128xf32> {jax.result_info = "result[269]"}, tensor<128xf32> {jax.result_info = "result[270]"}, tensor<128xf32> {jax.result_info = "result[271]"}, tensor<128xf32> {jax.result_info = "result[272]"}, tensor<128xf32> {jax.result_info = "result[273]"}, tensor<128xf32> {jax.result_info = "result[274]"}, tensor<128xf32> {jax.result_info = "result[275]"}, tensor<128xf32> {jax.result_info = "result[276]"}, tensor<128xf32> {jax.result_info = "result[277]"}, tensor<256xf32> {jax.result_info = "result[278]"}, tensor<256xf32> {jax.result_info = "result[279]"}, tensor<256xf32> {jax.result_info = "result[280]"}, tensor<256xf32> {jax.result_info = "result[281]"}, tensor<256xf32> {jax.result_info = "result[282]"}, tensor<256xf32> {jax.result_info = "result[283]"}, tensor<256xf32> {jax.result_info = "result[284]"}, tensor<256xf32> {jax.result_info = "result[285]"}, tensor<256xf32> {jax.result_info = "result[286]"}, tensor<256xf32> {jax.result_info = "result[287]"}, tensor<256xf32> {jax.result_info = "result[288]"}, tensor<256xf32> {jax.result_info = "result[289]"}, tensor<256xf32> {jax.result_info = "result[290]"}, tensor<256xf32> {jax.result_info = "result[291]"}, tensor<64xf32> {jax.result_info = "result[292]"}, tensor<64xf32> {jax.result_info = "result[293]"}, tensor<i32> {jax.result_info = "result[294]"}, tensor<f32> {jax.result_info = "result[295]"}, tensor<4xf32> {jax.result_info = "result[296]"}, tensor<4xf32> {jax.result_info = "result[297]"}, tensor<4xf32> {jax.result_info = "result[298]"}, tensor<4xf32> {jax.result_info = "result[299]"}) {
    %0:300 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg181, %arg182, %arg183, %arg184, %arg185, %arg186, %arg187, %arg188, %arg189, %arg190, %arg191, %arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %arg198, %arg199, %arg200, %arg201, %arg202, %arg203, %arg204, %arg205, %arg206, %arg207, %arg208, %arg209, %arg210, %arg211, %arg212, %arg213, %arg214, %arg215, %arg216, %arg217, %arg218, %arg219, %arg220, %arg221, %arg222, %arg223, %arg224, %arg225, %arg226, %arg227, %arg228, %arg229, %arg230, %arg231, %arg232, %arg233, %arg234, %arg235, %arg236, %arg237, %arg238, %arg239, %arg240, %arg241, %arg242, %arg243, %arg244, %arg245, %arg246, %arg247, %arg248, %arg249, %arg250, %arg251, %arg252, %arg253, %arg254, %arg255, %arg256, %arg257, %arg258, %arg259, %arg260, %arg261, %arg262, %arg263, %arg264, %arg265, %arg266, %arg267, %arg268, %arg269, %arg270, %arg271, %arg272, %arg273, %arg274, %arg275, %arg276, %arg277, %arg278, %arg279, %arg280, %arg281, %arg282, %arg283, %arg284, %arg285, %arg286, %arg287, %arg288, %arg289, %arg290, %arg291, %arg292, %arg293, %arg294, %arg295, %arg296, %arg297) in_shardings=[<@mesh, []>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, []>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, []>, <@mesh, [{"batch"}, {}, {}, {}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, []>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, []>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, []>, <@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>] manual_axes={"batch"} (%arg298: tensor<i32>, %arg299: tensor<1000xf32>, %arg300: tensor<512x1000xf32>, %arg301: tensor<64xf32>, %arg302: tensor<64xf32>, %arg303: tensor<64xf32>, %arg304: tensor<64xf32>, %arg305: tensor<3x3x64x64xf32>, %arg306: tensor<3x3x64x64xf32>, %arg307: tensor<64xf32>, %arg308: tensor<64xf32>, %arg309: tensor<64xf32>, %arg310: tensor<64xf32>, %arg311: tensor<3x3x64x64xf32>, %arg312: tensor<3x3x64x64xf32>, %arg313: tensor<256xf32>, %arg314: tensor<256xf32>, %arg315: tensor<256xf32>, %arg316: tensor<256xf32>, %arg317: tensor<3x3x256x256xf32>, %arg318: tensor<3x3x256x256xf32>, %arg319: tensor<256xf32>, %arg320: tensor<256xf32>, %arg321: tensor<256xf32>, %arg322: tensor<256xf32>, %arg323: tensor<3x3x256x256xf32>, %arg324: tensor<3x3x256x256xf32>, %arg325: tensor<256xf32>, %arg326: tensor<256xf32>, %arg327: tensor<256xf32>, %arg328: tensor<256xf32>, %arg329: tensor<3x3x256x256xf32>, %arg330: tensor<3x3x256x256xf32>, %arg331: tensor<512xf32>, %arg332: tensor<512xf32>, %arg333: tensor<512xf32>, %arg334: tensor<512xf32>, %arg335: tensor<3x3x256x512xf32>, %arg336: tensor<3x3x512x512xf32>, %arg337: tensor<1x1x256x512xf32>, %arg338: tensor<512xf32>, %arg339: tensor<512xf32>, %arg340: tensor<512xf32>, %arg341: tensor<512xf32>, %arg342: tensor<512xf32>, %arg343: tensor<512xf32>, %arg344: tensor<3x3x512x512xf32>, %arg345: tensor<3x3x512x512xf32>, %arg346: tensor<512xf32>, %arg347: tensor<512xf32>, %arg348: tensor<512xf32>, %arg349: tensor<512xf32>, %arg350: tensor<3x3x512x512xf32>, %arg351: tensor<3x3x512x512xf32>, %arg352: tensor<64xf32>, %arg353: tensor<64xf32>, %arg354: tensor<64xf32>, %arg355: tensor<64xf32>, %arg356: tensor<3x3x64x64xf32>, %arg357: tensor<3x3x64x64xf32>, %arg358: tensor<128xf32>, %arg359: tensor<128xf32>, %arg360: tensor<128xf32>, %arg361: tensor<128xf32>, %arg362: tensor<3x3x64x128xf32>, %arg363: tensor<3x3x128x128xf32>, %arg364: tensor<1x1x64x128xf32>, %arg365: tensor<128xf32>, %arg366: tensor<128xf32>, %arg367: tensor<128xf32>, %arg368: tensor<128xf32>, %arg369: tensor<128xf32>, %arg370: tensor<128xf32>, %arg371: tensor<3x3x128x128xf32>, %arg372: tensor<3x3x128x128xf32>, %arg373: tensor<128xf32>, %arg374: tensor<128xf32>, %arg375: tensor<128xf32>, %arg376: tensor<128xf32>, %arg377: tensor<3x3x128x128xf32>, %arg378: tensor<3x3x128x128xf32>, %arg379: tensor<128xf32>, %arg380: tensor<128xf32>, %arg381: tensor<128xf32>, %arg382: tensor<128xf32>, %arg383: tensor<3x3x128x128xf32>, %arg384: tensor<3x3x128x128xf32>, %arg385: tensor<256xf32>, %arg386: tensor<256xf32>, %arg387: tensor<256xf32>, %arg388: tensor<256xf32>, %arg389: tensor<3x3x128x256xf32>, %arg390: tensor<3x3x256x256xf32>, %arg391: tensor<1x1x128x256xf32>, %arg392: tensor<256xf32>, %arg393: tensor<256xf32>, %arg394: tensor<256xf32>, %arg395: tensor<256xf32>, %arg396: tensor<256xf32>, %arg397: tensor<256xf32>, %arg398: tensor<3x3x256x256xf32>, %arg399: tensor<3x3x256x256xf32>, %arg400: tensor<256xf32>, %arg401: tensor<256xf32>, %arg402: tensor<256xf32>, %arg403: tensor<256xf32>, %arg404: tensor<3x3x256x256xf32>, %arg405: tensor<3x3x256x256xf32>, %arg406: tensor<64xf32>, %arg407: tensor<64xf32>, %arg408: tensor<7x7x3x64xf32>, %arg409: tensor<1000xf32>, %arg410: tensor<512x1000xf32>, %arg411: tensor<64xf32>, %arg412: tensor<64xf32>, %arg413: tensor<64xf32>, %arg414: tensor<64xf32>, %arg415: tensor<3x3x64x64xf32>, %arg416: tensor<3x3x64x64xf32>, %arg417: tensor<64xf32>, %arg418: tensor<64xf32>, %arg419: tensor<64xf32>, %arg420: tensor<64xf32>, %arg421: tensor<3x3x64x64xf32>, %arg422: tensor<3x3x64x64xf32>, %arg423: tensor<256xf32>, %arg424: tensor<256xf32>, %arg425: tensor<256xf32>, %arg426: tensor<256xf32>, %arg427: tensor<3x3x256x256xf32>, %arg428: tensor<3x3x256x256xf32>, %arg429: tensor<256xf32>, %arg430: tensor<256xf32>, %arg431: tensor<256xf32>, %arg432: tensor<256xf32>, %arg433: tensor<3x3x256x256xf32>, %arg434: tensor<3x3x256x256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<256xf32>, %arg438: tensor<256xf32>, %arg439: tensor<3x3x256x256xf32>, %arg440: tensor<3x3x256x256xf32>, %arg441: tensor<512xf32>, %arg442: tensor<512xf32>, %arg443: tensor<512xf32>, %arg444: tensor<512xf32>, %arg445: tensor<3x3x256x512xf32>, %arg446: tensor<3x3x512x512xf32>, %arg447: tensor<1x1x256x512xf32>, %arg448: tensor<512xf32>, %arg449: tensor<512xf32>, %arg450: tensor<512xf32>, %arg451: tensor<512xf32>, %arg452: tensor<512xf32>, %arg453: tensor<512xf32>, %arg454: tensor<3x3x512x512xf32>, %arg455: tensor<3x3x512x512xf32>, %arg456: tensor<512xf32>, %arg457: tensor<512xf32>, %arg458: tensor<512xf32>, %arg459: tensor<512xf32>, %arg460: tensor<3x3x512x512xf32>, %arg461: tensor<3x3x512x512xf32>, %arg462: tensor<64xf32>, %arg463: tensor<64xf32>, %arg464: tensor<64xf32>, %arg465: tensor<64xf32>, %arg466: tensor<3x3x64x64xf32>, %arg467: tensor<3x3x64x64xf32>, %arg468: tensor<128xf32>, %arg469: tensor<128xf32>, %arg470: tensor<128xf32>, %arg471: tensor<128xf32>, %arg472: tensor<3x3x64x128xf32>, %arg473: tensor<3x3x128x128xf32>, %arg474: tensor<1x1x64x128xf32>, %arg475: tensor<128xf32>, %arg476: tensor<128xf32>, %arg477: tensor<128xf32>, %arg478: tensor<128xf32>, %arg479: tensor<128xf32>, %arg480: tensor<128xf32>, %arg481: tensor<3x3x128x128xf32>, %arg482: tensor<3x3x128x128xf32>, %arg483: tensor<128xf32>, %arg484: tensor<128xf32>, %arg485: tensor<128xf32>, %arg486: tensor<128xf32>, %arg487: tensor<3x3x128x128xf32>, %arg488: tensor<3x3x128x128xf32>, %arg489: tensor<128xf32>, %arg490: tensor<128xf32>, %arg491: tensor<128xf32>, %arg492: tensor<128xf32>, %arg493: tensor<3x3x128x128xf32>, %arg494: tensor<3x3x128x128xf32>, %arg495: tensor<256xf32>, %arg496: tensor<256xf32>, %arg497: tensor<256xf32>, %arg498: tensor<256xf32>, %arg499: tensor<3x3x128x256xf32>, %arg500: tensor<3x3x256x256xf32>, %arg501: tensor<1x1x128x256xf32>, %arg502: tensor<256xf32>, %arg503: tensor<256xf32>, %arg504: tensor<256xf32>, %arg505: tensor<256xf32>, %arg506: tensor<256xf32>, %arg507: tensor<256xf32>, %arg508: tensor<3x3x256x256xf32>, %arg509: tensor<3x3x256x256xf32>, %arg510: tensor<256xf32>, %arg511: tensor<256xf32>, %arg512: tensor<256xf32>, %arg513: tensor<256xf32>, %arg514: tensor<3x3x256x256xf32>, %arg515: tensor<3x3x256x256xf32>, %arg516: tensor<64xf32>, %arg517: tensor<64xf32>, %arg518: tensor<7x7x3x64xf32>, %arg519: tensor<i32>, %arg520: tensor<64xf32>, %arg521: tensor<64xf32>, %arg522: tensor<64xf32>, %arg523: tensor<64xf32>, %arg524: tensor<64xf32>, %arg525: tensor<64xf32>, %arg526: tensor<64xf32>, %arg527: tensor<64xf32>, %arg528: tensor<256xf32>, %arg529: tensor<256xf32>, %arg530: tensor<256xf32>, %arg531: tensor<256xf32>, %arg532: tensor<256xf32>, %arg533: tensor<256xf32>, %arg534: tensor<256xf32>, %arg535: tensor<256xf32>, %arg536: tensor<256xf32>, %arg537: tensor<256xf32>, %arg538: tensor<256xf32>, %arg539: tensor<256xf32>, %arg540: tensor<512xf32>, %arg541: tensor<512xf32>, %arg542: tensor<512xf32>, %arg543: tensor<512xf32>, %arg544: tensor<512xf32>, %arg545: tensor<512xf32>, %arg546: tensor<512xf32>, %arg547: tensor<512xf32>, %arg548: tensor<512xf32>, %arg549: tensor<512xf32>, %arg550: tensor<512xf32>, %arg551: tensor<512xf32>, %arg552: tensor<512xf32>, %arg553: tensor<512xf32>, %arg554: tensor<64xf32>, %arg555: tensor<64xf32>, %arg556: tensor<64xf32>, %arg557: tensor<64xf32>, %arg558: tensor<128xf32>, %arg559: tensor<128xf32>, %arg560: tensor<128xf32>, %arg561: tensor<128xf32>, %arg562: tensor<128xf32>, %arg563: tensor<128xf32>, %arg564: tensor<128xf32>, %arg565: tensor<128xf32>, %arg566: tensor<128xf32>, %arg567: tensor<128xf32>, %arg568: tensor<128xf32>, %arg569: tensor<128xf32>, %arg570: tensor<128xf32>, %arg571: tensor<128xf32>, %arg572: tensor<128xf32>, %arg573: tensor<128xf32>, %arg574: tensor<128xf32>, %arg575: tensor<128xf32>, %arg576: tensor<256xf32>, %arg577: tensor<256xf32>, %arg578: tensor<256xf32>, %arg579: tensor<256xf32>, %arg580: tensor<256xf32>, %arg581: tensor<256xf32>, %arg582: tensor<256xf32>, %arg583: tensor<256xf32>, %arg584: tensor<256xf32>, %arg585: tensor<256xf32>, %arg586: tensor<256xf32>, %arg587: tensor<256xf32>, %arg588: tensor<256xf32>, %arg589: tensor<256xf32>, %arg590: tensor<64xf32>, %arg591: tensor<64xf32>, %arg592: tensor<i32>, %arg593: tensor<f32>, %arg594: tensor<1x256x224x224x3xf16>, %arg595: tensor<1x256xi32>) {
      %1 = stablehlo.reshape %arg594 : (tensor<1x256x224x224x3xf16>) -> tensor<256x224x224x3xf16>
      %2 = stablehlo.reshape %arg595 : (tensor<1x256xi32>) -> tensor<256xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %3 = stablehlo.subtract %arg298, %c : tensor<i32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %cst = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %4 = func.call @clip(%3, %c_0, %cst) : (tensor<i32>, tensor<i32>, tensor<f32>) -> tensor<f32>
      %cst_1 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %5 = stablehlo.divide %4, %cst_1 : tensor<f32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %6 = stablehlo.subtract %cst_2, %5 : tensor<f32>
      %cst_3 = stablehlo.constant dense<-4.000000e-01> : tensor<f32>
      %7 = stablehlo.multiply %cst_3, %6 : tensor<f32>
      %cst_4 = stablehlo.constant dense<4.000000e-01> : tensor<f32>
      %8 = stablehlo.add %7, %cst_4 : tensor<f32>
      %9 = stablehlo.convert %arg298 : (tensor<i32>) -> tensor<f32>
      %cst_5 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %10 = stablehlo.compare  LT, %9, %cst_5,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %11 = stablehlo.convert %arg298 : (tensor<i32>) -> tensor<f32>
      %cst_6 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %12 = stablehlo.subtract %11, %cst_6 : tensor<f32>
      %cst_7 = stablehlo.constant dense<1.188450e+05> : tensor<f32>
      %13 = stablehlo.minimum %12, %cst_7 : tensor<f32>
      %cst_8 = stablehlo.constant dense<3.14159274> : tensor<f32>
      %14 = stablehlo.multiply %cst_8, %13 : tensor<f32>
      %cst_9 = stablehlo.constant dense<1.188450e+05> : tensor<f32>
      %15 = stablehlo.divide %14, %cst_9 : tensor<f32>
      %16 = stablehlo.cosine %15 : tensor<f32>
      %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %17 = stablehlo.add %cst_10, %16 : tensor<f32>
      %cst_11 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %18 = stablehlo.multiply %cst_11, %17 : tensor<f32>
      %cst_12 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %19 = stablehlo.power %18, %cst_12 : tensor<f32>
      %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %20 = stablehlo.multiply %cst_13, %19 : tensor<f32>
      %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %21 = stablehlo.add %20, %cst_14 : tensor<f32>
      %cst_15 = stablehlo.constant dense<4.000000e-01> : tensor<f32>
      %22 = stablehlo.multiply %cst_15, %21 : tensor<f32>
      %23 = func.call @_where(%10, %8, %22) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %24 = stablehlo.convert %arg408 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf16>
      %25 = stablehlo.convolution(%1, %24) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf16>, tensor<7x7x3x64xf16>) -> tensor<256x112x112x64xf16>
      %26 = stablehlo.convert %25 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
      %27 = stablehlo.multiply %26, %26 : tensor<256x112x112x64xf32>
      %cst_16 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %28 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<256x112x112x64xf32>
      %29 = stablehlo.multiply %28, %26 : tensor<256x112x112x64xf32>
      %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %30 = stablehlo.reduce(%26 init: %cst_17) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_18 = stablehlo.constant dense<0x4A440000> : tensor<f32>
      %31 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %32 = stablehlo.divide %30, %31 : tensor<64xf32>
      %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %33 = stablehlo.reduce(%27 init: %cst_19) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_20 = stablehlo.constant dense<0x4A440000> : tensor<f32>
      %34 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %35 = stablehlo.divide %33, %34 : tensor<64xf32>
      %36 = stablehlo.broadcast_in_dim %32, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %37 = stablehlo.broadcast_in_dim %35, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %38 = stablehlo.concatenate %36, %37, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
      %39 = "stablehlo.all_reduce"(%38) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %cst_21 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %40 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %41 = stablehlo.divide %39, %40 : tensor<2x64xf32>
      %42 = stablehlo.slice %41 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %43 = stablehlo.reshape %42 : (tensor<1x64xf32>) -> tensor<64xf32>
      %44 = stablehlo.slice %41 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %45 = stablehlo.reshape %44 : (tensor<1x64xf32>) -> tensor<64xf32>
      %46 = stablehlo.multiply %43, %43 : tensor<64xf32>
      %cst_22 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %47 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %48 = stablehlo.multiply %47, %43 : tensor<64xf32>
      %49 = stablehlo.subtract %45, %46 : tensor<64xf32>
      %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %50 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %51 = stablehlo.maximum %50, %49 : tensor<64xf32>
      %52 = stablehlo.compare  EQ, %49, %51,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %53 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %54 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %55 = stablehlo.select %52, %53, %54 : tensor<64xi1>, tensor<64xf32>
      %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %56 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %57 = stablehlo.compare  EQ, %56, %51,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_27 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %58 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_28 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %59 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %60 = stablehlo.select %57, %58, %59 : tensor<64xi1>, tensor<64xf32>
      %61 = stablehlo.divide %55, %60 : tensor<64xf32>
      %cst_29 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %62 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %63 = stablehlo.multiply %62, %arg590 : tensor<64xf32>
      %cst_30 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %64 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %65 = stablehlo.multiply %64, %43 : tensor<64xf32>
      %66 = stablehlo.add %63, %65 : tensor<64xf32>
      %cst_31 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %67 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %68 = stablehlo.multiply %67, %arg591 : tensor<64xf32>
      %cst_32 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %69 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %70 = stablehlo.multiply %69, %51 : tensor<64xf32>
      %71 = stablehlo.add %68, %70 : tensor<64xf32>
      %72 = stablehlo.broadcast_in_dim %43, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %73 = stablehlo.broadcast_in_dim %51, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %74 = stablehlo.convert %25 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
      %75 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %76 = stablehlo.subtract %74, %75 : tensor<256x112x112x64xf32>
      %cst_33 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %77 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %78 = stablehlo.add %73, %77 : tensor<1x1x1x64xf32>
      %79 = stablehlo.rsqrt %78 : tensor<1x1x1x64xf32>
      %80 = stablehlo.divide %79, %78 : tensor<1x1x1x64xf32>
      %cst_34 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %81 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %82 = stablehlo.multiply %81, %80 : tensor<1x1x1x64xf32>
      %83 = stablehlo.reshape %arg407 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %84 = stablehlo.multiply %79, %83 : tensor<1x1x1x64xf32>
      %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %86 = stablehlo.multiply %76, %85 : tensor<256x112x112x64xf32>
      %87 = stablehlo.reshape %arg406 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %89 = stablehlo.add %86, %88 : tensor<256x112x112x64xf32>
      %90 = stablehlo.convert %89 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %91 = func.call @relu(%90) : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16>
      %cst_35 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %92 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f16>) -> tensor<256x112x112x64xf16>
      %93 = stablehlo.compare  GT, %90, %92,  FLOAT : (tensor<256x112x112x64xf16>, tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xi1>
      %cst_36 = stablehlo.constant dense<0xFC00> : tensor<f16>
      %94 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f16>) -> tensor<f16>
      %95 = "stablehlo.reduce_window"(%91, %94) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg596: tensor<f16>, %arg597: tensor<f16>):
        %7234 = stablehlo.maximum %arg596, %arg597 : tensor<f16>
        stablehlo.return %7234 : tensor<f16>
      }) : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x56x56x64xf16>
      %96 = stablehlo.convert %arg305 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %97 = stablehlo.convolution(%95, %96) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %98 = stablehlo.convert %97 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %99 = stablehlo.multiply %98, %98 : tensor<256x56x56x64xf32>
      %cst_37 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %100 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
      %101 = stablehlo.multiply %100, %98 : tensor<256x56x56x64xf32>
      %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %102 = stablehlo.reduce(%98 init: %cst_38) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_39 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %103 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %104 = stablehlo.divide %102, %103 : tensor<64xf32>
      %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %105 = stablehlo.reduce(%99 init: %cst_40) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_41 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %106 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %107 = stablehlo.divide %105, %106 : tensor<64xf32>
      %108 = stablehlo.broadcast_in_dim %104, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %109 = stablehlo.broadcast_in_dim %107, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %110 = stablehlo.concatenate %108, %109, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
      %111 = "stablehlo.all_reduce"(%110) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %cst_42 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %112 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %113 = stablehlo.divide %111, %112 : tensor<2x64xf32>
      %114 = stablehlo.slice %113 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %115 = stablehlo.reshape %114 : (tensor<1x64xf32>) -> tensor<64xf32>
      %116 = stablehlo.slice %113 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %117 = stablehlo.reshape %116 : (tensor<1x64xf32>) -> tensor<64xf32>
      %118 = stablehlo.multiply %115, %115 : tensor<64xf32>
      %cst_43 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %119 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %120 = stablehlo.multiply %119, %115 : tensor<64xf32>
      %121 = stablehlo.subtract %117, %118 : tensor<64xf32>
      %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %122 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %123 = stablehlo.maximum %122, %121 : tensor<64xf32>
      %124 = stablehlo.compare  EQ, %121, %123,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_45 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %125 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %126 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %127 = stablehlo.select %124, %125, %126 : tensor<64xi1>, tensor<64xf32>
      %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %128 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %129 = stablehlo.compare  EQ, %128, %123,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_48 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %130 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_49 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %131 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %132 = stablehlo.select %129, %130, %131 : tensor<64xi1>, tensor<64xf32>
      %133 = stablehlo.divide %127, %132 : tensor<64xf32>
      %cst_50 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %134 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %135 = stablehlo.multiply %134, %arg520 : tensor<64xf32>
      %cst_51 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %136 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %137 = stablehlo.multiply %136, %115 : tensor<64xf32>
      %138 = stablehlo.add %135, %137 : tensor<64xf32>
      %cst_52 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %139 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %140 = stablehlo.multiply %139, %arg521 : tensor<64xf32>
      %cst_53 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %141 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %142 = stablehlo.multiply %141, %123 : tensor<64xf32>
      %143 = stablehlo.add %140, %142 : tensor<64xf32>
      %144 = stablehlo.broadcast_in_dim %115, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %145 = stablehlo.broadcast_in_dim %123, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %146 = stablehlo.convert %97 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %147 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %148 = stablehlo.subtract %146, %147 : tensor<256x56x56x64xf32>
      %cst_54 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %149 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %150 = stablehlo.add %145, %149 : tensor<1x1x1x64xf32>
      %151 = stablehlo.rsqrt %150 : tensor<1x1x1x64xf32>
      %152 = stablehlo.divide %151, %150 : tensor<1x1x1x64xf32>
      %cst_55 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %153 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %154 = stablehlo.multiply %153, %152 : tensor<1x1x1x64xf32>
      %155 = stablehlo.reshape %arg302 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %156 = stablehlo.multiply %151, %155 : tensor<1x1x1x64xf32>
      %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %158 = stablehlo.multiply %148, %157 : tensor<256x56x56x64xf32>
      %159 = stablehlo.reshape %arg301 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %160 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %161 = stablehlo.add %158, %160 : tensor<256x56x56x64xf32>
      %162 = stablehlo.convert %161 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %163 = func.call @relu_44(%162) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %164 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %165 = stablehlo.compare  GT, %162, %164,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %166 = stablehlo.convert %arg306 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %167 = stablehlo.convolution(%163, %166) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %168 = stablehlo.convert %167 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %169 = stablehlo.multiply %168, %168 : tensor<256x56x56x64xf32>
      %cst_57 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %170 = stablehlo.broadcast_in_dim %cst_57, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
      %171 = stablehlo.multiply %170, %168 : tensor<256x56x56x64xf32>
      %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %172 = stablehlo.reduce(%168 init: %cst_58) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_59 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %174 = stablehlo.divide %172, %173 : tensor<64xf32>
      %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %175 = stablehlo.reduce(%169 init: %cst_60) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_61 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %176 = stablehlo.broadcast_in_dim %cst_61, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %177 = stablehlo.divide %175, %176 : tensor<64xf32>
      %178 = stablehlo.broadcast_in_dim %174, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %179 = stablehlo.broadcast_in_dim %177, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %180 = stablehlo.concatenate %178, %179, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
      %181 = "stablehlo.all_reduce"(%180) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %cst_62 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %182 = stablehlo.broadcast_in_dim %cst_62, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %183 = stablehlo.divide %181, %182 : tensor<2x64xf32>
      %184 = stablehlo.slice %183 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %185 = stablehlo.reshape %184 : (tensor<1x64xf32>) -> tensor<64xf32>
      %186 = stablehlo.slice %183 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %187 = stablehlo.reshape %186 : (tensor<1x64xf32>) -> tensor<64xf32>
      %188 = stablehlo.multiply %185, %185 : tensor<64xf32>
      %cst_63 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %189 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %190 = stablehlo.multiply %189, %185 : tensor<64xf32>
      %191 = stablehlo.subtract %187, %188 : tensor<64xf32>
      %cst_64 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %192 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %193 = stablehlo.maximum %192, %191 : tensor<64xf32>
      %194 = stablehlo.compare  EQ, %191, %193,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_65 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %195 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %196 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %197 = stablehlo.select %194, %195, %196 : tensor<64xi1>, tensor<64xf32>
      %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %198 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %199 = stablehlo.compare  EQ, %198, %193,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_68 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %200 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_69 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %201 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %202 = stablehlo.select %199, %200, %201 : tensor<64xi1>, tensor<64xf32>
      %203 = stablehlo.divide %197, %202 : tensor<64xf32>
      %cst_70 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %204 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %205 = stablehlo.multiply %204, %arg522 : tensor<64xf32>
      %cst_71 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %206 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %207 = stablehlo.multiply %206, %185 : tensor<64xf32>
      %208 = stablehlo.add %205, %207 : tensor<64xf32>
      %cst_72 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %209 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %210 = stablehlo.multiply %209, %arg523 : tensor<64xf32>
      %cst_73 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %211 = stablehlo.broadcast_in_dim %cst_73, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %212 = stablehlo.multiply %211, %193 : tensor<64xf32>
      %213 = stablehlo.add %210, %212 : tensor<64xf32>
      %214 = stablehlo.broadcast_in_dim %185, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %215 = stablehlo.broadcast_in_dim %193, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %216 = stablehlo.convert %167 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %217 = stablehlo.broadcast_in_dim %214, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %218 = stablehlo.subtract %216, %217 : tensor<256x56x56x64xf32>
      %cst_74 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %219 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %220 = stablehlo.add %215, %219 : tensor<1x1x1x64xf32>
      %221 = stablehlo.rsqrt %220 : tensor<1x1x1x64xf32>
      %222 = stablehlo.divide %221, %220 : tensor<1x1x1x64xf32>
      %cst_75 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %223 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %224 = stablehlo.multiply %223, %222 : tensor<1x1x1x64xf32>
      %225 = stablehlo.reshape %arg304 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %226 = stablehlo.multiply %221, %225 : tensor<1x1x1x64xf32>
      %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %228 = stablehlo.multiply %218, %227 : tensor<256x56x56x64xf32>
      %229 = stablehlo.reshape %arg303 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %231 = stablehlo.add %228, %230 : tensor<256x56x56x64xf32>
      %232 = stablehlo.convert %231 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %233 = stablehlo.add %95, %232 : tensor<256x56x56x64xf16>
      %234 = func.call @relu_44(%233) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %235 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %236 = stablehlo.compare  GT, %233, %235,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %237 = stablehlo.convert %arg311 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %238 = stablehlo.convolution(%234, %237) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %239 = stablehlo.convert %238 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %240 = stablehlo.multiply %239, %239 : tensor<256x56x56x64xf32>
      %cst_77 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %241 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
      %242 = stablehlo.multiply %241, %239 : tensor<256x56x56x64xf32>
      %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %243 = stablehlo.reduce(%239 init: %cst_78) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_79 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %244 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %245 = stablehlo.divide %243, %244 : tensor<64xf32>
      %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %246 = stablehlo.reduce(%240 init: %cst_80) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_81 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %247 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %248 = stablehlo.divide %246, %247 : tensor<64xf32>
      %249 = stablehlo.broadcast_in_dim %245, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %250 = stablehlo.broadcast_in_dim %248, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %251 = stablehlo.concatenate %249, %250, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
      %252 = "stablehlo.all_reduce"(%251) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %cst_82 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %253 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %254 = stablehlo.divide %252, %253 : tensor<2x64xf32>
      %255 = stablehlo.slice %254 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %256 = stablehlo.reshape %255 : (tensor<1x64xf32>) -> tensor<64xf32>
      %257 = stablehlo.slice %254 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %258 = stablehlo.reshape %257 : (tensor<1x64xf32>) -> tensor<64xf32>
      %259 = stablehlo.multiply %256, %256 : tensor<64xf32>
      %cst_83 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %260 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %261 = stablehlo.multiply %260, %256 : tensor<64xf32>
      %262 = stablehlo.subtract %258, %259 : tensor<64xf32>
      %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %264 = stablehlo.maximum %263, %262 : tensor<64xf32>
      %265 = stablehlo.compare  EQ, %262, %264,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_85 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %266 = stablehlo.broadcast_in_dim %cst_85, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %267 = stablehlo.broadcast_in_dim %cst_86, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %268 = stablehlo.select %265, %266, %267 : tensor<64xi1>, tensor<64xf32>
      %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %269 = stablehlo.broadcast_in_dim %cst_87, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %270 = stablehlo.compare  EQ, %269, %264,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_88 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %271 = stablehlo.broadcast_in_dim %cst_88, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_89 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %272 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %273 = stablehlo.select %270, %271, %272 : tensor<64xi1>, tensor<64xf32>
      %274 = stablehlo.divide %268, %273 : tensor<64xf32>
      %cst_90 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %275 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %276 = stablehlo.multiply %275, %arg524 : tensor<64xf32>
      %cst_91 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %278 = stablehlo.multiply %277, %256 : tensor<64xf32>
      %279 = stablehlo.add %276, %278 : tensor<64xf32>
      %cst_92 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %280 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %281 = stablehlo.multiply %280, %arg525 : tensor<64xf32>
      %cst_93 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %282 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %283 = stablehlo.multiply %282, %264 : tensor<64xf32>
      %284 = stablehlo.add %281, %283 : tensor<64xf32>
      %285 = stablehlo.broadcast_in_dim %256, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %286 = stablehlo.broadcast_in_dim %264, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %287 = stablehlo.convert %238 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %288 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %289 = stablehlo.subtract %287, %288 : tensor<256x56x56x64xf32>
      %cst_94 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %291 = stablehlo.add %286, %290 : tensor<1x1x1x64xf32>
      %292 = stablehlo.rsqrt %291 : tensor<1x1x1x64xf32>
      %293 = stablehlo.divide %292, %291 : tensor<1x1x1x64xf32>
      %cst_95 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %294 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %295 = stablehlo.multiply %294, %293 : tensor<1x1x1x64xf32>
      %296 = stablehlo.reshape %arg308 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %297 = stablehlo.multiply %292, %296 : tensor<1x1x1x64xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %299 = stablehlo.multiply %289, %298 : tensor<256x56x56x64xf32>
      %300 = stablehlo.reshape %arg307 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %302 = stablehlo.add %299, %301 : tensor<256x56x56x64xf32>
      %303 = stablehlo.convert %302 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %304 = func.call @relu_44(%303) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %305 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %306 = stablehlo.compare  GT, %303, %305,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %307 = stablehlo.convert %arg312 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %308 = stablehlo.convolution(%304, %307) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %309 = stablehlo.convert %308 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %310 = stablehlo.multiply %309, %309 : tensor<256x56x56x64xf32>
      %cst_97 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
      %312 = stablehlo.multiply %311, %309 : tensor<256x56x56x64xf32>
      %cst_98 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%309 init: %cst_98) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_99 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_99, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %315 = stablehlo.divide %313, %314 : tensor<64xf32>
      %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %316 = stablehlo.reduce(%310 init: %cst_100) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_101 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %318 = stablehlo.divide %316, %317 : tensor<64xf32>
      %319 = stablehlo.broadcast_in_dim %315, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %320 = stablehlo.broadcast_in_dim %318, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %321 = stablehlo.concatenate %319, %320, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
      %322 = "stablehlo.all_reduce"(%321) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %cst_102 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %323 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %324 = stablehlo.divide %322, %323 : tensor<2x64xf32>
      %325 = stablehlo.slice %324 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %326 = stablehlo.reshape %325 : (tensor<1x64xf32>) -> tensor<64xf32>
      %327 = stablehlo.slice %324 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %328 = stablehlo.reshape %327 : (tensor<1x64xf32>) -> tensor<64xf32>
      %329 = stablehlo.multiply %326, %326 : tensor<64xf32>
      %cst_103 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %331 = stablehlo.multiply %330, %326 : tensor<64xf32>
      %332 = stablehlo.subtract %328, %329 : tensor<64xf32>
      %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %333 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %334 = stablehlo.maximum %333, %332 : tensor<64xf32>
      %335 = stablehlo.compare  EQ, %332, %334,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_105 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %338 = stablehlo.select %335, %336, %337 : tensor<64xi1>, tensor<64xf32>
      %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %340 = stablehlo.compare  EQ, %339, %334,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_108 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_109 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_109, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %343 = stablehlo.select %340, %341, %342 : tensor<64xi1>, tensor<64xf32>
      %344 = stablehlo.divide %338, %343 : tensor<64xf32>
      %cst_110 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %346 = stablehlo.multiply %345, %arg526 : tensor<64xf32>
      %cst_111 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %348 = stablehlo.multiply %347, %326 : tensor<64xf32>
      %349 = stablehlo.add %346, %348 : tensor<64xf32>
      %cst_112 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %351 = stablehlo.multiply %350, %arg527 : tensor<64xf32>
      %cst_113 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %352 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %353 = stablehlo.multiply %352, %334 : tensor<64xf32>
      %354 = stablehlo.add %351, %353 : tensor<64xf32>
      %355 = stablehlo.broadcast_in_dim %326, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %356 = stablehlo.broadcast_in_dim %334, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %357 = stablehlo.convert %308 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %358 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %359 = stablehlo.subtract %357, %358 : tensor<256x56x56x64xf32>
      %cst_114 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %361 = stablehlo.add %356, %360 : tensor<1x1x1x64xf32>
      %362 = stablehlo.rsqrt %361 : tensor<1x1x1x64xf32>
      %363 = stablehlo.divide %362, %361 : tensor<1x1x1x64xf32>
      %cst_115 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_115, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %365 = stablehlo.multiply %364, %363 : tensor<1x1x1x64xf32>
      %366 = stablehlo.reshape %arg310 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %367 = stablehlo.multiply %362, %366 : tensor<1x1x1x64xf32>
      %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %369 = stablehlo.multiply %359, %368 : tensor<256x56x56x64xf32>
      %370 = stablehlo.reshape %arg309 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %371 = stablehlo.broadcast_in_dim %370, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %372 = stablehlo.add %369, %371 : tensor<256x56x56x64xf32>
      %373 = stablehlo.convert %372 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %374 = stablehlo.add %234, %373 : tensor<256x56x56x64xf16>
      %375 = func.call @relu_44(%374) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %376 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %377 = stablehlo.compare  GT, %374, %376,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %378 = stablehlo.convert %arg356 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %379 = stablehlo.convolution(%375, %378) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %380 = stablehlo.convert %379 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %381 = stablehlo.multiply %380, %380 : tensor<256x56x56x64xf32>
      %cst_117 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_117, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
      %383 = stablehlo.multiply %382, %380 : tensor<256x56x56x64xf32>
      %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%380 init: %cst_118) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_119 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %386 = stablehlo.divide %384, %385 : tensor<64xf32>
      %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%381 init: %cst_120) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_121 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_121, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %389 = stablehlo.divide %387, %388 : tensor<64xf32>
      %390 = stablehlo.broadcast_in_dim %386, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %391 = stablehlo.broadcast_in_dim %389, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %392 = stablehlo.concatenate %390, %391, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
      %393 = "stablehlo.all_reduce"(%392) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %cst_122 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %395 = stablehlo.divide %393, %394 : tensor<2x64xf32>
      %396 = stablehlo.slice %395 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %397 = stablehlo.reshape %396 : (tensor<1x64xf32>) -> tensor<64xf32>
      %398 = stablehlo.slice %395 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %399 = stablehlo.reshape %398 : (tensor<1x64xf32>) -> tensor<64xf32>
      %400 = stablehlo.multiply %397, %397 : tensor<64xf32>
      %cst_123 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %402 = stablehlo.multiply %401, %397 : tensor<64xf32>
      %403 = stablehlo.subtract %399, %400 : tensor<64xf32>
      %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_124, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %405 = stablehlo.maximum %404, %403 : tensor<64xf32>
      %406 = stablehlo.compare  EQ, %403, %405,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_125 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %409 = stablehlo.select %406, %407, %408 : tensor<64xi1>, tensor<64xf32>
      %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_127, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %411 = stablehlo.compare  EQ, %410, %405,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_128 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_129 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %414 = stablehlo.select %411, %412, %413 : tensor<64xi1>, tensor<64xf32>
      %415 = stablehlo.divide %409, %414 : tensor<64xf32>
      %cst_130 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %417 = stablehlo.multiply %416, %arg554 : tensor<64xf32>
      %cst_131 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %419 = stablehlo.multiply %418, %397 : tensor<64xf32>
      %420 = stablehlo.add %417, %419 : tensor<64xf32>
      %cst_132 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %422 = stablehlo.multiply %421, %arg555 : tensor<64xf32>
      %cst_133 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_133, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %424 = stablehlo.multiply %423, %405 : tensor<64xf32>
      %425 = stablehlo.add %422, %424 : tensor<64xf32>
      %426 = stablehlo.broadcast_in_dim %397, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %427 = stablehlo.broadcast_in_dim %405, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %428 = stablehlo.convert %379 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %429 = stablehlo.broadcast_in_dim %426, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %430 = stablehlo.subtract %428, %429 : tensor<256x56x56x64xf32>
      %cst_134 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_134, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %432 = stablehlo.add %427, %431 : tensor<1x1x1x64xf32>
      %433 = stablehlo.rsqrt %432 : tensor<1x1x1x64xf32>
      %434 = stablehlo.divide %433, %432 : tensor<1x1x1x64xf32>
      %cst_135 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %436 = stablehlo.multiply %435, %434 : tensor<1x1x1x64xf32>
      %437 = stablehlo.reshape %arg353 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %438 = stablehlo.multiply %433, %437 : tensor<1x1x1x64xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %440 = stablehlo.multiply %430, %439 : tensor<256x56x56x64xf32>
      %441 = stablehlo.reshape %arg352 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %443 = stablehlo.add %440, %442 : tensor<256x56x56x64xf32>
      %444 = stablehlo.convert %443 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %445 = func.call @relu_44(%444) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %446 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %447 = stablehlo.compare  GT, %444, %446,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %448 = stablehlo.convert %arg357 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %449 = stablehlo.convolution(%445, %448) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %450 = stablehlo.convert %449 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %451 = stablehlo.multiply %450, %450 : tensor<256x56x56x64xf32>
      %cst_137 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_137, dims = [] : (tensor<f32>) -> tensor<256x56x56x64xf32>
      %453 = stablehlo.multiply %452, %450 : tensor<256x56x56x64xf32>
      %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.reduce(%450 init: %cst_138) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_139 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %456 = stablehlo.divide %454, %455 : tensor<64xf32>
      %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %457 = stablehlo.reduce(%451 init: %cst_140) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_141 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %459 = stablehlo.divide %457, %458 : tensor<64xf32>
      %460 = stablehlo.broadcast_in_dim %456, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %461 = stablehlo.broadcast_in_dim %459, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %462 = stablehlo.concatenate %460, %461, dim = 0 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<2x64xf32>
      %463 = "stablehlo.all_reduce"(%462) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %cst_142 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %465 = stablehlo.divide %463, %464 : tensor<2x64xf32>
      %466 = stablehlo.slice %465 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %467 = stablehlo.reshape %466 : (tensor<1x64xf32>) -> tensor<64xf32>
      %468 = stablehlo.slice %465 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %469 = stablehlo.reshape %468 : (tensor<1x64xf32>) -> tensor<64xf32>
      %470 = stablehlo.multiply %467, %467 : tensor<64xf32>
      %cst_143 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %472 = stablehlo.multiply %471, %467 : tensor<64xf32>
      %473 = stablehlo.subtract %469, %470 : tensor<64xf32>
      %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %475 = stablehlo.maximum %474, %473 : tensor<64xf32>
      %476 = stablehlo.compare  EQ, %473, %475,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_145 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %479 = stablehlo.select %476, %477, %478 : tensor<64xi1>, tensor<64xf32>
      %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %481 = stablehlo.compare  EQ, %480, %475,  FLOAT : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
      %cst_148 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %cst_149 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %484 = stablehlo.select %481, %482, %483 : tensor<64xi1>, tensor<64xf32>
      %485 = stablehlo.divide %479, %484 : tensor<64xf32>
      %cst_150 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %487 = stablehlo.multiply %486, %arg556 : tensor<64xf32>
      %cst_151 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %489 = stablehlo.multiply %488, %467 : tensor<64xf32>
      %490 = stablehlo.add %487, %489 : tensor<64xf32>
      %cst_152 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %492 = stablehlo.multiply %491, %arg557 : tensor<64xf32>
      %cst_153 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %494 = stablehlo.multiply %493, %475 : tensor<64xf32>
      %495 = stablehlo.add %492, %494 : tensor<64xf32>
      %496 = stablehlo.broadcast_in_dim %467, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %497 = stablehlo.broadcast_in_dim %475, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %498 = stablehlo.convert %449 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %499 = stablehlo.broadcast_in_dim %496, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %500 = stablehlo.subtract %498, %499 : tensor<256x56x56x64xf32>
      %cst_154 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %502 = stablehlo.add %497, %501 : tensor<1x1x1x64xf32>
      %503 = stablehlo.rsqrt %502 : tensor<1x1x1x64xf32>
      %504 = stablehlo.divide %503, %502 : tensor<1x1x1x64xf32>
      %cst_155 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_155, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %506 = stablehlo.multiply %505, %504 : tensor<1x1x1x64xf32>
      %507 = stablehlo.reshape %arg355 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %508 = stablehlo.multiply %503, %507 : tensor<1x1x1x64xf32>
      %509 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %510 = stablehlo.multiply %500, %509 : tensor<256x56x56x64xf32>
      %511 = stablehlo.reshape %arg354 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %512 = stablehlo.broadcast_in_dim %511, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %513 = stablehlo.add %510, %512 : tensor<256x56x56x64xf32>
      %514 = stablehlo.convert %513 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %515 = stablehlo.add %375, %514 : tensor<256x56x56x64xf16>
      %516 = func.call @relu_44(%515) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %517 = stablehlo.broadcast_in_dim %cst_156, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %518 = stablehlo.compare  GT, %515, %517,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %519 = stablehlo.convert %arg362 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf16>
      %520 = stablehlo.convolution(%516, %519) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x128xf16>) -> tensor<256x28x28x128xf16>
      %521 = stablehlo.convert %520 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %522 = stablehlo.multiply %521, %521 : tensor<256x28x28x128xf32>
      %cst_157 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %523 = stablehlo.broadcast_in_dim %cst_157, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %524 = stablehlo.multiply %523, %521 : tensor<256x28x28x128xf32>
      %cst_158 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %525 = stablehlo.reduce(%521 init: %cst_158) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_159 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %526 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %527 = stablehlo.divide %525, %526 : tensor<128xf32>
      %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %528 = stablehlo.reduce(%522 init: %cst_160) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_161 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %530 = stablehlo.divide %528, %529 : tensor<128xf32>
      %531 = stablehlo.broadcast_in_dim %527, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %532 = stablehlo.broadcast_in_dim %530, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %533 = stablehlo.concatenate %531, %532, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %534 = "stablehlo.all_reduce"(%533) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_162 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %535 = stablehlo.broadcast_in_dim %cst_162, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %536 = stablehlo.divide %534, %535 : tensor<2x128xf32>
      %537 = stablehlo.slice %536 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %538 = stablehlo.reshape %537 : (tensor<1x128xf32>) -> tensor<128xf32>
      %539 = stablehlo.slice %536 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %540 = stablehlo.reshape %539 : (tensor<1x128xf32>) -> tensor<128xf32>
      %541 = stablehlo.multiply %538, %538 : tensor<128xf32>
      %cst_163 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %542 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %543 = stablehlo.multiply %542, %538 : tensor<128xf32>
      %544 = stablehlo.subtract %540, %541 : tensor<128xf32>
      %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %545 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %546 = stablehlo.maximum %545, %544 : tensor<128xf32>
      %547 = stablehlo.compare  EQ, %544, %546,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_165 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %548 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %549 = stablehlo.broadcast_in_dim %cst_166, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %550 = stablehlo.select %547, %548, %549 : tensor<128xi1>, tensor<128xf32>
      %cst_167 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %551 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %552 = stablehlo.compare  EQ, %551, %546,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_168 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %553 = stablehlo.broadcast_in_dim %cst_168, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_169 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %554 = stablehlo.broadcast_in_dim %cst_169, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %555 = stablehlo.select %552, %553, %554 : tensor<128xi1>, tensor<128xf32>
      %556 = stablehlo.divide %550, %555 : tensor<128xf32>
      %cst_170 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %557 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %558 = stablehlo.multiply %557, %arg558 : tensor<128xf32>
      %cst_171 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %559 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %560 = stablehlo.multiply %559, %538 : tensor<128xf32>
      %561 = stablehlo.add %558, %560 : tensor<128xf32>
      %cst_172 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %562 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %563 = stablehlo.multiply %562, %arg559 : tensor<128xf32>
      %cst_173 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %564 = stablehlo.broadcast_in_dim %cst_173, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %565 = stablehlo.multiply %564, %546 : tensor<128xf32>
      %566 = stablehlo.add %563, %565 : tensor<128xf32>
      %567 = stablehlo.broadcast_in_dim %538, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %568 = stablehlo.broadcast_in_dim %546, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %569 = stablehlo.convert %520 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %570 = stablehlo.broadcast_in_dim %567, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %571 = stablehlo.subtract %569, %570 : tensor<256x28x28x128xf32>
      %cst_174 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %572 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %573 = stablehlo.add %568, %572 : tensor<1x1x1x128xf32>
      %574 = stablehlo.rsqrt %573 : tensor<1x1x1x128xf32>
      %575 = stablehlo.divide %574, %573 : tensor<1x1x1x128xf32>
      %cst_175 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %576 = stablehlo.broadcast_in_dim %cst_175, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %577 = stablehlo.multiply %576, %575 : tensor<1x1x1x128xf32>
      %578 = stablehlo.reshape %arg359 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %579 = stablehlo.multiply %574, %578 : tensor<1x1x1x128xf32>
      %580 = stablehlo.broadcast_in_dim %579, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %581 = stablehlo.multiply %571, %580 : tensor<256x28x28x128xf32>
      %582 = stablehlo.reshape %arg358 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %583 = stablehlo.broadcast_in_dim %582, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %584 = stablehlo.add %581, %583 : tensor<256x28x28x128xf32>
      %585 = stablehlo.convert %584 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %586 = func.call @relu_90(%585) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_176 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %587 = stablehlo.broadcast_in_dim %cst_176, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %588 = stablehlo.compare  GT, %585, %587,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %589 = stablehlo.convert %arg363 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %590 = stablehlo.convolution(%586, %589) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %591 = stablehlo.convert %590 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %592 = stablehlo.multiply %591, %591 : tensor<256x28x28x128xf32>
      %cst_177 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %593 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %594 = stablehlo.multiply %593, %591 : tensor<256x28x28x128xf32>
      %cst_178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %595 = stablehlo.reduce(%591 init: %cst_178) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_179 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %596 = stablehlo.broadcast_in_dim %cst_179, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %597 = stablehlo.divide %595, %596 : tensor<128xf32>
      %cst_180 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %598 = stablehlo.reduce(%592 init: %cst_180) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_181 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %599 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %600 = stablehlo.divide %598, %599 : tensor<128xf32>
      %601 = stablehlo.broadcast_in_dim %597, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %602 = stablehlo.broadcast_in_dim %600, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %603 = stablehlo.concatenate %601, %602, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %604 = "stablehlo.all_reduce"(%603) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_182 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %605 = stablehlo.broadcast_in_dim %cst_182, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %606 = stablehlo.divide %604, %605 : tensor<2x128xf32>
      %607 = stablehlo.slice %606 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %608 = stablehlo.reshape %607 : (tensor<1x128xf32>) -> tensor<128xf32>
      %609 = stablehlo.slice %606 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %610 = stablehlo.reshape %609 : (tensor<1x128xf32>) -> tensor<128xf32>
      %611 = stablehlo.multiply %608, %608 : tensor<128xf32>
      %cst_183 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %612 = stablehlo.broadcast_in_dim %cst_183, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %613 = stablehlo.multiply %612, %608 : tensor<128xf32>
      %614 = stablehlo.subtract %610, %611 : tensor<128xf32>
      %cst_184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %615 = stablehlo.broadcast_in_dim %cst_184, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %616 = stablehlo.maximum %615, %614 : tensor<128xf32>
      %617 = stablehlo.compare  EQ, %614, %616,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_185 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %618 = stablehlo.broadcast_in_dim %cst_185, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %619 = stablehlo.broadcast_in_dim %cst_186, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %620 = stablehlo.select %617, %618, %619 : tensor<128xi1>, tensor<128xf32>
      %cst_187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %621 = stablehlo.broadcast_in_dim %cst_187, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %622 = stablehlo.compare  EQ, %621, %616,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_188 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %623 = stablehlo.broadcast_in_dim %cst_188, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_189 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %624 = stablehlo.broadcast_in_dim %cst_189, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %625 = stablehlo.select %622, %623, %624 : tensor<128xi1>, tensor<128xf32>
      %626 = stablehlo.divide %620, %625 : tensor<128xf32>
      %cst_190 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %627 = stablehlo.broadcast_in_dim %cst_190, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %628 = stablehlo.multiply %627, %arg560 : tensor<128xf32>
      %cst_191 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %629 = stablehlo.broadcast_in_dim %cst_191, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %630 = stablehlo.multiply %629, %608 : tensor<128xf32>
      %631 = stablehlo.add %628, %630 : tensor<128xf32>
      %cst_192 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %632 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %633 = stablehlo.multiply %632, %arg561 : tensor<128xf32>
      %cst_193 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %634 = stablehlo.broadcast_in_dim %cst_193, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %635 = stablehlo.multiply %634, %616 : tensor<128xf32>
      %636 = stablehlo.add %633, %635 : tensor<128xf32>
      %637 = stablehlo.broadcast_in_dim %608, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %638 = stablehlo.broadcast_in_dim %616, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %639 = stablehlo.convert %590 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %640 = stablehlo.broadcast_in_dim %637, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %641 = stablehlo.subtract %639, %640 : tensor<256x28x28x128xf32>
      %cst_194 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_194, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %643 = stablehlo.add %638, %642 : tensor<1x1x1x128xf32>
      %644 = stablehlo.rsqrt %643 : tensor<1x1x1x128xf32>
      %645 = stablehlo.divide %644, %643 : tensor<1x1x1x128xf32>
      %cst_195 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %647 = stablehlo.multiply %646, %645 : tensor<1x1x1x128xf32>
      %648 = stablehlo.reshape %arg361 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %649 = stablehlo.multiply %644, %648 : tensor<1x1x1x128xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %651 = stablehlo.multiply %641, %650 : tensor<256x28x28x128xf32>
      %652 = stablehlo.reshape %arg360 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %654 = stablehlo.add %651, %653 : tensor<256x28x28x128xf32>
      %655 = stablehlo.convert %654 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %656 = stablehlo.convert %arg364 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf16>
      %657 = stablehlo.convolution(%516, %656) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x128xf16>) -> tensor<256x28x28x128xf16>
      %658 = stablehlo.convert %657 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %659 = stablehlo.multiply %658, %658 : tensor<256x28x28x128xf32>
      %cst_196 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %660 = stablehlo.broadcast_in_dim %cst_196, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %661 = stablehlo.multiply %660, %658 : tensor<256x28x28x128xf32>
      %cst_197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %662 = stablehlo.reduce(%658 init: %cst_197) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_198 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %663 = stablehlo.broadcast_in_dim %cst_198, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %664 = stablehlo.divide %662, %663 : tensor<128xf32>
      %cst_199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %665 = stablehlo.reduce(%659 init: %cst_199) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_200 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_200, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %667 = stablehlo.divide %665, %666 : tensor<128xf32>
      %668 = stablehlo.broadcast_in_dim %664, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %669 = stablehlo.broadcast_in_dim %667, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %670 = stablehlo.concatenate %668, %669, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %671 = "stablehlo.all_reduce"(%670) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_201 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %672 = stablehlo.broadcast_in_dim %cst_201, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %673 = stablehlo.divide %671, %672 : tensor<2x128xf32>
      %674 = stablehlo.slice %673 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %675 = stablehlo.reshape %674 : (tensor<1x128xf32>) -> tensor<128xf32>
      %676 = stablehlo.slice %673 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %677 = stablehlo.reshape %676 : (tensor<1x128xf32>) -> tensor<128xf32>
      %678 = stablehlo.multiply %675, %675 : tensor<128xf32>
      %cst_202 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %679 = stablehlo.broadcast_in_dim %cst_202, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %680 = stablehlo.multiply %679, %675 : tensor<128xf32>
      %681 = stablehlo.subtract %677, %678 : tensor<128xf32>
      %cst_203 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %682 = stablehlo.broadcast_in_dim %cst_203, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %683 = stablehlo.maximum %682, %681 : tensor<128xf32>
      %684 = stablehlo.compare  EQ, %681, %683,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_204 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_204, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_205 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %686 = stablehlo.broadcast_in_dim %cst_205, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %687 = stablehlo.select %684, %685, %686 : tensor<128xi1>, tensor<128xf32>
      %cst_206 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %688 = stablehlo.broadcast_in_dim %cst_206, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %689 = stablehlo.compare  EQ, %688, %683,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_207 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %690 = stablehlo.broadcast_in_dim %cst_207, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_208 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %691 = stablehlo.broadcast_in_dim %cst_208, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %692 = stablehlo.select %689, %690, %691 : tensor<128xi1>, tensor<128xf32>
      %693 = stablehlo.divide %687, %692 : tensor<128xf32>
      %cst_209 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %694 = stablehlo.broadcast_in_dim %cst_209, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %695 = stablehlo.multiply %694, %arg562 : tensor<128xf32>
      %cst_210 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %696 = stablehlo.broadcast_in_dim %cst_210, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %697 = stablehlo.multiply %696, %675 : tensor<128xf32>
      %698 = stablehlo.add %695, %697 : tensor<128xf32>
      %cst_211 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %699 = stablehlo.broadcast_in_dim %cst_211, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %700 = stablehlo.multiply %699, %arg563 : tensor<128xf32>
      %cst_212 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %701 = stablehlo.broadcast_in_dim %cst_212, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %702 = stablehlo.multiply %701, %683 : tensor<128xf32>
      %703 = stablehlo.add %700, %702 : tensor<128xf32>
      %704 = stablehlo.broadcast_in_dim %675, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %705 = stablehlo.broadcast_in_dim %683, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %706 = stablehlo.convert %657 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %707 = stablehlo.broadcast_in_dim %704, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %708 = stablehlo.subtract %706, %707 : tensor<256x28x28x128xf32>
      %cst_213 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %709 = stablehlo.broadcast_in_dim %cst_213, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %710 = stablehlo.add %705, %709 : tensor<1x1x1x128xf32>
      %711 = stablehlo.rsqrt %710 : tensor<1x1x1x128xf32>
      %712 = stablehlo.divide %711, %710 : tensor<1x1x1x128xf32>
      %cst_214 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %713 = stablehlo.broadcast_in_dim %cst_214, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %714 = stablehlo.multiply %713, %712 : tensor<1x1x1x128xf32>
      %715 = stablehlo.reshape %arg366 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %716 = stablehlo.multiply %711, %715 : tensor<1x1x1x128xf32>
      %717 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %718 = stablehlo.multiply %708, %717 : tensor<256x28x28x128xf32>
      %719 = stablehlo.reshape %arg365 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %720 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %721 = stablehlo.add %718, %720 : tensor<256x28x28x128xf32>
      %722 = stablehlo.convert %721 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %723 = stablehlo.add %722, %655 : tensor<256x28x28x128xf16>
      %724 = func.call @relu_90(%723) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_215 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %725 = stablehlo.broadcast_in_dim %cst_215, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %726 = stablehlo.compare  GT, %723, %725,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %727 = stablehlo.convert %arg371 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %728 = stablehlo.convolution(%724, %727) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %729 = stablehlo.convert %728 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %730 = stablehlo.multiply %729, %729 : tensor<256x28x28x128xf32>
      %cst_216 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %731 = stablehlo.broadcast_in_dim %cst_216, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %732 = stablehlo.multiply %731, %729 : tensor<256x28x28x128xf32>
      %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %733 = stablehlo.reduce(%729 init: %cst_217) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_218 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %734 = stablehlo.broadcast_in_dim %cst_218, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %735 = stablehlo.divide %733, %734 : tensor<128xf32>
      %cst_219 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %736 = stablehlo.reduce(%730 init: %cst_219) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_220 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %737 = stablehlo.broadcast_in_dim %cst_220, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %738 = stablehlo.divide %736, %737 : tensor<128xf32>
      %739 = stablehlo.broadcast_in_dim %735, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %740 = stablehlo.broadcast_in_dim %738, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %741 = stablehlo.concatenate %739, %740, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %742 = "stablehlo.all_reduce"(%741) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_221 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %743 = stablehlo.broadcast_in_dim %cst_221, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %744 = stablehlo.divide %742, %743 : tensor<2x128xf32>
      %745 = stablehlo.slice %744 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %746 = stablehlo.reshape %745 : (tensor<1x128xf32>) -> tensor<128xf32>
      %747 = stablehlo.slice %744 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %748 = stablehlo.reshape %747 : (tensor<1x128xf32>) -> tensor<128xf32>
      %749 = stablehlo.multiply %746, %746 : tensor<128xf32>
      %cst_222 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %750 = stablehlo.broadcast_in_dim %cst_222, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %751 = stablehlo.multiply %750, %746 : tensor<128xf32>
      %752 = stablehlo.subtract %748, %749 : tensor<128xf32>
      %cst_223 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %753 = stablehlo.broadcast_in_dim %cst_223, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %754 = stablehlo.maximum %753, %752 : tensor<128xf32>
      %755 = stablehlo.compare  EQ, %752, %754,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_224 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %756 = stablehlo.broadcast_in_dim %cst_224, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_225 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %757 = stablehlo.broadcast_in_dim %cst_225, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %758 = stablehlo.select %755, %756, %757 : tensor<128xi1>, tensor<128xf32>
      %cst_226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %759 = stablehlo.broadcast_in_dim %cst_226, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %760 = stablehlo.compare  EQ, %759, %754,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_227 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %761 = stablehlo.broadcast_in_dim %cst_227, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_228 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %762 = stablehlo.broadcast_in_dim %cst_228, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %763 = stablehlo.select %760, %761, %762 : tensor<128xi1>, tensor<128xf32>
      %764 = stablehlo.divide %758, %763 : tensor<128xf32>
      %cst_229 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %765 = stablehlo.broadcast_in_dim %cst_229, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %766 = stablehlo.multiply %765, %arg564 : tensor<128xf32>
      %cst_230 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %767 = stablehlo.broadcast_in_dim %cst_230, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %768 = stablehlo.multiply %767, %746 : tensor<128xf32>
      %769 = stablehlo.add %766, %768 : tensor<128xf32>
      %cst_231 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %770 = stablehlo.broadcast_in_dim %cst_231, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %771 = stablehlo.multiply %770, %arg565 : tensor<128xf32>
      %cst_232 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %772 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %773 = stablehlo.multiply %772, %754 : tensor<128xf32>
      %774 = stablehlo.add %771, %773 : tensor<128xf32>
      %775 = stablehlo.broadcast_in_dim %746, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %776 = stablehlo.broadcast_in_dim %754, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %777 = stablehlo.convert %728 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %778 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %779 = stablehlo.subtract %777, %778 : tensor<256x28x28x128xf32>
      %cst_233 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %780 = stablehlo.broadcast_in_dim %cst_233, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %781 = stablehlo.add %776, %780 : tensor<1x1x1x128xf32>
      %782 = stablehlo.rsqrt %781 : tensor<1x1x1x128xf32>
      %783 = stablehlo.divide %782, %781 : tensor<1x1x1x128xf32>
      %cst_234 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %784 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %785 = stablehlo.multiply %784, %783 : tensor<1x1x1x128xf32>
      %786 = stablehlo.reshape %arg368 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %787 = stablehlo.multiply %782, %786 : tensor<1x1x1x128xf32>
      %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %789 = stablehlo.multiply %779, %788 : tensor<256x28x28x128xf32>
      %790 = stablehlo.reshape %arg367 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %791 = stablehlo.broadcast_in_dim %790, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %792 = stablehlo.add %789, %791 : tensor<256x28x28x128xf32>
      %793 = stablehlo.convert %792 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %794 = func.call @relu_90(%793) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_235 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %795 = stablehlo.broadcast_in_dim %cst_235, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %796 = stablehlo.compare  GT, %793, %795,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %797 = stablehlo.convert %arg372 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %798 = stablehlo.convolution(%794, %797) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %799 = stablehlo.convert %798 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %800 = stablehlo.multiply %799, %799 : tensor<256x28x28x128xf32>
      %cst_236 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %801 = stablehlo.broadcast_in_dim %cst_236, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %802 = stablehlo.multiply %801, %799 : tensor<256x28x28x128xf32>
      %cst_237 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %803 = stablehlo.reduce(%799 init: %cst_237) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_238 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %804 = stablehlo.broadcast_in_dim %cst_238, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %805 = stablehlo.divide %803, %804 : tensor<128xf32>
      %cst_239 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %806 = stablehlo.reduce(%800 init: %cst_239) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_240 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %807 = stablehlo.broadcast_in_dim %cst_240, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %808 = stablehlo.divide %806, %807 : tensor<128xf32>
      %809 = stablehlo.broadcast_in_dim %805, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %810 = stablehlo.broadcast_in_dim %808, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %811 = stablehlo.concatenate %809, %810, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %812 = "stablehlo.all_reduce"(%811) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_241 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %813 = stablehlo.broadcast_in_dim %cst_241, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %814 = stablehlo.divide %812, %813 : tensor<2x128xf32>
      %815 = stablehlo.slice %814 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %816 = stablehlo.reshape %815 : (tensor<1x128xf32>) -> tensor<128xf32>
      %817 = stablehlo.slice %814 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %818 = stablehlo.reshape %817 : (tensor<1x128xf32>) -> tensor<128xf32>
      %819 = stablehlo.multiply %816, %816 : tensor<128xf32>
      %cst_242 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %820 = stablehlo.broadcast_in_dim %cst_242, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %821 = stablehlo.multiply %820, %816 : tensor<128xf32>
      %822 = stablehlo.subtract %818, %819 : tensor<128xf32>
      %cst_243 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %823 = stablehlo.broadcast_in_dim %cst_243, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %824 = stablehlo.maximum %823, %822 : tensor<128xf32>
      %825 = stablehlo.compare  EQ, %822, %824,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_244 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %826 = stablehlo.broadcast_in_dim %cst_244, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_245 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %827 = stablehlo.broadcast_in_dim %cst_245, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %828 = stablehlo.select %825, %826, %827 : tensor<128xi1>, tensor<128xf32>
      %cst_246 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %829 = stablehlo.broadcast_in_dim %cst_246, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %830 = stablehlo.compare  EQ, %829, %824,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_247 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %831 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_248 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %832 = stablehlo.broadcast_in_dim %cst_248, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %833 = stablehlo.select %830, %831, %832 : tensor<128xi1>, tensor<128xf32>
      %834 = stablehlo.divide %828, %833 : tensor<128xf32>
      %cst_249 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %835 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %836 = stablehlo.multiply %835, %arg566 : tensor<128xf32>
      %cst_250 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %837 = stablehlo.broadcast_in_dim %cst_250, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %838 = stablehlo.multiply %837, %816 : tensor<128xf32>
      %839 = stablehlo.add %836, %838 : tensor<128xf32>
      %cst_251 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %840 = stablehlo.broadcast_in_dim %cst_251, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %841 = stablehlo.multiply %840, %arg567 : tensor<128xf32>
      %cst_252 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %842 = stablehlo.broadcast_in_dim %cst_252, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %843 = stablehlo.multiply %842, %824 : tensor<128xf32>
      %844 = stablehlo.add %841, %843 : tensor<128xf32>
      %845 = stablehlo.broadcast_in_dim %816, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %846 = stablehlo.broadcast_in_dim %824, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %847 = stablehlo.convert %798 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %848 = stablehlo.broadcast_in_dim %845, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %849 = stablehlo.subtract %847, %848 : tensor<256x28x28x128xf32>
      %cst_253 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %850 = stablehlo.broadcast_in_dim %cst_253, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %851 = stablehlo.add %846, %850 : tensor<1x1x1x128xf32>
      %852 = stablehlo.rsqrt %851 : tensor<1x1x1x128xf32>
      %853 = stablehlo.divide %852, %851 : tensor<1x1x1x128xf32>
      %cst_254 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %854 = stablehlo.broadcast_in_dim %cst_254, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %855 = stablehlo.multiply %854, %853 : tensor<1x1x1x128xf32>
      %856 = stablehlo.reshape %arg370 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %857 = stablehlo.multiply %852, %856 : tensor<1x1x1x128xf32>
      %858 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %859 = stablehlo.multiply %849, %858 : tensor<256x28x28x128xf32>
      %860 = stablehlo.reshape %arg369 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %861 = stablehlo.broadcast_in_dim %860, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %862 = stablehlo.add %859, %861 : tensor<256x28x28x128xf32>
      %863 = stablehlo.convert %862 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %864 = stablehlo.add %724, %863 : tensor<256x28x28x128xf16>
      %865 = func.call @relu_90(%864) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_255 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %866 = stablehlo.broadcast_in_dim %cst_255, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %867 = stablehlo.compare  GT, %864, %866,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %868 = stablehlo.convert %arg377 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %869 = stablehlo.convolution(%865, %868) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %870 = stablehlo.convert %869 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %871 = stablehlo.multiply %870, %870 : tensor<256x28x28x128xf32>
      %cst_256 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %872 = stablehlo.broadcast_in_dim %cst_256, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %873 = stablehlo.multiply %872, %870 : tensor<256x28x28x128xf32>
      %cst_257 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %874 = stablehlo.reduce(%870 init: %cst_257) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_258 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %875 = stablehlo.broadcast_in_dim %cst_258, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %876 = stablehlo.divide %874, %875 : tensor<128xf32>
      %cst_259 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %877 = stablehlo.reduce(%871 init: %cst_259) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_260 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %878 = stablehlo.broadcast_in_dim %cst_260, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %879 = stablehlo.divide %877, %878 : tensor<128xf32>
      %880 = stablehlo.broadcast_in_dim %876, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %881 = stablehlo.broadcast_in_dim %879, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %882 = stablehlo.concatenate %880, %881, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %883 = "stablehlo.all_reduce"(%882) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_261 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %884 = stablehlo.broadcast_in_dim %cst_261, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %885 = stablehlo.divide %883, %884 : tensor<2x128xf32>
      %886 = stablehlo.slice %885 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %887 = stablehlo.reshape %886 : (tensor<1x128xf32>) -> tensor<128xf32>
      %888 = stablehlo.slice %885 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %889 = stablehlo.reshape %888 : (tensor<1x128xf32>) -> tensor<128xf32>
      %890 = stablehlo.multiply %887, %887 : tensor<128xf32>
      %cst_262 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %891 = stablehlo.broadcast_in_dim %cst_262, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %892 = stablehlo.multiply %891, %887 : tensor<128xf32>
      %893 = stablehlo.subtract %889, %890 : tensor<128xf32>
      %cst_263 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %894 = stablehlo.broadcast_in_dim %cst_263, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %895 = stablehlo.maximum %894, %893 : tensor<128xf32>
      %896 = stablehlo.compare  EQ, %893, %895,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_264 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %897 = stablehlo.broadcast_in_dim %cst_264, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_265 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %898 = stablehlo.broadcast_in_dim %cst_265, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %899 = stablehlo.select %896, %897, %898 : tensor<128xi1>, tensor<128xf32>
      %cst_266 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %900 = stablehlo.broadcast_in_dim %cst_266, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %901 = stablehlo.compare  EQ, %900, %895,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_267 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %902 = stablehlo.broadcast_in_dim %cst_267, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_268 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %903 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %904 = stablehlo.select %901, %902, %903 : tensor<128xi1>, tensor<128xf32>
      %905 = stablehlo.divide %899, %904 : tensor<128xf32>
      %cst_269 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %906 = stablehlo.broadcast_in_dim %cst_269, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %907 = stablehlo.multiply %906, %arg568 : tensor<128xf32>
      %cst_270 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %908 = stablehlo.broadcast_in_dim %cst_270, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %909 = stablehlo.multiply %908, %887 : tensor<128xf32>
      %910 = stablehlo.add %907, %909 : tensor<128xf32>
      %cst_271 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %911 = stablehlo.broadcast_in_dim %cst_271, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %912 = stablehlo.multiply %911, %arg569 : tensor<128xf32>
      %cst_272 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %913 = stablehlo.broadcast_in_dim %cst_272, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %914 = stablehlo.multiply %913, %895 : tensor<128xf32>
      %915 = stablehlo.add %912, %914 : tensor<128xf32>
      %916 = stablehlo.broadcast_in_dim %887, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %917 = stablehlo.broadcast_in_dim %895, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %918 = stablehlo.convert %869 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %919 = stablehlo.broadcast_in_dim %916, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %920 = stablehlo.subtract %918, %919 : tensor<256x28x28x128xf32>
      %cst_273 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %921 = stablehlo.broadcast_in_dim %cst_273, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %922 = stablehlo.add %917, %921 : tensor<1x1x1x128xf32>
      %923 = stablehlo.rsqrt %922 : tensor<1x1x1x128xf32>
      %924 = stablehlo.divide %923, %922 : tensor<1x1x1x128xf32>
      %cst_274 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %925 = stablehlo.broadcast_in_dim %cst_274, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %926 = stablehlo.multiply %925, %924 : tensor<1x1x1x128xf32>
      %927 = stablehlo.reshape %arg374 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %928 = stablehlo.multiply %923, %927 : tensor<1x1x1x128xf32>
      %929 = stablehlo.broadcast_in_dim %928, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %930 = stablehlo.multiply %920, %929 : tensor<256x28x28x128xf32>
      %931 = stablehlo.reshape %arg373 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %932 = stablehlo.broadcast_in_dim %931, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %933 = stablehlo.add %930, %932 : tensor<256x28x28x128xf32>
      %934 = stablehlo.convert %933 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %935 = func.call @relu_90(%934) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_275 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %936 = stablehlo.broadcast_in_dim %cst_275, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %937 = stablehlo.compare  GT, %934, %936,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %938 = stablehlo.convert %arg378 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %939 = stablehlo.convolution(%935, %938) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %940 = stablehlo.convert %939 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %941 = stablehlo.multiply %940, %940 : tensor<256x28x28x128xf32>
      %cst_276 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %942 = stablehlo.broadcast_in_dim %cst_276, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %943 = stablehlo.multiply %942, %940 : tensor<256x28x28x128xf32>
      %cst_277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %944 = stablehlo.reduce(%940 init: %cst_277) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_278 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %945 = stablehlo.broadcast_in_dim %cst_278, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %946 = stablehlo.divide %944, %945 : tensor<128xf32>
      %cst_279 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %947 = stablehlo.reduce(%941 init: %cst_279) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_280 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %948 = stablehlo.broadcast_in_dim %cst_280, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %949 = stablehlo.divide %947, %948 : tensor<128xf32>
      %950 = stablehlo.broadcast_in_dim %946, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %951 = stablehlo.broadcast_in_dim %949, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %952 = stablehlo.concatenate %950, %951, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %953 = "stablehlo.all_reduce"(%952) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_281 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %954 = stablehlo.broadcast_in_dim %cst_281, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %955 = stablehlo.divide %953, %954 : tensor<2x128xf32>
      %956 = stablehlo.slice %955 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %957 = stablehlo.reshape %956 : (tensor<1x128xf32>) -> tensor<128xf32>
      %958 = stablehlo.slice %955 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %959 = stablehlo.reshape %958 : (tensor<1x128xf32>) -> tensor<128xf32>
      %960 = stablehlo.multiply %957, %957 : tensor<128xf32>
      %cst_282 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %961 = stablehlo.broadcast_in_dim %cst_282, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %962 = stablehlo.multiply %961, %957 : tensor<128xf32>
      %963 = stablehlo.subtract %959, %960 : tensor<128xf32>
      %cst_283 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %964 = stablehlo.broadcast_in_dim %cst_283, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %965 = stablehlo.maximum %964, %963 : tensor<128xf32>
      %966 = stablehlo.compare  EQ, %963, %965,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_284 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %967 = stablehlo.broadcast_in_dim %cst_284, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_285 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %968 = stablehlo.broadcast_in_dim %cst_285, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %969 = stablehlo.select %966, %967, %968 : tensor<128xi1>, tensor<128xf32>
      %cst_286 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %970 = stablehlo.broadcast_in_dim %cst_286, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %971 = stablehlo.compare  EQ, %970, %965,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_287 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %972 = stablehlo.broadcast_in_dim %cst_287, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_288 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %973 = stablehlo.broadcast_in_dim %cst_288, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %974 = stablehlo.select %971, %972, %973 : tensor<128xi1>, tensor<128xf32>
      %975 = stablehlo.divide %969, %974 : tensor<128xf32>
      %cst_289 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %976 = stablehlo.broadcast_in_dim %cst_289, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %977 = stablehlo.multiply %976, %arg570 : tensor<128xf32>
      %cst_290 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %978 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %979 = stablehlo.multiply %978, %957 : tensor<128xf32>
      %980 = stablehlo.add %977, %979 : tensor<128xf32>
      %cst_291 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %981 = stablehlo.broadcast_in_dim %cst_291, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %982 = stablehlo.multiply %981, %arg571 : tensor<128xf32>
      %cst_292 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %983 = stablehlo.broadcast_in_dim %cst_292, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %984 = stablehlo.multiply %983, %965 : tensor<128xf32>
      %985 = stablehlo.add %982, %984 : tensor<128xf32>
      %986 = stablehlo.broadcast_in_dim %957, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %987 = stablehlo.broadcast_in_dim %965, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %988 = stablehlo.convert %939 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %989 = stablehlo.broadcast_in_dim %986, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %990 = stablehlo.subtract %988, %989 : tensor<256x28x28x128xf32>
      %cst_293 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %991 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %992 = stablehlo.add %987, %991 : tensor<1x1x1x128xf32>
      %993 = stablehlo.rsqrt %992 : tensor<1x1x1x128xf32>
      %994 = stablehlo.divide %993, %992 : tensor<1x1x1x128xf32>
      %cst_294 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %995 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %996 = stablehlo.multiply %995, %994 : tensor<1x1x1x128xf32>
      %997 = stablehlo.reshape %arg376 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %998 = stablehlo.multiply %993, %997 : tensor<1x1x1x128xf32>
      %999 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1000 = stablehlo.multiply %990, %999 : tensor<256x28x28x128xf32>
      %1001 = stablehlo.reshape %arg375 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1002 = stablehlo.broadcast_in_dim %1001, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1003 = stablehlo.add %1000, %1002 : tensor<256x28x28x128xf32>
      %1004 = stablehlo.convert %1003 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1005 = stablehlo.add %865, %1004 : tensor<256x28x28x128xf16>
      %1006 = func.call @relu_90(%1005) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_295 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1007 = stablehlo.broadcast_in_dim %cst_295, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %1008 = stablehlo.compare  GT, %1005, %1007,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %1009 = stablehlo.convert %arg383 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1010 = stablehlo.convolution(%1006, %1009) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1011 = stablehlo.convert %1010 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1012 = stablehlo.multiply %1011, %1011 : tensor<256x28x28x128xf32>
      %cst_296 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1013 = stablehlo.broadcast_in_dim %cst_296, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %1014 = stablehlo.multiply %1013, %1011 : tensor<256x28x28x128xf32>
      %cst_297 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1015 = stablehlo.reduce(%1011 init: %cst_297) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_298 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %1016 = stablehlo.broadcast_in_dim %cst_298, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1017 = stablehlo.divide %1015, %1016 : tensor<128xf32>
      %cst_299 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1018 = stablehlo.reduce(%1012 init: %cst_299) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_300 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %1019 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1020 = stablehlo.divide %1018, %1019 : tensor<128xf32>
      %1021 = stablehlo.broadcast_in_dim %1017, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %1022 = stablehlo.broadcast_in_dim %1020, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %1023 = stablehlo.concatenate %1021, %1022, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %1024 = "stablehlo.all_reduce"(%1023) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_301 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1025 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %1026 = stablehlo.divide %1024, %1025 : tensor<2x128xf32>
      %1027 = stablehlo.slice %1026 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %1028 = stablehlo.reshape %1027 : (tensor<1x128xf32>) -> tensor<128xf32>
      %1029 = stablehlo.slice %1026 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %1030 = stablehlo.reshape %1029 : (tensor<1x128xf32>) -> tensor<128xf32>
      %1031 = stablehlo.multiply %1028, %1028 : tensor<128xf32>
      %cst_302 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1032 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1033 = stablehlo.multiply %1032, %1028 : tensor<128xf32>
      %1034 = stablehlo.subtract %1030, %1031 : tensor<128xf32>
      %cst_303 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1035 = stablehlo.broadcast_in_dim %cst_303, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1036 = stablehlo.maximum %1035, %1034 : tensor<128xf32>
      %1037 = stablehlo.compare  EQ, %1034, %1036,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_304 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1038 = stablehlo.broadcast_in_dim %cst_304, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_305 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1039 = stablehlo.broadcast_in_dim %cst_305, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1040 = stablehlo.select %1037, %1038, %1039 : tensor<128xi1>, tensor<128xf32>
      %cst_306 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1041 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1042 = stablehlo.compare  EQ, %1041, %1036,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_307 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1043 = stablehlo.broadcast_in_dim %cst_307, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_308 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1044 = stablehlo.broadcast_in_dim %cst_308, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1045 = stablehlo.select %1042, %1043, %1044 : tensor<128xi1>, tensor<128xf32>
      %1046 = stablehlo.divide %1040, %1045 : tensor<128xf32>
      %cst_309 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1047 = stablehlo.broadcast_in_dim %cst_309, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1048 = stablehlo.multiply %1047, %arg572 : tensor<128xf32>
      %cst_310 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1049 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1050 = stablehlo.multiply %1049, %1028 : tensor<128xf32>
      %1051 = stablehlo.add %1048, %1050 : tensor<128xf32>
      %cst_311 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1052 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1053 = stablehlo.multiply %1052, %arg573 : tensor<128xf32>
      %cst_312 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1054 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1055 = stablehlo.multiply %1054, %1036 : tensor<128xf32>
      %1056 = stablehlo.add %1053, %1055 : tensor<128xf32>
      %1057 = stablehlo.broadcast_in_dim %1028, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1058 = stablehlo.broadcast_in_dim %1036, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1059 = stablehlo.convert %1010 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1060 = stablehlo.broadcast_in_dim %1057, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1061 = stablehlo.subtract %1059, %1060 : tensor<256x28x28x128xf32>
      %cst_313 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1062 = stablehlo.broadcast_in_dim %cst_313, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1063 = stablehlo.add %1058, %1062 : tensor<1x1x1x128xf32>
      %1064 = stablehlo.rsqrt %1063 : tensor<1x1x1x128xf32>
      %1065 = stablehlo.divide %1064, %1063 : tensor<1x1x1x128xf32>
      %cst_314 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1066 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1067 = stablehlo.multiply %1066, %1065 : tensor<1x1x1x128xf32>
      %1068 = stablehlo.reshape %arg380 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1069 = stablehlo.multiply %1064, %1068 : tensor<1x1x1x128xf32>
      %1070 = stablehlo.broadcast_in_dim %1069, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1071 = stablehlo.multiply %1061, %1070 : tensor<256x28x28x128xf32>
      %1072 = stablehlo.reshape %arg379 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1073 = stablehlo.broadcast_in_dim %1072, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1074 = stablehlo.add %1071, %1073 : tensor<256x28x28x128xf32>
      %1075 = stablehlo.convert %1074 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1076 = func.call @relu_90(%1075) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_315 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1077 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %1078 = stablehlo.compare  GT, %1075, %1077,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %1079 = stablehlo.convert %arg384 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1080 = stablehlo.convolution(%1076, %1079) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1081 = stablehlo.convert %1080 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1082 = stablehlo.multiply %1081, %1081 : tensor<256x28x28x128xf32>
      %cst_316 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1083 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %1084 = stablehlo.multiply %1083, %1081 : tensor<256x28x28x128xf32>
      %cst_317 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1085 = stablehlo.reduce(%1081 init: %cst_317) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_318 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %1086 = stablehlo.broadcast_in_dim %cst_318, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1087 = stablehlo.divide %1085, %1086 : tensor<128xf32>
      %cst_319 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1088 = stablehlo.reduce(%1082 init: %cst_319) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_320 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %1089 = stablehlo.broadcast_in_dim %cst_320, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1090 = stablehlo.divide %1088, %1089 : tensor<128xf32>
      %1091 = stablehlo.broadcast_in_dim %1087, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %1092 = stablehlo.broadcast_in_dim %1090, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %1093 = stablehlo.concatenate %1091, %1092, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %1094 = "stablehlo.all_reduce"(%1093) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_321 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1095 = stablehlo.broadcast_in_dim %cst_321, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %1096 = stablehlo.divide %1094, %1095 : tensor<2x128xf32>
      %1097 = stablehlo.slice %1096 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %1098 = stablehlo.reshape %1097 : (tensor<1x128xf32>) -> tensor<128xf32>
      %1099 = stablehlo.slice %1096 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %1100 = stablehlo.reshape %1099 : (tensor<1x128xf32>) -> tensor<128xf32>
      %1101 = stablehlo.multiply %1098, %1098 : tensor<128xf32>
      %cst_322 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1102 = stablehlo.broadcast_in_dim %cst_322, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1103 = stablehlo.multiply %1102, %1098 : tensor<128xf32>
      %1104 = stablehlo.subtract %1100, %1101 : tensor<128xf32>
      %cst_323 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1105 = stablehlo.broadcast_in_dim %cst_323, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1106 = stablehlo.maximum %1105, %1104 : tensor<128xf32>
      %1107 = stablehlo.compare  EQ, %1104, %1106,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_324 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1108 = stablehlo.broadcast_in_dim %cst_324, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_325 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1109 = stablehlo.broadcast_in_dim %cst_325, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1110 = stablehlo.select %1107, %1108, %1109 : tensor<128xi1>, tensor<128xf32>
      %cst_326 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1111 = stablehlo.broadcast_in_dim %cst_326, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1112 = stablehlo.compare  EQ, %1111, %1106,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_327 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1113 = stablehlo.broadcast_in_dim %cst_327, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_328 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1114 = stablehlo.broadcast_in_dim %cst_328, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1115 = stablehlo.select %1112, %1113, %1114 : tensor<128xi1>, tensor<128xf32>
      %1116 = stablehlo.divide %1110, %1115 : tensor<128xf32>
      %cst_329 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1117 = stablehlo.broadcast_in_dim %cst_329, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1118 = stablehlo.multiply %1117, %arg574 : tensor<128xf32>
      %cst_330 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1119 = stablehlo.broadcast_in_dim %cst_330, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1120 = stablehlo.multiply %1119, %1098 : tensor<128xf32>
      %1121 = stablehlo.add %1118, %1120 : tensor<128xf32>
      %cst_331 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1122 = stablehlo.broadcast_in_dim %cst_331, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1123 = stablehlo.multiply %1122, %arg575 : tensor<128xf32>
      %cst_332 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1124 = stablehlo.broadcast_in_dim %cst_332, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %1125 = stablehlo.multiply %1124, %1106 : tensor<128xf32>
      %1126 = stablehlo.add %1123, %1125 : tensor<128xf32>
      %1127 = stablehlo.broadcast_in_dim %1098, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1128 = stablehlo.broadcast_in_dim %1106, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1129 = stablehlo.convert %1080 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1130 = stablehlo.broadcast_in_dim %1127, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1131 = stablehlo.subtract %1129, %1130 : tensor<256x28x28x128xf32>
      %cst_333 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1132 = stablehlo.broadcast_in_dim %cst_333, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1133 = stablehlo.add %1128, %1132 : tensor<1x1x1x128xf32>
      %1134 = stablehlo.rsqrt %1133 : tensor<1x1x1x128xf32>
      %1135 = stablehlo.divide %1134, %1133 : tensor<1x1x1x128xf32>
      %cst_334 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1136 = stablehlo.broadcast_in_dim %cst_334, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1137 = stablehlo.multiply %1136, %1135 : tensor<1x1x1x128xf32>
      %1138 = stablehlo.reshape %arg382 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1139 = stablehlo.multiply %1134, %1138 : tensor<1x1x1x128xf32>
      %1140 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1141 = stablehlo.multiply %1131, %1140 : tensor<256x28x28x128xf32>
      %1142 = stablehlo.reshape %arg381 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1143 = stablehlo.broadcast_in_dim %1142, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1144 = stablehlo.add %1141, %1143 : tensor<256x28x28x128xf32>
      %1145 = stablehlo.convert %1144 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1146 = stablehlo.add %1006, %1145 : tensor<256x28x28x128xf16>
      %1147 = func.call @relu_90(%1146) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_335 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1148 = stablehlo.broadcast_in_dim %cst_335, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %1149 = stablehlo.compare  GT, %1146, %1148,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %1150 = stablehlo.convert %arg389 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf16>
      %1151 = stablehlo.convolution(%1147, %1150) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x256xf16>) -> tensor<256x14x14x256xf16>
      %1152 = stablehlo.convert %1151 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1153 = stablehlo.multiply %1152, %1152 : tensor<256x14x14x256xf32>
      %cst_336 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1154 = stablehlo.broadcast_in_dim %cst_336, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1155 = stablehlo.multiply %1154, %1152 : tensor<256x14x14x256xf32>
      %cst_337 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1156 = stablehlo.reduce(%1152 init: %cst_337) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_338 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1157 = stablehlo.broadcast_in_dim %cst_338, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1158 = stablehlo.divide %1156, %1157 : tensor<256xf32>
      %cst_339 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1159 = stablehlo.reduce(%1153 init: %cst_339) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_340 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1160 = stablehlo.broadcast_in_dim %cst_340, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1161 = stablehlo.divide %1159, %1160 : tensor<256xf32>
      %1162 = stablehlo.broadcast_in_dim %1158, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1163 = stablehlo.broadcast_in_dim %1161, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1164 = stablehlo.concatenate %1162, %1163, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1165 = "stablehlo.all_reduce"(%1164) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_341 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1166 = stablehlo.broadcast_in_dim %cst_341, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1167 = stablehlo.divide %1165, %1166 : tensor<2x256xf32>
      %1168 = stablehlo.slice %1167 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1169 = stablehlo.reshape %1168 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1170 = stablehlo.slice %1167 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1171 = stablehlo.reshape %1170 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1172 = stablehlo.multiply %1169, %1169 : tensor<256xf32>
      %cst_342 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1173 = stablehlo.broadcast_in_dim %cst_342, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1174 = stablehlo.multiply %1173, %1169 : tensor<256xf32>
      %1175 = stablehlo.subtract %1171, %1172 : tensor<256xf32>
      %cst_343 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1176 = stablehlo.broadcast_in_dim %cst_343, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1177 = stablehlo.maximum %1176, %1175 : tensor<256xf32>
      %1178 = stablehlo.compare  EQ, %1175, %1177,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_344 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1179 = stablehlo.broadcast_in_dim %cst_344, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_345 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1180 = stablehlo.broadcast_in_dim %cst_345, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1181 = stablehlo.select %1178, %1179, %1180 : tensor<256xi1>, tensor<256xf32>
      %cst_346 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1182 = stablehlo.broadcast_in_dim %cst_346, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1183 = stablehlo.compare  EQ, %1182, %1177,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_347 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1184 = stablehlo.broadcast_in_dim %cst_347, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_348 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1185 = stablehlo.broadcast_in_dim %cst_348, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1186 = stablehlo.select %1183, %1184, %1185 : tensor<256xi1>, tensor<256xf32>
      %1187 = stablehlo.divide %1181, %1186 : tensor<256xf32>
      %cst_349 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1188 = stablehlo.broadcast_in_dim %cst_349, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1189 = stablehlo.multiply %1188, %arg576 : tensor<256xf32>
      %cst_350 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1190 = stablehlo.broadcast_in_dim %cst_350, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1191 = stablehlo.multiply %1190, %1169 : tensor<256xf32>
      %1192 = stablehlo.add %1189, %1191 : tensor<256xf32>
      %cst_351 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1193 = stablehlo.broadcast_in_dim %cst_351, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1194 = stablehlo.multiply %1193, %arg577 : tensor<256xf32>
      %cst_352 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1195 = stablehlo.broadcast_in_dim %cst_352, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1196 = stablehlo.multiply %1195, %1177 : tensor<256xf32>
      %1197 = stablehlo.add %1194, %1196 : tensor<256xf32>
      %1198 = stablehlo.broadcast_in_dim %1169, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1199 = stablehlo.broadcast_in_dim %1177, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1200 = stablehlo.convert %1151 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1201 = stablehlo.broadcast_in_dim %1198, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1202 = stablehlo.subtract %1200, %1201 : tensor<256x14x14x256xf32>
      %cst_353 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1203 = stablehlo.broadcast_in_dim %cst_353, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1204 = stablehlo.add %1199, %1203 : tensor<1x1x1x256xf32>
      %1205 = stablehlo.rsqrt %1204 : tensor<1x1x1x256xf32>
      %1206 = stablehlo.divide %1205, %1204 : tensor<1x1x1x256xf32>
      %cst_354 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1207 = stablehlo.broadcast_in_dim %cst_354, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1208 = stablehlo.multiply %1207, %1206 : tensor<1x1x1x256xf32>
      %1209 = stablehlo.reshape %arg386 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1210 = stablehlo.multiply %1205, %1209 : tensor<1x1x1x256xf32>
      %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1212 = stablehlo.multiply %1202, %1211 : tensor<256x14x14x256xf32>
      %1213 = stablehlo.reshape %arg385 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1214 = stablehlo.broadcast_in_dim %1213, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1215 = stablehlo.add %1212, %1214 : tensor<256x14x14x256xf32>
      %1216 = stablehlo.convert %1215 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1217 = func.call @relu_142(%1216) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_355 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1218 = stablehlo.broadcast_in_dim %cst_355, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1219 = stablehlo.compare  GT, %1216, %1218,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1220 = stablehlo.convert %arg390 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1221 = stablehlo.convolution(%1217, %1220) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1222 = stablehlo.convert %1221 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1223 = stablehlo.multiply %1222, %1222 : tensor<256x14x14x256xf32>
      %cst_356 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1224 = stablehlo.broadcast_in_dim %cst_356, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1225 = stablehlo.multiply %1224, %1222 : tensor<256x14x14x256xf32>
      %cst_357 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1226 = stablehlo.reduce(%1222 init: %cst_357) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_358 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1227 = stablehlo.broadcast_in_dim %cst_358, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1228 = stablehlo.divide %1226, %1227 : tensor<256xf32>
      %cst_359 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1229 = stablehlo.reduce(%1223 init: %cst_359) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_360 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1230 = stablehlo.broadcast_in_dim %cst_360, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1231 = stablehlo.divide %1229, %1230 : tensor<256xf32>
      %1232 = stablehlo.broadcast_in_dim %1228, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1233 = stablehlo.broadcast_in_dim %1231, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1234 = stablehlo.concatenate %1232, %1233, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1235 = "stablehlo.all_reduce"(%1234) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_361 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1236 = stablehlo.broadcast_in_dim %cst_361, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1237 = stablehlo.divide %1235, %1236 : tensor<2x256xf32>
      %1238 = stablehlo.slice %1237 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1239 = stablehlo.reshape %1238 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1240 = stablehlo.slice %1237 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1241 = stablehlo.reshape %1240 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1242 = stablehlo.multiply %1239, %1239 : tensor<256xf32>
      %cst_362 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1243 = stablehlo.broadcast_in_dim %cst_362, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1244 = stablehlo.multiply %1243, %1239 : tensor<256xf32>
      %1245 = stablehlo.subtract %1241, %1242 : tensor<256xf32>
      %cst_363 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1246 = stablehlo.broadcast_in_dim %cst_363, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1247 = stablehlo.maximum %1246, %1245 : tensor<256xf32>
      %1248 = stablehlo.compare  EQ, %1245, %1247,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_364 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1249 = stablehlo.broadcast_in_dim %cst_364, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_365 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1250 = stablehlo.broadcast_in_dim %cst_365, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1251 = stablehlo.select %1248, %1249, %1250 : tensor<256xi1>, tensor<256xf32>
      %cst_366 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1252 = stablehlo.broadcast_in_dim %cst_366, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1253 = stablehlo.compare  EQ, %1252, %1247,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_367 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1254 = stablehlo.broadcast_in_dim %cst_367, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_368 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1255 = stablehlo.broadcast_in_dim %cst_368, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1256 = stablehlo.select %1253, %1254, %1255 : tensor<256xi1>, tensor<256xf32>
      %1257 = stablehlo.divide %1251, %1256 : tensor<256xf32>
      %cst_369 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1258 = stablehlo.broadcast_in_dim %cst_369, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1259 = stablehlo.multiply %1258, %arg578 : tensor<256xf32>
      %cst_370 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1260 = stablehlo.broadcast_in_dim %cst_370, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1261 = stablehlo.multiply %1260, %1239 : tensor<256xf32>
      %1262 = stablehlo.add %1259, %1261 : tensor<256xf32>
      %cst_371 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1263 = stablehlo.broadcast_in_dim %cst_371, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1264 = stablehlo.multiply %1263, %arg579 : tensor<256xf32>
      %cst_372 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1265 = stablehlo.broadcast_in_dim %cst_372, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1266 = stablehlo.multiply %1265, %1247 : tensor<256xf32>
      %1267 = stablehlo.add %1264, %1266 : tensor<256xf32>
      %1268 = stablehlo.broadcast_in_dim %1239, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1269 = stablehlo.broadcast_in_dim %1247, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1270 = stablehlo.convert %1221 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1271 = stablehlo.broadcast_in_dim %1268, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1272 = stablehlo.subtract %1270, %1271 : tensor<256x14x14x256xf32>
      %cst_373 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1273 = stablehlo.broadcast_in_dim %cst_373, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1274 = stablehlo.add %1269, %1273 : tensor<1x1x1x256xf32>
      %1275 = stablehlo.rsqrt %1274 : tensor<1x1x1x256xf32>
      %1276 = stablehlo.divide %1275, %1274 : tensor<1x1x1x256xf32>
      %cst_374 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1277 = stablehlo.broadcast_in_dim %cst_374, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1278 = stablehlo.multiply %1277, %1276 : tensor<1x1x1x256xf32>
      %1279 = stablehlo.reshape %arg388 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1280 = stablehlo.multiply %1275, %1279 : tensor<1x1x1x256xf32>
      %1281 = stablehlo.broadcast_in_dim %1280, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1282 = stablehlo.multiply %1272, %1281 : tensor<256x14x14x256xf32>
      %1283 = stablehlo.reshape %arg387 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1284 = stablehlo.broadcast_in_dim %1283, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1285 = stablehlo.add %1282, %1284 : tensor<256x14x14x256xf32>
      %1286 = stablehlo.convert %1285 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1287 = stablehlo.convert %arg391 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf16>
      %1288 = stablehlo.convolution(%1147, %1287) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x256xf16>) -> tensor<256x14x14x256xf16>
      %1289 = stablehlo.convert %1288 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1290 = stablehlo.multiply %1289, %1289 : tensor<256x14x14x256xf32>
      %cst_375 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1291 = stablehlo.broadcast_in_dim %cst_375, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1292 = stablehlo.multiply %1291, %1289 : tensor<256x14x14x256xf32>
      %cst_376 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1293 = stablehlo.reduce(%1289 init: %cst_376) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_377 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1294 = stablehlo.broadcast_in_dim %cst_377, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1295 = stablehlo.divide %1293, %1294 : tensor<256xf32>
      %cst_378 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1296 = stablehlo.reduce(%1290 init: %cst_378) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_379 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1297 = stablehlo.broadcast_in_dim %cst_379, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1298 = stablehlo.divide %1296, %1297 : tensor<256xf32>
      %1299 = stablehlo.broadcast_in_dim %1295, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1300 = stablehlo.broadcast_in_dim %1298, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1301 = stablehlo.concatenate %1299, %1300, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1302 = "stablehlo.all_reduce"(%1301) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_380 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1303 = stablehlo.broadcast_in_dim %cst_380, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1304 = stablehlo.divide %1302, %1303 : tensor<2x256xf32>
      %1305 = stablehlo.slice %1304 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1306 = stablehlo.reshape %1305 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1307 = stablehlo.slice %1304 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1308 = stablehlo.reshape %1307 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1309 = stablehlo.multiply %1306, %1306 : tensor<256xf32>
      %cst_381 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1310 = stablehlo.broadcast_in_dim %cst_381, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1311 = stablehlo.multiply %1310, %1306 : tensor<256xf32>
      %1312 = stablehlo.subtract %1308, %1309 : tensor<256xf32>
      %cst_382 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1313 = stablehlo.broadcast_in_dim %cst_382, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1314 = stablehlo.maximum %1313, %1312 : tensor<256xf32>
      %1315 = stablehlo.compare  EQ, %1312, %1314,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_383 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1316 = stablehlo.broadcast_in_dim %cst_383, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_384 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1317 = stablehlo.broadcast_in_dim %cst_384, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1318 = stablehlo.select %1315, %1316, %1317 : tensor<256xi1>, tensor<256xf32>
      %cst_385 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1319 = stablehlo.broadcast_in_dim %cst_385, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1320 = stablehlo.compare  EQ, %1319, %1314,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_386 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1321 = stablehlo.broadcast_in_dim %cst_386, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_387 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1322 = stablehlo.broadcast_in_dim %cst_387, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1323 = stablehlo.select %1320, %1321, %1322 : tensor<256xi1>, tensor<256xf32>
      %1324 = stablehlo.divide %1318, %1323 : tensor<256xf32>
      %cst_388 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1325 = stablehlo.broadcast_in_dim %cst_388, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1326 = stablehlo.multiply %1325, %arg580 : tensor<256xf32>
      %cst_389 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1327 = stablehlo.broadcast_in_dim %cst_389, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1328 = stablehlo.multiply %1327, %1306 : tensor<256xf32>
      %1329 = stablehlo.add %1326, %1328 : tensor<256xf32>
      %cst_390 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1330 = stablehlo.broadcast_in_dim %cst_390, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1331 = stablehlo.multiply %1330, %arg581 : tensor<256xf32>
      %cst_391 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1332 = stablehlo.broadcast_in_dim %cst_391, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1333 = stablehlo.multiply %1332, %1314 : tensor<256xf32>
      %1334 = stablehlo.add %1331, %1333 : tensor<256xf32>
      %1335 = stablehlo.broadcast_in_dim %1306, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1336 = stablehlo.broadcast_in_dim %1314, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1337 = stablehlo.convert %1288 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1338 = stablehlo.broadcast_in_dim %1335, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1339 = stablehlo.subtract %1337, %1338 : tensor<256x14x14x256xf32>
      %cst_392 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1340 = stablehlo.broadcast_in_dim %cst_392, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1341 = stablehlo.add %1336, %1340 : tensor<1x1x1x256xf32>
      %1342 = stablehlo.rsqrt %1341 : tensor<1x1x1x256xf32>
      %1343 = stablehlo.divide %1342, %1341 : tensor<1x1x1x256xf32>
      %cst_393 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1344 = stablehlo.broadcast_in_dim %cst_393, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1345 = stablehlo.multiply %1344, %1343 : tensor<1x1x1x256xf32>
      %1346 = stablehlo.reshape %arg393 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1347 = stablehlo.multiply %1342, %1346 : tensor<1x1x1x256xf32>
      %1348 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1349 = stablehlo.multiply %1339, %1348 : tensor<256x14x14x256xf32>
      %1350 = stablehlo.reshape %arg392 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1351 = stablehlo.broadcast_in_dim %1350, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1352 = stablehlo.add %1349, %1351 : tensor<256x14x14x256xf32>
      %1353 = stablehlo.convert %1352 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1354 = stablehlo.add %1353, %1286 : tensor<256x14x14x256xf16>
      %1355 = func.call @relu_142(%1354) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_394 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1356 = stablehlo.broadcast_in_dim %cst_394, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1357 = stablehlo.compare  GT, %1354, %1356,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1358 = stablehlo.convert %arg398 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1359 = stablehlo.convolution(%1355, %1358) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1360 = stablehlo.convert %1359 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1361 = stablehlo.multiply %1360, %1360 : tensor<256x14x14x256xf32>
      %cst_395 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1362 = stablehlo.broadcast_in_dim %cst_395, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1363 = stablehlo.multiply %1362, %1360 : tensor<256x14x14x256xf32>
      %cst_396 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1364 = stablehlo.reduce(%1360 init: %cst_396) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_397 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1365 = stablehlo.broadcast_in_dim %cst_397, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1366 = stablehlo.divide %1364, %1365 : tensor<256xf32>
      %cst_398 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1367 = stablehlo.reduce(%1361 init: %cst_398) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_399 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1368 = stablehlo.broadcast_in_dim %cst_399, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1369 = stablehlo.divide %1367, %1368 : tensor<256xf32>
      %1370 = stablehlo.broadcast_in_dim %1366, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1371 = stablehlo.broadcast_in_dim %1369, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1372 = stablehlo.concatenate %1370, %1371, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1373 = "stablehlo.all_reduce"(%1372) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_400 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1374 = stablehlo.broadcast_in_dim %cst_400, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1375 = stablehlo.divide %1373, %1374 : tensor<2x256xf32>
      %1376 = stablehlo.slice %1375 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1377 = stablehlo.reshape %1376 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1378 = stablehlo.slice %1375 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1379 = stablehlo.reshape %1378 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1380 = stablehlo.multiply %1377, %1377 : tensor<256xf32>
      %cst_401 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1381 = stablehlo.broadcast_in_dim %cst_401, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1382 = stablehlo.multiply %1381, %1377 : tensor<256xf32>
      %1383 = stablehlo.subtract %1379, %1380 : tensor<256xf32>
      %cst_402 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1384 = stablehlo.broadcast_in_dim %cst_402, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1385 = stablehlo.maximum %1384, %1383 : tensor<256xf32>
      %1386 = stablehlo.compare  EQ, %1383, %1385,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_403 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1387 = stablehlo.broadcast_in_dim %cst_403, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_404 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1388 = stablehlo.broadcast_in_dim %cst_404, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1389 = stablehlo.select %1386, %1387, %1388 : tensor<256xi1>, tensor<256xf32>
      %cst_405 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1390 = stablehlo.broadcast_in_dim %cst_405, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1391 = stablehlo.compare  EQ, %1390, %1385,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_406 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1392 = stablehlo.broadcast_in_dim %cst_406, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_407 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1393 = stablehlo.broadcast_in_dim %cst_407, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1394 = stablehlo.select %1391, %1392, %1393 : tensor<256xi1>, tensor<256xf32>
      %1395 = stablehlo.divide %1389, %1394 : tensor<256xf32>
      %cst_408 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1396 = stablehlo.broadcast_in_dim %cst_408, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1397 = stablehlo.multiply %1396, %arg582 : tensor<256xf32>
      %cst_409 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1398 = stablehlo.broadcast_in_dim %cst_409, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1399 = stablehlo.multiply %1398, %1377 : tensor<256xf32>
      %1400 = stablehlo.add %1397, %1399 : tensor<256xf32>
      %cst_410 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1401 = stablehlo.broadcast_in_dim %cst_410, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1402 = stablehlo.multiply %1401, %arg583 : tensor<256xf32>
      %cst_411 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1403 = stablehlo.broadcast_in_dim %cst_411, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1404 = stablehlo.multiply %1403, %1385 : tensor<256xf32>
      %1405 = stablehlo.add %1402, %1404 : tensor<256xf32>
      %1406 = stablehlo.broadcast_in_dim %1377, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1407 = stablehlo.broadcast_in_dim %1385, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1408 = stablehlo.convert %1359 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1409 = stablehlo.broadcast_in_dim %1406, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1410 = stablehlo.subtract %1408, %1409 : tensor<256x14x14x256xf32>
      %cst_412 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1411 = stablehlo.broadcast_in_dim %cst_412, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1412 = stablehlo.add %1407, %1411 : tensor<1x1x1x256xf32>
      %1413 = stablehlo.rsqrt %1412 : tensor<1x1x1x256xf32>
      %1414 = stablehlo.divide %1413, %1412 : tensor<1x1x1x256xf32>
      %cst_413 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1415 = stablehlo.broadcast_in_dim %cst_413, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1416 = stablehlo.multiply %1415, %1414 : tensor<1x1x1x256xf32>
      %1417 = stablehlo.reshape %arg395 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1418 = stablehlo.multiply %1413, %1417 : tensor<1x1x1x256xf32>
      %1419 = stablehlo.broadcast_in_dim %1418, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1420 = stablehlo.multiply %1410, %1419 : tensor<256x14x14x256xf32>
      %1421 = stablehlo.reshape %arg394 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1422 = stablehlo.broadcast_in_dim %1421, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1423 = stablehlo.add %1420, %1422 : tensor<256x14x14x256xf32>
      %1424 = stablehlo.convert %1423 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1425 = func.call @relu_142(%1424) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_414 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1426 = stablehlo.broadcast_in_dim %cst_414, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1427 = stablehlo.compare  GT, %1424, %1426,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1428 = stablehlo.convert %arg399 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1429 = stablehlo.convolution(%1425, %1428) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1430 = stablehlo.convert %1429 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1431 = stablehlo.multiply %1430, %1430 : tensor<256x14x14x256xf32>
      %cst_415 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1432 = stablehlo.broadcast_in_dim %cst_415, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1433 = stablehlo.multiply %1432, %1430 : tensor<256x14x14x256xf32>
      %cst_416 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1434 = stablehlo.reduce(%1430 init: %cst_416) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_417 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1435 = stablehlo.broadcast_in_dim %cst_417, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1436 = stablehlo.divide %1434, %1435 : tensor<256xf32>
      %cst_418 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1437 = stablehlo.reduce(%1431 init: %cst_418) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_419 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1438 = stablehlo.broadcast_in_dim %cst_419, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1439 = stablehlo.divide %1437, %1438 : tensor<256xf32>
      %1440 = stablehlo.broadcast_in_dim %1436, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1441 = stablehlo.broadcast_in_dim %1439, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1442 = stablehlo.concatenate %1440, %1441, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1443 = "stablehlo.all_reduce"(%1442) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_420 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1444 = stablehlo.broadcast_in_dim %cst_420, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1445 = stablehlo.divide %1443, %1444 : tensor<2x256xf32>
      %1446 = stablehlo.slice %1445 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1447 = stablehlo.reshape %1446 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1448 = stablehlo.slice %1445 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1449 = stablehlo.reshape %1448 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1450 = stablehlo.multiply %1447, %1447 : tensor<256xf32>
      %cst_421 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1451 = stablehlo.broadcast_in_dim %cst_421, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1452 = stablehlo.multiply %1451, %1447 : tensor<256xf32>
      %1453 = stablehlo.subtract %1449, %1450 : tensor<256xf32>
      %cst_422 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1454 = stablehlo.broadcast_in_dim %cst_422, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1455 = stablehlo.maximum %1454, %1453 : tensor<256xf32>
      %1456 = stablehlo.compare  EQ, %1453, %1455,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_423 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1457 = stablehlo.broadcast_in_dim %cst_423, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_424 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1458 = stablehlo.broadcast_in_dim %cst_424, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1459 = stablehlo.select %1456, %1457, %1458 : tensor<256xi1>, tensor<256xf32>
      %cst_425 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1460 = stablehlo.broadcast_in_dim %cst_425, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1461 = stablehlo.compare  EQ, %1460, %1455,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_426 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1462 = stablehlo.broadcast_in_dim %cst_426, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_427 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1463 = stablehlo.broadcast_in_dim %cst_427, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1464 = stablehlo.select %1461, %1462, %1463 : tensor<256xi1>, tensor<256xf32>
      %1465 = stablehlo.divide %1459, %1464 : tensor<256xf32>
      %cst_428 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1466 = stablehlo.broadcast_in_dim %cst_428, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1467 = stablehlo.multiply %1466, %arg584 : tensor<256xf32>
      %cst_429 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1468 = stablehlo.broadcast_in_dim %cst_429, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1469 = stablehlo.multiply %1468, %1447 : tensor<256xf32>
      %1470 = stablehlo.add %1467, %1469 : tensor<256xf32>
      %cst_430 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1471 = stablehlo.broadcast_in_dim %cst_430, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1472 = stablehlo.multiply %1471, %arg585 : tensor<256xf32>
      %cst_431 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1473 = stablehlo.broadcast_in_dim %cst_431, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1474 = stablehlo.multiply %1473, %1455 : tensor<256xf32>
      %1475 = stablehlo.add %1472, %1474 : tensor<256xf32>
      %1476 = stablehlo.broadcast_in_dim %1447, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1477 = stablehlo.broadcast_in_dim %1455, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1478 = stablehlo.convert %1429 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1479 = stablehlo.broadcast_in_dim %1476, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1480 = stablehlo.subtract %1478, %1479 : tensor<256x14x14x256xf32>
      %cst_432 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1481 = stablehlo.broadcast_in_dim %cst_432, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1482 = stablehlo.add %1477, %1481 : tensor<1x1x1x256xf32>
      %1483 = stablehlo.rsqrt %1482 : tensor<1x1x1x256xf32>
      %1484 = stablehlo.divide %1483, %1482 : tensor<1x1x1x256xf32>
      %cst_433 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1485 = stablehlo.broadcast_in_dim %cst_433, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1486 = stablehlo.multiply %1485, %1484 : tensor<1x1x1x256xf32>
      %1487 = stablehlo.reshape %arg397 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1488 = stablehlo.multiply %1483, %1487 : tensor<1x1x1x256xf32>
      %1489 = stablehlo.broadcast_in_dim %1488, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1490 = stablehlo.multiply %1480, %1489 : tensor<256x14x14x256xf32>
      %1491 = stablehlo.reshape %arg396 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1492 = stablehlo.broadcast_in_dim %1491, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1493 = stablehlo.add %1490, %1492 : tensor<256x14x14x256xf32>
      %1494 = stablehlo.convert %1493 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1495 = stablehlo.add %1355, %1494 : tensor<256x14x14x256xf16>
      %1496 = func.call @relu_142(%1495) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_434 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1497 = stablehlo.broadcast_in_dim %cst_434, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1498 = stablehlo.compare  GT, %1495, %1497,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1499 = stablehlo.convert %arg404 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1500 = stablehlo.convolution(%1496, %1499) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1501 = stablehlo.convert %1500 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1502 = stablehlo.multiply %1501, %1501 : tensor<256x14x14x256xf32>
      %cst_435 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1503 = stablehlo.broadcast_in_dim %cst_435, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1504 = stablehlo.multiply %1503, %1501 : tensor<256x14x14x256xf32>
      %cst_436 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1505 = stablehlo.reduce(%1501 init: %cst_436) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_437 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1506 = stablehlo.broadcast_in_dim %cst_437, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1507 = stablehlo.divide %1505, %1506 : tensor<256xf32>
      %cst_438 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1508 = stablehlo.reduce(%1502 init: %cst_438) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_439 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1509 = stablehlo.broadcast_in_dim %cst_439, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1510 = stablehlo.divide %1508, %1509 : tensor<256xf32>
      %1511 = stablehlo.broadcast_in_dim %1507, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1512 = stablehlo.broadcast_in_dim %1510, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1513 = stablehlo.concatenate %1511, %1512, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1514 = "stablehlo.all_reduce"(%1513) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_440 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1515 = stablehlo.broadcast_in_dim %cst_440, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1516 = stablehlo.divide %1514, %1515 : tensor<2x256xf32>
      %1517 = stablehlo.slice %1516 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1518 = stablehlo.reshape %1517 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1519 = stablehlo.slice %1516 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1520 = stablehlo.reshape %1519 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1521 = stablehlo.multiply %1518, %1518 : tensor<256xf32>
      %cst_441 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1522 = stablehlo.broadcast_in_dim %cst_441, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1523 = stablehlo.multiply %1522, %1518 : tensor<256xf32>
      %1524 = stablehlo.subtract %1520, %1521 : tensor<256xf32>
      %cst_442 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1525 = stablehlo.broadcast_in_dim %cst_442, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1526 = stablehlo.maximum %1525, %1524 : tensor<256xf32>
      %1527 = stablehlo.compare  EQ, %1524, %1526,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_443 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1528 = stablehlo.broadcast_in_dim %cst_443, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_444 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1529 = stablehlo.broadcast_in_dim %cst_444, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1530 = stablehlo.select %1527, %1528, %1529 : tensor<256xi1>, tensor<256xf32>
      %cst_445 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1531 = stablehlo.broadcast_in_dim %cst_445, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1532 = stablehlo.compare  EQ, %1531, %1526,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_446 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1533 = stablehlo.broadcast_in_dim %cst_446, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_447 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1534 = stablehlo.broadcast_in_dim %cst_447, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1535 = stablehlo.select %1532, %1533, %1534 : tensor<256xi1>, tensor<256xf32>
      %1536 = stablehlo.divide %1530, %1535 : tensor<256xf32>
      %cst_448 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1537 = stablehlo.broadcast_in_dim %cst_448, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1538 = stablehlo.multiply %1537, %arg586 : tensor<256xf32>
      %cst_449 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1539 = stablehlo.broadcast_in_dim %cst_449, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1540 = stablehlo.multiply %1539, %1518 : tensor<256xf32>
      %1541 = stablehlo.add %1538, %1540 : tensor<256xf32>
      %cst_450 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1542 = stablehlo.broadcast_in_dim %cst_450, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1543 = stablehlo.multiply %1542, %arg587 : tensor<256xf32>
      %cst_451 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1544 = stablehlo.broadcast_in_dim %cst_451, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1545 = stablehlo.multiply %1544, %1526 : tensor<256xf32>
      %1546 = stablehlo.add %1543, %1545 : tensor<256xf32>
      %1547 = stablehlo.broadcast_in_dim %1518, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1548 = stablehlo.broadcast_in_dim %1526, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1549 = stablehlo.convert %1500 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1550 = stablehlo.broadcast_in_dim %1547, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1551 = stablehlo.subtract %1549, %1550 : tensor<256x14x14x256xf32>
      %cst_452 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1552 = stablehlo.broadcast_in_dim %cst_452, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1553 = stablehlo.add %1548, %1552 : tensor<1x1x1x256xf32>
      %1554 = stablehlo.rsqrt %1553 : tensor<1x1x1x256xf32>
      %1555 = stablehlo.divide %1554, %1553 : tensor<1x1x1x256xf32>
      %cst_453 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1556 = stablehlo.broadcast_in_dim %cst_453, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1557 = stablehlo.multiply %1556, %1555 : tensor<1x1x1x256xf32>
      %1558 = stablehlo.reshape %arg401 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1559 = stablehlo.multiply %1554, %1558 : tensor<1x1x1x256xf32>
      %1560 = stablehlo.broadcast_in_dim %1559, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1561 = stablehlo.multiply %1551, %1560 : tensor<256x14x14x256xf32>
      %1562 = stablehlo.reshape %arg400 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1563 = stablehlo.broadcast_in_dim %1562, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1564 = stablehlo.add %1561, %1563 : tensor<256x14x14x256xf32>
      %1565 = stablehlo.convert %1564 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1566 = func.call @relu_142(%1565) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_454 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1567 = stablehlo.broadcast_in_dim %cst_454, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1568 = stablehlo.compare  GT, %1565, %1567,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1569 = stablehlo.convert %arg405 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1570 = stablehlo.convolution(%1566, %1569) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1571 = stablehlo.convert %1570 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1572 = stablehlo.multiply %1571, %1571 : tensor<256x14x14x256xf32>
      %cst_455 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1573 = stablehlo.broadcast_in_dim %cst_455, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1574 = stablehlo.multiply %1573, %1571 : tensor<256x14x14x256xf32>
      %cst_456 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1575 = stablehlo.reduce(%1571 init: %cst_456) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_457 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1576 = stablehlo.broadcast_in_dim %cst_457, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1577 = stablehlo.divide %1575, %1576 : tensor<256xf32>
      %cst_458 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1578 = stablehlo.reduce(%1572 init: %cst_458) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_459 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1579 = stablehlo.broadcast_in_dim %cst_459, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1580 = stablehlo.divide %1578, %1579 : tensor<256xf32>
      %1581 = stablehlo.broadcast_in_dim %1577, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1582 = stablehlo.broadcast_in_dim %1580, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1583 = stablehlo.concatenate %1581, %1582, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1584 = "stablehlo.all_reduce"(%1583) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_460 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1585 = stablehlo.broadcast_in_dim %cst_460, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1586 = stablehlo.divide %1584, %1585 : tensor<2x256xf32>
      %1587 = stablehlo.slice %1586 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1588 = stablehlo.reshape %1587 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1589 = stablehlo.slice %1586 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1590 = stablehlo.reshape %1589 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1591 = stablehlo.multiply %1588, %1588 : tensor<256xf32>
      %cst_461 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1592 = stablehlo.broadcast_in_dim %cst_461, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1593 = stablehlo.multiply %1592, %1588 : tensor<256xf32>
      %1594 = stablehlo.subtract %1590, %1591 : tensor<256xf32>
      %cst_462 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1595 = stablehlo.broadcast_in_dim %cst_462, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1596 = stablehlo.maximum %1595, %1594 : tensor<256xf32>
      %1597 = stablehlo.compare  EQ, %1594, %1596,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_463 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1598 = stablehlo.broadcast_in_dim %cst_463, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_464 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1599 = stablehlo.broadcast_in_dim %cst_464, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1600 = stablehlo.select %1597, %1598, %1599 : tensor<256xi1>, tensor<256xf32>
      %cst_465 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1601 = stablehlo.broadcast_in_dim %cst_465, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1602 = stablehlo.compare  EQ, %1601, %1596,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_466 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1603 = stablehlo.broadcast_in_dim %cst_466, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_467 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1604 = stablehlo.broadcast_in_dim %cst_467, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1605 = stablehlo.select %1602, %1603, %1604 : tensor<256xi1>, tensor<256xf32>
      %1606 = stablehlo.divide %1600, %1605 : tensor<256xf32>
      %cst_468 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1607 = stablehlo.broadcast_in_dim %cst_468, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1608 = stablehlo.multiply %1607, %arg588 : tensor<256xf32>
      %cst_469 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1609 = stablehlo.broadcast_in_dim %cst_469, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1610 = stablehlo.multiply %1609, %1588 : tensor<256xf32>
      %1611 = stablehlo.add %1608, %1610 : tensor<256xf32>
      %cst_470 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1612 = stablehlo.broadcast_in_dim %cst_470, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1613 = stablehlo.multiply %1612, %arg589 : tensor<256xf32>
      %cst_471 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1614 = stablehlo.broadcast_in_dim %cst_471, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1615 = stablehlo.multiply %1614, %1596 : tensor<256xf32>
      %1616 = stablehlo.add %1613, %1615 : tensor<256xf32>
      %1617 = stablehlo.broadcast_in_dim %1588, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1618 = stablehlo.broadcast_in_dim %1596, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1619 = stablehlo.convert %1570 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1620 = stablehlo.broadcast_in_dim %1617, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1621 = stablehlo.subtract %1619, %1620 : tensor<256x14x14x256xf32>
      %cst_472 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1622 = stablehlo.broadcast_in_dim %cst_472, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1623 = stablehlo.add %1618, %1622 : tensor<1x1x1x256xf32>
      %1624 = stablehlo.rsqrt %1623 : tensor<1x1x1x256xf32>
      %1625 = stablehlo.divide %1624, %1623 : tensor<1x1x1x256xf32>
      %cst_473 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1626 = stablehlo.broadcast_in_dim %cst_473, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1627 = stablehlo.multiply %1626, %1625 : tensor<1x1x1x256xf32>
      %1628 = stablehlo.reshape %arg403 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1629 = stablehlo.multiply %1624, %1628 : tensor<1x1x1x256xf32>
      %1630 = stablehlo.broadcast_in_dim %1629, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1631 = stablehlo.multiply %1621, %1630 : tensor<256x14x14x256xf32>
      %1632 = stablehlo.reshape %arg402 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1633 = stablehlo.broadcast_in_dim %1632, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1634 = stablehlo.add %1631, %1633 : tensor<256x14x14x256xf32>
      %1635 = stablehlo.convert %1634 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1636 = stablehlo.add %1496, %1635 : tensor<256x14x14x256xf16>
      %1637 = func.call @relu_142(%1636) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_474 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1638 = stablehlo.broadcast_in_dim %cst_474, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1639 = stablehlo.compare  GT, %1636, %1638,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1640 = stablehlo.convert %arg317 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1641 = stablehlo.convolution(%1637, %1640) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1642 = stablehlo.convert %1641 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1643 = stablehlo.multiply %1642, %1642 : tensor<256x14x14x256xf32>
      %cst_475 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1644 = stablehlo.broadcast_in_dim %cst_475, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1645 = stablehlo.multiply %1644, %1642 : tensor<256x14x14x256xf32>
      %cst_476 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1646 = stablehlo.reduce(%1642 init: %cst_476) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_477 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1647 = stablehlo.broadcast_in_dim %cst_477, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1648 = stablehlo.divide %1646, %1647 : tensor<256xf32>
      %cst_478 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1649 = stablehlo.reduce(%1643 init: %cst_478) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_479 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1650 = stablehlo.broadcast_in_dim %cst_479, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1651 = stablehlo.divide %1649, %1650 : tensor<256xf32>
      %1652 = stablehlo.broadcast_in_dim %1648, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1653 = stablehlo.broadcast_in_dim %1651, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1654 = stablehlo.concatenate %1652, %1653, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1655 = "stablehlo.all_reduce"(%1654) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_480 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1656 = stablehlo.broadcast_in_dim %cst_480, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1657 = stablehlo.divide %1655, %1656 : tensor<2x256xf32>
      %1658 = stablehlo.slice %1657 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1659 = stablehlo.reshape %1658 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1660 = stablehlo.slice %1657 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1661 = stablehlo.reshape %1660 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1662 = stablehlo.multiply %1659, %1659 : tensor<256xf32>
      %cst_481 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1663 = stablehlo.broadcast_in_dim %cst_481, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1664 = stablehlo.multiply %1663, %1659 : tensor<256xf32>
      %1665 = stablehlo.subtract %1661, %1662 : tensor<256xf32>
      %cst_482 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1666 = stablehlo.broadcast_in_dim %cst_482, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1667 = stablehlo.maximum %1666, %1665 : tensor<256xf32>
      %1668 = stablehlo.compare  EQ, %1665, %1667,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_483 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1669 = stablehlo.broadcast_in_dim %cst_483, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_484 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1670 = stablehlo.broadcast_in_dim %cst_484, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1671 = stablehlo.select %1668, %1669, %1670 : tensor<256xi1>, tensor<256xf32>
      %cst_485 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1672 = stablehlo.broadcast_in_dim %cst_485, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1673 = stablehlo.compare  EQ, %1672, %1667,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_486 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1674 = stablehlo.broadcast_in_dim %cst_486, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_487 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1675 = stablehlo.broadcast_in_dim %cst_487, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1676 = stablehlo.select %1673, %1674, %1675 : tensor<256xi1>, tensor<256xf32>
      %1677 = stablehlo.divide %1671, %1676 : tensor<256xf32>
      %cst_488 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1678 = stablehlo.broadcast_in_dim %cst_488, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1679 = stablehlo.multiply %1678, %arg528 : tensor<256xf32>
      %cst_489 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1680 = stablehlo.broadcast_in_dim %cst_489, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1681 = stablehlo.multiply %1680, %1659 : tensor<256xf32>
      %1682 = stablehlo.add %1679, %1681 : tensor<256xf32>
      %cst_490 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1683 = stablehlo.broadcast_in_dim %cst_490, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1684 = stablehlo.multiply %1683, %arg529 : tensor<256xf32>
      %cst_491 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1685 = stablehlo.broadcast_in_dim %cst_491, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1686 = stablehlo.multiply %1685, %1667 : tensor<256xf32>
      %1687 = stablehlo.add %1684, %1686 : tensor<256xf32>
      %1688 = stablehlo.broadcast_in_dim %1659, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1689 = stablehlo.broadcast_in_dim %1667, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1690 = stablehlo.convert %1641 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1691 = stablehlo.broadcast_in_dim %1688, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1692 = stablehlo.subtract %1690, %1691 : tensor<256x14x14x256xf32>
      %cst_492 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1693 = stablehlo.broadcast_in_dim %cst_492, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1694 = stablehlo.add %1689, %1693 : tensor<1x1x1x256xf32>
      %1695 = stablehlo.rsqrt %1694 : tensor<1x1x1x256xf32>
      %1696 = stablehlo.divide %1695, %1694 : tensor<1x1x1x256xf32>
      %cst_493 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1697 = stablehlo.broadcast_in_dim %cst_493, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1698 = stablehlo.multiply %1697, %1696 : tensor<1x1x1x256xf32>
      %1699 = stablehlo.reshape %arg314 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1700 = stablehlo.multiply %1695, %1699 : tensor<1x1x1x256xf32>
      %1701 = stablehlo.broadcast_in_dim %1700, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1702 = stablehlo.multiply %1692, %1701 : tensor<256x14x14x256xf32>
      %1703 = stablehlo.reshape %arg313 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1704 = stablehlo.broadcast_in_dim %1703, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1705 = stablehlo.add %1702, %1704 : tensor<256x14x14x256xf32>
      %1706 = stablehlo.convert %1705 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1707 = func.call @relu_142(%1706) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_494 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1708 = stablehlo.broadcast_in_dim %cst_494, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1709 = stablehlo.compare  GT, %1706, %1708,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1710 = stablehlo.convert %arg318 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1711 = stablehlo.convolution(%1707, %1710) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1712 = stablehlo.convert %1711 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1713 = stablehlo.multiply %1712, %1712 : tensor<256x14x14x256xf32>
      %cst_495 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1714 = stablehlo.broadcast_in_dim %cst_495, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1715 = stablehlo.multiply %1714, %1712 : tensor<256x14x14x256xf32>
      %cst_496 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1716 = stablehlo.reduce(%1712 init: %cst_496) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_497 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1717 = stablehlo.broadcast_in_dim %cst_497, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1718 = stablehlo.divide %1716, %1717 : tensor<256xf32>
      %cst_498 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1719 = stablehlo.reduce(%1713 init: %cst_498) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_499 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1720 = stablehlo.broadcast_in_dim %cst_499, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1721 = stablehlo.divide %1719, %1720 : tensor<256xf32>
      %1722 = stablehlo.broadcast_in_dim %1718, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1723 = stablehlo.broadcast_in_dim %1721, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1724 = stablehlo.concatenate %1722, %1723, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1725 = "stablehlo.all_reduce"(%1724) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_500 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1726 = stablehlo.broadcast_in_dim %cst_500, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1727 = stablehlo.divide %1725, %1726 : tensor<2x256xf32>
      %1728 = stablehlo.slice %1727 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1729 = stablehlo.reshape %1728 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1730 = stablehlo.slice %1727 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1731 = stablehlo.reshape %1730 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1732 = stablehlo.multiply %1729, %1729 : tensor<256xf32>
      %cst_501 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1733 = stablehlo.broadcast_in_dim %cst_501, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1734 = stablehlo.multiply %1733, %1729 : tensor<256xf32>
      %1735 = stablehlo.subtract %1731, %1732 : tensor<256xf32>
      %cst_502 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1736 = stablehlo.broadcast_in_dim %cst_502, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1737 = stablehlo.maximum %1736, %1735 : tensor<256xf32>
      %1738 = stablehlo.compare  EQ, %1735, %1737,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_503 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1739 = stablehlo.broadcast_in_dim %cst_503, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_504 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1740 = stablehlo.broadcast_in_dim %cst_504, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1741 = stablehlo.select %1738, %1739, %1740 : tensor<256xi1>, tensor<256xf32>
      %cst_505 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1742 = stablehlo.broadcast_in_dim %cst_505, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1743 = stablehlo.compare  EQ, %1742, %1737,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_506 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1744 = stablehlo.broadcast_in_dim %cst_506, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_507 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1745 = stablehlo.broadcast_in_dim %cst_507, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1746 = stablehlo.select %1743, %1744, %1745 : tensor<256xi1>, tensor<256xf32>
      %1747 = stablehlo.divide %1741, %1746 : tensor<256xf32>
      %cst_508 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1748 = stablehlo.broadcast_in_dim %cst_508, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1749 = stablehlo.multiply %1748, %arg530 : tensor<256xf32>
      %cst_509 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1750 = stablehlo.broadcast_in_dim %cst_509, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1751 = stablehlo.multiply %1750, %1729 : tensor<256xf32>
      %1752 = stablehlo.add %1749, %1751 : tensor<256xf32>
      %cst_510 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1753 = stablehlo.broadcast_in_dim %cst_510, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1754 = stablehlo.multiply %1753, %arg531 : tensor<256xf32>
      %cst_511 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1755 = stablehlo.broadcast_in_dim %cst_511, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1756 = stablehlo.multiply %1755, %1737 : tensor<256xf32>
      %1757 = stablehlo.add %1754, %1756 : tensor<256xf32>
      %1758 = stablehlo.broadcast_in_dim %1729, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1759 = stablehlo.broadcast_in_dim %1737, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1760 = stablehlo.convert %1711 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1761 = stablehlo.broadcast_in_dim %1758, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1762 = stablehlo.subtract %1760, %1761 : tensor<256x14x14x256xf32>
      %cst_512 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1763 = stablehlo.broadcast_in_dim %cst_512, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1764 = stablehlo.add %1759, %1763 : tensor<1x1x1x256xf32>
      %1765 = stablehlo.rsqrt %1764 : tensor<1x1x1x256xf32>
      %1766 = stablehlo.divide %1765, %1764 : tensor<1x1x1x256xf32>
      %cst_513 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1767 = stablehlo.broadcast_in_dim %cst_513, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1768 = stablehlo.multiply %1767, %1766 : tensor<1x1x1x256xf32>
      %1769 = stablehlo.reshape %arg316 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1770 = stablehlo.multiply %1765, %1769 : tensor<1x1x1x256xf32>
      %1771 = stablehlo.broadcast_in_dim %1770, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1772 = stablehlo.multiply %1762, %1771 : tensor<256x14x14x256xf32>
      %1773 = stablehlo.reshape %arg315 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1774 = stablehlo.broadcast_in_dim %1773, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1775 = stablehlo.add %1772, %1774 : tensor<256x14x14x256xf32>
      %1776 = stablehlo.convert %1775 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1777 = stablehlo.add %1637, %1776 : tensor<256x14x14x256xf16>
      %1778 = func.call @relu_142(%1777) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_514 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1779 = stablehlo.broadcast_in_dim %cst_514, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1780 = stablehlo.compare  GT, %1777, %1779,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1781 = stablehlo.convert %arg323 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1782 = stablehlo.convolution(%1778, %1781) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1783 = stablehlo.convert %1782 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1784 = stablehlo.multiply %1783, %1783 : tensor<256x14x14x256xf32>
      %cst_515 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1785 = stablehlo.broadcast_in_dim %cst_515, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1786 = stablehlo.multiply %1785, %1783 : tensor<256x14x14x256xf32>
      %cst_516 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1787 = stablehlo.reduce(%1783 init: %cst_516) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_517 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1788 = stablehlo.broadcast_in_dim %cst_517, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1789 = stablehlo.divide %1787, %1788 : tensor<256xf32>
      %cst_518 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1790 = stablehlo.reduce(%1784 init: %cst_518) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_519 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1791 = stablehlo.broadcast_in_dim %cst_519, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1792 = stablehlo.divide %1790, %1791 : tensor<256xf32>
      %1793 = stablehlo.broadcast_in_dim %1789, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1794 = stablehlo.broadcast_in_dim %1792, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1795 = stablehlo.concatenate %1793, %1794, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1796 = "stablehlo.all_reduce"(%1795) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_520 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1797 = stablehlo.broadcast_in_dim %cst_520, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1798 = stablehlo.divide %1796, %1797 : tensor<2x256xf32>
      %1799 = stablehlo.slice %1798 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1800 = stablehlo.reshape %1799 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1801 = stablehlo.slice %1798 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1802 = stablehlo.reshape %1801 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1803 = stablehlo.multiply %1800, %1800 : tensor<256xf32>
      %cst_521 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1804 = stablehlo.broadcast_in_dim %cst_521, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1805 = stablehlo.multiply %1804, %1800 : tensor<256xf32>
      %1806 = stablehlo.subtract %1802, %1803 : tensor<256xf32>
      %cst_522 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1807 = stablehlo.broadcast_in_dim %cst_522, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1808 = stablehlo.maximum %1807, %1806 : tensor<256xf32>
      %1809 = stablehlo.compare  EQ, %1806, %1808,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_523 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1810 = stablehlo.broadcast_in_dim %cst_523, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_524 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1811 = stablehlo.broadcast_in_dim %cst_524, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1812 = stablehlo.select %1809, %1810, %1811 : tensor<256xi1>, tensor<256xf32>
      %cst_525 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1813 = stablehlo.broadcast_in_dim %cst_525, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1814 = stablehlo.compare  EQ, %1813, %1808,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_526 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1815 = stablehlo.broadcast_in_dim %cst_526, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_527 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1816 = stablehlo.broadcast_in_dim %cst_527, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1817 = stablehlo.select %1814, %1815, %1816 : tensor<256xi1>, tensor<256xf32>
      %1818 = stablehlo.divide %1812, %1817 : tensor<256xf32>
      %cst_528 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1819 = stablehlo.broadcast_in_dim %cst_528, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1820 = stablehlo.multiply %1819, %arg532 : tensor<256xf32>
      %cst_529 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1821 = stablehlo.broadcast_in_dim %cst_529, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1822 = stablehlo.multiply %1821, %1800 : tensor<256xf32>
      %1823 = stablehlo.add %1820, %1822 : tensor<256xf32>
      %cst_530 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1824 = stablehlo.broadcast_in_dim %cst_530, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1825 = stablehlo.multiply %1824, %arg533 : tensor<256xf32>
      %cst_531 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1826 = stablehlo.broadcast_in_dim %cst_531, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1827 = stablehlo.multiply %1826, %1808 : tensor<256xf32>
      %1828 = stablehlo.add %1825, %1827 : tensor<256xf32>
      %1829 = stablehlo.broadcast_in_dim %1800, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1830 = stablehlo.broadcast_in_dim %1808, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1831 = stablehlo.convert %1782 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1832 = stablehlo.broadcast_in_dim %1829, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1833 = stablehlo.subtract %1831, %1832 : tensor<256x14x14x256xf32>
      %cst_532 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1834 = stablehlo.broadcast_in_dim %cst_532, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1835 = stablehlo.add %1830, %1834 : tensor<1x1x1x256xf32>
      %1836 = stablehlo.rsqrt %1835 : tensor<1x1x1x256xf32>
      %1837 = stablehlo.divide %1836, %1835 : tensor<1x1x1x256xf32>
      %cst_533 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1838 = stablehlo.broadcast_in_dim %cst_533, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1839 = stablehlo.multiply %1838, %1837 : tensor<1x1x1x256xf32>
      %1840 = stablehlo.reshape %arg320 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1841 = stablehlo.multiply %1836, %1840 : tensor<1x1x1x256xf32>
      %1842 = stablehlo.broadcast_in_dim %1841, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1843 = stablehlo.multiply %1833, %1842 : tensor<256x14x14x256xf32>
      %1844 = stablehlo.reshape %arg319 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1845 = stablehlo.broadcast_in_dim %1844, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1846 = stablehlo.add %1843, %1845 : tensor<256x14x14x256xf32>
      %1847 = stablehlo.convert %1846 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1848 = func.call @relu_142(%1847) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_534 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1849 = stablehlo.broadcast_in_dim %cst_534, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1850 = stablehlo.compare  GT, %1847, %1849,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1851 = stablehlo.convert %arg324 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1852 = stablehlo.convolution(%1848, %1851) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1853 = stablehlo.convert %1852 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1854 = stablehlo.multiply %1853, %1853 : tensor<256x14x14x256xf32>
      %cst_535 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1855 = stablehlo.broadcast_in_dim %cst_535, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1856 = stablehlo.multiply %1855, %1853 : tensor<256x14x14x256xf32>
      %cst_536 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1857 = stablehlo.reduce(%1853 init: %cst_536) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_537 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1858 = stablehlo.broadcast_in_dim %cst_537, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1859 = stablehlo.divide %1857, %1858 : tensor<256xf32>
      %cst_538 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1860 = stablehlo.reduce(%1854 init: %cst_538) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_539 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1861 = stablehlo.broadcast_in_dim %cst_539, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1862 = stablehlo.divide %1860, %1861 : tensor<256xf32>
      %1863 = stablehlo.broadcast_in_dim %1859, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1864 = stablehlo.broadcast_in_dim %1862, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1865 = stablehlo.concatenate %1863, %1864, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1866 = "stablehlo.all_reduce"(%1865) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_540 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1867 = stablehlo.broadcast_in_dim %cst_540, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1868 = stablehlo.divide %1866, %1867 : tensor<2x256xf32>
      %1869 = stablehlo.slice %1868 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1870 = stablehlo.reshape %1869 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1871 = stablehlo.slice %1868 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1872 = stablehlo.reshape %1871 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1873 = stablehlo.multiply %1870, %1870 : tensor<256xf32>
      %cst_541 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1874 = stablehlo.broadcast_in_dim %cst_541, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1875 = stablehlo.multiply %1874, %1870 : tensor<256xf32>
      %1876 = stablehlo.subtract %1872, %1873 : tensor<256xf32>
      %cst_542 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1877 = stablehlo.broadcast_in_dim %cst_542, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1878 = stablehlo.maximum %1877, %1876 : tensor<256xf32>
      %1879 = stablehlo.compare  EQ, %1876, %1878,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_543 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1880 = stablehlo.broadcast_in_dim %cst_543, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_544 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1881 = stablehlo.broadcast_in_dim %cst_544, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1882 = stablehlo.select %1879, %1880, %1881 : tensor<256xi1>, tensor<256xf32>
      %cst_545 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1883 = stablehlo.broadcast_in_dim %cst_545, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1884 = stablehlo.compare  EQ, %1883, %1878,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_546 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1885 = stablehlo.broadcast_in_dim %cst_546, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_547 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1886 = stablehlo.broadcast_in_dim %cst_547, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1887 = stablehlo.select %1884, %1885, %1886 : tensor<256xi1>, tensor<256xf32>
      %1888 = stablehlo.divide %1882, %1887 : tensor<256xf32>
      %cst_548 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1889 = stablehlo.broadcast_in_dim %cst_548, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1890 = stablehlo.multiply %1889, %arg534 : tensor<256xf32>
      %cst_549 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1891 = stablehlo.broadcast_in_dim %cst_549, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1892 = stablehlo.multiply %1891, %1870 : tensor<256xf32>
      %1893 = stablehlo.add %1890, %1892 : tensor<256xf32>
      %cst_550 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1894 = stablehlo.broadcast_in_dim %cst_550, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1895 = stablehlo.multiply %1894, %arg535 : tensor<256xf32>
      %cst_551 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1896 = stablehlo.broadcast_in_dim %cst_551, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1897 = stablehlo.multiply %1896, %1878 : tensor<256xf32>
      %1898 = stablehlo.add %1895, %1897 : tensor<256xf32>
      %1899 = stablehlo.broadcast_in_dim %1870, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1900 = stablehlo.broadcast_in_dim %1878, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1901 = stablehlo.convert %1852 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1902 = stablehlo.broadcast_in_dim %1899, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1903 = stablehlo.subtract %1901, %1902 : tensor<256x14x14x256xf32>
      %cst_552 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1904 = stablehlo.broadcast_in_dim %cst_552, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1905 = stablehlo.add %1900, %1904 : tensor<1x1x1x256xf32>
      %1906 = stablehlo.rsqrt %1905 : tensor<1x1x1x256xf32>
      %1907 = stablehlo.divide %1906, %1905 : tensor<1x1x1x256xf32>
      %cst_553 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1908 = stablehlo.broadcast_in_dim %cst_553, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1909 = stablehlo.multiply %1908, %1907 : tensor<1x1x1x256xf32>
      %1910 = stablehlo.reshape %arg322 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1911 = stablehlo.multiply %1906, %1910 : tensor<1x1x1x256xf32>
      %1912 = stablehlo.broadcast_in_dim %1911, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1913 = stablehlo.multiply %1903, %1912 : tensor<256x14x14x256xf32>
      %1914 = stablehlo.reshape %arg321 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1915 = stablehlo.broadcast_in_dim %1914, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1916 = stablehlo.add %1913, %1915 : tensor<256x14x14x256xf32>
      %1917 = stablehlo.convert %1916 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1918 = stablehlo.add %1778, %1917 : tensor<256x14x14x256xf16>
      %1919 = func.call @relu_142(%1918) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_554 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1920 = stablehlo.broadcast_in_dim %cst_554, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1921 = stablehlo.compare  GT, %1918, %1920,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1922 = stablehlo.convert %arg329 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1923 = stablehlo.convolution(%1919, %1922) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1924 = stablehlo.convert %1923 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1925 = stablehlo.multiply %1924, %1924 : tensor<256x14x14x256xf32>
      %cst_555 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1926 = stablehlo.broadcast_in_dim %cst_555, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1927 = stablehlo.multiply %1926, %1924 : tensor<256x14x14x256xf32>
      %cst_556 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1928 = stablehlo.reduce(%1924 init: %cst_556) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_557 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1929 = stablehlo.broadcast_in_dim %cst_557, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1930 = stablehlo.divide %1928, %1929 : tensor<256xf32>
      %cst_558 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1931 = stablehlo.reduce(%1925 init: %cst_558) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_559 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1932 = stablehlo.broadcast_in_dim %cst_559, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1933 = stablehlo.divide %1931, %1932 : tensor<256xf32>
      %1934 = stablehlo.broadcast_in_dim %1930, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1935 = stablehlo.broadcast_in_dim %1933, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1936 = stablehlo.concatenate %1934, %1935, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1937 = "stablehlo.all_reduce"(%1936) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_560 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1938 = stablehlo.broadcast_in_dim %cst_560, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1939 = stablehlo.divide %1937, %1938 : tensor<2x256xf32>
      %1940 = stablehlo.slice %1939 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1941 = stablehlo.reshape %1940 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1942 = stablehlo.slice %1939 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1943 = stablehlo.reshape %1942 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1944 = stablehlo.multiply %1941, %1941 : tensor<256xf32>
      %cst_561 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1945 = stablehlo.broadcast_in_dim %cst_561, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1946 = stablehlo.multiply %1945, %1941 : tensor<256xf32>
      %1947 = stablehlo.subtract %1943, %1944 : tensor<256xf32>
      %cst_562 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1948 = stablehlo.broadcast_in_dim %cst_562, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1949 = stablehlo.maximum %1948, %1947 : tensor<256xf32>
      %1950 = stablehlo.compare  EQ, %1947, %1949,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_563 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1951 = stablehlo.broadcast_in_dim %cst_563, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_564 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1952 = stablehlo.broadcast_in_dim %cst_564, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1953 = stablehlo.select %1950, %1951, %1952 : tensor<256xi1>, tensor<256xf32>
      %cst_565 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1954 = stablehlo.broadcast_in_dim %cst_565, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1955 = stablehlo.compare  EQ, %1954, %1949,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_566 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1956 = stablehlo.broadcast_in_dim %cst_566, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_567 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1957 = stablehlo.broadcast_in_dim %cst_567, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1958 = stablehlo.select %1955, %1956, %1957 : tensor<256xi1>, tensor<256xf32>
      %1959 = stablehlo.divide %1953, %1958 : tensor<256xf32>
      %cst_568 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1960 = stablehlo.broadcast_in_dim %cst_568, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1961 = stablehlo.multiply %1960, %arg536 : tensor<256xf32>
      %cst_569 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1962 = stablehlo.broadcast_in_dim %cst_569, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1963 = stablehlo.multiply %1962, %1941 : tensor<256xf32>
      %1964 = stablehlo.add %1961, %1963 : tensor<256xf32>
      %cst_570 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1965 = stablehlo.broadcast_in_dim %cst_570, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1966 = stablehlo.multiply %1965, %arg537 : tensor<256xf32>
      %cst_571 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1967 = stablehlo.broadcast_in_dim %cst_571, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1968 = stablehlo.multiply %1967, %1949 : tensor<256xf32>
      %1969 = stablehlo.add %1966, %1968 : tensor<256xf32>
      %1970 = stablehlo.broadcast_in_dim %1941, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1971 = stablehlo.broadcast_in_dim %1949, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1972 = stablehlo.convert %1923 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1973 = stablehlo.broadcast_in_dim %1970, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1974 = stablehlo.subtract %1972, %1973 : tensor<256x14x14x256xf32>
      %cst_572 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1975 = stablehlo.broadcast_in_dim %cst_572, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1976 = stablehlo.add %1971, %1975 : tensor<1x1x1x256xf32>
      %1977 = stablehlo.rsqrt %1976 : tensor<1x1x1x256xf32>
      %1978 = stablehlo.divide %1977, %1976 : tensor<1x1x1x256xf32>
      %cst_573 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1979 = stablehlo.broadcast_in_dim %cst_573, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1980 = stablehlo.multiply %1979, %1978 : tensor<1x1x1x256xf32>
      %1981 = stablehlo.reshape %arg326 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1982 = stablehlo.multiply %1977, %1981 : tensor<1x1x1x256xf32>
      %1983 = stablehlo.broadcast_in_dim %1982, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1984 = stablehlo.multiply %1974, %1983 : tensor<256x14x14x256xf32>
      %1985 = stablehlo.reshape %arg325 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1986 = stablehlo.broadcast_in_dim %1985, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1987 = stablehlo.add %1984, %1986 : tensor<256x14x14x256xf32>
      %1988 = stablehlo.convert %1987 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1989 = func.call @relu_142(%1988) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_574 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1990 = stablehlo.broadcast_in_dim %cst_574, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1991 = stablehlo.compare  GT, %1988, %1990,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1992 = stablehlo.convert %arg330 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1993 = stablehlo.convolution(%1989, %1992) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1994 = stablehlo.convert %1993 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1995 = stablehlo.multiply %1994, %1994 : tensor<256x14x14x256xf32>
      %cst_575 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1996 = stablehlo.broadcast_in_dim %cst_575, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1997 = stablehlo.multiply %1996, %1994 : tensor<256x14x14x256xf32>
      %cst_576 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1998 = stablehlo.reduce(%1994 init: %cst_576) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_577 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1999 = stablehlo.broadcast_in_dim %cst_577, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2000 = stablehlo.divide %1998, %1999 : tensor<256xf32>
      %cst_578 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2001 = stablehlo.reduce(%1995 init: %cst_578) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_579 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %2002 = stablehlo.broadcast_in_dim %cst_579, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2003 = stablehlo.divide %2001, %2002 : tensor<256xf32>
      %2004 = stablehlo.broadcast_in_dim %2000, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %2005 = stablehlo.broadcast_in_dim %2003, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %2006 = stablehlo.concatenate %2004, %2005, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %2007 = "stablehlo.all_reduce"(%2006) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_580 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2008 = stablehlo.broadcast_in_dim %cst_580, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %2009 = stablehlo.divide %2007, %2008 : tensor<2x256xf32>
      %2010 = stablehlo.slice %2009 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %2011 = stablehlo.reshape %2010 : (tensor<1x256xf32>) -> tensor<256xf32>
      %2012 = stablehlo.slice %2009 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %2013 = stablehlo.reshape %2012 : (tensor<1x256xf32>) -> tensor<256xf32>
      %2014 = stablehlo.multiply %2011, %2011 : tensor<256xf32>
      %cst_581 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2015 = stablehlo.broadcast_in_dim %cst_581, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2016 = stablehlo.multiply %2015, %2011 : tensor<256xf32>
      %2017 = stablehlo.subtract %2013, %2014 : tensor<256xf32>
      %cst_582 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2018 = stablehlo.broadcast_in_dim %cst_582, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2019 = stablehlo.maximum %2018, %2017 : tensor<256xf32>
      %2020 = stablehlo.compare  EQ, %2017, %2019,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_583 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2021 = stablehlo.broadcast_in_dim %cst_583, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_584 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2022 = stablehlo.broadcast_in_dim %cst_584, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2023 = stablehlo.select %2020, %2021, %2022 : tensor<256xi1>, tensor<256xf32>
      %cst_585 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2024 = stablehlo.broadcast_in_dim %cst_585, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2025 = stablehlo.compare  EQ, %2024, %2019,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_586 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2026 = stablehlo.broadcast_in_dim %cst_586, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_587 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2027 = stablehlo.broadcast_in_dim %cst_587, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2028 = stablehlo.select %2025, %2026, %2027 : tensor<256xi1>, tensor<256xf32>
      %2029 = stablehlo.divide %2023, %2028 : tensor<256xf32>
      %cst_588 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2030 = stablehlo.broadcast_in_dim %cst_588, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2031 = stablehlo.multiply %2030, %arg538 : tensor<256xf32>
      %cst_589 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2032 = stablehlo.broadcast_in_dim %cst_589, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2033 = stablehlo.multiply %2032, %2011 : tensor<256xf32>
      %2034 = stablehlo.add %2031, %2033 : tensor<256xf32>
      %cst_590 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2035 = stablehlo.broadcast_in_dim %cst_590, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2036 = stablehlo.multiply %2035, %arg539 : tensor<256xf32>
      %cst_591 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2037 = stablehlo.broadcast_in_dim %cst_591, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2038 = stablehlo.multiply %2037, %2019 : tensor<256xf32>
      %2039 = stablehlo.add %2036, %2038 : tensor<256xf32>
      %2040 = stablehlo.broadcast_in_dim %2011, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2041 = stablehlo.broadcast_in_dim %2019, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2042 = stablehlo.convert %1993 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2043 = stablehlo.broadcast_in_dim %2040, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2044 = stablehlo.subtract %2042, %2043 : tensor<256x14x14x256xf32>
      %cst_592 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2045 = stablehlo.broadcast_in_dim %cst_592, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2046 = stablehlo.add %2041, %2045 : tensor<1x1x1x256xf32>
      %2047 = stablehlo.rsqrt %2046 : tensor<1x1x1x256xf32>
      %2048 = stablehlo.divide %2047, %2046 : tensor<1x1x1x256xf32>
      %cst_593 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %2049 = stablehlo.broadcast_in_dim %cst_593, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2050 = stablehlo.multiply %2049, %2048 : tensor<1x1x1x256xf32>
      %2051 = stablehlo.reshape %arg328 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2052 = stablehlo.multiply %2047, %2051 : tensor<1x1x1x256xf32>
      %2053 = stablehlo.broadcast_in_dim %2052, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2054 = stablehlo.multiply %2044, %2053 : tensor<256x14x14x256xf32>
      %2055 = stablehlo.reshape %arg327 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2056 = stablehlo.broadcast_in_dim %2055, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2057 = stablehlo.add %2054, %2056 : tensor<256x14x14x256xf32>
      %2058 = stablehlo.convert %2057 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2059 = stablehlo.add %1919, %2058 : tensor<256x14x14x256xf16>
      %2060 = func.call @relu_142(%2059) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_594 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2061 = stablehlo.broadcast_in_dim %cst_594, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %2062 = stablehlo.compare  GT, %2059, %2061,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %2063 = stablehlo.convert %arg335 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf16>
      %2064 = stablehlo.convolution(%2060, %2063) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x512xf16>) -> tensor<256x7x7x512xf16>
      %2065 = stablehlo.convert %2064 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2066 = stablehlo.multiply %2065, %2065 : tensor<256x7x7x512xf32>
      %cst_595 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2067 = stablehlo.broadcast_in_dim %cst_595, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %2068 = stablehlo.multiply %2067, %2065 : tensor<256x7x7x512xf32>
      %cst_596 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2069 = stablehlo.reduce(%2065 init: %cst_596) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_597 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2070 = stablehlo.broadcast_in_dim %cst_597, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2071 = stablehlo.divide %2069, %2070 : tensor<512xf32>
      %cst_598 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2072 = stablehlo.reduce(%2066 init: %cst_598) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_599 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2073 = stablehlo.broadcast_in_dim %cst_599, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2074 = stablehlo.divide %2072, %2073 : tensor<512xf32>
      %2075 = stablehlo.broadcast_in_dim %2071, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2076 = stablehlo.broadcast_in_dim %2074, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2077 = stablehlo.concatenate %2075, %2076, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %2078 = "stablehlo.all_reduce"(%2077) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_600 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2079 = stablehlo.broadcast_in_dim %cst_600, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2080 = stablehlo.divide %2078, %2079 : tensor<2x512xf32>
      %2081 = stablehlo.slice %2080 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2082 = stablehlo.reshape %2081 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2083 = stablehlo.slice %2080 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2084 = stablehlo.reshape %2083 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2085 = stablehlo.multiply %2082, %2082 : tensor<512xf32>
      %cst_601 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2086 = stablehlo.broadcast_in_dim %cst_601, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2087 = stablehlo.multiply %2086, %2082 : tensor<512xf32>
      %2088 = stablehlo.subtract %2084, %2085 : tensor<512xf32>
      %cst_602 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2089 = stablehlo.broadcast_in_dim %cst_602, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2090 = stablehlo.maximum %2089, %2088 : tensor<512xf32>
      %2091 = stablehlo.compare  EQ, %2088, %2090,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_603 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2092 = stablehlo.broadcast_in_dim %cst_603, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_604 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2093 = stablehlo.broadcast_in_dim %cst_604, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2094 = stablehlo.select %2091, %2092, %2093 : tensor<512xi1>, tensor<512xf32>
      %cst_605 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2095 = stablehlo.broadcast_in_dim %cst_605, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2096 = stablehlo.compare  EQ, %2095, %2090,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_606 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2097 = stablehlo.broadcast_in_dim %cst_606, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_607 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2098 = stablehlo.broadcast_in_dim %cst_607, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2099 = stablehlo.select %2096, %2097, %2098 : tensor<512xi1>, tensor<512xf32>
      %2100 = stablehlo.divide %2094, %2099 : tensor<512xf32>
      %cst_608 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2101 = stablehlo.broadcast_in_dim %cst_608, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2102 = stablehlo.multiply %2101, %arg540 : tensor<512xf32>
      %cst_609 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2103 = stablehlo.broadcast_in_dim %cst_609, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2104 = stablehlo.multiply %2103, %2082 : tensor<512xf32>
      %2105 = stablehlo.add %2102, %2104 : tensor<512xf32>
      %cst_610 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2106 = stablehlo.broadcast_in_dim %cst_610, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2107 = stablehlo.multiply %2106, %arg541 : tensor<512xf32>
      %cst_611 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2108 = stablehlo.broadcast_in_dim %cst_611, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2109 = stablehlo.multiply %2108, %2090 : tensor<512xf32>
      %2110 = stablehlo.add %2107, %2109 : tensor<512xf32>
      %2111 = stablehlo.broadcast_in_dim %2082, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2112 = stablehlo.broadcast_in_dim %2090, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2113 = stablehlo.convert %2064 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2114 = stablehlo.broadcast_in_dim %2111, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2115 = stablehlo.subtract %2113, %2114 : tensor<256x7x7x512xf32>
      %cst_612 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2116 = stablehlo.broadcast_in_dim %cst_612, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2117 = stablehlo.add %2112, %2116 : tensor<1x1x1x512xf32>
      %2118 = stablehlo.rsqrt %2117 : tensor<1x1x1x512xf32>
      %2119 = stablehlo.divide %2118, %2117 : tensor<1x1x1x512xf32>
      %cst_613 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %2120 = stablehlo.broadcast_in_dim %cst_613, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2121 = stablehlo.multiply %2120, %2119 : tensor<1x1x1x512xf32>
      %2122 = stablehlo.reshape %arg332 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2123 = stablehlo.multiply %2118, %2122 : tensor<1x1x1x512xf32>
      %2124 = stablehlo.broadcast_in_dim %2123, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2125 = stablehlo.multiply %2115, %2124 : tensor<256x7x7x512xf32>
      %2126 = stablehlo.reshape %arg331 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2127 = stablehlo.broadcast_in_dim %2126, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2128 = stablehlo.add %2125, %2127 : tensor<256x7x7x512xf32>
      %2129 = stablehlo.convert %2128 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2130 = func.call @relu_198(%2129) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_614 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2131 = stablehlo.broadcast_in_dim %cst_614, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2132 = stablehlo.compare  GT, %2129, %2131,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %2133 = stablehlo.convert %arg336 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %2134 = stablehlo.convolution(%2130, %2133) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %2135 = stablehlo.convert %2134 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2136 = stablehlo.multiply %2135, %2135 : tensor<256x7x7x512xf32>
      %cst_615 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2137 = stablehlo.broadcast_in_dim %cst_615, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %2138 = stablehlo.multiply %2137, %2135 : tensor<256x7x7x512xf32>
      %cst_616 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2139 = stablehlo.reduce(%2135 init: %cst_616) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_617 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2140 = stablehlo.broadcast_in_dim %cst_617, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2141 = stablehlo.divide %2139, %2140 : tensor<512xf32>
      %cst_618 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2142 = stablehlo.reduce(%2136 init: %cst_618) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_619 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2143 = stablehlo.broadcast_in_dim %cst_619, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2144 = stablehlo.divide %2142, %2143 : tensor<512xf32>
      %2145 = stablehlo.broadcast_in_dim %2141, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2146 = stablehlo.broadcast_in_dim %2144, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2147 = stablehlo.concatenate %2145, %2146, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %2148 = "stablehlo.all_reduce"(%2147) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_620 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2149 = stablehlo.broadcast_in_dim %cst_620, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2150 = stablehlo.divide %2148, %2149 : tensor<2x512xf32>
      %2151 = stablehlo.slice %2150 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2152 = stablehlo.reshape %2151 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2153 = stablehlo.slice %2150 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2154 = stablehlo.reshape %2153 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2155 = stablehlo.multiply %2152, %2152 : tensor<512xf32>
      %cst_621 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2156 = stablehlo.broadcast_in_dim %cst_621, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2157 = stablehlo.multiply %2156, %2152 : tensor<512xf32>
      %2158 = stablehlo.subtract %2154, %2155 : tensor<512xf32>
      %cst_622 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2159 = stablehlo.broadcast_in_dim %cst_622, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2160 = stablehlo.maximum %2159, %2158 : tensor<512xf32>
      %2161 = stablehlo.compare  EQ, %2158, %2160,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_623 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2162 = stablehlo.broadcast_in_dim %cst_623, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_624 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2163 = stablehlo.broadcast_in_dim %cst_624, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2164 = stablehlo.select %2161, %2162, %2163 : tensor<512xi1>, tensor<512xf32>
      %cst_625 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2165 = stablehlo.broadcast_in_dim %cst_625, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2166 = stablehlo.compare  EQ, %2165, %2160,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_626 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2167 = stablehlo.broadcast_in_dim %cst_626, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_627 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2168 = stablehlo.broadcast_in_dim %cst_627, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2169 = stablehlo.select %2166, %2167, %2168 : tensor<512xi1>, tensor<512xf32>
      %2170 = stablehlo.divide %2164, %2169 : tensor<512xf32>
      %cst_628 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2171 = stablehlo.broadcast_in_dim %cst_628, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2172 = stablehlo.multiply %2171, %arg542 : tensor<512xf32>
      %cst_629 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2173 = stablehlo.broadcast_in_dim %cst_629, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2174 = stablehlo.multiply %2173, %2152 : tensor<512xf32>
      %2175 = stablehlo.add %2172, %2174 : tensor<512xf32>
      %cst_630 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2176 = stablehlo.broadcast_in_dim %cst_630, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2177 = stablehlo.multiply %2176, %arg543 : tensor<512xf32>
      %cst_631 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2178 = stablehlo.broadcast_in_dim %cst_631, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2179 = stablehlo.multiply %2178, %2160 : tensor<512xf32>
      %2180 = stablehlo.add %2177, %2179 : tensor<512xf32>
      %2181 = stablehlo.broadcast_in_dim %2152, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2182 = stablehlo.broadcast_in_dim %2160, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2183 = stablehlo.convert %2134 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2184 = stablehlo.broadcast_in_dim %2181, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2185 = stablehlo.subtract %2183, %2184 : tensor<256x7x7x512xf32>
      %cst_632 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2186 = stablehlo.broadcast_in_dim %cst_632, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2187 = stablehlo.add %2182, %2186 : tensor<1x1x1x512xf32>
      %2188 = stablehlo.rsqrt %2187 : tensor<1x1x1x512xf32>
      %2189 = stablehlo.divide %2188, %2187 : tensor<1x1x1x512xf32>
      %cst_633 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %2190 = stablehlo.broadcast_in_dim %cst_633, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2191 = stablehlo.multiply %2190, %2189 : tensor<1x1x1x512xf32>
      %2192 = stablehlo.reshape %arg334 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2193 = stablehlo.multiply %2188, %2192 : tensor<1x1x1x512xf32>
      %2194 = stablehlo.broadcast_in_dim %2193, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2195 = stablehlo.multiply %2185, %2194 : tensor<256x7x7x512xf32>
      %2196 = stablehlo.reshape %arg333 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2197 = stablehlo.broadcast_in_dim %2196, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2198 = stablehlo.add %2195, %2197 : tensor<256x7x7x512xf32>
      %2199 = stablehlo.convert %2198 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2200 = stablehlo.convert %arg337 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf16>
      %2201 = stablehlo.convolution(%2060, %2200) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x512xf16>) -> tensor<256x7x7x512xf16>
      %2202 = stablehlo.convert %2201 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2203 = stablehlo.multiply %2202, %2202 : tensor<256x7x7x512xf32>
      %cst_634 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2204 = stablehlo.broadcast_in_dim %cst_634, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %2205 = stablehlo.multiply %2204, %2202 : tensor<256x7x7x512xf32>
      %cst_635 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2206 = stablehlo.reduce(%2202 init: %cst_635) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_636 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2207 = stablehlo.broadcast_in_dim %cst_636, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2208 = stablehlo.divide %2206, %2207 : tensor<512xf32>
      %cst_637 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2209 = stablehlo.reduce(%2203 init: %cst_637) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_638 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2210 = stablehlo.broadcast_in_dim %cst_638, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2211 = stablehlo.divide %2209, %2210 : tensor<512xf32>
      %2212 = stablehlo.broadcast_in_dim %2208, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2213 = stablehlo.broadcast_in_dim %2211, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2214 = stablehlo.concatenate %2212, %2213, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %2215 = "stablehlo.all_reduce"(%2214) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_639 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2216 = stablehlo.broadcast_in_dim %cst_639, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2217 = stablehlo.divide %2215, %2216 : tensor<2x512xf32>
      %2218 = stablehlo.slice %2217 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2219 = stablehlo.reshape %2218 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2220 = stablehlo.slice %2217 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2221 = stablehlo.reshape %2220 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2222 = stablehlo.multiply %2219, %2219 : tensor<512xf32>
      %cst_640 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2223 = stablehlo.broadcast_in_dim %cst_640, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2224 = stablehlo.multiply %2223, %2219 : tensor<512xf32>
      %2225 = stablehlo.subtract %2221, %2222 : tensor<512xf32>
      %cst_641 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2226 = stablehlo.broadcast_in_dim %cst_641, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2227 = stablehlo.maximum %2226, %2225 : tensor<512xf32>
      %2228 = stablehlo.compare  EQ, %2225, %2227,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_642 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2229 = stablehlo.broadcast_in_dim %cst_642, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_643 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2230 = stablehlo.broadcast_in_dim %cst_643, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2231 = stablehlo.select %2228, %2229, %2230 : tensor<512xi1>, tensor<512xf32>
      %cst_644 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2232 = stablehlo.broadcast_in_dim %cst_644, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2233 = stablehlo.compare  EQ, %2232, %2227,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_645 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2234 = stablehlo.broadcast_in_dim %cst_645, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_646 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2235 = stablehlo.broadcast_in_dim %cst_646, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2236 = stablehlo.select %2233, %2234, %2235 : tensor<512xi1>, tensor<512xf32>
      %2237 = stablehlo.divide %2231, %2236 : tensor<512xf32>
      %cst_647 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2238 = stablehlo.broadcast_in_dim %cst_647, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2239 = stablehlo.multiply %2238, %arg544 : tensor<512xf32>
      %cst_648 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2240 = stablehlo.broadcast_in_dim %cst_648, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2241 = stablehlo.multiply %2240, %2219 : tensor<512xf32>
      %2242 = stablehlo.add %2239, %2241 : tensor<512xf32>
      %cst_649 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2243 = stablehlo.broadcast_in_dim %cst_649, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2244 = stablehlo.multiply %2243, %arg545 : tensor<512xf32>
      %cst_650 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2245 = stablehlo.broadcast_in_dim %cst_650, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2246 = stablehlo.multiply %2245, %2227 : tensor<512xf32>
      %2247 = stablehlo.add %2244, %2246 : tensor<512xf32>
      %2248 = stablehlo.broadcast_in_dim %2219, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2249 = stablehlo.broadcast_in_dim %2227, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2250 = stablehlo.convert %2201 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2251 = stablehlo.broadcast_in_dim %2248, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2252 = stablehlo.subtract %2250, %2251 : tensor<256x7x7x512xf32>
      %cst_651 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2253 = stablehlo.broadcast_in_dim %cst_651, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2254 = stablehlo.add %2249, %2253 : tensor<1x1x1x512xf32>
      %2255 = stablehlo.rsqrt %2254 : tensor<1x1x1x512xf32>
      %2256 = stablehlo.divide %2255, %2254 : tensor<1x1x1x512xf32>
      %cst_652 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %2257 = stablehlo.broadcast_in_dim %cst_652, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2258 = stablehlo.multiply %2257, %2256 : tensor<1x1x1x512xf32>
      %2259 = stablehlo.reshape %arg339 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2260 = stablehlo.multiply %2255, %2259 : tensor<1x1x1x512xf32>
      %2261 = stablehlo.broadcast_in_dim %2260, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2262 = stablehlo.multiply %2252, %2261 : tensor<256x7x7x512xf32>
      %2263 = stablehlo.reshape %arg338 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2264 = stablehlo.broadcast_in_dim %2263, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2265 = stablehlo.add %2262, %2264 : tensor<256x7x7x512xf32>
      %2266 = stablehlo.convert %2265 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2267 = stablehlo.add %2266, %2199 : tensor<256x7x7x512xf16>
      %2268 = func.call @relu_198(%2267) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_653 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2269 = stablehlo.broadcast_in_dim %cst_653, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2270 = stablehlo.compare  GT, %2267, %2269,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %2271 = stablehlo.convert %arg344 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %2272 = stablehlo.convolution(%2268, %2271) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %2273 = stablehlo.convert %2272 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2274 = stablehlo.multiply %2273, %2273 : tensor<256x7x7x512xf32>
      %cst_654 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2275 = stablehlo.broadcast_in_dim %cst_654, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %2276 = stablehlo.multiply %2275, %2273 : tensor<256x7x7x512xf32>
      %cst_655 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2277 = stablehlo.reduce(%2273 init: %cst_655) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_656 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2278 = stablehlo.broadcast_in_dim %cst_656, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2279 = stablehlo.divide %2277, %2278 : tensor<512xf32>
      %cst_657 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2280 = stablehlo.reduce(%2274 init: %cst_657) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_658 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2281 = stablehlo.broadcast_in_dim %cst_658, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2282 = stablehlo.divide %2280, %2281 : tensor<512xf32>
      %2283 = stablehlo.broadcast_in_dim %2279, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2284 = stablehlo.broadcast_in_dim %2282, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2285 = stablehlo.concatenate %2283, %2284, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %2286 = "stablehlo.all_reduce"(%2285) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_659 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2287 = stablehlo.broadcast_in_dim %cst_659, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2288 = stablehlo.divide %2286, %2287 : tensor<2x512xf32>
      %2289 = stablehlo.slice %2288 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2290 = stablehlo.reshape %2289 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2291 = stablehlo.slice %2288 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2292 = stablehlo.reshape %2291 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2293 = stablehlo.multiply %2290, %2290 : tensor<512xf32>
      %cst_660 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2294 = stablehlo.broadcast_in_dim %cst_660, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2295 = stablehlo.multiply %2294, %2290 : tensor<512xf32>
      %2296 = stablehlo.subtract %2292, %2293 : tensor<512xf32>
      %cst_661 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2297 = stablehlo.broadcast_in_dim %cst_661, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2298 = stablehlo.maximum %2297, %2296 : tensor<512xf32>
      %2299 = stablehlo.compare  EQ, %2296, %2298,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_662 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2300 = stablehlo.broadcast_in_dim %cst_662, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_663 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2301 = stablehlo.broadcast_in_dim %cst_663, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2302 = stablehlo.select %2299, %2300, %2301 : tensor<512xi1>, tensor<512xf32>
      %cst_664 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2303 = stablehlo.broadcast_in_dim %cst_664, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2304 = stablehlo.compare  EQ, %2303, %2298,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_665 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2305 = stablehlo.broadcast_in_dim %cst_665, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_666 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2306 = stablehlo.broadcast_in_dim %cst_666, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2307 = stablehlo.select %2304, %2305, %2306 : tensor<512xi1>, tensor<512xf32>
      %2308 = stablehlo.divide %2302, %2307 : tensor<512xf32>
      %cst_667 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2309 = stablehlo.broadcast_in_dim %cst_667, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2310 = stablehlo.multiply %2309, %arg546 : tensor<512xf32>
      %cst_668 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2311 = stablehlo.broadcast_in_dim %cst_668, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2312 = stablehlo.multiply %2311, %2290 : tensor<512xf32>
      %2313 = stablehlo.add %2310, %2312 : tensor<512xf32>
      %cst_669 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2314 = stablehlo.broadcast_in_dim %cst_669, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2315 = stablehlo.multiply %2314, %arg547 : tensor<512xf32>
      %cst_670 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2316 = stablehlo.broadcast_in_dim %cst_670, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2317 = stablehlo.multiply %2316, %2298 : tensor<512xf32>
      %2318 = stablehlo.add %2315, %2317 : tensor<512xf32>
      %2319 = stablehlo.broadcast_in_dim %2290, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2320 = stablehlo.broadcast_in_dim %2298, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2321 = stablehlo.convert %2272 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2322 = stablehlo.broadcast_in_dim %2319, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2323 = stablehlo.subtract %2321, %2322 : tensor<256x7x7x512xf32>
      %cst_671 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2324 = stablehlo.broadcast_in_dim %cst_671, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2325 = stablehlo.add %2320, %2324 : tensor<1x1x1x512xf32>
      %2326 = stablehlo.rsqrt %2325 : tensor<1x1x1x512xf32>
      %2327 = stablehlo.divide %2326, %2325 : tensor<1x1x1x512xf32>
      %cst_672 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %2328 = stablehlo.broadcast_in_dim %cst_672, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2329 = stablehlo.multiply %2328, %2327 : tensor<1x1x1x512xf32>
      %2330 = stablehlo.reshape %arg341 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2331 = stablehlo.multiply %2326, %2330 : tensor<1x1x1x512xf32>
      %2332 = stablehlo.broadcast_in_dim %2331, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2333 = stablehlo.multiply %2323, %2332 : tensor<256x7x7x512xf32>
      %2334 = stablehlo.reshape %arg340 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2335 = stablehlo.broadcast_in_dim %2334, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2336 = stablehlo.add %2333, %2335 : tensor<256x7x7x512xf32>
      %2337 = stablehlo.convert %2336 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2338 = func.call @relu_198(%2337) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_673 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2339 = stablehlo.broadcast_in_dim %cst_673, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2340 = stablehlo.compare  GT, %2337, %2339,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %2341 = stablehlo.convert %arg345 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %2342 = stablehlo.convolution(%2338, %2341) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %2343 = stablehlo.convert %2342 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2344 = stablehlo.multiply %2343, %2343 : tensor<256x7x7x512xf32>
      %cst_674 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2345 = stablehlo.broadcast_in_dim %cst_674, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %2346 = stablehlo.multiply %2345, %2343 : tensor<256x7x7x512xf32>
      %cst_675 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2347 = stablehlo.reduce(%2343 init: %cst_675) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_676 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2348 = stablehlo.broadcast_in_dim %cst_676, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2349 = stablehlo.divide %2347, %2348 : tensor<512xf32>
      %cst_677 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2350 = stablehlo.reduce(%2344 init: %cst_677) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_678 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2351 = stablehlo.broadcast_in_dim %cst_678, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2352 = stablehlo.divide %2350, %2351 : tensor<512xf32>
      %2353 = stablehlo.broadcast_in_dim %2349, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2354 = stablehlo.broadcast_in_dim %2352, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2355 = stablehlo.concatenate %2353, %2354, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %2356 = "stablehlo.all_reduce"(%2355) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_679 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2357 = stablehlo.broadcast_in_dim %cst_679, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2358 = stablehlo.divide %2356, %2357 : tensor<2x512xf32>
      %2359 = stablehlo.slice %2358 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2360 = stablehlo.reshape %2359 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2361 = stablehlo.slice %2358 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2362 = stablehlo.reshape %2361 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2363 = stablehlo.multiply %2360, %2360 : tensor<512xf32>
      %cst_680 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2364 = stablehlo.broadcast_in_dim %cst_680, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2365 = stablehlo.multiply %2364, %2360 : tensor<512xf32>
      %2366 = stablehlo.subtract %2362, %2363 : tensor<512xf32>
      %cst_681 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2367 = stablehlo.broadcast_in_dim %cst_681, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2368 = stablehlo.maximum %2367, %2366 : tensor<512xf32>
      %2369 = stablehlo.compare  EQ, %2366, %2368,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_682 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2370 = stablehlo.broadcast_in_dim %cst_682, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_683 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2371 = stablehlo.broadcast_in_dim %cst_683, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2372 = stablehlo.select %2369, %2370, %2371 : tensor<512xi1>, tensor<512xf32>
      %cst_684 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2373 = stablehlo.broadcast_in_dim %cst_684, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2374 = stablehlo.compare  EQ, %2373, %2368,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_685 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2375 = stablehlo.broadcast_in_dim %cst_685, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_686 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2376 = stablehlo.broadcast_in_dim %cst_686, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2377 = stablehlo.select %2374, %2375, %2376 : tensor<512xi1>, tensor<512xf32>
      %2378 = stablehlo.divide %2372, %2377 : tensor<512xf32>
      %cst_687 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2379 = stablehlo.broadcast_in_dim %cst_687, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2380 = stablehlo.multiply %2379, %arg548 : tensor<512xf32>
      %cst_688 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2381 = stablehlo.broadcast_in_dim %cst_688, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2382 = stablehlo.multiply %2381, %2360 : tensor<512xf32>
      %2383 = stablehlo.add %2380, %2382 : tensor<512xf32>
      %cst_689 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2384 = stablehlo.broadcast_in_dim %cst_689, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2385 = stablehlo.multiply %2384, %arg549 : tensor<512xf32>
      %cst_690 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2386 = stablehlo.broadcast_in_dim %cst_690, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2387 = stablehlo.multiply %2386, %2368 : tensor<512xf32>
      %2388 = stablehlo.add %2385, %2387 : tensor<512xf32>
      %2389 = stablehlo.broadcast_in_dim %2360, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2390 = stablehlo.broadcast_in_dim %2368, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2391 = stablehlo.convert %2342 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2392 = stablehlo.broadcast_in_dim %2389, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2393 = stablehlo.subtract %2391, %2392 : tensor<256x7x7x512xf32>
      %cst_691 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2394 = stablehlo.broadcast_in_dim %cst_691, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2395 = stablehlo.add %2390, %2394 : tensor<1x1x1x512xf32>
      %2396 = stablehlo.rsqrt %2395 : tensor<1x1x1x512xf32>
      %2397 = stablehlo.divide %2396, %2395 : tensor<1x1x1x512xf32>
      %cst_692 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %2398 = stablehlo.broadcast_in_dim %cst_692, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2399 = stablehlo.multiply %2398, %2397 : tensor<1x1x1x512xf32>
      %2400 = stablehlo.reshape %arg343 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2401 = stablehlo.multiply %2396, %2400 : tensor<1x1x1x512xf32>
      %2402 = stablehlo.broadcast_in_dim %2401, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2403 = stablehlo.multiply %2393, %2402 : tensor<256x7x7x512xf32>
      %2404 = stablehlo.reshape %arg342 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2405 = stablehlo.broadcast_in_dim %2404, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2406 = stablehlo.add %2403, %2405 : tensor<256x7x7x512xf32>
      %2407 = stablehlo.convert %2406 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2408 = stablehlo.add %2268, %2407 : tensor<256x7x7x512xf16>
      %2409 = func.call @relu_198(%2408) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_693 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2410 = stablehlo.broadcast_in_dim %cst_693, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2411 = stablehlo.compare  GT, %2408, %2410,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %2412 = stablehlo.convert %arg350 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %2413 = stablehlo.convolution(%2409, %2412) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %2414 = stablehlo.convert %2413 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2415 = stablehlo.multiply %2414, %2414 : tensor<256x7x7x512xf32>
      %cst_694 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2416 = stablehlo.broadcast_in_dim %cst_694, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %2417 = stablehlo.multiply %2416, %2414 : tensor<256x7x7x512xf32>
      %cst_695 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2418 = stablehlo.reduce(%2414 init: %cst_695) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_696 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2419 = stablehlo.broadcast_in_dim %cst_696, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2420 = stablehlo.divide %2418, %2419 : tensor<512xf32>
      %cst_697 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2421 = stablehlo.reduce(%2415 init: %cst_697) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_698 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2422 = stablehlo.broadcast_in_dim %cst_698, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2423 = stablehlo.divide %2421, %2422 : tensor<512xf32>
      %2424 = stablehlo.broadcast_in_dim %2420, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2425 = stablehlo.broadcast_in_dim %2423, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2426 = stablehlo.concatenate %2424, %2425, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %2427 = "stablehlo.all_reduce"(%2426) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_699 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2428 = stablehlo.broadcast_in_dim %cst_699, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2429 = stablehlo.divide %2427, %2428 : tensor<2x512xf32>
      %2430 = stablehlo.slice %2429 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2431 = stablehlo.reshape %2430 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2432 = stablehlo.slice %2429 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2433 = stablehlo.reshape %2432 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2434 = stablehlo.multiply %2431, %2431 : tensor<512xf32>
      %cst_700 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2435 = stablehlo.broadcast_in_dim %cst_700, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2436 = stablehlo.multiply %2435, %2431 : tensor<512xf32>
      %2437 = stablehlo.subtract %2433, %2434 : tensor<512xf32>
      %cst_701 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2438 = stablehlo.broadcast_in_dim %cst_701, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2439 = stablehlo.maximum %2438, %2437 : tensor<512xf32>
      %2440 = stablehlo.compare  EQ, %2437, %2439,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_702 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2441 = stablehlo.broadcast_in_dim %cst_702, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_703 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2442 = stablehlo.broadcast_in_dim %cst_703, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2443 = stablehlo.select %2440, %2441, %2442 : tensor<512xi1>, tensor<512xf32>
      %cst_704 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2444 = stablehlo.broadcast_in_dim %cst_704, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2445 = stablehlo.compare  EQ, %2444, %2439,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_705 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2446 = stablehlo.broadcast_in_dim %cst_705, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_706 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2447 = stablehlo.broadcast_in_dim %cst_706, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2448 = stablehlo.select %2445, %2446, %2447 : tensor<512xi1>, tensor<512xf32>
      %2449 = stablehlo.divide %2443, %2448 : tensor<512xf32>
      %cst_707 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2450 = stablehlo.broadcast_in_dim %cst_707, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2451 = stablehlo.multiply %2450, %arg550 : tensor<512xf32>
      %cst_708 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2452 = stablehlo.broadcast_in_dim %cst_708, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2453 = stablehlo.multiply %2452, %2431 : tensor<512xf32>
      %2454 = stablehlo.add %2451, %2453 : tensor<512xf32>
      %cst_709 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2455 = stablehlo.broadcast_in_dim %cst_709, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2456 = stablehlo.multiply %2455, %arg551 : tensor<512xf32>
      %cst_710 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2457 = stablehlo.broadcast_in_dim %cst_710, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2458 = stablehlo.multiply %2457, %2439 : tensor<512xf32>
      %2459 = stablehlo.add %2456, %2458 : tensor<512xf32>
      %2460 = stablehlo.broadcast_in_dim %2431, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2461 = stablehlo.broadcast_in_dim %2439, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2462 = stablehlo.convert %2413 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2463 = stablehlo.broadcast_in_dim %2460, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2464 = stablehlo.subtract %2462, %2463 : tensor<256x7x7x512xf32>
      %cst_711 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2465 = stablehlo.broadcast_in_dim %cst_711, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2466 = stablehlo.add %2461, %2465 : tensor<1x1x1x512xf32>
      %2467 = stablehlo.rsqrt %2466 : tensor<1x1x1x512xf32>
      %2468 = stablehlo.divide %2467, %2466 : tensor<1x1x1x512xf32>
      %cst_712 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %2469 = stablehlo.broadcast_in_dim %cst_712, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2470 = stablehlo.multiply %2469, %2468 : tensor<1x1x1x512xf32>
      %2471 = stablehlo.reshape %arg347 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2472 = stablehlo.multiply %2467, %2471 : tensor<1x1x1x512xf32>
      %2473 = stablehlo.broadcast_in_dim %2472, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2474 = stablehlo.multiply %2464, %2473 : tensor<256x7x7x512xf32>
      %2475 = stablehlo.reshape %arg346 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2476 = stablehlo.broadcast_in_dim %2475, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2477 = stablehlo.add %2474, %2476 : tensor<256x7x7x512xf32>
      %2478 = stablehlo.convert %2477 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2479 = func.call @relu_198(%2478) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_713 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2480 = stablehlo.broadcast_in_dim %cst_713, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2481 = stablehlo.compare  GT, %2478, %2480,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %2482 = stablehlo.convert %arg351 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %2483 = stablehlo.convolution(%2479, %2482) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %2484 = stablehlo.convert %2483 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2485 = stablehlo.multiply %2484, %2484 : tensor<256x7x7x512xf32>
      %cst_714 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2486 = stablehlo.broadcast_in_dim %cst_714, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %2487 = stablehlo.multiply %2486, %2484 : tensor<256x7x7x512xf32>
      %cst_715 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2488 = stablehlo.reduce(%2484 init: %cst_715) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_716 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2489 = stablehlo.broadcast_in_dim %cst_716, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2490 = stablehlo.divide %2488, %2489 : tensor<512xf32>
      %cst_717 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2491 = stablehlo.reduce(%2485 init: %cst_717) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_718 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2492 = stablehlo.broadcast_in_dim %cst_718, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2493 = stablehlo.divide %2491, %2492 : tensor<512xf32>
      %2494 = stablehlo.broadcast_in_dim %2490, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2495 = stablehlo.broadcast_in_dim %2493, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %2496 = stablehlo.concatenate %2494, %2495, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %2497 = "stablehlo.all_reduce"(%2496) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_719 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2498 = stablehlo.broadcast_in_dim %cst_719, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2499 = stablehlo.divide %2497, %2498 : tensor<2x512xf32>
      %2500 = stablehlo.slice %2499 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2501 = stablehlo.reshape %2500 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2502 = stablehlo.slice %2499 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2503 = stablehlo.reshape %2502 : (tensor<1x512xf32>) -> tensor<512xf32>
      %2504 = stablehlo.multiply %2501, %2501 : tensor<512xf32>
      %cst_720 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2505 = stablehlo.broadcast_in_dim %cst_720, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2506 = stablehlo.multiply %2505, %2501 : tensor<512xf32>
      %2507 = stablehlo.subtract %2503, %2504 : tensor<512xf32>
      %cst_721 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2508 = stablehlo.broadcast_in_dim %cst_721, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2509 = stablehlo.maximum %2508, %2507 : tensor<512xf32>
      %2510 = stablehlo.compare  EQ, %2507, %2509,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_722 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2511 = stablehlo.broadcast_in_dim %cst_722, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_723 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2512 = stablehlo.broadcast_in_dim %cst_723, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2513 = stablehlo.select %2510, %2511, %2512 : tensor<512xi1>, tensor<512xf32>
      %cst_724 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2514 = stablehlo.broadcast_in_dim %cst_724, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2515 = stablehlo.compare  EQ, %2514, %2509,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_725 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2516 = stablehlo.broadcast_in_dim %cst_725, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_726 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2517 = stablehlo.broadcast_in_dim %cst_726, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2518 = stablehlo.select %2515, %2516, %2517 : tensor<512xi1>, tensor<512xf32>
      %2519 = stablehlo.divide %2513, %2518 : tensor<512xf32>
      %cst_727 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2520 = stablehlo.broadcast_in_dim %cst_727, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2521 = stablehlo.multiply %2520, %arg552 : tensor<512xf32>
      %cst_728 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2522 = stablehlo.broadcast_in_dim %cst_728, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2523 = stablehlo.multiply %2522, %2501 : tensor<512xf32>
      %2524 = stablehlo.add %2521, %2523 : tensor<512xf32>
      %cst_729 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %2525 = stablehlo.broadcast_in_dim %cst_729, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2526 = stablehlo.multiply %2525, %arg553 : tensor<512xf32>
      %cst_730 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %2527 = stablehlo.broadcast_in_dim %cst_730, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2528 = stablehlo.multiply %2527, %2509 : tensor<512xf32>
      %2529 = stablehlo.add %2526, %2528 : tensor<512xf32>
      %2530 = stablehlo.broadcast_in_dim %2501, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2531 = stablehlo.broadcast_in_dim %2509, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2532 = stablehlo.convert %2483 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %2533 = stablehlo.broadcast_in_dim %2530, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2534 = stablehlo.subtract %2532, %2533 : tensor<256x7x7x512xf32>
      %cst_731 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2535 = stablehlo.broadcast_in_dim %cst_731, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2536 = stablehlo.add %2531, %2535 : tensor<1x1x1x512xf32>
      %2537 = stablehlo.rsqrt %2536 : tensor<1x1x1x512xf32>
      %2538 = stablehlo.divide %2537, %2536 : tensor<1x1x1x512xf32>
      %cst_732 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %2539 = stablehlo.broadcast_in_dim %cst_732, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %2540 = stablehlo.multiply %2539, %2538 : tensor<1x1x1x512xf32>
      %2541 = stablehlo.reshape %arg349 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2542 = stablehlo.multiply %2537, %2541 : tensor<1x1x1x512xf32>
      %2543 = stablehlo.broadcast_in_dim %2542, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2544 = stablehlo.multiply %2534, %2543 : tensor<256x7x7x512xf32>
      %2545 = stablehlo.reshape %arg348 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2546 = stablehlo.broadcast_in_dim %2545, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2547 = stablehlo.add %2544, %2546 : tensor<256x7x7x512xf32>
      %2548 = stablehlo.convert %2547 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2549 = stablehlo.add %2409, %2548 : tensor<256x7x7x512xf16>
      %2550 = func.call @relu_198(%2549) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_733 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2551 = stablehlo.broadcast_in_dim %cst_733, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2552 = stablehlo.compare  GT, %2549, %2551,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %2553 = stablehlo.convert %2550 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_734 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2554 = stablehlo.reduce(%2553 init: %cst_734) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<256x512xf32>
      %cst_735 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
      %2555 = stablehlo.broadcast_in_dim %cst_735, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
      %2556 = stablehlo.divide %2554, %2555 : tensor<256x512xf32>
      %2557 = stablehlo.convert %2556 : (tensor<256x512xf32>) -> tensor<256x512xf16>
      %2558 = stablehlo.convert %arg300 : (tensor<512x1000xf32>) -> tensor<512x1000xf16>
      %2559 = stablehlo.convert %arg299 : (tensor<1000xf32>) -> tensor<1000xf16>
      %2560 = stablehlo.dot_general %2557, %2558, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf16>, tensor<512x1000xf16>) -> tensor<256x1000xf16>
      %2561 = stablehlo.reshape %2559 : (tensor<1000xf16>) -> tensor<1x1000xf16>
      %2562 = stablehlo.broadcast_in_dim %2561, dims = [0, 1] : (tensor<1x1000xf16>) -> tensor<256x1000xf16>
      %2563 = stablehlo.add %2560, %2562 : tensor<256x1000xf16>
      %2564 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
      %2565 = stablehlo.iota dim = 0 : tensor<1000xi32>
      %2566 = stablehlo.reshape %2565 : (tensor<1000xi32>) -> tensor<1x1000xi32>
      %2567 = stablehlo.broadcast_in_dim %2564, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
      %2568 = stablehlo.broadcast_in_dim %2566, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
      %2569 = stablehlo.compare  EQ, %2567, %2568,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
      %cst_736 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2570 = stablehlo.broadcast_in_dim %cst_736, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %cst_737 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2571 = stablehlo.broadcast_in_dim %cst_737, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %2572 = stablehlo.select %2569, %2570, %2571 : tensor<256x1000xi1>, tensor<256x1000xf32>
      %2573 = stablehlo.convert %2572 : tensor<256x1000xf32>
      %2574:2 = func.call @log_softmax(%2563) : (tensor<256x1000xf16>) -> (tensor<256x1000xf16>, tensor<256x1xf16>)
      %cst_738 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2575 = stablehlo.broadcast_in_dim %cst_738, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %2576 = stablehlo.multiply %2575, %arg300 : tensor<512x1000xf32>
      %cst_739 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2577 = stablehlo.broadcast_in_dim %cst_739, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2578 = stablehlo.multiply %2577, %arg305 : tensor<3x3x64x64xf32>
      %cst_740 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2579 = stablehlo.broadcast_in_dim %cst_740, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2580 = stablehlo.multiply %2579, %arg306 : tensor<3x3x64x64xf32>
      %cst_741 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2581 = stablehlo.broadcast_in_dim %cst_741, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2582 = stablehlo.multiply %2581, %arg311 : tensor<3x3x64x64xf32>
      %cst_742 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2583 = stablehlo.broadcast_in_dim %cst_742, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2584 = stablehlo.multiply %2583, %arg312 : tensor<3x3x64x64xf32>
      %cst_743 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2585 = stablehlo.broadcast_in_dim %cst_743, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2586 = stablehlo.multiply %2585, %arg317 : tensor<3x3x256x256xf32>
      %cst_744 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2587 = stablehlo.broadcast_in_dim %cst_744, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2588 = stablehlo.multiply %2587, %arg318 : tensor<3x3x256x256xf32>
      %cst_745 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2589 = stablehlo.broadcast_in_dim %cst_745, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2590 = stablehlo.multiply %2589, %arg323 : tensor<3x3x256x256xf32>
      %cst_746 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2591 = stablehlo.broadcast_in_dim %cst_746, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2592 = stablehlo.multiply %2591, %arg324 : tensor<3x3x256x256xf32>
      %cst_747 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2593 = stablehlo.broadcast_in_dim %cst_747, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2594 = stablehlo.multiply %2593, %arg329 : tensor<3x3x256x256xf32>
      %cst_748 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2595 = stablehlo.broadcast_in_dim %cst_748, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2596 = stablehlo.multiply %2595, %arg330 : tensor<3x3x256x256xf32>
      %cst_749 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2597 = stablehlo.broadcast_in_dim %cst_749, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %2598 = stablehlo.multiply %2597, %arg335 : tensor<3x3x256x512xf32>
      %cst_750 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2599 = stablehlo.broadcast_in_dim %cst_750, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2600 = stablehlo.multiply %2599, %arg336 : tensor<3x3x512x512xf32>
      %cst_751 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2601 = stablehlo.broadcast_in_dim %cst_751, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %2602 = stablehlo.multiply %2601, %arg337 : tensor<1x1x256x512xf32>
      %cst_752 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2603 = stablehlo.broadcast_in_dim %cst_752, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2604 = stablehlo.multiply %2603, %arg344 : tensor<3x3x512x512xf32>
      %cst_753 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2605 = stablehlo.broadcast_in_dim %cst_753, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2606 = stablehlo.multiply %2605, %arg345 : tensor<3x3x512x512xf32>
      %cst_754 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2607 = stablehlo.broadcast_in_dim %cst_754, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2608 = stablehlo.multiply %2607, %arg350 : tensor<3x3x512x512xf32>
      %cst_755 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2609 = stablehlo.broadcast_in_dim %cst_755, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2610 = stablehlo.multiply %2609, %arg351 : tensor<3x3x512x512xf32>
      %cst_756 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2611 = stablehlo.broadcast_in_dim %cst_756, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2612 = stablehlo.multiply %2611, %arg356 : tensor<3x3x64x64xf32>
      %cst_757 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2613 = stablehlo.broadcast_in_dim %cst_757, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2614 = stablehlo.multiply %2613, %arg357 : tensor<3x3x64x64xf32>
      %cst_758 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2615 = stablehlo.broadcast_in_dim %cst_758, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %2616 = stablehlo.multiply %2615, %arg362 : tensor<3x3x64x128xf32>
      %cst_759 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2617 = stablehlo.broadcast_in_dim %cst_759, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2618 = stablehlo.multiply %2617, %arg363 : tensor<3x3x128x128xf32>
      %cst_760 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2619 = stablehlo.broadcast_in_dim %cst_760, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %2620 = stablehlo.multiply %2619, %arg364 : tensor<1x1x64x128xf32>
      %cst_761 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2621 = stablehlo.broadcast_in_dim %cst_761, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2622 = stablehlo.multiply %2621, %arg371 : tensor<3x3x128x128xf32>
      %cst_762 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2623 = stablehlo.broadcast_in_dim %cst_762, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2624 = stablehlo.multiply %2623, %arg372 : tensor<3x3x128x128xf32>
      %cst_763 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2625 = stablehlo.broadcast_in_dim %cst_763, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2626 = stablehlo.multiply %2625, %arg377 : tensor<3x3x128x128xf32>
      %cst_764 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2627 = stablehlo.broadcast_in_dim %cst_764, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2628 = stablehlo.multiply %2627, %arg378 : tensor<3x3x128x128xf32>
      %cst_765 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2629 = stablehlo.broadcast_in_dim %cst_765, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2630 = stablehlo.multiply %2629, %arg383 : tensor<3x3x128x128xf32>
      %cst_766 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2631 = stablehlo.broadcast_in_dim %cst_766, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2632 = stablehlo.multiply %2631, %arg384 : tensor<3x3x128x128xf32>
      %cst_767 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2633 = stablehlo.broadcast_in_dim %cst_767, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %2634 = stablehlo.multiply %2633, %arg389 : tensor<3x3x128x256xf32>
      %cst_768 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2635 = stablehlo.broadcast_in_dim %cst_768, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2636 = stablehlo.multiply %2635, %arg390 : tensor<3x3x256x256xf32>
      %cst_769 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2637 = stablehlo.broadcast_in_dim %cst_769, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %2638 = stablehlo.multiply %2637, %arg391 : tensor<1x1x128x256xf32>
      %cst_770 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2639 = stablehlo.broadcast_in_dim %cst_770, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2640 = stablehlo.multiply %2639, %arg398 : tensor<3x3x256x256xf32>
      %cst_771 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2641 = stablehlo.broadcast_in_dim %cst_771, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2642 = stablehlo.multiply %2641, %arg399 : tensor<3x3x256x256xf32>
      %cst_772 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2643 = stablehlo.broadcast_in_dim %cst_772, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2644 = stablehlo.multiply %2643, %arg404 : tensor<3x3x256x256xf32>
      %cst_773 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2645 = stablehlo.broadcast_in_dim %cst_773, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2646 = stablehlo.multiply %2645, %arg405 : tensor<3x3x256x256xf32>
      %cst_774 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2647 = stablehlo.broadcast_in_dim %cst_774, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %2648 = stablehlo.multiply %2647, %arg408 : tensor<7x7x3x64xf32>
      %2649 = stablehlo.convert %arg593 : tensor<f32>
      %cst_775 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2650 = stablehlo.multiply %2649, %cst_775 : tensor<f32>
      %cst_776 = stablehlo.constant dense<5.000000e-05> : tensor<f32>
      %2651 = stablehlo.multiply %cst_776, %2650 : tensor<f32>
      %2652 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %2653 = stablehlo.multiply %2652, %2648 : tensor<7x7x3x64xf32>
      %2654 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2655 = stablehlo.multiply %2654, %2646 : tensor<3x3x256x256xf32>
      %2656 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2657 = stablehlo.multiply %2656, %2644 : tensor<3x3x256x256xf32>
      %2658 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2659 = stablehlo.multiply %2658, %2642 : tensor<3x3x256x256xf32>
      %2660 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2661 = stablehlo.multiply %2660, %2640 : tensor<3x3x256x256xf32>
      %2662 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %2663 = stablehlo.multiply %2662, %2638 : tensor<1x1x128x256xf32>
      %2664 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2665 = stablehlo.multiply %2664, %2636 : tensor<3x3x256x256xf32>
      %2666 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %2667 = stablehlo.multiply %2666, %2634 : tensor<3x3x128x256xf32>
      %2668 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2669 = stablehlo.multiply %2668, %2632 : tensor<3x3x128x128xf32>
      %2670 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2671 = stablehlo.multiply %2670, %2630 : tensor<3x3x128x128xf32>
      %2672 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2673 = stablehlo.multiply %2672, %2628 : tensor<3x3x128x128xf32>
      %2674 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2675 = stablehlo.multiply %2674, %2626 : tensor<3x3x128x128xf32>
      %2676 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2677 = stablehlo.multiply %2676, %2624 : tensor<3x3x128x128xf32>
      %2678 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2679 = stablehlo.multiply %2678, %2622 : tensor<3x3x128x128xf32>
      %2680 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %2681 = stablehlo.multiply %2680, %2620 : tensor<1x1x64x128xf32>
      %2682 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2683 = stablehlo.multiply %2682, %2618 : tensor<3x3x128x128xf32>
      %2684 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %2685 = stablehlo.multiply %2684, %2616 : tensor<3x3x64x128xf32>
      %2686 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2687 = stablehlo.multiply %2686, %2614 : tensor<3x3x64x64xf32>
      %2688 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2689 = stablehlo.multiply %2688, %2612 : tensor<3x3x64x64xf32>
      %2690 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2691 = stablehlo.multiply %2690, %2610 : tensor<3x3x512x512xf32>
      %2692 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2693 = stablehlo.multiply %2692, %2608 : tensor<3x3x512x512xf32>
      %2694 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2695 = stablehlo.multiply %2694, %2606 : tensor<3x3x512x512xf32>
      %2696 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2697 = stablehlo.multiply %2696, %2604 : tensor<3x3x512x512xf32>
      %2698 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %2699 = stablehlo.multiply %2698, %2602 : tensor<1x1x256x512xf32>
      %2700 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2701 = stablehlo.multiply %2700, %2600 : tensor<3x3x512x512xf32>
      %2702 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %2703 = stablehlo.multiply %2702, %2598 : tensor<3x3x256x512xf32>
      %2704 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2705 = stablehlo.multiply %2704, %2596 : tensor<3x3x256x256xf32>
      %2706 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2707 = stablehlo.multiply %2706, %2594 : tensor<3x3x256x256xf32>
      %2708 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2709 = stablehlo.multiply %2708, %2592 : tensor<3x3x256x256xf32>
      %2710 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2711 = stablehlo.multiply %2710, %2590 : tensor<3x3x256x256xf32>
      %2712 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2713 = stablehlo.multiply %2712, %2588 : tensor<3x3x256x256xf32>
      %2714 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2715 = stablehlo.multiply %2714, %2586 : tensor<3x3x256x256xf32>
      %2716 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2717 = stablehlo.multiply %2716, %2584 : tensor<3x3x64x64xf32>
      %2718 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2719 = stablehlo.multiply %2718, %2582 : tensor<3x3x64x64xf32>
      %2720 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2721 = stablehlo.multiply %2720, %2580 : tensor<3x3x64x64xf32>
      %2722 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2723 = stablehlo.multiply %2722, %2578 : tensor<3x3x64x64xf32>
      %2724 = stablehlo.broadcast_in_dim %2651, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %2725 = stablehlo.multiply %2724, %2576 : tensor<512x1000xf32>
      %cst_777 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %2726 = stablehlo.divide %2650, %cst_777 : tensor<f32>
      %2727 = stablehlo.broadcast_in_dim %2726, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2728 = stablehlo.negate %2727 : tensor<256xf32>
      %2729 = stablehlo.broadcast_in_dim %2728, dims = [0] : (tensor<256xf32>) -> tensor<256x1000xf32>
      %2730 = stablehlo.multiply %2573, %2729 : tensor<256x1000xf32>
      %2731 = stablehlo.convert %2730 : (tensor<256x1000xf32>) -> tensor<256x1000xf16>
      %2732 = func.call @log_softmax_285(%2574#0, %2574#1, %2731) : (tensor<256x1000xf16>, tensor<256x1xf16>, tensor<256x1000xf16>) -> tensor<256x1000xf16>
      %cst_778 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2733 = stablehlo.reduce(%2732 init: %cst_778) applies stablehlo.add across dimensions = [0] : (tensor<256x1000xf16>, tensor<f16>) -> tensor<1000xf16>
      %2734 = stablehlo.reshape %2733 : (tensor<1000xf16>) -> tensor<1x1000xf16>
      %2735 = stablehlo.reshape %2734 : (tensor<1x1000xf16>) -> tensor<1000xf16>
      %2736 = stablehlo.dot_general %2732, %2557, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x1000xf16>, tensor<256x512xf16>) -> tensor<1000x512xf16>
      %2737 = stablehlo.transpose %2736, dims = [1, 0] : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
      %2738 = stablehlo.dot_general %2732, %2558, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1000xf16>, tensor<512x1000xf16>) -> tensor<256x512xf16>
      %2739 = stablehlo.convert %2735 : (tensor<1000xf16>) -> tensor<1000xf32>
      %2740 = stablehlo.convert %2737 : (tensor<512x1000xf16>) -> tensor<512x1000xf32>
      %2741 = stablehlo.add %2725, %2740 : tensor<512x1000xf32>
      %2742 = stablehlo.convert %2738 : (tensor<256x512xf16>) -> tensor<256x512xf32>
      %cst_779 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
      %2743 = stablehlo.broadcast_in_dim %cst_779, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
      %2744 = stablehlo.divide %2742, %2743 : tensor<256x512xf32>
      %2745 = stablehlo.broadcast_in_dim %2744, dims = [0, 3] : (tensor<256x512xf32>) -> tensor<256x7x7x512xf32>
      %2746 = stablehlo.convert %2745 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_780 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2747 = stablehlo.broadcast_in_dim %cst_780, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2748 = stablehlo.select %2552, %2746, %2747 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %2749 = stablehlo.convert %2748 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_781 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2750 = stablehlo.reduce(%2749 init: %cst_781) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2751 = stablehlo.reshape %2750 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2752 = stablehlo.reshape %2751 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2753 = stablehlo.multiply %2534, %2749 : tensor<256x7x7x512xf32>
      %cst_782 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2754 = stablehlo.reduce(%2753 init: %cst_782) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2755 = stablehlo.reshape %2754 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2756 = stablehlo.broadcast_in_dim %2542, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2757 = stablehlo.multiply %2749, %2756 : tensor<256x7x7x512xf32>
      %2758 = stablehlo.multiply %2537, %2755 : tensor<1x1x1x512xf32>
      %2759 = stablehlo.multiply %2755, %2541 : tensor<1x1x1x512xf32>
      %2760 = stablehlo.reshape %2758 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2761 = stablehlo.multiply %2759, %2540 : tensor<1x1x1x512xf32>
      %2762 = stablehlo.negate %2757 : tensor<256x7x7x512xf32>
      %cst_783 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2763 = stablehlo.reduce(%2762 init: %cst_783) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2764 = stablehlo.reshape %2763 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2765 = stablehlo.convert %2757 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_784 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2766 = stablehlo.reduce(%2761 init: %cst_784) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_785 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2767 = stablehlo.reduce(%2764 init: %cst_785) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2768 = stablehlo.multiply %2766, %2519 : tensor<512xf32>
      %2769 = stablehlo.negate %2768 : tensor<512xf32>
      %2770 = stablehlo.multiply %2769, %2506 : tensor<512xf32>
      %2771 = stablehlo.add %2767, %2770 : tensor<512xf32>
      %2772 = stablehlo.broadcast_in_dim %2768, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_786 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2773 = stablehlo.pad %2772, %cst_786, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2774 = stablehlo.broadcast_in_dim %2771, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_787 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2775 = stablehlo.pad %2774, %cst_787, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2776 = stablehlo.add %2773, %2775 : tensor<2x512xf32>
      %cst_788 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2777 = stablehlo.broadcast_in_dim %cst_788, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2778 = stablehlo.divide %2776, %2777 : tensor<2x512xf32>
      %2779 = "stablehlo.all_reduce"(%2778) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %2780 = stablehlo.slice %2779 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2781 = stablehlo.slice %2779 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_789 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2782 = stablehlo.reduce(%2781 init: %cst_789) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_790 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2783 = stablehlo.reduce(%2780 init: %cst_790) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_791 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2784 = stablehlo.broadcast_in_dim %cst_791, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2785 = stablehlo.divide %2782, %2784 : tensor<512xf32>
      %2786 = stablehlo.broadcast_in_dim %2785, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_792 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2787 = stablehlo.broadcast_in_dim %cst_792, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2788 = stablehlo.divide %2783, %2787 : tensor<512xf32>
      %2789 = stablehlo.broadcast_in_dim %2788, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %2790 = stablehlo.multiply %2786, %2487 : tensor<256x7x7x512xf32>
      %2791 = stablehlo.add %2789, %2790 : tensor<256x7x7x512xf32>
      %2792 = stablehlo.convert %2791 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2793 = stablehlo.add %2765, %2792 : tensor<256x7x7x512xf16>
      %2794 = stablehlo.convolution(%2479, %2793) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x512x512xf16>
      %2795 = stablehlo.reverse %2482, dims = [0, 1] : tensor<3x3x512x512xf16>
      %2796 = stablehlo.convolution(%2793, %2795) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %2797 = stablehlo.convert %2794 : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf32>
      %2798 = stablehlo.add %2691, %2797 : tensor<3x3x512x512xf32>
      %cst_793 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2799 = stablehlo.broadcast_in_dim %cst_793, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2800 = stablehlo.select %2481, %2796, %2799 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %2801 = stablehlo.convert %2800 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_794 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2802 = stablehlo.reduce(%2801 init: %cst_794) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2803 = stablehlo.reshape %2802 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2804 = stablehlo.reshape %2803 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2805 = stablehlo.multiply %2464, %2801 : tensor<256x7x7x512xf32>
      %cst_795 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2806 = stablehlo.reduce(%2805 init: %cst_795) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2807 = stablehlo.reshape %2806 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2808 = stablehlo.broadcast_in_dim %2472, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2809 = stablehlo.multiply %2801, %2808 : tensor<256x7x7x512xf32>
      %2810 = stablehlo.multiply %2467, %2807 : tensor<1x1x1x512xf32>
      %2811 = stablehlo.multiply %2807, %2471 : tensor<1x1x1x512xf32>
      %2812 = stablehlo.reshape %2810 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2813 = stablehlo.multiply %2811, %2470 : tensor<1x1x1x512xf32>
      %2814 = stablehlo.negate %2809 : tensor<256x7x7x512xf32>
      %cst_796 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2815 = stablehlo.reduce(%2814 init: %cst_796) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2816 = stablehlo.reshape %2815 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2817 = stablehlo.convert %2809 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_797 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2818 = stablehlo.reduce(%2813 init: %cst_797) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_798 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2819 = stablehlo.reduce(%2816 init: %cst_798) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2820 = stablehlo.multiply %2818, %2449 : tensor<512xf32>
      %2821 = stablehlo.negate %2820 : tensor<512xf32>
      %2822 = stablehlo.multiply %2821, %2436 : tensor<512xf32>
      %2823 = stablehlo.add %2819, %2822 : tensor<512xf32>
      %2824 = stablehlo.broadcast_in_dim %2820, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_799 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2825 = stablehlo.pad %2824, %cst_799, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2826 = stablehlo.broadcast_in_dim %2823, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_800 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2827 = stablehlo.pad %2826, %cst_800, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2828 = stablehlo.add %2825, %2827 : tensor<2x512xf32>
      %cst_801 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2829 = stablehlo.broadcast_in_dim %cst_801, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2830 = stablehlo.divide %2828, %2829 : tensor<2x512xf32>
      %2831 = "stablehlo.all_reduce"(%2830) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %2832 = stablehlo.slice %2831 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2833 = stablehlo.slice %2831 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_802 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2834 = stablehlo.reduce(%2833 init: %cst_802) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_803 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2835 = stablehlo.reduce(%2832 init: %cst_803) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_804 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2836 = stablehlo.broadcast_in_dim %cst_804, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2837 = stablehlo.divide %2834, %2836 : tensor<512xf32>
      %2838 = stablehlo.broadcast_in_dim %2837, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_805 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2839 = stablehlo.broadcast_in_dim %cst_805, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2840 = stablehlo.divide %2835, %2839 : tensor<512xf32>
      %2841 = stablehlo.broadcast_in_dim %2840, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %2842 = stablehlo.multiply %2838, %2417 : tensor<256x7x7x512xf32>
      %2843 = stablehlo.add %2841, %2842 : tensor<256x7x7x512xf32>
      %2844 = stablehlo.convert %2843 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2845 = stablehlo.add %2817, %2844 : tensor<256x7x7x512xf16>
      %2846 = stablehlo.convolution(%2409, %2845) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x512x512xf16>
      %2847 = stablehlo.reverse %2412, dims = [0, 1] : tensor<3x3x512x512xf16>
      %2848 = stablehlo.convolution(%2845, %2847) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %2849 = stablehlo.add %2748, %2848 : tensor<256x7x7x512xf16>
      %2850 = stablehlo.convert %2846 : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf32>
      %2851 = stablehlo.add %2693, %2850 : tensor<3x3x512x512xf32>
      %cst_806 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2852 = stablehlo.broadcast_in_dim %cst_806, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2853 = stablehlo.select %2411, %2849, %2852 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %2854 = stablehlo.convert %2853 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_807 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2855 = stablehlo.reduce(%2854 init: %cst_807) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2856 = stablehlo.reshape %2855 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2857 = stablehlo.reshape %2856 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2858 = stablehlo.multiply %2393, %2854 : tensor<256x7x7x512xf32>
      %cst_808 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2859 = stablehlo.reduce(%2858 init: %cst_808) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2860 = stablehlo.reshape %2859 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2861 = stablehlo.broadcast_in_dim %2401, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2862 = stablehlo.multiply %2854, %2861 : tensor<256x7x7x512xf32>
      %2863 = stablehlo.multiply %2396, %2860 : tensor<1x1x1x512xf32>
      %2864 = stablehlo.multiply %2860, %2400 : tensor<1x1x1x512xf32>
      %2865 = stablehlo.reshape %2863 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2866 = stablehlo.multiply %2864, %2399 : tensor<1x1x1x512xf32>
      %2867 = stablehlo.negate %2862 : tensor<256x7x7x512xf32>
      %cst_809 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2868 = stablehlo.reduce(%2867 init: %cst_809) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2869 = stablehlo.reshape %2868 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2870 = stablehlo.convert %2862 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_810 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2871 = stablehlo.reduce(%2866 init: %cst_810) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_811 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2872 = stablehlo.reduce(%2869 init: %cst_811) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2873 = stablehlo.multiply %2871, %2378 : tensor<512xf32>
      %2874 = stablehlo.negate %2873 : tensor<512xf32>
      %2875 = stablehlo.multiply %2874, %2365 : tensor<512xf32>
      %2876 = stablehlo.add %2872, %2875 : tensor<512xf32>
      %2877 = stablehlo.broadcast_in_dim %2873, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_812 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2878 = stablehlo.pad %2877, %cst_812, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2879 = stablehlo.broadcast_in_dim %2876, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_813 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2880 = stablehlo.pad %2879, %cst_813, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2881 = stablehlo.add %2878, %2880 : tensor<2x512xf32>
      %cst_814 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2882 = stablehlo.broadcast_in_dim %cst_814, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2883 = stablehlo.divide %2881, %2882 : tensor<2x512xf32>
      %2884 = "stablehlo.all_reduce"(%2883) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %2885 = stablehlo.slice %2884 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2886 = stablehlo.slice %2884 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_815 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2887 = stablehlo.reduce(%2886 init: %cst_815) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_816 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2888 = stablehlo.reduce(%2885 init: %cst_816) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_817 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2889 = stablehlo.broadcast_in_dim %cst_817, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2890 = stablehlo.divide %2887, %2889 : tensor<512xf32>
      %2891 = stablehlo.broadcast_in_dim %2890, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_818 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2892 = stablehlo.broadcast_in_dim %cst_818, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2893 = stablehlo.divide %2888, %2892 : tensor<512xf32>
      %2894 = stablehlo.broadcast_in_dim %2893, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %2895 = stablehlo.multiply %2891, %2346 : tensor<256x7x7x512xf32>
      %2896 = stablehlo.add %2894, %2895 : tensor<256x7x7x512xf32>
      %2897 = stablehlo.convert %2896 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2898 = stablehlo.add %2870, %2897 : tensor<256x7x7x512xf16>
      %2899 = stablehlo.convolution(%2338, %2898) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x512x512xf16>
      %2900 = stablehlo.reverse %2341, dims = [0, 1] : tensor<3x3x512x512xf16>
      %2901 = stablehlo.convolution(%2898, %2900) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %2902 = stablehlo.convert %2899 : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf32>
      %2903 = stablehlo.add %2695, %2902 : tensor<3x3x512x512xf32>
      %cst_819 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2904 = stablehlo.broadcast_in_dim %cst_819, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2905 = stablehlo.select %2340, %2901, %2904 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %2906 = stablehlo.convert %2905 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_820 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2907 = stablehlo.reduce(%2906 init: %cst_820) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2908 = stablehlo.reshape %2907 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2909 = stablehlo.reshape %2908 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2910 = stablehlo.multiply %2323, %2906 : tensor<256x7x7x512xf32>
      %cst_821 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2911 = stablehlo.reduce(%2910 init: %cst_821) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2912 = stablehlo.reshape %2911 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2913 = stablehlo.broadcast_in_dim %2331, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2914 = stablehlo.multiply %2906, %2913 : tensor<256x7x7x512xf32>
      %2915 = stablehlo.multiply %2326, %2912 : tensor<1x1x1x512xf32>
      %2916 = stablehlo.multiply %2912, %2330 : tensor<1x1x1x512xf32>
      %2917 = stablehlo.reshape %2915 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2918 = stablehlo.multiply %2916, %2329 : tensor<1x1x1x512xf32>
      %2919 = stablehlo.negate %2914 : tensor<256x7x7x512xf32>
      %cst_822 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2920 = stablehlo.reduce(%2919 init: %cst_822) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2921 = stablehlo.reshape %2920 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2922 = stablehlo.convert %2914 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_823 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2923 = stablehlo.reduce(%2918 init: %cst_823) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_824 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2924 = stablehlo.reduce(%2921 init: %cst_824) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2925 = stablehlo.multiply %2923, %2308 : tensor<512xf32>
      %2926 = stablehlo.negate %2925 : tensor<512xf32>
      %2927 = stablehlo.multiply %2926, %2295 : tensor<512xf32>
      %2928 = stablehlo.add %2924, %2927 : tensor<512xf32>
      %2929 = stablehlo.broadcast_in_dim %2925, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_825 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2930 = stablehlo.pad %2929, %cst_825, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2931 = stablehlo.broadcast_in_dim %2928, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_826 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2932 = stablehlo.pad %2931, %cst_826, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2933 = stablehlo.add %2930, %2932 : tensor<2x512xf32>
      %cst_827 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2934 = stablehlo.broadcast_in_dim %cst_827, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2935 = stablehlo.divide %2933, %2934 : tensor<2x512xf32>
      %2936 = "stablehlo.all_reduce"(%2935) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %2937 = stablehlo.slice %2936 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2938 = stablehlo.slice %2936 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_828 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2939 = stablehlo.reduce(%2938 init: %cst_828) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_829 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2940 = stablehlo.reduce(%2937 init: %cst_829) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_830 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2941 = stablehlo.broadcast_in_dim %cst_830, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2942 = stablehlo.divide %2939, %2941 : tensor<512xf32>
      %2943 = stablehlo.broadcast_in_dim %2942, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_831 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2944 = stablehlo.broadcast_in_dim %cst_831, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2945 = stablehlo.divide %2940, %2944 : tensor<512xf32>
      %2946 = stablehlo.broadcast_in_dim %2945, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %2947 = stablehlo.multiply %2943, %2276 : tensor<256x7x7x512xf32>
      %2948 = stablehlo.add %2946, %2947 : tensor<256x7x7x512xf32>
      %2949 = stablehlo.convert %2948 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %2950 = stablehlo.add %2922, %2949 : tensor<256x7x7x512xf16>
      %2951 = stablehlo.convolution(%2268, %2950) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x512x512xf16>
      %2952 = stablehlo.reverse %2271, dims = [0, 1] : tensor<3x3x512x512xf16>
      %2953 = stablehlo.convolution(%2950, %2952) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %2954 = stablehlo.add %2853, %2953 : tensor<256x7x7x512xf16>
      %2955 = stablehlo.convert %2951 : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf32>
      %2956 = stablehlo.add %2697, %2955 : tensor<3x3x512x512xf32>
      %cst_832 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2957 = stablehlo.broadcast_in_dim %cst_832, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %2958 = stablehlo.select %2270, %2954, %2957 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %2959 = stablehlo.convert %2958 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_833 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2960 = stablehlo.reduce(%2959 init: %cst_833) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2961 = stablehlo.reshape %2960 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2962 = stablehlo.reshape %2961 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2963 = stablehlo.multiply %2252, %2959 : tensor<256x7x7x512xf32>
      %cst_834 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2964 = stablehlo.reduce(%2963 init: %cst_834) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2965 = stablehlo.reshape %2964 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2966 = stablehlo.broadcast_in_dim %2260, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %2967 = stablehlo.multiply %2959, %2966 : tensor<256x7x7x512xf32>
      %2968 = stablehlo.multiply %2255, %2965 : tensor<1x1x1x512xf32>
      %2969 = stablehlo.multiply %2965, %2259 : tensor<1x1x1x512xf32>
      %2970 = stablehlo.reshape %2968 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %2971 = stablehlo.multiply %2969, %2258 : tensor<1x1x1x512xf32>
      %2972 = stablehlo.negate %2967 : tensor<256x7x7x512xf32>
      %cst_835 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2973 = stablehlo.reduce(%2972 init: %cst_835) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2974 = stablehlo.reshape %2973 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %2975 = stablehlo.convert %2967 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_836 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2976 = stablehlo.reduce(%2971 init: %cst_836) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_837 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2977 = stablehlo.reduce(%2974 init: %cst_837) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %2978 = stablehlo.multiply %2976, %2237 : tensor<512xf32>
      %2979 = stablehlo.negate %2978 : tensor<512xf32>
      %2980 = stablehlo.multiply %2979, %2224 : tensor<512xf32>
      %2981 = stablehlo.add %2977, %2980 : tensor<512xf32>
      %2982 = stablehlo.broadcast_in_dim %2978, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_838 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2983 = stablehlo.pad %2982, %cst_838, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2984 = stablehlo.broadcast_in_dim %2981, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_839 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2985 = stablehlo.pad %2984, %cst_839, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %2986 = stablehlo.add %2983, %2985 : tensor<2x512xf32>
      %cst_840 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2987 = stablehlo.broadcast_in_dim %cst_840, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %2988 = stablehlo.divide %2986, %2987 : tensor<2x512xf32>
      %2989 = "stablehlo.all_reduce"(%2988) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %2990 = stablehlo.slice %2989 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %2991 = stablehlo.slice %2989 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_841 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2992 = stablehlo.reduce(%2991 init: %cst_841) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_842 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2993 = stablehlo.reduce(%2990 init: %cst_842) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_843 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2994 = stablehlo.broadcast_in_dim %cst_843, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2995 = stablehlo.divide %2992, %2994 : tensor<512xf32>
      %2996 = stablehlo.broadcast_in_dim %2995, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_844 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %2997 = stablehlo.broadcast_in_dim %cst_844, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2998 = stablehlo.divide %2993, %2997 : tensor<512xf32>
      %2999 = stablehlo.broadcast_in_dim %2998, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %3000 = stablehlo.multiply %2996, %2205 : tensor<256x7x7x512xf32>
      %3001 = stablehlo.add %2999, %3000 : tensor<256x7x7x512xf32>
      %3002 = stablehlo.convert %3001 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %3003 = stablehlo.add %2975, %3002 : tensor<256x7x7x512xf16>
      %3004 = stablehlo.convolution(%2060, %3003) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x7x7x512xf16>) -> tensor<1x1x256x512xf16>
      %3005 = stablehlo.reverse %2200, dims = [0, 1] : tensor<1x1x256x512xf16>
      %3006 = stablehlo.convolution(%3003, %3005) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x256x512xf16>) -> tensor<256x14x14x256xf16>
      %3007 = stablehlo.convert %3004 : (tensor<1x1x256x512xf16>) -> tensor<1x1x256x512xf32>
      %3008 = stablehlo.add %2699, %3007 : tensor<1x1x256x512xf32>
      %3009 = stablehlo.convert %2958 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_845 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3010 = stablehlo.reduce(%3009 init: %cst_845) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %3011 = stablehlo.reshape %3010 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3012 = stablehlo.reshape %3011 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %3013 = stablehlo.multiply %2185, %3009 : tensor<256x7x7x512xf32>
      %cst_846 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3014 = stablehlo.reduce(%3013 init: %cst_846) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %3015 = stablehlo.reshape %3014 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3016 = stablehlo.broadcast_in_dim %2193, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3017 = stablehlo.multiply %3009, %3016 : tensor<256x7x7x512xf32>
      %3018 = stablehlo.multiply %2188, %3015 : tensor<1x1x1x512xf32>
      %3019 = stablehlo.multiply %3015, %2192 : tensor<1x1x1x512xf32>
      %3020 = stablehlo.reshape %3018 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %3021 = stablehlo.multiply %3019, %2191 : tensor<1x1x1x512xf32>
      %3022 = stablehlo.negate %3017 : tensor<256x7x7x512xf32>
      %cst_847 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3023 = stablehlo.reduce(%3022 init: %cst_847) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %3024 = stablehlo.reshape %3023 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3025 = stablehlo.convert %3017 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_848 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3026 = stablehlo.reduce(%3021 init: %cst_848) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_849 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3027 = stablehlo.reduce(%3024 init: %cst_849) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %3028 = stablehlo.multiply %3026, %2170 : tensor<512xf32>
      %3029 = stablehlo.negate %3028 : tensor<512xf32>
      %3030 = stablehlo.multiply %3029, %2157 : tensor<512xf32>
      %3031 = stablehlo.add %3027, %3030 : tensor<512xf32>
      %3032 = stablehlo.broadcast_in_dim %3028, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_850 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3033 = stablehlo.pad %3032, %cst_850, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %3034 = stablehlo.broadcast_in_dim %3031, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_851 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3035 = stablehlo.pad %3034, %cst_851, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %3036 = stablehlo.add %3033, %3035 : tensor<2x512xf32>
      %cst_852 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3037 = stablehlo.broadcast_in_dim %cst_852, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %3038 = stablehlo.divide %3036, %3037 : tensor<2x512xf32>
      %3039 = "stablehlo.all_reduce"(%3038) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %3040 = stablehlo.slice %3039 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %3041 = stablehlo.slice %3039 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_853 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3042 = stablehlo.reduce(%3041 init: %cst_853) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_854 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3043 = stablehlo.reduce(%3040 init: %cst_854) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_855 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %3044 = stablehlo.broadcast_in_dim %cst_855, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3045 = stablehlo.divide %3042, %3044 : tensor<512xf32>
      %3046 = stablehlo.broadcast_in_dim %3045, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_856 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %3047 = stablehlo.broadcast_in_dim %cst_856, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3048 = stablehlo.divide %3043, %3047 : tensor<512xf32>
      %3049 = stablehlo.broadcast_in_dim %3048, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %3050 = stablehlo.multiply %3046, %2138 : tensor<256x7x7x512xf32>
      %3051 = stablehlo.add %3049, %3050 : tensor<256x7x7x512xf32>
      %3052 = stablehlo.convert %3051 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %3053 = stablehlo.add %3025, %3052 : tensor<256x7x7x512xf16>
      %3054 = stablehlo.convolution(%2130, %3053) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x512x512xf16>
      %3055 = stablehlo.reverse %2133, dims = [0, 1] : tensor<3x3x512x512xf16>
      %3056 = stablehlo.convolution(%3053, %3055) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %3057 = stablehlo.convert %3054 : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf32>
      %3058 = stablehlo.add %2701, %3057 : tensor<3x3x512x512xf32>
      %cst_857 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3059 = stablehlo.broadcast_in_dim %cst_857, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %3060 = stablehlo.select %2132, %3056, %3059 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %3061 = stablehlo.convert %3060 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_858 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3062 = stablehlo.reduce(%3061 init: %cst_858) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %3063 = stablehlo.reshape %3062 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3064 = stablehlo.reshape %3063 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %3065 = stablehlo.multiply %2115, %3061 : tensor<256x7x7x512xf32>
      %cst_859 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3066 = stablehlo.reduce(%3065 init: %cst_859) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %3067 = stablehlo.reshape %3066 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3068 = stablehlo.broadcast_in_dim %2123, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3069 = stablehlo.multiply %3061, %3068 : tensor<256x7x7x512xf32>
      %3070 = stablehlo.multiply %2118, %3067 : tensor<1x1x1x512xf32>
      %3071 = stablehlo.multiply %3067, %2122 : tensor<1x1x1x512xf32>
      %3072 = stablehlo.reshape %3070 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %3073 = stablehlo.multiply %3071, %2121 : tensor<1x1x1x512xf32>
      %3074 = stablehlo.negate %3069 : tensor<256x7x7x512xf32>
      %cst_860 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3075 = stablehlo.reduce(%3074 init: %cst_860) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %3076 = stablehlo.reshape %3075 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3077 = stablehlo.convert %3069 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_861 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3078 = stablehlo.reduce(%3073 init: %cst_861) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_862 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3079 = stablehlo.reduce(%3076 init: %cst_862) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %3080 = stablehlo.multiply %3078, %2100 : tensor<512xf32>
      %3081 = stablehlo.negate %3080 : tensor<512xf32>
      %3082 = stablehlo.multiply %3081, %2087 : tensor<512xf32>
      %3083 = stablehlo.add %3079, %3082 : tensor<512xf32>
      %3084 = stablehlo.broadcast_in_dim %3080, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_863 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3085 = stablehlo.pad %3084, %cst_863, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %3086 = stablehlo.broadcast_in_dim %3083, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_864 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3087 = stablehlo.pad %3086, %cst_864, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %3088 = stablehlo.add %3085, %3087 : tensor<2x512xf32>
      %cst_865 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3089 = stablehlo.broadcast_in_dim %cst_865, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %3090 = stablehlo.divide %3088, %3089 : tensor<2x512xf32>
      %3091 = "stablehlo.all_reduce"(%3090) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %3092 = stablehlo.slice %3091 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %3093 = stablehlo.slice %3091 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_866 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3094 = stablehlo.reduce(%3093 init: %cst_866) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_867 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3095 = stablehlo.reduce(%3092 init: %cst_867) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_868 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %3096 = stablehlo.broadcast_in_dim %cst_868, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3097 = stablehlo.divide %3094, %3096 : tensor<512xf32>
      %3098 = stablehlo.broadcast_in_dim %3097, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_869 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %3099 = stablehlo.broadcast_in_dim %cst_869, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3100 = stablehlo.divide %3095, %3099 : tensor<512xf32>
      %3101 = stablehlo.broadcast_in_dim %3100, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %3102 = stablehlo.multiply %3098, %2068 : tensor<256x7x7x512xf32>
      %3103 = stablehlo.add %3101, %3102 : tensor<256x7x7x512xf32>
      %3104 = stablehlo.convert %3103 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %3105 = stablehlo.add %3077, %3104 : tensor<256x7x7x512xf16>
      %3106 = stablehlo.convolution(%2060, %3105) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x256x512xf16>
      %3107 = stablehlo.reverse %2063, dims = [0, 1] : tensor<3x3x256x512xf16>
      %3108 = stablehlo.convolution(%3105, %3107) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x256x512xf16>) -> tensor<256x14x14x256xf16>
      %3109 = stablehlo.add %3006, %3108 : tensor<256x14x14x256xf16>
      %3110 = stablehlo.convert %3106 : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf32>
      %3111 = stablehlo.add %2703, %3110 : tensor<3x3x256x512xf32>
      %cst_870 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3112 = stablehlo.broadcast_in_dim %cst_870, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3113 = stablehlo.select %2062, %3109, %3112 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3114 = stablehlo.convert %3113 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_871 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3115 = stablehlo.reduce(%3114 init: %cst_871) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3116 = stablehlo.reshape %3115 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3117 = stablehlo.reshape %3116 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3118 = stablehlo.multiply %2044, %3114 : tensor<256x14x14x256xf32>
      %cst_872 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3119 = stablehlo.reduce(%3118 init: %cst_872) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3120 = stablehlo.reshape %3119 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3121 = stablehlo.broadcast_in_dim %2052, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3122 = stablehlo.multiply %3114, %3121 : tensor<256x14x14x256xf32>
      %3123 = stablehlo.multiply %2047, %3120 : tensor<1x1x1x256xf32>
      %3124 = stablehlo.multiply %3120, %2051 : tensor<1x1x1x256xf32>
      %3125 = stablehlo.reshape %3123 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3126 = stablehlo.multiply %3124, %2050 : tensor<1x1x1x256xf32>
      %3127 = stablehlo.negate %3122 : tensor<256x14x14x256xf32>
      %cst_873 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3128 = stablehlo.reduce(%3127 init: %cst_873) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3129 = stablehlo.reshape %3128 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3130 = stablehlo.convert %3122 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_874 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3131 = stablehlo.reduce(%3126 init: %cst_874) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_875 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3132 = stablehlo.reduce(%3129 init: %cst_875) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3133 = stablehlo.multiply %3131, %2029 : tensor<256xf32>
      %3134 = stablehlo.negate %3133 : tensor<256xf32>
      %3135 = stablehlo.multiply %3134, %2016 : tensor<256xf32>
      %3136 = stablehlo.add %3132, %3135 : tensor<256xf32>
      %3137 = stablehlo.broadcast_in_dim %3133, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_876 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3138 = stablehlo.pad %3137, %cst_876, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3139 = stablehlo.broadcast_in_dim %3136, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_877 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3140 = stablehlo.pad %3139, %cst_877, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3141 = stablehlo.add %3138, %3140 : tensor<2x256xf32>
      %cst_878 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3142 = stablehlo.broadcast_in_dim %cst_878, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3143 = stablehlo.divide %3141, %3142 : tensor<2x256xf32>
      %3144 = "stablehlo.all_reduce"(%3143) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3145 = stablehlo.slice %3144 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3146 = stablehlo.slice %3144 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_879 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3147 = stablehlo.reduce(%3146 init: %cst_879) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_880 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3148 = stablehlo.reduce(%3145 init: %cst_880) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_881 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3149 = stablehlo.broadcast_in_dim %cst_881, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3150 = stablehlo.divide %3147, %3149 : tensor<256xf32>
      %3151 = stablehlo.broadcast_in_dim %3150, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_882 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3152 = stablehlo.broadcast_in_dim %cst_882, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3153 = stablehlo.divide %3148, %3152 : tensor<256xf32>
      %3154 = stablehlo.broadcast_in_dim %3153, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3155 = stablehlo.multiply %3151, %1997 : tensor<256x14x14x256xf32>
      %3156 = stablehlo.add %3154, %3155 : tensor<256x14x14x256xf32>
      %3157 = stablehlo.convert %3156 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3158 = stablehlo.add %3130, %3157 : tensor<256x14x14x256xf16>
      %3159 = stablehlo.convolution(%1989, %3158) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3160 = stablehlo.reverse %1992, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3161 = stablehlo.convolution(%3158, %3160) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3162 = stablehlo.convert %3159 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3163 = stablehlo.add %2705, %3162 : tensor<3x3x256x256xf32>
      %cst_883 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3164 = stablehlo.broadcast_in_dim %cst_883, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3165 = stablehlo.select %1991, %3161, %3164 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3166 = stablehlo.convert %3165 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_884 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3167 = stablehlo.reduce(%3166 init: %cst_884) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3168 = stablehlo.reshape %3167 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3169 = stablehlo.reshape %3168 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3170 = stablehlo.multiply %1974, %3166 : tensor<256x14x14x256xf32>
      %cst_885 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3171 = stablehlo.reduce(%3170 init: %cst_885) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3172 = stablehlo.reshape %3171 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3173 = stablehlo.broadcast_in_dim %1982, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3174 = stablehlo.multiply %3166, %3173 : tensor<256x14x14x256xf32>
      %3175 = stablehlo.multiply %1977, %3172 : tensor<1x1x1x256xf32>
      %3176 = stablehlo.multiply %3172, %1981 : tensor<1x1x1x256xf32>
      %3177 = stablehlo.reshape %3175 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3178 = stablehlo.multiply %3176, %1980 : tensor<1x1x1x256xf32>
      %3179 = stablehlo.negate %3174 : tensor<256x14x14x256xf32>
      %cst_886 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3180 = stablehlo.reduce(%3179 init: %cst_886) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3181 = stablehlo.reshape %3180 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3182 = stablehlo.convert %3174 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_887 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3183 = stablehlo.reduce(%3178 init: %cst_887) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_888 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3184 = stablehlo.reduce(%3181 init: %cst_888) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3185 = stablehlo.multiply %3183, %1959 : tensor<256xf32>
      %3186 = stablehlo.negate %3185 : tensor<256xf32>
      %3187 = stablehlo.multiply %3186, %1946 : tensor<256xf32>
      %3188 = stablehlo.add %3184, %3187 : tensor<256xf32>
      %3189 = stablehlo.broadcast_in_dim %3185, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_889 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3190 = stablehlo.pad %3189, %cst_889, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3191 = stablehlo.broadcast_in_dim %3188, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_890 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3192 = stablehlo.pad %3191, %cst_890, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3193 = stablehlo.add %3190, %3192 : tensor<2x256xf32>
      %cst_891 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3194 = stablehlo.broadcast_in_dim %cst_891, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3195 = stablehlo.divide %3193, %3194 : tensor<2x256xf32>
      %3196 = "stablehlo.all_reduce"(%3195) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3197 = stablehlo.slice %3196 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3198 = stablehlo.slice %3196 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_892 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3199 = stablehlo.reduce(%3198 init: %cst_892) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_893 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3200 = stablehlo.reduce(%3197 init: %cst_893) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_894 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3201 = stablehlo.broadcast_in_dim %cst_894, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3202 = stablehlo.divide %3199, %3201 : tensor<256xf32>
      %3203 = stablehlo.broadcast_in_dim %3202, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_895 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3204 = stablehlo.broadcast_in_dim %cst_895, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3205 = stablehlo.divide %3200, %3204 : tensor<256xf32>
      %3206 = stablehlo.broadcast_in_dim %3205, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3207 = stablehlo.multiply %3203, %1927 : tensor<256x14x14x256xf32>
      %3208 = stablehlo.add %3206, %3207 : tensor<256x14x14x256xf32>
      %3209 = stablehlo.convert %3208 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3210 = stablehlo.add %3182, %3209 : tensor<256x14x14x256xf16>
      %3211 = stablehlo.convolution(%1919, %3210) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3212 = stablehlo.reverse %1922, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3213 = stablehlo.convolution(%3210, %3212) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3214 = stablehlo.add %3113, %3213 : tensor<256x14x14x256xf16>
      %3215 = stablehlo.convert %3211 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3216 = stablehlo.add %2707, %3215 : tensor<3x3x256x256xf32>
      %cst_896 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3217 = stablehlo.broadcast_in_dim %cst_896, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3218 = stablehlo.select %1921, %3214, %3217 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3219 = stablehlo.convert %3218 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_897 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3220 = stablehlo.reduce(%3219 init: %cst_897) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3221 = stablehlo.reshape %3220 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3222 = stablehlo.reshape %3221 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3223 = stablehlo.multiply %1903, %3219 : tensor<256x14x14x256xf32>
      %cst_898 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3224 = stablehlo.reduce(%3223 init: %cst_898) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3225 = stablehlo.reshape %3224 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3226 = stablehlo.broadcast_in_dim %1911, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3227 = stablehlo.multiply %3219, %3226 : tensor<256x14x14x256xf32>
      %3228 = stablehlo.multiply %1906, %3225 : tensor<1x1x1x256xf32>
      %3229 = stablehlo.multiply %3225, %1910 : tensor<1x1x1x256xf32>
      %3230 = stablehlo.reshape %3228 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3231 = stablehlo.multiply %3229, %1909 : tensor<1x1x1x256xf32>
      %3232 = stablehlo.negate %3227 : tensor<256x14x14x256xf32>
      %cst_899 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3233 = stablehlo.reduce(%3232 init: %cst_899) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3234 = stablehlo.reshape %3233 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3235 = stablehlo.convert %3227 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_900 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3236 = stablehlo.reduce(%3231 init: %cst_900) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_901 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3237 = stablehlo.reduce(%3234 init: %cst_901) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3238 = stablehlo.multiply %3236, %1888 : tensor<256xf32>
      %3239 = stablehlo.negate %3238 : tensor<256xf32>
      %3240 = stablehlo.multiply %3239, %1875 : tensor<256xf32>
      %3241 = stablehlo.add %3237, %3240 : tensor<256xf32>
      %3242 = stablehlo.broadcast_in_dim %3238, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_902 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3243 = stablehlo.pad %3242, %cst_902, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3244 = stablehlo.broadcast_in_dim %3241, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_903 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3245 = stablehlo.pad %3244, %cst_903, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3246 = stablehlo.add %3243, %3245 : tensor<2x256xf32>
      %cst_904 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3247 = stablehlo.broadcast_in_dim %cst_904, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3248 = stablehlo.divide %3246, %3247 : tensor<2x256xf32>
      %3249 = "stablehlo.all_reduce"(%3248) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3250 = stablehlo.slice %3249 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3251 = stablehlo.slice %3249 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_905 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3252 = stablehlo.reduce(%3251 init: %cst_905) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_906 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3253 = stablehlo.reduce(%3250 init: %cst_906) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_907 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3254 = stablehlo.broadcast_in_dim %cst_907, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3255 = stablehlo.divide %3252, %3254 : tensor<256xf32>
      %3256 = stablehlo.broadcast_in_dim %3255, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_908 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3257 = stablehlo.broadcast_in_dim %cst_908, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3258 = stablehlo.divide %3253, %3257 : tensor<256xf32>
      %3259 = stablehlo.broadcast_in_dim %3258, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3260 = stablehlo.multiply %3256, %1856 : tensor<256x14x14x256xf32>
      %3261 = stablehlo.add %3259, %3260 : tensor<256x14x14x256xf32>
      %3262 = stablehlo.convert %3261 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3263 = stablehlo.add %3235, %3262 : tensor<256x14x14x256xf16>
      %3264 = stablehlo.convolution(%1848, %3263) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3265 = stablehlo.reverse %1851, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3266 = stablehlo.convolution(%3263, %3265) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3267 = stablehlo.convert %3264 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3268 = stablehlo.add %2709, %3267 : tensor<3x3x256x256xf32>
      %cst_909 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3269 = stablehlo.broadcast_in_dim %cst_909, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3270 = stablehlo.select %1850, %3266, %3269 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3271 = stablehlo.convert %3270 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_910 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3272 = stablehlo.reduce(%3271 init: %cst_910) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3273 = stablehlo.reshape %3272 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3274 = stablehlo.reshape %3273 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3275 = stablehlo.multiply %1833, %3271 : tensor<256x14x14x256xf32>
      %cst_911 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3276 = stablehlo.reduce(%3275 init: %cst_911) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3277 = stablehlo.reshape %3276 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3278 = stablehlo.broadcast_in_dim %1841, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3279 = stablehlo.multiply %3271, %3278 : tensor<256x14x14x256xf32>
      %3280 = stablehlo.multiply %1836, %3277 : tensor<1x1x1x256xf32>
      %3281 = stablehlo.multiply %3277, %1840 : tensor<1x1x1x256xf32>
      %3282 = stablehlo.reshape %3280 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3283 = stablehlo.multiply %3281, %1839 : tensor<1x1x1x256xf32>
      %3284 = stablehlo.negate %3279 : tensor<256x14x14x256xf32>
      %cst_912 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3285 = stablehlo.reduce(%3284 init: %cst_912) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3286 = stablehlo.reshape %3285 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3287 = stablehlo.convert %3279 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_913 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3288 = stablehlo.reduce(%3283 init: %cst_913) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_914 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3289 = stablehlo.reduce(%3286 init: %cst_914) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3290 = stablehlo.multiply %3288, %1818 : tensor<256xf32>
      %3291 = stablehlo.negate %3290 : tensor<256xf32>
      %3292 = stablehlo.multiply %3291, %1805 : tensor<256xf32>
      %3293 = stablehlo.add %3289, %3292 : tensor<256xf32>
      %3294 = stablehlo.broadcast_in_dim %3290, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_915 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3295 = stablehlo.pad %3294, %cst_915, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3296 = stablehlo.broadcast_in_dim %3293, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_916 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3297 = stablehlo.pad %3296, %cst_916, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3298 = stablehlo.add %3295, %3297 : tensor<2x256xf32>
      %cst_917 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3299 = stablehlo.broadcast_in_dim %cst_917, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3300 = stablehlo.divide %3298, %3299 : tensor<2x256xf32>
      %3301 = "stablehlo.all_reduce"(%3300) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3302 = stablehlo.slice %3301 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3303 = stablehlo.slice %3301 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_918 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3304 = stablehlo.reduce(%3303 init: %cst_918) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_919 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3305 = stablehlo.reduce(%3302 init: %cst_919) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_920 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3306 = stablehlo.broadcast_in_dim %cst_920, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3307 = stablehlo.divide %3304, %3306 : tensor<256xf32>
      %3308 = stablehlo.broadcast_in_dim %3307, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_921 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3309 = stablehlo.broadcast_in_dim %cst_921, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3310 = stablehlo.divide %3305, %3309 : tensor<256xf32>
      %3311 = stablehlo.broadcast_in_dim %3310, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3312 = stablehlo.multiply %3308, %1786 : tensor<256x14x14x256xf32>
      %3313 = stablehlo.add %3311, %3312 : tensor<256x14x14x256xf32>
      %3314 = stablehlo.convert %3313 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3315 = stablehlo.add %3287, %3314 : tensor<256x14x14x256xf16>
      %3316 = stablehlo.convolution(%1778, %3315) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3317 = stablehlo.reverse %1781, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3318 = stablehlo.convolution(%3315, %3317) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3319 = stablehlo.add %3218, %3318 : tensor<256x14x14x256xf16>
      %3320 = stablehlo.convert %3316 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3321 = stablehlo.add %2711, %3320 : tensor<3x3x256x256xf32>
      %cst_922 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3322 = stablehlo.broadcast_in_dim %cst_922, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3323 = stablehlo.select %1780, %3319, %3322 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3324 = stablehlo.convert %3323 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_923 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3325 = stablehlo.reduce(%3324 init: %cst_923) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3326 = stablehlo.reshape %3325 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3327 = stablehlo.reshape %3326 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3328 = stablehlo.multiply %1762, %3324 : tensor<256x14x14x256xf32>
      %cst_924 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3329 = stablehlo.reduce(%3328 init: %cst_924) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3330 = stablehlo.reshape %3329 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3331 = stablehlo.broadcast_in_dim %1770, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3332 = stablehlo.multiply %3324, %3331 : tensor<256x14x14x256xf32>
      %3333 = stablehlo.multiply %1765, %3330 : tensor<1x1x1x256xf32>
      %3334 = stablehlo.multiply %3330, %1769 : tensor<1x1x1x256xf32>
      %3335 = stablehlo.reshape %3333 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3336 = stablehlo.multiply %3334, %1768 : tensor<1x1x1x256xf32>
      %3337 = stablehlo.negate %3332 : tensor<256x14x14x256xf32>
      %cst_925 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3338 = stablehlo.reduce(%3337 init: %cst_925) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3339 = stablehlo.reshape %3338 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3340 = stablehlo.convert %3332 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_926 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3341 = stablehlo.reduce(%3336 init: %cst_926) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_927 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3342 = stablehlo.reduce(%3339 init: %cst_927) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3343 = stablehlo.multiply %3341, %1747 : tensor<256xf32>
      %3344 = stablehlo.negate %3343 : tensor<256xf32>
      %3345 = stablehlo.multiply %3344, %1734 : tensor<256xf32>
      %3346 = stablehlo.add %3342, %3345 : tensor<256xf32>
      %3347 = stablehlo.broadcast_in_dim %3343, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_928 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3348 = stablehlo.pad %3347, %cst_928, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3349 = stablehlo.broadcast_in_dim %3346, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_929 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3350 = stablehlo.pad %3349, %cst_929, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3351 = stablehlo.add %3348, %3350 : tensor<2x256xf32>
      %cst_930 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3352 = stablehlo.broadcast_in_dim %cst_930, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3353 = stablehlo.divide %3351, %3352 : tensor<2x256xf32>
      %3354 = "stablehlo.all_reduce"(%3353) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3355 = stablehlo.slice %3354 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3356 = stablehlo.slice %3354 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_931 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3357 = stablehlo.reduce(%3356 init: %cst_931) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_932 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3358 = stablehlo.reduce(%3355 init: %cst_932) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_933 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3359 = stablehlo.broadcast_in_dim %cst_933, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3360 = stablehlo.divide %3357, %3359 : tensor<256xf32>
      %3361 = stablehlo.broadcast_in_dim %3360, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_934 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3362 = stablehlo.broadcast_in_dim %cst_934, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3363 = stablehlo.divide %3358, %3362 : tensor<256xf32>
      %3364 = stablehlo.broadcast_in_dim %3363, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3365 = stablehlo.multiply %3361, %1715 : tensor<256x14x14x256xf32>
      %3366 = stablehlo.add %3364, %3365 : tensor<256x14x14x256xf32>
      %3367 = stablehlo.convert %3366 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3368 = stablehlo.add %3340, %3367 : tensor<256x14x14x256xf16>
      %3369 = stablehlo.convolution(%1707, %3368) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3370 = stablehlo.reverse %1710, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3371 = stablehlo.convolution(%3368, %3370) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3372 = stablehlo.convert %3369 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3373 = stablehlo.add %2713, %3372 : tensor<3x3x256x256xf32>
      %cst_935 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3374 = stablehlo.broadcast_in_dim %cst_935, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3375 = stablehlo.select %1709, %3371, %3374 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3376 = stablehlo.convert %3375 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_936 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3377 = stablehlo.reduce(%3376 init: %cst_936) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3378 = stablehlo.reshape %3377 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3379 = stablehlo.reshape %3378 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3380 = stablehlo.multiply %1692, %3376 : tensor<256x14x14x256xf32>
      %cst_937 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3381 = stablehlo.reduce(%3380 init: %cst_937) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3382 = stablehlo.reshape %3381 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3383 = stablehlo.broadcast_in_dim %1700, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3384 = stablehlo.multiply %3376, %3383 : tensor<256x14x14x256xf32>
      %3385 = stablehlo.multiply %1695, %3382 : tensor<1x1x1x256xf32>
      %3386 = stablehlo.multiply %3382, %1699 : tensor<1x1x1x256xf32>
      %3387 = stablehlo.reshape %3385 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3388 = stablehlo.multiply %3386, %1698 : tensor<1x1x1x256xf32>
      %3389 = stablehlo.negate %3384 : tensor<256x14x14x256xf32>
      %cst_938 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3390 = stablehlo.reduce(%3389 init: %cst_938) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3391 = stablehlo.reshape %3390 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3392 = stablehlo.convert %3384 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_939 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3393 = stablehlo.reduce(%3388 init: %cst_939) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_940 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3394 = stablehlo.reduce(%3391 init: %cst_940) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3395 = stablehlo.multiply %3393, %1677 : tensor<256xf32>
      %3396 = stablehlo.negate %3395 : tensor<256xf32>
      %3397 = stablehlo.multiply %3396, %1664 : tensor<256xf32>
      %3398 = stablehlo.add %3394, %3397 : tensor<256xf32>
      %3399 = stablehlo.broadcast_in_dim %3395, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_941 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3400 = stablehlo.pad %3399, %cst_941, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3401 = stablehlo.broadcast_in_dim %3398, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_942 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3402 = stablehlo.pad %3401, %cst_942, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3403 = stablehlo.add %3400, %3402 : tensor<2x256xf32>
      %cst_943 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3404 = stablehlo.broadcast_in_dim %cst_943, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3405 = stablehlo.divide %3403, %3404 : tensor<2x256xf32>
      %3406 = "stablehlo.all_reduce"(%3405) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3407 = stablehlo.slice %3406 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3408 = stablehlo.slice %3406 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_944 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3409 = stablehlo.reduce(%3408 init: %cst_944) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_945 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3410 = stablehlo.reduce(%3407 init: %cst_945) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_946 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3411 = stablehlo.broadcast_in_dim %cst_946, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3412 = stablehlo.divide %3409, %3411 : tensor<256xf32>
      %3413 = stablehlo.broadcast_in_dim %3412, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_947 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3414 = stablehlo.broadcast_in_dim %cst_947, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3415 = stablehlo.divide %3410, %3414 : tensor<256xf32>
      %3416 = stablehlo.broadcast_in_dim %3415, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3417 = stablehlo.multiply %3413, %1645 : tensor<256x14x14x256xf32>
      %3418 = stablehlo.add %3416, %3417 : tensor<256x14x14x256xf32>
      %3419 = stablehlo.convert %3418 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3420 = stablehlo.add %3392, %3419 : tensor<256x14x14x256xf16>
      %3421 = stablehlo.convolution(%1637, %3420) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3422 = stablehlo.reverse %1640, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3423 = stablehlo.convolution(%3420, %3422) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3424 = stablehlo.add %3323, %3423 : tensor<256x14x14x256xf16>
      %3425 = stablehlo.convert %3421 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3426 = stablehlo.add %2715, %3425 : tensor<3x3x256x256xf32>
      %cst_948 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3427 = stablehlo.broadcast_in_dim %cst_948, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3428 = stablehlo.select %1639, %3424, %3427 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3429 = stablehlo.convert %3428 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_949 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3430 = stablehlo.reduce(%3429 init: %cst_949) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3431 = stablehlo.reshape %3430 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3432 = stablehlo.reshape %3431 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3433 = stablehlo.multiply %1621, %3429 : tensor<256x14x14x256xf32>
      %cst_950 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3434 = stablehlo.reduce(%3433 init: %cst_950) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3435 = stablehlo.reshape %3434 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3436 = stablehlo.broadcast_in_dim %1629, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3437 = stablehlo.multiply %3429, %3436 : tensor<256x14x14x256xf32>
      %3438 = stablehlo.multiply %1624, %3435 : tensor<1x1x1x256xf32>
      %3439 = stablehlo.multiply %3435, %1628 : tensor<1x1x1x256xf32>
      %3440 = stablehlo.reshape %3438 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3441 = stablehlo.multiply %3439, %1627 : tensor<1x1x1x256xf32>
      %3442 = stablehlo.negate %3437 : tensor<256x14x14x256xf32>
      %cst_951 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3443 = stablehlo.reduce(%3442 init: %cst_951) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3444 = stablehlo.reshape %3443 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3445 = stablehlo.convert %3437 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_952 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3446 = stablehlo.reduce(%3441 init: %cst_952) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_953 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3447 = stablehlo.reduce(%3444 init: %cst_953) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3448 = stablehlo.multiply %3446, %1606 : tensor<256xf32>
      %3449 = stablehlo.negate %3448 : tensor<256xf32>
      %3450 = stablehlo.multiply %3449, %1593 : tensor<256xf32>
      %3451 = stablehlo.add %3447, %3450 : tensor<256xf32>
      %3452 = stablehlo.broadcast_in_dim %3448, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_954 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3453 = stablehlo.pad %3452, %cst_954, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3454 = stablehlo.broadcast_in_dim %3451, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_955 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3455 = stablehlo.pad %3454, %cst_955, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3456 = stablehlo.add %3453, %3455 : tensor<2x256xf32>
      %cst_956 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3457 = stablehlo.broadcast_in_dim %cst_956, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3458 = stablehlo.divide %3456, %3457 : tensor<2x256xf32>
      %3459 = "stablehlo.all_reduce"(%3458) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3460 = stablehlo.slice %3459 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3461 = stablehlo.slice %3459 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_957 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3462 = stablehlo.reduce(%3461 init: %cst_957) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_958 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3463 = stablehlo.reduce(%3460 init: %cst_958) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_959 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3464 = stablehlo.broadcast_in_dim %cst_959, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3465 = stablehlo.divide %3462, %3464 : tensor<256xf32>
      %3466 = stablehlo.broadcast_in_dim %3465, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_960 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3467 = stablehlo.broadcast_in_dim %cst_960, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3468 = stablehlo.divide %3463, %3467 : tensor<256xf32>
      %3469 = stablehlo.broadcast_in_dim %3468, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3470 = stablehlo.multiply %3466, %1574 : tensor<256x14x14x256xf32>
      %3471 = stablehlo.add %3469, %3470 : tensor<256x14x14x256xf32>
      %3472 = stablehlo.convert %3471 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3473 = stablehlo.add %3445, %3472 : tensor<256x14x14x256xf16>
      %3474 = stablehlo.convolution(%1566, %3473) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3475 = stablehlo.reverse %1569, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3476 = stablehlo.convolution(%3473, %3475) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3477 = stablehlo.convert %3474 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3478 = stablehlo.add %2655, %3477 : tensor<3x3x256x256xf32>
      %cst_961 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3479 = stablehlo.broadcast_in_dim %cst_961, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3480 = stablehlo.select %1568, %3476, %3479 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3481 = stablehlo.convert %3480 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_962 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3482 = stablehlo.reduce(%3481 init: %cst_962) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3483 = stablehlo.reshape %3482 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3484 = stablehlo.reshape %3483 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3485 = stablehlo.multiply %1551, %3481 : tensor<256x14x14x256xf32>
      %cst_963 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3486 = stablehlo.reduce(%3485 init: %cst_963) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3487 = stablehlo.reshape %3486 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3488 = stablehlo.broadcast_in_dim %1559, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3489 = stablehlo.multiply %3481, %3488 : tensor<256x14x14x256xf32>
      %3490 = stablehlo.multiply %1554, %3487 : tensor<1x1x1x256xf32>
      %3491 = stablehlo.multiply %3487, %1558 : tensor<1x1x1x256xf32>
      %3492 = stablehlo.reshape %3490 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3493 = stablehlo.multiply %3491, %1557 : tensor<1x1x1x256xf32>
      %3494 = stablehlo.negate %3489 : tensor<256x14x14x256xf32>
      %cst_964 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3495 = stablehlo.reduce(%3494 init: %cst_964) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3496 = stablehlo.reshape %3495 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3497 = stablehlo.convert %3489 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_965 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3498 = stablehlo.reduce(%3493 init: %cst_965) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_966 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3499 = stablehlo.reduce(%3496 init: %cst_966) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3500 = stablehlo.multiply %3498, %1536 : tensor<256xf32>
      %3501 = stablehlo.negate %3500 : tensor<256xf32>
      %3502 = stablehlo.multiply %3501, %1523 : tensor<256xf32>
      %3503 = stablehlo.add %3499, %3502 : tensor<256xf32>
      %3504 = stablehlo.broadcast_in_dim %3500, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_967 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3505 = stablehlo.pad %3504, %cst_967, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3506 = stablehlo.broadcast_in_dim %3503, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_968 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3507 = stablehlo.pad %3506, %cst_968, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3508 = stablehlo.add %3505, %3507 : tensor<2x256xf32>
      %cst_969 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3509 = stablehlo.broadcast_in_dim %cst_969, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3510 = stablehlo.divide %3508, %3509 : tensor<2x256xf32>
      %3511 = "stablehlo.all_reduce"(%3510) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3512 = stablehlo.slice %3511 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3513 = stablehlo.slice %3511 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_970 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3514 = stablehlo.reduce(%3513 init: %cst_970) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_971 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3515 = stablehlo.reduce(%3512 init: %cst_971) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_972 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3516 = stablehlo.broadcast_in_dim %cst_972, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3517 = stablehlo.divide %3514, %3516 : tensor<256xf32>
      %3518 = stablehlo.broadcast_in_dim %3517, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_973 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3519 = stablehlo.broadcast_in_dim %cst_973, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3520 = stablehlo.divide %3515, %3519 : tensor<256xf32>
      %3521 = stablehlo.broadcast_in_dim %3520, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3522 = stablehlo.multiply %3518, %1504 : tensor<256x14x14x256xf32>
      %3523 = stablehlo.add %3521, %3522 : tensor<256x14x14x256xf32>
      %3524 = stablehlo.convert %3523 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3525 = stablehlo.add %3497, %3524 : tensor<256x14x14x256xf16>
      %3526 = stablehlo.convolution(%1496, %3525) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3527 = stablehlo.reverse %1499, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3528 = stablehlo.convolution(%3525, %3527) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3529 = stablehlo.add %3428, %3528 : tensor<256x14x14x256xf16>
      %3530 = stablehlo.convert %3526 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3531 = stablehlo.add %2657, %3530 : tensor<3x3x256x256xf32>
      %cst_974 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3532 = stablehlo.broadcast_in_dim %cst_974, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3533 = stablehlo.select %1498, %3529, %3532 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3534 = stablehlo.convert %3533 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_975 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3535 = stablehlo.reduce(%3534 init: %cst_975) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3536 = stablehlo.reshape %3535 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3537 = stablehlo.reshape %3536 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3538 = stablehlo.multiply %1480, %3534 : tensor<256x14x14x256xf32>
      %cst_976 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3539 = stablehlo.reduce(%3538 init: %cst_976) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3540 = stablehlo.reshape %3539 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3541 = stablehlo.broadcast_in_dim %1488, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3542 = stablehlo.multiply %3534, %3541 : tensor<256x14x14x256xf32>
      %3543 = stablehlo.multiply %1483, %3540 : tensor<1x1x1x256xf32>
      %3544 = stablehlo.multiply %3540, %1487 : tensor<1x1x1x256xf32>
      %3545 = stablehlo.reshape %3543 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3546 = stablehlo.multiply %3544, %1486 : tensor<1x1x1x256xf32>
      %3547 = stablehlo.negate %3542 : tensor<256x14x14x256xf32>
      %cst_977 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3548 = stablehlo.reduce(%3547 init: %cst_977) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3549 = stablehlo.reshape %3548 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3550 = stablehlo.convert %3542 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_978 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3551 = stablehlo.reduce(%3546 init: %cst_978) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_979 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3552 = stablehlo.reduce(%3549 init: %cst_979) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3553 = stablehlo.multiply %3551, %1465 : tensor<256xf32>
      %3554 = stablehlo.negate %3553 : tensor<256xf32>
      %3555 = stablehlo.multiply %3554, %1452 : tensor<256xf32>
      %3556 = stablehlo.add %3552, %3555 : tensor<256xf32>
      %3557 = stablehlo.broadcast_in_dim %3553, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_980 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3558 = stablehlo.pad %3557, %cst_980, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3559 = stablehlo.broadcast_in_dim %3556, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_981 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3560 = stablehlo.pad %3559, %cst_981, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3561 = stablehlo.add %3558, %3560 : tensor<2x256xf32>
      %cst_982 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3562 = stablehlo.broadcast_in_dim %cst_982, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3563 = stablehlo.divide %3561, %3562 : tensor<2x256xf32>
      %3564 = "stablehlo.all_reduce"(%3563) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3565 = stablehlo.slice %3564 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3566 = stablehlo.slice %3564 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_983 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3567 = stablehlo.reduce(%3566 init: %cst_983) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_984 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3568 = stablehlo.reduce(%3565 init: %cst_984) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_985 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3569 = stablehlo.broadcast_in_dim %cst_985, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3570 = stablehlo.divide %3567, %3569 : tensor<256xf32>
      %3571 = stablehlo.broadcast_in_dim %3570, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_986 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3572 = stablehlo.broadcast_in_dim %cst_986, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3573 = stablehlo.divide %3568, %3572 : tensor<256xf32>
      %3574 = stablehlo.broadcast_in_dim %3573, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3575 = stablehlo.multiply %3571, %1433 : tensor<256x14x14x256xf32>
      %3576 = stablehlo.add %3574, %3575 : tensor<256x14x14x256xf32>
      %3577 = stablehlo.convert %3576 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3578 = stablehlo.add %3550, %3577 : tensor<256x14x14x256xf16>
      %3579 = stablehlo.convolution(%1425, %3578) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3580 = stablehlo.reverse %1428, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3581 = stablehlo.convolution(%3578, %3580) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3582 = stablehlo.convert %3579 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3583 = stablehlo.add %2659, %3582 : tensor<3x3x256x256xf32>
      %cst_987 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3584 = stablehlo.broadcast_in_dim %cst_987, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3585 = stablehlo.select %1427, %3581, %3584 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3586 = stablehlo.convert %3585 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_988 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3587 = stablehlo.reduce(%3586 init: %cst_988) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3588 = stablehlo.reshape %3587 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3589 = stablehlo.reshape %3588 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3590 = stablehlo.multiply %1410, %3586 : tensor<256x14x14x256xf32>
      %cst_989 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3591 = stablehlo.reduce(%3590 init: %cst_989) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3592 = stablehlo.reshape %3591 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3593 = stablehlo.broadcast_in_dim %1418, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3594 = stablehlo.multiply %3586, %3593 : tensor<256x14x14x256xf32>
      %3595 = stablehlo.multiply %1413, %3592 : tensor<1x1x1x256xf32>
      %3596 = stablehlo.multiply %3592, %1417 : tensor<1x1x1x256xf32>
      %3597 = stablehlo.reshape %3595 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3598 = stablehlo.multiply %3596, %1416 : tensor<1x1x1x256xf32>
      %3599 = stablehlo.negate %3594 : tensor<256x14x14x256xf32>
      %cst_990 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3600 = stablehlo.reduce(%3599 init: %cst_990) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3601 = stablehlo.reshape %3600 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3602 = stablehlo.convert %3594 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_991 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3603 = stablehlo.reduce(%3598 init: %cst_991) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_992 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3604 = stablehlo.reduce(%3601 init: %cst_992) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3605 = stablehlo.multiply %3603, %1395 : tensor<256xf32>
      %3606 = stablehlo.negate %3605 : tensor<256xf32>
      %3607 = stablehlo.multiply %3606, %1382 : tensor<256xf32>
      %3608 = stablehlo.add %3604, %3607 : tensor<256xf32>
      %3609 = stablehlo.broadcast_in_dim %3605, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_993 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3610 = stablehlo.pad %3609, %cst_993, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3611 = stablehlo.broadcast_in_dim %3608, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_994 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3612 = stablehlo.pad %3611, %cst_994, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3613 = stablehlo.add %3610, %3612 : tensor<2x256xf32>
      %cst_995 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3614 = stablehlo.broadcast_in_dim %cst_995, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3615 = stablehlo.divide %3613, %3614 : tensor<2x256xf32>
      %3616 = "stablehlo.all_reduce"(%3615) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3617 = stablehlo.slice %3616 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3618 = stablehlo.slice %3616 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_996 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3619 = stablehlo.reduce(%3618 init: %cst_996) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_997 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3620 = stablehlo.reduce(%3617 init: %cst_997) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_998 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3621 = stablehlo.broadcast_in_dim %cst_998, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3622 = stablehlo.divide %3619, %3621 : tensor<256xf32>
      %3623 = stablehlo.broadcast_in_dim %3622, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_999 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3624 = stablehlo.broadcast_in_dim %cst_999, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3625 = stablehlo.divide %3620, %3624 : tensor<256xf32>
      %3626 = stablehlo.broadcast_in_dim %3625, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3627 = stablehlo.multiply %3623, %1363 : tensor<256x14x14x256xf32>
      %3628 = stablehlo.add %3626, %3627 : tensor<256x14x14x256xf32>
      %3629 = stablehlo.convert %3628 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3630 = stablehlo.add %3602, %3629 : tensor<256x14x14x256xf16>
      %3631 = stablehlo.convolution(%1355, %3630) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3632 = stablehlo.reverse %1358, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3633 = stablehlo.convolution(%3630, %3632) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3634 = stablehlo.add %3533, %3633 : tensor<256x14x14x256xf16>
      %3635 = stablehlo.convert %3631 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3636 = stablehlo.add %2661, %3635 : tensor<3x3x256x256xf32>
      %cst_1000 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3637 = stablehlo.broadcast_in_dim %cst_1000, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3638 = stablehlo.select %1357, %3634, %3637 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3639 = stablehlo.convert %3638 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_1001 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3640 = stablehlo.reduce(%3639 init: %cst_1001) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3641 = stablehlo.reshape %3640 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3642 = stablehlo.reshape %3641 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3643 = stablehlo.multiply %1339, %3639 : tensor<256x14x14x256xf32>
      %cst_1002 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3644 = stablehlo.reduce(%3643 init: %cst_1002) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3645 = stablehlo.reshape %3644 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3646 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3647 = stablehlo.multiply %3639, %3646 : tensor<256x14x14x256xf32>
      %3648 = stablehlo.multiply %1342, %3645 : tensor<1x1x1x256xf32>
      %3649 = stablehlo.multiply %3645, %1346 : tensor<1x1x1x256xf32>
      %3650 = stablehlo.reshape %3648 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3651 = stablehlo.multiply %3649, %1345 : tensor<1x1x1x256xf32>
      %3652 = stablehlo.negate %3647 : tensor<256x14x14x256xf32>
      %cst_1003 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3653 = stablehlo.reduce(%3652 init: %cst_1003) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3654 = stablehlo.reshape %3653 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3655 = stablehlo.convert %3647 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_1004 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3656 = stablehlo.reduce(%3651 init: %cst_1004) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_1005 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3657 = stablehlo.reduce(%3654 init: %cst_1005) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3658 = stablehlo.multiply %3656, %1324 : tensor<256xf32>
      %3659 = stablehlo.negate %3658 : tensor<256xf32>
      %3660 = stablehlo.multiply %3659, %1311 : tensor<256xf32>
      %3661 = stablehlo.add %3657, %3660 : tensor<256xf32>
      %3662 = stablehlo.broadcast_in_dim %3658, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_1006 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3663 = stablehlo.pad %3662, %cst_1006, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3664 = stablehlo.broadcast_in_dim %3661, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_1007 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3665 = stablehlo.pad %3664, %cst_1007, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3666 = stablehlo.add %3663, %3665 : tensor<2x256xf32>
      %cst_1008 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3667 = stablehlo.broadcast_in_dim %cst_1008, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3668 = stablehlo.divide %3666, %3667 : tensor<2x256xf32>
      %3669 = "stablehlo.all_reduce"(%3668) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3670 = stablehlo.slice %3669 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3671 = stablehlo.slice %3669 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_1009 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3672 = stablehlo.reduce(%3671 init: %cst_1009) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_1010 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3673 = stablehlo.reduce(%3670 init: %cst_1010) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_1011 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3674 = stablehlo.broadcast_in_dim %cst_1011, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3675 = stablehlo.divide %3672, %3674 : tensor<256xf32>
      %3676 = stablehlo.broadcast_in_dim %3675, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_1012 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3677 = stablehlo.broadcast_in_dim %cst_1012, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3678 = stablehlo.divide %3673, %3677 : tensor<256xf32>
      %3679 = stablehlo.broadcast_in_dim %3678, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3680 = stablehlo.multiply %3676, %1292 : tensor<256x14x14x256xf32>
      %3681 = stablehlo.add %3679, %3680 : tensor<256x14x14x256xf32>
      %3682 = stablehlo.convert %3681 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3683 = stablehlo.add %3655, %3682 : tensor<256x14x14x256xf16>
      %3684 = stablehlo.convolution(%1147, %3683) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x14x14x256xf16>) -> tensor<1x1x128x256xf16>
      %3685 = stablehlo.reverse %1287, dims = [0, 1] : tensor<1x1x128x256xf16>
      %3686 = stablehlo.convolution(%3683, %3685) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x128x256xf16>) -> tensor<256x28x28x128xf16>
      %3687 = stablehlo.convert %3684 : (tensor<1x1x128x256xf16>) -> tensor<1x1x128x256xf32>
      %3688 = stablehlo.add %2663, %3687 : tensor<1x1x128x256xf32>
      %3689 = stablehlo.convert %3638 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_1013 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3690 = stablehlo.reduce(%3689 init: %cst_1013) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3691 = stablehlo.reshape %3690 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3692 = stablehlo.reshape %3691 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3693 = stablehlo.multiply %1272, %3689 : tensor<256x14x14x256xf32>
      %cst_1014 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3694 = stablehlo.reduce(%3693 init: %cst_1014) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3695 = stablehlo.reshape %3694 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3696 = stablehlo.broadcast_in_dim %1280, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3697 = stablehlo.multiply %3689, %3696 : tensor<256x14x14x256xf32>
      %3698 = stablehlo.multiply %1275, %3695 : tensor<1x1x1x256xf32>
      %3699 = stablehlo.multiply %3695, %1279 : tensor<1x1x1x256xf32>
      %3700 = stablehlo.reshape %3698 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3701 = stablehlo.multiply %3699, %1278 : tensor<1x1x1x256xf32>
      %3702 = stablehlo.negate %3697 : tensor<256x14x14x256xf32>
      %cst_1015 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3703 = stablehlo.reduce(%3702 init: %cst_1015) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3704 = stablehlo.reshape %3703 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3705 = stablehlo.convert %3697 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_1016 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3706 = stablehlo.reduce(%3701 init: %cst_1016) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_1017 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3707 = stablehlo.reduce(%3704 init: %cst_1017) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3708 = stablehlo.multiply %3706, %1257 : tensor<256xf32>
      %3709 = stablehlo.negate %3708 : tensor<256xf32>
      %3710 = stablehlo.multiply %3709, %1244 : tensor<256xf32>
      %3711 = stablehlo.add %3707, %3710 : tensor<256xf32>
      %3712 = stablehlo.broadcast_in_dim %3708, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_1018 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3713 = stablehlo.pad %3712, %cst_1018, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3714 = stablehlo.broadcast_in_dim %3711, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_1019 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3715 = stablehlo.pad %3714, %cst_1019, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3716 = stablehlo.add %3713, %3715 : tensor<2x256xf32>
      %cst_1020 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3717 = stablehlo.broadcast_in_dim %cst_1020, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3718 = stablehlo.divide %3716, %3717 : tensor<2x256xf32>
      %3719 = "stablehlo.all_reduce"(%3718) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3720 = stablehlo.slice %3719 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3721 = stablehlo.slice %3719 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_1021 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3722 = stablehlo.reduce(%3721 init: %cst_1021) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_1022 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3723 = stablehlo.reduce(%3720 init: %cst_1022) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_1023 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3724 = stablehlo.broadcast_in_dim %cst_1023, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3725 = stablehlo.divide %3722, %3724 : tensor<256xf32>
      %3726 = stablehlo.broadcast_in_dim %3725, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_1024 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3727 = stablehlo.broadcast_in_dim %cst_1024, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3728 = stablehlo.divide %3723, %3727 : tensor<256xf32>
      %3729 = stablehlo.broadcast_in_dim %3728, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3730 = stablehlo.multiply %3726, %1225 : tensor<256x14x14x256xf32>
      %3731 = stablehlo.add %3729, %3730 : tensor<256x14x14x256xf32>
      %3732 = stablehlo.convert %3731 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3733 = stablehlo.add %3705, %3732 : tensor<256x14x14x256xf16>
      %3734 = stablehlo.convolution(%1217, %3733) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %3735 = stablehlo.reverse %1220, dims = [0, 1] : tensor<3x3x256x256xf16>
      %3736 = stablehlo.convolution(%3733, %3735) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3737 = stablehlo.convert %3734 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %3738 = stablehlo.add %2665, %3737 : tensor<3x3x256x256xf32>
      %cst_1025 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3739 = stablehlo.broadcast_in_dim %cst_1025, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %3740 = stablehlo.select %1219, %3736, %3739 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %3741 = stablehlo.convert %3740 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_1026 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3742 = stablehlo.reduce(%3741 init: %cst_1026) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3743 = stablehlo.reshape %3742 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3744 = stablehlo.reshape %3743 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3745 = stablehlo.multiply %1202, %3741 : tensor<256x14x14x256xf32>
      %cst_1027 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3746 = stablehlo.reduce(%3745 init: %cst_1027) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3747 = stablehlo.reshape %3746 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3748 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3749 = stablehlo.multiply %3741, %3748 : tensor<256x14x14x256xf32>
      %3750 = stablehlo.multiply %1205, %3747 : tensor<1x1x1x256xf32>
      %3751 = stablehlo.multiply %3747, %1209 : tensor<1x1x1x256xf32>
      %3752 = stablehlo.reshape %3750 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %3753 = stablehlo.multiply %3751, %1208 : tensor<1x1x1x256xf32>
      %3754 = stablehlo.negate %3749 : tensor<256x14x14x256xf32>
      %cst_1028 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3755 = stablehlo.reduce(%3754 init: %cst_1028) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3756 = stablehlo.reshape %3755 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3757 = stablehlo.convert %3749 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_1029 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3758 = stablehlo.reduce(%3753 init: %cst_1029) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_1030 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3759 = stablehlo.reduce(%3756 init: %cst_1030) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %3760 = stablehlo.multiply %3758, %1187 : tensor<256xf32>
      %3761 = stablehlo.negate %3760 : tensor<256xf32>
      %3762 = stablehlo.multiply %3761, %1174 : tensor<256xf32>
      %3763 = stablehlo.add %3759, %3762 : tensor<256xf32>
      %3764 = stablehlo.broadcast_in_dim %3760, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_1031 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3765 = stablehlo.pad %3764, %cst_1031, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3766 = stablehlo.broadcast_in_dim %3763, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_1032 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3767 = stablehlo.pad %3766, %cst_1032, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %3768 = stablehlo.add %3765, %3767 : tensor<2x256xf32>
      %cst_1033 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3769 = stablehlo.broadcast_in_dim %cst_1033, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %3770 = stablehlo.divide %3768, %3769 : tensor<2x256xf32>
      %3771 = "stablehlo.all_reduce"(%3770) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %3772 = stablehlo.slice %3771 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %3773 = stablehlo.slice %3771 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_1034 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3774 = stablehlo.reduce(%3773 init: %cst_1034) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_1035 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3775 = stablehlo.reduce(%3772 init: %cst_1035) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_1036 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3776 = stablehlo.broadcast_in_dim %cst_1036, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3777 = stablehlo.divide %3774, %3776 : tensor<256xf32>
      %3778 = stablehlo.broadcast_in_dim %3777, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_1037 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %3779 = stablehlo.broadcast_in_dim %cst_1037, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3780 = stablehlo.divide %3775, %3779 : tensor<256xf32>
      %3781 = stablehlo.broadcast_in_dim %3780, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %3782 = stablehlo.multiply %3778, %1155 : tensor<256x14x14x256xf32>
      %3783 = stablehlo.add %3781, %3782 : tensor<256x14x14x256xf32>
      %3784 = stablehlo.convert %3783 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3785 = stablehlo.add %3757, %3784 : tensor<256x14x14x256xf16>
      %3786 = stablehlo.convolution(%1147, %3785) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x128x256xf16>
      %3787 = stablehlo.reverse %1150, dims = [0, 1] : tensor<3x3x128x256xf16>
      %3788 = stablehlo.convolution(%3785, %3787) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x128x256xf16>) -> tensor<256x28x28x128xf16>
      %3789 = stablehlo.add %3686, %3788 : tensor<256x28x28x128xf16>
      %3790 = stablehlo.convert %3786 : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf32>
      %3791 = stablehlo.add %2667, %3790 : tensor<3x3x128x256xf32>
      %cst_1038 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3792 = stablehlo.broadcast_in_dim %cst_1038, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %3793 = stablehlo.select %1149, %3789, %3792 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %3794 = stablehlo.convert %3793 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_1039 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3795 = stablehlo.reduce(%3794 init: %cst_1039) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3796 = stablehlo.reshape %3795 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3797 = stablehlo.reshape %3796 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %3798 = stablehlo.multiply %1131, %3794 : tensor<256x28x28x128xf32>
      %cst_1040 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3799 = stablehlo.reduce(%3798 init: %cst_1040) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3800 = stablehlo.reshape %3799 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3801 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %3802 = stablehlo.multiply %3794, %3801 : tensor<256x28x28x128xf32>
      %3803 = stablehlo.multiply %1134, %3800 : tensor<1x1x1x128xf32>
      %3804 = stablehlo.multiply %3800, %1138 : tensor<1x1x1x128xf32>
      %3805 = stablehlo.reshape %3803 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %3806 = stablehlo.multiply %3804, %1137 : tensor<1x1x1x128xf32>
      %3807 = stablehlo.negate %3802 : tensor<256x28x28x128xf32>
      %cst_1041 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3808 = stablehlo.reduce(%3807 init: %cst_1041) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3809 = stablehlo.reshape %3808 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3810 = stablehlo.convert %3802 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_1042 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3811 = stablehlo.reduce(%3806 init: %cst_1042) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1043 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3812 = stablehlo.reduce(%3809 init: %cst_1043) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3813 = stablehlo.multiply %3811, %1116 : tensor<128xf32>
      %3814 = stablehlo.negate %3813 : tensor<128xf32>
      %3815 = stablehlo.multiply %3814, %1103 : tensor<128xf32>
      %3816 = stablehlo.add %3812, %3815 : tensor<128xf32>
      %3817 = stablehlo.broadcast_in_dim %3813, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1044 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3818 = stablehlo.pad %3817, %cst_1044, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %3819 = stablehlo.broadcast_in_dim %3816, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1045 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3820 = stablehlo.pad %3819, %cst_1045, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %3821 = stablehlo.add %3818, %3820 : tensor<2x128xf32>
      %cst_1046 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3822 = stablehlo.broadcast_in_dim %cst_1046, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %3823 = stablehlo.divide %3821, %3822 : tensor<2x128xf32>
      %3824 = "stablehlo.all_reduce"(%3823) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %3825 = stablehlo.slice %3824 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %3826 = stablehlo.slice %3824 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_1047 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3827 = stablehlo.reduce(%3826 init: %cst_1047) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1048 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3828 = stablehlo.reduce(%3825 init: %cst_1048) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1049 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %3829 = stablehlo.broadcast_in_dim %cst_1049, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3830 = stablehlo.divide %3827, %3829 : tensor<128xf32>
      %3831 = stablehlo.broadcast_in_dim %3830, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_1050 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %3832 = stablehlo.broadcast_in_dim %cst_1050, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3833 = stablehlo.divide %3828, %3832 : tensor<128xf32>
      %3834 = stablehlo.broadcast_in_dim %3833, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %3835 = stablehlo.multiply %3831, %1084 : tensor<256x28x28x128xf32>
      %3836 = stablehlo.add %3834, %3835 : tensor<256x28x28x128xf32>
      %3837 = stablehlo.convert %3836 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %3838 = stablehlo.add %3810, %3837 : tensor<256x28x28x128xf16>
      %3839 = stablehlo.convolution(%1076, %3838) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %3840 = stablehlo.reverse %1079, dims = [0, 1] : tensor<3x3x128x128xf16>
      %3841 = stablehlo.convolution(%3838, %3840) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %3842 = stablehlo.convert %3839 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %3843 = stablehlo.add %2669, %3842 : tensor<3x3x128x128xf32>
      %cst_1051 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3844 = stablehlo.broadcast_in_dim %cst_1051, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %3845 = stablehlo.select %1078, %3841, %3844 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %3846 = stablehlo.convert %3845 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_1052 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3847 = stablehlo.reduce(%3846 init: %cst_1052) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3848 = stablehlo.reshape %3847 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3849 = stablehlo.reshape %3848 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %3850 = stablehlo.multiply %1061, %3846 : tensor<256x28x28x128xf32>
      %cst_1053 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3851 = stablehlo.reduce(%3850 init: %cst_1053) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3852 = stablehlo.reshape %3851 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3853 = stablehlo.broadcast_in_dim %1069, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %3854 = stablehlo.multiply %3846, %3853 : tensor<256x28x28x128xf32>
      %3855 = stablehlo.multiply %1064, %3852 : tensor<1x1x1x128xf32>
      %3856 = stablehlo.multiply %3852, %1068 : tensor<1x1x1x128xf32>
      %3857 = stablehlo.reshape %3855 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %3858 = stablehlo.multiply %3856, %1067 : tensor<1x1x1x128xf32>
      %3859 = stablehlo.negate %3854 : tensor<256x28x28x128xf32>
      %cst_1054 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3860 = stablehlo.reduce(%3859 init: %cst_1054) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3861 = stablehlo.reshape %3860 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3862 = stablehlo.convert %3854 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_1055 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3863 = stablehlo.reduce(%3858 init: %cst_1055) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1056 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3864 = stablehlo.reduce(%3861 init: %cst_1056) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3865 = stablehlo.multiply %3863, %1046 : tensor<128xf32>
      %3866 = stablehlo.negate %3865 : tensor<128xf32>
      %3867 = stablehlo.multiply %3866, %1033 : tensor<128xf32>
      %3868 = stablehlo.add %3864, %3867 : tensor<128xf32>
      %3869 = stablehlo.broadcast_in_dim %3865, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1057 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3870 = stablehlo.pad %3869, %cst_1057, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %3871 = stablehlo.broadcast_in_dim %3868, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1058 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3872 = stablehlo.pad %3871, %cst_1058, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %3873 = stablehlo.add %3870, %3872 : tensor<2x128xf32>
      %cst_1059 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3874 = stablehlo.broadcast_in_dim %cst_1059, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %3875 = stablehlo.divide %3873, %3874 : tensor<2x128xf32>
      %3876 = "stablehlo.all_reduce"(%3875) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %3877 = stablehlo.slice %3876 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %3878 = stablehlo.slice %3876 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_1060 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3879 = stablehlo.reduce(%3878 init: %cst_1060) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1061 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3880 = stablehlo.reduce(%3877 init: %cst_1061) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1062 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %3881 = stablehlo.broadcast_in_dim %cst_1062, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3882 = stablehlo.divide %3879, %3881 : tensor<128xf32>
      %3883 = stablehlo.broadcast_in_dim %3882, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_1063 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %3884 = stablehlo.broadcast_in_dim %cst_1063, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3885 = stablehlo.divide %3880, %3884 : tensor<128xf32>
      %3886 = stablehlo.broadcast_in_dim %3885, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %3887 = stablehlo.multiply %3883, %1014 : tensor<256x28x28x128xf32>
      %3888 = stablehlo.add %3886, %3887 : tensor<256x28x28x128xf32>
      %3889 = stablehlo.convert %3888 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %3890 = stablehlo.add %3862, %3889 : tensor<256x28x28x128xf16>
      %3891 = stablehlo.convolution(%1006, %3890) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %3892 = stablehlo.reverse %1009, dims = [0, 1] : tensor<3x3x128x128xf16>
      %3893 = stablehlo.convolution(%3890, %3892) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %3894 = stablehlo.add %3793, %3893 : tensor<256x28x28x128xf16>
      %3895 = stablehlo.convert %3891 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %3896 = stablehlo.add %2671, %3895 : tensor<3x3x128x128xf32>
      %cst_1064 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3897 = stablehlo.broadcast_in_dim %cst_1064, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %3898 = stablehlo.select %1008, %3894, %3897 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %3899 = stablehlo.convert %3898 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_1065 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3900 = stablehlo.reduce(%3899 init: %cst_1065) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3901 = stablehlo.reshape %3900 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3902 = stablehlo.reshape %3901 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %3903 = stablehlo.multiply %990, %3899 : tensor<256x28x28x128xf32>
      %cst_1066 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3904 = stablehlo.reduce(%3903 init: %cst_1066) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3905 = stablehlo.reshape %3904 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3906 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %3907 = stablehlo.multiply %3899, %3906 : tensor<256x28x28x128xf32>
      %3908 = stablehlo.multiply %993, %3905 : tensor<1x1x1x128xf32>
      %3909 = stablehlo.multiply %3905, %997 : tensor<1x1x1x128xf32>
      %3910 = stablehlo.reshape %3908 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %3911 = stablehlo.multiply %3909, %996 : tensor<1x1x1x128xf32>
      %3912 = stablehlo.negate %3907 : tensor<256x28x28x128xf32>
      %cst_1067 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3913 = stablehlo.reduce(%3912 init: %cst_1067) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3914 = stablehlo.reshape %3913 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3915 = stablehlo.convert %3907 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_1068 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3916 = stablehlo.reduce(%3911 init: %cst_1068) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1069 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3917 = stablehlo.reduce(%3914 init: %cst_1069) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3918 = stablehlo.multiply %3916, %975 : tensor<128xf32>
      %3919 = stablehlo.negate %3918 : tensor<128xf32>
      %3920 = stablehlo.multiply %3919, %962 : tensor<128xf32>
      %3921 = stablehlo.add %3917, %3920 : tensor<128xf32>
      %3922 = stablehlo.broadcast_in_dim %3918, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1070 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3923 = stablehlo.pad %3922, %cst_1070, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %3924 = stablehlo.broadcast_in_dim %3921, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1071 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3925 = stablehlo.pad %3924, %cst_1071, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %3926 = stablehlo.add %3923, %3925 : tensor<2x128xf32>
      %cst_1072 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3927 = stablehlo.broadcast_in_dim %cst_1072, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %3928 = stablehlo.divide %3926, %3927 : tensor<2x128xf32>
      %3929 = "stablehlo.all_reduce"(%3928) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %3930 = stablehlo.slice %3929 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %3931 = stablehlo.slice %3929 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_1073 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3932 = stablehlo.reduce(%3931 init: %cst_1073) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1074 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3933 = stablehlo.reduce(%3930 init: %cst_1074) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1075 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %3934 = stablehlo.broadcast_in_dim %cst_1075, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3935 = stablehlo.divide %3932, %3934 : tensor<128xf32>
      %3936 = stablehlo.broadcast_in_dim %3935, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_1076 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %3937 = stablehlo.broadcast_in_dim %cst_1076, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3938 = stablehlo.divide %3933, %3937 : tensor<128xf32>
      %3939 = stablehlo.broadcast_in_dim %3938, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %3940 = stablehlo.multiply %3936, %943 : tensor<256x28x28x128xf32>
      %3941 = stablehlo.add %3939, %3940 : tensor<256x28x28x128xf32>
      %3942 = stablehlo.convert %3941 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %3943 = stablehlo.add %3915, %3942 : tensor<256x28x28x128xf16>
      %3944 = stablehlo.convolution(%935, %3943) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %3945 = stablehlo.reverse %938, dims = [0, 1] : tensor<3x3x128x128xf16>
      %3946 = stablehlo.convolution(%3943, %3945) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %3947 = stablehlo.convert %3944 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %3948 = stablehlo.add %2673, %3947 : tensor<3x3x128x128xf32>
      %cst_1077 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3949 = stablehlo.broadcast_in_dim %cst_1077, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %3950 = stablehlo.select %937, %3946, %3949 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %3951 = stablehlo.convert %3950 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_1078 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3952 = stablehlo.reduce(%3951 init: %cst_1078) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3953 = stablehlo.reshape %3952 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3954 = stablehlo.reshape %3953 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %3955 = stablehlo.multiply %920, %3951 : tensor<256x28x28x128xf32>
      %cst_1079 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3956 = stablehlo.reduce(%3955 init: %cst_1079) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3957 = stablehlo.reshape %3956 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3958 = stablehlo.broadcast_in_dim %928, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %3959 = stablehlo.multiply %3951, %3958 : tensor<256x28x28x128xf32>
      %3960 = stablehlo.multiply %923, %3957 : tensor<1x1x1x128xf32>
      %3961 = stablehlo.multiply %3957, %927 : tensor<1x1x1x128xf32>
      %3962 = stablehlo.reshape %3960 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %3963 = stablehlo.multiply %3961, %926 : tensor<1x1x1x128xf32>
      %3964 = stablehlo.negate %3959 : tensor<256x28x28x128xf32>
      %cst_1080 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3965 = stablehlo.reduce(%3964 init: %cst_1080) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3966 = stablehlo.reshape %3965 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %3967 = stablehlo.convert %3959 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_1081 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3968 = stablehlo.reduce(%3963 init: %cst_1081) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1082 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3969 = stablehlo.reduce(%3966 init: %cst_1082) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %3970 = stablehlo.multiply %3968, %905 : tensor<128xf32>
      %3971 = stablehlo.negate %3970 : tensor<128xf32>
      %3972 = stablehlo.multiply %3971, %892 : tensor<128xf32>
      %3973 = stablehlo.add %3969, %3972 : tensor<128xf32>
      %3974 = stablehlo.broadcast_in_dim %3970, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1083 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3975 = stablehlo.pad %3974, %cst_1083, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %3976 = stablehlo.broadcast_in_dim %3973, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1084 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3977 = stablehlo.pad %3976, %cst_1084, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %3978 = stablehlo.add %3975, %3977 : tensor<2x128xf32>
      %cst_1085 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3979 = stablehlo.broadcast_in_dim %cst_1085, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %3980 = stablehlo.divide %3978, %3979 : tensor<2x128xf32>
      %3981 = "stablehlo.all_reduce"(%3980) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %3982 = stablehlo.slice %3981 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %3983 = stablehlo.slice %3981 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_1086 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3984 = stablehlo.reduce(%3983 init: %cst_1086) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1087 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3985 = stablehlo.reduce(%3982 init: %cst_1087) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1088 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %3986 = stablehlo.broadcast_in_dim %cst_1088, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3987 = stablehlo.divide %3984, %3986 : tensor<128xf32>
      %3988 = stablehlo.broadcast_in_dim %3987, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_1089 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %3989 = stablehlo.broadcast_in_dim %cst_1089, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3990 = stablehlo.divide %3985, %3989 : tensor<128xf32>
      %3991 = stablehlo.broadcast_in_dim %3990, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %3992 = stablehlo.multiply %3988, %873 : tensor<256x28x28x128xf32>
      %3993 = stablehlo.add %3991, %3992 : tensor<256x28x28x128xf32>
      %3994 = stablehlo.convert %3993 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %3995 = stablehlo.add %3967, %3994 : tensor<256x28x28x128xf16>
      %3996 = stablehlo.convolution(%865, %3995) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %3997 = stablehlo.reverse %868, dims = [0, 1] : tensor<3x3x128x128xf16>
      %3998 = stablehlo.convolution(%3995, %3997) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %3999 = stablehlo.add %3898, %3998 : tensor<256x28x28x128xf16>
      %4000 = stablehlo.convert %3996 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %4001 = stablehlo.add %2675, %4000 : tensor<3x3x128x128xf32>
      %cst_1090 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4002 = stablehlo.broadcast_in_dim %cst_1090, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %4003 = stablehlo.select %867, %3999, %4002 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %4004 = stablehlo.convert %4003 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_1091 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4005 = stablehlo.reduce(%4004 init: %cst_1091) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4006 = stablehlo.reshape %4005 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4007 = stablehlo.reshape %4006 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4008 = stablehlo.multiply %849, %4004 : tensor<256x28x28x128xf32>
      %cst_1092 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4009 = stablehlo.reduce(%4008 init: %cst_1092) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4010 = stablehlo.reshape %4009 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4011 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %4012 = stablehlo.multiply %4004, %4011 : tensor<256x28x28x128xf32>
      %4013 = stablehlo.multiply %852, %4010 : tensor<1x1x1x128xf32>
      %4014 = stablehlo.multiply %4010, %856 : tensor<1x1x1x128xf32>
      %4015 = stablehlo.reshape %4013 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4016 = stablehlo.multiply %4014, %855 : tensor<1x1x1x128xf32>
      %4017 = stablehlo.negate %4012 : tensor<256x28x28x128xf32>
      %cst_1093 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4018 = stablehlo.reduce(%4017 init: %cst_1093) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4019 = stablehlo.reshape %4018 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4020 = stablehlo.convert %4012 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_1094 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4021 = stablehlo.reduce(%4016 init: %cst_1094) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1095 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4022 = stablehlo.reduce(%4019 init: %cst_1095) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4023 = stablehlo.multiply %4021, %834 : tensor<128xf32>
      %4024 = stablehlo.negate %4023 : tensor<128xf32>
      %4025 = stablehlo.multiply %4024, %821 : tensor<128xf32>
      %4026 = stablehlo.add %4022, %4025 : tensor<128xf32>
      %4027 = stablehlo.broadcast_in_dim %4023, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1096 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4028 = stablehlo.pad %4027, %cst_1096, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4029 = stablehlo.broadcast_in_dim %4026, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1097 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4030 = stablehlo.pad %4029, %cst_1097, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4031 = stablehlo.add %4028, %4030 : tensor<2x128xf32>
      %cst_1098 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4032 = stablehlo.broadcast_in_dim %cst_1098, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %4033 = stablehlo.divide %4031, %4032 : tensor<2x128xf32>
      %4034 = "stablehlo.all_reduce"(%4033) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %4035 = stablehlo.slice %4034 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %4036 = stablehlo.slice %4034 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_1099 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4037 = stablehlo.reduce(%4036 init: %cst_1099) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4038 = stablehlo.reduce(%4035 init: %cst_1100) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1101 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4039 = stablehlo.broadcast_in_dim %cst_1101, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4040 = stablehlo.divide %4037, %4039 : tensor<128xf32>
      %4041 = stablehlo.broadcast_in_dim %4040, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_1102 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4042 = stablehlo.broadcast_in_dim %cst_1102, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4043 = stablehlo.divide %4038, %4042 : tensor<128xf32>
      %4044 = stablehlo.broadcast_in_dim %4043, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %4045 = stablehlo.multiply %4041, %802 : tensor<256x28x28x128xf32>
      %4046 = stablehlo.add %4044, %4045 : tensor<256x28x28x128xf32>
      %4047 = stablehlo.convert %4046 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %4048 = stablehlo.add %4020, %4047 : tensor<256x28x28x128xf16>
      %4049 = stablehlo.convolution(%794, %4048) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %4050 = stablehlo.reverse %797, dims = [0, 1] : tensor<3x3x128x128xf16>
      %4051 = stablehlo.convolution(%4048, %4050) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %4052 = stablehlo.convert %4049 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %4053 = stablehlo.add %2677, %4052 : tensor<3x3x128x128xf32>
      %cst_1103 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4054 = stablehlo.broadcast_in_dim %cst_1103, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %4055 = stablehlo.select %796, %4051, %4054 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %4056 = stablehlo.convert %4055 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_1104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4057 = stablehlo.reduce(%4056 init: %cst_1104) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4058 = stablehlo.reshape %4057 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4059 = stablehlo.reshape %4058 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4060 = stablehlo.multiply %779, %4056 : tensor<256x28x28x128xf32>
      %cst_1105 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4061 = stablehlo.reduce(%4060 init: %cst_1105) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4062 = stablehlo.reshape %4061 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4063 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %4064 = stablehlo.multiply %4056, %4063 : tensor<256x28x28x128xf32>
      %4065 = stablehlo.multiply %782, %4062 : tensor<1x1x1x128xf32>
      %4066 = stablehlo.multiply %4062, %786 : tensor<1x1x1x128xf32>
      %4067 = stablehlo.reshape %4065 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4068 = stablehlo.multiply %4066, %785 : tensor<1x1x1x128xf32>
      %4069 = stablehlo.negate %4064 : tensor<256x28x28x128xf32>
      %cst_1106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4070 = stablehlo.reduce(%4069 init: %cst_1106) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4071 = stablehlo.reshape %4070 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4072 = stablehlo.convert %4064 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_1107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4073 = stablehlo.reduce(%4068 init: %cst_1107) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4074 = stablehlo.reduce(%4071 init: %cst_1108) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4075 = stablehlo.multiply %4073, %764 : tensor<128xf32>
      %4076 = stablehlo.negate %4075 : tensor<128xf32>
      %4077 = stablehlo.multiply %4076, %751 : tensor<128xf32>
      %4078 = stablehlo.add %4074, %4077 : tensor<128xf32>
      %4079 = stablehlo.broadcast_in_dim %4075, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4080 = stablehlo.pad %4079, %cst_1109, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4081 = stablehlo.broadcast_in_dim %4078, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4082 = stablehlo.pad %4081, %cst_1110, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4083 = stablehlo.add %4080, %4082 : tensor<2x128xf32>
      %cst_1111 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4084 = stablehlo.broadcast_in_dim %cst_1111, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %4085 = stablehlo.divide %4083, %4084 : tensor<2x128xf32>
      %4086 = "stablehlo.all_reduce"(%4085) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %4087 = stablehlo.slice %4086 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %4088 = stablehlo.slice %4086 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_1112 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4089 = stablehlo.reduce(%4088 init: %cst_1112) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1113 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4090 = stablehlo.reduce(%4087 init: %cst_1113) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1114 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4091 = stablehlo.broadcast_in_dim %cst_1114, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4092 = stablehlo.divide %4089, %4091 : tensor<128xf32>
      %4093 = stablehlo.broadcast_in_dim %4092, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_1115 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4094 = stablehlo.broadcast_in_dim %cst_1115, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4095 = stablehlo.divide %4090, %4094 : tensor<128xf32>
      %4096 = stablehlo.broadcast_in_dim %4095, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %4097 = stablehlo.multiply %4093, %732 : tensor<256x28x28x128xf32>
      %4098 = stablehlo.add %4096, %4097 : tensor<256x28x28x128xf32>
      %4099 = stablehlo.convert %4098 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %4100 = stablehlo.add %4072, %4099 : tensor<256x28x28x128xf16>
      %4101 = stablehlo.convolution(%724, %4100) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %4102 = stablehlo.reverse %727, dims = [0, 1] : tensor<3x3x128x128xf16>
      %4103 = stablehlo.convolution(%4100, %4102) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %4104 = stablehlo.add %4003, %4103 : tensor<256x28x28x128xf16>
      %4105 = stablehlo.convert %4101 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %4106 = stablehlo.add %2679, %4105 : tensor<3x3x128x128xf32>
      %cst_1116 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4107 = stablehlo.broadcast_in_dim %cst_1116, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %4108 = stablehlo.select %726, %4104, %4107 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %4109 = stablehlo.convert %4108 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_1117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4110 = stablehlo.reduce(%4109 init: %cst_1117) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4111 = stablehlo.reshape %4110 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4112 = stablehlo.reshape %4111 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4113 = stablehlo.multiply %708, %4109 : tensor<256x28x28x128xf32>
      %cst_1118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4114 = stablehlo.reduce(%4113 init: %cst_1118) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4115 = stablehlo.reshape %4114 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4116 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %4117 = stablehlo.multiply %4109, %4116 : tensor<256x28x28x128xf32>
      %4118 = stablehlo.multiply %711, %4115 : tensor<1x1x1x128xf32>
      %4119 = stablehlo.multiply %4115, %715 : tensor<1x1x1x128xf32>
      %4120 = stablehlo.reshape %4118 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4121 = stablehlo.multiply %4119, %714 : tensor<1x1x1x128xf32>
      %4122 = stablehlo.negate %4117 : tensor<256x28x28x128xf32>
      %cst_1119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4123 = stablehlo.reduce(%4122 init: %cst_1119) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4124 = stablehlo.reshape %4123 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4125 = stablehlo.convert %4117 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_1120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4126 = stablehlo.reduce(%4121 init: %cst_1120) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4127 = stablehlo.reduce(%4124 init: %cst_1121) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4128 = stablehlo.multiply %4126, %693 : tensor<128xf32>
      %4129 = stablehlo.negate %4128 : tensor<128xf32>
      %4130 = stablehlo.multiply %4129, %680 : tensor<128xf32>
      %4131 = stablehlo.add %4127, %4130 : tensor<128xf32>
      %4132 = stablehlo.broadcast_in_dim %4128, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4133 = stablehlo.pad %4132, %cst_1122, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4134 = stablehlo.broadcast_in_dim %4131, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4135 = stablehlo.pad %4134, %cst_1123, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4136 = stablehlo.add %4133, %4135 : tensor<2x128xf32>
      %cst_1124 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4137 = stablehlo.broadcast_in_dim %cst_1124, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %4138 = stablehlo.divide %4136, %4137 : tensor<2x128xf32>
      %4139 = "stablehlo.all_reduce"(%4138) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %4140 = stablehlo.slice %4139 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %4141 = stablehlo.slice %4139 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_1125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4142 = stablehlo.reduce(%4141 init: %cst_1125) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4143 = stablehlo.reduce(%4140 init: %cst_1126) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1127 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4144 = stablehlo.broadcast_in_dim %cst_1127, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4145 = stablehlo.divide %4142, %4144 : tensor<128xf32>
      %4146 = stablehlo.broadcast_in_dim %4145, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_1128 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4147 = stablehlo.broadcast_in_dim %cst_1128, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4148 = stablehlo.divide %4143, %4147 : tensor<128xf32>
      %4149 = stablehlo.broadcast_in_dim %4148, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %4150 = stablehlo.multiply %4146, %661 : tensor<256x28x28x128xf32>
      %4151 = stablehlo.add %4149, %4150 : tensor<256x28x28x128xf32>
      %4152 = stablehlo.convert %4151 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %4153 = stablehlo.add %4125, %4152 : tensor<256x28x28x128xf16>
      %4154 = stablehlo.convolution(%516, %4153) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x28x28x128xf16>) -> tensor<1x1x64x128xf16>
      %4155 = stablehlo.reverse %656, dims = [0, 1] : tensor<1x1x64x128xf16>
      %4156 = stablehlo.convolution(%4153, %4155) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x64x128xf16>) -> tensor<256x56x56x64xf16>
      %4157 = stablehlo.convert %4154 : (tensor<1x1x64x128xf16>) -> tensor<1x1x64x128xf32>
      %4158 = stablehlo.add %2681, %4157 : tensor<1x1x64x128xf32>
      %4159 = stablehlo.convert %4108 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_1129 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4160 = stablehlo.reduce(%4159 init: %cst_1129) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4161 = stablehlo.reshape %4160 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4162 = stablehlo.reshape %4161 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4163 = stablehlo.multiply %641, %4159 : tensor<256x28x28x128xf32>
      %cst_1130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4164 = stablehlo.reduce(%4163 init: %cst_1130) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4165 = stablehlo.reshape %4164 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4166 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %4167 = stablehlo.multiply %4159, %4166 : tensor<256x28x28x128xf32>
      %4168 = stablehlo.multiply %644, %4165 : tensor<1x1x1x128xf32>
      %4169 = stablehlo.multiply %4165, %648 : tensor<1x1x1x128xf32>
      %4170 = stablehlo.reshape %4168 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4171 = stablehlo.multiply %4169, %647 : tensor<1x1x1x128xf32>
      %4172 = stablehlo.negate %4167 : tensor<256x28x28x128xf32>
      %cst_1131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4173 = stablehlo.reduce(%4172 init: %cst_1131) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4174 = stablehlo.reshape %4173 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4175 = stablehlo.convert %4167 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_1132 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4176 = stablehlo.reduce(%4171 init: %cst_1132) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4177 = stablehlo.reduce(%4174 init: %cst_1133) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4178 = stablehlo.multiply %4176, %626 : tensor<128xf32>
      %4179 = stablehlo.negate %4178 : tensor<128xf32>
      %4180 = stablehlo.multiply %4179, %613 : tensor<128xf32>
      %4181 = stablehlo.add %4177, %4180 : tensor<128xf32>
      %4182 = stablehlo.broadcast_in_dim %4178, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4183 = stablehlo.pad %4182, %cst_1134, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4184 = stablehlo.broadcast_in_dim %4181, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4185 = stablehlo.pad %4184, %cst_1135, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4186 = stablehlo.add %4183, %4185 : tensor<2x128xf32>
      %cst_1136 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4187 = stablehlo.broadcast_in_dim %cst_1136, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %4188 = stablehlo.divide %4186, %4187 : tensor<2x128xf32>
      %4189 = "stablehlo.all_reduce"(%4188) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %4190 = stablehlo.slice %4189 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %4191 = stablehlo.slice %4189 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_1137 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4192 = stablehlo.reduce(%4191 init: %cst_1137) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4193 = stablehlo.reduce(%4190 init: %cst_1138) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1139 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4194 = stablehlo.broadcast_in_dim %cst_1139, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4195 = stablehlo.divide %4192, %4194 : tensor<128xf32>
      %4196 = stablehlo.broadcast_in_dim %4195, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_1140 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4197 = stablehlo.broadcast_in_dim %cst_1140, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4198 = stablehlo.divide %4193, %4197 : tensor<128xf32>
      %4199 = stablehlo.broadcast_in_dim %4198, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %4200 = stablehlo.multiply %4196, %594 : tensor<256x28x28x128xf32>
      %4201 = stablehlo.add %4199, %4200 : tensor<256x28x28x128xf32>
      %4202 = stablehlo.convert %4201 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %4203 = stablehlo.add %4175, %4202 : tensor<256x28x28x128xf16>
      %4204 = stablehlo.convolution(%586, %4203) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %4205 = stablehlo.reverse %589, dims = [0, 1] : tensor<3x3x128x128xf16>
      %4206 = stablehlo.convolution(%4203, %4205) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %4207 = stablehlo.convert %4204 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %4208 = stablehlo.add %2683, %4207 : tensor<3x3x128x128xf32>
      %cst_1141 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4209 = stablehlo.broadcast_in_dim %cst_1141, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %4210 = stablehlo.select %588, %4206, %4209 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %4211 = stablehlo.convert %4210 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_1142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4212 = stablehlo.reduce(%4211 init: %cst_1142) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4213 = stablehlo.reshape %4212 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4214 = stablehlo.reshape %4213 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4215 = stablehlo.multiply %571, %4211 : tensor<256x28x28x128xf32>
      %cst_1143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4216 = stablehlo.reduce(%4215 init: %cst_1143) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4217 = stablehlo.reshape %4216 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4218 = stablehlo.broadcast_in_dim %579, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %4219 = stablehlo.multiply %4211, %4218 : tensor<256x28x28x128xf32>
      %4220 = stablehlo.multiply %574, %4217 : tensor<1x1x1x128xf32>
      %4221 = stablehlo.multiply %4217, %578 : tensor<1x1x1x128xf32>
      %4222 = stablehlo.reshape %4220 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %4223 = stablehlo.multiply %4221, %577 : tensor<1x1x1x128xf32>
      %4224 = stablehlo.negate %4219 : tensor<256x28x28x128xf32>
      %cst_1144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4225 = stablehlo.reduce(%4224 init: %cst_1144) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4226 = stablehlo.reshape %4225 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %4227 = stablehlo.convert %4219 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_1145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4228 = stablehlo.reduce(%4223 init: %cst_1145) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4229 = stablehlo.reduce(%4226 init: %cst_1146) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %4230 = stablehlo.multiply %4228, %556 : tensor<128xf32>
      %4231 = stablehlo.negate %4230 : tensor<128xf32>
      %4232 = stablehlo.multiply %4231, %543 : tensor<128xf32>
      %4233 = stablehlo.add %4229, %4232 : tensor<128xf32>
      %4234 = stablehlo.broadcast_in_dim %4230, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4235 = stablehlo.pad %4234, %cst_1147, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4236 = stablehlo.broadcast_in_dim %4233, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_1148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4237 = stablehlo.pad %4236, %cst_1148, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %4238 = stablehlo.add %4235, %4237 : tensor<2x128xf32>
      %cst_1149 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4239 = stablehlo.broadcast_in_dim %cst_1149, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %4240 = stablehlo.divide %4238, %4239 : tensor<2x128xf32>
      %4241 = "stablehlo.all_reduce"(%4240) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %4242 = stablehlo.slice %4241 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %4243 = stablehlo.slice %4241 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_1150 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4244 = stablehlo.reduce(%4243 init: %cst_1150) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4245 = stablehlo.reduce(%4242 init: %cst_1151) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_1152 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4246 = stablehlo.broadcast_in_dim %cst_1152, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4247 = stablehlo.divide %4244, %4246 : tensor<128xf32>
      %4248 = stablehlo.broadcast_in_dim %4247, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_1153 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %4249 = stablehlo.broadcast_in_dim %cst_1153, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4250 = stablehlo.divide %4245, %4249 : tensor<128xf32>
      %4251 = stablehlo.broadcast_in_dim %4250, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %4252 = stablehlo.multiply %4248, %524 : tensor<256x28x28x128xf32>
      %4253 = stablehlo.add %4251, %4252 : tensor<256x28x28x128xf32>
      %4254 = stablehlo.convert %4253 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %4255 = stablehlo.add %4227, %4254 : tensor<256x28x28x128xf16>
      %4256 = stablehlo.convolution(%516, %4255) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x64x128xf16>
      %4257 = stablehlo.reverse %519, dims = [0, 1] : tensor<3x3x64x128xf16>
      %4258 = stablehlo.convolution(%4255, %4257) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x64x128xf16>) -> tensor<256x56x56x64xf16>
      %4259 = stablehlo.add %4156, %4258 : tensor<256x56x56x64xf16>
      %4260 = stablehlo.convert %4256 : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf32>
      %4261 = stablehlo.add %2685, %4260 : tensor<3x3x64x128xf32>
      %cst_1154 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4262 = stablehlo.broadcast_in_dim %cst_1154, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %4263 = stablehlo.select %518, %4259, %4262 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %4264 = stablehlo.convert %4263 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_1155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4265 = stablehlo.reduce(%4264 init: %cst_1155) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4266 = stablehlo.reshape %4265 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4267 = stablehlo.reshape %4266 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4268 = stablehlo.multiply %500, %4264 : tensor<256x56x56x64xf32>
      %cst_1156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4269 = stablehlo.reduce(%4268 init: %cst_1156) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4270 = stablehlo.reshape %4269 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4271 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %4272 = stablehlo.multiply %4264, %4271 : tensor<256x56x56x64xf32>
      %4273 = stablehlo.multiply %503, %4270 : tensor<1x1x1x64xf32>
      %4274 = stablehlo.multiply %4270, %507 : tensor<1x1x1x64xf32>
      %4275 = stablehlo.reshape %4273 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4276 = stablehlo.multiply %4274, %506 : tensor<1x1x1x64xf32>
      %4277 = stablehlo.negate %4272 : tensor<256x56x56x64xf32>
      %cst_1157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4278 = stablehlo.reduce(%4277 init: %cst_1157) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4279 = stablehlo.reshape %4278 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4280 = stablehlo.convert %4272 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_1158 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4281 = stablehlo.reduce(%4276 init: %cst_1158) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4282 = stablehlo.reduce(%4279 init: %cst_1159) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4283 = stablehlo.multiply %4281, %485 : tensor<64xf32>
      %4284 = stablehlo.negate %4283 : tensor<64xf32>
      %4285 = stablehlo.multiply %4284, %472 : tensor<64xf32>
      %4286 = stablehlo.add %4282, %4285 : tensor<64xf32>
      %4287 = stablehlo.broadcast_in_dim %4283, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4288 = stablehlo.pad %4287, %cst_1160, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4289 = stablehlo.broadcast_in_dim %4286, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4290 = stablehlo.pad %4289, %cst_1161, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4291 = stablehlo.add %4288, %4290 : tensor<2x64xf32>
      %cst_1162 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4292 = stablehlo.broadcast_in_dim %cst_1162, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %4293 = stablehlo.divide %4291, %4292 : tensor<2x64xf32>
      %4294 = "stablehlo.all_reduce"(%4293) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %4295 = stablehlo.slice %4294 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %4296 = stablehlo.slice %4294 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_1163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4297 = stablehlo.reduce(%4296 init: %cst_1163) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4298 = stablehlo.reduce(%4295 init: %cst_1164) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1165 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4299 = stablehlo.broadcast_in_dim %cst_1165, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4300 = stablehlo.divide %4297, %4299 : tensor<64xf32>
      %4301 = stablehlo.broadcast_in_dim %4300, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_1166 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4302 = stablehlo.broadcast_in_dim %cst_1166, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4303 = stablehlo.divide %4298, %4302 : tensor<64xf32>
      %4304 = stablehlo.broadcast_in_dim %4303, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %4305 = stablehlo.multiply %4301, %453 : tensor<256x56x56x64xf32>
      %4306 = stablehlo.add %4304, %4305 : tensor<256x56x56x64xf32>
      %4307 = stablehlo.convert %4306 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %4308 = stablehlo.add %4280, %4307 : tensor<256x56x56x64xf16>
      %4309 = stablehlo.convolution(%445, %4308) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %4310 = stablehlo.reverse %448, dims = [0, 1] : tensor<3x3x64x64xf16>
      %4311 = stablehlo.convolution(%4308, %4310) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %4312 = stablehlo.convert %4309 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %4313 = stablehlo.add %2687, %4312 : tensor<3x3x64x64xf32>
      %cst_1167 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4314 = stablehlo.broadcast_in_dim %cst_1167, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %4315 = stablehlo.select %447, %4311, %4314 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %4316 = stablehlo.convert %4315 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_1168 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4317 = stablehlo.reduce(%4316 init: %cst_1168) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4318 = stablehlo.reshape %4317 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4319 = stablehlo.reshape %4318 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4320 = stablehlo.multiply %430, %4316 : tensor<256x56x56x64xf32>
      %cst_1169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4321 = stablehlo.reduce(%4320 init: %cst_1169) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4322 = stablehlo.reshape %4321 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4323 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %4324 = stablehlo.multiply %4316, %4323 : tensor<256x56x56x64xf32>
      %4325 = stablehlo.multiply %433, %4322 : tensor<1x1x1x64xf32>
      %4326 = stablehlo.multiply %4322, %437 : tensor<1x1x1x64xf32>
      %4327 = stablehlo.reshape %4325 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4328 = stablehlo.multiply %4326, %436 : tensor<1x1x1x64xf32>
      %4329 = stablehlo.negate %4324 : tensor<256x56x56x64xf32>
      %cst_1170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4330 = stablehlo.reduce(%4329 init: %cst_1170) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4331 = stablehlo.reshape %4330 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4332 = stablehlo.convert %4324 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_1171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4333 = stablehlo.reduce(%4328 init: %cst_1171) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4334 = stablehlo.reduce(%4331 init: %cst_1172) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4335 = stablehlo.multiply %4333, %415 : tensor<64xf32>
      %4336 = stablehlo.negate %4335 : tensor<64xf32>
      %4337 = stablehlo.multiply %4336, %402 : tensor<64xf32>
      %4338 = stablehlo.add %4334, %4337 : tensor<64xf32>
      %4339 = stablehlo.broadcast_in_dim %4335, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4340 = stablehlo.pad %4339, %cst_1173, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4341 = stablehlo.broadcast_in_dim %4338, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4342 = stablehlo.pad %4341, %cst_1174, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4343 = stablehlo.add %4340, %4342 : tensor<2x64xf32>
      %cst_1175 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4344 = stablehlo.broadcast_in_dim %cst_1175, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %4345 = stablehlo.divide %4343, %4344 : tensor<2x64xf32>
      %4346 = "stablehlo.all_reduce"(%4345) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %4347 = stablehlo.slice %4346 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %4348 = stablehlo.slice %4346 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_1176 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4349 = stablehlo.reduce(%4348 init: %cst_1176) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4350 = stablehlo.reduce(%4347 init: %cst_1177) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1178 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4351 = stablehlo.broadcast_in_dim %cst_1178, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4352 = stablehlo.divide %4349, %4351 : tensor<64xf32>
      %4353 = stablehlo.broadcast_in_dim %4352, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_1179 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4354 = stablehlo.broadcast_in_dim %cst_1179, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4355 = stablehlo.divide %4350, %4354 : tensor<64xf32>
      %4356 = stablehlo.broadcast_in_dim %4355, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %4357 = stablehlo.multiply %4353, %383 : tensor<256x56x56x64xf32>
      %4358 = stablehlo.add %4356, %4357 : tensor<256x56x56x64xf32>
      %4359 = stablehlo.convert %4358 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %4360 = stablehlo.add %4332, %4359 : tensor<256x56x56x64xf16>
      %4361 = stablehlo.convolution(%375, %4360) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %4362 = stablehlo.reverse %378, dims = [0, 1] : tensor<3x3x64x64xf16>
      %4363 = stablehlo.convolution(%4360, %4362) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %4364 = stablehlo.add %4263, %4363 : tensor<256x56x56x64xf16>
      %4365 = stablehlo.convert %4361 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %4366 = stablehlo.add %2689, %4365 : tensor<3x3x64x64xf32>
      %cst_1180 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4367 = stablehlo.broadcast_in_dim %cst_1180, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %4368 = stablehlo.select %377, %4364, %4367 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %4369 = stablehlo.convert %4368 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_1181 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4370 = stablehlo.reduce(%4369 init: %cst_1181) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4371 = stablehlo.reshape %4370 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4372 = stablehlo.reshape %4371 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4373 = stablehlo.multiply %359, %4369 : tensor<256x56x56x64xf32>
      %cst_1182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4374 = stablehlo.reduce(%4373 init: %cst_1182) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4375 = stablehlo.reshape %4374 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4376 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %4377 = stablehlo.multiply %4369, %4376 : tensor<256x56x56x64xf32>
      %4378 = stablehlo.multiply %362, %4375 : tensor<1x1x1x64xf32>
      %4379 = stablehlo.multiply %4375, %366 : tensor<1x1x1x64xf32>
      %4380 = stablehlo.reshape %4378 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4381 = stablehlo.multiply %4379, %365 : tensor<1x1x1x64xf32>
      %4382 = stablehlo.negate %4377 : tensor<256x56x56x64xf32>
      %cst_1183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4383 = stablehlo.reduce(%4382 init: %cst_1183) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4384 = stablehlo.reshape %4383 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4385 = stablehlo.convert %4377 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_1184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4386 = stablehlo.reduce(%4381 init: %cst_1184) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4387 = stablehlo.reduce(%4384 init: %cst_1185) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4388 = stablehlo.multiply %4386, %344 : tensor<64xf32>
      %4389 = stablehlo.negate %4388 : tensor<64xf32>
      %4390 = stablehlo.multiply %4389, %331 : tensor<64xf32>
      %4391 = stablehlo.add %4387, %4390 : tensor<64xf32>
      %4392 = stablehlo.broadcast_in_dim %4388, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4393 = stablehlo.pad %4392, %cst_1186, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4394 = stablehlo.broadcast_in_dim %4391, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4395 = stablehlo.pad %4394, %cst_1187, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4396 = stablehlo.add %4393, %4395 : tensor<2x64xf32>
      %cst_1188 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4397 = stablehlo.broadcast_in_dim %cst_1188, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %4398 = stablehlo.divide %4396, %4397 : tensor<2x64xf32>
      %4399 = "stablehlo.all_reduce"(%4398) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %4400 = stablehlo.slice %4399 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %4401 = stablehlo.slice %4399 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_1189 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4402 = stablehlo.reduce(%4401 init: %cst_1189) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4403 = stablehlo.reduce(%4400 init: %cst_1190) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1191 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4404 = stablehlo.broadcast_in_dim %cst_1191, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4405 = stablehlo.divide %4402, %4404 : tensor<64xf32>
      %4406 = stablehlo.broadcast_in_dim %4405, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_1192 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4407 = stablehlo.broadcast_in_dim %cst_1192, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4408 = stablehlo.divide %4403, %4407 : tensor<64xf32>
      %4409 = stablehlo.broadcast_in_dim %4408, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %4410 = stablehlo.multiply %4406, %312 : tensor<256x56x56x64xf32>
      %4411 = stablehlo.add %4409, %4410 : tensor<256x56x56x64xf32>
      %4412 = stablehlo.convert %4411 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %4413 = stablehlo.add %4385, %4412 : tensor<256x56x56x64xf16>
      %4414 = stablehlo.convolution(%304, %4413) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %4415 = stablehlo.reverse %307, dims = [0, 1] : tensor<3x3x64x64xf16>
      %4416 = stablehlo.convolution(%4413, %4415) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %4417 = stablehlo.convert %4414 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %4418 = stablehlo.add %2717, %4417 : tensor<3x3x64x64xf32>
      %cst_1193 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4419 = stablehlo.broadcast_in_dim %cst_1193, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %4420 = stablehlo.select %306, %4416, %4419 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %4421 = stablehlo.convert %4420 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_1194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4422 = stablehlo.reduce(%4421 init: %cst_1194) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4423 = stablehlo.reshape %4422 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4424 = stablehlo.reshape %4423 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4425 = stablehlo.multiply %289, %4421 : tensor<256x56x56x64xf32>
      %cst_1195 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4426 = stablehlo.reduce(%4425 init: %cst_1195) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4427 = stablehlo.reshape %4426 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4428 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %4429 = stablehlo.multiply %4421, %4428 : tensor<256x56x56x64xf32>
      %4430 = stablehlo.multiply %292, %4427 : tensor<1x1x1x64xf32>
      %4431 = stablehlo.multiply %4427, %296 : tensor<1x1x1x64xf32>
      %4432 = stablehlo.reshape %4430 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4433 = stablehlo.multiply %4431, %295 : tensor<1x1x1x64xf32>
      %4434 = stablehlo.negate %4429 : tensor<256x56x56x64xf32>
      %cst_1196 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4435 = stablehlo.reduce(%4434 init: %cst_1196) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4436 = stablehlo.reshape %4435 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4437 = stablehlo.convert %4429 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_1197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4438 = stablehlo.reduce(%4433 init: %cst_1197) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1198 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4439 = stablehlo.reduce(%4436 init: %cst_1198) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4440 = stablehlo.multiply %4438, %274 : tensor<64xf32>
      %4441 = stablehlo.negate %4440 : tensor<64xf32>
      %4442 = stablehlo.multiply %4441, %261 : tensor<64xf32>
      %4443 = stablehlo.add %4439, %4442 : tensor<64xf32>
      %4444 = stablehlo.broadcast_in_dim %4440, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4445 = stablehlo.pad %4444, %cst_1199, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4446 = stablehlo.broadcast_in_dim %4443, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1200 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4447 = stablehlo.pad %4446, %cst_1200, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4448 = stablehlo.add %4445, %4447 : tensor<2x64xf32>
      %cst_1201 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4449 = stablehlo.broadcast_in_dim %cst_1201, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %4450 = stablehlo.divide %4448, %4449 : tensor<2x64xf32>
      %4451 = "stablehlo.all_reduce"(%4450) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %4452 = stablehlo.slice %4451 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %4453 = stablehlo.slice %4451 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_1202 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4454 = stablehlo.reduce(%4453 init: %cst_1202) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1203 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4455 = stablehlo.reduce(%4452 init: %cst_1203) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1204 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4456 = stablehlo.broadcast_in_dim %cst_1204, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4457 = stablehlo.divide %4454, %4456 : tensor<64xf32>
      %4458 = stablehlo.broadcast_in_dim %4457, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_1205 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4459 = stablehlo.broadcast_in_dim %cst_1205, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4460 = stablehlo.divide %4455, %4459 : tensor<64xf32>
      %4461 = stablehlo.broadcast_in_dim %4460, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %4462 = stablehlo.multiply %4458, %242 : tensor<256x56x56x64xf32>
      %4463 = stablehlo.add %4461, %4462 : tensor<256x56x56x64xf32>
      %4464 = stablehlo.convert %4463 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %4465 = stablehlo.add %4437, %4464 : tensor<256x56x56x64xf16>
      %4466 = stablehlo.convolution(%234, %4465) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %4467 = stablehlo.reverse %237, dims = [0, 1] : tensor<3x3x64x64xf16>
      %4468 = stablehlo.convolution(%4465, %4467) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %4469 = stablehlo.add %4368, %4468 : tensor<256x56x56x64xf16>
      %4470 = stablehlo.convert %4466 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %4471 = stablehlo.add %2719, %4470 : tensor<3x3x64x64xf32>
      %cst_1206 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4472 = stablehlo.broadcast_in_dim %cst_1206, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %4473 = stablehlo.select %236, %4469, %4472 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %4474 = stablehlo.convert %4473 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_1207 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4475 = stablehlo.reduce(%4474 init: %cst_1207) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4476 = stablehlo.reshape %4475 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4477 = stablehlo.reshape %4476 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4478 = stablehlo.multiply %218, %4474 : tensor<256x56x56x64xf32>
      %cst_1208 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4479 = stablehlo.reduce(%4478 init: %cst_1208) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4480 = stablehlo.reshape %4479 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4481 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %4482 = stablehlo.multiply %4474, %4481 : tensor<256x56x56x64xf32>
      %4483 = stablehlo.multiply %221, %4480 : tensor<1x1x1x64xf32>
      %4484 = stablehlo.multiply %4480, %225 : tensor<1x1x1x64xf32>
      %4485 = stablehlo.reshape %4483 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4486 = stablehlo.multiply %4484, %224 : tensor<1x1x1x64xf32>
      %4487 = stablehlo.negate %4482 : tensor<256x56x56x64xf32>
      %cst_1209 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4488 = stablehlo.reduce(%4487 init: %cst_1209) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4489 = stablehlo.reshape %4488 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4490 = stablehlo.convert %4482 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_1210 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4491 = stablehlo.reduce(%4486 init: %cst_1210) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1211 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4492 = stablehlo.reduce(%4489 init: %cst_1211) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4493 = stablehlo.multiply %4491, %203 : tensor<64xf32>
      %4494 = stablehlo.negate %4493 : tensor<64xf32>
      %4495 = stablehlo.multiply %4494, %190 : tensor<64xf32>
      %4496 = stablehlo.add %4492, %4495 : tensor<64xf32>
      %4497 = stablehlo.broadcast_in_dim %4493, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1212 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4498 = stablehlo.pad %4497, %cst_1212, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4499 = stablehlo.broadcast_in_dim %4496, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1213 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4500 = stablehlo.pad %4499, %cst_1213, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4501 = stablehlo.add %4498, %4500 : tensor<2x64xf32>
      %cst_1214 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4502 = stablehlo.broadcast_in_dim %cst_1214, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %4503 = stablehlo.divide %4501, %4502 : tensor<2x64xf32>
      %4504 = "stablehlo.all_reduce"(%4503) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %4505 = stablehlo.slice %4504 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %4506 = stablehlo.slice %4504 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_1215 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4507 = stablehlo.reduce(%4506 init: %cst_1215) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1216 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4508 = stablehlo.reduce(%4505 init: %cst_1216) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1217 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4509 = stablehlo.broadcast_in_dim %cst_1217, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4510 = stablehlo.divide %4507, %4509 : tensor<64xf32>
      %4511 = stablehlo.broadcast_in_dim %4510, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_1218 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4512 = stablehlo.broadcast_in_dim %cst_1218, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4513 = stablehlo.divide %4508, %4512 : tensor<64xf32>
      %4514 = stablehlo.broadcast_in_dim %4513, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %4515 = stablehlo.multiply %4511, %171 : tensor<256x56x56x64xf32>
      %4516 = stablehlo.add %4514, %4515 : tensor<256x56x56x64xf32>
      %4517 = stablehlo.convert %4516 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %4518 = stablehlo.add %4490, %4517 : tensor<256x56x56x64xf16>
      %4519 = stablehlo.convolution(%163, %4518) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %4520 = stablehlo.reverse %166, dims = [0, 1] : tensor<3x3x64x64xf16>
      %4521 = stablehlo.convolution(%4518, %4520) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %4522 = stablehlo.convert %4519 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %4523 = stablehlo.add %2721, %4522 : tensor<3x3x64x64xf32>
      %cst_1219 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4524 = stablehlo.broadcast_in_dim %cst_1219, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %4525 = stablehlo.select %165, %4521, %4524 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %4526 = stablehlo.convert %4525 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_1220 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4527 = stablehlo.reduce(%4526 init: %cst_1220) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4528 = stablehlo.reshape %4527 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4529 = stablehlo.reshape %4528 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4530 = stablehlo.multiply %148, %4526 : tensor<256x56x56x64xf32>
      %cst_1221 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4531 = stablehlo.reduce(%4530 init: %cst_1221) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4532 = stablehlo.reshape %4531 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4533 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %4534 = stablehlo.multiply %4526, %4533 : tensor<256x56x56x64xf32>
      %4535 = stablehlo.multiply %151, %4532 : tensor<1x1x1x64xf32>
      %4536 = stablehlo.multiply %4532, %155 : tensor<1x1x1x64xf32>
      %4537 = stablehlo.reshape %4535 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4538 = stablehlo.multiply %4536, %154 : tensor<1x1x1x64xf32>
      %4539 = stablehlo.negate %4534 : tensor<256x56x56x64xf32>
      %cst_1222 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4540 = stablehlo.reduce(%4539 init: %cst_1222) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4541 = stablehlo.reshape %4540 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4542 = stablehlo.convert %4534 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_1223 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4543 = stablehlo.reduce(%4538 init: %cst_1223) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1224 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4544 = stablehlo.reduce(%4541 init: %cst_1224) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4545 = stablehlo.multiply %4543, %133 : tensor<64xf32>
      %4546 = stablehlo.negate %4545 : tensor<64xf32>
      %4547 = stablehlo.multiply %4546, %120 : tensor<64xf32>
      %4548 = stablehlo.add %4544, %4547 : tensor<64xf32>
      %4549 = stablehlo.broadcast_in_dim %4545, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1225 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4550 = stablehlo.pad %4549, %cst_1225, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4551 = stablehlo.broadcast_in_dim %4548, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4552 = stablehlo.pad %4551, %cst_1226, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4553 = stablehlo.add %4550, %4552 : tensor<2x64xf32>
      %cst_1227 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4554 = stablehlo.broadcast_in_dim %cst_1227, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %4555 = stablehlo.divide %4553, %4554 : tensor<2x64xf32>
      %4556 = "stablehlo.all_reduce"(%4555) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %4557 = stablehlo.slice %4556 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %4558 = stablehlo.slice %4556 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_1228 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4559 = stablehlo.reduce(%4558 init: %cst_1228) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1229 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4560 = stablehlo.reduce(%4557 init: %cst_1229) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1230 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4561 = stablehlo.broadcast_in_dim %cst_1230, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4562 = stablehlo.divide %4559, %4561 : tensor<64xf32>
      %4563 = stablehlo.broadcast_in_dim %4562, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_1231 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %4564 = stablehlo.broadcast_in_dim %cst_1231, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4565 = stablehlo.divide %4560, %4564 : tensor<64xf32>
      %4566 = stablehlo.broadcast_in_dim %4565, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %4567 = stablehlo.multiply %4563, %101 : tensor<256x56x56x64xf32>
      %4568 = stablehlo.add %4566, %4567 : tensor<256x56x56x64xf32>
      %4569 = stablehlo.convert %4568 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %4570 = stablehlo.add %4542, %4569 : tensor<256x56x56x64xf16>
      %4571 = stablehlo.convolution(%95, %4570) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %4572 = stablehlo.reverse %96, dims = [0, 1] : tensor<3x3x64x64xf16>
      %4573 = stablehlo.convolution(%4570, %4572) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %4574 = stablehlo.add %4473, %4573 : tensor<256x56x56x64xf16>
      %4575 = stablehlo.convert %4571 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %4576 = stablehlo.add %2723, %4575 : tensor<3x3x64x64xf32>
      %cst_1232 = stablehlo.constant dense<0xFC00> : tensor<f16>
      %4577 = stablehlo.pad %91, %cst_1232, low = [0, 0, 0, 0], high = [0, 1, 1, 0], interior = [0, 0, 0, 0] : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x113x113x64xf16>
      %cst_1233 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4578 = "stablehlo.select_and_scatter"(%4577, %4574, %cst_1233) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg596: tensor<f16>, %arg597: tensor<f16>):
        %7234 = stablehlo.compare  GE, %arg596, %arg597,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
        stablehlo.return %7234 : tensor<i1>
      }, {
      ^bb0(%arg596: tensor<f16>, %arg597: tensor<f16>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f16>
        stablehlo.return %7234 : tensor<f16>
      }) : (tensor<256x113x113x64xf16>, tensor<256x56x56x64xf16>, tensor<f16>) -> tensor<256x113x113x64xf16>
      %4579 = stablehlo.slice %4578 [0:256, 0:112, 0:112, 0:64] : (tensor<256x113x113x64xf16>) -> tensor<256x112x112x64xf16>
      %cst_1234 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4580 = stablehlo.broadcast_in_dim %cst_1234, dims = [] : (tensor<f16>) -> tensor<256x112x112x64xf16>
      %4581 = stablehlo.select %93, %4579, %4580 : tensor<256x112x112x64xi1>, tensor<256x112x112x64xf16>
      %4582 = stablehlo.convert %4581 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
      %cst_1235 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4583 = stablehlo.reduce(%4582 init: %cst_1235) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4584 = stablehlo.reshape %4583 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4585 = stablehlo.reshape %4584 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4586 = stablehlo.multiply %76, %4582 : tensor<256x112x112x64xf32>
      %cst_1236 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4587 = stablehlo.reduce(%4586 init: %cst_1236) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4588 = stablehlo.reshape %4587 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4589 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %4590 = stablehlo.multiply %4582, %4589 : tensor<256x112x112x64xf32>
      %4591 = stablehlo.multiply %79, %4588 : tensor<1x1x1x64xf32>
      %4592 = stablehlo.multiply %4588, %83 : tensor<1x1x1x64xf32>
      %4593 = stablehlo.reshape %4591 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %4594 = stablehlo.multiply %4592, %82 : tensor<1x1x1x64xf32>
      %4595 = stablehlo.negate %4590 : tensor<256x112x112x64xf32>
      %cst_1237 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4596 = stablehlo.reduce(%4595 init: %cst_1237) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4597 = stablehlo.reshape %4596 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %4598 = stablehlo.convert %4590 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %cst_1238 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4599 = stablehlo.reduce(%4594 init: %cst_1238) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1239 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4600 = stablehlo.reduce(%4597 init: %cst_1239) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %4601 = stablehlo.multiply %4599, %61 : tensor<64xf32>
      %4602 = stablehlo.negate %4601 : tensor<64xf32>
      %4603 = stablehlo.multiply %4602, %48 : tensor<64xf32>
      %4604 = stablehlo.add %4600, %4603 : tensor<64xf32>
      %4605 = stablehlo.broadcast_in_dim %4601, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1240 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4606 = stablehlo.pad %4605, %cst_1240, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4607 = stablehlo.broadcast_in_dim %4604, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_1241 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4608 = stablehlo.pad %4607, %cst_1241, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %4609 = stablehlo.add %4606, %4608 : tensor<2x64xf32>
      %cst_1242 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %4610 = stablehlo.broadcast_in_dim %cst_1242, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %4611 = stablehlo.divide %4609, %4610 : tensor<2x64xf32>
      %4612 = "stablehlo.all_reduce"(%4611) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %4613 = stablehlo.slice %4612 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %4614 = stablehlo.slice %4612 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_1243 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4615 = stablehlo.reduce(%4614 init: %cst_1243) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1244 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4616 = stablehlo.reduce(%4613 init: %cst_1244) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_1245 = stablehlo.constant dense<0x4A440000> : tensor<f32>
      %4617 = stablehlo.broadcast_in_dim %cst_1245, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4618 = stablehlo.divide %4615, %4617 : tensor<64xf32>
      %4619 = stablehlo.broadcast_in_dim %4618, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %cst_1246 = stablehlo.constant dense<0x4A440000> : tensor<f32>
      %4620 = stablehlo.broadcast_in_dim %cst_1246, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4621 = stablehlo.divide %4616, %4620 : tensor<64xf32>
      %4622 = stablehlo.broadcast_in_dim %4621, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %4623 = stablehlo.multiply %4619, %29 : tensor<256x112x112x64xf32>
      %4624 = stablehlo.add %4622, %4623 : tensor<256x112x112x64xf32>
      %4625 = stablehlo.convert %4624 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %4626 = stablehlo.add %4598, %4625 : tensor<256x112x112x64xf16>
      %4627 = stablehlo.convolution(%1, %4626) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf16>, tensor<256x112x112x64xf16>) -> tensor<7x7x3x64xf16>
      %4628 = stablehlo.convert %4627 : (tensor<7x7x3x64xf16>) -> tensor<7x7x3x64xf32>
      %4629 = stablehlo.add %2653, %4628 : tensor<7x7x3x64xf32>
      %4630 = stablehlo.convert %arg593 : tensor<f32>
      %4631 = stablehlo.broadcast_in_dim %4630, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %4632 = stablehlo.divide %2739, %4631 : tensor<1000xf32>
      %4633 = stablehlo.convert %arg593 : tensor<f32>
      %4634 = stablehlo.broadcast_in_dim %4633, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %4635 = stablehlo.divide %2741, %4634 : tensor<512x1000xf32>
      %4636 = stablehlo.convert %arg593 : tensor<f32>
      %4637 = stablehlo.broadcast_in_dim %4636, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4638 = stablehlo.divide %4529, %4637 : tensor<64xf32>
      %4639 = stablehlo.convert %arg593 : tensor<f32>
      %4640 = stablehlo.broadcast_in_dim %4639, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4641 = stablehlo.divide %4537, %4640 : tensor<64xf32>
      %4642 = stablehlo.convert %arg593 : tensor<f32>
      %4643 = stablehlo.broadcast_in_dim %4642, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4644 = stablehlo.divide %4477, %4643 : tensor<64xf32>
      %4645 = stablehlo.convert %arg593 : tensor<f32>
      %4646 = stablehlo.broadcast_in_dim %4645, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4647 = stablehlo.divide %4485, %4646 : tensor<64xf32>
      %4648 = stablehlo.convert %arg593 : tensor<f32>
      %4649 = stablehlo.broadcast_in_dim %4648, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %4650 = stablehlo.divide %4576, %4649 : tensor<3x3x64x64xf32>
      %4651 = stablehlo.convert %arg593 : tensor<f32>
      %4652 = stablehlo.broadcast_in_dim %4651, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %4653 = stablehlo.divide %4523, %4652 : tensor<3x3x64x64xf32>
      %4654 = stablehlo.convert %arg593 : tensor<f32>
      %4655 = stablehlo.broadcast_in_dim %4654, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4656 = stablehlo.divide %4424, %4655 : tensor<64xf32>
      %4657 = stablehlo.convert %arg593 : tensor<f32>
      %4658 = stablehlo.broadcast_in_dim %4657, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4659 = stablehlo.divide %4432, %4658 : tensor<64xf32>
      %4660 = stablehlo.convert %arg593 : tensor<f32>
      %4661 = stablehlo.broadcast_in_dim %4660, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4662 = stablehlo.divide %4372, %4661 : tensor<64xf32>
      %4663 = stablehlo.convert %arg593 : tensor<f32>
      %4664 = stablehlo.broadcast_in_dim %4663, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4665 = stablehlo.divide %4380, %4664 : tensor<64xf32>
      %4666 = stablehlo.convert %arg593 : tensor<f32>
      %4667 = stablehlo.broadcast_in_dim %4666, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %4668 = stablehlo.divide %4471, %4667 : tensor<3x3x64x64xf32>
      %4669 = stablehlo.convert %arg593 : tensor<f32>
      %4670 = stablehlo.broadcast_in_dim %4669, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %4671 = stablehlo.divide %4418, %4670 : tensor<3x3x64x64xf32>
      %4672 = stablehlo.convert %arg593 : tensor<f32>
      %4673 = stablehlo.broadcast_in_dim %4672, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4674 = stablehlo.divide %3379, %4673 : tensor<256xf32>
      %4675 = stablehlo.convert %arg593 : tensor<f32>
      %4676 = stablehlo.broadcast_in_dim %4675, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4677 = stablehlo.divide %3387, %4676 : tensor<256xf32>
      %4678 = stablehlo.convert %arg593 : tensor<f32>
      %4679 = stablehlo.broadcast_in_dim %4678, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4680 = stablehlo.divide %3327, %4679 : tensor<256xf32>
      %4681 = stablehlo.convert %arg593 : tensor<f32>
      %4682 = stablehlo.broadcast_in_dim %4681, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4683 = stablehlo.divide %3335, %4682 : tensor<256xf32>
      %4684 = stablehlo.convert %arg593 : tensor<f32>
      %4685 = stablehlo.broadcast_in_dim %4684, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4686 = stablehlo.divide %3426, %4685 : tensor<3x3x256x256xf32>
      %4687 = stablehlo.convert %arg593 : tensor<f32>
      %4688 = stablehlo.broadcast_in_dim %4687, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4689 = stablehlo.divide %3373, %4688 : tensor<3x3x256x256xf32>
      %4690 = stablehlo.convert %arg593 : tensor<f32>
      %4691 = stablehlo.broadcast_in_dim %4690, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4692 = stablehlo.divide %3274, %4691 : tensor<256xf32>
      %4693 = stablehlo.convert %arg593 : tensor<f32>
      %4694 = stablehlo.broadcast_in_dim %4693, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4695 = stablehlo.divide %3282, %4694 : tensor<256xf32>
      %4696 = stablehlo.convert %arg593 : tensor<f32>
      %4697 = stablehlo.broadcast_in_dim %4696, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4698 = stablehlo.divide %3222, %4697 : tensor<256xf32>
      %4699 = stablehlo.convert %arg593 : tensor<f32>
      %4700 = stablehlo.broadcast_in_dim %4699, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4701 = stablehlo.divide %3230, %4700 : tensor<256xf32>
      %4702 = stablehlo.convert %arg593 : tensor<f32>
      %4703 = stablehlo.broadcast_in_dim %4702, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4704 = stablehlo.divide %3321, %4703 : tensor<3x3x256x256xf32>
      %4705 = stablehlo.convert %arg593 : tensor<f32>
      %4706 = stablehlo.broadcast_in_dim %4705, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4707 = stablehlo.divide %3268, %4706 : tensor<3x3x256x256xf32>
      %4708 = stablehlo.convert %arg593 : tensor<f32>
      %4709 = stablehlo.broadcast_in_dim %4708, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4710 = stablehlo.divide %3169, %4709 : tensor<256xf32>
      %4711 = stablehlo.convert %arg593 : tensor<f32>
      %4712 = stablehlo.broadcast_in_dim %4711, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4713 = stablehlo.divide %3177, %4712 : tensor<256xf32>
      %4714 = stablehlo.convert %arg593 : tensor<f32>
      %4715 = stablehlo.broadcast_in_dim %4714, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4716 = stablehlo.divide %3117, %4715 : tensor<256xf32>
      %4717 = stablehlo.convert %arg593 : tensor<f32>
      %4718 = stablehlo.broadcast_in_dim %4717, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4719 = stablehlo.divide %3125, %4718 : tensor<256xf32>
      %4720 = stablehlo.convert %arg593 : tensor<f32>
      %4721 = stablehlo.broadcast_in_dim %4720, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4722 = stablehlo.divide %3216, %4721 : tensor<3x3x256x256xf32>
      %4723 = stablehlo.convert %arg593 : tensor<f32>
      %4724 = stablehlo.broadcast_in_dim %4723, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4725 = stablehlo.divide %3163, %4724 : tensor<3x3x256x256xf32>
      %4726 = stablehlo.convert %arg593 : tensor<f32>
      %4727 = stablehlo.broadcast_in_dim %4726, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4728 = stablehlo.divide %3064, %4727 : tensor<512xf32>
      %4729 = stablehlo.convert %arg593 : tensor<f32>
      %4730 = stablehlo.broadcast_in_dim %4729, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4731 = stablehlo.divide %3072, %4730 : tensor<512xf32>
      %4732 = stablehlo.convert %arg593 : tensor<f32>
      %4733 = stablehlo.broadcast_in_dim %4732, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4734 = stablehlo.divide %3012, %4733 : tensor<512xf32>
      %4735 = stablehlo.convert %arg593 : tensor<f32>
      %4736 = stablehlo.broadcast_in_dim %4735, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4737 = stablehlo.divide %3020, %4736 : tensor<512xf32>
      %4738 = stablehlo.convert %arg593 : tensor<f32>
      %4739 = stablehlo.broadcast_in_dim %4738, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %4740 = stablehlo.divide %3111, %4739 : tensor<3x3x256x512xf32>
      %4741 = stablehlo.convert %arg593 : tensor<f32>
      %4742 = stablehlo.broadcast_in_dim %4741, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %4743 = stablehlo.divide %3058, %4742 : tensor<3x3x512x512xf32>
      %4744 = stablehlo.convert %arg593 : tensor<f32>
      %4745 = stablehlo.broadcast_in_dim %4744, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %4746 = stablehlo.divide %3008, %4745 : tensor<1x1x256x512xf32>
      %4747 = stablehlo.convert %arg593 : tensor<f32>
      %4748 = stablehlo.broadcast_in_dim %4747, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4749 = stablehlo.divide %2962, %4748 : tensor<512xf32>
      %4750 = stablehlo.convert %arg593 : tensor<f32>
      %4751 = stablehlo.broadcast_in_dim %4750, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4752 = stablehlo.divide %2970, %4751 : tensor<512xf32>
      %4753 = stablehlo.convert %arg593 : tensor<f32>
      %4754 = stablehlo.broadcast_in_dim %4753, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4755 = stablehlo.divide %2909, %4754 : tensor<512xf32>
      %4756 = stablehlo.convert %arg593 : tensor<f32>
      %4757 = stablehlo.broadcast_in_dim %4756, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4758 = stablehlo.divide %2917, %4757 : tensor<512xf32>
      %4759 = stablehlo.convert %arg593 : tensor<f32>
      %4760 = stablehlo.broadcast_in_dim %4759, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4761 = stablehlo.divide %2857, %4760 : tensor<512xf32>
      %4762 = stablehlo.convert %arg593 : tensor<f32>
      %4763 = stablehlo.broadcast_in_dim %4762, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4764 = stablehlo.divide %2865, %4763 : tensor<512xf32>
      %4765 = stablehlo.convert %arg593 : tensor<f32>
      %4766 = stablehlo.broadcast_in_dim %4765, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %4767 = stablehlo.divide %2956, %4766 : tensor<3x3x512x512xf32>
      %4768 = stablehlo.convert %arg593 : tensor<f32>
      %4769 = stablehlo.broadcast_in_dim %4768, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %4770 = stablehlo.divide %2903, %4769 : tensor<3x3x512x512xf32>
      %4771 = stablehlo.convert %arg593 : tensor<f32>
      %4772 = stablehlo.broadcast_in_dim %4771, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4773 = stablehlo.divide %2804, %4772 : tensor<512xf32>
      %4774 = stablehlo.convert %arg593 : tensor<f32>
      %4775 = stablehlo.broadcast_in_dim %4774, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4776 = stablehlo.divide %2812, %4775 : tensor<512xf32>
      %4777 = stablehlo.convert %arg593 : tensor<f32>
      %4778 = stablehlo.broadcast_in_dim %4777, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4779 = stablehlo.divide %2752, %4778 : tensor<512xf32>
      %4780 = stablehlo.convert %arg593 : tensor<f32>
      %4781 = stablehlo.broadcast_in_dim %4780, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %4782 = stablehlo.divide %2760, %4781 : tensor<512xf32>
      %4783 = stablehlo.convert %arg593 : tensor<f32>
      %4784 = stablehlo.broadcast_in_dim %4783, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %4785 = stablehlo.divide %2851, %4784 : tensor<3x3x512x512xf32>
      %4786 = stablehlo.convert %arg593 : tensor<f32>
      %4787 = stablehlo.broadcast_in_dim %4786, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %4788 = stablehlo.divide %2798, %4787 : tensor<3x3x512x512xf32>
      %4789 = stablehlo.convert %arg593 : tensor<f32>
      %4790 = stablehlo.broadcast_in_dim %4789, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4791 = stablehlo.divide %4319, %4790 : tensor<64xf32>
      %4792 = stablehlo.convert %arg593 : tensor<f32>
      %4793 = stablehlo.broadcast_in_dim %4792, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4794 = stablehlo.divide %4327, %4793 : tensor<64xf32>
      %4795 = stablehlo.convert %arg593 : tensor<f32>
      %4796 = stablehlo.broadcast_in_dim %4795, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4797 = stablehlo.divide %4267, %4796 : tensor<64xf32>
      %4798 = stablehlo.convert %arg593 : tensor<f32>
      %4799 = stablehlo.broadcast_in_dim %4798, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4800 = stablehlo.divide %4275, %4799 : tensor<64xf32>
      %4801 = stablehlo.convert %arg593 : tensor<f32>
      %4802 = stablehlo.broadcast_in_dim %4801, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %4803 = stablehlo.divide %4366, %4802 : tensor<3x3x64x64xf32>
      %4804 = stablehlo.convert %arg593 : tensor<f32>
      %4805 = stablehlo.broadcast_in_dim %4804, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %4806 = stablehlo.divide %4313, %4805 : tensor<3x3x64x64xf32>
      %4807 = stablehlo.convert %arg593 : tensor<f32>
      %4808 = stablehlo.broadcast_in_dim %4807, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4809 = stablehlo.divide %4214, %4808 : tensor<128xf32>
      %4810 = stablehlo.convert %arg593 : tensor<f32>
      %4811 = stablehlo.broadcast_in_dim %4810, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4812 = stablehlo.divide %4222, %4811 : tensor<128xf32>
      %4813 = stablehlo.convert %arg593 : tensor<f32>
      %4814 = stablehlo.broadcast_in_dim %4813, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4815 = stablehlo.divide %4162, %4814 : tensor<128xf32>
      %4816 = stablehlo.convert %arg593 : tensor<f32>
      %4817 = stablehlo.broadcast_in_dim %4816, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4818 = stablehlo.divide %4170, %4817 : tensor<128xf32>
      %4819 = stablehlo.convert %arg593 : tensor<f32>
      %4820 = stablehlo.broadcast_in_dim %4819, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %4821 = stablehlo.divide %4261, %4820 : tensor<3x3x64x128xf32>
      %4822 = stablehlo.convert %arg593 : tensor<f32>
      %4823 = stablehlo.broadcast_in_dim %4822, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %4824 = stablehlo.divide %4208, %4823 : tensor<3x3x128x128xf32>
      %4825 = stablehlo.convert %arg593 : tensor<f32>
      %4826 = stablehlo.broadcast_in_dim %4825, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %4827 = stablehlo.divide %4158, %4826 : tensor<1x1x64x128xf32>
      %4828 = stablehlo.convert %arg593 : tensor<f32>
      %4829 = stablehlo.broadcast_in_dim %4828, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4830 = stablehlo.divide %4112, %4829 : tensor<128xf32>
      %4831 = stablehlo.convert %arg593 : tensor<f32>
      %4832 = stablehlo.broadcast_in_dim %4831, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4833 = stablehlo.divide %4120, %4832 : tensor<128xf32>
      %4834 = stablehlo.convert %arg593 : tensor<f32>
      %4835 = stablehlo.broadcast_in_dim %4834, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4836 = stablehlo.divide %4059, %4835 : tensor<128xf32>
      %4837 = stablehlo.convert %arg593 : tensor<f32>
      %4838 = stablehlo.broadcast_in_dim %4837, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4839 = stablehlo.divide %4067, %4838 : tensor<128xf32>
      %4840 = stablehlo.convert %arg593 : tensor<f32>
      %4841 = stablehlo.broadcast_in_dim %4840, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4842 = stablehlo.divide %4007, %4841 : tensor<128xf32>
      %4843 = stablehlo.convert %arg593 : tensor<f32>
      %4844 = stablehlo.broadcast_in_dim %4843, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4845 = stablehlo.divide %4015, %4844 : tensor<128xf32>
      %4846 = stablehlo.convert %arg593 : tensor<f32>
      %4847 = stablehlo.broadcast_in_dim %4846, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %4848 = stablehlo.divide %4106, %4847 : tensor<3x3x128x128xf32>
      %4849 = stablehlo.convert %arg593 : tensor<f32>
      %4850 = stablehlo.broadcast_in_dim %4849, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %4851 = stablehlo.divide %4053, %4850 : tensor<3x3x128x128xf32>
      %4852 = stablehlo.convert %arg593 : tensor<f32>
      %4853 = stablehlo.broadcast_in_dim %4852, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4854 = stablehlo.divide %3954, %4853 : tensor<128xf32>
      %4855 = stablehlo.convert %arg593 : tensor<f32>
      %4856 = stablehlo.broadcast_in_dim %4855, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4857 = stablehlo.divide %3962, %4856 : tensor<128xf32>
      %4858 = stablehlo.convert %arg593 : tensor<f32>
      %4859 = stablehlo.broadcast_in_dim %4858, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4860 = stablehlo.divide %3902, %4859 : tensor<128xf32>
      %4861 = stablehlo.convert %arg593 : tensor<f32>
      %4862 = stablehlo.broadcast_in_dim %4861, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4863 = stablehlo.divide %3910, %4862 : tensor<128xf32>
      %4864 = stablehlo.convert %arg593 : tensor<f32>
      %4865 = stablehlo.broadcast_in_dim %4864, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %4866 = stablehlo.divide %4001, %4865 : tensor<3x3x128x128xf32>
      %4867 = stablehlo.convert %arg593 : tensor<f32>
      %4868 = stablehlo.broadcast_in_dim %4867, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %4869 = stablehlo.divide %3948, %4868 : tensor<3x3x128x128xf32>
      %4870 = stablehlo.convert %arg593 : tensor<f32>
      %4871 = stablehlo.broadcast_in_dim %4870, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4872 = stablehlo.divide %3849, %4871 : tensor<128xf32>
      %4873 = stablehlo.convert %arg593 : tensor<f32>
      %4874 = stablehlo.broadcast_in_dim %4873, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4875 = stablehlo.divide %3857, %4874 : tensor<128xf32>
      %4876 = stablehlo.convert %arg593 : tensor<f32>
      %4877 = stablehlo.broadcast_in_dim %4876, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4878 = stablehlo.divide %3797, %4877 : tensor<128xf32>
      %4879 = stablehlo.convert %arg593 : tensor<f32>
      %4880 = stablehlo.broadcast_in_dim %4879, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %4881 = stablehlo.divide %3805, %4880 : tensor<128xf32>
      %4882 = stablehlo.convert %arg593 : tensor<f32>
      %4883 = stablehlo.broadcast_in_dim %4882, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %4884 = stablehlo.divide %3896, %4883 : tensor<3x3x128x128xf32>
      %4885 = stablehlo.convert %arg593 : tensor<f32>
      %4886 = stablehlo.broadcast_in_dim %4885, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %4887 = stablehlo.divide %3843, %4886 : tensor<3x3x128x128xf32>
      %4888 = stablehlo.convert %arg593 : tensor<f32>
      %4889 = stablehlo.broadcast_in_dim %4888, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4890 = stablehlo.divide %3744, %4889 : tensor<256xf32>
      %4891 = stablehlo.convert %arg593 : tensor<f32>
      %4892 = stablehlo.broadcast_in_dim %4891, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4893 = stablehlo.divide %3752, %4892 : tensor<256xf32>
      %4894 = stablehlo.convert %arg593 : tensor<f32>
      %4895 = stablehlo.broadcast_in_dim %4894, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4896 = stablehlo.divide %3692, %4895 : tensor<256xf32>
      %4897 = stablehlo.convert %arg593 : tensor<f32>
      %4898 = stablehlo.broadcast_in_dim %4897, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4899 = stablehlo.divide %3700, %4898 : tensor<256xf32>
      %4900 = stablehlo.convert %arg593 : tensor<f32>
      %4901 = stablehlo.broadcast_in_dim %4900, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %4902 = stablehlo.divide %3791, %4901 : tensor<3x3x128x256xf32>
      %4903 = stablehlo.convert %arg593 : tensor<f32>
      %4904 = stablehlo.broadcast_in_dim %4903, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4905 = stablehlo.divide %3738, %4904 : tensor<3x3x256x256xf32>
      %4906 = stablehlo.convert %arg593 : tensor<f32>
      %4907 = stablehlo.broadcast_in_dim %4906, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %4908 = stablehlo.divide %3688, %4907 : tensor<1x1x128x256xf32>
      %4909 = stablehlo.convert %arg593 : tensor<f32>
      %4910 = stablehlo.broadcast_in_dim %4909, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4911 = stablehlo.divide %3642, %4910 : tensor<256xf32>
      %4912 = stablehlo.convert %arg593 : tensor<f32>
      %4913 = stablehlo.broadcast_in_dim %4912, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4914 = stablehlo.divide %3650, %4913 : tensor<256xf32>
      %4915 = stablehlo.convert %arg593 : tensor<f32>
      %4916 = stablehlo.broadcast_in_dim %4915, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4917 = stablehlo.divide %3589, %4916 : tensor<256xf32>
      %4918 = stablehlo.convert %arg593 : tensor<f32>
      %4919 = stablehlo.broadcast_in_dim %4918, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4920 = stablehlo.divide %3597, %4919 : tensor<256xf32>
      %4921 = stablehlo.convert %arg593 : tensor<f32>
      %4922 = stablehlo.broadcast_in_dim %4921, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4923 = stablehlo.divide %3537, %4922 : tensor<256xf32>
      %4924 = stablehlo.convert %arg593 : tensor<f32>
      %4925 = stablehlo.broadcast_in_dim %4924, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4926 = stablehlo.divide %3545, %4925 : tensor<256xf32>
      %4927 = stablehlo.convert %arg593 : tensor<f32>
      %4928 = stablehlo.broadcast_in_dim %4927, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4929 = stablehlo.divide %3636, %4928 : tensor<3x3x256x256xf32>
      %4930 = stablehlo.convert %arg593 : tensor<f32>
      %4931 = stablehlo.broadcast_in_dim %4930, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4932 = stablehlo.divide %3583, %4931 : tensor<3x3x256x256xf32>
      %4933 = stablehlo.convert %arg593 : tensor<f32>
      %4934 = stablehlo.broadcast_in_dim %4933, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4935 = stablehlo.divide %3484, %4934 : tensor<256xf32>
      %4936 = stablehlo.convert %arg593 : tensor<f32>
      %4937 = stablehlo.broadcast_in_dim %4936, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4938 = stablehlo.divide %3492, %4937 : tensor<256xf32>
      %4939 = stablehlo.convert %arg593 : tensor<f32>
      %4940 = stablehlo.broadcast_in_dim %4939, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4941 = stablehlo.divide %3432, %4940 : tensor<256xf32>
      %4942 = stablehlo.convert %arg593 : tensor<f32>
      %4943 = stablehlo.broadcast_in_dim %4942, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %4944 = stablehlo.divide %3440, %4943 : tensor<256xf32>
      %4945 = stablehlo.convert %arg593 : tensor<f32>
      %4946 = stablehlo.broadcast_in_dim %4945, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4947 = stablehlo.divide %3531, %4946 : tensor<3x3x256x256xf32>
      %4948 = stablehlo.convert %arg593 : tensor<f32>
      %4949 = stablehlo.broadcast_in_dim %4948, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %4950 = stablehlo.divide %3478, %4949 : tensor<3x3x256x256xf32>
      %4951 = stablehlo.convert %arg593 : tensor<f32>
      %4952 = stablehlo.broadcast_in_dim %4951, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4953 = stablehlo.divide %4585, %4952 : tensor<64xf32>
      %4954 = stablehlo.convert %arg593 : tensor<f32>
      %4955 = stablehlo.broadcast_in_dim %4954, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %4956 = stablehlo.divide %4593, %4955 : tensor<64xf32>
      %4957 = stablehlo.convert %arg593 : tensor<f32>
      %4958 = stablehlo.broadcast_in_dim %4957, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %4959 = stablehlo.divide %4629, %4958 : tensor<7x7x3x64xf32>
      %4960 = "stablehlo.all_reduce"(%4632) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<1000xf32>) -> tensor<1000xf32>
      %4961 = "stablehlo.all_reduce"(%4635) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512x1000xf32>) -> tensor<512x1000xf32>
      %4962 = "stablehlo.all_reduce"(%4638) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %4963 = "stablehlo.all_reduce"(%4641) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %4964 = "stablehlo.all_reduce"(%4644) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %4965 = "stablehlo.all_reduce"(%4647) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %4966 = "stablehlo.all_reduce"(%4650) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %4967 = "stablehlo.all_reduce"(%4653) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %4968 = "stablehlo.all_reduce"(%4656) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %4969 = "stablehlo.all_reduce"(%4659) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %4970 = "stablehlo.all_reduce"(%4662) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %4971 = "stablehlo.all_reduce"(%4665) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %4972 = "stablehlo.all_reduce"(%4668) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %4973 = "stablehlo.all_reduce"(%4671) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %4974 = "stablehlo.all_reduce"(%4674) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4975 = "stablehlo.all_reduce"(%4677) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4976 = "stablehlo.all_reduce"(%4680) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4977 = "stablehlo.all_reduce"(%4683) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4978 = "stablehlo.all_reduce"(%4686) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4979 = "stablehlo.all_reduce"(%4689) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4980 = "stablehlo.all_reduce"(%4692) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4981 = "stablehlo.all_reduce"(%4695) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4982 = "stablehlo.all_reduce"(%4698) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4983 = "stablehlo.all_reduce"(%4701) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4984 = "stablehlo.all_reduce"(%4704) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4985 = "stablehlo.all_reduce"(%4707) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4986 = "stablehlo.all_reduce"(%4710) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4987 = "stablehlo.all_reduce"(%4713) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4988 = "stablehlo.all_reduce"(%4716) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4989 = "stablehlo.all_reduce"(%4719) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %4990 = "stablehlo.all_reduce"(%4722) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4991 = "stablehlo.all_reduce"(%4725) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4992 = "stablehlo.all_reduce"(%4728) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %4993 = "stablehlo.all_reduce"(%4731) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %4994 = "stablehlo.all_reduce"(%4734) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %4995 = "stablehlo.all_reduce"(%4737) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %4996 = "stablehlo.all_reduce"(%4740) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
      %4997 = "stablehlo.all_reduce"(%4743) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %4998 = "stablehlo.all_reduce"(%4746) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
      %4999 = "stablehlo.all_reduce"(%4749) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5000 = "stablehlo.all_reduce"(%4752) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5001 = "stablehlo.all_reduce"(%4755) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5002 = "stablehlo.all_reduce"(%4758) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5003 = "stablehlo.all_reduce"(%4761) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5004 = "stablehlo.all_reduce"(%4764) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5005 = "stablehlo.all_reduce"(%4767) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %5006 = "stablehlo.all_reduce"(%4770) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %5007 = "stablehlo.all_reduce"(%4773) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5008 = "stablehlo.all_reduce"(%4776) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5009 = "stablehlo.all_reduce"(%4779) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5010 = "stablehlo.all_reduce"(%4782) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5011 = "stablehlo.all_reduce"(%4785) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %5012 = "stablehlo.all_reduce"(%4788) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %5013 = "stablehlo.all_reduce"(%4791) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5014 = "stablehlo.all_reduce"(%4794) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5015 = "stablehlo.all_reduce"(%4797) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5016 = "stablehlo.all_reduce"(%4800) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5017 = "stablehlo.all_reduce"(%4803) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %5018 = "stablehlo.all_reduce"(%4806) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %5019 = "stablehlo.all_reduce"(%4809) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5020 = "stablehlo.all_reduce"(%4812) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5021 = "stablehlo.all_reduce"(%4815) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5022 = "stablehlo.all_reduce"(%4818) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5023 = "stablehlo.all_reduce"(%4821) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
      %5024 = "stablehlo.all_reduce"(%4824) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %5025 = "stablehlo.all_reduce"(%4827) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
      %5026 = "stablehlo.all_reduce"(%4830) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5027 = "stablehlo.all_reduce"(%4833) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5028 = "stablehlo.all_reduce"(%4836) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5029 = "stablehlo.all_reduce"(%4839) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5030 = "stablehlo.all_reduce"(%4842) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5031 = "stablehlo.all_reduce"(%4845) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5032 = "stablehlo.all_reduce"(%4848) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %5033 = "stablehlo.all_reduce"(%4851) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %5034 = "stablehlo.all_reduce"(%4854) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5035 = "stablehlo.all_reduce"(%4857) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5036 = "stablehlo.all_reduce"(%4860) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5037 = "stablehlo.all_reduce"(%4863) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5038 = "stablehlo.all_reduce"(%4866) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %5039 = "stablehlo.all_reduce"(%4869) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %5040 = "stablehlo.all_reduce"(%4872) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5041 = "stablehlo.all_reduce"(%4875) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5042 = "stablehlo.all_reduce"(%4878) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5043 = "stablehlo.all_reduce"(%4881) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5044 = "stablehlo.all_reduce"(%4884) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %5045 = "stablehlo.all_reduce"(%4887) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %5046 = "stablehlo.all_reduce"(%4890) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5047 = "stablehlo.all_reduce"(%4893) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5048 = "stablehlo.all_reduce"(%4896) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5049 = "stablehlo.all_reduce"(%4899) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5050 = "stablehlo.all_reduce"(%4902) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
      %5051 = "stablehlo.all_reduce"(%4905) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %5052 = "stablehlo.all_reduce"(%4908) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
      %5053 = "stablehlo.all_reduce"(%4911) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5054 = "stablehlo.all_reduce"(%4914) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5055 = "stablehlo.all_reduce"(%4917) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5056 = "stablehlo.all_reduce"(%4920) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5057 = "stablehlo.all_reduce"(%4923) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5058 = "stablehlo.all_reduce"(%4926) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5059 = "stablehlo.all_reduce"(%4929) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %5060 = "stablehlo.all_reduce"(%4932) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %5061 = "stablehlo.all_reduce"(%4935) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5062 = "stablehlo.all_reduce"(%4938) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5063 = "stablehlo.all_reduce"(%4941) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5064 = "stablehlo.all_reduce"(%4944) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5065 = "stablehlo.all_reduce"(%4947) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %5066 = "stablehlo.all_reduce"(%4950) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %5067 = "stablehlo.all_reduce"(%4953) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5068 = "stablehlo.all_reduce"(%4956) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5069 = "stablehlo.all_reduce"(%4959) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
      %cst_1247 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5070 = stablehlo.broadcast_in_dim %cst_1247, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %5071 = stablehlo.divide %4960, %5070 : tensor<1000xf32>
      %cst_1248 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5072 = stablehlo.broadcast_in_dim %cst_1248, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %5073 = stablehlo.divide %4961, %5072 : tensor<512x1000xf32>
      %cst_1249 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5074 = stablehlo.broadcast_in_dim %cst_1249, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5075 = stablehlo.divide %4962, %5074 : tensor<64xf32>
      %cst_1250 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5076 = stablehlo.broadcast_in_dim %cst_1250, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5077 = stablehlo.divide %4963, %5076 : tensor<64xf32>
      %cst_1251 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5078 = stablehlo.broadcast_in_dim %cst_1251, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5079 = stablehlo.divide %4964, %5078 : tensor<64xf32>
      %cst_1252 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5080 = stablehlo.broadcast_in_dim %cst_1252, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5081 = stablehlo.divide %4965, %5080 : tensor<64xf32>
      %cst_1253 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5082 = stablehlo.broadcast_in_dim %cst_1253, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5083 = stablehlo.divide %4966, %5082 : tensor<3x3x64x64xf32>
      %cst_1254 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5084 = stablehlo.broadcast_in_dim %cst_1254, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5085 = stablehlo.divide %4967, %5084 : tensor<3x3x64x64xf32>
      %cst_1255 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5086 = stablehlo.broadcast_in_dim %cst_1255, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5087 = stablehlo.divide %4968, %5086 : tensor<64xf32>
      %cst_1256 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5088 = stablehlo.broadcast_in_dim %cst_1256, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5089 = stablehlo.divide %4969, %5088 : tensor<64xf32>
      %cst_1257 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5090 = stablehlo.broadcast_in_dim %cst_1257, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5091 = stablehlo.divide %4970, %5090 : tensor<64xf32>
      %cst_1258 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5092 = stablehlo.broadcast_in_dim %cst_1258, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5093 = stablehlo.divide %4971, %5092 : tensor<64xf32>
      %cst_1259 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5094 = stablehlo.broadcast_in_dim %cst_1259, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5095 = stablehlo.divide %4972, %5094 : tensor<3x3x64x64xf32>
      %cst_1260 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5096 = stablehlo.broadcast_in_dim %cst_1260, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5097 = stablehlo.divide %4973, %5096 : tensor<3x3x64x64xf32>
      %cst_1261 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5098 = stablehlo.broadcast_in_dim %cst_1261, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5099 = stablehlo.divide %4974, %5098 : tensor<256xf32>
      %cst_1262 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5100 = stablehlo.broadcast_in_dim %cst_1262, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5101 = stablehlo.divide %4975, %5100 : tensor<256xf32>
      %cst_1263 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5102 = stablehlo.broadcast_in_dim %cst_1263, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5103 = stablehlo.divide %4976, %5102 : tensor<256xf32>
      %cst_1264 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5104 = stablehlo.broadcast_in_dim %cst_1264, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5105 = stablehlo.divide %4977, %5104 : tensor<256xf32>
      %cst_1265 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5106 = stablehlo.broadcast_in_dim %cst_1265, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5107 = stablehlo.divide %4978, %5106 : tensor<3x3x256x256xf32>
      %cst_1266 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5108 = stablehlo.broadcast_in_dim %cst_1266, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5109 = stablehlo.divide %4979, %5108 : tensor<3x3x256x256xf32>
      %cst_1267 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5110 = stablehlo.broadcast_in_dim %cst_1267, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5111 = stablehlo.divide %4980, %5110 : tensor<256xf32>
      %cst_1268 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5112 = stablehlo.broadcast_in_dim %cst_1268, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5113 = stablehlo.divide %4981, %5112 : tensor<256xf32>
      %cst_1269 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5114 = stablehlo.broadcast_in_dim %cst_1269, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5115 = stablehlo.divide %4982, %5114 : tensor<256xf32>
      %cst_1270 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5116 = stablehlo.broadcast_in_dim %cst_1270, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5117 = stablehlo.divide %4983, %5116 : tensor<256xf32>
      %cst_1271 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5118 = stablehlo.broadcast_in_dim %cst_1271, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5119 = stablehlo.divide %4984, %5118 : tensor<3x3x256x256xf32>
      %cst_1272 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5120 = stablehlo.broadcast_in_dim %cst_1272, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5121 = stablehlo.divide %4985, %5120 : tensor<3x3x256x256xf32>
      %cst_1273 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5122 = stablehlo.broadcast_in_dim %cst_1273, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5123 = stablehlo.divide %4986, %5122 : tensor<256xf32>
      %cst_1274 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5124 = stablehlo.broadcast_in_dim %cst_1274, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5125 = stablehlo.divide %4987, %5124 : tensor<256xf32>
      %cst_1275 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5126 = stablehlo.broadcast_in_dim %cst_1275, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5127 = stablehlo.divide %4988, %5126 : tensor<256xf32>
      %cst_1276 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5128 = stablehlo.broadcast_in_dim %cst_1276, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5129 = stablehlo.divide %4989, %5128 : tensor<256xf32>
      %cst_1277 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5130 = stablehlo.broadcast_in_dim %cst_1277, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5131 = stablehlo.divide %4990, %5130 : tensor<3x3x256x256xf32>
      %cst_1278 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5132 = stablehlo.broadcast_in_dim %cst_1278, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5133 = stablehlo.divide %4991, %5132 : tensor<3x3x256x256xf32>
      %cst_1279 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5134 = stablehlo.broadcast_in_dim %cst_1279, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5135 = stablehlo.divide %4992, %5134 : tensor<512xf32>
      %cst_1280 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5136 = stablehlo.broadcast_in_dim %cst_1280, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5137 = stablehlo.divide %4993, %5136 : tensor<512xf32>
      %cst_1281 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5138 = stablehlo.broadcast_in_dim %cst_1281, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5139 = stablehlo.divide %4994, %5138 : tensor<512xf32>
      %cst_1282 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5140 = stablehlo.broadcast_in_dim %cst_1282, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5141 = stablehlo.divide %4995, %5140 : tensor<512xf32>
      %cst_1283 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5142 = stablehlo.broadcast_in_dim %cst_1283, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %5143 = stablehlo.divide %4996, %5142 : tensor<3x3x256x512xf32>
      %cst_1284 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5144 = stablehlo.broadcast_in_dim %cst_1284, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %5145 = stablehlo.divide %4997, %5144 : tensor<3x3x512x512xf32>
      %cst_1285 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5146 = stablehlo.broadcast_in_dim %cst_1285, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %5147 = stablehlo.divide %4998, %5146 : tensor<1x1x256x512xf32>
      %cst_1286 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5148 = stablehlo.broadcast_in_dim %cst_1286, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5149 = stablehlo.divide %4999, %5148 : tensor<512xf32>
      %cst_1287 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5150 = stablehlo.broadcast_in_dim %cst_1287, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5151 = stablehlo.divide %5000, %5150 : tensor<512xf32>
      %cst_1288 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5152 = stablehlo.broadcast_in_dim %cst_1288, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5153 = stablehlo.divide %5001, %5152 : tensor<512xf32>
      %cst_1289 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5154 = stablehlo.broadcast_in_dim %cst_1289, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5155 = stablehlo.divide %5002, %5154 : tensor<512xf32>
      %cst_1290 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5156 = stablehlo.broadcast_in_dim %cst_1290, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5157 = stablehlo.divide %5003, %5156 : tensor<512xf32>
      %cst_1291 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5158 = stablehlo.broadcast_in_dim %cst_1291, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5159 = stablehlo.divide %5004, %5158 : tensor<512xf32>
      %cst_1292 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5160 = stablehlo.broadcast_in_dim %cst_1292, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %5161 = stablehlo.divide %5005, %5160 : tensor<3x3x512x512xf32>
      %cst_1293 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5162 = stablehlo.broadcast_in_dim %cst_1293, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %5163 = stablehlo.divide %5006, %5162 : tensor<3x3x512x512xf32>
      %cst_1294 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5164 = stablehlo.broadcast_in_dim %cst_1294, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5165 = stablehlo.divide %5007, %5164 : tensor<512xf32>
      %cst_1295 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5166 = stablehlo.broadcast_in_dim %cst_1295, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5167 = stablehlo.divide %5008, %5166 : tensor<512xf32>
      %cst_1296 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5168 = stablehlo.broadcast_in_dim %cst_1296, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5169 = stablehlo.divide %5009, %5168 : tensor<512xf32>
      %cst_1297 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5170 = stablehlo.broadcast_in_dim %cst_1297, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5171 = stablehlo.divide %5010, %5170 : tensor<512xf32>
      %cst_1298 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5172 = stablehlo.broadcast_in_dim %cst_1298, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %5173 = stablehlo.divide %5011, %5172 : tensor<3x3x512x512xf32>
      %cst_1299 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5174 = stablehlo.broadcast_in_dim %cst_1299, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %5175 = stablehlo.divide %5012, %5174 : tensor<3x3x512x512xf32>
      %cst_1300 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5176 = stablehlo.broadcast_in_dim %cst_1300, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5177 = stablehlo.divide %5013, %5176 : tensor<64xf32>
      %cst_1301 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5178 = stablehlo.broadcast_in_dim %cst_1301, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5179 = stablehlo.divide %5014, %5178 : tensor<64xf32>
      %cst_1302 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5180 = stablehlo.broadcast_in_dim %cst_1302, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5181 = stablehlo.divide %5015, %5180 : tensor<64xf32>
      %cst_1303 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5182 = stablehlo.broadcast_in_dim %cst_1303, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5183 = stablehlo.divide %5016, %5182 : tensor<64xf32>
      %cst_1304 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5184 = stablehlo.broadcast_in_dim %cst_1304, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5185 = stablehlo.divide %5017, %5184 : tensor<3x3x64x64xf32>
      %cst_1305 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5186 = stablehlo.broadcast_in_dim %cst_1305, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5187 = stablehlo.divide %5018, %5186 : tensor<3x3x64x64xf32>
      %cst_1306 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5188 = stablehlo.broadcast_in_dim %cst_1306, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5189 = stablehlo.divide %5019, %5188 : tensor<128xf32>
      %cst_1307 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5190 = stablehlo.broadcast_in_dim %cst_1307, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5191 = stablehlo.divide %5020, %5190 : tensor<128xf32>
      %cst_1308 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5192 = stablehlo.broadcast_in_dim %cst_1308, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5193 = stablehlo.divide %5021, %5192 : tensor<128xf32>
      %cst_1309 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5194 = stablehlo.broadcast_in_dim %cst_1309, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5195 = stablehlo.divide %5022, %5194 : tensor<128xf32>
      %cst_1310 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5196 = stablehlo.broadcast_in_dim %cst_1310, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %5197 = stablehlo.divide %5023, %5196 : tensor<3x3x64x128xf32>
      %cst_1311 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5198 = stablehlo.broadcast_in_dim %cst_1311, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %5199 = stablehlo.divide %5024, %5198 : tensor<3x3x128x128xf32>
      %cst_1312 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5200 = stablehlo.broadcast_in_dim %cst_1312, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %5201 = stablehlo.divide %5025, %5200 : tensor<1x1x64x128xf32>
      %cst_1313 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5202 = stablehlo.broadcast_in_dim %cst_1313, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5203 = stablehlo.divide %5026, %5202 : tensor<128xf32>
      %cst_1314 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5204 = stablehlo.broadcast_in_dim %cst_1314, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5205 = stablehlo.divide %5027, %5204 : tensor<128xf32>
      %cst_1315 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5206 = stablehlo.broadcast_in_dim %cst_1315, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5207 = stablehlo.divide %5028, %5206 : tensor<128xf32>
      %cst_1316 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5208 = stablehlo.broadcast_in_dim %cst_1316, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5209 = stablehlo.divide %5029, %5208 : tensor<128xf32>
      %cst_1317 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5210 = stablehlo.broadcast_in_dim %cst_1317, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5211 = stablehlo.divide %5030, %5210 : tensor<128xf32>
      %cst_1318 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5212 = stablehlo.broadcast_in_dim %cst_1318, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5213 = stablehlo.divide %5031, %5212 : tensor<128xf32>
      %cst_1319 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5214 = stablehlo.broadcast_in_dim %cst_1319, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %5215 = stablehlo.divide %5032, %5214 : tensor<3x3x128x128xf32>
      %cst_1320 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5216 = stablehlo.broadcast_in_dim %cst_1320, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %5217 = stablehlo.divide %5033, %5216 : tensor<3x3x128x128xf32>
      %cst_1321 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5218 = stablehlo.broadcast_in_dim %cst_1321, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5219 = stablehlo.divide %5034, %5218 : tensor<128xf32>
      %cst_1322 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5220 = stablehlo.broadcast_in_dim %cst_1322, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5221 = stablehlo.divide %5035, %5220 : tensor<128xf32>
      %cst_1323 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5222 = stablehlo.broadcast_in_dim %cst_1323, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5223 = stablehlo.divide %5036, %5222 : tensor<128xf32>
      %cst_1324 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5224 = stablehlo.broadcast_in_dim %cst_1324, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5225 = stablehlo.divide %5037, %5224 : tensor<128xf32>
      %cst_1325 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5226 = stablehlo.broadcast_in_dim %cst_1325, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %5227 = stablehlo.divide %5038, %5226 : tensor<3x3x128x128xf32>
      %cst_1326 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5228 = stablehlo.broadcast_in_dim %cst_1326, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %5229 = stablehlo.divide %5039, %5228 : tensor<3x3x128x128xf32>
      %cst_1327 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5230 = stablehlo.broadcast_in_dim %cst_1327, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5231 = stablehlo.divide %5040, %5230 : tensor<128xf32>
      %cst_1328 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5232 = stablehlo.broadcast_in_dim %cst_1328, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5233 = stablehlo.divide %5041, %5232 : tensor<128xf32>
      %cst_1329 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5234 = stablehlo.broadcast_in_dim %cst_1329, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5235 = stablehlo.divide %5042, %5234 : tensor<128xf32>
      %cst_1330 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5236 = stablehlo.broadcast_in_dim %cst_1330, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5237 = stablehlo.divide %5043, %5236 : tensor<128xf32>
      %cst_1331 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5238 = stablehlo.broadcast_in_dim %cst_1331, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %5239 = stablehlo.divide %5044, %5238 : tensor<3x3x128x128xf32>
      %cst_1332 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5240 = stablehlo.broadcast_in_dim %cst_1332, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %5241 = stablehlo.divide %5045, %5240 : tensor<3x3x128x128xf32>
      %cst_1333 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5242 = stablehlo.broadcast_in_dim %cst_1333, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5243 = stablehlo.divide %5046, %5242 : tensor<256xf32>
      %cst_1334 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5244 = stablehlo.broadcast_in_dim %cst_1334, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5245 = stablehlo.divide %5047, %5244 : tensor<256xf32>
      %cst_1335 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5246 = stablehlo.broadcast_in_dim %cst_1335, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5247 = stablehlo.divide %5048, %5246 : tensor<256xf32>
      %cst_1336 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5248 = stablehlo.broadcast_in_dim %cst_1336, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5249 = stablehlo.divide %5049, %5248 : tensor<256xf32>
      %cst_1337 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5250 = stablehlo.broadcast_in_dim %cst_1337, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %5251 = stablehlo.divide %5050, %5250 : tensor<3x3x128x256xf32>
      %cst_1338 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5252 = stablehlo.broadcast_in_dim %cst_1338, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5253 = stablehlo.divide %5051, %5252 : tensor<3x3x256x256xf32>
      %cst_1339 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5254 = stablehlo.broadcast_in_dim %cst_1339, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %5255 = stablehlo.divide %5052, %5254 : tensor<1x1x128x256xf32>
      %cst_1340 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5256 = stablehlo.broadcast_in_dim %cst_1340, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5257 = stablehlo.divide %5053, %5256 : tensor<256xf32>
      %cst_1341 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5258 = stablehlo.broadcast_in_dim %cst_1341, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5259 = stablehlo.divide %5054, %5258 : tensor<256xf32>
      %cst_1342 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5260 = stablehlo.broadcast_in_dim %cst_1342, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5261 = stablehlo.divide %5055, %5260 : tensor<256xf32>
      %cst_1343 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5262 = stablehlo.broadcast_in_dim %cst_1343, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5263 = stablehlo.divide %5056, %5262 : tensor<256xf32>
      %cst_1344 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5264 = stablehlo.broadcast_in_dim %cst_1344, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5265 = stablehlo.divide %5057, %5264 : tensor<256xf32>
      %cst_1345 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5266 = stablehlo.broadcast_in_dim %cst_1345, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5267 = stablehlo.divide %5058, %5266 : tensor<256xf32>
      %cst_1346 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5268 = stablehlo.broadcast_in_dim %cst_1346, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5269 = stablehlo.divide %5059, %5268 : tensor<3x3x256x256xf32>
      %cst_1347 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5270 = stablehlo.broadcast_in_dim %cst_1347, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5271 = stablehlo.divide %5060, %5270 : tensor<3x3x256x256xf32>
      %cst_1348 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5272 = stablehlo.broadcast_in_dim %cst_1348, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5273 = stablehlo.divide %5061, %5272 : tensor<256xf32>
      %cst_1349 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5274 = stablehlo.broadcast_in_dim %cst_1349, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5275 = stablehlo.divide %5062, %5274 : tensor<256xf32>
      %cst_1350 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5276 = stablehlo.broadcast_in_dim %cst_1350, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5277 = stablehlo.divide %5063, %5276 : tensor<256xf32>
      %cst_1351 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5278 = stablehlo.broadcast_in_dim %cst_1351, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5279 = stablehlo.divide %5064, %5278 : tensor<256xf32>
      %cst_1352 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5280 = stablehlo.broadcast_in_dim %cst_1352, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5281 = stablehlo.divide %5065, %5280 : tensor<3x3x256x256xf32>
      %cst_1353 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5282 = stablehlo.broadcast_in_dim %cst_1353, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5283 = stablehlo.divide %5066, %5282 : tensor<3x3x256x256xf32>
      %cst_1354 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5284 = stablehlo.broadcast_in_dim %cst_1354, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5285 = stablehlo.divide %5067, %5284 : tensor<64xf32>
      %cst_1355 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5286 = stablehlo.broadcast_in_dim %cst_1355, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5287 = stablehlo.divide %5068, %5286 : tensor<64xf32>
      %cst_1356 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5288 = stablehlo.broadcast_in_dim %cst_1356, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %5289 = stablehlo.divide %5069, %5288 : tensor<7x7x3x64xf32>
      %5290 = stablehlo.is_finite %5071 : (tensor<1000xf32>) -> tensor<1000xi1>
      %c_1357 = stablehlo.constant dense<true> : tensor<i1>
      %5291 = stablehlo.reduce(%5290 init: %c_1357) applies stablehlo.and across dimensions = [0] : (tensor<1000xi1>, tensor<i1>) -> tensor<i1>
      %c_1358 = stablehlo.constant dense<true> : tensor<i1>
      %5292 = stablehlo.and %c_1358, %5291 : tensor<i1>
      %5293 = stablehlo.is_finite %5073 : (tensor<512x1000xf32>) -> tensor<512x1000xi1>
      %c_1359 = stablehlo.constant dense<true> : tensor<i1>
      %5294 = stablehlo.reduce(%5293 init: %c_1359) applies stablehlo.and across dimensions = [0, 1] : (tensor<512x1000xi1>, tensor<i1>) -> tensor<i1>
      %5295 = stablehlo.and %5292, %5294 : tensor<i1>
      %5296 = stablehlo.is_finite %5075 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1360 = stablehlo.constant dense<true> : tensor<i1>
      %5297 = stablehlo.reduce(%5296 init: %c_1360) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5298 = stablehlo.and %5295, %5297 : tensor<i1>
      %5299 = stablehlo.is_finite %5077 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1361 = stablehlo.constant dense<true> : tensor<i1>
      %5300 = stablehlo.reduce(%5299 init: %c_1361) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5301 = stablehlo.and %5298, %5300 : tensor<i1>
      %5302 = stablehlo.is_finite %5079 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1362 = stablehlo.constant dense<true> : tensor<i1>
      %5303 = stablehlo.reduce(%5302 init: %c_1362) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5304 = stablehlo.and %5301, %5303 : tensor<i1>
      %5305 = stablehlo.is_finite %5081 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1363 = stablehlo.constant dense<true> : tensor<i1>
      %5306 = stablehlo.reduce(%5305 init: %c_1363) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5307 = stablehlo.and %5304, %5306 : tensor<i1>
      %5308 = stablehlo.is_finite %5083 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_1364 = stablehlo.constant dense<true> : tensor<i1>
      %5309 = stablehlo.reduce(%5308 init: %c_1364) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %5310 = stablehlo.and %5307, %5309 : tensor<i1>
      %5311 = stablehlo.is_finite %5085 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_1365 = stablehlo.constant dense<true> : tensor<i1>
      %5312 = stablehlo.reduce(%5311 init: %c_1365) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %5313 = stablehlo.and %5310, %5312 : tensor<i1>
      %5314 = stablehlo.is_finite %5087 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1366 = stablehlo.constant dense<true> : tensor<i1>
      %5315 = stablehlo.reduce(%5314 init: %c_1366) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5316 = stablehlo.and %5313, %5315 : tensor<i1>
      %5317 = stablehlo.is_finite %5089 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1367 = stablehlo.constant dense<true> : tensor<i1>
      %5318 = stablehlo.reduce(%5317 init: %c_1367) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5319 = stablehlo.and %5316, %5318 : tensor<i1>
      %5320 = stablehlo.is_finite %5091 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1368 = stablehlo.constant dense<true> : tensor<i1>
      %5321 = stablehlo.reduce(%5320 init: %c_1368) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5322 = stablehlo.and %5319, %5321 : tensor<i1>
      %5323 = stablehlo.is_finite %5093 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1369 = stablehlo.constant dense<true> : tensor<i1>
      %5324 = stablehlo.reduce(%5323 init: %c_1369) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5325 = stablehlo.and %5322, %5324 : tensor<i1>
      %5326 = stablehlo.is_finite %5095 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_1370 = stablehlo.constant dense<true> : tensor<i1>
      %5327 = stablehlo.reduce(%5326 init: %c_1370) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %5328 = stablehlo.and %5325, %5327 : tensor<i1>
      %5329 = stablehlo.is_finite %5097 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_1371 = stablehlo.constant dense<true> : tensor<i1>
      %5330 = stablehlo.reduce(%5329 init: %c_1371) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %5331 = stablehlo.and %5328, %5330 : tensor<i1>
      %5332 = stablehlo.is_finite %5099 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1372 = stablehlo.constant dense<true> : tensor<i1>
      %5333 = stablehlo.reduce(%5332 init: %c_1372) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5334 = stablehlo.and %5331, %5333 : tensor<i1>
      %5335 = stablehlo.is_finite %5101 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1373 = stablehlo.constant dense<true> : tensor<i1>
      %5336 = stablehlo.reduce(%5335 init: %c_1373) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5337 = stablehlo.and %5334, %5336 : tensor<i1>
      %5338 = stablehlo.is_finite %5103 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1374 = stablehlo.constant dense<true> : tensor<i1>
      %5339 = stablehlo.reduce(%5338 init: %c_1374) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5340 = stablehlo.and %5337, %5339 : tensor<i1>
      %5341 = stablehlo.is_finite %5105 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1375 = stablehlo.constant dense<true> : tensor<i1>
      %5342 = stablehlo.reduce(%5341 init: %c_1375) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5343 = stablehlo.and %5340, %5342 : tensor<i1>
      %5344 = stablehlo.is_finite %5107 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1376 = stablehlo.constant dense<true> : tensor<i1>
      %5345 = stablehlo.reduce(%5344 init: %c_1376) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5346 = stablehlo.and %5343, %5345 : tensor<i1>
      %5347 = stablehlo.is_finite %5109 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1377 = stablehlo.constant dense<true> : tensor<i1>
      %5348 = stablehlo.reduce(%5347 init: %c_1377) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5349 = stablehlo.and %5346, %5348 : tensor<i1>
      %5350 = stablehlo.is_finite %5111 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1378 = stablehlo.constant dense<true> : tensor<i1>
      %5351 = stablehlo.reduce(%5350 init: %c_1378) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5352 = stablehlo.and %5349, %5351 : tensor<i1>
      %5353 = stablehlo.is_finite %5113 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1379 = stablehlo.constant dense<true> : tensor<i1>
      %5354 = stablehlo.reduce(%5353 init: %c_1379) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5355 = stablehlo.and %5352, %5354 : tensor<i1>
      %5356 = stablehlo.is_finite %5115 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1380 = stablehlo.constant dense<true> : tensor<i1>
      %5357 = stablehlo.reduce(%5356 init: %c_1380) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5358 = stablehlo.and %5355, %5357 : tensor<i1>
      %5359 = stablehlo.is_finite %5117 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1381 = stablehlo.constant dense<true> : tensor<i1>
      %5360 = stablehlo.reduce(%5359 init: %c_1381) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5361 = stablehlo.and %5358, %5360 : tensor<i1>
      %5362 = stablehlo.is_finite %5119 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1382 = stablehlo.constant dense<true> : tensor<i1>
      %5363 = stablehlo.reduce(%5362 init: %c_1382) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5364 = stablehlo.and %5361, %5363 : tensor<i1>
      %5365 = stablehlo.is_finite %5121 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1383 = stablehlo.constant dense<true> : tensor<i1>
      %5366 = stablehlo.reduce(%5365 init: %c_1383) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5367 = stablehlo.and %5364, %5366 : tensor<i1>
      %5368 = stablehlo.is_finite %5123 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1384 = stablehlo.constant dense<true> : tensor<i1>
      %5369 = stablehlo.reduce(%5368 init: %c_1384) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5370 = stablehlo.and %5367, %5369 : tensor<i1>
      %5371 = stablehlo.is_finite %5125 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1385 = stablehlo.constant dense<true> : tensor<i1>
      %5372 = stablehlo.reduce(%5371 init: %c_1385) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5373 = stablehlo.and %5370, %5372 : tensor<i1>
      %5374 = stablehlo.is_finite %5127 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1386 = stablehlo.constant dense<true> : tensor<i1>
      %5375 = stablehlo.reduce(%5374 init: %c_1386) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5376 = stablehlo.and %5373, %5375 : tensor<i1>
      %5377 = stablehlo.is_finite %5129 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1387 = stablehlo.constant dense<true> : tensor<i1>
      %5378 = stablehlo.reduce(%5377 init: %c_1387) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5379 = stablehlo.and %5376, %5378 : tensor<i1>
      %5380 = stablehlo.is_finite %5131 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1388 = stablehlo.constant dense<true> : tensor<i1>
      %5381 = stablehlo.reduce(%5380 init: %c_1388) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5382 = stablehlo.and %5379, %5381 : tensor<i1>
      %5383 = stablehlo.is_finite %5133 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1389 = stablehlo.constant dense<true> : tensor<i1>
      %5384 = stablehlo.reduce(%5383 init: %c_1389) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5385 = stablehlo.and %5382, %5384 : tensor<i1>
      %5386 = stablehlo.is_finite %5135 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1390 = stablehlo.constant dense<true> : tensor<i1>
      %5387 = stablehlo.reduce(%5386 init: %c_1390) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5388 = stablehlo.and %5385, %5387 : tensor<i1>
      %5389 = stablehlo.is_finite %5137 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1391 = stablehlo.constant dense<true> : tensor<i1>
      %5390 = stablehlo.reduce(%5389 init: %c_1391) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5391 = stablehlo.and %5388, %5390 : tensor<i1>
      %5392 = stablehlo.is_finite %5139 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1392 = stablehlo.constant dense<true> : tensor<i1>
      %5393 = stablehlo.reduce(%5392 init: %c_1392) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5394 = stablehlo.and %5391, %5393 : tensor<i1>
      %5395 = stablehlo.is_finite %5141 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1393 = stablehlo.constant dense<true> : tensor<i1>
      %5396 = stablehlo.reduce(%5395 init: %c_1393) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5397 = stablehlo.and %5394, %5396 : tensor<i1>
      %5398 = stablehlo.is_finite %5143 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xi1>
      %c_1394 = stablehlo.constant dense<true> : tensor<i1>
      %5399 = stablehlo.reduce(%5398 init: %c_1394) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x512xi1>, tensor<i1>) -> tensor<i1>
      %5400 = stablehlo.and %5397, %5399 : tensor<i1>
      %5401 = stablehlo.is_finite %5145 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
      %c_1395 = stablehlo.constant dense<true> : tensor<i1>
      %5402 = stablehlo.reduce(%5401 init: %c_1395) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
      %5403 = stablehlo.and %5400, %5402 : tensor<i1>
      %5404 = stablehlo.is_finite %5147 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xi1>
      %c_1396 = stablehlo.constant dense<true> : tensor<i1>
      %5405 = stablehlo.reduce(%5404 init: %c_1396) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x256x512xi1>, tensor<i1>) -> tensor<i1>
      %5406 = stablehlo.and %5403, %5405 : tensor<i1>
      %5407 = stablehlo.is_finite %5149 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1397 = stablehlo.constant dense<true> : tensor<i1>
      %5408 = stablehlo.reduce(%5407 init: %c_1397) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5409 = stablehlo.and %5406, %5408 : tensor<i1>
      %5410 = stablehlo.is_finite %5151 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1398 = stablehlo.constant dense<true> : tensor<i1>
      %5411 = stablehlo.reduce(%5410 init: %c_1398) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5412 = stablehlo.and %5409, %5411 : tensor<i1>
      %5413 = stablehlo.is_finite %5153 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1399 = stablehlo.constant dense<true> : tensor<i1>
      %5414 = stablehlo.reduce(%5413 init: %c_1399) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5415 = stablehlo.and %5412, %5414 : tensor<i1>
      %5416 = stablehlo.is_finite %5155 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1400 = stablehlo.constant dense<true> : tensor<i1>
      %5417 = stablehlo.reduce(%5416 init: %c_1400) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5418 = stablehlo.and %5415, %5417 : tensor<i1>
      %5419 = stablehlo.is_finite %5157 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1401 = stablehlo.constant dense<true> : tensor<i1>
      %5420 = stablehlo.reduce(%5419 init: %c_1401) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5421 = stablehlo.and %5418, %5420 : tensor<i1>
      %5422 = stablehlo.is_finite %5159 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1402 = stablehlo.constant dense<true> : tensor<i1>
      %5423 = stablehlo.reduce(%5422 init: %c_1402) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5424 = stablehlo.and %5421, %5423 : tensor<i1>
      %5425 = stablehlo.is_finite %5161 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
      %c_1403 = stablehlo.constant dense<true> : tensor<i1>
      %5426 = stablehlo.reduce(%5425 init: %c_1403) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
      %5427 = stablehlo.and %5424, %5426 : tensor<i1>
      %5428 = stablehlo.is_finite %5163 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
      %c_1404 = stablehlo.constant dense<true> : tensor<i1>
      %5429 = stablehlo.reduce(%5428 init: %c_1404) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
      %5430 = stablehlo.and %5427, %5429 : tensor<i1>
      %5431 = stablehlo.is_finite %5165 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1405 = stablehlo.constant dense<true> : tensor<i1>
      %5432 = stablehlo.reduce(%5431 init: %c_1405) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5433 = stablehlo.and %5430, %5432 : tensor<i1>
      %5434 = stablehlo.is_finite %5167 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1406 = stablehlo.constant dense<true> : tensor<i1>
      %5435 = stablehlo.reduce(%5434 init: %c_1406) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5436 = stablehlo.and %5433, %5435 : tensor<i1>
      %5437 = stablehlo.is_finite %5169 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1407 = stablehlo.constant dense<true> : tensor<i1>
      %5438 = stablehlo.reduce(%5437 init: %c_1407) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5439 = stablehlo.and %5436, %5438 : tensor<i1>
      %5440 = stablehlo.is_finite %5171 : (tensor<512xf32>) -> tensor<512xi1>
      %c_1408 = stablehlo.constant dense<true> : tensor<i1>
      %5441 = stablehlo.reduce(%5440 init: %c_1408) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %5442 = stablehlo.and %5439, %5441 : tensor<i1>
      %5443 = stablehlo.is_finite %5173 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
      %c_1409 = stablehlo.constant dense<true> : tensor<i1>
      %5444 = stablehlo.reduce(%5443 init: %c_1409) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
      %5445 = stablehlo.and %5442, %5444 : tensor<i1>
      %5446 = stablehlo.is_finite %5175 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
      %c_1410 = stablehlo.constant dense<true> : tensor<i1>
      %5447 = stablehlo.reduce(%5446 init: %c_1410) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
      %5448 = stablehlo.and %5445, %5447 : tensor<i1>
      %5449 = stablehlo.is_finite %5177 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1411 = stablehlo.constant dense<true> : tensor<i1>
      %5450 = stablehlo.reduce(%5449 init: %c_1411) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5451 = stablehlo.and %5448, %5450 : tensor<i1>
      %5452 = stablehlo.is_finite %5179 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1412 = stablehlo.constant dense<true> : tensor<i1>
      %5453 = stablehlo.reduce(%5452 init: %c_1412) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5454 = stablehlo.and %5451, %5453 : tensor<i1>
      %5455 = stablehlo.is_finite %5181 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1413 = stablehlo.constant dense<true> : tensor<i1>
      %5456 = stablehlo.reduce(%5455 init: %c_1413) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5457 = stablehlo.and %5454, %5456 : tensor<i1>
      %5458 = stablehlo.is_finite %5183 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1414 = stablehlo.constant dense<true> : tensor<i1>
      %5459 = stablehlo.reduce(%5458 init: %c_1414) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5460 = stablehlo.and %5457, %5459 : tensor<i1>
      %5461 = stablehlo.is_finite %5185 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_1415 = stablehlo.constant dense<true> : tensor<i1>
      %5462 = stablehlo.reduce(%5461 init: %c_1415) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %5463 = stablehlo.and %5460, %5462 : tensor<i1>
      %5464 = stablehlo.is_finite %5187 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_1416 = stablehlo.constant dense<true> : tensor<i1>
      %5465 = stablehlo.reduce(%5464 init: %c_1416) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %5466 = stablehlo.and %5463, %5465 : tensor<i1>
      %5467 = stablehlo.is_finite %5189 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1417 = stablehlo.constant dense<true> : tensor<i1>
      %5468 = stablehlo.reduce(%5467 init: %c_1417) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5469 = stablehlo.and %5466, %5468 : tensor<i1>
      %5470 = stablehlo.is_finite %5191 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1418 = stablehlo.constant dense<true> : tensor<i1>
      %5471 = stablehlo.reduce(%5470 init: %c_1418) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5472 = stablehlo.and %5469, %5471 : tensor<i1>
      %5473 = stablehlo.is_finite %5193 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1419 = stablehlo.constant dense<true> : tensor<i1>
      %5474 = stablehlo.reduce(%5473 init: %c_1419) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5475 = stablehlo.and %5472, %5474 : tensor<i1>
      %5476 = stablehlo.is_finite %5195 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1420 = stablehlo.constant dense<true> : tensor<i1>
      %5477 = stablehlo.reduce(%5476 init: %c_1420) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5478 = stablehlo.and %5475, %5477 : tensor<i1>
      %5479 = stablehlo.is_finite %5197 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xi1>
      %c_1421 = stablehlo.constant dense<true> : tensor<i1>
      %5480 = stablehlo.reduce(%5479 init: %c_1421) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x128xi1>, tensor<i1>) -> tensor<i1>
      %5481 = stablehlo.and %5478, %5480 : tensor<i1>
      %5482 = stablehlo.is_finite %5199 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_1422 = stablehlo.constant dense<true> : tensor<i1>
      %5483 = stablehlo.reduce(%5482 init: %c_1422) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %5484 = stablehlo.and %5481, %5483 : tensor<i1>
      %5485 = stablehlo.is_finite %5201 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xi1>
      %c_1423 = stablehlo.constant dense<true> : tensor<i1>
      %5486 = stablehlo.reduce(%5485 init: %c_1423) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x64x128xi1>, tensor<i1>) -> tensor<i1>
      %5487 = stablehlo.and %5484, %5486 : tensor<i1>
      %5488 = stablehlo.is_finite %5203 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1424 = stablehlo.constant dense<true> : tensor<i1>
      %5489 = stablehlo.reduce(%5488 init: %c_1424) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5490 = stablehlo.and %5487, %5489 : tensor<i1>
      %5491 = stablehlo.is_finite %5205 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1425 = stablehlo.constant dense<true> : tensor<i1>
      %5492 = stablehlo.reduce(%5491 init: %c_1425) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5493 = stablehlo.and %5490, %5492 : tensor<i1>
      %5494 = stablehlo.is_finite %5207 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1426 = stablehlo.constant dense<true> : tensor<i1>
      %5495 = stablehlo.reduce(%5494 init: %c_1426) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5496 = stablehlo.and %5493, %5495 : tensor<i1>
      %5497 = stablehlo.is_finite %5209 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1427 = stablehlo.constant dense<true> : tensor<i1>
      %5498 = stablehlo.reduce(%5497 init: %c_1427) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5499 = stablehlo.and %5496, %5498 : tensor<i1>
      %5500 = stablehlo.is_finite %5211 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1428 = stablehlo.constant dense<true> : tensor<i1>
      %5501 = stablehlo.reduce(%5500 init: %c_1428) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5502 = stablehlo.and %5499, %5501 : tensor<i1>
      %5503 = stablehlo.is_finite %5213 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1429 = stablehlo.constant dense<true> : tensor<i1>
      %5504 = stablehlo.reduce(%5503 init: %c_1429) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5505 = stablehlo.and %5502, %5504 : tensor<i1>
      %5506 = stablehlo.is_finite %5215 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_1430 = stablehlo.constant dense<true> : tensor<i1>
      %5507 = stablehlo.reduce(%5506 init: %c_1430) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %5508 = stablehlo.and %5505, %5507 : tensor<i1>
      %5509 = stablehlo.is_finite %5217 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_1431 = stablehlo.constant dense<true> : tensor<i1>
      %5510 = stablehlo.reduce(%5509 init: %c_1431) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %5511 = stablehlo.and %5508, %5510 : tensor<i1>
      %5512 = stablehlo.is_finite %5219 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1432 = stablehlo.constant dense<true> : tensor<i1>
      %5513 = stablehlo.reduce(%5512 init: %c_1432) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5514 = stablehlo.and %5511, %5513 : tensor<i1>
      %5515 = stablehlo.is_finite %5221 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1433 = stablehlo.constant dense<true> : tensor<i1>
      %5516 = stablehlo.reduce(%5515 init: %c_1433) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5517 = stablehlo.and %5514, %5516 : tensor<i1>
      %5518 = stablehlo.is_finite %5223 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1434 = stablehlo.constant dense<true> : tensor<i1>
      %5519 = stablehlo.reduce(%5518 init: %c_1434) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5520 = stablehlo.and %5517, %5519 : tensor<i1>
      %5521 = stablehlo.is_finite %5225 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1435 = stablehlo.constant dense<true> : tensor<i1>
      %5522 = stablehlo.reduce(%5521 init: %c_1435) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5523 = stablehlo.and %5520, %5522 : tensor<i1>
      %5524 = stablehlo.is_finite %5227 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_1436 = stablehlo.constant dense<true> : tensor<i1>
      %5525 = stablehlo.reduce(%5524 init: %c_1436) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %5526 = stablehlo.and %5523, %5525 : tensor<i1>
      %5527 = stablehlo.is_finite %5229 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_1437 = stablehlo.constant dense<true> : tensor<i1>
      %5528 = stablehlo.reduce(%5527 init: %c_1437) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %5529 = stablehlo.and %5526, %5528 : tensor<i1>
      %5530 = stablehlo.is_finite %5231 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1438 = stablehlo.constant dense<true> : tensor<i1>
      %5531 = stablehlo.reduce(%5530 init: %c_1438) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5532 = stablehlo.and %5529, %5531 : tensor<i1>
      %5533 = stablehlo.is_finite %5233 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1439 = stablehlo.constant dense<true> : tensor<i1>
      %5534 = stablehlo.reduce(%5533 init: %c_1439) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5535 = stablehlo.and %5532, %5534 : tensor<i1>
      %5536 = stablehlo.is_finite %5235 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1440 = stablehlo.constant dense<true> : tensor<i1>
      %5537 = stablehlo.reduce(%5536 init: %c_1440) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5538 = stablehlo.and %5535, %5537 : tensor<i1>
      %5539 = stablehlo.is_finite %5237 : (tensor<128xf32>) -> tensor<128xi1>
      %c_1441 = stablehlo.constant dense<true> : tensor<i1>
      %5540 = stablehlo.reduce(%5539 init: %c_1441) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %5541 = stablehlo.and %5538, %5540 : tensor<i1>
      %5542 = stablehlo.is_finite %5239 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_1442 = stablehlo.constant dense<true> : tensor<i1>
      %5543 = stablehlo.reduce(%5542 init: %c_1442) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %5544 = stablehlo.and %5541, %5543 : tensor<i1>
      %5545 = stablehlo.is_finite %5241 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_1443 = stablehlo.constant dense<true> : tensor<i1>
      %5546 = stablehlo.reduce(%5545 init: %c_1443) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %5547 = stablehlo.and %5544, %5546 : tensor<i1>
      %5548 = stablehlo.is_finite %5243 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1444 = stablehlo.constant dense<true> : tensor<i1>
      %5549 = stablehlo.reduce(%5548 init: %c_1444) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5550 = stablehlo.and %5547, %5549 : tensor<i1>
      %5551 = stablehlo.is_finite %5245 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1445 = stablehlo.constant dense<true> : tensor<i1>
      %5552 = stablehlo.reduce(%5551 init: %c_1445) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5553 = stablehlo.and %5550, %5552 : tensor<i1>
      %5554 = stablehlo.is_finite %5247 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1446 = stablehlo.constant dense<true> : tensor<i1>
      %5555 = stablehlo.reduce(%5554 init: %c_1446) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5556 = stablehlo.and %5553, %5555 : tensor<i1>
      %5557 = stablehlo.is_finite %5249 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1447 = stablehlo.constant dense<true> : tensor<i1>
      %5558 = stablehlo.reduce(%5557 init: %c_1447) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5559 = stablehlo.and %5556, %5558 : tensor<i1>
      %5560 = stablehlo.is_finite %5251 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xi1>
      %c_1448 = stablehlo.constant dense<true> : tensor<i1>
      %5561 = stablehlo.reduce(%5560 init: %c_1448) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x256xi1>, tensor<i1>) -> tensor<i1>
      %5562 = stablehlo.and %5559, %5561 : tensor<i1>
      %5563 = stablehlo.is_finite %5253 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1449 = stablehlo.constant dense<true> : tensor<i1>
      %5564 = stablehlo.reduce(%5563 init: %c_1449) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5565 = stablehlo.and %5562, %5564 : tensor<i1>
      %5566 = stablehlo.is_finite %5255 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xi1>
      %c_1450 = stablehlo.constant dense<true> : tensor<i1>
      %5567 = stablehlo.reduce(%5566 init: %c_1450) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x128x256xi1>, tensor<i1>) -> tensor<i1>
      %5568 = stablehlo.and %5565, %5567 : tensor<i1>
      %5569 = stablehlo.is_finite %5257 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1451 = stablehlo.constant dense<true> : tensor<i1>
      %5570 = stablehlo.reduce(%5569 init: %c_1451) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5571 = stablehlo.and %5568, %5570 : tensor<i1>
      %5572 = stablehlo.is_finite %5259 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1452 = stablehlo.constant dense<true> : tensor<i1>
      %5573 = stablehlo.reduce(%5572 init: %c_1452) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5574 = stablehlo.and %5571, %5573 : tensor<i1>
      %5575 = stablehlo.is_finite %5261 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1453 = stablehlo.constant dense<true> : tensor<i1>
      %5576 = stablehlo.reduce(%5575 init: %c_1453) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5577 = stablehlo.and %5574, %5576 : tensor<i1>
      %5578 = stablehlo.is_finite %5263 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1454 = stablehlo.constant dense<true> : tensor<i1>
      %5579 = stablehlo.reduce(%5578 init: %c_1454) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5580 = stablehlo.and %5577, %5579 : tensor<i1>
      %5581 = stablehlo.is_finite %5265 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1455 = stablehlo.constant dense<true> : tensor<i1>
      %5582 = stablehlo.reduce(%5581 init: %c_1455) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5583 = stablehlo.and %5580, %5582 : tensor<i1>
      %5584 = stablehlo.is_finite %5267 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1456 = stablehlo.constant dense<true> : tensor<i1>
      %5585 = stablehlo.reduce(%5584 init: %c_1456) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5586 = stablehlo.and %5583, %5585 : tensor<i1>
      %5587 = stablehlo.is_finite %5269 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1457 = stablehlo.constant dense<true> : tensor<i1>
      %5588 = stablehlo.reduce(%5587 init: %c_1457) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5589 = stablehlo.and %5586, %5588 : tensor<i1>
      %5590 = stablehlo.is_finite %5271 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1458 = stablehlo.constant dense<true> : tensor<i1>
      %5591 = stablehlo.reduce(%5590 init: %c_1458) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5592 = stablehlo.and %5589, %5591 : tensor<i1>
      %5593 = stablehlo.is_finite %5273 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1459 = stablehlo.constant dense<true> : tensor<i1>
      %5594 = stablehlo.reduce(%5593 init: %c_1459) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5595 = stablehlo.and %5592, %5594 : tensor<i1>
      %5596 = stablehlo.is_finite %5275 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1460 = stablehlo.constant dense<true> : tensor<i1>
      %5597 = stablehlo.reduce(%5596 init: %c_1460) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5598 = stablehlo.and %5595, %5597 : tensor<i1>
      %5599 = stablehlo.is_finite %5277 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1461 = stablehlo.constant dense<true> : tensor<i1>
      %5600 = stablehlo.reduce(%5599 init: %c_1461) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5601 = stablehlo.and %5598, %5600 : tensor<i1>
      %5602 = stablehlo.is_finite %5279 : (tensor<256xf32>) -> tensor<256xi1>
      %c_1462 = stablehlo.constant dense<true> : tensor<i1>
      %5603 = stablehlo.reduce(%5602 init: %c_1462) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %5604 = stablehlo.and %5601, %5603 : tensor<i1>
      %5605 = stablehlo.is_finite %5281 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1463 = stablehlo.constant dense<true> : tensor<i1>
      %5606 = stablehlo.reduce(%5605 init: %c_1463) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5607 = stablehlo.and %5604, %5606 : tensor<i1>
      %5608 = stablehlo.is_finite %5283 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_1464 = stablehlo.constant dense<true> : tensor<i1>
      %5609 = stablehlo.reduce(%5608 init: %c_1464) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %5610 = stablehlo.and %5607, %5609 : tensor<i1>
      %5611 = stablehlo.is_finite %5285 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1465 = stablehlo.constant dense<true> : tensor<i1>
      %5612 = stablehlo.reduce(%5611 init: %c_1465) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5613 = stablehlo.and %5610, %5612 : tensor<i1>
      %5614 = stablehlo.is_finite %5287 : (tensor<64xf32>) -> tensor<64xi1>
      %c_1466 = stablehlo.constant dense<true> : tensor<i1>
      %5615 = stablehlo.reduce(%5614 init: %c_1466) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %5616 = stablehlo.and %5613, %5615 : tensor<i1>
      %5617 = stablehlo.is_finite %5289 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xi1>
      %c_1467 = stablehlo.constant dense<true> : tensor<i1>
      %5618 = stablehlo.reduce(%5617 init: %c_1467) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<7x7x3x64xi1>, tensor<i1>) -> tensor<i1>
      %5619 = stablehlo.and %5616, %5618 : tensor<i1>
      %c_1468 = stablehlo.constant dense<2000> : tensor<i32>
      %5620 = stablehlo.compare  EQ, %arg592, %c_1468,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_1469 = stablehlo.constant dense<false> : tensor<i1>
      %5621 = stablehlo.broadcast_in_dim %c_1469, dims = [] : (tensor<i1>) -> tensor<i1>
      %5622 = stablehlo.compare  NE, %5620, %5621,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %5623 = stablehlo.convert %5622 : tensor<i1>
      %5624 = stablehlo.and %5623, %5619 : tensor<i1>
      %cst_1470 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %5625 = stablehlo.multiply %arg593, %cst_1470 : tensor<f32>
      %5626 = stablehlo.convert %5625 : tensor<f32>
      %cst_1471 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
      %5627 = stablehlo.minimum %5626, %cst_1471 : tensor<f32>
      %5628 = func.call @_where_493(%5624, %5627, %arg593) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_1472 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %5629 = stablehlo.multiply %arg593, %cst_1472 : tensor<f32>
      %5630 = stablehlo.convert %5629 : tensor<f32>
      %cst_1473 = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
      %5631 = stablehlo.maximum %5630, %cst_1473 : tensor<f32>
      %5632 = func.call @_where_496(%5619, %5628, %5631) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %5633 = stablehlo.not %5619 : tensor<i1>
      %c_1474 = stablehlo.constant dense<false> : tensor<i1>
      %5634 = stablehlo.broadcast_in_dim %c_1474, dims = [] : (tensor<i1>) -> tensor<i1>
      %5635 = stablehlo.compare  NE, %5620, %5634,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %5636 = stablehlo.convert %5635 : tensor<i1>
      %5637 = stablehlo.or %5636, %5633 : tensor<i1>
      %c_1475 = stablehlo.constant dense<1> : tensor<i32>
      %5638 = stablehlo.add %arg592, %c_1475 : tensor<i32>
      %c_1476 = stablehlo.constant dense<0> : tensor<i32>
      %5639 = func.call @_where_498(%5637, %c_1476, %5638) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %5640 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
      %5641 = stablehlo.iota dim = 0 : tensor<1000xi32>
      %5642 = stablehlo.reshape %5641 : (tensor<1000xi32>) -> tensor<1x1000xi32>
      %5643 = stablehlo.broadcast_in_dim %5640, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
      %5644 = stablehlo.broadcast_in_dim %5642, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
      %5645 = stablehlo.compare  EQ, %5643, %5644,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
      %cst_1477 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %5646 = stablehlo.broadcast_in_dim %cst_1477, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %cst_1478 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5647 = stablehlo.broadcast_in_dim %cst_1478, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %5648 = stablehlo.select %5645, %5646, %5647 : tensor<256x1000xi1>, tensor<256x1000xf32>
      %5649 = stablehlo.convert %5648 : tensor<256x1000xf32>
      %5650 = func.call @log_softmax_500(%2563) : (tensor<256x1000xf16>) -> tensor<256x1000xf16>
      %5651 = stablehlo.convert %5650 : (tensor<256x1000xf16>) -> tensor<256x1000xf32>
      %5652 = stablehlo.multiply %5649, %5651 : tensor<256x1000xf32>
      %cst_1479 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5653 = stablehlo.reduce(%5652 init: %cst_1479) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
      %5654 = stablehlo.negate %5653 : tensor<256xf32>
      %cst_1480 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5655 = stablehlo.reduce(%5654 init: %cst_1480) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_1481 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %5656 = stablehlo.divide %5655, %cst_1481 : tensor<f32>
      %5657 = func.call @argmax(%2563) : (tensor<256x1000xf16>) -> tensor<256xi32>
      %5658 = stablehlo.compare  EQ, %5657, %2,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
      %5659 = stablehlo.convert %5658 : (tensor<256xi1>) -> tensor<256xi32>
      %5660 = stablehlo.convert %5659 : (tensor<256xi32>) -> tensor<256xf32>
      %cst_1482 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %5661 = stablehlo.reduce(%5660 init: %cst_1482) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_1483 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %5662 = stablehlo.divide %5661, %cst_1483 : tensor<f32>
      %5663 = "stablehlo.all_reduce"(%5662) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %5664 = "stablehlo.all_reduce"(%5656) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_1484 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5665 = stablehlo.divide %5663, %cst_1484 : tensor<f32>
      %cst_1485 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5666 = stablehlo.divide %5664, %cst_1485 : tensor<f32>
      %5667 = "stablehlo.all_reduce"(%138) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5668 = "stablehlo.all_reduce"(%143) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5669 = "stablehlo.all_reduce"(%208) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5670 = "stablehlo.all_reduce"(%213) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5671 = "stablehlo.all_reduce"(%279) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5672 = "stablehlo.all_reduce"(%284) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5673 = "stablehlo.all_reduce"(%349) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5674 = "stablehlo.all_reduce"(%354) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5675 = "stablehlo.all_reduce"(%1682) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5676 = "stablehlo.all_reduce"(%1687) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5677 = "stablehlo.all_reduce"(%1752) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5678 = "stablehlo.all_reduce"(%1757) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5679 = "stablehlo.all_reduce"(%1823) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5680 = "stablehlo.all_reduce"(%1828) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5681 = "stablehlo.all_reduce"(%1893) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5682 = "stablehlo.all_reduce"(%1898) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5683 = "stablehlo.all_reduce"(%1964) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5684 = "stablehlo.all_reduce"(%1969) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5685 = "stablehlo.all_reduce"(%2034) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5686 = "stablehlo.all_reduce"(%2039) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5687 = "stablehlo.all_reduce"(%2105) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5688 = "stablehlo.all_reduce"(%2110) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5689 = "stablehlo.all_reduce"(%2175) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5690 = "stablehlo.all_reduce"(%2180) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5691 = "stablehlo.all_reduce"(%2242) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5692 = "stablehlo.all_reduce"(%2247) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5693 = "stablehlo.all_reduce"(%2313) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5694 = "stablehlo.all_reduce"(%2318) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5695 = "stablehlo.all_reduce"(%2383) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5696 = "stablehlo.all_reduce"(%2388) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5697 = "stablehlo.all_reduce"(%2454) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5698 = "stablehlo.all_reduce"(%2459) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5699 = "stablehlo.all_reduce"(%2524) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5700 = "stablehlo.all_reduce"(%2529) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %5701 = "stablehlo.all_reduce"(%420) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5702 = "stablehlo.all_reduce"(%425) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5703 = "stablehlo.all_reduce"(%490) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5704 = "stablehlo.all_reduce"(%495) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5705 = "stablehlo.all_reduce"(%561) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5706 = "stablehlo.all_reduce"(%566) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5707 = "stablehlo.all_reduce"(%631) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5708 = "stablehlo.all_reduce"(%636) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5709 = "stablehlo.all_reduce"(%698) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5710 = "stablehlo.all_reduce"(%703) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5711 = "stablehlo.all_reduce"(%769) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5712 = "stablehlo.all_reduce"(%774) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5713 = "stablehlo.all_reduce"(%839) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5714 = "stablehlo.all_reduce"(%844) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5715 = "stablehlo.all_reduce"(%910) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5716 = "stablehlo.all_reduce"(%915) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5717 = "stablehlo.all_reduce"(%980) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5718 = "stablehlo.all_reduce"(%985) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5719 = "stablehlo.all_reduce"(%1051) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5720 = "stablehlo.all_reduce"(%1056) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5721 = "stablehlo.all_reduce"(%1121) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5722 = "stablehlo.all_reduce"(%1126) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %5723 = "stablehlo.all_reduce"(%1192) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5724 = "stablehlo.all_reduce"(%1197) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5725 = "stablehlo.all_reduce"(%1262) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5726 = "stablehlo.all_reduce"(%1267) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5727 = "stablehlo.all_reduce"(%1329) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5728 = "stablehlo.all_reduce"(%1334) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5729 = "stablehlo.all_reduce"(%1400) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5730 = "stablehlo.all_reduce"(%1405) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5731 = "stablehlo.all_reduce"(%1470) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5732 = "stablehlo.all_reduce"(%1475) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5733 = "stablehlo.all_reduce"(%1541) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5734 = "stablehlo.all_reduce"(%1546) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5735 = "stablehlo.all_reduce"(%1611) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5736 = "stablehlo.all_reduce"(%1616) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %5737 = "stablehlo.all_reduce"(%66) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %5738 = "stablehlo.all_reduce"(%71) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg596: tensor<f32>, %arg597: tensor<f32>):
        %7234 = stablehlo.add %arg596, %arg597 : tensor<f32>
        stablehlo.return %7234 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %cst_1486 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5739 = stablehlo.broadcast_in_dim %cst_1486, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5740 = stablehlo.divide %5667, %5739 : tensor<64xf32>
      %cst_1487 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5741 = stablehlo.broadcast_in_dim %cst_1487, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5742 = stablehlo.divide %5668, %5741 : tensor<64xf32>
      %cst_1488 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5743 = stablehlo.broadcast_in_dim %cst_1488, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5744 = stablehlo.divide %5669, %5743 : tensor<64xf32>
      %cst_1489 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5745 = stablehlo.broadcast_in_dim %cst_1489, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5746 = stablehlo.divide %5670, %5745 : tensor<64xf32>
      %cst_1490 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5747 = stablehlo.broadcast_in_dim %cst_1490, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5748 = stablehlo.divide %5671, %5747 : tensor<64xf32>
      %cst_1491 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5749 = stablehlo.broadcast_in_dim %cst_1491, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5750 = stablehlo.divide %5672, %5749 : tensor<64xf32>
      %cst_1492 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5751 = stablehlo.broadcast_in_dim %cst_1492, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5752 = stablehlo.divide %5673, %5751 : tensor<64xf32>
      %cst_1493 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5753 = stablehlo.broadcast_in_dim %cst_1493, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5754 = stablehlo.divide %5674, %5753 : tensor<64xf32>
      %cst_1494 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5755 = stablehlo.broadcast_in_dim %cst_1494, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5756 = stablehlo.divide %5675, %5755 : tensor<256xf32>
      %cst_1495 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5757 = stablehlo.broadcast_in_dim %cst_1495, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5758 = stablehlo.divide %5676, %5757 : tensor<256xf32>
      %cst_1496 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5759 = stablehlo.broadcast_in_dim %cst_1496, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5760 = stablehlo.divide %5677, %5759 : tensor<256xf32>
      %cst_1497 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5761 = stablehlo.broadcast_in_dim %cst_1497, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5762 = stablehlo.divide %5678, %5761 : tensor<256xf32>
      %cst_1498 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5763 = stablehlo.broadcast_in_dim %cst_1498, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5764 = stablehlo.divide %5679, %5763 : tensor<256xf32>
      %cst_1499 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5765 = stablehlo.broadcast_in_dim %cst_1499, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5766 = stablehlo.divide %5680, %5765 : tensor<256xf32>
      %cst_1500 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5767 = stablehlo.broadcast_in_dim %cst_1500, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5768 = stablehlo.divide %5681, %5767 : tensor<256xf32>
      %cst_1501 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5769 = stablehlo.broadcast_in_dim %cst_1501, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5770 = stablehlo.divide %5682, %5769 : tensor<256xf32>
      %cst_1502 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5771 = stablehlo.broadcast_in_dim %cst_1502, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5772 = stablehlo.divide %5683, %5771 : tensor<256xf32>
      %cst_1503 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5773 = stablehlo.broadcast_in_dim %cst_1503, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5774 = stablehlo.divide %5684, %5773 : tensor<256xf32>
      %cst_1504 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5775 = stablehlo.broadcast_in_dim %cst_1504, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5776 = stablehlo.divide %5685, %5775 : tensor<256xf32>
      %cst_1505 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5777 = stablehlo.broadcast_in_dim %cst_1505, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5778 = stablehlo.divide %5686, %5777 : tensor<256xf32>
      %cst_1506 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5779 = stablehlo.broadcast_in_dim %cst_1506, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5780 = stablehlo.divide %5687, %5779 : tensor<512xf32>
      %cst_1507 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5781 = stablehlo.broadcast_in_dim %cst_1507, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5782 = stablehlo.divide %5688, %5781 : tensor<512xf32>
      %cst_1508 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5783 = stablehlo.broadcast_in_dim %cst_1508, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5784 = stablehlo.divide %5689, %5783 : tensor<512xf32>
      %cst_1509 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5785 = stablehlo.broadcast_in_dim %cst_1509, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5786 = stablehlo.divide %5690, %5785 : tensor<512xf32>
      %cst_1510 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5787 = stablehlo.broadcast_in_dim %cst_1510, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5788 = stablehlo.divide %5691, %5787 : tensor<512xf32>
      %cst_1511 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5789 = stablehlo.broadcast_in_dim %cst_1511, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5790 = stablehlo.divide %5692, %5789 : tensor<512xf32>
      %cst_1512 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5791 = stablehlo.broadcast_in_dim %cst_1512, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5792 = stablehlo.divide %5693, %5791 : tensor<512xf32>
      %cst_1513 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5793 = stablehlo.broadcast_in_dim %cst_1513, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5794 = stablehlo.divide %5694, %5793 : tensor<512xf32>
      %cst_1514 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5795 = stablehlo.broadcast_in_dim %cst_1514, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5796 = stablehlo.divide %5695, %5795 : tensor<512xf32>
      %cst_1515 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5797 = stablehlo.broadcast_in_dim %cst_1515, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5798 = stablehlo.divide %5696, %5797 : tensor<512xf32>
      %cst_1516 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5799 = stablehlo.broadcast_in_dim %cst_1516, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5800 = stablehlo.divide %5697, %5799 : tensor<512xf32>
      %cst_1517 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5801 = stablehlo.broadcast_in_dim %cst_1517, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5802 = stablehlo.divide %5698, %5801 : tensor<512xf32>
      %cst_1518 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5803 = stablehlo.broadcast_in_dim %cst_1518, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5804 = stablehlo.divide %5699, %5803 : tensor<512xf32>
      %cst_1519 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5805 = stablehlo.broadcast_in_dim %cst_1519, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5806 = stablehlo.divide %5700, %5805 : tensor<512xf32>
      %cst_1520 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5807 = stablehlo.broadcast_in_dim %cst_1520, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5808 = stablehlo.divide %5701, %5807 : tensor<64xf32>
      %cst_1521 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5809 = stablehlo.broadcast_in_dim %cst_1521, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5810 = stablehlo.divide %5702, %5809 : tensor<64xf32>
      %cst_1522 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5811 = stablehlo.broadcast_in_dim %cst_1522, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5812 = stablehlo.divide %5703, %5811 : tensor<64xf32>
      %cst_1523 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5813 = stablehlo.broadcast_in_dim %cst_1523, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5814 = stablehlo.divide %5704, %5813 : tensor<64xf32>
      %cst_1524 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5815 = stablehlo.broadcast_in_dim %cst_1524, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5816 = stablehlo.divide %5705, %5815 : tensor<128xf32>
      %cst_1525 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5817 = stablehlo.broadcast_in_dim %cst_1525, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5818 = stablehlo.divide %5706, %5817 : tensor<128xf32>
      %cst_1526 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5819 = stablehlo.broadcast_in_dim %cst_1526, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5820 = stablehlo.divide %5707, %5819 : tensor<128xf32>
      %cst_1527 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5821 = stablehlo.broadcast_in_dim %cst_1527, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5822 = stablehlo.divide %5708, %5821 : tensor<128xf32>
      %cst_1528 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5823 = stablehlo.broadcast_in_dim %cst_1528, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5824 = stablehlo.divide %5709, %5823 : tensor<128xf32>
      %cst_1529 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5825 = stablehlo.broadcast_in_dim %cst_1529, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5826 = stablehlo.divide %5710, %5825 : tensor<128xf32>
      %cst_1530 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5827 = stablehlo.broadcast_in_dim %cst_1530, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5828 = stablehlo.divide %5711, %5827 : tensor<128xf32>
      %cst_1531 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5829 = stablehlo.broadcast_in_dim %cst_1531, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5830 = stablehlo.divide %5712, %5829 : tensor<128xf32>
      %cst_1532 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5831 = stablehlo.broadcast_in_dim %cst_1532, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5832 = stablehlo.divide %5713, %5831 : tensor<128xf32>
      %cst_1533 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5833 = stablehlo.broadcast_in_dim %cst_1533, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5834 = stablehlo.divide %5714, %5833 : tensor<128xf32>
      %cst_1534 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5835 = stablehlo.broadcast_in_dim %cst_1534, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5836 = stablehlo.divide %5715, %5835 : tensor<128xf32>
      %cst_1535 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5837 = stablehlo.broadcast_in_dim %cst_1535, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5838 = stablehlo.divide %5716, %5837 : tensor<128xf32>
      %cst_1536 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5839 = stablehlo.broadcast_in_dim %cst_1536, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5840 = stablehlo.divide %5717, %5839 : tensor<128xf32>
      %cst_1537 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5841 = stablehlo.broadcast_in_dim %cst_1537, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5842 = stablehlo.divide %5718, %5841 : tensor<128xf32>
      %cst_1538 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5843 = stablehlo.broadcast_in_dim %cst_1538, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5844 = stablehlo.divide %5719, %5843 : tensor<128xf32>
      %cst_1539 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5845 = stablehlo.broadcast_in_dim %cst_1539, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5846 = stablehlo.divide %5720, %5845 : tensor<128xf32>
      %cst_1540 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5847 = stablehlo.broadcast_in_dim %cst_1540, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5848 = stablehlo.divide %5721, %5847 : tensor<128xf32>
      %cst_1541 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5849 = stablehlo.broadcast_in_dim %cst_1541, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %5850 = stablehlo.divide %5722, %5849 : tensor<128xf32>
      %cst_1542 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5851 = stablehlo.broadcast_in_dim %cst_1542, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5852 = stablehlo.divide %5723, %5851 : tensor<256xf32>
      %cst_1543 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5853 = stablehlo.broadcast_in_dim %cst_1543, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5854 = stablehlo.divide %5724, %5853 : tensor<256xf32>
      %cst_1544 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5855 = stablehlo.broadcast_in_dim %cst_1544, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5856 = stablehlo.divide %5725, %5855 : tensor<256xf32>
      %cst_1545 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5857 = stablehlo.broadcast_in_dim %cst_1545, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5858 = stablehlo.divide %5726, %5857 : tensor<256xf32>
      %cst_1546 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5859 = stablehlo.broadcast_in_dim %cst_1546, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5860 = stablehlo.divide %5727, %5859 : tensor<256xf32>
      %cst_1547 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5861 = stablehlo.broadcast_in_dim %cst_1547, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5862 = stablehlo.divide %5728, %5861 : tensor<256xf32>
      %cst_1548 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5863 = stablehlo.broadcast_in_dim %cst_1548, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5864 = stablehlo.divide %5729, %5863 : tensor<256xf32>
      %cst_1549 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5865 = stablehlo.broadcast_in_dim %cst_1549, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5866 = stablehlo.divide %5730, %5865 : tensor<256xf32>
      %cst_1550 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5867 = stablehlo.broadcast_in_dim %cst_1550, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5868 = stablehlo.divide %5731, %5867 : tensor<256xf32>
      %cst_1551 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5869 = stablehlo.broadcast_in_dim %cst_1551, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5870 = stablehlo.divide %5732, %5869 : tensor<256xf32>
      %cst_1552 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5871 = stablehlo.broadcast_in_dim %cst_1552, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5872 = stablehlo.divide %5733, %5871 : tensor<256xf32>
      %cst_1553 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5873 = stablehlo.broadcast_in_dim %cst_1553, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5874 = stablehlo.divide %5734, %5873 : tensor<256xf32>
      %cst_1554 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5875 = stablehlo.broadcast_in_dim %cst_1554, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5876 = stablehlo.divide %5735, %5875 : tensor<256xf32>
      %cst_1555 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5877 = stablehlo.broadcast_in_dim %cst_1555, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5878 = stablehlo.divide %5736, %5877 : tensor<256xf32>
      %cst_1556 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5879 = stablehlo.broadcast_in_dim %cst_1556, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5880 = stablehlo.divide %5737, %5879 : tensor<64xf32>
      %cst_1557 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %5881 = stablehlo.broadcast_in_dim %cst_1557, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5882 = stablehlo.divide %5738, %5881 : tensor<64xf32>
      %cst_1558 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5883 = stablehlo.broadcast_in_dim %cst_1558, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %5884 = stablehlo.multiply %5883, %arg409 : tensor<1000xf32>
      %5885 = stablehlo.add %5071, %5884 : tensor<1000xf32>
      %cst_1559 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5886 = stablehlo.broadcast_in_dim %cst_1559, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %5887 = stablehlo.multiply %5886, %arg410 : tensor<512x1000xf32>
      %5888 = stablehlo.add %5073, %5887 : tensor<512x1000xf32>
      %cst_1560 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5889 = stablehlo.broadcast_in_dim %cst_1560, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5890 = stablehlo.multiply %5889, %arg411 : tensor<64xf32>
      %5891 = stablehlo.add %5075, %5890 : tensor<64xf32>
      %cst_1561 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5892 = stablehlo.broadcast_in_dim %cst_1561, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5893 = stablehlo.multiply %5892, %arg412 : tensor<64xf32>
      %5894 = stablehlo.add %5077, %5893 : tensor<64xf32>
      %cst_1562 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5895 = stablehlo.broadcast_in_dim %cst_1562, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5896 = stablehlo.multiply %5895, %arg413 : tensor<64xf32>
      %5897 = stablehlo.add %5079, %5896 : tensor<64xf32>
      %cst_1563 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5898 = stablehlo.broadcast_in_dim %cst_1563, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5899 = stablehlo.multiply %5898, %arg414 : tensor<64xf32>
      %5900 = stablehlo.add %5081, %5899 : tensor<64xf32>
      %cst_1564 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5901 = stablehlo.broadcast_in_dim %cst_1564, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5902 = stablehlo.multiply %5901, %arg415 : tensor<3x3x64x64xf32>
      %5903 = stablehlo.add %5083, %5902 : tensor<3x3x64x64xf32>
      %cst_1565 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5904 = stablehlo.broadcast_in_dim %cst_1565, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5905 = stablehlo.multiply %5904, %arg416 : tensor<3x3x64x64xf32>
      %5906 = stablehlo.add %5085, %5905 : tensor<3x3x64x64xf32>
      %cst_1566 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5907 = stablehlo.broadcast_in_dim %cst_1566, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5908 = stablehlo.multiply %5907, %arg417 : tensor<64xf32>
      %5909 = stablehlo.add %5087, %5908 : tensor<64xf32>
      %cst_1567 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5910 = stablehlo.broadcast_in_dim %cst_1567, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5911 = stablehlo.multiply %5910, %arg418 : tensor<64xf32>
      %5912 = stablehlo.add %5089, %5911 : tensor<64xf32>
      %cst_1568 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5913 = stablehlo.broadcast_in_dim %cst_1568, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5914 = stablehlo.multiply %5913, %arg419 : tensor<64xf32>
      %5915 = stablehlo.add %5091, %5914 : tensor<64xf32>
      %cst_1569 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5916 = stablehlo.broadcast_in_dim %cst_1569, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %5917 = stablehlo.multiply %5916, %arg420 : tensor<64xf32>
      %5918 = stablehlo.add %5093, %5917 : tensor<64xf32>
      %cst_1570 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5919 = stablehlo.broadcast_in_dim %cst_1570, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5920 = stablehlo.multiply %5919, %arg421 : tensor<3x3x64x64xf32>
      %5921 = stablehlo.add %5095, %5920 : tensor<3x3x64x64xf32>
      %cst_1571 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5922 = stablehlo.broadcast_in_dim %cst_1571, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %5923 = stablehlo.multiply %5922, %arg422 : tensor<3x3x64x64xf32>
      %5924 = stablehlo.add %5097, %5923 : tensor<3x3x64x64xf32>
      %cst_1572 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5925 = stablehlo.broadcast_in_dim %cst_1572, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5926 = stablehlo.multiply %5925, %arg423 : tensor<256xf32>
      %5927 = stablehlo.add %5099, %5926 : tensor<256xf32>
      %cst_1573 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5928 = stablehlo.broadcast_in_dim %cst_1573, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5929 = stablehlo.multiply %5928, %arg424 : tensor<256xf32>
      %5930 = stablehlo.add %5101, %5929 : tensor<256xf32>
      %cst_1574 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5931 = stablehlo.broadcast_in_dim %cst_1574, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5932 = stablehlo.multiply %5931, %arg425 : tensor<256xf32>
      %5933 = stablehlo.add %5103, %5932 : tensor<256xf32>
      %cst_1575 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5934 = stablehlo.broadcast_in_dim %cst_1575, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5935 = stablehlo.multiply %5934, %arg426 : tensor<256xf32>
      %5936 = stablehlo.add %5105, %5935 : tensor<256xf32>
      %cst_1576 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5937 = stablehlo.broadcast_in_dim %cst_1576, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5938 = stablehlo.multiply %5937, %arg427 : tensor<3x3x256x256xf32>
      %5939 = stablehlo.add %5107, %5938 : tensor<3x3x256x256xf32>
      %cst_1577 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5940 = stablehlo.broadcast_in_dim %cst_1577, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5941 = stablehlo.multiply %5940, %arg428 : tensor<3x3x256x256xf32>
      %5942 = stablehlo.add %5109, %5941 : tensor<3x3x256x256xf32>
      %cst_1578 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5943 = stablehlo.broadcast_in_dim %cst_1578, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5944 = stablehlo.multiply %5943, %arg429 : tensor<256xf32>
      %5945 = stablehlo.add %5111, %5944 : tensor<256xf32>
      %cst_1579 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5946 = stablehlo.broadcast_in_dim %cst_1579, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5947 = stablehlo.multiply %5946, %arg430 : tensor<256xf32>
      %5948 = stablehlo.add %5113, %5947 : tensor<256xf32>
      %cst_1580 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5949 = stablehlo.broadcast_in_dim %cst_1580, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5950 = stablehlo.multiply %5949, %arg431 : tensor<256xf32>
      %5951 = stablehlo.add %5115, %5950 : tensor<256xf32>
      %cst_1581 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5952 = stablehlo.broadcast_in_dim %cst_1581, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5953 = stablehlo.multiply %5952, %arg432 : tensor<256xf32>
      %5954 = stablehlo.add %5117, %5953 : tensor<256xf32>
      %cst_1582 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5955 = stablehlo.broadcast_in_dim %cst_1582, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5956 = stablehlo.multiply %5955, %arg433 : tensor<3x3x256x256xf32>
      %5957 = stablehlo.add %5119, %5956 : tensor<3x3x256x256xf32>
      %cst_1583 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5958 = stablehlo.broadcast_in_dim %cst_1583, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5959 = stablehlo.multiply %5958, %arg434 : tensor<3x3x256x256xf32>
      %5960 = stablehlo.add %5121, %5959 : tensor<3x3x256x256xf32>
      %cst_1584 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5961 = stablehlo.broadcast_in_dim %cst_1584, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5962 = stablehlo.multiply %5961, %arg435 : tensor<256xf32>
      %5963 = stablehlo.add %5123, %5962 : tensor<256xf32>
      %cst_1585 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5964 = stablehlo.broadcast_in_dim %cst_1585, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5965 = stablehlo.multiply %5964, %arg436 : tensor<256xf32>
      %5966 = stablehlo.add %5125, %5965 : tensor<256xf32>
      %cst_1586 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5967 = stablehlo.broadcast_in_dim %cst_1586, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5968 = stablehlo.multiply %5967, %arg437 : tensor<256xf32>
      %5969 = stablehlo.add %5127, %5968 : tensor<256xf32>
      %cst_1587 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5970 = stablehlo.broadcast_in_dim %cst_1587, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %5971 = stablehlo.multiply %5970, %arg438 : tensor<256xf32>
      %5972 = stablehlo.add %5129, %5971 : tensor<256xf32>
      %cst_1588 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5973 = stablehlo.broadcast_in_dim %cst_1588, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5974 = stablehlo.multiply %5973, %arg439 : tensor<3x3x256x256xf32>
      %5975 = stablehlo.add %5131, %5974 : tensor<3x3x256x256xf32>
      %cst_1589 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5976 = stablehlo.broadcast_in_dim %cst_1589, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %5977 = stablehlo.multiply %5976, %arg440 : tensor<3x3x256x256xf32>
      %5978 = stablehlo.add %5133, %5977 : tensor<3x3x256x256xf32>
      %cst_1590 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5979 = stablehlo.broadcast_in_dim %cst_1590, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5980 = stablehlo.multiply %5979, %arg441 : tensor<512xf32>
      %5981 = stablehlo.add %5135, %5980 : tensor<512xf32>
      %cst_1591 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5982 = stablehlo.broadcast_in_dim %cst_1591, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5983 = stablehlo.multiply %5982, %arg442 : tensor<512xf32>
      %5984 = stablehlo.add %5137, %5983 : tensor<512xf32>
      %cst_1592 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5985 = stablehlo.broadcast_in_dim %cst_1592, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5986 = stablehlo.multiply %5985, %arg443 : tensor<512xf32>
      %5987 = stablehlo.add %5139, %5986 : tensor<512xf32>
      %cst_1593 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5988 = stablehlo.broadcast_in_dim %cst_1593, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %5989 = stablehlo.multiply %5988, %arg444 : tensor<512xf32>
      %5990 = stablehlo.add %5141, %5989 : tensor<512xf32>
      %cst_1594 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5991 = stablehlo.broadcast_in_dim %cst_1594, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %5992 = stablehlo.multiply %5991, %arg445 : tensor<3x3x256x512xf32>
      %5993 = stablehlo.add %5143, %5992 : tensor<3x3x256x512xf32>
      %cst_1595 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5994 = stablehlo.broadcast_in_dim %cst_1595, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %5995 = stablehlo.multiply %5994, %arg446 : tensor<3x3x512x512xf32>
      %5996 = stablehlo.add %5145, %5995 : tensor<3x3x512x512xf32>
      %cst_1596 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %5997 = stablehlo.broadcast_in_dim %cst_1596, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %5998 = stablehlo.multiply %5997, %arg447 : tensor<1x1x256x512xf32>
      %5999 = stablehlo.add %5147, %5998 : tensor<1x1x256x512xf32>
      %cst_1597 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6000 = stablehlo.broadcast_in_dim %cst_1597, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6001 = stablehlo.multiply %6000, %arg448 : tensor<512xf32>
      %6002 = stablehlo.add %5149, %6001 : tensor<512xf32>
      %cst_1598 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6003 = stablehlo.broadcast_in_dim %cst_1598, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6004 = stablehlo.multiply %6003, %arg449 : tensor<512xf32>
      %6005 = stablehlo.add %5151, %6004 : tensor<512xf32>
      %cst_1599 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6006 = stablehlo.broadcast_in_dim %cst_1599, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6007 = stablehlo.multiply %6006, %arg450 : tensor<512xf32>
      %6008 = stablehlo.add %5153, %6007 : tensor<512xf32>
      %cst_1600 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6009 = stablehlo.broadcast_in_dim %cst_1600, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6010 = stablehlo.multiply %6009, %arg451 : tensor<512xf32>
      %6011 = stablehlo.add %5155, %6010 : tensor<512xf32>
      %cst_1601 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6012 = stablehlo.broadcast_in_dim %cst_1601, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6013 = stablehlo.multiply %6012, %arg452 : tensor<512xf32>
      %6014 = stablehlo.add %5157, %6013 : tensor<512xf32>
      %cst_1602 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6015 = stablehlo.broadcast_in_dim %cst_1602, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6016 = stablehlo.multiply %6015, %arg453 : tensor<512xf32>
      %6017 = stablehlo.add %5159, %6016 : tensor<512xf32>
      %cst_1603 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6018 = stablehlo.broadcast_in_dim %cst_1603, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6019 = stablehlo.multiply %6018, %arg454 : tensor<3x3x512x512xf32>
      %6020 = stablehlo.add %5161, %6019 : tensor<3x3x512x512xf32>
      %cst_1604 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6021 = stablehlo.broadcast_in_dim %cst_1604, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6022 = stablehlo.multiply %6021, %arg455 : tensor<3x3x512x512xf32>
      %6023 = stablehlo.add %5163, %6022 : tensor<3x3x512x512xf32>
      %cst_1605 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6024 = stablehlo.broadcast_in_dim %cst_1605, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6025 = stablehlo.multiply %6024, %arg456 : tensor<512xf32>
      %6026 = stablehlo.add %5165, %6025 : tensor<512xf32>
      %cst_1606 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6027 = stablehlo.broadcast_in_dim %cst_1606, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6028 = stablehlo.multiply %6027, %arg457 : tensor<512xf32>
      %6029 = stablehlo.add %5167, %6028 : tensor<512xf32>
      %cst_1607 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6030 = stablehlo.broadcast_in_dim %cst_1607, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6031 = stablehlo.multiply %6030, %arg458 : tensor<512xf32>
      %6032 = stablehlo.add %5169, %6031 : tensor<512xf32>
      %cst_1608 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6033 = stablehlo.broadcast_in_dim %cst_1608, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6034 = stablehlo.multiply %6033, %arg459 : tensor<512xf32>
      %6035 = stablehlo.add %5171, %6034 : tensor<512xf32>
      %cst_1609 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6036 = stablehlo.broadcast_in_dim %cst_1609, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6037 = stablehlo.multiply %6036, %arg460 : tensor<3x3x512x512xf32>
      %6038 = stablehlo.add %5173, %6037 : tensor<3x3x512x512xf32>
      %cst_1610 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6039 = stablehlo.broadcast_in_dim %cst_1610, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6040 = stablehlo.multiply %6039, %arg461 : tensor<3x3x512x512xf32>
      %6041 = stablehlo.add %5175, %6040 : tensor<3x3x512x512xf32>
      %cst_1611 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6042 = stablehlo.broadcast_in_dim %cst_1611, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6043 = stablehlo.multiply %6042, %arg462 : tensor<64xf32>
      %6044 = stablehlo.add %5177, %6043 : tensor<64xf32>
      %cst_1612 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6045 = stablehlo.broadcast_in_dim %cst_1612, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6046 = stablehlo.multiply %6045, %arg463 : tensor<64xf32>
      %6047 = stablehlo.add %5179, %6046 : tensor<64xf32>
      %cst_1613 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6048 = stablehlo.broadcast_in_dim %cst_1613, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6049 = stablehlo.multiply %6048, %arg464 : tensor<64xf32>
      %6050 = stablehlo.add %5181, %6049 : tensor<64xf32>
      %cst_1614 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6051 = stablehlo.broadcast_in_dim %cst_1614, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6052 = stablehlo.multiply %6051, %arg465 : tensor<64xf32>
      %6053 = stablehlo.add %5183, %6052 : tensor<64xf32>
      %cst_1615 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6054 = stablehlo.broadcast_in_dim %cst_1615, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6055 = stablehlo.multiply %6054, %arg466 : tensor<3x3x64x64xf32>
      %6056 = stablehlo.add %5185, %6055 : tensor<3x3x64x64xf32>
      %cst_1616 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6057 = stablehlo.broadcast_in_dim %cst_1616, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6058 = stablehlo.multiply %6057, %arg467 : tensor<3x3x64x64xf32>
      %6059 = stablehlo.add %5187, %6058 : tensor<3x3x64x64xf32>
      %cst_1617 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6060 = stablehlo.broadcast_in_dim %cst_1617, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6061 = stablehlo.multiply %6060, %arg468 : tensor<128xf32>
      %6062 = stablehlo.add %5189, %6061 : tensor<128xf32>
      %cst_1618 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6063 = stablehlo.broadcast_in_dim %cst_1618, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6064 = stablehlo.multiply %6063, %arg469 : tensor<128xf32>
      %6065 = stablehlo.add %5191, %6064 : tensor<128xf32>
      %cst_1619 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6066 = stablehlo.broadcast_in_dim %cst_1619, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6067 = stablehlo.multiply %6066, %arg470 : tensor<128xf32>
      %6068 = stablehlo.add %5193, %6067 : tensor<128xf32>
      %cst_1620 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6069 = stablehlo.broadcast_in_dim %cst_1620, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6070 = stablehlo.multiply %6069, %arg471 : tensor<128xf32>
      %6071 = stablehlo.add %5195, %6070 : tensor<128xf32>
      %cst_1621 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6072 = stablehlo.broadcast_in_dim %cst_1621, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %6073 = stablehlo.multiply %6072, %arg472 : tensor<3x3x64x128xf32>
      %6074 = stablehlo.add %5197, %6073 : tensor<3x3x64x128xf32>
      %cst_1622 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6075 = stablehlo.broadcast_in_dim %cst_1622, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6076 = stablehlo.multiply %6075, %arg473 : tensor<3x3x128x128xf32>
      %6077 = stablehlo.add %5199, %6076 : tensor<3x3x128x128xf32>
      %cst_1623 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6078 = stablehlo.broadcast_in_dim %cst_1623, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %6079 = stablehlo.multiply %6078, %arg474 : tensor<1x1x64x128xf32>
      %6080 = stablehlo.add %5201, %6079 : tensor<1x1x64x128xf32>
      %cst_1624 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6081 = stablehlo.broadcast_in_dim %cst_1624, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6082 = stablehlo.multiply %6081, %arg475 : tensor<128xf32>
      %6083 = stablehlo.add %5203, %6082 : tensor<128xf32>
      %cst_1625 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6084 = stablehlo.broadcast_in_dim %cst_1625, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6085 = stablehlo.multiply %6084, %arg476 : tensor<128xf32>
      %6086 = stablehlo.add %5205, %6085 : tensor<128xf32>
      %cst_1626 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6087 = stablehlo.broadcast_in_dim %cst_1626, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6088 = stablehlo.multiply %6087, %arg477 : tensor<128xf32>
      %6089 = stablehlo.add %5207, %6088 : tensor<128xf32>
      %cst_1627 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6090 = stablehlo.broadcast_in_dim %cst_1627, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6091 = stablehlo.multiply %6090, %arg478 : tensor<128xf32>
      %6092 = stablehlo.add %5209, %6091 : tensor<128xf32>
      %cst_1628 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6093 = stablehlo.broadcast_in_dim %cst_1628, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6094 = stablehlo.multiply %6093, %arg479 : tensor<128xf32>
      %6095 = stablehlo.add %5211, %6094 : tensor<128xf32>
      %cst_1629 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6096 = stablehlo.broadcast_in_dim %cst_1629, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6097 = stablehlo.multiply %6096, %arg480 : tensor<128xf32>
      %6098 = stablehlo.add %5213, %6097 : tensor<128xf32>
      %cst_1630 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6099 = stablehlo.broadcast_in_dim %cst_1630, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6100 = stablehlo.multiply %6099, %arg481 : tensor<3x3x128x128xf32>
      %6101 = stablehlo.add %5215, %6100 : tensor<3x3x128x128xf32>
      %cst_1631 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6102 = stablehlo.broadcast_in_dim %cst_1631, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6103 = stablehlo.multiply %6102, %arg482 : tensor<3x3x128x128xf32>
      %6104 = stablehlo.add %5217, %6103 : tensor<3x3x128x128xf32>
      %cst_1632 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6105 = stablehlo.broadcast_in_dim %cst_1632, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6106 = stablehlo.multiply %6105, %arg483 : tensor<128xf32>
      %6107 = stablehlo.add %5219, %6106 : tensor<128xf32>
      %cst_1633 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6108 = stablehlo.broadcast_in_dim %cst_1633, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6109 = stablehlo.multiply %6108, %arg484 : tensor<128xf32>
      %6110 = stablehlo.add %5221, %6109 : tensor<128xf32>
      %cst_1634 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6111 = stablehlo.broadcast_in_dim %cst_1634, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6112 = stablehlo.multiply %6111, %arg485 : tensor<128xf32>
      %6113 = stablehlo.add %5223, %6112 : tensor<128xf32>
      %cst_1635 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6114 = stablehlo.broadcast_in_dim %cst_1635, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6115 = stablehlo.multiply %6114, %arg486 : tensor<128xf32>
      %6116 = stablehlo.add %5225, %6115 : tensor<128xf32>
      %cst_1636 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6117 = stablehlo.broadcast_in_dim %cst_1636, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6118 = stablehlo.multiply %6117, %arg487 : tensor<3x3x128x128xf32>
      %6119 = stablehlo.add %5227, %6118 : tensor<3x3x128x128xf32>
      %cst_1637 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6120 = stablehlo.broadcast_in_dim %cst_1637, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6121 = stablehlo.multiply %6120, %arg488 : tensor<3x3x128x128xf32>
      %6122 = stablehlo.add %5229, %6121 : tensor<3x3x128x128xf32>
      %cst_1638 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6123 = stablehlo.broadcast_in_dim %cst_1638, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6124 = stablehlo.multiply %6123, %arg489 : tensor<128xf32>
      %6125 = stablehlo.add %5231, %6124 : tensor<128xf32>
      %cst_1639 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6126 = stablehlo.broadcast_in_dim %cst_1639, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6127 = stablehlo.multiply %6126, %arg490 : tensor<128xf32>
      %6128 = stablehlo.add %5233, %6127 : tensor<128xf32>
      %cst_1640 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6129 = stablehlo.broadcast_in_dim %cst_1640, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6130 = stablehlo.multiply %6129, %arg491 : tensor<128xf32>
      %6131 = stablehlo.add %5235, %6130 : tensor<128xf32>
      %cst_1641 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6132 = stablehlo.broadcast_in_dim %cst_1641, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6133 = stablehlo.multiply %6132, %arg492 : tensor<128xf32>
      %6134 = stablehlo.add %5237, %6133 : tensor<128xf32>
      %cst_1642 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6135 = stablehlo.broadcast_in_dim %cst_1642, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6136 = stablehlo.multiply %6135, %arg493 : tensor<3x3x128x128xf32>
      %6137 = stablehlo.add %5239, %6136 : tensor<3x3x128x128xf32>
      %cst_1643 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6138 = stablehlo.broadcast_in_dim %cst_1643, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6139 = stablehlo.multiply %6138, %arg494 : tensor<3x3x128x128xf32>
      %6140 = stablehlo.add %5241, %6139 : tensor<3x3x128x128xf32>
      %cst_1644 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6141 = stablehlo.broadcast_in_dim %cst_1644, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6142 = stablehlo.multiply %6141, %arg495 : tensor<256xf32>
      %6143 = stablehlo.add %5243, %6142 : tensor<256xf32>
      %cst_1645 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6144 = stablehlo.broadcast_in_dim %cst_1645, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6145 = stablehlo.multiply %6144, %arg496 : tensor<256xf32>
      %6146 = stablehlo.add %5245, %6145 : tensor<256xf32>
      %cst_1646 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6147 = stablehlo.broadcast_in_dim %cst_1646, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6148 = stablehlo.multiply %6147, %arg497 : tensor<256xf32>
      %6149 = stablehlo.add %5247, %6148 : tensor<256xf32>
      %cst_1647 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6150 = stablehlo.broadcast_in_dim %cst_1647, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6151 = stablehlo.multiply %6150, %arg498 : tensor<256xf32>
      %6152 = stablehlo.add %5249, %6151 : tensor<256xf32>
      %cst_1648 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6153 = stablehlo.broadcast_in_dim %cst_1648, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %6154 = stablehlo.multiply %6153, %arg499 : tensor<3x3x128x256xf32>
      %6155 = stablehlo.add %5251, %6154 : tensor<3x3x128x256xf32>
      %cst_1649 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6156 = stablehlo.broadcast_in_dim %cst_1649, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6157 = stablehlo.multiply %6156, %arg500 : tensor<3x3x256x256xf32>
      %6158 = stablehlo.add %5253, %6157 : tensor<3x3x256x256xf32>
      %cst_1650 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6159 = stablehlo.broadcast_in_dim %cst_1650, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %6160 = stablehlo.multiply %6159, %arg501 : tensor<1x1x128x256xf32>
      %6161 = stablehlo.add %5255, %6160 : tensor<1x1x128x256xf32>
      %cst_1651 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6162 = stablehlo.broadcast_in_dim %cst_1651, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6163 = stablehlo.multiply %6162, %arg502 : tensor<256xf32>
      %6164 = stablehlo.add %5257, %6163 : tensor<256xf32>
      %cst_1652 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6165 = stablehlo.broadcast_in_dim %cst_1652, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6166 = stablehlo.multiply %6165, %arg503 : tensor<256xf32>
      %6167 = stablehlo.add %5259, %6166 : tensor<256xf32>
      %cst_1653 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6168 = stablehlo.broadcast_in_dim %cst_1653, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6169 = stablehlo.multiply %6168, %arg504 : tensor<256xf32>
      %6170 = stablehlo.add %5261, %6169 : tensor<256xf32>
      %cst_1654 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6171 = stablehlo.broadcast_in_dim %cst_1654, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6172 = stablehlo.multiply %6171, %arg505 : tensor<256xf32>
      %6173 = stablehlo.add %5263, %6172 : tensor<256xf32>
      %cst_1655 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6174 = stablehlo.broadcast_in_dim %cst_1655, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6175 = stablehlo.multiply %6174, %arg506 : tensor<256xf32>
      %6176 = stablehlo.add %5265, %6175 : tensor<256xf32>
      %cst_1656 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6177 = stablehlo.broadcast_in_dim %cst_1656, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6178 = stablehlo.multiply %6177, %arg507 : tensor<256xf32>
      %6179 = stablehlo.add %5267, %6178 : tensor<256xf32>
      %cst_1657 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6180 = stablehlo.broadcast_in_dim %cst_1657, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6181 = stablehlo.multiply %6180, %arg508 : tensor<3x3x256x256xf32>
      %6182 = stablehlo.add %5269, %6181 : tensor<3x3x256x256xf32>
      %cst_1658 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6183 = stablehlo.broadcast_in_dim %cst_1658, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6184 = stablehlo.multiply %6183, %arg509 : tensor<3x3x256x256xf32>
      %6185 = stablehlo.add %5271, %6184 : tensor<3x3x256x256xf32>
      %cst_1659 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6186 = stablehlo.broadcast_in_dim %cst_1659, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6187 = stablehlo.multiply %6186, %arg510 : tensor<256xf32>
      %6188 = stablehlo.add %5273, %6187 : tensor<256xf32>
      %cst_1660 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6189 = stablehlo.broadcast_in_dim %cst_1660, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6190 = stablehlo.multiply %6189, %arg511 : tensor<256xf32>
      %6191 = stablehlo.add %5275, %6190 : tensor<256xf32>
      %cst_1661 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6192 = stablehlo.broadcast_in_dim %cst_1661, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6193 = stablehlo.multiply %6192, %arg512 : tensor<256xf32>
      %6194 = stablehlo.add %5277, %6193 : tensor<256xf32>
      %cst_1662 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6195 = stablehlo.broadcast_in_dim %cst_1662, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6196 = stablehlo.multiply %6195, %arg513 : tensor<256xf32>
      %6197 = stablehlo.add %5279, %6196 : tensor<256xf32>
      %cst_1663 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6198 = stablehlo.broadcast_in_dim %cst_1663, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6199 = stablehlo.multiply %6198, %arg514 : tensor<3x3x256x256xf32>
      %6200 = stablehlo.add %5281, %6199 : tensor<3x3x256x256xf32>
      %cst_1664 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6201 = stablehlo.broadcast_in_dim %cst_1664, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6202 = stablehlo.multiply %6201, %arg515 : tensor<3x3x256x256xf32>
      %6203 = stablehlo.add %5283, %6202 : tensor<3x3x256x256xf32>
      %cst_1665 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6204 = stablehlo.broadcast_in_dim %cst_1665, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6205 = stablehlo.multiply %6204, %arg516 : tensor<64xf32>
      %6206 = stablehlo.add %5285, %6205 : tensor<64xf32>
      %cst_1666 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6207 = stablehlo.broadcast_in_dim %cst_1666, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6208 = stablehlo.multiply %6207, %arg517 : tensor<64xf32>
      %6209 = stablehlo.add %5287, %6208 : tensor<64xf32>
      %cst_1667 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6210 = stablehlo.broadcast_in_dim %cst_1667, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %6211 = stablehlo.multiply %6210, %arg518 : tensor<7x7x3x64xf32>
      %6212 = stablehlo.add %5289, %6211 : tensor<7x7x3x64xf32>
      %cst_1668 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6213 = stablehlo.broadcast_in_dim %cst_1668, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %6214 = stablehlo.multiply %6213, %5885 : tensor<1000xf32>
      %6215 = stablehlo.add %5071, %6214 : tensor<1000xf32>
      %cst_1669 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6216 = stablehlo.broadcast_in_dim %cst_1669, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %6217 = stablehlo.multiply %6216, %5888 : tensor<512x1000xf32>
      %6218 = stablehlo.add %5073, %6217 : tensor<512x1000xf32>
      %cst_1670 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6219 = stablehlo.broadcast_in_dim %cst_1670, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6220 = stablehlo.multiply %6219, %5891 : tensor<64xf32>
      %6221 = stablehlo.add %5075, %6220 : tensor<64xf32>
      %cst_1671 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6222 = stablehlo.broadcast_in_dim %cst_1671, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6223 = stablehlo.multiply %6222, %5894 : tensor<64xf32>
      %6224 = stablehlo.add %5077, %6223 : tensor<64xf32>
      %cst_1672 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6225 = stablehlo.broadcast_in_dim %cst_1672, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6226 = stablehlo.multiply %6225, %5897 : tensor<64xf32>
      %6227 = stablehlo.add %5079, %6226 : tensor<64xf32>
      %cst_1673 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6228 = stablehlo.broadcast_in_dim %cst_1673, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6229 = stablehlo.multiply %6228, %5900 : tensor<64xf32>
      %6230 = stablehlo.add %5081, %6229 : tensor<64xf32>
      %cst_1674 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6231 = stablehlo.broadcast_in_dim %cst_1674, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6232 = stablehlo.multiply %6231, %5903 : tensor<3x3x64x64xf32>
      %6233 = stablehlo.add %5083, %6232 : tensor<3x3x64x64xf32>
      %cst_1675 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6234 = stablehlo.broadcast_in_dim %cst_1675, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6235 = stablehlo.multiply %6234, %5906 : tensor<3x3x64x64xf32>
      %6236 = stablehlo.add %5085, %6235 : tensor<3x3x64x64xf32>
      %cst_1676 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6237 = stablehlo.broadcast_in_dim %cst_1676, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6238 = stablehlo.multiply %6237, %5909 : tensor<64xf32>
      %6239 = stablehlo.add %5087, %6238 : tensor<64xf32>
      %cst_1677 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6240 = stablehlo.broadcast_in_dim %cst_1677, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6241 = stablehlo.multiply %6240, %5912 : tensor<64xf32>
      %6242 = stablehlo.add %5089, %6241 : tensor<64xf32>
      %cst_1678 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6243 = stablehlo.broadcast_in_dim %cst_1678, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6244 = stablehlo.multiply %6243, %5915 : tensor<64xf32>
      %6245 = stablehlo.add %5091, %6244 : tensor<64xf32>
      %cst_1679 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6246 = stablehlo.broadcast_in_dim %cst_1679, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6247 = stablehlo.multiply %6246, %5918 : tensor<64xf32>
      %6248 = stablehlo.add %5093, %6247 : tensor<64xf32>
      %cst_1680 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6249 = stablehlo.broadcast_in_dim %cst_1680, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6250 = stablehlo.multiply %6249, %5921 : tensor<3x3x64x64xf32>
      %6251 = stablehlo.add %5095, %6250 : tensor<3x3x64x64xf32>
      %cst_1681 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6252 = stablehlo.broadcast_in_dim %cst_1681, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6253 = stablehlo.multiply %6252, %5924 : tensor<3x3x64x64xf32>
      %6254 = stablehlo.add %5097, %6253 : tensor<3x3x64x64xf32>
      %cst_1682 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6255 = stablehlo.broadcast_in_dim %cst_1682, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6256 = stablehlo.multiply %6255, %5927 : tensor<256xf32>
      %6257 = stablehlo.add %5099, %6256 : tensor<256xf32>
      %cst_1683 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6258 = stablehlo.broadcast_in_dim %cst_1683, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6259 = stablehlo.multiply %6258, %5930 : tensor<256xf32>
      %6260 = stablehlo.add %5101, %6259 : tensor<256xf32>
      %cst_1684 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6261 = stablehlo.broadcast_in_dim %cst_1684, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6262 = stablehlo.multiply %6261, %5933 : tensor<256xf32>
      %6263 = stablehlo.add %5103, %6262 : tensor<256xf32>
      %cst_1685 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6264 = stablehlo.broadcast_in_dim %cst_1685, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6265 = stablehlo.multiply %6264, %5936 : tensor<256xf32>
      %6266 = stablehlo.add %5105, %6265 : tensor<256xf32>
      %cst_1686 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6267 = stablehlo.broadcast_in_dim %cst_1686, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6268 = stablehlo.multiply %6267, %5939 : tensor<3x3x256x256xf32>
      %6269 = stablehlo.add %5107, %6268 : tensor<3x3x256x256xf32>
      %cst_1687 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6270 = stablehlo.broadcast_in_dim %cst_1687, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6271 = stablehlo.multiply %6270, %5942 : tensor<3x3x256x256xf32>
      %6272 = stablehlo.add %5109, %6271 : tensor<3x3x256x256xf32>
      %cst_1688 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6273 = stablehlo.broadcast_in_dim %cst_1688, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6274 = stablehlo.multiply %6273, %5945 : tensor<256xf32>
      %6275 = stablehlo.add %5111, %6274 : tensor<256xf32>
      %cst_1689 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6276 = stablehlo.broadcast_in_dim %cst_1689, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6277 = stablehlo.multiply %6276, %5948 : tensor<256xf32>
      %6278 = stablehlo.add %5113, %6277 : tensor<256xf32>
      %cst_1690 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6279 = stablehlo.broadcast_in_dim %cst_1690, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6280 = stablehlo.multiply %6279, %5951 : tensor<256xf32>
      %6281 = stablehlo.add %5115, %6280 : tensor<256xf32>
      %cst_1691 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6282 = stablehlo.broadcast_in_dim %cst_1691, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6283 = stablehlo.multiply %6282, %5954 : tensor<256xf32>
      %6284 = stablehlo.add %5117, %6283 : tensor<256xf32>
      %cst_1692 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6285 = stablehlo.broadcast_in_dim %cst_1692, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6286 = stablehlo.multiply %6285, %5957 : tensor<3x3x256x256xf32>
      %6287 = stablehlo.add %5119, %6286 : tensor<3x3x256x256xf32>
      %cst_1693 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6288 = stablehlo.broadcast_in_dim %cst_1693, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6289 = stablehlo.multiply %6288, %5960 : tensor<3x3x256x256xf32>
      %6290 = stablehlo.add %5121, %6289 : tensor<3x3x256x256xf32>
      %cst_1694 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6291 = stablehlo.broadcast_in_dim %cst_1694, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6292 = stablehlo.multiply %6291, %5963 : tensor<256xf32>
      %6293 = stablehlo.add %5123, %6292 : tensor<256xf32>
      %cst_1695 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6294 = stablehlo.broadcast_in_dim %cst_1695, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6295 = stablehlo.multiply %6294, %5966 : tensor<256xf32>
      %6296 = stablehlo.add %5125, %6295 : tensor<256xf32>
      %cst_1696 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6297 = stablehlo.broadcast_in_dim %cst_1696, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6298 = stablehlo.multiply %6297, %5969 : tensor<256xf32>
      %6299 = stablehlo.add %5127, %6298 : tensor<256xf32>
      %cst_1697 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6300 = stablehlo.broadcast_in_dim %cst_1697, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6301 = stablehlo.multiply %6300, %5972 : tensor<256xf32>
      %6302 = stablehlo.add %5129, %6301 : tensor<256xf32>
      %cst_1698 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6303 = stablehlo.broadcast_in_dim %cst_1698, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6304 = stablehlo.multiply %6303, %5975 : tensor<3x3x256x256xf32>
      %6305 = stablehlo.add %5131, %6304 : tensor<3x3x256x256xf32>
      %cst_1699 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6306 = stablehlo.broadcast_in_dim %cst_1699, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6307 = stablehlo.multiply %6306, %5978 : tensor<3x3x256x256xf32>
      %6308 = stablehlo.add %5133, %6307 : tensor<3x3x256x256xf32>
      %cst_1700 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6309 = stablehlo.broadcast_in_dim %cst_1700, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6310 = stablehlo.multiply %6309, %5981 : tensor<512xf32>
      %6311 = stablehlo.add %5135, %6310 : tensor<512xf32>
      %cst_1701 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6312 = stablehlo.broadcast_in_dim %cst_1701, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6313 = stablehlo.multiply %6312, %5984 : tensor<512xf32>
      %6314 = stablehlo.add %5137, %6313 : tensor<512xf32>
      %cst_1702 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6315 = stablehlo.broadcast_in_dim %cst_1702, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6316 = stablehlo.multiply %6315, %5987 : tensor<512xf32>
      %6317 = stablehlo.add %5139, %6316 : tensor<512xf32>
      %cst_1703 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6318 = stablehlo.broadcast_in_dim %cst_1703, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6319 = stablehlo.multiply %6318, %5990 : tensor<512xf32>
      %6320 = stablehlo.add %5141, %6319 : tensor<512xf32>
      %cst_1704 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6321 = stablehlo.broadcast_in_dim %cst_1704, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %6322 = stablehlo.multiply %6321, %5993 : tensor<3x3x256x512xf32>
      %6323 = stablehlo.add %5143, %6322 : tensor<3x3x256x512xf32>
      %cst_1705 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6324 = stablehlo.broadcast_in_dim %cst_1705, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6325 = stablehlo.multiply %6324, %5996 : tensor<3x3x512x512xf32>
      %6326 = stablehlo.add %5145, %6325 : tensor<3x3x512x512xf32>
      %cst_1706 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6327 = stablehlo.broadcast_in_dim %cst_1706, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %6328 = stablehlo.multiply %6327, %5999 : tensor<1x1x256x512xf32>
      %6329 = stablehlo.add %5147, %6328 : tensor<1x1x256x512xf32>
      %cst_1707 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6330 = stablehlo.broadcast_in_dim %cst_1707, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6331 = stablehlo.multiply %6330, %6002 : tensor<512xf32>
      %6332 = stablehlo.add %5149, %6331 : tensor<512xf32>
      %cst_1708 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6333 = stablehlo.broadcast_in_dim %cst_1708, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6334 = stablehlo.multiply %6333, %6005 : tensor<512xf32>
      %6335 = stablehlo.add %5151, %6334 : tensor<512xf32>
      %cst_1709 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6336 = stablehlo.broadcast_in_dim %cst_1709, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6337 = stablehlo.multiply %6336, %6008 : tensor<512xf32>
      %6338 = stablehlo.add %5153, %6337 : tensor<512xf32>
      %cst_1710 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6339 = stablehlo.broadcast_in_dim %cst_1710, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6340 = stablehlo.multiply %6339, %6011 : tensor<512xf32>
      %6341 = stablehlo.add %5155, %6340 : tensor<512xf32>
      %cst_1711 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6342 = stablehlo.broadcast_in_dim %cst_1711, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6343 = stablehlo.multiply %6342, %6014 : tensor<512xf32>
      %6344 = stablehlo.add %5157, %6343 : tensor<512xf32>
      %cst_1712 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6345 = stablehlo.broadcast_in_dim %cst_1712, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6346 = stablehlo.multiply %6345, %6017 : tensor<512xf32>
      %6347 = stablehlo.add %5159, %6346 : tensor<512xf32>
      %cst_1713 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6348 = stablehlo.broadcast_in_dim %cst_1713, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6349 = stablehlo.multiply %6348, %6020 : tensor<3x3x512x512xf32>
      %6350 = stablehlo.add %5161, %6349 : tensor<3x3x512x512xf32>
      %cst_1714 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6351 = stablehlo.broadcast_in_dim %cst_1714, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6352 = stablehlo.multiply %6351, %6023 : tensor<3x3x512x512xf32>
      %6353 = stablehlo.add %5163, %6352 : tensor<3x3x512x512xf32>
      %cst_1715 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6354 = stablehlo.broadcast_in_dim %cst_1715, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6355 = stablehlo.multiply %6354, %6026 : tensor<512xf32>
      %6356 = stablehlo.add %5165, %6355 : tensor<512xf32>
      %cst_1716 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6357 = stablehlo.broadcast_in_dim %cst_1716, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6358 = stablehlo.multiply %6357, %6029 : tensor<512xf32>
      %6359 = stablehlo.add %5167, %6358 : tensor<512xf32>
      %cst_1717 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6360 = stablehlo.broadcast_in_dim %cst_1717, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6361 = stablehlo.multiply %6360, %6032 : tensor<512xf32>
      %6362 = stablehlo.add %5169, %6361 : tensor<512xf32>
      %cst_1718 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6363 = stablehlo.broadcast_in_dim %cst_1718, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6364 = stablehlo.multiply %6363, %6035 : tensor<512xf32>
      %6365 = stablehlo.add %5171, %6364 : tensor<512xf32>
      %cst_1719 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6366 = stablehlo.broadcast_in_dim %cst_1719, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6367 = stablehlo.multiply %6366, %6038 : tensor<3x3x512x512xf32>
      %6368 = stablehlo.add %5173, %6367 : tensor<3x3x512x512xf32>
      %cst_1720 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6369 = stablehlo.broadcast_in_dim %cst_1720, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6370 = stablehlo.multiply %6369, %6041 : tensor<3x3x512x512xf32>
      %6371 = stablehlo.add %5175, %6370 : tensor<3x3x512x512xf32>
      %cst_1721 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6372 = stablehlo.broadcast_in_dim %cst_1721, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6373 = stablehlo.multiply %6372, %6044 : tensor<64xf32>
      %6374 = stablehlo.add %5177, %6373 : tensor<64xf32>
      %cst_1722 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6375 = stablehlo.broadcast_in_dim %cst_1722, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6376 = stablehlo.multiply %6375, %6047 : tensor<64xf32>
      %6377 = stablehlo.add %5179, %6376 : tensor<64xf32>
      %cst_1723 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6378 = stablehlo.broadcast_in_dim %cst_1723, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6379 = stablehlo.multiply %6378, %6050 : tensor<64xf32>
      %6380 = stablehlo.add %5181, %6379 : tensor<64xf32>
      %cst_1724 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6381 = stablehlo.broadcast_in_dim %cst_1724, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6382 = stablehlo.multiply %6381, %6053 : tensor<64xf32>
      %6383 = stablehlo.add %5183, %6382 : tensor<64xf32>
      %cst_1725 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6384 = stablehlo.broadcast_in_dim %cst_1725, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6385 = stablehlo.multiply %6384, %6056 : tensor<3x3x64x64xf32>
      %6386 = stablehlo.add %5185, %6385 : tensor<3x3x64x64xf32>
      %cst_1726 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6387 = stablehlo.broadcast_in_dim %cst_1726, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6388 = stablehlo.multiply %6387, %6059 : tensor<3x3x64x64xf32>
      %6389 = stablehlo.add %5187, %6388 : tensor<3x3x64x64xf32>
      %cst_1727 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6390 = stablehlo.broadcast_in_dim %cst_1727, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6391 = stablehlo.multiply %6390, %6062 : tensor<128xf32>
      %6392 = stablehlo.add %5189, %6391 : tensor<128xf32>
      %cst_1728 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6393 = stablehlo.broadcast_in_dim %cst_1728, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6394 = stablehlo.multiply %6393, %6065 : tensor<128xf32>
      %6395 = stablehlo.add %5191, %6394 : tensor<128xf32>
      %cst_1729 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6396 = stablehlo.broadcast_in_dim %cst_1729, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6397 = stablehlo.multiply %6396, %6068 : tensor<128xf32>
      %6398 = stablehlo.add %5193, %6397 : tensor<128xf32>
      %cst_1730 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6399 = stablehlo.broadcast_in_dim %cst_1730, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6400 = stablehlo.multiply %6399, %6071 : tensor<128xf32>
      %6401 = stablehlo.add %5195, %6400 : tensor<128xf32>
      %cst_1731 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6402 = stablehlo.broadcast_in_dim %cst_1731, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %6403 = stablehlo.multiply %6402, %6074 : tensor<3x3x64x128xf32>
      %6404 = stablehlo.add %5197, %6403 : tensor<3x3x64x128xf32>
      %cst_1732 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6405 = stablehlo.broadcast_in_dim %cst_1732, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6406 = stablehlo.multiply %6405, %6077 : tensor<3x3x128x128xf32>
      %6407 = stablehlo.add %5199, %6406 : tensor<3x3x128x128xf32>
      %cst_1733 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6408 = stablehlo.broadcast_in_dim %cst_1733, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %6409 = stablehlo.multiply %6408, %6080 : tensor<1x1x64x128xf32>
      %6410 = stablehlo.add %5201, %6409 : tensor<1x1x64x128xf32>
      %cst_1734 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6411 = stablehlo.broadcast_in_dim %cst_1734, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6412 = stablehlo.multiply %6411, %6083 : tensor<128xf32>
      %6413 = stablehlo.add %5203, %6412 : tensor<128xf32>
      %cst_1735 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6414 = stablehlo.broadcast_in_dim %cst_1735, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6415 = stablehlo.multiply %6414, %6086 : tensor<128xf32>
      %6416 = stablehlo.add %5205, %6415 : tensor<128xf32>
      %cst_1736 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6417 = stablehlo.broadcast_in_dim %cst_1736, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6418 = stablehlo.multiply %6417, %6089 : tensor<128xf32>
      %6419 = stablehlo.add %5207, %6418 : tensor<128xf32>
      %cst_1737 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6420 = stablehlo.broadcast_in_dim %cst_1737, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6421 = stablehlo.multiply %6420, %6092 : tensor<128xf32>
      %6422 = stablehlo.add %5209, %6421 : tensor<128xf32>
      %cst_1738 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6423 = stablehlo.broadcast_in_dim %cst_1738, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6424 = stablehlo.multiply %6423, %6095 : tensor<128xf32>
      %6425 = stablehlo.add %5211, %6424 : tensor<128xf32>
      %cst_1739 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6426 = stablehlo.broadcast_in_dim %cst_1739, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6427 = stablehlo.multiply %6426, %6098 : tensor<128xf32>
      %6428 = stablehlo.add %5213, %6427 : tensor<128xf32>
      %cst_1740 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6429 = stablehlo.broadcast_in_dim %cst_1740, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6430 = stablehlo.multiply %6429, %6101 : tensor<3x3x128x128xf32>
      %6431 = stablehlo.add %5215, %6430 : tensor<3x3x128x128xf32>
      %cst_1741 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6432 = stablehlo.broadcast_in_dim %cst_1741, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6433 = stablehlo.multiply %6432, %6104 : tensor<3x3x128x128xf32>
      %6434 = stablehlo.add %5217, %6433 : tensor<3x3x128x128xf32>
      %cst_1742 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6435 = stablehlo.broadcast_in_dim %cst_1742, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6436 = stablehlo.multiply %6435, %6107 : tensor<128xf32>
      %6437 = stablehlo.add %5219, %6436 : tensor<128xf32>
      %cst_1743 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6438 = stablehlo.broadcast_in_dim %cst_1743, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6439 = stablehlo.multiply %6438, %6110 : tensor<128xf32>
      %6440 = stablehlo.add %5221, %6439 : tensor<128xf32>
      %cst_1744 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6441 = stablehlo.broadcast_in_dim %cst_1744, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6442 = stablehlo.multiply %6441, %6113 : tensor<128xf32>
      %6443 = stablehlo.add %5223, %6442 : tensor<128xf32>
      %cst_1745 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6444 = stablehlo.broadcast_in_dim %cst_1745, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6445 = stablehlo.multiply %6444, %6116 : tensor<128xf32>
      %6446 = stablehlo.add %5225, %6445 : tensor<128xf32>
      %cst_1746 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6447 = stablehlo.broadcast_in_dim %cst_1746, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6448 = stablehlo.multiply %6447, %6119 : tensor<3x3x128x128xf32>
      %6449 = stablehlo.add %5227, %6448 : tensor<3x3x128x128xf32>
      %cst_1747 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6450 = stablehlo.broadcast_in_dim %cst_1747, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6451 = stablehlo.multiply %6450, %6122 : tensor<3x3x128x128xf32>
      %6452 = stablehlo.add %5229, %6451 : tensor<3x3x128x128xf32>
      %cst_1748 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6453 = stablehlo.broadcast_in_dim %cst_1748, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6454 = stablehlo.multiply %6453, %6125 : tensor<128xf32>
      %6455 = stablehlo.add %5231, %6454 : tensor<128xf32>
      %cst_1749 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6456 = stablehlo.broadcast_in_dim %cst_1749, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6457 = stablehlo.multiply %6456, %6128 : tensor<128xf32>
      %6458 = stablehlo.add %5233, %6457 : tensor<128xf32>
      %cst_1750 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6459 = stablehlo.broadcast_in_dim %cst_1750, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6460 = stablehlo.multiply %6459, %6131 : tensor<128xf32>
      %6461 = stablehlo.add %5235, %6460 : tensor<128xf32>
      %cst_1751 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6462 = stablehlo.broadcast_in_dim %cst_1751, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6463 = stablehlo.multiply %6462, %6134 : tensor<128xf32>
      %6464 = stablehlo.add %5237, %6463 : tensor<128xf32>
      %cst_1752 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6465 = stablehlo.broadcast_in_dim %cst_1752, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6466 = stablehlo.multiply %6465, %6137 : tensor<3x3x128x128xf32>
      %6467 = stablehlo.add %5239, %6466 : tensor<3x3x128x128xf32>
      %cst_1753 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6468 = stablehlo.broadcast_in_dim %cst_1753, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6469 = stablehlo.multiply %6468, %6140 : tensor<3x3x128x128xf32>
      %6470 = stablehlo.add %5241, %6469 : tensor<3x3x128x128xf32>
      %cst_1754 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6471 = stablehlo.broadcast_in_dim %cst_1754, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6472 = stablehlo.multiply %6471, %6143 : tensor<256xf32>
      %6473 = stablehlo.add %5243, %6472 : tensor<256xf32>
      %cst_1755 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6474 = stablehlo.broadcast_in_dim %cst_1755, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6475 = stablehlo.multiply %6474, %6146 : tensor<256xf32>
      %6476 = stablehlo.add %5245, %6475 : tensor<256xf32>
      %cst_1756 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6477 = stablehlo.broadcast_in_dim %cst_1756, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6478 = stablehlo.multiply %6477, %6149 : tensor<256xf32>
      %6479 = stablehlo.add %5247, %6478 : tensor<256xf32>
      %cst_1757 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6480 = stablehlo.broadcast_in_dim %cst_1757, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6481 = stablehlo.multiply %6480, %6152 : tensor<256xf32>
      %6482 = stablehlo.add %5249, %6481 : tensor<256xf32>
      %cst_1758 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6483 = stablehlo.broadcast_in_dim %cst_1758, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %6484 = stablehlo.multiply %6483, %6155 : tensor<3x3x128x256xf32>
      %6485 = stablehlo.add %5251, %6484 : tensor<3x3x128x256xf32>
      %cst_1759 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6486 = stablehlo.broadcast_in_dim %cst_1759, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6487 = stablehlo.multiply %6486, %6158 : tensor<3x3x256x256xf32>
      %6488 = stablehlo.add %5253, %6487 : tensor<3x3x256x256xf32>
      %cst_1760 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6489 = stablehlo.broadcast_in_dim %cst_1760, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %6490 = stablehlo.multiply %6489, %6161 : tensor<1x1x128x256xf32>
      %6491 = stablehlo.add %5255, %6490 : tensor<1x1x128x256xf32>
      %cst_1761 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6492 = stablehlo.broadcast_in_dim %cst_1761, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6493 = stablehlo.multiply %6492, %6164 : tensor<256xf32>
      %6494 = stablehlo.add %5257, %6493 : tensor<256xf32>
      %cst_1762 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6495 = stablehlo.broadcast_in_dim %cst_1762, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6496 = stablehlo.multiply %6495, %6167 : tensor<256xf32>
      %6497 = stablehlo.add %5259, %6496 : tensor<256xf32>
      %cst_1763 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6498 = stablehlo.broadcast_in_dim %cst_1763, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6499 = stablehlo.multiply %6498, %6170 : tensor<256xf32>
      %6500 = stablehlo.add %5261, %6499 : tensor<256xf32>
      %cst_1764 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6501 = stablehlo.broadcast_in_dim %cst_1764, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6502 = stablehlo.multiply %6501, %6173 : tensor<256xf32>
      %6503 = stablehlo.add %5263, %6502 : tensor<256xf32>
      %cst_1765 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6504 = stablehlo.broadcast_in_dim %cst_1765, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6505 = stablehlo.multiply %6504, %6176 : tensor<256xf32>
      %6506 = stablehlo.add %5265, %6505 : tensor<256xf32>
      %cst_1766 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6507 = stablehlo.broadcast_in_dim %cst_1766, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6508 = stablehlo.multiply %6507, %6179 : tensor<256xf32>
      %6509 = stablehlo.add %5267, %6508 : tensor<256xf32>
      %cst_1767 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6510 = stablehlo.broadcast_in_dim %cst_1767, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6511 = stablehlo.multiply %6510, %6182 : tensor<3x3x256x256xf32>
      %6512 = stablehlo.add %5269, %6511 : tensor<3x3x256x256xf32>
      %cst_1768 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6513 = stablehlo.broadcast_in_dim %cst_1768, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6514 = stablehlo.multiply %6513, %6185 : tensor<3x3x256x256xf32>
      %6515 = stablehlo.add %5271, %6514 : tensor<3x3x256x256xf32>
      %cst_1769 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6516 = stablehlo.broadcast_in_dim %cst_1769, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6517 = stablehlo.multiply %6516, %6188 : tensor<256xf32>
      %6518 = stablehlo.add %5273, %6517 : tensor<256xf32>
      %cst_1770 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6519 = stablehlo.broadcast_in_dim %cst_1770, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6520 = stablehlo.multiply %6519, %6191 : tensor<256xf32>
      %6521 = stablehlo.add %5275, %6520 : tensor<256xf32>
      %cst_1771 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6522 = stablehlo.broadcast_in_dim %cst_1771, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6523 = stablehlo.multiply %6522, %6194 : tensor<256xf32>
      %6524 = stablehlo.add %5277, %6523 : tensor<256xf32>
      %cst_1772 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6525 = stablehlo.broadcast_in_dim %cst_1772, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6526 = stablehlo.multiply %6525, %6197 : tensor<256xf32>
      %6527 = stablehlo.add %5279, %6526 : tensor<256xf32>
      %cst_1773 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6528 = stablehlo.broadcast_in_dim %cst_1773, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6529 = stablehlo.multiply %6528, %6200 : tensor<3x3x256x256xf32>
      %6530 = stablehlo.add %5281, %6529 : tensor<3x3x256x256xf32>
      %cst_1774 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6531 = stablehlo.broadcast_in_dim %cst_1774, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6532 = stablehlo.multiply %6531, %6203 : tensor<3x3x256x256xf32>
      %6533 = stablehlo.add %5283, %6532 : tensor<3x3x256x256xf32>
      %cst_1775 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6534 = stablehlo.broadcast_in_dim %cst_1775, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6535 = stablehlo.multiply %6534, %6206 : tensor<64xf32>
      %6536 = stablehlo.add %5285, %6535 : tensor<64xf32>
      %cst_1776 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6537 = stablehlo.broadcast_in_dim %cst_1776, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6538 = stablehlo.multiply %6537, %6209 : tensor<64xf32>
      %6539 = stablehlo.add %5287, %6538 : tensor<64xf32>
      %cst_1777 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %6540 = stablehlo.broadcast_in_dim %cst_1777, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %6541 = stablehlo.multiply %6540, %6212 : tensor<7x7x3x64xf32>
      %6542 = stablehlo.add %5289, %6541 : tensor<7x7x3x64xf32>
      %c_1778 = stablehlo.constant dense<0> : tensor<i32>
      %6543 = stablehlo.subtract %arg519, %c_1778 : tensor<i32>
      %c_1779 = stablehlo.constant dense<0> : tensor<i32>
      %cst_1780 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %6544 = func.call @clip_542(%6543, %c_1779, %cst_1780) : (tensor<i32>, tensor<i32>, tensor<f32>) -> tensor<f32>
      %cst_1781 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %6545 = stablehlo.divide %6544, %cst_1781 : tensor<f32>
      %cst_1782 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %6546 = stablehlo.subtract %cst_1782, %6545 : tensor<f32>
      %cst_1783 = stablehlo.constant dense<-4.000000e-01> : tensor<f32>
      %6547 = stablehlo.multiply %cst_1783, %6546 : tensor<f32>
      %cst_1784 = stablehlo.constant dense<4.000000e-01> : tensor<f32>
      %6548 = stablehlo.add %6547, %cst_1784 : tensor<f32>
      %6549 = stablehlo.convert %arg519 : (tensor<i32>) -> tensor<f32>
      %cst_1785 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %6550 = stablehlo.compare  LT, %6549, %cst_1785,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6551 = stablehlo.convert %arg519 : (tensor<i32>) -> tensor<f32>
      %cst_1786 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %6552 = stablehlo.subtract %6551, %cst_1786 : tensor<f32>
      %cst_1787 = stablehlo.constant dense<1.188450e+05> : tensor<f32>
      %6553 = stablehlo.minimum %6552, %cst_1787 : tensor<f32>
      %cst_1788 = stablehlo.constant dense<3.14159274> : tensor<f32>
      %6554 = stablehlo.multiply %cst_1788, %6553 : tensor<f32>
      %cst_1789 = stablehlo.constant dense<1.188450e+05> : tensor<f32>
      %6555 = stablehlo.divide %6554, %cst_1789 : tensor<f32>
      %6556 = stablehlo.cosine %6555 : tensor<f32>
      %cst_1790 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %6557 = stablehlo.add %cst_1790, %6556 : tensor<f32>
      %cst_1791 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %6558 = stablehlo.multiply %cst_1791, %6557 : tensor<f32>
      %cst_1792 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %6559 = stablehlo.power %6558, %cst_1792 : tensor<f32>
      %cst_1793 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %6560 = stablehlo.multiply %cst_1793, %6559 : tensor<f32>
      %cst_1794 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %6561 = stablehlo.add %6560, %cst_1794 : tensor<f32>
      %cst_1795 = stablehlo.constant dense<4.000000e-01> : tensor<f32>
      %6562 = stablehlo.multiply %cst_1795, %6561 : tensor<f32>
      %6563 = func.call @_where(%6550, %6548, %6562) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_1796 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
      %6564 = stablehlo.multiply %cst_1796, %6563 : tensor<f32>
      %6565 = stablehlo.convert %6564 : tensor<f32>
      %6566 = stablehlo.broadcast_in_dim %6565, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %6567 = stablehlo.multiply %6566, %6215 : tensor<1000xf32>
      %6568 = stablehlo.convert %6564 : tensor<f32>
      %6569 = stablehlo.broadcast_in_dim %6568, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %6570 = stablehlo.multiply %6569, %6218 : tensor<512x1000xf32>
      %6571 = stablehlo.convert %6564 : tensor<f32>
      %6572 = stablehlo.broadcast_in_dim %6571, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6573 = stablehlo.multiply %6572, %6221 : tensor<64xf32>
      %6574 = stablehlo.convert %6564 : tensor<f32>
      %6575 = stablehlo.broadcast_in_dim %6574, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6576 = stablehlo.multiply %6575, %6224 : tensor<64xf32>
      %6577 = stablehlo.convert %6564 : tensor<f32>
      %6578 = stablehlo.broadcast_in_dim %6577, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6579 = stablehlo.multiply %6578, %6227 : tensor<64xf32>
      %6580 = stablehlo.convert %6564 : tensor<f32>
      %6581 = stablehlo.broadcast_in_dim %6580, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6582 = stablehlo.multiply %6581, %6230 : tensor<64xf32>
      %6583 = stablehlo.convert %6564 : tensor<f32>
      %6584 = stablehlo.broadcast_in_dim %6583, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6585 = stablehlo.multiply %6584, %6233 : tensor<3x3x64x64xf32>
      %6586 = stablehlo.convert %6564 : tensor<f32>
      %6587 = stablehlo.broadcast_in_dim %6586, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6588 = stablehlo.multiply %6587, %6236 : tensor<3x3x64x64xf32>
      %6589 = stablehlo.convert %6564 : tensor<f32>
      %6590 = stablehlo.broadcast_in_dim %6589, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6591 = stablehlo.multiply %6590, %6239 : tensor<64xf32>
      %6592 = stablehlo.convert %6564 : tensor<f32>
      %6593 = stablehlo.broadcast_in_dim %6592, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6594 = stablehlo.multiply %6593, %6242 : tensor<64xf32>
      %6595 = stablehlo.convert %6564 : tensor<f32>
      %6596 = stablehlo.broadcast_in_dim %6595, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6597 = stablehlo.multiply %6596, %6245 : tensor<64xf32>
      %6598 = stablehlo.convert %6564 : tensor<f32>
      %6599 = stablehlo.broadcast_in_dim %6598, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6600 = stablehlo.multiply %6599, %6248 : tensor<64xf32>
      %6601 = stablehlo.convert %6564 : tensor<f32>
      %6602 = stablehlo.broadcast_in_dim %6601, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6603 = stablehlo.multiply %6602, %6251 : tensor<3x3x64x64xf32>
      %6604 = stablehlo.convert %6564 : tensor<f32>
      %6605 = stablehlo.broadcast_in_dim %6604, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6606 = stablehlo.multiply %6605, %6254 : tensor<3x3x64x64xf32>
      %6607 = stablehlo.convert %6564 : tensor<f32>
      %6608 = stablehlo.broadcast_in_dim %6607, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6609 = stablehlo.multiply %6608, %6257 : tensor<256xf32>
      %6610 = stablehlo.convert %6564 : tensor<f32>
      %6611 = stablehlo.broadcast_in_dim %6610, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6612 = stablehlo.multiply %6611, %6260 : tensor<256xf32>
      %6613 = stablehlo.convert %6564 : tensor<f32>
      %6614 = stablehlo.broadcast_in_dim %6613, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6615 = stablehlo.multiply %6614, %6263 : tensor<256xf32>
      %6616 = stablehlo.convert %6564 : tensor<f32>
      %6617 = stablehlo.broadcast_in_dim %6616, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6618 = stablehlo.multiply %6617, %6266 : tensor<256xf32>
      %6619 = stablehlo.convert %6564 : tensor<f32>
      %6620 = stablehlo.broadcast_in_dim %6619, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6621 = stablehlo.multiply %6620, %6269 : tensor<3x3x256x256xf32>
      %6622 = stablehlo.convert %6564 : tensor<f32>
      %6623 = stablehlo.broadcast_in_dim %6622, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6624 = stablehlo.multiply %6623, %6272 : tensor<3x3x256x256xf32>
      %6625 = stablehlo.convert %6564 : tensor<f32>
      %6626 = stablehlo.broadcast_in_dim %6625, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6627 = stablehlo.multiply %6626, %6275 : tensor<256xf32>
      %6628 = stablehlo.convert %6564 : tensor<f32>
      %6629 = stablehlo.broadcast_in_dim %6628, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6630 = stablehlo.multiply %6629, %6278 : tensor<256xf32>
      %6631 = stablehlo.convert %6564 : tensor<f32>
      %6632 = stablehlo.broadcast_in_dim %6631, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6633 = stablehlo.multiply %6632, %6281 : tensor<256xf32>
      %6634 = stablehlo.convert %6564 : tensor<f32>
      %6635 = stablehlo.broadcast_in_dim %6634, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6636 = stablehlo.multiply %6635, %6284 : tensor<256xf32>
      %6637 = stablehlo.convert %6564 : tensor<f32>
      %6638 = stablehlo.broadcast_in_dim %6637, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6639 = stablehlo.multiply %6638, %6287 : tensor<3x3x256x256xf32>
      %6640 = stablehlo.convert %6564 : tensor<f32>
      %6641 = stablehlo.broadcast_in_dim %6640, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6642 = stablehlo.multiply %6641, %6290 : tensor<3x3x256x256xf32>
      %6643 = stablehlo.convert %6564 : tensor<f32>
      %6644 = stablehlo.broadcast_in_dim %6643, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6645 = stablehlo.multiply %6644, %6293 : tensor<256xf32>
      %6646 = stablehlo.convert %6564 : tensor<f32>
      %6647 = stablehlo.broadcast_in_dim %6646, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6648 = stablehlo.multiply %6647, %6296 : tensor<256xf32>
      %6649 = stablehlo.convert %6564 : tensor<f32>
      %6650 = stablehlo.broadcast_in_dim %6649, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6651 = stablehlo.multiply %6650, %6299 : tensor<256xf32>
      %6652 = stablehlo.convert %6564 : tensor<f32>
      %6653 = stablehlo.broadcast_in_dim %6652, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6654 = stablehlo.multiply %6653, %6302 : tensor<256xf32>
      %6655 = stablehlo.convert %6564 : tensor<f32>
      %6656 = stablehlo.broadcast_in_dim %6655, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6657 = stablehlo.multiply %6656, %6305 : tensor<3x3x256x256xf32>
      %6658 = stablehlo.convert %6564 : tensor<f32>
      %6659 = stablehlo.broadcast_in_dim %6658, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6660 = stablehlo.multiply %6659, %6308 : tensor<3x3x256x256xf32>
      %6661 = stablehlo.convert %6564 : tensor<f32>
      %6662 = stablehlo.broadcast_in_dim %6661, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6663 = stablehlo.multiply %6662, %6311 : tensor<512xf32>
      %6664 = stablehlo.convert %6564 : tensor<f32>
      %6665 = stablehlo.broadcast_in_dim %6664, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6666 = stablehlo.multiply %6665, %6314 : tensor<512xf32>
      %6667 = stablehlo.convert %6564 : tensor<f32>
      %6668 = stablehlo.broadcast_in_dim %6667, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6669 = stablehlo.multiply %6668, %6317 : tensor<512xf32>
      %6670 = stablehlo.convert %6564 : tensor<f32>
      %6671 = stablehlo.broadcast_in_dim %6670, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6672 = stablehlo.multiply %6671, %6320 : tensor<512xf32>
      %6673 = stablehlo.convert %6564 : tensor<f32>
      %6674 = stablehlo.broadcast_in_dim %6673, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %6675 = stablehlo.multiply %6674, %6323 : tensor<3x3x256x512xf32>
      %6676 = stablehlo.convert %6564 : tensor<f32>
      %6677 = stablehlo.broadcast_in_dim %6676, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6678 = stablehlo.multiply %6677, %6326 : tensor<3x3x512x512xf32>
      %6679 = stablehlo.convert %6564 : tensor<f32>
      %6680 = stablehlo.broadcast_in_dim %6679, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %6681 = stablehlo.multiply %6680, %6329 : tensor<1x1x256x512xf32>
      %6682 = stablehlo.convert %6564 : tensor<f32>
      %6683 = stablehlo.broadcast_in_dim %6682, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6684 = stablehlo.multiply %6683, %6332 : tensor<512xf32>
      %6685 = stablehlo.convert %6564 : tensor<f32>
      %6686 = stablehlo.broadcast_in_dim %6685, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6687 = stablehlo.multiply %6686, %6335 : tensor<512xf32>
      %6688 = stablehlo.convert %6564 : tensor<f32>
      %6689 = stablehlo.broadcast_in_dim %6688, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6690 = stablehlo.multiply %6689, %6338 : tensor<512xf32>
      %6691 = stablehlo.convert %6564 : tensor<f32>
      %6692 = stablehlo.broadcast_in_dim %6691, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6693 = stablehlo.multiply %6692, %6341 : tensor<512xf32>
      %6694 = stablehlo.convert %6564 : tensor<f32>
      %6695 = stablehlo.broadcast_in_dim %6694, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6696 = stablehlo.multiply %6695, %6344 : tensor<512xf32>
      %6697 = stablehlo.convert %6564 : tensor<f32>
      %6698 = stablehlo.broadcast_in_dim %6697, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6699 = stablehlo.multiply %6698, %6347 : tensor<512xf32>
      %6700 = stablehlo.convert %6564 : tensor<f32>
      %6701 = stablehlo.broadcast_in_dim %6700, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6702 = stablehlo.multiply %6701, %6350 : tensor<3x3x512x512xf32>
      %6703 = stablehlo.convert %6564 : tensor<f32>
      %6704 = stablehlo.broadcast_in_dim %6703, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6705 = stablehlo.multiply %6704, %6353 : tensor<3x3x512x512xf32>
      %6706 = stablehlo.convert %6564 : tensor<f32>
      %6707 = stablehlo.broadcast_in_dim %6706, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6708 = stablehlo.multiply %6707, %6356 : tensor<512xf32>
      %6709 = stablehlo.convert %6564 : tensor<f32>
      %6710 = stablehlo.broadcast_in_dim %6709, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6711 = stablehlo.multiply %6710, %6359 : tensor<512xf32>
      %6712 = stablehlo.convert %6564 : tensor<f32>
      %6713 = stablehlo.broadcast_in_dim %6712, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6714 = stablehlo.multiply %6713, %6362 : tensor<512xf32>
      %6715 = stablehlo.convert %6564 : tensor<f32>
      %6716 = stablehlo.broadcast_in_dim %6715, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %6717 = stablehlo.multiply %6716, %6365 : tensor<512xf32>
      %6718 = stablehlo.convert %6564 : tensor<f32>
      %6719 = stablehlo.broadcast_in_dim %6718, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6720 = stablehlo.multiply %6719, %6368 : tensor<3x3x512x512xf32>
      %6721 = stablehlo.convert %6564 : tensor<f32>
      %6722 = stablehlo.broadcast_in_dim %6721, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %6723 = stablehlo.multiply %6722, %6371 : tensor<3x3x512x512xf32>
      %6724 = stablehlo.convert %6564 : tensor<f32>
      %6725 = stablehlo.broadcast_in_dim %6724, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6726 = stablehlo.multiply %6725, %6374 : tensor<64xf32>
      %6727 = stablehlo.convert %6564 : tensor<f32>
      %6728 = stablehlo.broadcast_in_dim %6727, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6729 = stablehlo.multiply %6728, %6377 : tensor<64xf32>
      %6730 = stablehlo.convert %6564 : tensor<f32>
      %6731 = stablehlo.broadcast_in_dim %6730, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6732 = stablehlo.multiply %6731, %6380 : tensor<64xf32>
      %6733 = stablehlo.convert %6564 : tensor<f32>
      %6734 = stablehlo.broadcast_in_dim %6733, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6735 = stablehlo.multiply %6734, %6383 : tensor<64xf32>
      %6736 = stablehlo.convert %6564 : tensor<f32>
      %6737 = stablehlo.broadcast_in_dim %6736, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6738 = stablehlo.multiply %6737, %6386 : tensor<3x3x64x64xf32>
      %6739 = stablehlo.convert %6564 : tensor<f32>
      %6740 = stablehlo.broadcast_in_dim %6739, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %6741 = stablehlo.multiply %6740, %6389 : tensor<3x3x64x64xf32>
      %6742 = stablehlo.convert %6564 : tensor<f32>
      %6743 = stablehlo.broadcast_in_dim %6742, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6744 = stablehlo.multiply %6743, %6392 : tensor<128xf32>
      %6745 = stablehlo.convert %6564 : tensor<f32>
      %6746 = stablehlo.broadcast_in_dim %6745, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6747 = stablehlo.multiply %6746, %6395 : tensor<128xf32>
      %6748 = stablehlo.convert %6564 : tensor<f32>
      %6749 = stablehlo.broadcast_in_dim %6748, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6750 = stablehlo.multiply %6749, %6398 : tensor<128xf32>
      %6751 = stablehlo.convert %6564 : tensor<f32>
      %6752 = stablehlo.broadcast_in_dim %6751, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6753 = stablehlo.multiply %6752, %6401 : tensor<128xf32>
      %6754 = stablehlo.convert %6564 : tensor<f32>
      %6755 = stablehlo.broadcast_in_dim %6754, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %6756 = stablehlo.multiply %6755, %6404 : tensor<3x3x64x128xf32>
      %6757 = stablehlo.convert %6564 : tensor<f32>
      %6758 = stablehlo.broadcast_in_dim %6757, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6759 = stablehlo.multiply %6758, %6407 : tensor<3x3x128x128xf32>
      %6760 = stablehlo.convert %6564 : tensor<f32>
      %6761 = stablehlo.broadcast_in_dim %6760, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %6762 = stablehlo.multiply %6761, %6410 : tensor<1x1x64x128xf32>
      %6763 = stablehlo.convert %6564 : tensor<f32>
      %6764 = stablehlo.broadcast_in_dim %6763, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6765 = stablehlo.multiply %6764, %6413 : tensor<128xf32>
      %6766 = stablehlo.convert %6564 : tensor<f32>
      %6767 = stablehlo.broadcast_in_dim %6766, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6768 = stablehlo.multiply %6767, %6416 : tensor<128xf32>
      %6769 = stablehlo.convert %6564 : tensor<f32>
      %6770 = stablehlo.broadcast_in_dim %6769, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6771 = stablehlo.multiply %6770, %6419 : tensor<128xf32>
      %6772 = stablehlo.convert %6564 : tensor<f32>
      %6773 = stablehlo.broadcast_in_dim %6772, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6774 = stablehlo.multiply %6773, %6422 : tensor<128xf32>
      %6775 = stablehlo.convert %6564 : tensor<f32>
      %6776 = stablehlo.broadcast_in_dim %6775, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6777 = stablehlo.multiply %6776, %6425 : tensor<128xf32>
      %6778 = stablehlo.convert %6564 : tensor<f32>
      %6779 = stablehlo.broadcast_in_dim %6778, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6780 = stablehlo.multiply %6779, %6428 : tensor<128xf32>
      %6781 = stablehlo.convert %6564 : tensor<f32>
      %6782 = stablehlo.broadcast_in_dim %6781, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6783 = stablehlo.multiply %6782, %6431 : tensor<3x3x128x128xf32>
      %6784 = stablehlo.convert %6564 : tensor<f32>
      %6785 = stablehlo.broadcast_in_dim %6784, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6786 = stablehlo.multiply %6785, %6434 : tensor<3x3x128x128xf32>
      %6787 = stablehlo.convert %6564 : tensor<f32>
      %6788 = stablehlo.broadcast_in_dim %6787, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6789 = stablehlo.multiply %6788, %6437 : tensor<128xf32>
      %6790 = stablehlo.convert %6564 : tensor<f32>
      %6791 = stablehlo.broadcast_in_dim %6790, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6792 = stablehlo.multiply %6791, %6440 : tensor<128xf32>
      %6793 = stablehlo.convert %6564 : tensor<f32>
      %6794 = stablehlo.broadcast_in_dim %6793, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6795 = stablehlo.multiply %6794, %6443 : tensor<128xf32>
      %6796 = stablehlo.convert %6564 : tensor<f32>
      %6797 = stablehlo.broadcast_in_dim %6796, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6798 = stablehlo.multiply %6797, %6446 : tensor<128xf32>
      %6799 = stablehlo.convert %6564 : tensor<f32>
      %6800 = stablehlo.broadcast_in_dim %6799, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6801 = stablehlo.multiply %6800, %6449 : tensor<3x3x128x128xf32>
      %6802 = stablehlo.convert %6564 : tensor<f32>
      %6803 = stablehlo.broadcast_in_dim %6802, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6804 = stablehlo.multiply %6803, %6452 : tensor<3x3x128x128xf32>
      %6805 = stablehlo.convert %6564 : tensor<f32>
      %6806 = stablehlo.broadcast_in_dim %6805, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6807 = stablehlo.multiply %6806, %6455 : tensor<128xf32>
      %6808 = stablehlo.convert %6564 : tensor<f32>
      %6809 = stablehlo.broadcast_in_dim %6808, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6810 = stablehlo.multiply %6809, %6458 : tensor<128xf32>
      %6811 = stablehlo.convert %6564 : tensor<f32>
      %6812 = stablehlo.broadcast_in_dim %6811, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6813 = stablehlo.multiply %6812, %6461 : tensor<128xf32>
      %6814 = stablehlo.convert %6564 : tensor<f32>
      %6815 = stablehlo.broadcast_in_dim %6814, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %6816 = stablehlo.multiply %6815, %6464 : tensor<128xf32>
      %6817 = stablehlo.convert %6564 : tensor<f32>
      %6818 = stablehlo.broadcast_in_dim %6817, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6819 = stablehlo.multiply %6818, %6467 : tensor<3x3x128x128xf32>
      %6820 = stablehlo.convert %6564 : tensor<f32>
      %6821 = stablehlo.broadcast_in_dim %6820, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %6822 = stablehlo.multiply %6821, %6470 : tensor<3x3x128x128xf32>
      %6823 = stablehlo.convert %6564 : tensor<f32>
      %6824 = stablehlo.broadcast_in_dim %6823, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6825 = stablehlo.multiply %6824, %6473 : tensor<256xf32>
      %6826 = stablehlo.convert %6564 : tensor<f32>
      %6827 = stablehlo.broadcast_in_dim %6826, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6828 = stablehlo.multiply %6827, %6476 : tensor<256xf32>
      %6829 = stablehlo.convert %6564 : tensor<f32>
      %6830 = stablehlo.broadcast_in_dim %6829, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6831 = stablehlo.multiply %6830, %6479 : tensor<256xf32>
      %6832 = stablehlo.convert %6564 : tensor<f32>
      %6833 = stablehlo.broadcast_in_dim %6832, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6834 = stablehlo.multiply %6833, %6482 : tensor<256xf32>
      %6835 = stablehlo.convert %6564 : tensor<f32>
      %6836 = stablehlo.broadcast_in_dim %6835, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %6837 = stablehlo.multiply %6836, %6485 : tensor<3x3x128x256xf32>
      %6838 = stablehlo.convert %6564 : tensor<f32>
      %6839 = stablehlo.broadcast_in_dim %6838, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6840 = stablehlo.multiply %6839, %6488 : tensor<3x3x256x256xf32>
      %6841 = stablehlo.convert %6564 : tensor<f32>
      %6842 = stablehlo.broadcast_in_dim %6841, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %6843 = stablehlo.multiply %6842, %6491 : tensor<1x1x128x256xf32>
      %6844 = stablehlo.convert %6564 : tensor<f32>
      %6845 = stablehlo.broadcast_in_dim %6844, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6846 = stablehlo.multiply %6845, %6494 : tensor<256xf32>
      %6847 = stablehlo.convert %6564 : tensor<f32>
      %6848 = stablehlo.broadcast_in_dim %6847, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6849 = stablehlo.multiply %6848, %6497 : tensor<256xf32>
      %6850 = stablehlo.convert %6564 : tensor<f32>
      %6851 = stablehlo.broadcast_in_dim %6850, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6852 = stablehlo.multiply %6851, %6500 : tensor<256xf32>
      %6853 = stablehlo.convert %6564 : tensor<f32>
      %6854 = stablehlo.broadcast_in_dim %6853, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6855 = stablehlo.multiply %6854, %6503 : tensor<256xf32>
      %6856 = stablehlo.convert %6564 : tensor<f32>
      %6857 = stablehlo.broadcast_in_dim %6856, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6858 = stablehlo.multiply %6857, %6506 : tensor<256xf32>
      %6859 = stablehlo.convert %6564 : tensor<f32>
      %6860 = stablehlo.broadcast_in_dim %6859, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6861 = stablehlo.multiply %6860, %6509 : tensor<256xf32>
      %6862 = stablehlo.convert %6564 : tensor<f32>
      %6863 = stablehlo.broadcast_in_dim %6862, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6864 = stablehlo.multiply %6863, %6512 : tensor<3x3x256x256xf32>
      %6865 = stablehlo.convert %6564 : tensor<f32>
      %6866 = stablehlo.broadcast_in_dim %6865, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6867 = stablehlo.multiply %6866, %6515 : tensor<3x3x256x256xf32>
      %6868 = stablehlo.convert %6564 : tensor<f32>
      %6869 = stablehlo.broadcast_in_dim %6868, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6870 = stablehlo.multiply %6869, %6518 : tensor<256xf32>
      %6871 = stablehlo.convert %6564 : tensor<f32>
      %6872 = stablehlo.broadcast_in_dim %6871, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6873 = stablehlo.multiply %6872, %6521 : tensor<256xf32>
      %6874 = stablehlo.convert %6564 : tensor<f32>
      %6875 = stablehlo.broadcast_in_dim %6874, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6876 = stablehlo.multiply %6875, %6524 : tensor<256xf32>
      %6877 = stablehlo.convert %6564 : tensor<f32>
      %6878 = stablehlo.broadcast_in_dim %6877, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %6879 = stablehlo.multiply %6878, %6527 : tensor<256xf32>
      %6880 = stablehlo.convert %6564 : tensor<f32>
      %6881 = stablehlo.broadcast_in_dim %6880, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6882 = stablehlo.multiply %6881, %6530 : tensor<3x3x256x256xf32>
      %6883 = stablehlo.convert %6564 : tensor<f32>
      %6884 = stablehlo.broadcast_in_dim %6883, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %6885 = stablehlo.multiply %6884, %6533 : tensor<3x3x256x256xf32>
      %6886 = stablehlo.convert %6564 : tensor<f32>
      %6887 = stablehlo.broadcast_in_dim %6886, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6888 = stablehlo.multiply %6887, %6536 : tensor<64xf32>
      %6889 = stablehlo.convert %6564 : tensor<f32>
      %6890 = stablehlo.broadcast_in_dim %6889, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %6891 = stablehlo.multiply %6890, %6539 : tensor<64xf32>
      %6892 = stablehlo.convert %6564 : tensor<f32>
      %6893 = stablehlo.broadcast_in_dim %6892, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %6894 = stablehlo.multiply %6893, %6542 : tensor<7x7x3x64xf32>
      %c_1797 = stablehlo.constant dense<2147483647> : tensor<i32>
      %6895 = stablehlo.compare  LT, %arg519, %c_1797,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_1798 = stablehlo.constant dense<1> : tensor<i32>
      %6896 = stablehlo.add %arg519, %c_1798 : tensor<i32>
      %c_1799 = stablehlo.constant dense<2147483647> : tensor<i32>
      %6897 = func.call @_where_550(%6895, %6896, %c_1799) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %6898 = stablehlo.add %arg299, %6567 : tensor<1000xf32>
      %6899 = stablehlo.add %arg300, %6570 : tensor<512x1000xf32>
      %6900 = stablehlo.add %arg301, %6573 : tensor<64xf32>
      %6901 = stablehlo.add %arg302, %6576 : tensor<64xf32>
      %6902 = stablehlo.add %arg303, %6579 : tensor<64xf32>
      %6903 = stablehlo.add %arg304, %6582 : tensor<64xf32>
      %6904 = stablehlo.add %arg305, %6585 : tensor<3x3x64x64xf32>
      %6905 = stablehlo.add %arg306, %6588 : tensor<3x3x64x64xf32>
      %6906 = stablehlo.add %arg307, %6591 : tensor<64xf32>
      %6907 = stablehlo.add %arg308, %6594 : tensor<64xf32>
      %6908 = stablehlo.add %arg309, %6597 : tensor<64xf32>
      %6909 = stablehlo.add %arg310, %6600 : tensor<64xf32>
      %6910 = stablehlo.add %arg311, %6603 : tensor<3x3x64x64xf32>
      %6911 = stablehlo.add %arg312, %6606 : tensor<3x3x64x64xf32>
      %6912 = stablehlo.add %arg313, %6609 : tensor<256xf32>
      %6913 = stablehlo.add %arg314, %6612 : tensor<256xf32>
      %6914 = stablehlo.add %arg315, %6615 : tensor<256xf32>
      %6915 = stablehlo.add %arg316, %6618 : tensor<256xf32>
      %6916 = stablehlo.add %arg317, %6621 : tensor<3x3x256x256xf32>
      %6917 = stablehlo.add %arg318, %6624 : tensor<3x3x256x256xf32>
      %6918 = stablehlo.add %arg319, %6627 : tensor<256xf32>
      %6919 = stablehlo.add %arg320, %6630 : tensor<256xf32>
      %6920 = stablehlo.add %arg321, %6633 : tensor<256xf32>
      %6921 = stablehlo.add %arg322, %6636 : tensor<256xf32>
      %6922 = stablehlo.add %arg323, %6639 : tensor<3x3x256x256xf32>
      %6923 = stablehlo.add %arg324, %6642 : tensor<3x3x256x256xf32>
      %6924 = stablehlo.add %arg325, %6645 : tensor<256xf32>
      %6925 = stablehlo.add %arg326, %6648 : tensor<256xf32>
      %6926 = stablehlo.add %arg327, %6651 : tensor<256xf32>
      %6927 = stablehlo.add %arg328, %6654 : tensor<256xf32>
      %6928 = stablehlo.add %arg329, %6657 : tensor<3x3x256x256xf32>
      %6929 = stablehlo.add %arg330, %6660 : tensor<3x3x256x256xf32>
      %6930 = stablehlo.add %arg331, %6663 : tensor<512xf32>
      %6931 = stablehlo.add %arg332, %6666 : tensor<512xf32>
      %6932 = stablehlo.add %arg333, %6669 : tensor<512xf32>
      %6933 = stablehlo.add %arg334, %6672 : tensor<512xf32>
      %6934 = stablehlo.add %arg335, %6675 : tensor<3x3x256x512xf32>
      %6935 = stablehlo.add %arg336, %6678 : tensor<3x3x512x512xf32>
      %6936 = stablehlo.add %arg337, %6681 : tensor<1x1x256x512xf32>
      %6937 = stablehlo.add %arg338, %6684 : tensor<512xf32>
      %6938 = stablehlo.add %arg339, %6687 : tensor<512xf32>
      %6939 = stablehlo.add %arg340, %6690 : tensor<512xf32>
      %6940 = stablehlo.add %arg341, %6693 : tensor<512xf32>
      %6941 = stablehlo.add %arg342, %6696 : tensor<512xf32>
      %6942 = stablehlo.add %arg343, %6699 : tensor<512xf32>
      %6943 = stablehlo.add %arg344, %6702 : tensor<3x3x512x512xf32>
      %6944 = stablehlo.add %arg345, %6705 : tensor<3x3x512x512xf32>
      %6945 = stablehlo.add %arg346, %6708 : tensor<512xf32>
      %6946 = stablehlo.add %arg347, %6711 : tensor<512xf32>
      %6947 = stablehlo.add %arg348, %6714 : tensor<512xf32>
      %6948 = stablehlo.add %arg349, %6717 : tensor<512xf32>
      %6949 = stablehlo.add %arg350, %6720 : tensor<3x3x512x512xf32>
      %6950 = stablehlo.add %arg351, %6723 : tensor<3x3x512x512xf32>
      %6951 = stablehlo.add %arg352, %6726 : tensor<64xf32>
      %6952 = stablehlo.add %arg353, %6729 : tensor<64xf32>
      %6953 = stablehlo.add %arg354, %6732 : tensor<64xf32>
      %6954 = stablehlo.add %arg355, %6735 : tensor<64xf32>
      %6955 = stablehlo.add %arg356, %6738 : tensor<3x3x64x64xf32>
      %6956 = stablehlo.add %arg357, %6741 : tensor<3x3x64x64xf32>
      %6957 = stablehlo.add %arg358, %6744 : tensor<128xf32>
      %6958 = stablehlo.add %arg359, %6747 : tensor<128xf32>
      %6959 = stablehlo.add %arg360, %6750 : tensor<128xf32>
      %6960 = stablehlo.add %arg361, %6753 : tensor<128xf32>
      %6961 = stablehlo.add %arg362, %6756 : tensor<3x3x64x128xf32>
      %6962 = stablehlo.add %arg363, %6759 : tensor<3x3x128x128xf32>
      %6963 = stablehlo.add %arg364, %6762 : tensor<1x1x64x128xf32>
      %6964 = stablehlo.add %arg365, %6765 : tensor<128xf32>
      %6965 = stablehlo.add %arg366, %6768 : tensor<128xf32>
      %6966 = stablehlo.add %arg367, %6771 : tensor<128xf32>
      %6967 = stablehlo.add %arg368, %6774 : tensor<128xf32>
      %6968 = stablehlo.add %arg369, %6777 : tensor<128xf32>
      %6969 = stablehlo.add %arg370, %6780 : tensor<128xf32>
      %6970 = stablehlo.add %arg371, %6783 : tensor<3x3x128x128xf32>
      %6971 = stablehlo.add %arg372, %6786 : tensor<3x3x128x128xf32>
      %6972 = stablehlo.add %arg373, %6789 : tensor<128xf32>
      %6973 = stablehlo.add %arg374, %6792 : tensor<128xf32>
      %6974 = stablehlo.add %arg375, %6795 : tensor<128xf32>
      %6975 = stablehlo.add %arg376, %6798 : tensor<128xf32>
      %6976 = stablehlo.add %arg377, %6801 : tensor<3x3x128x128xf32>
      %6977 = stablehlo.add %arg378, %6804 : tensor<3x3x128x128xf32>
      %6978 = stablehlo.add %arg379, %6807 : tensor<128xf32>
      %6979 = stablehlo.add %arg380, %6810 : tensor<128xf32>
      %6980 = stablehlo.add %arg381, %6813 : tensor<128xf32>
      %6981 = stablehlo.add %arg382, %6816 : tensor<128xf32>
      %6982 = stablehlo.add %arg383, %6819 : tensor<3x3x128x128xf32>
      %6983 = stablehlo.add %arg384, %6822 : tensor<3x3x128x128xf32>
      %6984 = stablehlo.add %arg385, %6825 : tensor<256xf32>
      %6985 = stablehlo.add %arg386, %6828 : tensor<256xf32>
      %6986 = stablehlo.add %arg387, %6831 : tensor<256xf32>
      %6987 = stablehlo.add %arg388, %6834 : tensor<256xf32>
      %6988 = stablehlo.add %arg389, %6837 : tensor<3x3x128x256xf32>
      %6989 = stablehlo.add %arg390, %6840 : tensor<3x3x256x256xf32>
      %6990 = stablehlo.add %arg391, %6843 : tensor<1x1x128x256xf32>
      %6991 = stablehlo.add %arg392, %6846 : tensor<256xf32>
      %6992 = stablehlo.add %arg393, %6849 : tensor<256xf32>
      %6993 = stablehlo.add %arg394, %6852 : tensor<256xf32>
      %6994 = stablehlo.add %arg395, %6855 : tensor<256xf32>
      %6995 = stablehlo.add %arg396, %6858 : tensor<256xf32>
      %6996 = stablehlo.add %arg397, %6861 : tensor<256xf32>
      %6997 = stablehlo.add %arg398, %6864 : tensor<3x3x256x256xf32>
      %6998 = stablehlo.add %arg399, %6867 : tensor<3x3x256x256xf32>
      %6999 = stablehlo.add %arg400, %6870 : tensor<256xf32>
      %7000 = stablehlo.add %arg401, %6873 : tensor<256xf32>
      %7001 = stablehlo.add %arg402, %6876 : tensor<256xf32>
      %7002 = stablehlo.add %arg403, %6879 : tensor<256xf32>
      %7003 = stablehlo.add %arg404, %6882 : tensor<3x3x256x256xf32>
      %7004 = stablehlo.add %arg405, %6885 : tensor<3x3x256x256xf32>
      %7005 = stablehlo.add %arg406, %6888 : tensor<64xf32>
      %7006 = stablehlo.add %arg407, %6891 : tensor<64xf32>
      %7007 = stablehlo.add %arg408, %6894 : tensor<7x7x3x64xf32>
      %c_1800 = stablehlo.constant dense<1> : tensor<i32>
      %7008 = stablehlo.add %arg298, %c_1800 : tensor<i32>
      %7009 = func.call @_where_552(%5619, %5885, %arg409) : (tensor<i1>, tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
      %7010 = func.call @_where_554(%5619, %5888, %arg410) : (tensor<i1>, tensor<512x1000xf32>, tensor<512x1000xf32>) -> tensor<512x1000xf32>
      %7011 = func.call @_where_556(%5619, %5891, %arg411) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7012 = func.call @_where_556(%5619, %5894, %arg412) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7013 = func.call @_where_556(%5619, %5897, %arg413) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7014 = func.call @_where_556(%5619, %5900, %arg414) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7015 = func.call @_where_558(%5619, %5903, %arg415) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7016 = func.call @_where_558(%5619, %5906, %arg416) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7017 = func.call @_where_556(%5619, %5909, %arg417) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7018 = func.call @_where_556(%5619, %5912, %arg418) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7019 = func.call @_where_556(%5619, %5915, %arg419) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7020 = func.call @_where_556(%5619, %5918, %arg420) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7021 = func.call @_where_558(%5619, %5921, %arg421) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7022 = func.call @_where_558(%5619, %5924, %arg422) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7023 = func.call @_where_560(%5619, %5927, %arg423) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7024 = func.call @_where_560(%5619, %5930, %arg424) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7025 = func.call @_where_560(%5619, %5933, %arg425) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7026 = func.call @_where_560(%5619, %5936, %arg426) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7027 = func.call @_where_562(%5619, %5939, %arg427) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7028 = func.call @_where_562(%5619, %5942, %arg428) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7029 = func.call @_where_560(%5619, %5945, %arg429) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7030 = func.call @_where_560(%5619, %5948, %arg430) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7031 = func.call @_where_560(%5619, %5951, %arg431) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7032 = func.call @_where_560(%5619, %5954, %arg432) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7033 = func.call @_where_562(%5619, %5957, %arg433) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7034 = func.call @_where_562(%5619, %5960, %arg434) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7035 = func.call @_where_560(%5619, %5963, %arg435) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7036 = func.call @_where_560(%5619, %5966, %arg436) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7037 = func.call @_where_560(%5619, %5969, %arg437) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7038 = func.call @_where_560(%5619, %5972, %arg438) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7039 = func.call @_where_562(%5619, %5975, %arg439) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7040 = func.call @_where_562(%5619, %5978, %arg440) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7041 = func.call @_where_564(%5619, %5981, %arg441) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7042 = func.call @_where_564(%5619, %5984, %arg442) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7043 = func.call @_where_564(%5619, %5987, %arg443) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7044 = func.call @_where_564(%5619, %5990, %arg444) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7045 = func.call @_where_566(%5619, %5993, %arg445) : (tensor<i1>, tensor<3x3x256x512xf32>, tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
      %7046 = func.call @_where_568(%5619, %5996, %arg446) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7047 = func.call @_where_570(%5619, %5999, %arg447) : (tensor<i1>, tensor<1x1x256x512xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
      %7048 = func.call @_where_564(%5619, %6002, %arg448) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7049 = func.call @_where_564(%5619, %6005, %arg449) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7050 = func.call @_where_564(%5619, %6008, %arg450) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7051 = func.call @_where_564(%5619, %6011, %arg451) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7052 = func.call @_where_564(%5619, %6014, %arg452) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7053 = func.call @_where_564(%5619, %6017, %arg453) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7054 = func.call @_where_568(%5619, %6020, %arg454) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7055 = func.call @_where_568(%5619, %6023, %arg455) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7056 = func.call @_where_564(%5619, %6026, %arg456) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7057 = func.call @_where_564(%5619, %6029, %arg457) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7058 = func.call @_where_564(%5619, %6032, %arg458) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7059 = func.call @_where_564(%5619, %6035, %arg459) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7060 = func.call @_where_568(%5619, %6038, %arg460) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7061 = func.call @_where_568(%5619, %6041, %arg461) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7062 = func.call @_where_556(%5619, %6044, %arg462) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7063 = func.call @_where_556(%5619, %6047, %arg463) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7064 = func.call @_where_556(%5619, %6050, %arg464) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7065 = func.call @_where_556(%5619, %6053, %arg465) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7066 = func.call @_where_558(%5619, %6056, %arg466) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7067 = func.call @_where_558(%5619, %6059, %arg467) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7068 = func.call @_where_572(%5619, %6062, %arg468) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7069 = func.call @_where_572(%5619, %6065, %arg469) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7070 = func.call @_where_572(%5619, %6068, %arg470) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7071 = func.call @_where_572(%5619, %6071, %arg471) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7072 = func.call @_where_574(%5619, %6074, %arg472) : (tensor<i1>, tensor<3x3x64x128xf32>, tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
      %7073 = func.call @_where_576(%5619, %6077, %arg473) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7074 = func.call @_where_578(%5619, %6080, %arg474) : (tensor<i1>, tensor<1x1x64x128xf32>, tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
      %7075 = func.call @_where_572(%5619, %6083, %arg475) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7076 = func.call @_where_572(%5619, %6086, %arg476) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7077 = func.call @_where_572(%5619, %6089, %arg477) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7078 = func.call @_where_572(%5619, %6092, %arg478) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7079 = func.call @_where_572(%5619, %6095, %arg479) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7080 = func.call @_where_572(%5619, %6098, %arg480) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7081 = func.call @_where_576(%5619, %6101, %arg481) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7082 = func.call @_where_576(%5619, %6104, %arg482) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7083 = func.call @_where_572(%5619, %6107, %arg483) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7084 = func.call @_where_572(%5619, %6110, %arg484) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7085 = func.call @_where_572(%5619, %6113, %arg485) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7086 = func.call @_where_572(%5619, %6116, %arg486) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7087 = func.call @_where_576(%5619, %6119, %arg487) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7088 = func.call @_where_576(%5619, %6122, %arg488) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7089 = func.call @_where_572(%5619, %6125, %arg489) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7090 = func.call @_where_572(%5619, %6128, %arg490) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7091 = func.call @_where_572(%5619, %6131, %arg491) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7092 = func.call @_where_572(%5619, %6134, %arg492) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7093 = func.call @_where_576(%5619, %6137, %arg493) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7094 = func.call @_where_576(%5619, %6140, %arg494) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7095 = func.call @_where_560(%5619, %6143, %arg495) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7096 = func.call @_where_560(%5619, %6146, %arg496) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7097 = func.call @_where_560(%5619, %6149, %arg497) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7098 = func.call @_where_560(%5619, %6152, %arg498) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7099 = func.call @_where_580(%5619, %6155, %arg499) : (tensor<i1>, tensor<3x3x128x256xf32>, tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
      %7100 = func.call @_where_562(%5619, %6158, %arg500) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7101 = func.call @_where_582(%5619, %6161, %arg501) : (tensor<i1>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
      %7102 = func.call @_where_560(%5619, %6164, %arg502) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7103 = func.call @_where_560(%5619, %6167, %arg503) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7104 = func.call @_where_560(%5619, %6170, %arg504) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7105 = func.call @_where_560(%5619, %6173, %arg505) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7106 = func.call @_where_560(%5619, %6176, %arg506) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7107 = func.call @_where_560(%5619, %6179, %arg507) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7108 = func.call @_where_562(%5619, %6182, %arg508) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7109 = func.call @_where_562(%5619, %6185, %arg509) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7110 = func.call @_where_560(%5619, %6188, %arg510) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7111 = func.call @_where_560(%5619, %6191, %arg511) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7112 = func.call @_where_560(%5619, %6194, %arg512) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7113 = func.call @_where_560(%5619, %6197, %arg513) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7114 = func.call @_where_562(%5619, %6200, %arg514) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7115 = func.call @_where_562(%5619, %6203, %arg515) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7116 = func.call @_where_556(%5619, %6206, %arg516) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7117 = func.call @_where_556(%5619, %6209, %arg517) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7118 = func.call @_where_584(%5619, %6212, %arg518) : (tensor<i1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
      %7119 = func.call @_where_550(%5619, %6897, %arg519) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %7120 = func.call @_where_552(%5619, %6898, %arg299) : (tensor<i1>, tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
      %7121 = func.call @_where_554(%5619, %6899, %arg300) : (tensor<i1>, tensor<512x1000xf32>, tensor<512x1000xf32>) -> tensor<512x1000xf32>
      %7122 = func.call @_where_556(%5619, %6900, %arg301) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7123 = func.call @_where_556(%5619, %6901, %arg302) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7124 = func.call @_where_556(%5619, %6902, %arg303) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7125 = func.call @_where_556(%5619, %6903, %arg304) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7126 = func.call @_where_558(%5619, %6904, %arg305) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7127 = func.call @_where_558(%5619, %6905, %arg306) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7128 = func.call @_where_556(%5619, %6906, %arg307) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7129 = func.call @_where_556(%5619, %6907, %arg308) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7130 = func.call @_where_556(%5619, %6908, %arg309) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7131 = func.call @_where_556(%5619, %6909, %arg310) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7132 = func.call @_where_558(%5619, %6910, %arg311) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7133 = func.call @_where_558(%5619, %6911, %arg312) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7134 = func.call @_where_560(%5619, %6912, %arg313) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7135 = func.call @_where_560(%5619, %6913, %arg314) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7136 = func.call @_where_560(%5619, %6914, %arg315) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7137 = func.call @_where_560(%5619, %6915, %arg316) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7138 = func.call @_where_562(%5619, %6916, %arg317) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7139 = func.call @_where_562(%5619, %6917, %arg318) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7140 = func.call @_where_560(%5619, %6918, %arg319) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7141 = func.call @_where_560(%5619, %6919, %arg320) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7142 = func.call @_where_560(%5619, %6920, %arg321) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7143 = func.call @_where_560(%5619, %6921, %arg322) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7144 = func.call @_where_562(%5619, %6922, %arg323) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7145 = func.call @_where_562(%5619, %6923, %arg324) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7146 = func.call @_where_560(%5619, %6924, %arg325) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7147 = func.call @_where_560(%5619, %6925, %arg326) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7148 = func.call @_where_560(%5619, %6926, %arg327) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7149 = func.call @_where_560(%5619, %6927, %arg328) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7150 = func.call @_where_562(%5619, %6928, %arg329) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7151 = func.call @_where_562(%5619, %6929, %arg330) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7152 = func.call @_where_564(%5619, %6930, %arg331) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7153 = func.call @_where_564(%5619, %6931, %arg332) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7154 = func.call @_where_564(%5619, %6932, %arg333) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7155 = func.call @_where_564(%5619, %6933, %arg334) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7156 = func.call @_where_566(%5619, %6934, %arg335) : (tensor<i1>, tensor<3x3x256x512xf32>, tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
      %7157 = func.call @_where_568(%5619, %6935, %arg336) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7158 = func.call @_where_570(%5619, %6936, %arg337) : (tensor<i1>, tensor<1x1x256x512xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
      %7159 = func.call @_where_564(%5619, %6937, %arg338) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7160 = func.call @_where_564(%5619, %6938, %arg339) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7161 = func.call @_where_564(%5619, %6939, %arg340) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7162 = func.call @_where_564(%5619, %6940, %arg341) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7163 = func.call @_where_564(%5619, %6941, %arg342) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7164 = func.call @_where_564(%5619, %6942, %arg343) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7165 = func.call @_where_568(%5619, %6943, %arg344) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7166 = func.call @_where_568(%5619, %6944, %arg345) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7167 = func.call @_where_564(%5619, %6945, %arg346) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7168 = func.call @_where_564(%5619, %6946, %arg347) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7169 = func.call @_where_564(%5619, %6947, %arg348) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7170 = func.call @_where_564(%5619, %6948, %arg349) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %7171 = func.call @_where_568(%5619, %6949, %arg350) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7172 = func.call @_where_568(%5619, %6950, %arg351) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %7173 = func.call @_where_556(%5619, %6951, %arg352) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7174 = func.call @_where_556(%5619, %6952, %arg353) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7175 = func.call @_where_556(%5619, %6953, %arg354) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7176 = func.call @_where_556(%5619, %6954, %arg355) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7177 = func.call @_where_558(%5619, %6955, %arg356) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7178 = func.call @_where_558(%5619, %6956, %arg357) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %7179 = func.call @_where_572(%5619, %6957, %arg358) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7180 = func.call @_where_572(%5619, %6958, %arg359) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7181 = func.call @_where_572(%5619, %6959, %arg360) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7182 = func.call @_where_572(%5619, %6960, %arg361) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7183 = func.call @_where_574(%5619, %6961, %arg362) : (tensor<i1>, tensor<3x3x64x128xf32>, tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
      %7184 = func.call @_where_576(%5619, %6962, %arg363) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7185 = func.call @_where_578(%5619, %6963, %arg364) : (tensor<i1>, tensor<1x1x64x128xf32>, tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
      %7186 = func.call @_where_572(%5619, %6964, %arg365) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7187 = func.call @_where_572(%5619, %6965, %arg366) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7188 = func.call @_where_572(%5619, %6966, %arg367) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7189 = func.call @_where_572(%5619, %6967, %arg368) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7190 = func.call @_where_572(%5619, %6968, %arg369) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7191 = func.call @_where_572(%5619, %6969, %arg370) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7192 = func.call @_where_576(%5619, %6970, %arg371) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7193 = func.call @_where_576(%5619, %6971, %arg372) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7194 = func.call @_where_572(%5619, %6972, %arg373) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7195 = func.call @_where_572(%5619, %6973, %arg374) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7196 = func.call @_where_572(%5619, %6974, %arg375) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7197 = func.call @_where_572(%5619, %6975, %arg376) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7198 = func.call @_where_576(%5619, %6976, %arg377) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7199 = func.call @_where_576(%5619, %6977, %arg378) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7200 = func.call @_where_572(%5619, %6978, %arg379) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7201 = func.call @_where_572(%5619, %6979, %arg380) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7202 = func.call @_where_572(%5619, %6980, %arg381) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7203 = func.call @_where_572(%5619, %6981, %arg382) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7204 = func.call @_where_576(%5619, %6982, %arg383) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7205 = func.call @_where_576(%5619, %6983, %arg384) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %7206 = func.call @_where_560(%5619, %6984, %arg385) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7207 = func.call @_where_560(%5619, %6985, %arg386) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7208 = func.call @_where_560(%5619, %6986, %arg387) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7209 = func.call @_where_560(%5619, %6987, %arg388) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7210 = func.call @_where_580(%5619, %6988, %arg389) : (tensor<i1>, tensor<3x3x128x256xf32>, tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
      %7211 = func.call @_where_562(%5619, %6989, %arg390) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7212 = func.call @_where_582(%5619, %6990, %arg391) : (tensor<i1>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
      %7213 = func.call @_where_560(%5619, %6991, %arg392) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7214 = func.call @_where_560(%5619, %6992, %arg393) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7215 = func.call @_where_560(%5619, %6993, %arg394) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7216 = func.call @_where_560(%5619, %6994, %arg395) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7217 = func.call @_where_560(%5619, %6995, %arg396) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7218 = func.call @_where_560(%5619, %6996, %arg397) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7219 = func.call @_where_562(%5619, %6997, %arg398) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7220 = func.call @_where_562(%5619, %6998, %arg399) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7221 = func.call @_where_560(%5619, %6999, %arg400) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7222 = func.call @_where_560(%5619, %7000, %arg401) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7223 = func.call @_where_560(%5619, %7001, %arg402) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7224 = func.call @_where_560(%5619, %7002, %arg403) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %7225 = func.call @_where_562(%5619, %7003, %arg404) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7226 = func.call @_where_562(%5619, %7004, %arg405) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %7227 = func.call @_where_556(%5619, %7005, %arg406) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7228 = func.call @_where_556(%5619, %7006, %arg407) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %7229 = func.call @_where_584(%5619, %7007, %arg408) : (tensor<i1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
      %7230 = stablehlo.broadcast_in_dim %5665, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %7231 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %7232 = stablehlo.broadcast_in_dim %5666, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %7233 = stablehlo.broadcast_in_dim %5632, dims = [] : (tensor<f32>) -> tensor<1xf32>
      sdy.return %7008, %7120, %7121, %7122, %7123, %7124, %7125, %7126, %7127, %7128, %7129, %7130, %7131, %7132, %7133, %7134, %7135, %7136, %7137, %7138, %7139, %7140, %7141, %7142, %7143, %7144, %7145, %7146, %7147, %7148, %7149, %7150, %7151, %7152, %7153, %7154, %7155, %7156, %7157, %7158, %7159, %7160, %7161, %7162, %7163, %7164, %7165, %7166, %7167, %7168, %7169, %7170, %7171, %7172, %7173, %7174, %7175, %7176, %7177, %7178, %7179, %7180, %7181, %7182, %7183, %7184, %7185, %7186, %7187, %7188, %7189, %7190, %7191, %7192, %7193, %7194, %7195, %7196, %7197, %7198, %7199, %7200, %7201, %7202, %7203, %7204, %7205, %7206, %7207, %7208, %7209, %7210, %7211, %7212, %7213, %7214, %7215, %7216, %7217, %7218, %7219, %7220, %7221, %7222, %7223, %7224, %7225, %7226, %7227, %7228, %7229, %7009, %7010, %7011, %7012, %7013, %7014, %7015, %7016, %7017, %7018, %7019, %7020, %7021, %7022, %7023, %7024, %7025, %7026, %7027, %7028, %7029, %7030, %7031, %7032, %7033, %7034, %7035, %7036, %7037, %7038, %7039, %7040, %7041, %7042, %7043, %7044, %7045, %7046, %7047, %7048, %7049, %7050, %7051, %7052, %7053, %7054, %7055, %7056, %7057, %7058, %7059, %7060, %7061, %7062, %7063, %7064, %7065, %7066, %7067, %7068, %7069, %7070, %7071, %7072, %7073, %7074, %7075, %7076, %7077, %7078, %7079, %7080, %7081, %7082, %7083, %7084, %7085, %7086, %7087, %7088, %7089, %7090, %7091, %7092, %7093, %7094, %7095, %7096, %7097, %7098, %7099, %7100, %7101, %7102, %7103, %7104, %7105, %7106, %7107, %7108, %7109, %7110, %7111, %7112, %7113, %7114, %7115, %7116, %7117, %7118, %7119, %5740, %5742, %5744, %5746, %5748, %5750, %5752, %5754, %5756, %5758, %5760, %5762, %5764, %5766, %5768, %5770, %5772, %5774, %5776, %5778, %5780, %5782, %5784, %5786, %5788, %5790, %5792, %5794, %5796, %5798, %5800, %5802, %5804, %5806, %5808, %5810, %5812, %5814, %5816, %5818, %5820, %5822, %5824, %5826, %5828, %5830, %5832, %5834, %5836, %5838, %5840, %5842, %5844, %5846, %5848, %5850, %5852, %5854, %5856, %5858, %5860, %5862, %5864, %5866, %5868, %5870, %5872, %5874, %5876, %5878, %5880, %5882, %5639, %5632, %7230, %7231, %7232, %7233 : tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
    } : (tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<4x256x224x224x3xf16>, tensor<4x256xi32>) -> (tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8, %0#9, %0#10, %0#11, %0#12, %0#13, %0#14, %0#15, %0#16, %0#17, %0#18, %0#19, %0#20, %0#21, %0#22, %0#23, %0#24, %0#25, %0#26, %0#27, %0#28, %0#29, %0#30, %0#31, %0#32, %0#33, %0#34, %0#35, %0#36, %0#37, %0#38, %0#39, %0#40, %0#41, %0#42, %0#43, %0#44, %0#45, %0#46, %0#47, %0#48, %0#49, %0#50, %0#51, %0#52, %0#53, %0#54, %0#55, %0#56, %0#57, %0#58, %0#59, %0#60, %0#61, %0#62, %0#63, %0#64, %0#65, %0#66, %0#67, %0#68, %0#69, %0#70, %0#71, %0#72, %0#73, %0#74, %0#75, %0#76, %0#77, %0#78, %0#79, %0#80, %0#81, %0#82, %0#83, %0#84, %0#85, %0#86, %0#87, %0#88, %0#89, %0#90, %0#91, %0#92, %0#93, %0#94, %0#95, %0#96, %0#97, %0#98, %0#99, %0#100, %0#101, %0#102, %0#103, %0#104, %0#105, %0#106, %0#107, %0#108, %0#109, %0#110, %0#111, %0#112, %0#113, %0#114, %0#115, %0#116, %0#117, %0#118, %0#119, %0#120, %0#121, %0#122, %0#123, %0#124, %0#125, %0#126, %0#127, %0#128, %0#129, %0#130, %0#131, %0#132, %0#133, %0#134, %0#135, %0#136, %0#137, %0#138, %0#139, %0#140, %0#141, %0#142, %0#143, %0#144, %0#145, %0#146, %0#147, %0#148, %0#149, %0#150, %0#151, %0#152, %0#153, %0#154, %0#155, %0#156, %0#157, %0#158, %0#159, %0#160, %0#161, %0#162, %0#163, %0#164, %0#165, %0#166, %0#167, %0#168, %0#169, %0#170, %0#171, %0#172, %0#173, %0#174, %0#175, %0#176, %0#177, %0#178, %0#179, %0#180, %0#181, %0#182, %0#183, %0#184, %0#185, %0#186, %0#187, %0#188, %0#189, %0#190, %0#191, %0#192, %0#193, %0#194, %0#195, %0#196, %0#197, %0#198, %0#199, %0#200, %0#201, %0#202, %0#203, %0#204, %0#205, %0#206, %0#207, %0#208, %0#209, %0#210, %0#211, %0#212, %0#213, %0#214, %0#215, %0#216, %0#217, %0#218, %0#219, %0#220, %0#221, %0#222, %0#223, %0#224, %0#225, %0#226, %0#227, %0#228, %0#229, %0#230, %0#231, %0#232, %0#233, %0#234, %0#235, %0#236, %0#237, %0#238, %0#239, %0#240, %0#241, %0#242, %0#243, %0#244, %0#245, %0#246, %0#247, %0#248, %0#249, %0#250, %0#251, %0#252, %0#253, %0#254, %0#255, %0#256, %0#257, %0#258, %0#259, %0#260, %0#261, %0#262, %0#263, %0#264, %0#265, %0#266, %0#267, %0#268, %0#269, %0#270, %0#271, %0#272, %0#273, %0#274, %0#275, %0#276, %0#277, %0#278, %0#279, %0#280, %0#281, %0#282, %0#283, %0#284, %0#285, %0#286, %0#287, %0#288, %0#289, %0#290, %0#291, %0#292, %0#293, %0#294, %0#295, %0#296, %0#297, %0#298, %0#299 : tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
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
  func.func private @relu(%arg0: tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x112x112x64xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x112x112x64xf16>
    return %1 : tensor<256x112x112x64xf16>
  }
  func.func private @relu_44(%arg0: tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x64xf16>
    return %1 : tensor<256x56x56x64xf16>
  }
  func.func private @relu_90(%arg0: tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xf16>
    return %1 : tensor<256x28x28x128xf16>
  }
  func.func private @relu_142(%arg0: tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xf16>
    return %1 : tensor<256x14x14x256xf16>
  }
  func.func private @relu_198(%arg0: tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xf16>
    return %1 : tensor<256x7x7x512xf16>
  }
  func.func private @log_softmax(%arg0: tensor<256x1000xf16>) -> (tensor<256x1000xf16>, tensor<256x1xf16>) {
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
    return %6, %10 : tensor<256x1000xf16>, tensor<256x1xf16>
  }
  func.func private @log_softmax_285(%arg0: tensor<256x1000xf16>, %arg1: tensor<256x1xf16>, %arg2: tensor<256x1000xf16>) -> tensor<256x1000xf16> {
    %0 = stablehlo.negate %arg2 : tensor<256x1000xf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf16>, tensor<f16>) -> tensor<256xf16>
    %2 = stablehlo.reshape %1 : (tensor<256xf16>) -> tensor<256x1xf16>
    %3 = stablehlo.divide %2, %arg1 : tensor<256x1xf16>
    %4 = stablehlo.convert %3 : (tensor<256x1xf16>) -> tensor<256x1xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<256x1xf32>, tensor<f32>) -> tensor<256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<256xf32>) -> tensor<256x1000xf32>
    %7 = stablehlo.convert %6 : (tensor<256x1000xf32>) -> tensor<256x1000xf16>
    %8 = stablehlo.multiply %7, %arg0 : tensor<256x1000xf16>
    %9 = stablehlo.add %arg2, %8 : tensor<256x1000xf16>
    return %9 : tensor<256x1000xf16>
  }
  func.func private @_where_493(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_496(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_498(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @log_softmax_500(%arg0: tensor<256x1000xf16>) -> tensor<256x1000xf16> {
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
  func.func private @clip_542(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %3 = stablehlo.minimum %arg2, %2 : tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @_where_550(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @_where_552(%arg0: tensor<i1>, %arg1: tensor<1000xf32>, %arg2: tensor<1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
  func.func private @_where_554(%arg0: tensor<i1>, %arg1: tensor<512x1000xf32>, %arg2: tensor<512x1000xf32>) -> tensor<512x1000xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<512x1000xf32>
    return %0 : tensor<512x1000xf32>
  }
  func.func private @_where_556(%arg0: tensor<i1>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @_where_558(%arg0: tensor<i1>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x64x64xf32>
    return %0 : tensor<3x3x64x64xf32>
  }
  func.func private @_where_560(%arg0: tensor<i1>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  func.func private @_where_562(%arg0: tensor<i1>, %arg1: tensor<3x3x256x256xf32>, %arg2: tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x256x256xf32>
    return %0 : tensor<3x3x256x256xf32>
  }
  func.func private @_where_564(%arg0: tensor<i1>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @_where_566(%arg0: tensor<i1>, %arg1: tensor<3x3x256x512xf32>, %arg2: tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x256x512xf32>
    return %0 : tensor<3x3x256x512xf32>
  }
  func.func private @_where_568(%arg0: tensor<i1>, %arg1: tensor<3x3x512x512xf32>, %arg2: tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x512x512xf32>
    return %0 : tensor<3x3x512x512xf32>
  }
  func.func private @_where_570(%arg0: tensor<i1>, %arg1: tensor<1x1x256x512xf32>, %arg2: tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x256x512xf32>
    return %0 : tensor<1x1x256x512xf32>
  }
  func.func private @_where_572(%arg0: tensor<i1>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  func.func private @_where_574(%arg0: tensor<i1>, %arg1: tensor<3x3x64x128xf32>, %arg2: tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x64x128xf32>
    return %0 : tensor<3x3x64x128xf32>
  }
  func.func private @_where_576(%arg0: tensor<i1>, %arg1: tensor<3x3x128x128xf32>, %arg2: tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x128x128xf32>
    return %0 : tensor<3x3x128x128xf32>
  }
  func.func private @_where_578(%arg0: tensor<i1>, %arg1: tensor<1x1x64x128xf32>, %arg2: tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x64x128xf32>
    return %0 : tensor<1x1x64x128xf32>
  }
  func.func private @_where_580(%arg0: tensor<i1>, %arg1: tensor<3x3x128x256xf32>, %arg2: tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x128x256xf32>
    return %0 : tensor<3x3x128x256xf32>
  }
  func.func private @_where_582(%arg0: tensor<i1>, %arg1: tensor<1x1x128x256xf32>, %arg2: tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x128x256xf32>
    return %0 : tensor<1x1x128x256xf32>
  }
  func.func private @_where_584(%arg0: tensor<i1>, %arg1: tensor<7x7x3x64xf32>, %arg2: tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<7x7x3x64xf32>
    return %0 : tensor<7x7x3x64xf32>
  }
}
