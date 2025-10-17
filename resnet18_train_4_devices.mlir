module @jit_call_wrapped attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  sdy.mesh @mesh = <["batch"=4]>
  func.func public @main(%arg0: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg1: tensor<1000xf32>, %arg2: tensor<512x1000xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<3x3x64x64xf32>, %arg8: tensor<3x3x64x64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64xf32>, %arg13: tensor<3x3x64x64xf32>, %arg14: tensor<3x3x64x64xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<3x3x64x128xf32>, %arg20: tensor<3x3x128x128xf32>, %arg21: tensor<1x1x64x128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128xf32>, %arg28: tensor<3x3x128x128xf32>, %arg29: tensor<3x3x128x128xf32>, %arg30: tensor<256xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256xf32>, %arg34: tensor<3x3x128x256xf32>, %arg35: tensor<3x3x256x256xf32>, %arg36: tensor<1x1x128x256xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256xf32>, %arg43: tensor<3x3x256x256xf32>, %arg44: tensor<3x3x256x256xf32>, %arg45: tensor<512xf32>, %arg46: tensor<512xf32>, %arg47: tensor<512xf32>, %arg48: tensor<512xf32>, %arg49: tensor<3x3x256x512xf32>, %arg50: tensor<3x3x512x512xf32>, %arg51: tensor<1x1x256x512xf32>, %arg52: tensor<512xf32>, %arg53: tensor<512xf32>, %arg54: tensor<512xf32>, %arg55: tensor<512xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512xf32>, %arg58: tensor<3x3x512x512xf32>, %arg59: tensor<3x3x512x512xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64xf32>, %arg62: tensor<7x7x3x64xf32>, %arg63: tensor<1000xf32>, %arg64: tensor<512x1000xf32>, %arg65: tensor<64xf32>, %arg66: tensor<64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<64xf32>, %arg69: tensor<3x3x64x64xf32>, %arg70: tensor<3x3x64x64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<64xf32>, %arg75: tensor<3x3x64x64xf32>, %arg76: tensor<3x3x64x64xf32>, %arg77: tensor<128xf32>, %arg78: tensor<128xf32>, %arg79: tensor<128xf32>, %arg80: tensor<128xf32>, %arg81: tensor<3x3x64x128xf32>, %arg82: tensor<3x3x128x128xf32>, %arg83: tensor<1x1x64x128xf32>, %arg84: tensor<128xf32>, %arg85: tensor<128xf32>, %arg86: tensor<128xf32>, %arg87: tensor<128xf32>, %arg88: tensor<128xf32>, %arg89: tensor<128xf32>, %arg90: tensor<3x3x128x128xf32>, %arg91: tensor<3x3x128x128xf32>, %arg92: tensor<256xf32>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256xf32>, %arg96: tensor<3x3x128x256xf32>, %arg97: tensor<3x3x256x256xf32>, %arg98: tensor<1x1x128x256xf32>, %arg99: tensor<256xf32>, %arg100: tensor<256xf32>, %arg101: tensor<256xf32>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256xf32>, %arg105: tensor<3x3x256x256xf32>, %arg106: tensor<3x3x256x256xf32>, %arg107: tensor<512xf32>, %arg108: tensor<512xf32>, %arg109: tensor<512xf32>, %arg110: tensor<512xf32>, %arg111: tensor<3x3x256x512xf32>, %arg112: tensor<3x3x512x512xf32>, %arg113: tensor<1x1x256x512xf32>, %arg114: tensor<512xf32>, %arg115: tensor<512xf32>, %arg116: tensor<512xf32>, %arg117: tensor<512xf32>, %arg118: tensor<512xf32>, %arg119: tensor<512xf32>, %arg120: tensor<3x3x512x512xf32>, %arg121: tensor<3x3x512x512xf32>, %arg122: tensor<64xf32>, %arg123: tensor<64xf32>, %arg124: tensor<7x7x3x64xf32>, %arg125: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg126: tensor<64xf32>, %arg127: tensor<64xf32>, %arg128: tensor<64xf32>, %arg129: tensor<64xf32>, %arg130: tensor<64xf32>, %arg131: tensor<64xf32>, %arg132: tensor<64xf32>, %arg133: tensor<64xf32>, %arg134: tensor<128xf32>, %arg135: tensor<128xf32>, %arg136: tensor<128xf32>, %arg137: tensor<128xf32>, %arg138: tensor<128xf32>, %arg139: tensor<128xf32>, %arg140: tensor<128xf32>, %arg141: tensor<128xf32>, %arg142: tensor<128xf32>, %arg143: tensor<128xf32>, %arg144: tensor<256xf32>, %arg145: tensor<256xf32>, %arg146: tensor<256xf32>, %arg147: tensor<256xf32>, %arg148: tensor<256xf32>, %arg149: tensor<256xf32>, %arg150: tensor<256xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256xf32>, %arg153: tensor<256xf32>, %arg154: tensor<512xf32>, %arg155: tensor<512xf32>, %arg156: tensor<512xf32>, %arg157: tensor<512xf32>, %arg158: tensor<512xf32>, %arg159: tensor<512xf32>, %arg160: tensor<512xf32>, %arg161: tensor<512xf32>, %arg162: tensor<512xf32>, %arg163: tensor<512xf32>, %arg164: tensor<64xf32>, %arg165: tensor<64xf32>, %arg166: tensor<i32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg167: tensor<f32> {sdy.sharding = #sdy.sharding<@empty_mesh, []>}, %arg168: tensor<4x256x224x224x3xf16>, %arg169: tensor<4x256xi32>) -> (tensor<i32> {jax.result_info = "result[0]"}, tensor<1000xf32> {jax.result_info = "result[1]"}, tensor<512x1000xf32> {jax.result_info = "result[2]"}, tensor<64xf32> {jax.result_info = "result[3]"}, tensor<64xf32> {jax.result_info = "result[4]"}, tensor<64xf32> {jax.result_info = "result[5]"}, tensor<64xf32> {jax.result_info = "result[6]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[7]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[8]"}, tensor<64xf32> {jax.result_info = "result[9]"}, tensor<64xf32> {jax.result_info = "result[10]"}, tensor<64xf32> {jax.result_info = "result[11]"}, tensor<64xf32> {jax.result_info = "result[12]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[13]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[14]"}, tensor<128xf32> {jax.result_info = "result[15]"}, tensor<128xf32> {jax.result_info = "result[16]"}, tensor<128xf32> {jax.result_info = "result[17]"}, tensor<128xf32> {jax.result_info = "result[18]"}, tensor<3x3x64x128xf32> {jax.result_info = "result[19]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[20]"}, tensor<1x1x64x128xf32> {jax.result_info = "result[21]"}, tensor<128xf32> {jax.result_info = "result[22]"}, tensor<128xf32> {jax.result_info = "result[23]"}, tensor<128xf32> {jax.result_info = "result[24]"}, tensor<128xf32> {jax.result_info = "result[25]"}, tensor<128xf32> {jax.result_info = "result[26]"}, tensor<128xf32> {jax.result_info = "result[27]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[28]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[29]"}, tensor<256xf32> {jax.result_info = "result[30]"}, tensor<256xf32> {jax.result_info = "result[31]"}, tensor<256xf32> {jax.result_info = "result[32]"}, tensor<256xf32> {jax.result_info = "result[33]"}, tensor<3x3x128x256xf32> {jax.result_info = "result[34]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[35]"}, tensor<1x1x128x256xf32> {jax.result_info = "result[36]"}, tensor<256xf32> {jax.result_info = "result[37]"}, tensor<256xf32> {jax.result_info = "result[38]"}, tensor<256xf32> {jax.result_info = "result[39]"}, tensor<256xf32> {jax.result_info = "result[40]"}, tensor<256xf32> {jax.result_info = "result[41]"}, tensor<256xf32> {jax.result_info = "result[42]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[43]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[44]"}, tensor<512xf32> {jax.result_info = "result[45]"}, tensor<512xf32> {jax.result_info = "result[46]"}, tensor<512xf32> {jax.result_info = "result[47]"}, tensor<512xf32> {jax.result_info = "result[48]"}, tensor<3x3x256x512xf32> {jax.result_info = "result[49]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[50]"}, tensor<1x1x256x512xf32> {jax.result_info = "result[51]"}, tensor<512xf32> {jax.result_info = "result[52]"}, tensor<512xf32> {jax.result_info = "result[53]"}, tensor<512xf32> {jax.result_info = "result[54]"}, tensor<512xf32> {jax.result_info = "result[55]"}, tensor<512xf32> {jax.result_info = "result[56]"}, tensor<512xf32> {jax.result_info = "result[57]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[58]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[59]"}, tensor<64xf32> {jax.result_info = "result[60]"}, tensor<64xf32> {jax.result_info = "result[61]"}, tensor<7x7x3x64xf32> {jax.result_info = "result[62]"}, tensor<1000xf32> {jax.result_info = "result[63]"}, tensor<512x1000xf32> {jax.result_info = "result[64]"}, tensor<64xf32> {jax.result_info = "result[65]"}, tensor<64xf32> {jax.result_info = "result[66]"}, tensor<64xf32> {jax.result_info = "result[67]"}, tensor<64xf32> {jax.result_info = "result[68]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[69]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[70]"}, tensor<64xf32> {jax.result_info = "result[71]"}, tensor<64xf32> {jax.result_info = "result[72]"}, tensor<64xf32> {jax.result_info = "result[73]"}, tensor<64xf32> {jax.result_info = "result[74]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[75]"}, tensor<3x3x64x64xf32> {jax.result_info = "result[76]"}, tensor<128xf32> {jax.result_info = "result[77]"}, tensor<128xf32> {jax.result_info = "result[78]"}, tensor<128xf32> {jax.result_info = "result[79]"}, tensor<128xf32> {jax.result_info = "result[80]"}, tensor<3x3x64x128xf32> {jax.result_info = "result[81]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[82]"}, tensor<1x1x64x128xf32> {jax.result_info = "result[83]"}, tensor<128xf32> {jax.result_info = "result[84]"}, tensor<128xf32> {jax.result_info = "result[85]"}, tensor<128xf32> {jax.result_info = "result[86]"}, tensor<128xf32> {jax.result_info = "result[87]"}, tensor<128xf32> {jax.result_info = "result[88]"}, tensor<128xf32> {jax.result_info = "result[89]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[90]"}, tensor<3x3x128x128xf32> {jax.result_info = "result[91]"}, tensor<256xf32> {jax.result_info = "result[92]"}, tensor<256xf32> {jax.result_info = "result[93]"}, tensor<256xf32> {jax.result_info = "result[94]"}, tensor<256xf32> {jax.result_info = "result[95]"}, tensor<3x3x128x256xf32> {jax.result_info = "result[96]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[97]"}, tensor<1x1x128x256xf32> {jax.result_info = "result[98]"}, tensor<256xf32> {jax.result_info = "result[99]"}, tensor<256xf32> {jax.result_info = "result[100]"}, tensor<256xf32> {jax.result_info = "result[101]"}, tensor<256xf32> {jax.result_info = "result[102]"}, tensor<256xf32> {jax.result_info = "result[103]"}, tensor<256xf32> {jax.result_info = "result[104]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[105]"}, tensor<3x3x256x256xf32> {jax.result_info = "result[106]"}, tensor<512xf32> {jax.result_info = "result[107]"}, tensor<512xf32> {jax.result_info = "result[108]"}, tensor<512xf32> {jax.result_info = "result[109]"}, tensor<512xf32> {jax.result_info = "result[110]"}, tensor<3x3x256x512xf32> {jax.result_info = "result[111]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[112]"}, tensor<1x1x256x512xf32> {jax.result_info = "result[113]"}, tensor<512xf32> {jax.result_info = "result[114]"}, tensor<512xf32> {jax.result_info = "result[115]"}, tensor<512xf32> {jax.result_info = "result[116]"}, tensor<512xf32> {jax.result_info = "result[117]"}, tensor<512xf32> {jax.result_info = "result[118]"}, tensor<512xf32> {jax.result_info = "result[119]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[120]"}, tensor<3x3x512x512xf32> {jax.result_info = "result[121]"}, tensor<64xf32> {jax.result_info = "result[122]"}, tensor<64xf32> {jax.result_info = "result[123]"}, tensor<7x7x3x64xf32> {jax.result_info = "result[124]"}, tensor<i32> {jax.result_info = "result[125]"}, tensor<64xf32> {jax.result_info = "result[126]"}, tensor<64xf32> {jax.result_info = "result[127]"}, tensor<64xf32> {jax.result_info = "result[128]"}, tensor<64xf32> {jax.result_info = "result[129]"}, tensor<64xf32> {jax.result_info = "result[130]"}, tensor<64xf32> {jax.result_info = "result[131]"}, tensor<64xf32> {jax.result_info = "result[132]"}, tensor<64xf32> {jax.result_info = "result[133]"}, tensor<128xf32> {jax.result_info = "result[134]"}, tensor<128xf32> {jax.result_info = "result[135]"}, tensor<128xf32> {jax.result_info = "result[136]"}, tensor<128xf32> {jax.result_info = "result[137]"}, tensor<128xf32> {jax.result_info = "result[138]"}, tensor<128xf32> {jax.result_info = "result[139]"}, tensor<128xf32> {jax.result_info = "result[140]"}, tensor<128xf32> {jax.result_info = "result[141]"}, tensor<128xf32> {jax.result_info = "result[142]"}, tensor<128xf32> {jax.result_info = "result[143]"}, tensor<256xf32> {jax.result_info = "result[144]"}, tensor<256xf32> {jax.result_info = "result[145]"}, tensor<256xf32> {jax.result_info = "result[146]"}, tensor<256xf32> {jax.result_info = "result[147]"}, tensor<256xf32> {jax.result_info = "result[148]"}, tensor<256xf32> {jax.result_info = "result[149]"}, tensor<256xf32> {jax.result_info = "result[150]"}, tensor<256xf32> {jax.result_info = "result[151]"}, tensor<256xf32> {jax.result_info = "result[152]"}, tensor<256xf32> {jax.result_info = "result[153]"}, tensor<512xf32> {jax.result_info = "result[154]"}, tensor<512xf32> {jax.result_info = "result[155]"}, tensor<512xf32> {jax.result_info = "result[156]"}, tensor<512xf32> {jax.result_info = "result[157]"}, tensor<512xf32> {jax.result_info = "result[158]"}, tensor<512xf32> {jax.result_info = "result[159]"}, tensor<512xf32> {jax.result_info = "result[160]"}, tensor<512xf32> {jax.result_info = "result[161]"}, tensor<512xf32> {jax.result_info = "result[162]"}, tensor<512xf32> {jax.result_info = "result[163]"}, tensor<64xf32> {jax.result_info = "result[164]"}, tensor<64xf32> {jax.result_info = "result[165]"}, tensor<i32> {jax.result_info = "result[166]"}, tensor<f32> {jax.result_info = "result[167]"}, tensor<4xf32> {jax.result_info = "result[168]"}, tensor<4xf32> {jax.result_info = "result[169]"}, tensor<4xf32> {jax.result_info = "result[170]"}, tensor<4xf32> {jax.result_info = "result[171]"}) {
    %0:172 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169) in_shardings=[<@mesh, []>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, []>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, []>, <@mesh, [{"batch"}, {}, {}, {}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, []>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, []>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, []>, <@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>] manual_axes={"batch"} (%arg170: tensor<i32>, %arg171: tensor<1000xf32>, %arg172: tensor<512x1000xf32>, %arg173: tensor<64xf32>, %arg174: tensor<64xf32>, %arg175: tensor<64xf32>, %arg176: tensor<64xf32>, %arg177: tensor<3x3x64x64xf32>, %arg178: tensor<3x3x64x64xf32>, %arg179: tensor<64xf32>, %arg180: tensor<64xf32>, %arg181: tensor<64xf32>, %arg182: tensor<64xf32>, %arg183: tensor<3x3x64x64xf32>, %arg184: tensor<3x3x64x64xf32>, %arg185: tensor<128xf32>, %arg186: tensor<128xf32>, %arg187: tensor<128xf32>, %arg188: tensor<128xf32>, %arg189: tensor<3x3x64x128xf32>, %arg190: tensor<3x3x128x128xf32>, %arg191: tensor<1x1x64x128xf32>, %arg192: tensor<128xf32>, %arg193: tensor<128xf32>, %arg194: tensor<128xf32>, %arg195: tensor<128xf32>, %arg196: tensor<128xf32>, %arg197: tensor<128xf32>, %arg198: tensor<3x3x128x128xf32>, %arg199: tensor<3x3x128x128xf32>, %arg200: tensor<256xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256xf32>, %arg203: tensor<256xf32>, %arg204: tensor<3x3x128x256xf32>, %arg205: tensor<3x3x256x256xf32>, %arg206: tensor<1x1x128x256xf32>, %arg207: tensor<256xf32>, %arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<3x3x256x256xf32>, %arg214: tensor<3x3x256x256xf32>, %arg215: tensor<512xf32>, %arg216: tensor<512xf32>, %arg217: tensor<512xf32>, %arg218: tensor<512xf32>, %arg219: tensor<3x3x256x512xf32>, %arg220: tensor<3x3x512x512xf32>, %arg221: tensor<1x1x256x512xf32>, %arg222: tensor<512xf32>, %arg223: tensor<512xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<512xf32>, %arg228: tensor<3x3x512x512xf32>, %arg229: tensor<3x3x512x512xf32>, %arg230: tensor<64xf32>, %arg231: tensor<64xf32>, %arg232: tensor<7x7x3x64xf32>, %arg233: tensor<1000xf32>, %arg234: tensor<512x1000xf32>, %arg235: tensor<64xf32>, %arg236: tensor<64xf32>, %arg237: tensor<64xf32>, %arg238: tensor<64xf32>, %arg239: tensor<3x3x64x64xf32>, %arg240: tensor<3x3x64x64xf32>, %arg241: tensor<64xf32>, %arg242: tensor<64xf32>, %arg243: tensor<64xf32>, %arg244: tensor<64xf32>, %arg245: tensor<3x3x64x64xf32>, %arg246: tensor<3x3x64x64xf32>, %arg247: tensor<128xf32>, %arg248: tensor<128xf32>, %arg249: tensor<128xf32>, %arg250: tensor<128xf32>, %arg251: tensor<3x3x64x128xf32>, %arg252: tensor<3x3x128x128xf32>, %arg253: tensor<1x1x64x128xf32>, %arg254: tensor<128xf32>, %arg255: tensor<128xf32>, %arg256: tensor<128xf32>, %arg257: tensor<128xf32>, %arg258: tensor<128xf32>, %arg259: tensor<128xf32>, %arg260: tensor<3x3x128x128xf32>, %arg261: tensor<3x3x128x128xf32>, %arg262: tensor<256xf32>, %arg263: tensor<256xf32>, %arg264: tensor<256xf32>, %arg265: tensor<256xf32>, %arg266: tensor<3x3x128x256xf32>, %arg267: tensor<3x3x256x256xf32>, %arg268: tensor<1x1x128x256xf32>, %arg269: tensor<256xf32>, %arg270: tensor<256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<256xf32>, %arg275: tensor<3x3x256x256xf32>, %arg276: tensor<3x3x256x256xf32>, %arg277: tensor<512xf32>, %arg278: tensor<512xf32>, %arg279: tensor<512xf32>, %arg280: tensor<512xf32>, %arg281: tensor<3x3x256x512xf32>, %arg282: tensor<3x3x512x512xf32>, %arg283: tensor<1x1x256x512xf32>, %arg284: tensor<512xf32>, %arg285: tensor<512xf32>, %arg286: tensor<512xf32>, %arg287: tensor<512xf32>, %arg288: tensor<512xf32>, %arg289: tensor<512xf32>, %arg290: tensor<3x3x512x512xf32>, %arg291: tensor<3x3x512x512xf32>, %arg292: tensor<64xf32>, %arg293: tensor<64xf32>, %arg294: tensor<7x7x3x64xf32>, %arg295: tensor<i32>, %arg296: tensor<64xf32>, %arg297: tensor<64xf32>, %arg298: tensor<64xf32>, %arg299: tensor<64xf32>, %arg300: tensor<64xf32>, %arg301: tensor<64xf32>, %arg302: tensor<64xf32>, %arg303: tensor<64xf32>, %arg304: tensor<128xf32>, %arg305: tensor<128xf32>, %arg306: tensor<128xf32>, %arg307: tensor<128xf32>, %arg308: tensor<128xf32>, %arg309: tensor<128xf32>, %arg310: tensor<128xf32>, %arg311: tensor<128xf32>, %arg312: tensor<128xf32>, %arg313: tensor<128xf32>, %arg314: tensor<256xf32>, %arg315: tensor<256xf32>, %arg316: tensor<256xf32>, %arg317: tensor<256xf32>, %arg318: tensor<256xf32>, %arg319: tensor<256xf32>, %arg320: tensor<256xf32>, %arg321: tensor<256xf32>, %arg322: tensor<256xf32>, %arg323: tensor<256xf32>, %arg324: tensor<512xf32>, %arg325: tensor<512xf32>, %arg326: tensor<512xf32>, %arg327: tensor<512xf32>, %arg328: tensor<512xf32>, %arg329: tensor<512xf32>, %arg330: tensor<512xf32>, %arg331: tensor<512xf32>, %arg332: tensor<512xf32>, %arg333: tensor<512xf32>, %arg334: tensor<64xf32>, %arg335: tensor<64xf32>, %arg336: tensor<i32>, %arg337: tensor<f32>, %arg338: tensor<1x256x224x224x3xf16>, %arg339: tensor<1x256xi32>) {
      %1 = stablehlo.reshape %arg338 : (tensor<1x256x224x224x3xf16>) -> tensor<256x224x224x3xf16>
      %2 = stablehlo.reshape %arg339 : (tensor<1x256xi32>) -> tensor<256xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %3 = stablehlo.subtract %arg170, %c : tensor<i32>
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
      %9 = stablehlo.convert %arg170 : (tensor<i32>) -> tensor<f32>
      %cst_5 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %10 = stablehlo.compare  LT, %9, %cst_5,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %11 = stablehlo.convert %arg170 : (tensor<i32>) -> tensor<f32>
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
      %24 = stablehlo.convert %arg232 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf16>
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
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
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
      %63 = stablehlo.multiply %62, %arg334 : tensor<64xf32>
      %cst_30 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %64 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %65 = stablehlo.multiply %64, %43 : tensor<64xf32>
      %66 = stablehlo.add %63, %65 : tensor<64xf32>
      %cst_31 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %67 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %68 = stablehlo.multiply %67, %arg335 : tensor<64xf32>
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
      %83 = stablehlo.reshape %arg231 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %84 = stablehlo.multiply %79, %83 : tensor<1x1x1x64xf32>
      %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %86 = stablehlo.multiply %76, %85 : tensor<256x112x112x64xf32>
      %87 = stablehlo.reshape %arg230 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
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
      ^bb0(%arg340: tensor<f16>, %arg341: tensor<f16>):
        %4098 = stablehlo.maximum %arg340, %arg341 : tensor<f16>
        stablehlo.return %4098 : tensor<f16>
      }) : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x56x56x64xf16>
      %96 = stablehlo.convert %arg177 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
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
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
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
      %135 = stablehlo.multiply %134, %arg296 : tensor<64xf32>
      %cst_51 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %136 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %137 = stablehlo.multiply %136, %115 : tensor<64xf32>
      %138 = stablehlo.add %135, %137 : tensor<64xf32>
      %cst_52 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %139 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %140 = stablehlo.multiply %139, %arg297 : tensor<64xf32>
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
      %155 = stablehlo.reshape %arg174 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %156 = stablehlo.multiply %151, %155 : tensor<1x1x1x64xf32>
      %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %158 = stablehlo.multiply %148, %157 : tensor<256x56x56x64xf32>
      %159 = stablehlo.reshape %arg173 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %160 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %161 = stablehlo.add %158, %160 : tensor<256x56x56x64xf32>
      %162 = stablehlo.convert %161 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %163 = func.call @relu_44(%162) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %164 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %165 = stablehlo.compare  GT, %162, %164,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %166 = stablehlo.convert %arg178 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
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
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
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
      %205 = stablehlo.multiply %204, %arg298 : tensor<64xf32>
      %cst_71 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %206 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %207 = stablehlo.multiply %206, %185 : tensor<64xf32>
      %208 = stablehlo.add %205, %207 : tensor<64xf32>
      %cst_72 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %209 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %210 = stablehlo.multiply %209, %arg299 : tensor<64xf32>
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
      %225 = stablehlo.reshape %arg176 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %226 = stablehlo.multiply %221, %225 : tensor<1x1x1x64xf32>
      %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %228 = stablehlo.multiply %218, %227 : tensor<256x56x56x64xf32>
      %229 = stablehlo.reshape %arg175 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %231 = stablehlo.add %228, %230 : tensor<256x56x56x64xf32>
      %232 = stablehlo.convert %231 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %233 = stablehlo.add %95, %232 : tensor<256x56x56x64xf16>
      %234 = func.call @relu_44(%233) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %235 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %236 = stablehlo.compare  GT, %233, %235,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %237 = stablehlo.convert %arg183 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
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
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
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
      %276 = stablehlo.multiply %275, %arg300 : tensor<64xf32>
      %cst_91 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %277 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %278 = stablehlo.multiply %277, %256 : tensor<64xf32>
      %279 = stablehlo.add %276, %278 : tensor<64xf32>
      %cst_92 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %280 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %281 = stablehlo.multiply %280, %arg301 : tensor<64xf32>
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
      %296 = stablehlo.reshape %arg180 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %297 = stablehlo.multiply %292, %296 : tensor<1x1x1x64xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %299 = stablehlo.multiply %289, %298 : tensor<256x56x56x64xf32>
      %300 = stablehlo.reshape %arg179 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %302 = stablehlo.add %299, %301 : tensor<256x56x56x64xf32>
      %303 = stablehlo.convert %302 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %304 = func.call @relu_44(%303) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %305 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %306 = stablehlo.compare  GT, %303, %305,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %307 = stablehlo.convert %arg184 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
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
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
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
      %346 = stablehlo.multiply %345, %arg302 : tensor<64xf32>
      %cst_111 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %348 = stablehlo.multiply %347, %326 : tensor<64xf32>
      %349 = stablehlo.add %346, %348 : tensor<64xf32>
      %cst_112 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %350 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %351 = stablehlo.multiply %350, %arg303 : tensor<64xf32>
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
      %366 = stablehlo.reshape %arg182 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %367 = stablehlo.multiply %362, %366 : tensor<1x1x1x64xf32>
      %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %369 = stablehlo.multiply %359, %368 : tensor<256x56x56x64xf32>
      %370 = stablehlo.reshape %arg181 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %371 = stablehlo.broadcast_in_dim %370, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %372 = stablehlo.add %369, %371 : tensor<256x56x56x64xf32>
      %373 = stablehlo.convert %372 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %374 = stablehlo.add %234, %373 : tensor<256x56x56x64xf16>
      %375 = func.call @relu_44(%374) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %376 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %377 = stablehlo.compare  GT, %374, %376,  FLOAT : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xi1>
      %378 = stablehlo.convert %arg189 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf16>
      %379 = stablehlo.convolution(%375, %378) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x128xf16>) -> tensor<256x28x28x128xf16>
      %380 = stablehlo.convert %379 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %381 = stablehlo.multiply %380, %380 : tensor<256x28x28x128xf32>
      %cst_117 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_117, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %383 = stablehlo.multiply %382, %380 : tensor<256x28x28x128xf32>
      %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%380 init: %cst_118) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_119 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %386 = stablehlo.divide %384, %385 : tensor<128xf32>
      %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%381 init: %cst_120) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_121 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_121, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %389 = stablehlo.divide %387, %388 : tensor<128xf32>
      %390 = stablehlo.broadcast_in_dim %386, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %391 = stablehlo.broadcast_in_dim %389, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %392 = stablehlo.concatenate %390, %391, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %393 = "stablehlo.all_reduce"(%392) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_122 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %394 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %395 = stablehlo.divide %393, %394 : tensor<2x128xf32>
      %396 = stablehlo.slice %395 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %397 = stablehlo.reshape %396 : (tensor<1x128xf32>) -> tensor<128xf32>
      %398 = stablehlo.slice %395 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %399 = stablehlo.reshape %398 : (tensor<1x128xf32>) -> tensor<128xf32>
      %400 = stablehlo.multiply %397, %397 : tensor<128xf32>
      %cst_123 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %402 = stablehlo.multiply %401, %397 : tensor<128xf32>
      %403 = stablehlo.subtract %399, %400 : tensor<128xf32>
      %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_124, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %405 = stablehlo.maximum %404, %403 : tensor<128xf32>
      %406 = stablehlo.compare  EQ, %403, %405,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_125 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %409 = stablehlo.select %406, %407, %408 : tensor<128xi1>, tensor<128xf32>
      %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_127, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %411 = stablehlo.compare  EQ, %410, %405,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_128 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %412 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_129 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %414 = stablehlo.select %411, %412, %413 : tensor<128xi1>, tensor<128xf32>
      %415 = stablehlo.divide %409, %414 : tensor<128xf32>
      %cst_130 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %417 = stablehlo.multiply %416, %arg304 : tensor<128xf32>
      %cst_131 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %419 = stablehlo.multiply %418, %397 : tensor<128xf32>
      %420 = stablehlo.add %417, %419 : tensor<128xf32>
      %cst_132 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %422 = stablehlo.multiply %421, %arg305 : tensor<128xf32>
      %cst_133 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_133, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %424 = stablehlo.multiply %423, %405 : tensor<128xf32>
      %425 = stablehlo.add %422, %424 : tensor<128xf32>
      %426 = stablehlo.broadcast_in_dim %397, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %427 = stablehlo.broadcast_in_dim %405, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %428 = stablehlo.convert %379 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %429 = stablehlo.broadcast_in_dim %426, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %430 = stablehlo.subtract %428, %429 : tensor<256x28x28x128xf32>
      %cst_134 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_134, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %432 = stablehlo.add %427, %431 : tensor<1x1x1x128xf32>
      %433 = stablehlo.rsqrt %432 : tensor<1x1x1x128xf32>
      %434 = stablehlo.divide %433, %432 : tensor<1x1x1x128xf32>
      %cst_135 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %436 = stablehlo.multiply %435, %434 : tensor<1x1x1x128xf32>
      %437 = stablehlo.reshape %arg186 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %438 = stablehlo.multiply %433, %437 : tensor<1x1x1x128xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %440 = stablehlo.multiply %430, %439 : tensor<256x28x28x128xf32>
      %441 = stablehlo.reshape %arg185 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %443 = stablehlo.add %440, %442 : tensor<256x28x28x128xf32>
      %444 = stablehlo.convert %443 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %445 = func.call @relu_88(%444) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %446 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %447 = stablehlo.compare  GT, %444, %446,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %448 = stablehlo.convert %arg190 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %449 = stablehlo.convolution(%445, %448) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %450 = stablehlo.convert %449 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %451 = stablehlo.multiply %450, %450 : tensor<256x28x28x128xf32>
      %cst_137 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_137, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %453 = stablehlo.multiply %452, %450 : tensor<256x28x28x128xf32>
      %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.reduce(%450 init: %cst_138) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_139 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %456 = stablehlo.divide %454, %455 : tensor<128xf32>
      %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %457 = stablehlo.reduce(%451 init: %cst_140) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_141 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %459 = stablehlo.divide %457, %458 : tensor<128xf32>
      %460 = stablehlo.broadcast_in_dim %456, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %461 = stablehlo.broadcast_in_dim %459, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %462 = stablehlo.concatenate %460, %461, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %463 = "stablehlo.all_reduce"(%462) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_142 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %465 = stablehlo.divide %463, %464 : tensor<2x128xf32>
      %466 = stablehlo.slice %465 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %467 = stablehlo.reshape %466 : (tensor<1x128xf32>) -> tensor<128xf32>
      %468 = stablehlo.slice %465 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %469 = stablehlo.reshape %468 : (tensor<1x128xf32>) -> tensor<128xf32>
      %470 = stablehlo.multiply %467, %467 : tensor<128xf32>
      %cst_143 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %471 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %472 = stablehlo.multiply %471, %467 : tensor<128xf32>
      %473 = stablehlo.subtract %469, %470 : tensor<128xf32>
      %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %474 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %475 = stablehlo.maximum %474, %473 : tensor<128xf32>
      %476 = stablehlo.compare  EQ, %473, %475,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_145 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %479 = stablehlo.select %476, %477, %478 : tensor<128xi1>, tensor<128xf32>
      %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %480 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %481 = stablehlo.compare  EQ, %480, %475,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_148 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_149 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %483 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %484 = stablehlo.select %481, %482, %483 : tensor<128xi1>, tensor<128xf32>
      %485 = stablehlo.divide %479, %484 : tensor<128xf32>
      %cst_150 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %486 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %487 = stablehlo.multiply %486, %arg306 : tensor<128xf32>
      %cst_151 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %489 = stablehlo.multiply %488, %467 : tensor<128xf32>
      %490 = stablehlo.add %487, %489 : tensor<128xf32>
      %cst_152 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %491 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %492 = stablehlo.multiply %491, %arg307 : tensor<128xf32>
      %cst_153 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %494 = stablehlo.multiply %493, %475 : tensor<128xf32>
      %495 = stablehlo.add %492, %494 : tensor<128xf32>
      %496 = stablehlo.broadcast_in_dim %467, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %497 = stablehlo.broadcast_in_dim %475, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %498 = stablehlo.convert %449 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %499 = stablehlo.broadcast_in_dim %496, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %500 = stablehlo.subtract %498, %499 : tensor<256x28x28x128xf32>
      %cst_154 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %502 = stablehlo.add %497, %501 : tensor<1x1x1x128xf32>
      %503 = stablehlo.rsqrt %502 : tensor<1x1x1x128xf32>
      %504 = stablehlo.divide %503, %502 : tensor<1x1x1x128xf32>
      %cst_155 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %505 = stablehlo.broadcast_in_dim %cst_155, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %506 = stablehlo.multiply %505, %504 : tensor<1x1x1x128xf32>
      %507 = stablehlo.reshape %arg188 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %508 = stablehlo.multiply %503, %507 : tensor<1x1x1x128xf32>
      %509 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %510 = stablehlo.multiply %500, %509 : tensor<256x28x28x128xf32>
      %511 = stablehlo.reshape %arg187 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %512 = stablehlo.broadcast_in_dim %511, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %513 = stablehlo.add %510, %512 : tensor<256x28x28x128xf32>
      %514 = stablehlo.convert %513 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %515 = stablehlo.convert %arg191 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf16>
      %516 = stablehlo.convolution(%375, %515) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x128xf16>) -> tensor<256x28x28x128xf16>
      %517 = stablehlo.convert %516 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %518 = stablehlo.multiply %517, %517 : tensor<256x28x28x128xf32>
      %cst_156 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %519 = stablehlo.broadcast_in_dim %cst_156, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %520 = stablehlo.multiply %519, %517 : tensor<256x28x28x128xf32>
      %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %521 = stablehlo.reduce(%517 init: %cst_157) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_158 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %522 = stablehlo.broadcast_in_dim %cst_158, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %523 = stablehlo.divide %521, %522 : tensor<128xf32>
      %cst_159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %524 = stablehlo.reduce(%518 init: %cst_159) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_160 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %525 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %526 = stablehlo.divide %524, %525 : tensor<128xf32>
      %527 = stablehlo.broadcast_in_dim %523, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %528 = stablehlo.broadcast_in_dim %526, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %529 = stablehlo.concatenate %527, %528, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %530 = "stablehlo.all_reduce"(%529) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_161 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %531 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %532 = stablehlo.divide %530, %531 : tensor<2x128xf32>
      %533 = stablehlo.slice %532 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %534 = stablehlo.reshape %533 : (tensor<1x128xf32>) -> tensor<128xf32>
      %535 = stablehlo.slice %532 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %536 = stablehlo.reshape %535 : (tensor<1x128xf32>) -> tensor<128xf32>
      %537 = stablehlo.multiply %534, %534 : tensor<128xf32>
      %cst_162 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %538 = stablehlo.broadcast_in_dim %cst_162, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %539 = stablehlo.multiply %538, %534 : tensor<128xf32>
      %540 = stablehlo.subtract %536, %537 : tensor<128xf32>
      %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %541 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %542 = stablehlo.maximum %541, %540 : tensor<128xf32>
      %543 = stablehlo.compare  EQ, %540, %542,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_164 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %545 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %546 = stablehlo.select %543, %544, %545 : tensor<128xi1>, tensor<128xf32>
      %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %547 = stablehlo.broadcast_in_dim %cst_166, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %548 = stablehlo.compare  EQ, %547, %542,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_167 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %549 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_168 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %550 = stablehlo.broadcast_in_dim %cst_168, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %551 = stablehlo.select %548, %549, %550 : tensor<128xi1>, tensor<128xf32>
      %552 = stablehlo.divide %546, %551 : tensor<128xf32>
      %cst_169 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %553 = stablehlo.broadcast_in_dim %cst_169, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %554 = stablehlo.multiply %553, %arg308 : tensor<128xf32>
      %cst_170 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %555 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %556 = stablehlo.multiply %555, %534 : tensor<128xf32>
      %557 = stablehlo.add %554, %556 : tensor<128xf32>
      %cst_171 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %558 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %559 = stablehlo.multiply %558, %arg309 : tensor<128xf32>
      %cst_172 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %560 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %561 = stablehlo.multiply %560, %542 : tensor<128xf32>
      %562 = stablehlo.add %559, %561 : tensor<128xf32>
      %563 = stablehlo.broadcast_in_dim %534, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %564 = stablehlo.broadcast_in_dim %542, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %565 = stablehlo.convert %516 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %566 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %567 = stablehlo.subtract %565, %566 : tensor<256x28x28x128xf32>
      %cst_173 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %568 = stablehlo.broadcast_in_dim %cst_173, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %569 = stablehlo.add %564, %568 : tensor<1x1x1x128xf32>
      %570 = stablehlo.rsqrt %569 : tensor<1x1x1x128xf32>
      %571 = stablehlo.divide %570, %569 : tensor<1x1x1x128xf32>
      %cst_174 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %572 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %573 = stablehlo.multiply %572, %571 : tensor<1x1x1x128xf32>
      %574 = stablehlo.reshape %arg193 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %575 = stablehlo.multiply %570, %574 : tensor<1x1x1x128xf32>
      %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %577 = stablehlo.multiply %567, %576 : tensor<256x28x28x128xf32>
      %578 = stablehlo.reshape %arg192 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %579 = stablehlo.broadcast_in_dim %578, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %580 = stablehlo.add %577, %579 : tensor<256x28x28x128xf32>
      %581 = stablehlo.convert %580 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %582 = stablehlo.add %581, %514 : tensor<256x28x28x128xf16>
      %583 = func.call @relu_88(%582) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_175 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %584 = stablehlo.broadcast_in_dim %cst_175, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %585 = stablehlo.compare  GT, %582, %584,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %586 = stablehlo.convert %arg198 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %587 = stablehlo.convolution(%583, %586) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %588 = stablehlo.convert %587 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %589 = stablehlo.multiply %588, %588 : tensor<256x28x28x128xf32>
      %cst_176 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %590 = stablehlo.broadcast_in_dim %cst_176, dims = [] : (tensor<f32>) -> tensor<256x28x28x128xf32>
      %591 = stablehlo.multiply %590, %588 : tensor<256x28x28x128xf32>
      %cst_177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %592 = stablehlo.reduce(%588 init: %cst_177) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_178 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %593 = stablehlo.broadcast_in_dim %cst_178, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %594 = stablehlo.divide %592, %593 : tensor<128xf32>
      %cst_179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %595 = stablehlo.reduce(%589 init: %cst_179) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_180 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %596 = stablehlo.broadcast_in_dim %cst_180, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %597 = stablehlo.divide %595, %596 : tensor<128xf32>
      %598 = stablehlo.broadcast_in_dim %594, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %599 = stablehlo.broadcast_in_dim %597, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %600 = stablehlo.concatenate %598, %599, dim = 0 : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>
      %601 = "stablehlo.all_reduce"(%600) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %cst_181 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %602 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %603 = stablehlo.divide %601, %602 : tensor<2x128xf32>
      %604 = stablehlo.slice %603 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %605 = stablehlo.reshape %604 : (tensor<1x128xf32>) -> tensor<128xf32>
      %606 = stablehlo.slice %603 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %607 = stablehlo.reshape %606 : (tensor<1x128xf32>) -> tensor<128xf32>
      %608 = stablehlo.multiply %605, %605 : tensor<128xf32>
      %cst_182 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %609 = stablehlo.broadcast_in_dim %cst_182, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %610 = stablehlo.multiply %609, %605 : tensor<128xf32>
      %611 = stablehlo.subtract %607, %608 : tensor<128xf32>
      %cst_183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %612 = stablehlo.broadcast_in_dim %cst_183, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %613 = stablehlo.maximum %612, %611 : tensor<128xf32>
      %614 = stablehlo.compare  EQ, %611, %613,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_184 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %615 = stablehlo.broadcast_in_dim %cst_184, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %616 = stablehlo.broadcast_in_dim %cst_185, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %617 = stablehlo.select %614, %615, %616 : tensor<128xi1>, tensor<128xf32>
      %cst_186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %618 = stablehlo.broadcast_in_dim %cst_186, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %619 = stablehlo.compare  EQ, %618, %613,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
      %cst_187 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %620 = stablehlo.broadcast_in_dim %cst_187, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_188 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %621 = stablehlo.broadcast_in_dim %cst_188, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %622 = stablehlo.select %619, %620, %621 : tensor<128xi1>, tensor<128xf32>
      %623 = stablehlo.divide %617, %622 : tensor<128xf32>
      %cst_189 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %624 = stablehlo.broadcast_in_dim %cst_189, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %625 = stablehlo.multiply %624, %arg310 : tensor<128xf32>
      %cst_190 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %626 = stablehlo.broadcast_in_dim %cst_190, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %627 = stablehlo.multiply %626, %605 : tensor<128xf32>
      %628 = stablehlo.add %625, %627 : tensor<128xf32>
      %cst_191 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %629 = stablehlo.broadcast_in_dim %cst_191, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %630 = stablehlo.multiply %629, %arg311 : tensor<128xf32>
      %cst_192 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %631 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %632 = stablehlo.multiply %631, %613 : tensor<128xf32>
      %633 = stablehlo.add %630, %632 : tensor<128xf32>
      %634 = stablehlo.broadcast_in_dim %605, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %635 = stablehlo.broadcast_in_dim %613, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %636 = stablehlo.convert %587 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %637 = stablehlo.broadcast_in_dim %634, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %638 = stablehlo.subtract %636, %637 : tensor<256x28x28x128xf32>
      %cst_193 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %639 = stablehlo.broadcast_in_dim %cst_193, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %640 = stablehlo.add %635, %639 : tensor<1x1x1x128xf32>
      %641 = stablehlo.rsqrt %640 : tensor<1x1x1x128xf32>
      %642 = stablehlo.divide %641, %640 : tensor<1x1x1x128xf32>
      %cst_194 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_194, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %644 = stablehlo.multiply %643, %642 : tensor<1x1x1x128xf32>
      %645 = stablehlo.reshape %arg195 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %646 = stablehlo.multiply %641, %645 : tensor<1x1x1x128xf32>
      %647 = stablehlo.broadcast_in_dim %646, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %648 = stablehlo.multiply %638, %647 : tensor<256x28x28x128xf32>
      %649 = stablehlo.reshape %arg194 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %651 = stablehlo.add %648, %650 : tensor<256x28x28x128xf32>
      %652 = stablehlo.convert %651 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %653 = func.call @relu_88(%652) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_195 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %654 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %655 = stablehlo.compare  GT, %652, %654,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %656 = stablehlo.convert %arg199 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %657 = stablehlo.convolution(%653, %656) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
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
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
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
      %695 = stablehlo.multiply %694, %arg312 : tensor<128xf32>
      %cst_210 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %696 = stablehlo.broadcast_in_dim %cst_210, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %697 = stablehlo.multiply %696, %675 : tensor<128xf32>
      %698 = stablehlo.add %695, %697 : tensor<128xf32>
      %cst_211 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %699 = stablehlo.broadcast_in_dim %cst_211, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %700 = stablehlo.multiply %699, %arg313 : tensor<128xf32>
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
      %715 = stablehlo.reshape %arg197 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %716 = stablehlo.multiply %711, %715 : tensor<1x1x1x128xf32>
      %717 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %718 = stablehlo.multiply %708, %717 : tensor<256x28x28x128xf32>
      %719 = stablehlo.reshape %arg196 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %720 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %721 = stablehlo.add %718, %720 : tensor<256x28x28x128xf32>
      %722 = stablehlo.convert %721 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %723 = stablehlo.add %583, %722 : tensor<256x28x28x128xf16>
      %724 = func.call @relu_88(%723) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %cst_215 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %725 = stablehlo.broadcast_in_dim %cst_215, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %726 = stablehlo.compare  GT, %723, %725,  FLOAT : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xi1>
      %727 = stablehlo.convert %arg204 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf16>
      %728 = stablehlo.convolution(%724, %727) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x256xf16>) -> tensor<256x14x14x256xf16>
      %729 = stablehlo.convert %728 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %730 = stablehlo.multiply %729, %729 : tensor<256x14x14x256xf32>
      %cst_216 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %731 = stablehlo.broadcast_in_dim %cst_216, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %732 = stablehlo.multiply %731, %729 : tensor<256x14x14x256xf32>
      %cst_217 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %733 = stablehlo.reduce(%729 init: %cst_217) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_218 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %734 = stablehlo.broadcast_in_dim %cst_218, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %735 = stablehlo.divide %733, %734 : tensor<256xf32>
      %cst_219 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %736 = stablehlo.reduce(%730 init: %cst_219) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_220 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %737 = stablehlo.broadcast_in_dim %cst_220, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %738 = stablehlo.divide %736, %737 : tensor<256xf32>
      %739 = stablehlo.broadcast_in_dim %735, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %740 = stablehlo.broadcast_in_dim %738, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %741 = stablehlo.concatenate %739, %740, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %742 = "stablehlo.all_reduce"(%741) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_221 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %743 = stablehlo.broadcast_in_dim %cst_221, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %744 = stablehlo.divide %742, %743 : tensor<2x256xf32>
      %745 = stablehlo.slice %744 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %746 = stablehlo.reshape %745 : (tensor<1x256xf32>) -> tensor<256xf32>
      %747 = stablehlo.slice %744 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %748 = stablehlo.reshape %747 : (tensor<1x256xf32>) -> tensor<256xf32>
      %749 = stablehlo.multiply %746, %746 : tensor<256xf32>
      %cst_222 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %750 = stablehlo.broadcast_in_dim %cst_222, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %751 = stablehlo.multiply %750, %746 : tensor<256xf32>
      %752 = stablehlo.subtract %748, %749 : tensor<256xf32>
      %cst_223 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %753 = stablehlo.broadcast_in_dim %cst_223, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %754 = stablehlo.maximum %753, %752 : tensor<256xf32>
      %755 = stablehlo.compare  EQ, %752, %754,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_224 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %756 = stablehlo.broadcast_in_dim %cst_224, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_225 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %757 = stablehlo.broadcast_in_dim %cst_225, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %758 = stablehlo.select %755, %756, %757 : tensor<256xi1>, tensor<256xf32>
      %cst_226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %759 = stablehlo.broadcast_in_dim %cst_226, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %760 = stablehlo.compare  EQ, %759, %754,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_227 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %761 = stablehlo.broadcast_in_dim %cst_227, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_228 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %762 = stablehlo.broadcast_in_dim %cst_228, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %763 = stablehlo.select %760, %761, %762 : tensor<256xi1>, tensor<256xf32>
      %764 = stablehlo.divide %758, %763 : tensor<256xf32>
      %cst_229 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %765 = stablehlo.broadcast_in_dim %cst_229, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %766 = stablehlo.multiply %765, %arg314 : tensor<256xf32>
      %cst_230 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %767 = stablehlo.broadcast_in_dim %cst_230, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %768 = stablehlo.multiply %767, %746 : tensor<256xf32>
      %769 = stablehlo.add %766, %768 : tensor<256xf32>
      %cst_231 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %770 = stablehlo.broadcast_in_dim %cst_231, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %771 = stablehlo.multiply %770, %arg315 : tensor<256xf32>
      %cst_232 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %772 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %773 = stablehlo.multiply %772, %754 : tensor<256xf32>
      %774 = stablehlo.add %771, %773 : tensor<256xf32>
      %775 = stablehlo.broadcast_in_dim %746, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %776 = stablehlo.broadcast_in_dim %754, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %777 = stablehlo.convert %728 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %778 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %779 = stablehlo.subtract %777, %778 : tensor<256x14x14x256xf32>
      %cst_233 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %780 = stablehlo.broadcast_in_dim %cst_233, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %781 = stablehlo.add %776, %780 : tensor<1x1x1x256xf32>
      %782 = stablehlo.rsqrt %781 : tensor<1x1x1x256xf32>
      %783 = stablehlo.divide %782, %781 : tensor<1x1x1x256xf32>
      %cst_234 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %784 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %785 = stablehlo.multiply %784, %783 : tensor<1x1x1x256xf32>
      %786 = stablehlo.reshape %arg201 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %787 = stablehlo.multiply %782, %786 : tensor<1x1x1x256xf32>
      %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %789 = stablehlo.multiply %779, %788 : tensor<256x14x14x256xf32>
      %790 = stablehlo.reshape %arg200 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %791 = stablehlo.broadcast_in_dim %790, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %792 = stablehlo.add %789, %791 : tensor<256x14x14x256xf32>
      %793 = stablehlo.convert %792 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %794 = func.call @relu_136(%793) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_235 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %795 = stablehlo.broadcast_in_dim %cst_235, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %796 = stablehlo.compare  GT, %793, %795,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %797 = stablehlo.convert %arg205 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %798 = stablehlo.convolution(%794, %797) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %799 = stablehlo.convert %798 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %800 = stablehlo.multiply %799, %799 : tensor<256x14x14x256xf32>
      %cst_236 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %801 = stablehlo.broadcast_in_dim %cst_236, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %802 = stablehlo.multiply %801, %799 : tensor<256x14x14x256xf32>
      %cst_237 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %803 = stablehlo.reduce(%799 init: %cst_237) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_238 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %804 = stablehlo.broadcast_in_dim %cst_238, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %805 = stablehlo.divide %803, %804 : tensor<256xf32>
      %cst_239 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %806 = stablehlo.reduce(%800 init: %cst_239) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_240 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %807 = stablehlo.broadcast_in_dim %cst_240, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %808 = stablehlo.divide %806, %807 : tensor<256xf32>
      %809 = stablehlo.broadcast_in_dim %805, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %810 = stablehlo.broadcast_in_dim %808, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %811 = stablehlo.concatenate %809, %810, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %812 = "stablehlo.all_reduce"(%811) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_241 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %813 = stablehlo.broadcast_in_dim %cst_241, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %814 = stablehlo.divide %812, %813 : tensor<2x256xf32>
      %815 = stablehlo.slice %814 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %816 = stablehlo.reshape %815 : (tensor<1x256xf32>) -> tensor<256xf32>
      %817 = stablehlo.slice %814 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %818 = stablehlo.reshape %817 : (tensor<1x256xf32>) -> tensor<256xf32>
      %819 = stablehlo.multiply %816, %816 : tensor<256xf32>
      %cst_242 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %820 = stablehlo.broadcast_in_dim %cst_242, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %821 = stablehlo.multiply %820, %816 : tensor<256xf32>
      %822 = stablehlo.subtract %818, %819 : tensor<256xf32>
      %cst_243 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %823 = stablehlo.broadcast_in_dim %cst_243, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %824 = stablehlo.maximum %823, %822 : tensor<256xf32>
      %825 = stablehlo.compare  EQ, %822, %824,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_244 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %826 = stablehlo.broadcast_in_dim %cst_244, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_245 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %827 = stablehlo.broadcast_in_dim %cst_245, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %828 = stablehlo.select %825, %826, %827 : tensor<256xi1>, tensor<256xf32>
      %cst_246 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %829 = stablehlo.broadcast_in_dim %cst_246, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %830 = stablehlo.compare  EQ, %829, %824,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_247 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %831 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_248 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %832 = stablehlo.broadcast_in_dim %cst_248, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %833 = stablehlo.select %830, %831, %832 : tensor<256xi1>, tensor<256xf32>
      %834 = stablehlo.divide %828, %833 : tensor<256xf32>
      %cst_249 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %835 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %836 = stablehlo.multiply %835, %arg316 : tensor<256xf32>
      %cst_250 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %837 = stablehlo.broadcast_in_dim %cst_250, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %838 = stablehlo.multiply %837, %816 : tensor<256xf32>
      %839 = stablehlo.add %836, %838 : tensor<256xf32>
      %cst_251 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %840 = stablehlo.broadcast_in_dim %cst_251, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %841 = stablehlo.multiply %840, %arg317 : tensor<256xf32>
      %cst_252 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %842 = stablehlo.broadcast_in_dim %cst_252, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %843 = stablehlo.multiply %842, %824 : tensor<256xf32>
      %844 = stablehlo.add %841, %843 : tensor<256xf32>
      %845 = stablehlo.broadcast_in_dim %816, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %846 = stablehlo.broadcast_in_dim %824, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %847 = stablehlo.convert %798 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %848 = stablehlo.broadcast_in_dim %845, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %849 = stablehlo.subtract %847, %848 : tensor<256x14x14x256xf32>
      %cst_253 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %850 = stablehlo.broadcast_in_dim %cst_253, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %851 = stablehlo.add %846, %850 : tensor<1x1x1x256xf32>
      %852 = stablehlo.rsqrt %851 : tensor<1x1x1x256xf32>
      %853 = stablehlo.divide %852, %851 : tensor<1x1x1x256xf32>
      %cst_254 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %854 = stablehlo.broadcast_in_dim %cst_254, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %855 = stablehlo.multiply %854, %853 : tensor<1x1x1x256xf32>
      %856 = stablehlo.reshape %arg203 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %857 = stablehlo.multiply %852, %856 : tensor<1x1x1x256xf32>
      %858 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %859 = stablehlo.multiply %849, %858 : tensor<256x14x14x256xf32>
      %860 = stablehlo.reshape %arg202 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %861 = stablehlo.broadcast_in_dim %860, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %862 = stablehlo.add %859, %861 : tensor<256x14x14x256xf32>
      %863 = stablehlo.convert %862 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %864 = stablehlo.convert %arg206 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf16>
      %865 = stablehlo.convolution(%724, %864) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x256xf16>) -> tensor<256x14x14x256xf16>
      %866 = stablehlo.convert %865 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %867 = stablehlo.multiply %866, %866 : tensor<256x14x14x256xf32>
      %cst_255 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %868 = stablehlo.broadcast_in_dim %cst_255, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %869 = stablehlo.multiply %868, %866 : tensor<256x14x14x256xf32>
      %cst_256 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %870 = stablehlo.reduce(%866 init: %cst_256) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_257 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %871 = stablehlo.broadcast_in_dim %cst_257, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %872 = stablehlo.divide %870, %871 : tensor<256xf32>
      %cst_258 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %873 = stablehlo.reduce(%867 init: %cst_258) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_259 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %874 = stablehlo.broadcast_in_dim %cst_259, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %875 = stablehlo.divide %873, %874 : tensor<256xf32>
      %876 = stablehlo.broadcast_in_dim %872, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %877 = stablehlo.broadcast_in_dim %875, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %878 = stablehlo.concatenate %876, %877, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %879 = "stablehlo.all_reduce"(%878) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_260 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %880 = stablehlo.broadcast_in_dim %cst_260, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %881 = stablehlo.divide %879, %880 : tensor<2x256xf32>
      %882 = stablehlo.slice %881 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %883 = stablehlo.reshape %882 : (tensor<1x256xf32>) -> tensor<256xf32>
      %884 = stablehlo.slice %881 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %885 = stablehlo.reshape %884 : (tensor<1x256xf32>) -> tensor<256xf32>
      %886 = stablehlo.multiply %883, %883 : tensor<256xf32>
      %cst_261 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %887 = stablehlo.broadcast_in_dim %cst_261, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %888 = stablehlo.multiply %887, %883 : tensor<256xf32>
      %889 = stablehlo.subtract %885, %886 : tensor<256xf32>
      %cst_262 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %890 = stablehlo.broadcast_in_dim %cst_262, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %891 = stablehlo.maximum %890, %889 : tensor<256xf32>
      %892 = stablehlo.compare  EQ, %889, %891,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_263 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %893 = stablehlo.broadcast_in_dim %cst_263, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_264 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %894 = stablehlo.broadcast_in_dim %cst_264, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %895 = stablehlo.select %892, %893, %894 : tensor<256xi1>, tensor<256xf32>
      %cst_265 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %896 = stablehlo.broadcast_in_dim %cst_265, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %897 = stablehlo.compare  EQ, %896, %891,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_266 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %898 = stablehlo.broadcast_in_dim %cst_266, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_267 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %899 = stablehlo.broadcast_in_dim %cst_267, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %900 = stablehlo.select %897, %898, %899 : tensor<256xi1>, tensor<256xf32>
      %901 = stablehlo.divide %895, %900 : tensor<256xf32>
      %cst_268 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %902 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %903 = stablehlo.multiply %902, %arg318 : tensor<256xf32>
      %cst_269 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %904 = stablehlo.broadcast_in_dim %cst_269, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %905 = stablehlo.multiply %904, %883 : tensor<256xf32>
      %906 = stablehlo.add %903, %905 : tensor<256xf32>
      %cst_270 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %907 = stablehlo.broadcast_in_dim %cst_270, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %908 = stablehlo.multiply %907, %arg319 : tensor<256xf32>
      %cst_271 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %909 = stablehlo.broadcast_in_dim %cst_271, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %910 = stablehlo.multiply %909, %891 : tensor<256xf32>
      %911 = stablehlo.add %908, %910 : tensor<256xf32>
      %912 = stablehlo.broadcast_in_dim %883, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %913 = stablehlo.broadcast_in_dim %891, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %914 = stablehlo.convert %865 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %915 = stablehlo.broadcast_in_dim %912, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %916 = stablehlo.subtract %914, %915 : tensor<256x14x14x256xf32>
      %cst_272 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %917 = stablehlo.broadcast_in_dim %cst_272, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %918 = stablehlo.add %913, %917 : tensor<1x1x1x256xf32>
      %919 = stablehlo.rsqrt %918 : tensor<1x1x1x256xf32>
      %920 = stablehlo.divide %919, %918 : tensor<1x1x1x256xf32>
      %cst_273 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %921 = stablehlo.broadcast_in_dim %cst_273, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %922 = stablehlo.multiply %921, %920 : tensor<1x1x1x256xf32>
      %923 = stablehlo.reshape %arg208 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %924 = stablehlo.multiply %919, %923 : tensor<1x1x1x256xf32>
      %925 = stablehlo.broadcast_in_dim %924, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %926 = stablehlo.multiply %916, %925 : tensor<256x14x14x256xf32>
      %927 = stablehlo.reshape %arg207 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %928 = stablehlo.broadcast_in_dim %927, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %929 = stablehlo.add %926, %928 : tensor<256x14x14x256xf32>
      %930 = stablehlo.convert %929 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %931 = stablehlo.add %930, %863 : tensor<256x14x14x256xf16>
      %932 = func.call @relu_136(%931) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_274 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %933 = stablehlo.broadcast_in_dim %cst_274, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %934 = stablehlo.compare  GT, %931, %933,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %935 = stablehlo.convert %arg213 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %936 = stablehlo.convolution(%932, %935) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %937 = stablehlo.convert %936 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %938 = stablehlo.multiply %937, %937 : tensor<256x14x14x256xf32>
      %cst_275 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %939 = stablehlo.broadcast_in_dim %cst_275, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %940 = stablehlo.multiply %939, %937 : tensor<256x14x14x256xf32>
      %cst_276 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %941 = stablehlo.reduce(%937 init: %cst_276) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_277 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %942 = stablehlo.broadcast_in_dim %cst_277, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %943 = stablehlo.divide %941, %942 : tensor<256xf32>
      %cst_278 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %944 = stablehlo.reduce(%938 init: %cst_278) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_279 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %945 = stablehlo.broadcast_in_dim %cst_279, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %946 = stablehlo.divide %944, %945 : tensor<256xf32>
      %947 = stablehlo.broadcast_in_dim %943, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %948 = stablehlo.broadcast_in_dim %946, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %949 = stablehlo.concatenate %947, %948, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %950 = "stablehlo.all_reduce"(%949) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_280 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %951 = stablehlo.broadcast_in_dim %cst_280, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %952 = stablehlo.divide %950, %951 : tensor<2x256xf32>
      %953 = stablehlo.slice %952 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %954 = stablehlo.reshape %953 : (tensor<1x256xf32>) -> tensor<256xf32>
      %955 = stablehlo.slice %952 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %956 = stablehlo.reshape %955 : (tensor<1x256xf32>) -> tensor<256xf32>
      %957 = stablehlo.multiply %954, %954 : tensor<256xf32>
      %cst_281 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %958 = stablehlo.broadcast_in_dim %cst_281, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %959 = stablehlo.multiply %958, %954 : tensor<256xf32>
      %960 = stablehlo.subtract %956, %957 : tensor<256xf32>
      %cst_282 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %961 = stablehlo.broadcast_in_dim %cst_282, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %962 = stablehlo.maximum %961, %960 : tensor<256xf32>
      %963 = stablehlo.compare  EQ, %960, %962,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_283 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %964 = stablehlo.broadcast_in_dim %cst_283, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_284 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %965 = stablehlo.broadcast_in_dim %cst_284, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %966 = stablehlo.select %963, %964, %965 : tensor<256xi1>, tensor<256xf32>
      %cst_285 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %967 = stablehlo.broadcast_in_dim %cst_285, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %968 = stablehlo.compare  EQ, %967, %962,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_286 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %969 = stablehlo.broadcast_in_dim %cst_286, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_287 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %970 = stablehlo.broadcast_in_dim %cst_287, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %971 = stablehlo.select %968, %969, %970 : tensor<256xi1>, tensor<256xf32>
      %972 = stablehlo.divide %966, %971 : tensor<256xf32>
      %cst_288 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %973 = stablehlo.broadcast_in_dim %cst_288, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %974 = stablehlo.multiply %973, %arg320 : tensor<256xf32>
      %cst_289 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %975 = stablehlo.broadcast_in_dim %cst_289, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %976 = stablehlo.multiply %975, %954 : tensor<256xf32>
      %977 = stablehlo.add %974, %976 : tensor<256xf32>
      %cst_290 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %978 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %979 = stablehlo.multiply %978, %arg321 : tensor<256xf32>
      %cst_291 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %980 = stablehlo.broadcast_in_dim %cst_291, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %981 = stablehlo.multiply %980, %962 : tensor<256xf32>
      %982 = stablehlo.add %979, %981 : tensor<256xf32>
      %983 = stablehlo.broadcast_in_dim %954, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %984 = stablehlo.broadcast_in_dim %962, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %985 = stablehlo.convert %936 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %986 = stablehlo.broadcast_in_dim %983, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %987 = stablehlo.subtract %985, %986 : tensor<256x14x14x256xf32>
      %cst_292 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %988 = stablehlo.broadcast_in_dim %cst_292, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %989 = stablehlo.add %984, %988 : tensor<1x1x1x256xf32>
      %990 = stablehlo.rsqrt %989 : tensor<1x1x1x256xf32>
      %991 = stablehlo.divide %990, %989 : tensor<1x1x1x256xf32>
      %cst_293 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %992 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %993 = stablehlo.multiply %992, %991 : tensor<1x1x1x256xf32>
      %994 = stablehlo.reshape %arg210 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %995 = stablehlo.multiply %990, %994 : tensor<1x1x1x256xf32>
      %996 = stablehlo.broadcast_in_dim %995, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %997 = stablehlo.multiply %987, %996 : tensor<256x14x14x256xf32>
      %998 = stablehlo.reshape %arg209 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %999 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1000 = stablehlo.add %997, %999 : tensor<256x14x14x256xf32>
      %1001 = stablehlo.convert %1000 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1002 = func.call @relu_136(%1001) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_294 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1003 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1004 = stablehlo.compare  GT, %1001, %1003,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1005 = stablehlo.convert %arg214 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1006 = stablehlo.convolution(%1002, %1005) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1007 = stablehlo.convert %1006 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1008 = stablehlo.multiply %1007, %1007 : tensor<256x14x14x256xf32>
      %cst_295 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1009 = stablehlo.broadcast_in_dim %cst_295, dims = [] : (tensor<f32>) -> tensor<256x14x14x256xf32>
      %1010 = stablehlo.multiply %1009, %1007 : tensor<256x14x14x256xf32>
      %cst_296 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1011 = stablehlo.reduce(%1007 init: %cst_296) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_297 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1012 = stablehlo.broadcast_in_dim %cst_297, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1013 = stablehlo.divide %1011, %1012 : tensor<256xf32>
      %cst_298 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1014 = stablehlo.reduce(%1008 init: %cst_298) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_299 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1015 = stablehlo.broadcast_in_dim %cst_299, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1016 = stablehlo.divide %1014, %1015 : tensor<256xf32>
      %1017 = stablehlo.broadcast_in_dim %1013, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1018 = stablehlo.broadcast_in_dim %1016, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1019 = stablehlo.concatenate %1017, %1018, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
      %1020 = "stablehlo.all_reduce"(%1019) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %cst_300 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1021 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1022 = stablehlo.divide %1020, %1021 : tensor<2x256xf32>
      %1023 = stablehlo.slice %1022 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1024 = stablehlo.reshape %1023 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1025 = stablehlo.slice %1022 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1026 = stablehlo.reshape %1025 : (tensor<1x256xf32>) -> tensor<256xf32>
      %1027 = stablehlo.multiply %1024, %1024 : tensor<256xf32>
      %cst_301 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1028 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1029 = stablehlo.multiply %1028, %1024 : tensor<256xf32>
      %1030 = stablehlo.subtract %1026, %1027 : tensor<256xf32>
      %cst_302 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1031 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1032 = stablehlo.maximum %1031, %1030 : tensor<256xf32>
      %1033 = stablehlo.compare  EQ, %1030, %1032,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_303 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1034 = stablehlo.broadcast_in_dim %cst_303, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_304 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1035 = stablehlo.broadcast_in_dim %cst_304, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1036 = stablehlo.select %1033, %1034, %1035 : tensor<256xi1>, tensor<256xf32>
      %cst_305 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1037 = stablehlo.broadcast_in_dim %cst_305, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1038 = stablehlo.compare  EQ, %1037, %1032,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_306 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1039 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %cst_307 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1040 = stablehlo.broadcast_in_dim %cst_307, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1041 = stablehlo.select %1038, %1039, %1040 : tensor<256xi1>, tensor<256xf32>
      %1042 = stablehlo.divide %1036, %1041 : tensor<256xf32>
      %cst_308 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1043 = stablehlo.broadcast_in_dim %cst_308, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1044 = stablehlo.multiply %1043, %arg322 : tensor<256xf32>
      %cst_309 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1045 = stablehlo.broadcast_in_dim %cst_309, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1046 = stablehlo.multiply %1045, %1024 : tensor<256xf32>
      %1047 = stablehlo.add %1044, %1046 : tensor<256xf32>
      %cst_310 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1048 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1049 = stablehlo.multiply %1048, %arg323 : tensor<256xf32>
      %cst_311 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1050 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1051 = stablehlo.multiply %1050, %1032 : tensor<256xf32>
      %1052 = stablehlo.add %1049, %1051 : tensor<256xf32>
      %1053 = stablehlo.broadcast_in_dim %1024, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1054 = stablehlo.broadcast_in_dim %1032, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1055 = stablehlo.convert %1006 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1056 = stablehlo.broadcast_in_dim %1053, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1057 = stablehlo.subtract %1055, %1056 : tensor<256x14x14x256xf32>
      %cst_312 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1058 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1059 = stablehlo.add %1054, %1058 : tensor<1x1x1x256xf32>
      %1060 = stablehlo.rsqrt %1059 : tensor<1x1x1x256xf32>
      %1061 = stablehlo.divide %1060, %1059 : tensor<1x1x1x256xf32>
      %cst_313 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1062 = stablehlo.broadcast_in_dim %cst_313, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1063 = stablehlo.multiply %1062, %1061 : tensor<1x1x1x256xf32>
      %1064 = stablehlo.reshape %arg212 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1065 = stablehlo.multiply %1060, %1064 : tensor<1x1x1x256xf32>
      %1066 = stablehlo.broadcast_in_dim %1065, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1067 = stablehlo.multiply %1057, %1066 : tensor<256x14x14x256xf32>
      %1068 = stablehlo.reshape %arg211 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1069 = stablehlo.broadcast_in_dim %1068, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1070 = stablehlo.add %1067, %1069 : tensor<256x14x14x256xf32>
      %1071 = stablehlo.convert %1070 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1072 = stablehlo.add %932, %1071 : tensor<256x14x14x256xf16>
      %1073 = func.call @relu_136(%1072) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %cst_314 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1074 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1075 = stablehlo.compare  GT, %1072, %1074,  FLOAT : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xi1>
      %1076 = stablehlo.convert %arg219 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf16>
      %1077 = stablehlo.convolution(%1073, %1076) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x512xf16>) -> tensor<256x7x7x512xf16>
      %1078 = stablehlo.convert %1077 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1079 = stablehlo.multiply %1078, %1078 : tensor<256x7x7x512xf32>
      %cst_315 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1080 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %1081 = stablehlo.multiply %1080, %1078 : tensor<256x7x7x512xf32>
      %cst_316 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1082 = stablehlo.reduce(%1078 init: %cst_316) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_317 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1083 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1084 = stablehlo.divide %1082, %1083 : tensor<512xf32>
      %cst_318 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1085 = stablehlo.reduce(%1079 init: %cst_318) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_319 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1086 = stablehlo.broadcast_in_dim %cst_319, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1087 = stablehlo.divide %1085, %1086 : tensor<512xf32>
      %1088 = stablehlo.broadcast_in_dim %1084, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1089 = stablehlo.broadcast_in_dim %1087, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1090 = stablehlo.concatenate %1088, %1089, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %1091 = "stablehlo.all_reduce"(%1090) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_320 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1092 = stablehlo.broadcast_in_dim %cst_320, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1093 = stablehlo.divide %1091, %1092 : tensor<2x512xf32>
      %1094 = stablehlo.slice %1093 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1095 = stablehlo.reshape %1094 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1096 = stablehlo.slice %1093 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1097 = stablehlo.reshape %1096 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1098 = stablehlo.multiply %1095, %1095 : tensor<512xf32>
      %cst_321 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1099 = stablehlo.broadcast_in_dim %cst_321, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1100 = stablehlo.multiply %1099, %1095 : tensor<512xf32>
      %1101 = stablehlo.subtract %1097, %1098 : tensor<512xf32>
      %cst_322 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1102 = stablehlo.broadcast_in_dim %cst_322, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1103 = stablehlo.maximum %1102, %1101 : tensor<512xf32>
      %1104 = stablehlo.compare  EQ, %1101, %1103,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_323 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1105 = stablehlo.broadcast_in_dim %cst_323, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_324 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1106 = stablehlo.broadcast_in_dim %cst_324, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1107 = stablehlo.select %1104, %1105, %1106 : tensor<512xi1>, tensor<512xf32>
      %cst_325 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1108 = stablehlo.broadcast_in_dim %cst_325, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1109 = stablehlo.compare  EQ, %1108, %1103,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_326 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1110 = stablehlo.broadcast_in_dim %cst_326, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_327 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1111 = stablehlo.broadcast_in_dim %cst_327, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1112 = stablehlo.select %1109, %1110, %1111 : tensor<512xi1>, tensor<512xf32>
      %1113 = stablehlo.divide %1107, %1112 : tensor<512xf32>
      %cst_328 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1114 = stablehlo.broadcast_in_dim %cst_328, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1115 = stablehlo.multiply %1114, %arg324 : tensor<512xf32>
      %cst_329 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1116 = stablehlo.broadcast_in_dim %cst_329, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1117 = stablehlo.multiply %1116, %1095 : tensor<512xf32>
      %1118 = stablehlo.add %1115, %1117 : tensor<512xf32>
      %cst_330 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1119 = stablehlo.broadcast_in_dim %cst_330, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1120 = stablehlo.multiply %1119, %arg325 : tensor<512xf32>
      %cst_331 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1121 = stablehlo.broadcast_in_dim %cst_331, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1122 = stablehlo.multiply %1121, %1103 : tensor<512xf32>
      %1123 = stablehlo.add %1120, %1122 : tensor<512xf32>
      %1124 = stablehlo.broadcast_in_dim %1095, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1125 = stablehlo.broadcast_in_dim %1103, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1126 = stablehlo.convert %1077 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1127 = stablehlo.broadcast_in_dim %1124, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1128 = stablehlo.subtract %1126, %1127 : tensor<256x7x7x512xf32>
      %cst_332 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1129 = stablehlo.broadcast_in_dim %cst_332, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1130 = stablehlo.add %1125, %1129 : tensor<1x1x1x512xf32>
      %1131 = stablehlo.rsqrt %1130 : tensor<1x1x1x512xf32>
      %1132 = stablehlo.divide %1131, %1130 : tensor<1x1x1x512xf32>
      %cst_333 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1133 = stablehlo.broadcast_in_dim %cst_333, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1134 = stablehlo.multiply %1133, %1132 : tensor<1x1x1x512xf32>
      %1135 = stablehlo.reshape %arg216 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1136 = stablehlo.multiply %1131, %1135 : tensor<1x1x1x512xf32>
      %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1138 = stablehlo.multiply %1128, %1137 : tensor<256x7x7x512xf32>
      %1139 = stablehlo.reshape %arg215 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1140 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1141 = stablehlo.add %1138, %1140 : tensor<256x7x7x512xf32>
      %1142 = stablehlo.convert %1141 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1143 = func.call @relu_184(%1142) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_334 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1144 = stablehlo.broadcast_in_dim %cst_334, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %1145 = stablehlo.compare  GT, %1142, %1144,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %1146 = stablehlo.convert %arg220 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %1147 = stablehlo.convolution(%1143, %1146) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %1148 = stablehlo.convert %1147 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1149 = stablehlo.multiply %1148, %1148 : tensor<256x7x7x512xf32>
      %cst_335 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1150 = stablehlo.broadcast_in_dim %cst_335, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %1151 = stablehlo.multiply %1150, %1148 : tensor<256x7x7x512xf32>
      %cst_336 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1152 = stablehlo.reduce(%1148 init: %cst_336) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_337 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1153 = stablehlo.broadcast_in_dim %cst_337, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1154 = stablehlo.divide %1152, %1153 : tensor<512xf32>
      %cst_338 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1155 = stablehlo.reduce(%1149 init: %cst_338) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_339 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1156 = stablehlo.broadcast_in_dim %cst_339, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1157 = stablehlo.divide %1155, %1156 : tensor<512xf32>
      %1158 = stablehlo.broadcast_in_dim %1154, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1159 = stablehlo.broadcast_in_dim %1157, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1160 = stablehlo.concatenate %1158, %1159, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %1161 = "stablehlo.all_reduce"(%1160) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_340 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1162 = stablehlo.broadcast_in_dim %cst_340, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1163 = stablehlo.divide %1161, %1162 : tensor<2x512xf32>
      %1164 = stablehlo.slice %1163 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1165 = stablehlo.reshape %1164 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1166 = stablehlo.slice %1163 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1167 = stablehlo.reshape %1166 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1168 = stablehlo.multiply %1165, %1165 : tensor<512xf32>
      %cst_341 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1169 = stablehlo.broadcast_in_dim %cst_341, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1170 = stablehlo.multiply %1169, %1165 : tensor<512xf32>
      %1171 = stablehlo.subtract %1167, %1168 : tensor<512xf32>
      %cst_342 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1172 = stablehlo.broadcast_in_dim %cst_342, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1173 = stablehlo.maximum %1172, %1171 : tensor<512xf32>
      %1174 = stablehlo.compare  EQ, %1171, %1173,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_343 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1175 = stablehlo.broadcast_in_dim %cst_343, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_344 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1176 = stablehlo.broadcast_in_dim %cst_344, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1177 = stablehlo.select %1174, %1175, %1176 : tensor<512xi1>, tensor<512xf32>
      %cst_345 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1178 = stablehlo.broadcast_in_dim %cst_345, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1179 = stablehlo.compare  EQ, %1178, %1173,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_346 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1180 = stablehlo.broadcast_in_dim %cst_346, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_347 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1181 = stablehlo.broadcast_in_dim %cst_347, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1182 = stablehlo.select %1179, %1180, %1181 : tensor<512xi1>, tensor<512xf32>
      %1183 = stablehlo.divide %1177, %1182 : tensor<512xf32>
      %cst_348 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1184 = stablehlo.broadcast_in_dim %cst_348, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1185 = stablehlo.multiply %1184, %arg326 : tensor<512xf32>
      %cst_349 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1186 = stablehlo.broadcast_in_dim %cst_349, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1187 = stablehlo.multiply %1186, %1165 : tensor<512xf32>
      %1188 = stablehlo.add %1185, %1187 : tensor<512xf32>
      %cst_350 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1189 = stablehlo.broadcast_in_dim %cst_350, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1190 = stablehlo.multiply %1189, %arg327 : tensor<512xf32>
      %cst_351 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1191 = stablehlo.broadcast_in_dim %cst_351, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1192 = stablehlo.multiply %1191, %1173 : tensor<512xf32>
      %1193 = stablehlo.add %1190, %1192 : tensor<512xf32>
      %1194 = stablehlo.broadcast_in_dim %1165, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1195 = stablehlo.broadcast_in_dim %1173, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1196 = stablehlo.convert %1147 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1197 = stablehlo.broadcast_in_dim %1194, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1198 = stablehlo.subtract %1196, %1197 : tensor<256x7x7x512xf32>
      %cst_352 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1199 = stablehlo.broadcast_in_dim %cst_352, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1200 = stablehlo.add %1195, %1199 : tensor<1x1x1x512xf32>
      %1201 = stablehlo.rsqrt %1200 : tensor<1x1x1x512xf32>
      %1202 = stablehlo.divide %1201, %1200 : tensor<1x1x1x512xf32>
      %cst_353 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1203 = stablehlo.broadcast_in_dim %cst_353, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1204 = stablehlo.multiply %1203, %1202 : tensor<1x1x1x512xf32>
      %1205 = stablehlo.reshape %arg218 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1206 = stablehlo.multiply %1201, %1205 : tensor<1x1x1x512xf32>
      %1207 = stablehlo.broadcast_in_dim %1206, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1208 = stablehlo.multiply %1198, %1207 : tensor<256x7x7x512xf32>
      %1209 = stablehlo.reshape %arg217 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1210 = stablehlo.broadcast_in_dim %1209, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1211 = stablehlo.add %1208, %1210 : tensor<256x7x7x512xf32>
      %1212 = stablehlo.convert %1211 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1213 = stablehlo.convert %arg221 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf16>
      %1214 = stablehlo.convolution(%1073, %1213) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x512xf16>) -> tensor<256x7x7x512xf16>
      %1215 = stablehlo.convert %1214 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1216 = stablehlo.multiply %1215, %1215 : tensor<256x7x7x512xf32>
      %cst_354 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1217 = stablehlo.broadcast_in_dim %cst_354, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %1218 = stablehlo.multiply %1217, %1215 : tensor<256x7x7x512xf32>
      %cst_355 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1219 = stablehlo.reduce(%1215 init: %cst_355) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_356 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1220 = stablehlo.broadcast_in_dim %cst_356, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1221 = stablehlo.divide %1219, %1220 : tensor<512xf32>
      %cst_357 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1222 = stablehlo.reduce(%1216 init: %cst_357) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_358 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1223 = stablehlo.broadcast_in_dim %cst_358, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1224 = stablehlo.divide %1222, %1223 : tensor<512xf32>
      %1225 = stablehlo.broadcast_in_dim %1221, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1226 = stablehlo.broadcast_in_dim %1224, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1227 = stablehlo.concatenate %1225, %1226, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %1228 = "stablehlo.all_reduce"(%1227) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_359 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1229 = stablehlo.broadcast_in_dim %cst_359, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1230 = stablehlo.divide %1228, %1229 : tensor<2x512xf32>
      %1231 = stablehlo.slice %1230 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1232 = stablehlo.reshape %1231 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1233 = stablehlo.slice %1230 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1234 = stablehlo.reshape %1233 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1235 = stablehlo.multiply %1232, %1232 : tensor<512xf32>
      %cst_360 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1236 = stablehlo.broadcast_in_dim %cst_360, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1237 = stablehlo.multiply %1236, %1232 : tensor<512xf32>
      %1238 = stablehlo.subtract %1234, %1235 : tensor<512xf32>
      %cst_361 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1239 = stablehlo.broadcast_in_dim %cst_361, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1240 = stablehlo.maximum %1239, %1238 : tensor<512xf32>
      %1241 = stablehlo.compare  EQ, %1238, %1240,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_362 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1242 = stablehlo.broadcast_in_dim %cst_362, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_363 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1243 = stablehlo.broadcast_in_dim %cst_363, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1244 = stablehlo.select %1241, %1242, %1243 : tensor<512xi1>, tensor<512xf32>
      %cst_364 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1245 = stablehlo.broadcast_in_dim %cst_364, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1246 = stablehlo.compare  EQ, %1245, %1240,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_365 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1247 = stablehlo.broadcast_in_dim %cst_365, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_366 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1248 = stablehlo.broadcast_in_dim %cst_366, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1249 = stablehlo.select %1246, %1247, %1248 : tensor<512xi1>, tensor<512xf32>
      %1250 = stablehlo.divide %1244, %1249 : tensor<512xf32>
      %cst_367 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1251 = stablehlo.broadcast_in_dim %cst_367, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1252 = stablehlo.multiply %1251, %arg328 : tensor<512xf32>
      %cst_368 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1253 = stablehlo.broadcast_in_dim %cst_368, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1254 = stablehlo.multiply %1253, %1232 : tensor<512xf32>
      %1255 = stablehlo.add %1252, %1254 : tensor<512xf32>
      %cst_369 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1256 = stablehlo.broadcast_in_dim %cst_369, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1257 = stablehlo.multiply %1256, %arg329 : tensor<512xf32>
      %cst_370 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1258 = stablehlo.broadcast_in_dim %cst_370, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1259 = stablehlo.multiply %1258, %1240 : tensor<512xf32>
      %1260 = stablehlo.add %1257, %1259 : tensor<512xf32>
      %1261 = stablehlo.broadcast_in_dim %1232, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1262 = stablehlo.broadcast_in_dim %1240, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1263 = stablehlo.convert %1214 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1264 = stablehlo.broadcast_in_dim %1261, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1265 = stablehlo.subtract %1263, %1264 : tensor<256x7x7x512xf32>
      %cst_371 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1266 = stablehlo.broadcast_in_dim %cst_371, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1267 = stablehlo.add %1262, %1266 : tensor<1x1x1x512xf32>
      %1268 = stablehlo.rsqrt %1267 : tensor<1x1x1x512xf32>
      %1269 = stablehlo.divide %1268, %1267 : tensor<1x1x1x512xf32>
      %cst_372 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1270 = stablehlo.broadcast_in_dim %cst_372, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1271 = stablehlo.multiply %1270, %1269 : tensor<1x1x1x512xf32>
      %1272 = stablehlo.reshape %arg223 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1273 = stablehlo.multiply %1268, %1272 : tensor<1x1x1x512xf32>
      %1274 = stablehlo.broadcast_in_dim %1273, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1275 = stablehlo.multiply %1265, %1274 : tensor<256x7x7x512xf32>
      %1276 = stablehlo.reshape %arg222 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1277 = stablehlo.broadcast_in_dim %1276, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1278 = stablehlo.add %1275, %1277 : tensor<256x7x7x512xf32>
      %1279 = stablehlo.convert %1278 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1280 = stablehlo.add %1279, %1212 : tensor<256x7x7x512xf16>
      %1281 = func.call @relu_184(%1280) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_373 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1282 = stablehlo.broadcast_in_dim %cst_373, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %1283 = stablehlo.compare  GT, %1280, %1282,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %1284 = stablehlo.convert %arg228 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %1285 = stablehlo.convolution(%1281, %1284) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %1286 = stablehlo.convert %1285 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1287 = stablehlo.multiply %1286, %1286 : tensor<256x7x7x512xf32>
      %cst_374 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1288 = stablehlo.broadcast_in_dim %cst_374, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %1289 = stablehlo.multiply %1288, %1286 : tensor<256x7x7x512xf32>
      %cst_375 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1290 = stablehlo.reduce(%1286 init: %cst_375) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_376 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1291 = stablehlo.broadcast_in_dim %cst_376, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1292 = stablehlo.divide %1290, %1291 : tensor<512xf32>
      %cst_377 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1293 = stablehlo.reduce(%1287 init: %cst_377) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_378 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1294 = stablehlo.broadcast_in_dim %cst_378, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1295 = stablehlo.divide %1293, %1294 : tensor<512xf32>
      %1296 = stablehlo.broadcast_in_dim %1292, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1297 = stablehlo.broadcast_in_dim %1295, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1298 = stablehlo.concatenate %1296, %1297, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %1299 = "stablehlo.all_reduce"(%1298) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_379 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1300 = stablehlo.broadcast_in_dim %cst_379, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1301 = stablehlo.divide %1299, %1300 : tensor<2x512xf32>
      %1302 = stablehlo.slice %1301 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1303 = stablehlo.reshape %1302 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1304 = stablehlo.slice %1301 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1305 = stablehlo.reshape %1304 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1306 = stablehlo.multiply %1303, %1303 : tensor<512xf32>
      %cst_380 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1307 = stablehlo.broadcast_in_dim %cst_380, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1308 = stablehlo.multiply %1307, %1303 : tensor<512xf32>
      %1309 = stablehlo.subtract %1305, %1306 : tensor<512xf32>
      %cst_381 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1310 = stablehlo.broadcast_in_dim %cst_381, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1311 = stablehlo.maximum %1310, %1309 : tensor<512xf32>
      %1312 = stablehlo.compare  EQ, %1309, %1311,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_382 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1313 = stablehlo.broadcast_in_dim %cst_382, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_383 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1314 = stablehlo.broadcast_in_dim %cst_383, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1315 = stablehlo.select %1312, %1313, %1314 : tensor<512xi1>, tensor<512xf32>
      %cst_384 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1316 = stablehlo.broadcast_in_dim %cst_384, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1317 = stablehlo.compare  EQ, %1316, %1311,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_385 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1318 = stablehlo.broadcast_in_dim %cst_385, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_386 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1319 = stablehlo.broadcast_in_dim %cst_386, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1320 = stablehlo.select %1317, %1318, %1319 : tensor<512xi1>, tensor<512xf32>
      %1321 = stablehlo.divide %1315, %1320 : tensor<512xf32>
      %cst_387 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1322 = stablehlo.broadcast_in_dim %cst_387, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1323 = stablehlo.multiply %1322, %arg330 : tensor<512xf32>
      %cst_388 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1324 = stablehlo.broadcast_in_dim %cst_388, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1325 = stablehlo.multiply %1324, %1303 : tensor<512xf32>
      %1326 = stablehlo.add %1323, %1325 : tensor<512xf32>
      %cst_389 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1327 = stablehlo.broadcast_in_dim %cst_389, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1328 = stablehlo.multiply %1327, %arg331 : tensor<512xf32>
      %cst_390 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1329 = stablehlo.broadcast_in_dim %cst_390, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1330 = stablehlo.multiply %1329, %1311 : tensor<512xf32>
      %1331 = stablehlo.add %1328, %1330 : tensor<512xf32>
      %1332 = stablehlo.broadcast_in_dim %1303, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1333 = stablehlo.broadcast_in_dim %1311, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1334 = stablehlo.convert %1285 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1335 = stablehlo.broadcast_in_dim %1332, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1336 = stablehlo.subtract %1334, %1335 : tensor<256x7x7x512xf32>
      %cst_391 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1337 = stablehlo.broadcast_in_dim %cst_391, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1338 = stablehlo.add %1333, %1337 : tensor<1x1x1x512xf32>
      %1339 = stablehlo.rsqrt %1338 : tensor<1x1x1x512xf32>
      %1340 = stablehlo.divide %1339, %1338 : tensor<1x1x1x512xf32>
      %cst_392 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1341 = stablehlo.broadcast_in_dim %cst_392, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1342 = stablehlo.multiply %1341, %1340 : tensor<1x1x1x512xf32>
      %1343 = stablehlo.reshape %arg225 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1344 = stablehlo.multiply %1339, %1343 : tensor<1x1x1x512xf32>
      %1345 = stablehlo.broadcast_in_dim %1344, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1346 = stablehlo.multiply %1336, %1345 : tensor<256x7x7x512xf32>
      %1347 = stablehlo.reshape %arg224 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1348 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1349 = stablehlo.add %1346, %1348 : tensor<256x7x7x512xf32>
      %1350 = stablehlo.convert %1349 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1351 = func.call @relu_184(%1350) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_393 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1352 = stablehlo.broadcast_in_dim %cst_393, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %1353 = stablehlo.compare  GT, %1350, %1352,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %1354 = stablehlo.convert %arg229 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %1355 = stablehlo.convolution(%1351, %1354) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %1356 = stablehlo.convert %1355 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1357 = stablehlo.multiply %1356, %1356 : tensor<256x7x7x512xf32>
      %cst_394 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1358 = stablehlo.broadcast_in_dim %cst_394, dims = [] : (tensor<f32>) -> tensor<256x7x7x512xf32>
      %1359 = stablehlo.multiply %1358, %1356 : tensor<256x7x7x512xf32>
      %cst_395 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1360 = stablehlo.reduce(%1356 init: %cst_395) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_396 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1361 = stablehlo.broadcast_in_dim %cst_396, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1362 = stablehlo.divide %1360, %1361 : tensor<512xf32>
      %cst_397 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1363 = stablehlo.reduce(%1357 init: %cst_397) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_398 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1364 = stablehlo.broadcast_in_dim %cst_398, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1365 = stablehlo.divide %1363, %1364 : tensor<512xf32>
      %1366 = stablehlo.broadcast_in_dim %1362, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1367 = stablehlo.broadcast_in_dim %1365, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %1368 = stablehlo.concatenate %1366, %1367, dim = 0 : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<2x512xf32>
      %1369 = "stablehlo.all_reduce"(%1368) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %cst_399 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1370 = stablehlo.broadcast_in_dim %cst_399, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1371 = stablehlo.divide %1369, %1370 : tensor<2x512xf32>
      %1372 = stablehlo.slice %1371 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1373 = stablehlo.reshape %1372 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1374 = stablehlo.slice %1371 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1375 = stablehlo.reshape %1374 : (tensor<1x512xf32>) -> tensor<512xf32>
      %1376 = stablehlo.multiply %1373, %1373 : tensor<512xf32>
      %cst_400 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1377 = stablehlo.broadcast_in_dim %cst_400, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1378 = stablehlo.multiply %1377, %1373 : tensor<512xf32>
      %1379 = stablehlo.subtract %1375, %1376 : tensor<512xf32>
      %cst_401 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1380 = stablehlo.broadcast_in_dim %cst_401, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1381 = stablehlo.maximum %1380, %1379 : tensor<512xf32>
      %1382 = stablehlo.compare  EQ, %1379, %1381,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_402 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1383 = stablehlo.broadcast_in_dim %cst_402, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_403 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1384 = stablehlo.broadcast_in_dim %cst_403, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1385 = stablehlo.select %1382, %1383, %1384 : tensor<512xi1>, tensor<512xf32>
      %cst_404 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1386 = stablehlo.broadcast_in_dim %cst_404, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1387 = stablehlo.compare  EQ, %1386, %1381,  FLOAT : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xi1>
      %cst_405 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1388 = stablehlo.broadcast_in_dim %cst_405, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %cst_406 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1389 = stablehlo.broadcast_in_dim %cst_406, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1390 = stablehlo.select %1387, %1388, %1389 : tensor<512xi1>, tensor<512xf32>
      %1391 = stablehlo.divide %1385, %1390 : tensor<512xf32>
      %cst_407 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1392 = stablehlo.broadcast_in_dim %cst_407, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1393 = stablehlo.multiply %1392, %arg332 : tensor<512xf32>
      %cst_408 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1394 = stablehlo.broadcast_in_dim %cst_408, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1395 = stablehlo.multiply %1394, %1373 : tensor<512xf32>
      %1396 = stablehlo.add %1393, %1395 : tensor<512xf32>
      %cst_409 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %1397 = stablehlo.broadcast_in_dim %cst_409, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1398 = stablehlo.multiply %1397, %arg333 : tensor<512xf32>
      %cst_410 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %1399 = stablehlo.broadcast_in_dim %cst_410, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1400 = stablehlo.multiply %1399, %1381 : tensor<512xf32>
      %1401 = stablehlo.add %1398, %1400 : tensor<512xf32>
      %1402 = stablehlo.broadcast_in_dim %1373, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1403 = stablehlo.broadcast_in_dim %1381, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1404 = stablehlo.convert %1355 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1405 = stablehlo.broadcast_in_dim %1402, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1406 = stablehlo.subtract %1404, %1405 : tensor<256x7x7x512xf32>
      %cst_411 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1407 = stablehlo.broadcast_in_dim %cst_411, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1408 = stablehlo.add %1403, %1407 : tensor<1x1x1x512xf32>
      %1409 = stablehlo.rsqrt %1408 : tensor<1x1x1x512xf32>
      %1410 = stablehlo.divide %1409, %1408 : tensor<1x1x1x512xf32>
      %cst_412 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %1411 = stablehlo.broadcast_in_dim %cst_412, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1412 = stablehlo.multiply %1411, %1410 : tensor<1x1x1x512xf32>
      %1413 = stablehlo.reshape %arg227 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1414 = stablehlo.multiply %1409, %1413 : tensor<1x1x1x512xf32>
      %1415 = stablehlo.broadcast_in_dim %1414, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1416 = stablehlo.multiply %1406, %1415 : tensor<256x7x7x512xf32>
      %1417 = stablehlo.reshape %arg226 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1418 = stablehlo.broadcast_in_dim %1417, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1419 = stablehlo.add %1416, %1418 : tensor<256x7x7x512xf32>
      %1420 = stablehlo.convert %1419 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1421 = stablehlo.add %1281, %1420 : tensor<256x7x7x512xf16>
      %1422 = func.call @relu_184(%1421) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %cst_413 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1423 = stablehlo.broadcast_in_dim %cst_413, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %1424 = stablehlo.compare  GT, %1421, %1423,  FLOAT : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xi1>
      %1425 = stablehlo.convert %1422 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_414 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1426 = stablehlo.reduce(%1425 init: %cst_414) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<256x512xf32>
      %cst_415 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
      %1427 = stablehlo.broadcast_in_dim %cst_415, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
      %1428 = stablehlo.divide %1426, %1427 : tensor<256x512xf32>
      %1429 = stablehlo.convert %1428 : (tensor<256x512xf32>) -> tensor<256x512xf16>
      %1430 = stablehlo.convert %arg172 : (tensor<512x1000xf32>) -> tensor<512x1000xf16>
      %1431 = stablehlo.convert %arg171 : (tensor<1000xf32>) -> tensor<1000xf16>
      %1432 = stablehlo.dot_general %1429, %1430, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf16>, tensor<512x1000xf16>) -> tensor<256x1000xf16>
      %1433 = stablehlo.reshape %1431 : (tensor<1000xf16>) -> tensor<1x1000xf16>
      %1434 = stablehlo.broadcast_in_dim %1433, dims = [0, 1] : (tensor<1x1000xf16>) -> tensor<256x1000xf16>
      %1435 = stablehlo.add %1432, %1434 : tensor<256x1000xf16>
      %1436 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
      %1437 = stablehlo.iota dim = 0 : tensor<1000xi32>
      %1438 = stablehlo.reshape %1437 : (tensor<1000xi32>) -> tensor<1x1000xi32>
      %1439 = stablehlo.broadcast_in_dim %1436, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
      %1440 = stablehlo.broadcast_in_dim %1438, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
      %1441 = stablehlo.compare  EQ, %1439, %1440,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
      %cst_416 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1442 = stablehlo.broadcast_in_dim %cst_416, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %cst_417 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1443 = stablehlo.broadcast_in_dim %cst_417, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %1444 = stablehlo.select %1441, %1442, %1443 : tensor<256x1000xi1>, tensor<256x1000xf32>
      %1445 = stablehlo.convert %1444 : tensor<256x1000xf32>
      %1446:2 = func.call @log_softmax(%1435) : (tensor<256x1000xf16>) -> (tensor<256x1000xf16>, tensor<256x1xf16>)
      %cst_418 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1447 = stablehlo.broadcast_in_dim %cst_418, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %1448 = stablehlo.multiply %1447, %arg172 : tensor<512x1000xf32>
      %cst_419 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1449 = stablehlo.broadcast_in_dim %cst_419, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %1450 = stablehlo.multiply %1449, %arg177 : tensor<3x3x64x64xf32>
      %cst_420 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1451 = stablehlo.broadcast_in_dim %cst_420, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %1452 = stablehlo.multiply %1451, %arg178 : tensor<3x3x64x64xf32>
      %cst_421 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1453 = stablehlo.broadcast_in_dim %cst_421, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %1454 = stablehlo.multiply %1453, %arg183 : tensor<3x3x64x64xf32>
      %cst_422 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1455 = stablehlo.broadcast_in_dim %cst_422, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %1456 = stablehlo.multiply %1455, %arg184 : tensor<3x3x64x64xf32>
      %cst_423 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1457 = stablehlo.broadcast_in_dim %cst_423, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %1458 = stablehlo.multiply %1457, %arg189 : tensor<3x3x64x128xf32>
      %cst_424 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1459 = stablehlo.broadcast_in_dim %cst_424, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %1460 = stablehlo.multiply %1459, %arg190 : tensor<3x3x128x128xf32>
      %cst_425 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1461 = stablehlo.broadcast_in_dim %cst_425, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %1462 = stablehlo.multiply %1461, %arg191 : tensor<1x1x64x128xf32>
      %cst_426 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1463 = stablehlo.broadcast_in_dim %cst_426, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %1464 = stablehlo.multiply %1463, %arg198 : tensor<3x3x128x128xf32>
      %cst_427 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1465 = stablehlo.broadcast_in_dim %cst_427, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %1466 = stablehlo.multiply %1465, %arg199 : tensor<3x3x128x128xf32>
      %cst_428 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1467 = stablehlo.broadcast_in_dim %cst_428, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %1468 = stablehlo.multiply %1467, %arg204 : tensor<3x3x128x256xf32>
      %cst_429 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1469 = stablehlo.broadcast_in_dim %cst_429, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %1470 = stablehlo.multiply %1469, %arg205 : tensor<3x3x256x256xf32>
      %cst_430 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1471 = stablehlo.broadcast_in_dim %cst_430, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %1472 = stablehlo.multiply %1471, %arg206 : tensor<1x1x128x256xf32>
      %cst_431 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1473 = stablehlo.broadcast_in_dim %cst_431, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %1474 = stablehlo.multiply %1473, %arg213 : tensor<3x3x256x256xf32>
      %cst_432 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1475 = stablehlo.broadcast_in_dim %cst_432, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %1476 = stablehlo.multiply %1475, %arg214 : tensor<3x3x256x256xf32>
      %cst_433 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1477 = stablehlo.broadcast_in_dim %cst_433, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %1478 = stablehlo.multiply %1477, %arg219 : tensor<3x3x256x512xf32>
      %cst_434 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1479 = stablehlo.broadcast_in_dim %cst_434, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %1480 = stablehlo.multiply %1479, %arg220 : tensor<3x3x512x512xf32>
      %cst_435 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1481 = stablehlo.broadcast_in_dim %cst_435, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %1482 = stablehlo.multiply %1481, %arg221 : tensor<1x1x256x512xf32>
      %cst_436 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1483 = stablehlo.broadcast_in_dim %cst_436, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %1484 = stablehlo.multiply %1483, %arg228 : tensor<3x3x512x512xf32>
      %cst_437 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1485 = stablehlo.broadcast_in_dim %cst_437, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %1486 = stablehlo.multiply %1485, %arg229 : tensor<3x3x512x512xf32>
      %cst_438 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1487 = stablehlo.broadcast_in_dim %cst_438, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %1488 = stablehlo.multiply %1487, %arg232 : tensor<7x7x3x64xf32>
      %1489 = stablehlo.convert %arg337 : tensor<f32>
      %cst_439 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1490 = stablehlo.multiply %1489, %cst_439 : tensor<f32>
      %cst_440 = stablehlo.constant dense<5.000000e-05> : tensor<f32>
      %1491 = stablehlo.multiply %cst_440, %1490 : tensor<f32>
      %1492 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %1493 = stablehlo.multiply %1492, %1488 : tensor<7x7x3x64xf32>
      %1494 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %1495 = stablehlo.multiply %1494, %1486 : tensor<3x3x512x512xf32>
      %1496 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %1497 = stablehlo.multiply %1496, %1484 : tensor<3x3x512x512xf32>
      %1498 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %1499 = stablehlo.multiply %1498, %1482 : tensor<1x1x256x512xf32>
      %1500 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %1501 = stablehlo.multiply %1500, %1480 : tensor<3x3x512x512xf32>
      %1502 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %1503 = stablehlo.multiply %1502, %1478 : tensor<3x3x256x512xf32>
      %1504 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %1505 = stablehlo.multiply %1504, %1476 : tensor<3x3x256x256xf32>
      %1506 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %1507 = stablehlo.multiply %1506, %1474 : tensor<3x3x256x256xf32>
      %1508 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %1509 = stablehlo.multiply %1508, %1472 : tensor<1x1x128x256xf32>
      %1510 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %1511 = stablehlo.multiply %1510, %1470 : tensor<3x3x256x256xf32>
      %1512 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %1513 = stablehlo.multiply %1512, %1468 : tensor<3x3x128x256xf32>
      %1514 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %1515 = stablehlo.multiply %1514, %1466 : tensor<3x3x128x128xf32>
      %1516 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %1517 = stablehlo.multiply %1516, %1464 : tensor<3x3x128x128xf32>
      %1518 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %1519 = stablehlo.multiply %1518, %1462 : tensor<1x1x64x128xf32>
      %1520 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %1521 = stablehlo.multiply %1520, %1460 : tensor<3x3x128x128xf32>
      %1522 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %1523 = stablehlo.multiply %1522, %1458 : tensor<3x3x64x128xf32>
      %1524 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %1525 = stablehlo.multiply %1524, %1456 : tensor<3x3x64x64xf32>
      %1526 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %1527 = stablehlo.multiply %1526, %1454 : tensor<3x3x64x64xf32>
      %1528 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %1529 = stablehlo.multiply %1528, %1452 : tensor<3x3x64x64xf32>
      %1530 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %1531 = stablehlo.multiply %1530, %1450 : tensor<3x3x64x64xf32>
      %1532 = stablehlo.broadcast_in_dim %1491, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %1533 = stablehlo.multiply %1532, %1448 : tensor<512x1000xf32>
      %cst_441 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %1534 = stablehlo.divide %1490, %cst_441 : tensor<f32>
      %1535 = stablehlo.broadcast_in_dim %1534, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1536 = stablehlo.negate %1535 : tensor<256xf32>
      %1537 = stablehlo.broadcast_in_dim %1536, dims = [0] : (tensor<256xf32>) -> tensor<256x1000xf32>
      %1538 = stablehlo.multiply %1445, %1537 : tensor<256x1000xf32>
      %1539 = stablehlo.convert %1538 : (tensor<256x1000xf32>) -> tensor<256x1000xf16>
      %1540 = func.call @log_softmax_269(%1446#0, %1446#1, %1539) : (tensor<256x1000xf16>, tensor<256x1xf16>, tensor<256x1000xf16>) -> tensor<256x1000xf16>
      %cst_442 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1541 = stablehlo.reduce(%1540 init: %cst_442) applies stablehlo.add across dimensions = [0] : (tensor<256x1000xf16>, tensor<f16>) -> tensor<1000xf16>
      %1542 = stablehlo.reshape %1541 : (tensor<1000xf16>) -> tensor<1x1000xf16>
      %1543 = stablehlo.reshape %1542 : (tensor<1x1000xf16>) -> tensor<1000xf16>
      %1544 = stablehlo.dot_general %1540, %1429, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x1000xf16>, tensor<256x512xf16>) -> tensor<1000x512xf16>
      %1545 = stablehlo.transpose %1544, dims = [1, 0] : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
      %1546 = stablehlo.dot_general %1540, %1430, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1000xf16>, tensor<512x1000xf16>) -> tensor<256x512xf16>
      %1547 = stablehlo.convert %1543 : (tensor<1000xf16>) -> tensor<1000xf32>
      %1548 = stablehlo.convert %1545 : (tensor<512x1000xf16>) -> tensor<512x1000xf32>
      %1549 = stablehlo.add %1533, %1548 : tensor<512x1000xf32>
      %1550 = stablehlo.convert %1546 : (tensor<256x512xf16>) -> tensor<256x512xf32>
      %cst_443 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
      %1551 = stablehlo.broadcast_in_dim %cst_443, dims = [] : (tensor<f32>) -> tensor<256x512xf32>
      %1552 = stablehlo.divide %1550, %1551 : tensor<256x512xf32>
      %1553 = stablehlo.broadcast_in_dim %1552, dims = [0, 3] : (tensor<256x512xf32>) -> tensor<256x7x7x512xf32>
      %1554 = stablehlo.convert %1553 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_444 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1555 = stablehlo.broadcast_in_dim %cst_444, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %1556 = stablehlo.select %1424, %1554, %1555 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %1557 = stablehlo.convert %1556 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_445 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1558 = stablehlo.reduce(%1557 init: %cst_445) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1559 = stablehlo.reshape %1558 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1560 = stablehlo.reshape %1559 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1561 = stablehlo.multiply %1406, %1557 : tensor<256x7x7x512xf32>
      %cst_446 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1562 = stablehlo.reduce(%1561 init: %cst_446) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1563 = stablehlo.reshape %1562 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1564 = stablehlo.broadcast_in_dim %1414, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1565 = stablehlo.multiply %1557, %1564 : tensor<256x7x7x512xf32>
      %1566 = stablehlo.multiply %1409, %1563 : tensor<1x1x1x512xf32>
      %1567 = stablehlo.multiply %1563, %1413 : tensor<1x1x1x512xf32>
      %1568 = stablehlo.reshape %1566 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1569 = stablehlo.multiply %1567, %1412 : tensor<1x1x1x512xf32>
      %1570 = stablehlo.negate %1565 : tensor<256x7x7x512xf32>
      %cst_447 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1571 = stablehlo.reduce(%1570 init: %cst_447) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1572 = stablehlo.reshape %1571 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1573 = stablehlo.convert %1565 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_448 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1574 = stablehlo.reduce(%1569 init: %cst_448) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_449 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1575 = stablehlo.reduce(%1572 init: %cst_449) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1576 = stablehlo.multiply %1574, %1391 : tensor<512xf32>
      %1577 = stablehlo.negate %1576 : tensor<512xf32>
      %1578 = stablehlo.multiply %1577, %1378 : tensor<512xf32>
      %1579 = stablehlo.add %1575, %1578 : tensor<512xf32>
      %1580 = stablehlo.broadcast_in_dim %1576, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_450 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1581 = stablehlo.pad %1580, %cst_450, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1582 = stablehlo.broadcast_in_dim %1579, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_451 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1583 = stablehlo.pad %1582, %cst_451, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1584 = stablehlo.add %1581, %1583 : tensor<2x512xf32>
      %cst_452 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1585 = stablehlo.broadcast_in_dim %cst_452, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1586 = stablehlo.divide %1584, %1585 : tensor<2x512xf32>
      %1587 = "stablehlo.all_reduce"(%1586) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %1588 = stablehlo.slice %1587 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1589 = stablehlo.slice %1587 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_453 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1590 = stablehlo.reduce(%1589 init: %cst_453) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_454 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1591 = stablehlo.reduce(%1588 init: %cst_454) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_455 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1592 = stablehlo.broadcast_in_dim %cst_455, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1593 = stablehlo.divide %1590, %1592 : tensor<512xf32>
      %1594 = stablehlo.broadcast_in_dim %1593, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_456 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1595 = stablehlo.broadcast_in_dim %cst_456, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1596 = stablehlo.divide %1591, %1595 : tensor<512xf32>
      %1597 = stablehlo.broadcast_in_dim %1596, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %1598 = stablehlo.multiply %1594, %1359 : tensor<256x7x7x512xf32>
      %1599 = stablehlo.add %1597, %1598 : tensor<256x7x7x512xf32>
      %1600 = stablehlo.convert %1599 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1601 = stablehlo.add %1573, %1600 : tensor<256x7x7x512xf16>
      %1602 = stablehlo.convolution(%1351, %1601) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x512x512xf16>
      %1603 = stablehlo.reverse %1354, dims = [0, 1] : tensor<3x3x512x512xf16>
      %1604 = stablehlo.convolution(%1601, %1603) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %1605 = stablehlo.convert %1602 : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf32>
      %1606 = stablehlo.add %1495, %1605 : tensor<3x3x512x512xf32>
      %cst_457 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1607 = stablehlo.broadcast_in_dim %cst_457, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %1608 = stablehlo.select %1353, %1604, %1607 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %1609 = stablehlo.convert %1608 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_458 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1610 = stablehlo.reduce(%1609 init: %cst_458) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1611 = stablehlo.reshape %1610 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1612 = stablehlo.reshape %1611 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1613 = stablehlo.multiply %1336, %1609 : tensor<256x7x7x512xf32>
      %cst_459 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1614 = stablehlo.reduce(%1613 init: %cst_459) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1615 = stablehlo.reshape %1614 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1616 = stablehlo.broadcast_in_dim %1344, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1617 = stablehlo.multiply %1609, %1616 : tensor<256x7x7x512xf32>
      %1618 = stablehlo.multiply %1339, %1615 : tensor<1x1x1x512xf32>
      %1619 = stablehlo.multiply %1615, %1343 : tensor<1x1x1x512xf32>
      %1620 = stablehlo.reshape %1618 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1621 = stablehlo.multiply %1619, %1342 : tensor<1x1x1x512xf32>
      %1622 = stablehlo.negate %1617 : tensor<256x7x7x512xf32>
      %cst_460 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1623 = stablehlo.reduce(%1622 init: %cst_460) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1624 = stablehlo.reshape %1623 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1625 = stablehlo.convert %1617 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_461 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1626 = stablehlo.reduce(%1621 init: %cst_461) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_462 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1627 = stablehlo.reduce(%1624 init: %cst_462) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1628 = stablehlo.multiply %1626, %1321 : tensor<512xf32>
      %1629 = stablehlo.negate %1628 : tensor<512xf32>
      %1630 = stablehlo.multiply %1629, %1308 : tensor<512xf32>
      %1631 = stablehlo.add %1627, %1630 : tensor<512xf32>
      %1632 = stablehlo.broadcast_in_dim %1628, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_463 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1633 = stablehlo.pad %1632, %cst_463, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1634 = stablehlo.broadcast_in_dim %1631, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_464 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1635 = stablehlo.pad %1634, %cst_464, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1636 = stablehlo.add %1633, %1635 : tensor<2x512xf32>
      %cst_465 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1637 = stablehlo.broadcast_in_dim %cst_465, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1638 = stablehlo.divide %1636, %1637 : tensor<2x512xf32>
      %1639 = "stablehlo.all_reduce"(%1638) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %1640 = stablehlo.slice %1639 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1641 = stablehlo.slice %1639 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_466 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1642 = stablehlo.reduce(%1641 init: %cst_466) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_467 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1643 = stablehlo.reduce(%1640 init: %cst_467) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_468 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1644 = stablehlo.broadcast_in_dim %cst_468, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1645 = stablehlo.divide %1642, %1644 : tensor<512xf32>
      %1646 = stablehlo.broadcast_in_dim %1645, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_469 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1647 = stablehlo.broadcast_in_dim %cst_469, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1648 = stablehlo.divide %1643, %1647 : tensor<512xf32>
      %1649 = stablehlo.broadcast_in_dim %1648, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %1650 = stablehlo.multiply %1646, %1289 : tensor<256x7x7x512xf32>
      %1651 = stablehlo.add %1649, %1650 : tensor<256x7x7x512xf32>
      %1652 = stablehlo.convert %1651 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1653 = stablehlo.add %1625, %1652 : tensor<256x7x7x512xf16>
      %1654 = stablehlo.convolution(%1281, %1653) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x512x512xf16>
      %1655 = stablehlo.reverse %1284, dims = [0, 1] : tensor<3x3x512x512xf16>
      %1656 = stablehlo.convolution(%1653, %1655) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %1657 = stablehlo.add %1556, %1656 : tensor<256x7x7x512xf16>
      %1658 = stablehlo.convert %1654 : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf32>
      %1659 = stablehlo.add %1497, %1658 : tensor<3x3x512x512xf32>
      %cst_470 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1660 = stablehlo.broadcast_in_dim %cst_470, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %1661 = stablehlo.select %1283, %1657, %1660 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %1662 = stablehlo.convert %1661 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_471 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1663 = stablehlo.reduce(%1662 init: %cst_471) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1664 = stablehlo.reshape %1663 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1665 = stablehlo.reshape %1664 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1666 = stablehlo.multiply %1265, %1662 : tensor<256x7x7x512xf32>
      %cst_472 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1667 = stablehlo.reduce(%1666 init: %cst_472) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1668 = stablehlo.reshape %1667 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1669 = stablehlo.broadcast_in_dim %1273, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1670 = stablehlo.multiply %1662, %1669 : tensor<256x7x7x512xf32>
      %1671 = stablehlo.multiply %1268, %1668 : tensor<1x1x1x512xf32>
      %1672 = stablehlo.multiply %1668, %1272 : tensor<1x1x1x512xf32>
      %1673 = stablehlo.reshape %1671 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1674 = stablehlo.multiply %1672, %1271 : tensor<1x1x1x512xf32>
      %1675 = stablehlo.negate %1670 : tensor<256x7x7x512xf32>
      %cst_473 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1676 = stablehlo.reduce(%1675 init: %cst_473) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1677 = stablehlo.reshape %1676 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1678 = stablehlo.convert %1670 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_474 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1679 = stablehlo.reduce(%1674 init: %cst_474) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_475 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1680 = stablehlo.reduce(%1677 init: %cst_475) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1681 = stablehlo.multiply %1679, %1250 : tensor<512xf32>
      %1682 = stablehlo.negate %1681 : tensor<512xf32>
      %1683 = stablehlo.multiply %1682, %1237 : tensor<512xf32>
      %1684 = stablehlo.add %1680, %1683 : tensor<512xf32>
      %1685 = stablehlo.broadcast_in_dim %1681, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_476 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1686 = stablehlo.pad %1685, %cst_476, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1687 = stablehlo.broadcast_in_dim %1684, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_477 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1688 = stablehlo.pad %1687, %cst_477, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1689 = stablehlo.add %1686, %1688 : tensor<2x512xf32>
      %cst_478 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1690 = stablehlo.broadcast_in_dim %cst_478, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1691 = stablehlo.divide %1689, %1690 : tensor<2x512xf32>
      %1692 = "stablehlo.all_reduce"(%1691) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %1693 = stablehlo.slice %1692 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1694 = stablehlo.slice %1692 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_479 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1695 = stablehlo.reduce(%1694 init: %cst_479) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_480 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1696 = stablehlo.reduce(%1693 init: %cst_480) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_481 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1697 = stablehlo.broadcast_in_dim %cst_481, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1698 = stablehlo.divide %1695, %1697 : tensor<512xf32>
      %1699 = stablehlo.broadcast_in_dim %1698, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_482 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1700 = stablehlo.broadcast_in_dim %cst_482, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1701 = stablehlo.divide %1696, %1700 : tensor<512xf32>
      %1702 = stablehlo.broadcast_in_dim %1701, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %1703 = stablehlo.multiply %1699, %1218 : tensor<256x7x7x512xf32>
      %1704 = stablehlo.add %1702, %1703 : tensor<256x7x7x512xf32>
      %1705 = stablehlo.convert %1704 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1706 = stablehlo.add %1678, %1705 : tensor<256x7x7x512xf16>
      %1707 = stablehlo.convolution(%1073, %1706) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x7x7x512xf16>) -> tensor<1x1x256x512xf16>
      %1708 = stablehlo.reverse %1213, dims = [0, 1] : tensor<1x1x256x512xf16>
      %1709 = stablehlo.convolution(%1706, %1708) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x256x512xf16>) -> tensor<256x14x14x256xf16>
      %1710 = stablehlo.convert %1707 : (tensor<1x1x256x512xf16>) -> tensor<1x1x256x512xf32>
      %1711 = stablehlo.add %1499, %1710 : tensor<1x1x256x512xf32>
      %1712 = stablehlo.convert %1661 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_483 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1713 = stablehlo.reduce(%1712 init: %cst_483) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1714 = stablehlo.reshape %1713 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1715 = stablehlo.reshape %1714 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1716 = stablehlo.multiply %1198, %1712 : tensor<256x7x7x512xf32>
      %cst_484 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1717 = stablehlo.reduce(%1716 init: %cst_484) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1718 = stablehlo.reshape %1717 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1719 = stablehlo.broadcast_in_dim %1206, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1720 = stablehlo.multiply %1712, %1719 : tensor<256x7x7x512xf32>
      %1721 = stablehlo.multiply %1201, %1718 : tensor<1x1x1x512xf32>
      %1722 = stablehlo.multiply %1718, %1205 : tensor<1x1x1x512xf32>
      %1723 = stablehlo.reshape %1721 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1724 = stablehlo.multiply %1722, %1204 : tensor<1x1x1x512xf32>
      %1725 = stablehlo.negate %1720 : tensor<256x7x7x512xf32>
      %cst_485 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1726 = stablehlo.reduce(%1725 init: %cst_485) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1727 = stablehlo.reshape %1726 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1728 = stablehlo.convert %1720 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_486 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1729 = stablehlo.reduce(%1724 init: %cst_486) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_487 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1730 = stablehlo.reduce(%1727 init: %cst_487) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1731 = stablehlo.multiply %1729, %1183 : tensor<512xf32>
      %1732 = stablehlo.negate %1731 : tensor<512xf32>
      %1733 = stablehlo.multiply %1732, %1170 : tensor<512xf32>
      %1734 = stablehlo.add %1730, %1733 : tensor<512xf32>
      %1735 = stablehlo.broadcast_in_dim %1731, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_488 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1736 = stablehlo.pad %1735, %cst_488, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1737 = stablehlo.broadcast_in_dim %1734, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_489 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1738 = stablehlo.pad %1737, %cst_489, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1739 = stablehlo.add %1736, %1738 : tensor<2x512xf32>
      %cst_490 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1740 = stablehlo.broadcast_in_dim %cst_490, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1741 = stablehlo.divide %1739, %1740 : tensor<2x512xf32>
      %1742 = "stablehlo.all_reduce"(%1741) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %1743 = stablehlo.slice %1742 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1744 = stablehlo.slice %1742 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_491 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1745 = stablehlo.reduce(%1744 init: %cst_491) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_492 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1746 = stablehlo.reduce(%1743 init: %cst_492) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_493 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1747 = stablehlo.broadcast_in_dim %cst_493, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1748 = stablehlo.divide %1745, %1747 : tensor<512xf32>
      %1749 = stablehlo.broadcast_in_dim %1748, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_494 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1750 = stablehlo.broadcast_in_dim %cst_494, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1751 = stablehlo.divide %1746, %1750 : tensor<512xf32>
      %1752 = stablehlo.broadcast_in_dim %1751, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %1753 = stablehlo.multiply %1749, %1151 : tensor<256x7x7x512xf32>
      %1754 = stablehlo.add %1752, %1753 : tensor<256x7x7x512xf32>
      %1755 = stablehlo.convert %1754 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1756 = stablehlo.add %1728, %1755 : tensor<256x7x7x512xf16>
      %1757 = stablehlo.convolution(%1143, %1756) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x512x512xf16>
      %1758 = stablehlo.reverse %1146, dims = [0, 1] : tensor<3x3x512x512xf16>
      %1759 = stablehlo.convolution(%1756, %1758) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %1760 = stablehlo.convert %1757 : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf32>
      %1761 = stablehlo.add %1501, %1760 : tensor<3x3x512x512xf32>
      %cst_495 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1762 = stablehlo.broadcast_in_dim %cst_495, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
      %1763 = stablehlo.select %1145, %1759, %1762 : tensor<256x7x7x512xi1>, tensor<256x7x7x512xf16>
      %1764 = stablehlo.convert %1763 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %cst_496 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1765 = stablehlo.reduce(%1764 init: %cst_496) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1766 = stablehlo.reshape %1765 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1767 = stablehlo.reshape %1766 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1768 = stablehlo.multiply %1128, %1764 : tensor<256x7x7x512xf32>
      %cst_497 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1769 = stablehlo.reduce(%1768 init: %cst_497) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1770 = stablehlo.reshape %1769 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1771 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1772 = stablehlo.multiply %1764, %1771 : tensor<256x7x7x512xf32>
      %1773 = stablehlo.multiply %1131, %1770 : tensor<1x1x1x512xf32>
      %1774 = stablehlo.multiply %1770, %1135 : tensor<1x1x1x512xf32>
      %1775 = stablehlo.reshape %1773 : (tensor<1x1x1x512xf32>) -> tensor<512xf32>
      %1776 = stablehlo.multiply %1774, %1134 : tensor<1x1x1x512xf32>
      %1777 = stablehlo.negate %1772 : tensor<256x7x7x512xf32>
      %cst_498 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1778 = stablehlo.reduce(%1777 init: %cst_498) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x7x7x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1779 = stablehlo.reshape %1778 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1780 = stablehlo.convert %1772 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %cst_499 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1781 = stablehlo.reduce(%1776 init: %cst_499) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_500 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1782 = stablehlo.reduce(%1779 init: %cst_500) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %1783 = stablehlo.multiply %1781, %1113 : tensor<512xf32>
      %1784 = stablehlo.negate %1783 : tensor<512xf32>
      %1785 = stablehlo.multiply %1784, %1100 : tensor<512xf32>
      %1786 = stablehlo.add %1782, %1785 : tensor<512xf32>
      %1787 = stablehlo.broadcast_in_dim %1783, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_501 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1788 = stablehlo.pad %1787, %cst_501, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1789 = stablehlo.broadcast_in_dim %1786, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
      %cst_502 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1790 = stablehlo.pad %1789, %cst_502, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<2x512xf32>
      %1791 = stablehlo.add %1788, %1790 : tensor<2x512xf32>
      %cst_503 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1792 = stablehlo.broadcast_in_dim %cst_503, dims = [] : (tensor<f32>) -> tensor<2x512xf32>
      %1793 = stablehlo.divide %1791, %1792 : tensor<2x512xf32>
      %1794 = "stablehlo.all_reduce"(%1793) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x512xf32>) -> tensor<2x512xf32>
      %1795 = stablehlo.slice %1794 [0:1, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %1796 = stablehlo.slice %1794 [1:2, 0:512] : (tensor<2x512xf32>) -> tensor<1x512xf32>
      %cst_504 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1797 = stablehlo.reduce(%1796 init: %cst_504) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_505 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1798 = stablehlo.reduce(%1795 init: %cst_505) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_506 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1799 = stablehlo.broadcast_in_dim %cst_506, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1800 = stablehlo.divide %1797, %1799 : tensor<512xf32>
      %1801 = stablehlo.broadcast_in_dim %1800, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %cst_507 = stablehlo.constant dense<1.254400e+04> : tensor<f32>
      %1802 = stablehlo.broadcast_in_dim %cst_507, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %1803 = stablehlo.divide %1798, %1802 : tensor<512xf32>
      %1804 = stablehlo.broadcast_in_dim %1803, dims = [3] : (tensor<512xf32>) -> tensor<256x7x7x512xf32>
      %1805 = stablehlo.multiply %1801, %1081 : tensor<256x7x7x512xf32>
      %1806 = stablehlo.add %1804, %1805 : tensor<256x7x7x512xf32>
      %1807 = stablehlo.convert %1806 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1808 = stablehlo.add %1780, %1807 : tensor<256x7x7x512xf16>
      %1809 = stablehlo.convolution(%1073, %1808) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x7x7x512xf16>) -> tensor<3x3x256x512xf16>
      %1810 = stablehlo.reverse %1076, dims = [0, 1] : tensor<3x3x256x512xf16>
      %1811 = stablehlo.convolution(%1808, %1810) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x256x512xf16>) -> tensor<256x14x14x256xf16>
      %1812 = stablehlo.add %1709, %1811 : tensor<256x14x14x256xf16>
      %1813 = stablehlo.convert %1809 : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf32>
      %1814 = stablehlo.add %1503, %1813 : tensor<3x3x256x512xf32>
      %cst_508 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1815 = stablehlo.broadcast_in_dim %cst_508, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1816 = stablehlo.select %1075, %1812, %1815 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %1817 = stablehlo.convert %1816 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_509 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1818 = stablehlo.reduce(%1817 init: %cst_509) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1819 = stablehlo.reshape %1818 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1820 = stablehlo.reshape %1819 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %1821 = stablehlo.multiply %1057, %1817 : tensor<256x14x14x256xf32>
      %cst_510 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1822 = stablehlo.reduce(%1821 init: %cst_510) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1823 = stablehlo.reshape %1822 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1824 = stablehlo.broadcast_in_dim %1065, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1825 = stablehlo.multiply %1817, %1824 : tensor<256x14x14x256xf32>
      %1826 = stablehlo.multiply %1060, %1823 : tensor<1x1x1x256xf32>
      %1827 = stablehlo.multiply %1823, %1064 : tensor<1x1x1x256xf32>
      %1828 = stablehlo.reshape %1826 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %1829 = stablehlo.multiply %1827, %1063 : tensor<1x1x1x256xf32>
      %1830 = stablehlo.negate %1825 : tensor<256x14x14x256xf32>
      %cst_511 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1831 = stablehlo.reduce(%1830 init: %cst_511) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1832 = stablehlo.reshape %1831 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1833 = stablehlo.convert %1825 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_512 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1834 = stablehlo.reduce(%1829 init: %cst_512) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_513 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1835 = stablehlo.reduce(%1832 init: %cst_513) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1836 = stablehlo.multiply %1834, %1042 : tensor<256xf32>
      %1837 = stablehlo.negate %1836 : tensor<256xf32>
      %1838 = stablehlo.multiply %1837, %1029 : tensor<256xf32>
      %1839 = stablehlo.add %1835, %1838 : tensor<256xf32>
      %1840 = stablehlo.broadcast_in_dim %1836, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_514 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1841 = stablehlo.pad %1840, %cst_514, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %1842 = stablehlo.broadcast_in_dim %1839, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_515 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1843 = stablehlo.pad %1842, %cst_515, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %1844 = stablehlo.add %1841, %1843 : tensor<2x256xf32>
      %cst_516 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1845 = stablehlo.broadcast_in_dim %cst_516, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1846 = stablehlo.divide %1844, %1845 : tensor<2x256xf32>
      %1847 = "stablehlo.all_reduce"(%1846) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %1848 = stablehlo.slice %1847 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1849 = stablehlo.slice %1847 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_517 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1850 = stablehlo.reduce(%1849 init: %cst_517) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_518 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1851 = stablehlo.reduce(%1848 init: %cst_518) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_519 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1852 = stablehlo.broadcast_in_dim %cst_519, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1853 = stablehlo.divide %1850, %1852 : tensor<256xf32>
      %1854 = stablehlo.broadcast_in_dim %1853, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_520 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1855 = stablehlo.broadcast_in_dim %cst_520, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1856 = stablehlo.divide %1851, %1855 : tensor<256xf32>
      %1857 = stablehlo.broadcast_in_dim %1856, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %1858 = stablehlo.multiply %1854, %1010 : tensor<256x14x14x256xf32>
      %1859 = stablehlo.add %1857, %1858 : tensor<256x14x14x256xf32>
      %1860 = stablehlo.convert %1859 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1861 = stablehlo.add %1833, %1860 : tensor<256x14x14x256xf16>
      %1862 = stablehlo.convolution(%1002, %1861) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %1863 = stablehlo.reverse %1005, dims = [0, 1] : tensor<3x3x256x256xf16>
      %1864 = stablehlo.convolution(%1861, %1863) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1865 = stablehlo.convert %1862 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %1866 = stablehlo.add %1505, %1865 : tensor<3x3x256x256xf32>
      %cst_521 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1867 = stablehlo.broadcast_in_dim %cst_521, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1868 = stablehlo.select %1004, %1864, %1867 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %1869 = stablehlo.convert %1868 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_522 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1870 = stablehlo.reduce(%1869 init: %cst_522) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1871 = stablehlo.reshape %1870 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1872 = stablehlo.reshape %1871 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %1873 = stablehlo.multiply %987, %1869 : tensor<256x14x14x256xf32>
      %cst_523 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1874 = stablehlo.reduce(%1873 init: %cst_523) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1875 = stablehlo.reshape %1874 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1876 = stablehlo.broadcast_in_dim %995, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1877 = stablehlo.multiply %1869, %1876 : tensor<256x14x14x256xf32>
      %1878 = stablehlo.multiply %990, %1875 : tensor<1x1x1x256xf32>
      %1879 = stablehlo.multiply %1875, %994 : tensor<1x1x1x256xf32>
      %1880 = stablehlo.reshape %1878 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %1881 = stablehlo.multiply %1879, %993 : tensor<1x1x1x256xf32>
      %1882 = stablehlo.negate %1877 : tensor<256x14x14x256xf32>
      %cst_524 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1883 = stablehlo.reduce(%1882 init: %cst_524) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1884 = stablehlo.reshape %1883 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1885 = stablehlo.convert %1877 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_525 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1886 = stablehlo.reduce(%1881 init: %cst_525) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_526 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1887 = stablehlo.reduce(%1884 init: %cst_526) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1888 = stablehlo.multiply %1886, %972 : tensor<256xf32>
      %1889 = stablehlo.negate %1888 : tensor<256xf32>
      %1890 = stablehlo.multiply %1889, %959 : tensor<256xf32>
      %1891 = stablehlo.add %1887, %1890 : tensor<256xf32>
      %1892 = stablehlo.broadcast_in_dim %1888, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_527 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1893 = stablehlo.pad %1892, %cst_527, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %1894 = stablehlo.broadcast_in_dim %1891, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_528 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1895 = stablehlo.pad %1894, %cst_528, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %1896 = stablehlo.add %1893, %1895 : tensor<2x256xf32>
      %cst_529 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1897 = stablehlo.broadcast_in_dim %cst_529, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1898 = stablehlo.divide %1896, %1897 : tensor<2x256xf32>
      %1899 = "stablehlo.all_reduce"(%1898) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %1900 = stablehlo.slice %1899 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1901 = stablehlo.slice %1899 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_530 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1902 = stablehlo.reduce(%1901 init: %cst_530) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_531 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1903 = stablehlo.reduce(%1900 init: %cst_531) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_532 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1904 = stablehlo.broadcast_in_dim %cst_532, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1905 = stablehlo.divide %1902, %1904 : tensor<256xf32>
      %1906 = stablehlo.broadcast_in_dim %1905, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_533 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1907 = stablehlo.broadcast_in_dim %cst_533, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1908 = stablehlo.divide %1903, %1907 : tensor<256xf32>
      %1909 = stablehlo.broadcast_in_dim %1908, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %1910 = stablehlo.multiply %1906, %940 : tensor<256x14x14x256xf32>
      %1911 = stablehlo.add %1909, %1910 : tensor<256x14x14x256xf32>
      %1912 = stablehlo.convert %1911 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1913 = stablehlo.add %1885, %1912 : tensor<256x14x14x256xf16>
      %1914 = stablehlo.convolution(%932, %1913) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %1915 = stablehlo.reverse %935, dims = [0, 1] : tensor<3x3x256x256xf16>
      %1916 = stablehlo.convolution(%1913, %1915) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1917 = stablehlo.add %1816, %1916 : tensor<256x14x14x256xf16>
      %1918 = stablehlo.convert %1914 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %1919 = stablehlo.add %1507, %1918 : tensor<3x3x256x256xf32>
      %cst_534 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %1920 = stablehlo.broadcast_in_dim %cst_534, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %1921 = stablehlo.select %934, %1917, %1920 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %1922 = stablehlo.convert %1921 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_535 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1923 = stablehlo.reduce(%1922 init: %cst_535) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1924 = stablehlo.reshape %1923 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1925 = stablehlo.reshape %1924 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %1926 = stablehlo.multiply %916, %1922 : tensor<256x14x14x256xf32>
      %cst_536 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1927 = stablehlo.reduce(%1926 init: %cst_536) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1928 = stablehlo.reshape %1927 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1929 = stablehlo.broadcast_in_dim %924, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1930 = stablehlo.multiply %1922, %1929 : tensor<256x14x14x256xf32>
      %1931 = stablehlo.multiply %919, %1928 : tensor<1x1x1x256xf32>
      %1932 = stablehlo.multiply %1928, %923 : tensor<1x1x1x256xf32>
      %1933 = stablehlo.reshape %1931 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %1934 = stablehlo.multiply %1932, %922 : tensor<1x1x1x256xf32>
      %1935 = stablehlo.negate %1930 : tensor<256x14x14x256xf32>
      %cst_537 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1936 = stablehlo.reduce(%1935 init: %cst_537) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1937 = stablehlo.reshape %1936 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1938 = stablehlo.convert %1930 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_538 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1939 = stablehlo.reduce(%1934 init: %cst_538) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_539 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1940 = stablehlo.reduce(%1937 init: %cst_539) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1941 = stablehlo.multiply %1939, %901 : tensor<256xf32>
      %1942 = stablehlo.negate %1941 : tensor<256xf32>
      %1943 = stablehlo.multiply %1942, %888 : tensor<256xf32>
      %1944 = stablehlo.add %1940, %1943 : tensor<256xf32>
      %1945 = stablehlo.broadcast_in_dim %1941, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_540 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1946 = stablehlo.pad %1945, %cst_540, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %1947 = stablehlo.broadcast_in_dim %1944, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_541 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1948 = stablehlo.pad %1947, %cst_541, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %1949 = stablehlo.add %1946, %1948 : tensor<2x256xf32>
      %cst_542 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1950 = stablehlo.broadcast_in_dim %cst_542, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %1951 = stablehlo.divide %1949, %1950 : tensor<2x256xf32>
      %1952 = "stablehlo.all_reduce"(%1951) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %1953 = stablehlo.slice %1952 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %1954 = stablehlo.slice %1952 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_543 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1955 = stablehlo.reduce(%1954 init: %cst_543) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_544 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1956 = stablehlo.reduce(%1953 init: %cst_544) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_545 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1957 = stablehlo.broadcast_in_dim %cst_545, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1958 = stablehlo.divide %1955, %1957 : tensor<256xf32>
      %1959 = stablehlo.broadcast_in_dim %1958, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_546 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %1960 = stablehlo.broadcast_in_dim %cst_546, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1961 = stablehlo.divide %1956, %1960 : tensor<256xf32>
      %1962 = stablehlo.broadcast_in_dim %1961, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %1963 = stablehlo.multiply %1959, %869 : tensor<256x14x14x256xf32>
      %1964 = stablehlo.add %1962, %1963 : tensor<256x14x14x256xf32>
      %1965 = stablehlo.convert %1964 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1966 = stablehlo.add %1938, %1965 : tensor<256x14x14x256xf16>
      %1967 = stablehlo.convolution(%724, %1966) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x14x14x256xf16>) -> tensor<1x1x128x256xf16>
      %1968 = stablehlo.reverse %864, dims = [0, 1] : tensor<1x1x128x256xf16>
      %1969 = stablehlo.convolution(%1966, %1968) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x128x256xf16>) -> tensor<256x28x28x128xf16>
      %1970 = stablehlo.convert %1967 : (tensor<1x1x128x256xf16>) -> tensor<1x1x128x256xf32>
      %1971 = stablehlo.add %1509, %1970 : tensor<1x1x128x256xf32>
      %1972 = stablehlo.convert %1921 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_547 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1973 = stablehlo.reduce(%1972 init: %cst_547) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1974 = stablehlo.reshape %1973 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1975 = stablehlo.reshape %1974 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %1976 = stablehlo.multiply %849, %1972 : tensor<256x14x14x256xf32>
      %cst_548 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1977 = stablehlo.reduce(%1976 init: %cst_548) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1978 = stablehlo.reshape %1977 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1979 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1980 = stablehlo.multiply %1972, %1979 : tensor<256x14x14x256xf32>
      %1981 = stablehlo.multiply %852, %1978 : tensor<1x1x1x256xf32>
      %1982 = stablehlo.multiply %1978, %856 : tensor<1x1x1x256xf32>
      %1983 = stablehlo.reshape %1981 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %1984 = stablehlo.multiply %1982, %855 : tensor<1x1x1x256xf32>
      %1985 = stablehlo.negate %1980 : tensor<256x14x14x256xf32>
      %cst_549 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1986 = stablehlo.reduce(%1985 init: %cst_549) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1987 = stablehlo.reshape %1986 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1988 = stablehlo.convert %1980 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_550 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1989 = stablehlo.reduce(%1984 init: %cst_550) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_551 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1990 = stablehlo.reduce(%1987 init: %cst_551) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %1991 = stablehlo.multiply %1989, %834 : tensor<256xf32>
      %1992 = stablehlo.negate %1991 : tensor<256xf32>
      %1993 = stablehlo.multiply %1992, %821 : tensor<256xf32>
      %1994 = stablehlo.add %1990, %1993 : tensor<256xf32>
      %1995 = stablehlo.broadcast_in_dim %1991, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_552 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1996 = stablehlo.pad %1995, %cst_552, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %1997 = stablehlo.broadcast_in_dim %1994, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_553 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1998 = stablehlo.pad %1997, %cst_553, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %1999 = stablehlo.add %1996, %1998 : tensor<2x256xf32>
      %cst_554 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2000 = stablehlo.broadcast_in_dim %cst_554, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %2001 = stablehlo.divide %1999, %2000 : tensor<2x256xf32>
      %2002 = "stablehlo.all_reduce"(%2001) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %2003 = stablehlo.slice %2002 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %2004 = stablehlo.slice %2002 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_555 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2005 = stablehlo.reduce(%2004 init: %cst_555) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_556 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2006 = stablehlo.reduce(%2003 init: %cst_556) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_557 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %2007 = stablehlo.broadcast_in_dim %cst_557, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2008 = stablehlo.divide %2005, %2007 : tensor<256xf32>
      %2009 = stablehlo.broadcast_in_dim %2008, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_558 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %2010 = stablehlo.broadcast_in_dim %cst_558, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2011 = stablehlo.divide %2006, %2010 : tensor<256xf32>
      %2012 = stablehlo.broadcast_in_dim %2011, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %2013 = stablehlo.multiply %2009, %802 : tensor<256x14x14x256xf32>
      %2014 = stablehlo.add %2012, %2013 : tensor<256x14x14x256xf32>
      %2015 = stablehlo.convert %2014 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2016 = stablehlo.add %1988, %2015 : tensor<256x14x14x256xf16>
      %2017 = stablehlo.convolution(%794, %2016) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x256x256xf16>
      %2018 = stablehlo.reverse %797, dims = [0, 1] : tensor<3x3x256x256xf16>
      %2019 = stablehlo.convolution(%2016, %2018) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2020 = stablehlo.convert %2017 : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf32>
      %2021 = stablehlo.add %1511, %2020 : tensor<3x3x256x256xf32>
      %cst_559 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2022 = stablehlo.broadcast_in_dim %cst_559, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
      %2023 = stablehlo.select %796, %2019, %2022 : tensor<256x14x14x256xi1>, tensor<256x14x14x256xf16>
      %2024 = stablehlo.convert %2023 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %cst_560 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2025 = stablehlo.reduce(%2024 init: %cst_560) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %2026 = stablehlo.reshape %2025 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2027 = stablehlo.reshape %2026 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %2028 = stablehlo.multiply %779, %2024 : tensor<256x14x14x256xf32>
      %cst_561 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2029 = stablehlo.reduce(%2028 init: %cst_561) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %2030 = stablehlo.reshape %2029 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2031 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2032 = stablehlo.multiply %2024, %2031 : tensor<256x14x14x256xf32>
      %2033 = stablehlo.multiply %782, %2030 : tensor<1x1x1x256xf32>
      %2034 = stablehlo.multiply %2030, %786 : tensor<1x1x1x256xf32>
      %2035 = stablehlo.reshape %2033 : (tensor<1x1x1x256xf32>) -> tensor<256xf32>
      %2036 = stablehlo.multiply %2034, %785 : tensor<1x1x1x256xf32>
      %2037 = stablehlo.negate %2032 : tensor<256x14x14x256xf32>
      %cst_562 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2038 = stablehlo.reduce(%2037 init: %cst_562) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x14x14x256xf32>, tensor<f32>) -> tensor<256xf32>
      %2039 = stablehlo.reshape %2038 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2040 = stablehlo.convert %2032 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %cst_563 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2041 = stablehlo.reduce(%2036 init: %cst_563) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_564 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2042 = stablehlo.reduce(%2039 init: %cst_564) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %2043 = stablehlo.multiply %2041, %764 : tensor<256xf32>
      %2044 = stablehlo.negate %2043 : tensor<256xf32>
      %2045 = stablehlo.multiply %2044, %751 : tensor<256xf32>
      %2046 = stablehlo.add %2042, %2045 : tensor<256xf32>
      %2047 = stablehlo.broadcast_in_dim %2043, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_565 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2048 = stablehlo.pad %2047, %cst_565, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %2049 = stablehlo.broadcast_in_dim %2046, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %cst_566 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2050 = stablehlo.pad %2049, %cst_566, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<2x256xf32>
      %2051 = stablehlo.add %2048, %2050 : tensor<2x256xf32>
      %cst_567 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2052 = stablehlo.broadcast_in_dim %cst_567, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
      %2053 = stablehlo.divide %2051, %2052 : tensor<2x256xf32>
      %2054 = "stablehlo.all_reduce"(%2053) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x256xf32>) -> tensor<2x256xf32>
      %2055 = stablehlo.slice %2054 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %2056 = stablehlo.slice %2054 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
      %cst_568 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2057 = stablehlo.reduce(%2056 init: %cst_568) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_569 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2058 = stablehlo.reduce(%2055 init: %cst_569) applies stablehlo.add across dimensions = [0] : (tensor<1x256xf32>, tensor<f32>) -> tensor<256xf32>
      %cst_570 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %2059 = stablehlo.broadcast_in_dim %cst_570, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2060 = stablehlo.divide %2057, %2059 : tensor<256xf32>
      %2061 = stablehlo.broadcast_in_dim %2060, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %cst_571 = stablehlo.constant dense<5.017600e+04> : tensor<f32>
      %2062 = stablehlo.broadcast_in_dim %cst_571, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2063 = stablehlo.divide %2058, %2062 : tensor<256xf32>
      %2064 = stablehlo.broadcast_in_dim %2063, dims = [3] : (tensor<256xf32>) -> tensor<256x14x14x256xf32>
      %2065 = stablehlo.multiply %2061, %732 : tensor<256x14x14x256xf32>
      %2066 = stablehlo.add %2064, %2065 : tensor<256x14x14x256xf32>
      %2067 = stablehlo.convert %2066 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2068 = stablehlo.add %2040, %2067 : tensor<256x14x14x256xf16>
      %2069 = stablehlo.convolution(%724, %2068) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x14x14x256xf16>) -> tensor<3x3x128x256xf16>
      %2070 = stablehlo.reverse %727, dims = [0, 1] : tensor<3x3x128x256xf16>
      %2071 = stablehlo.convolution(%2068, %2070) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x128x256xf16>) -> tensor<256x28x28x128xf16>
      %2072 = stablehlo.add %1969, %2071 : tensor<256x28x28x128xf16>
      %2073 = stablehlo.convert %2069 : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf32>
      %2074 = stablehlo.add %1513, %2073 : tensor<3x3x128x256xf32>
      %cst_572 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2075 = stablehlo.broadcast_in_dim %cst_572, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %2076 = stablehlo.select %726, %2072, %2075 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %2077 = stablehlo.convert %2076 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_573 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2078 = stablehlo.reduce(%2077 init: %cst_573) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2079 = stablehlo.reshape %2078 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2080 = stablehlo.reshape %2079 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2081 = stablehlo.multiply %708, %2077 : tensor<256x28x28x128xf32>
      %cst_574 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2082 = stablehlo.reduce(%2081 init: %cst_574) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2083 = stablehlo.reshape %2082 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2084 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %2085 = stablehlo.multiply %2077, %2084 : tensor<256x28x28x128xf32>
      %2086 = stablehlo.multiply %711, %2083 : tensor<1x1x1x128xf32>
      %2087 = stablehlo.multiply %2083, %715 : tensor<1x1x1x128xf32>
      %2088 = stablehlo.reshape %2086 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2089 = stablehlo.multiply %2087, %714 : tensor<1x1x1x128xf32>
      %2090 = stablehlo.negate %2085 : tensor<256x28x28x128xf32>
      %cst_575 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2091 = stablehlo.reduce(%2090 init: %cst_575) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2092 = stablehlo.reshape %2091 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2093 = stablehlo.convert %2085 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_576 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2094 = stablehlo.reduce(%2089 init: %cst_576) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_577 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2095 = stablehlo.reduce(%2092 init: %cst_577) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2096 = stablehlo.multiply %2094, %693 : tensor<128xf32>
      %2097 = stablehlo.negate %2096 : tensor<128xf32>
      %2098 = stablehlo.multiply %2097, %680 : tensor<128xf32>
      %2099 = stablehlo.add %2095, %2098 : tensor<128xf32>
      %2100 = stablehlo.broadcast_in_dim %2096, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_578 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2101 = stablehlo.pad %2100, %cst_578, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2102 = stablehlo.broadcast_in_dim %2099, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_579 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2103 = stablehlo.pad %2102, %cst_579, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2104 = stablehlo.add %2101, %2103 : tensor<2x128xf32>
      %cst_580 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2105 = stablehlo.broadcast_in_dim %cst_580, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %2106 = stablehlo.divide %2104, %2105 : tensor<2x128xf32>
      %2107 = "stablehlo.all_reduce"(%2106) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %2108 = stablehlo.slice %2107 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %2109 = stablehlo.slice %2107 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_581 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2110 = stablehlo.reduce(%2109 init: %cst_581) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_582 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2111 = stablehlo.reduce(%2108 init: %cst_582) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_583 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2112 = stablehlo.broadcast_in_dim %cst_583, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2113 = stablehlo.divide %2110, %2112 : tensor<128xf32>
      %2114 = stablehlo.broadcast_in_dim %2113, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_584 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2115 = stablehlo.broadcast_in_dim %cst_584, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2116 = stablehlo.divide %2111, %2115 : tensor<128xf32>
      %2117 = stablehlo.broadcast_in_dim %2116, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %2118 = stablehlo.multiply %2114, %661 : tensor<256x28x28x128xf32>
      %2119 = stablehlo.add %2117, %2118 : tensor<256x28x28x128xf32>
      %2120 = stablehlo.convert %2119 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %2121 = stablehlo.add %2093, %2120 : tensor<256x28x28x128xf16>
      %2122 = stablehlo.convolution(%653, %2121) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %2123 = stablehlo.reverse %656, dims = [0, 1] : tensor<3x3x128x128xf16>
      %2124 = stablehlo.convolution(%2121, %2123) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %2125 = stablehlo.convert %2122 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %2126 = stablehlo.add %1515, %2125 : tensor<3x3x128x128xf32>
      %cst_585 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2127 = stablehlo.broadcast_in_dim %cst_585, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %2128 = stablehlo.select %655, %2124, %2127 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %2129 = stablehlo.convert %2128 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_586 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2130 = stablehlo.reduce(%2129 init: %cst_586) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2131 = stablehlo.reshape %2130 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2132 = stablehlo.reshape %2131 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2133 = stablehlo.multiply %638, %2129 : tensor<256x28x28x128xf32>
      %cst_587 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2134 = stablehlo.reduce(%2133 init: %cst_587) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2135 = stablehlo.reshape %2134 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2136 = stablehlo.broadcast_in_dim %646, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %2137 = stablehlo.multiply %2129, %2136 : tensor<256x28x28x128xf32>
      %2138 = stablehlo.multiply %641, %2135 : tensor<1x1x1x128xf32>
      %2139 = stablehlo.multiply %2135, %645 : tensor<1x1x1x128xf32>
      %2140 = stablehlo.reshape %2138 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2141 = stablehlo.multiply %2139, %644 : tensor<1x1x1x128xf32>
      %2142 = stablehlo.negate %2137 : tensor<256x28x28x128xf32>
      %cst_588 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2143 = stablehlo.reduce(%2142 init: %cst_588) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2144 = stablehlo.reshape %2143 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2145 = stablehlo.convert %2137 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_589 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2146 = stablehlo.reduce(%2141 init: %cst_589) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_590 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2147 = stablehlo.reduce(%2144 init: %cst_590) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2148 = stablehlo.multiply %2146, %623 : tensor<128xf32>
      %2149 = stablehlo.negate %2148 : tensor<128xf32>
      %2150 = stablehlo.multiply %2149, %610 : tensor<128xf32>
      %2151 = stablehlo.add %2147, %2150 : tensor<128xf32>
      %2152 = stablehlo.broadcast_in_dim %2148, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_591 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2153 = stablehlo.pad %2152, %cst_591, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2154 = stablehlo.broadcast_in_dim %2151, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_592 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2155 = stablehlo.pad %2154, %cst_592, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2156 = stablehlo.add %2153, %2155 : tensor<2x128xf32>
      %cst_593 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2157 = stablehlo.broadcast_in_dim %cst_593, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %2158 = stablehlo.divide %2156, %2157 : tensor<2x128xf32>
      %2159 = "stablehlo.all_reduce"(%2158) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %2160 = stablehlo.slice %2159 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %2161 = stablehlo.slice %2159 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_594 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2162 = stablehlo.reduce(%2161 init: %cst_594) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_595 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2163 = stablehlo.reduce(%2160 init: %cst_595) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_596 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2164 = stablehlo.broadcast_in_dim %cst_596, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2165 = stablehlo.divide %2162, %2164 : tensor<128xf32>
      %2166 = stablehlo.broadcast_in_dim %2165, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_597 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2167 = stablehlo.broadcast_in_dim %cst_597, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2168 = stablehlo.divide %2163, %2167 : tensor<128xf32>
      %2169 = stablehlo.broadcast_in_dim %2168, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %2170 = stablehlo.multiply %2166, %591 : tensor<256x28x28x128xf32>
      %2171 = stablehlo.add %2169, %2170 : tensor<256x28x28x128xf32>
      %2172 = stablehlo.convert %2171 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %2173 = stablehlo.add %2145, %2172 : tensor<256x28x28x128xf16>
      %2174 = stablehlo.convolution(%583, %2173) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %2175 = stablehlo.reverse %586, dims = [0, 1] : tensor<3x3x128x128xf16>
      %2176 = stablehlo.convolution(%2173, %2175) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %2177 = stablehlo.add %2076, %2176 : tensor<256x28x28x128xf16>
      %2178 = stablehlo.convert %2174 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %2179 = stablehlo.add %1517, %2178 : tensor<3x3x128x128xf32>
      %cst_598 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2180 = stablehlo.broadcast_in_dim %cst_598, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %2181 = stablehlo.select %585, %2177, %2180 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %2182 = stablehlo.convert %2181 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_599 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2183 = stablehlo.reduce(%2182 init: %cst_599) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2184 = stablehlo.reshape %2183 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2185 = stablehlo.reshape %2184 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2186 = stablehlo.multiply %567, %2182 : tensor<256x28x28x128xf32>
      %cst_600 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2187 = stablehlo.reduce(%2186 init: %cst_600) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2188 = stablehlo.reshape %2187 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2189 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %2190 = stablehlo.multiply %2182, %2189 : tensor<256x28x28x128xf32>
      %2191 = stablehlo.multiply %570, %2188 : tensor<1x1x1x128xf32>
      %2192 = stablehlo.multiply %2188, %574 : tensor<1x1x1x128xf32>
      %2193 = stablehlo.reshape %2191 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2194 = stablehlo.multiply %2192, %573 : tensor<1x1x1x128xf32>
      %2195 = stablehlo.negate %2190 : tensor<256x28x28x128xf32>
      %cst_601 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2196 = stablehlo.reduce(%2195 init: %cst_601) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2197 = stablehlo.reshape %2196 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2198 = stablehlo.convert %2190 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_602 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2199 = stablehlo.reduce(%2194 init: %cst_602) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_603 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2200 = stablehlo.reduce(%2197 init: %cst_603) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2201 = stablehlo.multiply %2199, %552 : tensor<128xf32>
      %2202 = stablehlo.negate %2201 : tensor<128xf32>
      %2203 = stablehlo.multiply %2202, %539 : tensor<128xf32>
      %2204 = stablehlo.add %2200, %2203 : tensor<128xf32>
      %2205 = stablehlo.broadcast_in_dim %2201, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_604 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2206 = stablehlo.pad %2205, %cst_604, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2207 = stablehlo.broadcast_in_dim %2204, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_605 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2208 = stablehlo.pad %2207, %cst_605, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2209 = stablehlo.add %2206, %2208 : tensor<2x128xf32>
      %cst_606 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2210 = stablehlo.broadcast_in_dim %cst_606, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %2211 = stablehlo.divide %2209, %2210 : tensor<2x128xf32>
      %2212 = "stablehlo.all_reduce"(%2211) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %2213 = stablehlo.slice %2212 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %2214 = stablehlo.slice %2212 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_607 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2215 = stablehlo.reduce(%2214 init: %cst_607) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_608 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2216 = stablehlo.reduce(%2213 init: %cst_608) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_609 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2217 = stablehlo.broadcast_in_dim %cst_609, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2218 = stablehlo.divide %2215, %2217 : tensor<128xf32>
      %2219 = stablehlo.broadcast_in_dim %2218, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_610 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2220 = stablehlo.broadcast_in_dim %cst_610, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2221 = stablehlo.divide %2216, %2220 : tensor<128xf32>
      %2222 = stablehlo.broadcast_in_dim %2221, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %2223 = stablehlo.multiply %2219, %520 : tensor<256x28x28x128xf32>
      %2224 = stablehlo.add %2222, %2223 : tensor<256x28x28x128xf32>
      %2225 = stablehlo.convert %2224 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %2226 = stablehlo.add %2198, %2225 : tensor<256x28x28x128xf16>
      %2227 = stablehlo.convolution(%375, %2226) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x28x28x128xf16>) -> tensor<1x1x64x128xf16>
      %2228 = stablehlo.reverse %515, dims = [0, 1] : tensor<1x1x64x128xf16>
      %2229 = stablehlo.convolution(%2226, %2228) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x64x128xf16>) -> tensor<256x56x56x64xf16>
      %2230 = stablehlo.convert %2227 : (tensor<1x1x64x128xf16>) -> tensor<1x1x64x128xf32>
      %2231 = stablehlo.add %1519, %2230 : tensor<1x1x64x128xf32>
      %2232 = stablehlo.convert %2181 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_611 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2233 = stablehlo.reduce(%2232 init: %cst_611) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2234 = stablehlo.reshape %2233 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2235 = stablehlo.reshape %2234 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2236 = stablehlo.multiply %500, %2232 : tensor<256x28x28x128xf32>
      %cst_612 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2237 = stablehlo.reduce(%2236 init: %cst_612) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2238 = stablehlo.reshape %2237 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2239 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %2240 = stablehlo.multiply %2232, %2239 : tensor<256x28x28x128xf32>
      %2241 = stablehlo.multiply %503, %2238 : tensor<1x1x1x128xf32>
      %2242 = stablehlo.multiply %2238, %507 : tensor<1x1x1x128xf32>
      %2243 = stablehlo.reshape %2241 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2244 = stablehlo.multiply %2242, %506 : tensor<1x1x1x128xf32>
      %2245 = stablehlo.negate %2240 : tensor<256x28x28x128xf32>
      %cst_613 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2246 = stablehlo.reduce(%2245 init: %cst_613) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2247 = stablehlo.reshape %2246 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2248 = stablehlo.convert %2240 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_614 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2249 = stablehlo.reduce(%2244 init: %cst_614) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_615 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2250 = stablehlo.reduce(%2247 init: %cst_615) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2251 = stablehlo.multiply %2249, %485 : tensor<128xf32>
      %2252 = stablehlo.negate %2251 : tensor<128xf32>
      %2253 = stablehlo.multiply %2252, %472 : tensor<128xf32>
      %2254 = stablehlo.add %2250, %2253 : tensor<128xf32>
      %2255 = stablehlo.broadcast_in_dim %2251, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_616 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2256 = stablehlo.pad %2255, %cst_616, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2257 = stablehlo.broadcast_in_dim %2254, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_617 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2258 = stablehlo.pad %2257, %cst_617, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2259 = stablehlo.add %2256, %2258 : tensor<2x128xf32>
      %cst_618 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2260 = stablehlo.broadcast_in_dim %cst_618, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %2261 = stablehlo.divide %2259, %2260 : tensor<2x128xf32>
      %2262 = "stablehlo.all_reduce"(%2261) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %2263 = stablehlo.slice %2262 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %2264 = stablehlo.slice %2262 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_619 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2265 = stablehlo.reduce(%2264 init: %cst_619) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_620 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2266 = stablehlo.reduce(%2263 init: %cst_620) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_621 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2267 = stablehlo.broadcast_in_dim %cst_621, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2268 = stablehlo.divide %2265, %2267 : tensor<128xf32>
      %2269 = stablehlo.broadcast_in_dim %2268, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_622 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2270 = stablehlo.broadcast_in_dim %cst_622, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2271 = stablehlo.divide %2266, %2270 : tensor<128xf32>
      %2272 = stablehlo.broadcast_in_dim %2271, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %2273 = stablehlo.multiply %2269, %453 : tensor<256x28x28x128xf32>
      %2274 = stablehlo.add %2272, %2273 : tensor<256x28x28x128xf32>
      %2275 = stablehlo.convert %2274 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %2276 = stablehlo.add %2248, %2275 : tensor<256x28x28x128xf16>
      %2277 = stablehlo.convolution(%445, %2276) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x128x128xf16>
      %2278 = stablehlo.reverse %448, dims = [0, 1] : tensor<3x3x128x128xf16>
      %2279 = stablehlo.convolution(%2276, %2278) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %2280 = stablehlo.convert %2277 : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf32>
      %2281 = stablehlo.add %1521, %2280 : tensor<3x3x128x128xf32>
      %cst_623 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2282 = stablehlo.broadcast_in_dim %cst_623, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
      %2283 = stablehlo.select %447, %2279, %2282 : tensor<256x28x28x128xi1>, tensor<256x28x28x128xf16>
      %2284 = stablehlo.convert %2283 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %cst_624 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2285 = stablehlo.reduce(%2284 init: %cst_624) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2286 = stablehlo.reshape %2285 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2287 = stablehlo.reshape %2286 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2288 = stablehlo.multiply %430, %2284 : tensor<256x28x28x128xf32>
      %cst_625 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2289 = stablehlo.reduce(%2288 init: %cst_625) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2290 = stablehlo.reshape %2289 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2291 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %2292 = stablehlo.multiply %2284, %2291 : tensor<256x28x28x128xf32>
      %2293 = stablehlo.multiply %433, %2290 : tensor<1x1x1x128xf32>
      %2294 = stablehlo.multiply %2290, %437 : tensor<1x1x1x128xf32>
      %2295 = stablehlo.reshape %2293 : (tensor<1x1x1x128xf32>) -> tensor<128xf32>
      %2296 = stablehlo.multiply %2294, %436 : tensor<1x1x1x128xf32>
      %2297 = stablehlo.negate %2292 : tensor<256x28x28x128xf32>
      %cst_626 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2298 = stablehlo.reduce(%2297 init: %cst_626) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x28x28x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2299 = stablehlo.reshape %2298 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %2300 = stablehlo.convert %2292 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %cst_627 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2301 = stablehlo.reduce(%2296 init: %cst_627) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_628 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2302 = stablehlo.reduce(%2299 init: %cst_628) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %2303 = stablehlo.multiply %2301, %415 : tensor<128xf32>
      %2304 = stablehlo.negate %2303 : tensor<128xf32>
      %2305 = stablehlo.multiply %2304, %402 : tensor<128xf32>
      %2306 = stablehlo.add %2302, %2305 : tensor<128xf32>
      %2307 = stablehlo.broadcast_in_dim %2303, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_629 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2308 = stablehlo.pad %2307, %cst_629, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2309 = stablehlo.broadcast_in_dim %2306, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %cst_630 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2310 = stablehlo.pad %2309, %cst_630, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<2x128xf32>
      %2311 = stablehlo.add %2308, %2310 : tensor<2x128xf32>
      %cst_631 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2312 = stablehlo.broadcast_in_dim %cst_631, dims = [] : (tensor<f32>) -> tensor<2x128xf32>
      %2313 = stablehlo.divide %2311, %2312 : tensor<2x128xf32>
      %2314 = "stablehlo.all_reduce"(%2313) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x128xf32>) -> tensor<2x128xf32>
      %2315 = stablehlo.slice %2314 [0:1, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %2316 = stablehlo.slice %2314 [1:2, 0:128] : (tensor<2x128xf32>) -> tensor<1x128xf32>
      %cst_632 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2317 = stablehlo.reduce(%2316 init: %cst_632) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_633 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2318 = stablehlo.reduce(%2315 init: %cst_633) applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<128xf32>
      %cst_634 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2319 = stablehlo.broadcast_in_dim %cst_634, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2320 = stablehlo.divide %2317, %2319 : tensor<128xf32>
      %2321 = stablehlo.broadcast_in_dim %2320, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %cst_635 = stablehlo.constant dense<2.007040e+05> : tensor<f32>
      %2322 = stablehlo.broadcast_in_dim %cst_635, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2323 = stablehlo.divide %2318, %2322 : tensor<128xf32>
      %2324 = stablehlo.broadcast_in_dim %2323, dims = [3] : (tensor<128xf32>) -> tensor<256x28x28x128xf32>
      %2325 = stablehlo.multiply %2321, %383 : tensor<256x28x28x128xf32>
      %2326 = stablehlo.add %2324, %2325 : tensor<256x28x28x128xf32>
      %2327 = stablehlo.convert %2326 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %2328 = stablehlo.add %2300, %2327 : tensor<256x28x28x128xf16>
      %2329 = stablehlo.convolution(%375, %2328) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x28x28x128xf16>) -> tensor<3x3x64x128xf16>
      %2330 = stablehlo.reverse %378, dims = [0, 1] : tensor<3x3x64x128xf16>
      %2331 = stablehlo.convolution(%2328, %2330) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x64x128xf16>) -> tensor<256x56x56x64xf16>
      %2332 = stablehlo.add %2229, %2331 : tensor<256x56x56x64xf16>
      %2333 = stablehlo.convert %2329 : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf32>
      %2334 = stablehlo.add %1523, %2333 : tensor<3x3x64x128xf32>
      %cst_636 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2335 = stablehlo.broadcast_in_dim %cst_636, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %2336 = stablehlo.select %377, %2332, %2335 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %2337 = stablehlo.convert %2336 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_637 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2338 = stablehlo.reduce(%2337 init: %cst_637) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2339 = stablehlo.reshape %2338 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2340 = stablehlo.reshape %2339 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2341 = stablehlo.multiply %359, %2337 : tensor<256x56x56x64xf32>
      %cst_638 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2342 = stablehlo.reduce(%2341 init: %cst_638) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2343 = stablehlo.reshape %2342 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2344 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %2345 = stablehlo.multiply %2337, %2344 : tensor<256x56x56x64xf32>
      %2346 = stablehlo.multiply %362, %2343 : tensor<1x1x1x64xf32>
      %2347 = stablehlo.multiply %2343, %366 : tensor<1x1x1x64xf32>
      %2348 = stablehlo.reshape %2346 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2349 = stablehlo.multiply %2347, %365 : tensor<1x1x1x64xf32>
      %2350 = stablehlo.negate %2345 : tensor<256x56x56x64xf32>
      %cst_639 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2351 = stablehlo.reduce(%2350 init: %cst_639) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2352 = stablehlo.reshape %2351 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2353 = stablehlo.convert %2345 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_640 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2354 = stablehlo.reduce(%2349 init: %cst_640) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_641 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2355 = stablehlo.reduce(%2352 init: %cst_641) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2356 = stablehlo.multiply %2354, %344 : tensor<64xf32>
      %2357 = stablehlo.negate %2356 : tensor<64xf32>
      %2358 = stablehlo.multiply %2357, %331 : tensor<64xf32>
      %2359 = stablehlo.add %2355, %2358 : tensor<64xf32>
      %2360 = stablehlo.broadcast_in_dim %2356, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_642 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2361 = stablehlo.pad %2360, %cst_642, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2362 = stablehlo.broadcast_in_dim %2359, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_643 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2363 = stablehlo.pad %2362, %cst_643, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2364 = stablehlo.add %2361, %2363 : tensor<2x64xf32>
      %cst_644 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2365 = stablehlo.broadcast_in_dim %cst_644, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %2366 = stablehlo.divide %2364, %2365 : tensor<2x64xf32>
      %2367 = "stablehlo.all_reduce"(%2366) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %2368 = stablehlo.slice %2367 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %2369 = stablehlo.slice %2367 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_645 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2370 = stablehlo.reduce(%2369 init: %cst_645) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_646 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2371 = stablehlo.reduce(%2368 init: %cst_646) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_647 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %2372 = stablehlo.broadcast_in_dim %cst_647, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2373 = stablehlo.divide %2370, %2372 : tensor<64xf32>
      %2374 = stablehlo.broadcast_in_dim %2373, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_648 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %2375 = stablehlo.broadcast_in_dim %cst_648, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2376 = stablehlo.divide %2371, %2375 : tensor<64xf32>
      %2377 = stablehlo.broadcast_in_dim %2376, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %2378 = stablehlo.multiply %2374, %312 : tensor<256x56x56x64xf32>
      %2379 = stablehlo.add %2377, %2378 : tensor<256x56x56x64xf32>
      %2380 = stablehlo.convert %2379 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %2381 = stablehlo.add %2353, %2380 : tensor<256x56x56x64xf16>
      %2382 = stablehlo.convolution(%304, %2381) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %2383 = stablehlo.reverse %307, dims = [0, 1] : tensor<3x3x64x64xf16>
      %2384 = stablehlo.convolution(%2381, %2383) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %2385 = stablehlo.convert %2382 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %2386 = stablehlo.add %1525, %2385 : tensor<3x3x64x64xf32>
      %cst_649 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2387 = stablehlo.broadcast_in_dim %cst_649, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %2388 = stablehlo.select %306, %2384, %2387 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %2389 = stablehlo.convert %2388 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_650 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2390 = stablehlo.reduce(%2389 init: %cst_650) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2391 = stablehlo.reshape %2390 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2392 = stablehlo.reshape %2391 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2393 = stablehlo.multiply %289, %2389 : tensor<256x56x56x64xf32>
      %cst_651 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2394 = stablehlo.reduce(%2393 init: %cst_651) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2395 = stablehlo.reshape %2394 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2396 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %2397 = stablehlo.multiply %2389, %2396 : tensor<256x56x56x64xf32>
      %2398 = stablehlo.multiply %292, %2395 : tensor<1x1x1x64xf32>
      %2399 = stablehlo.multiply %2395, %296 : tensor<1x1x1x64xf32>
      %2400 = stablehlo.reshape %2398 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2401 = stablehlo.multiply %2399, %295 : tensor<1x1x1x64xf32>
      %2402 = stablehlo.negate %2397 : tensor<256x56x56x64xf32>
      %cst_652 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2403 = stablehlo.reduce(%2402 init: %cst_652) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2404 = stablehlo.reshape %2403 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2405 = stablehlo.convert %2397 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_653 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2406 = stablehlo.reduce(%2401 init: %cst_653) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_654 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2407 = stablehlo.reduce(%2404 init: %cst_654) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2408 = stablehlo.multiply %2406, %274 : tensor<64xf32>
      %2409 = stablehlo.negate %2408 : tensor<64xf32>
      %2410 = stablehlo.multiply %2409, %261 : tensor<64xf32>
      %2411 = stablehlo.add %2407, %2410 : tensor<64xf32>
      %2412 = stablehlo.broadcast_in_dim %2408, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_655 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2413 = stablehlo.pad %2412, %cst_655, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2414 = stablehlo.broadcast_in_dim %2411, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_656 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2415 = stablehlo.pad %2414, %cst_656, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2416 = stablehlo.add %2413, %2415 : tensor<2x64xf32>
      %cst_657 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2417 = stablehlo.broadcast_in_dim %cst_657, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %2418 = stablehlo.divide %2416, %2417 : tensor<2x64xf32>
      %2419 = "stablehlo.all_reduce"(%2418) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %2420 = stablehlo.slice %2419 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %2421 = stablehlo.slice %2419 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_658 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2422 = stablehlo.reduce(%2421 init: %cst_658) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_659 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2423 = stablehlo.reduce(%2420 init: %cst_659) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_660 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %2424 = stablehlo.broadcast_in_dim %cst_660, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2425 = stablehlo.divide %2422, %2424 : tensor<64xf32>
      %2426 = stablehlo.broadcast_in_dim %2425, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_661 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %2427 = stablehlo.broadcast_in_dim %cst_661, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2428 = stablehlo.divide %2423, %2427 : tensor<64xf32>
      %2429 = stablehlo.broadcast_in_dim %2428, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %2430 = stablehlo.multiply %2426, %242 : tensor<256x56x56x64xf32>
      %2431 = stablehlo.add %2429, %2430 : tensor<256x56x56x64xf32>
      %2432 = stablehlo.convert %2431 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %2433 = stablehlo.add %2405, %2432 : tensor<256x56x56x64xf16>
      %2434 = stablehlo.convolution(%234, %2433) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %2435 = stablehlo.reverse %237, dims = [0, 1] : tensor<3x3x64x64xf16>
      %2436 = stablehlo.convolution(%2433, %2435) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %2437 = stablehlo.add %2336, %2436 : tensor<256x56x56x64xf16>
      %2438 = stablehlo.convert %2434 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %2439 = stablehlo.add %1527, %2438 : tensor<3x3x64x64xf32>
      %cst_662 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2440 = stablehlo.broadcast_in_dim %cst_662, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %2441 = stablehlo.select %236, %2437, %2440 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %2442 = stablehlo.convert %2441 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_663 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2443 = stablehlo.reduce(%2442 init: %cst_663) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2444 = stablehlo.reshape %2443 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2445 = stablehlo.reshape %2444 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2446 = stablehlo.multiply %218, %2442 : tensor<256x56x56x64xf32>
      %cst_664 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2447 = stablehlo.reduce(%2446 init: %cst_664) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2448 = stablehlo.reshape %2447 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2449 = stablehlo.broadcast_in_dim %226, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %2450 = stablehlo.multiply %2442, %2449 : tensor<256x56x56x64xf32>
      %2451 = stablehlo.multiply %221, %2448 : tensor<1x1x1x64xf32>
      %2452 = stablehlo.multiply %2448, %225 : tensor<1x1x1x64xf32>
      %2453 = stablehlo.reshape %2451 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2454 = stablehlo.multiply %2452, %224 : tensor<1x1x1x64xf32>
      %2455 = stablehlo.negate %2450 : tensor<256x56x56x64xf32>
      %cst_665 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2456 = stablehlo.reduce(%2455 init: %cst_665) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2457 = stablehlo.reshape %2456 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2458 = stablehlo.convert %2450 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_666 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2459 = stablehlo.reduce(%2454 init: %cst_666) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_667 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2460 = stablehlo.reduce(%2457 init: %cst_667) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2461 = stablehlo.multiply %2459, %203 : tensor<64xf32>
      %2462 = stablehlo.negate %2461 : tensor<64xf32>
      %2463 = stablehlo.multiply %2462, %190 : tensor<64xf32>
      %2464 = stablehlo.add %2460, %2463 : tensor<64xf32>
      %2465 = stablehlo.broadcast_in_dim %2461, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_668 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2466 = stablehlo.pad %2465, %cst_668, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2467 = stablehlo.broadcast_in_dim %2464, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_669 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2468 = stablehlo.pad %2467, %cst_669, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2469 = stablehlo.add %2466, %2468 : tensor<2x64xf32>
      %cst_670 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2470 = stablehlo.broadcast_in_dim %cst_670, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %2471 = stablehlo.divide %2469, %2470 : tensor<2x64xf32>
      %2472 = "stablehlo.all_reduce"(%2471) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %2473 = stablehlo.slice %2472 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %2474 = stablehlo.slice %2472 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_671 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2475 = stablehlo.reduce(%2474 init: %cst_671) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_672 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2476 = stablehlo.reduce(%2473 init: %cst_672) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_673 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %2477 = stablehlo.broadcast_in_dim %cst_673, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2478 = stablehlo.divide %2475, %2477 : tensor<64xf32>
      %2479 = stablehlo.broadcast_in_dim %2478, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_674 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %2480 = stablehlo.broadcast_in_dim %cst_674, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2481 = stablehlo.divide %2476, %2480 : tensor<64xf32>
      %2482 = stablehlo.broadcast_in_dim %2481, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %2483 = stablehlo.multiply %2479, %171 : tensor<256x56x56x64xf32>
      %2484 = stablehlo.add %2482, %2483 : tensor<256x56x56x64xf32>
      %2485 = stablehlo.convert %2484 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %2486 = stablehlo.add %2458, %2485 : tensor<256x56x56x64xf16>
      %2487 = stablehlo.convolution(%163, %2486) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %2488 = stablehlo.reverse %166, dims = [0, 1] : tensor<3x3x64x64xf16>
      %2489 = stablehlo.convolution(%2486, %2488) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %2490 = stablehlo.convert %2487 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %2491 = stablehlo.add %1529, %2490 : tensor<3x3x64x64xf32>
      %cst_675 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2492 = stablehlo.broadcast_in_dim %cst_675, dims = [] : (tensor<f16>) -> tensor<256x56x56x64xf16>
      %2493 = stablehlo.select %165, %2489, %2492 : tensor<256x56x56x64xi1>, tensor<256x56x56x64xf16>
      %2494 = stablehlo.convert %2493 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %cst_676 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2495 = stablehlo.reduce(%2494 init: %cst_676) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2496 = stablehlo.reshape %2495 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2497 = stablehlo.reshape %2496 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2498 = stablehlo.multiply %148, %2494 : tensor<256x56x56x64xf32>
      %cst_677 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2499 = stablehlo.reduce(%2498 init: %cst_677) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2500 = stablehlo.reshape %2499 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2501 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %2502 = stablehlo.multiply %2494, %2501 : tensor<256x56x56x64xf32>
      %2503 = stablehlo.multiply %151, %2500 : tensor<1x1x1x64xf32>
      %2504 = stablehlo.multiply %2500, %155 : tensor<1x1x1x64xf32>
      %2505 = stablehlo.reshape %2503 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2506 = stablehlo.multiply %2504, %154 : tensor<1x1x1x64xf32>
      %2507 = stablehlo.negate %2502 : tensor<256x56x56x64xf32>
      %cst_678 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2508 = stablehlo.reduce(%2507 init: %cst_678) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2509 = stablehlo.reshape %2508 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2510 = stablehlo.convert %2502 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %cst_679 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2511 = stablehlo.reduce(%2506 init: %cst_679) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_680 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2512 = stablehlo.reduce(%2509 init: %cst_680) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2513 = stablehlo.multiply %2511, %133 : tensor<64xf32>
      %2514 = stablehlo.negate %2513 : tensor<64xf32>
      %2515 = stablehlo.multiply %2514, %120 : tensor<64xf32>
      %2516 = stablehlo.add %2512, %2515 : tensor<64xf32>
      %2517 = stablehlo.broadcast_in_dim %2513, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_681 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2518 = stablehlo.pad %2517, %cst_681, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2519 = stablehlo.broadcast_in_dim %2516, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_682 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2520 = stablehlo.pad %2519, %cst_682, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2521 = stablehlo.add %2518, %2520 : tensor<2x64xf32>
      %cst_683 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2522 = stablehlo.broadcast_in_dim %cst_683, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %2523 = stablehlo.divide %2521, %2522 : tensor<2x64xf32>
      %2524 = "stablehlo.all_reduce"(%2523) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %2525 = stablehlo.slice %2524 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %2526 = stablehlo.slice %2524 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_684 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2527 = stablehlo.reduce(%2526 init: %cst_684) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_685 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2528 = stablehlo.reduce(%2525 init: %cst_685) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_686 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %2529 = stablehlo.broadcast_in_dim %cst_686, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2530 = stablehlo.divide %2527, %2529 : tensor<64xf32>
      %2531 = stablehlo.broadcast_in_dim %2530, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %cst_687 = stablehlo.constant dense<8.028160e+05> : tensor<f32>
      %2532 = stablehlo.broadcast_in_dim %cst_687, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2533 = stablehlo.divide %2528, %2532 : tensor<64xf32>
      %2534 = stablehlo.broadcast_in_dim %2533, dims = [3] : (tensor<64xf32>) -> tensor<256x56x56x64xf32>
      %2535 = stablehlo.multiply %2531, %101 : tensor<256x56x56x64xf32>
      %2536 = stablehlo.add %2534, %2535 : tensor<256x56x56x64xf32>
      %2537 = stablehlo.convert %2536 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %2538 = stablehlo.add %2510, %2537 : tensor<256x56x56x64xf16>
      %2539 = stablehlo.convolution(%95, %2538) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<256x56x56x64xf16>) -> tensor<3x3x64x64xf16>
      %2540 = stablehlo.reverse %96, dims = [0, 1] : tensor<3x3x64x64xf16>
      %2541 = stablehlo.convolution(%2538, %2540) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %2542 = stablehlo.add %2441, %2541 : tensor<256x56x56x64xf16>
      %2543 = stablehlo.convert %2539 : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf32>
      %2544 = stablehlo.add %1531, %2543 : tensor<3x3x64x64xf32>
      %cst_688 = stablehlo.constant dense<0xFC00> : tensor<f16>
      %2545 = stablehlo.pad %91, %cst_688, low = [0, 0, 0, 0], high = [0, 1, 1, 0], interior = [0, 0, 0, 0] : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x113x113x64xf16>
      %cst_689 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2546 = "stablehlo.select_and_scatter"(%2545, %2542, %cst_689) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg340: tensor<f16>, %arg341: tensor<f16>):
        %4098 = stablehlo.compare  GE, %arg340, %arg341,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
        stablehlo.return %4098 : tensor<i1>
      }, {
      ^bb0(%arg340: tensor<f16>, %arg341: tensor<f16>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f16>
        stablehlo.return %4098 : tensor<f16>
      }) : (tensor<256x113x113x64xf16>, tensor<256x56x56x64xf16>, tensor<f16>) -> tensor<256x113x113x64xf16>
      %2547 = stablehlo.slice %2546 [0:256, 0:112, 0:112, 0:64] : (tensor<256x113x113x64xf16>) -> tensor<256x112x112x64xf16>
      %cst_690 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %2548 = stablehlo.broadcast_in_dim %cst_690, dims = [] : (tensor<f16>) -> tensor<256x112x112x64xf16>
      %2549 = stablehlo.select %93, %2547, %2548 : tensor<256x112x112x64xi1>, tensor<256x112x112x64xf16>
      %2550 = stablehlo.convert %2549 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
      %cst_691 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2551 = stablehlo.reduce(%2550 init: %cst_691) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2552 = stablehlo.reshape %2551 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2553 = stablehlo.reshape %2552 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2554 = stablehlo.multiply %76, %2550 : tensor<256x112x112x64xf32>
      %cst_692 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2555 = stablehlo.reduce(%2554 init: %cst_692) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2556 = stablehlo.reshape %2555 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2557 = stablehlo.broadcast_in_dim %84, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %2558 = stablehlo.multiply %2550, %2557 : tensor<256x112x112x64xf32>
      %2559 = stablehlo.multiply %79, %2556 : tensor<1x1x1x64xf32>
      %2560 = stablehlo.multiply %2556, %83 : tensor<1x1x1x64xf32>
      %2561 = stablehlo.reshape %2559 : (tensor<1x1x1x64xf32>) -> tensor<64xf32>
      %2562 = stablehlo.multiply %2560, %82 : tensor<1x1x1x64xf32>
      %2563 = stablehlo.negate %2558 : tensor<256x112x112x64xf32>
      %cst_693 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2564 = stablehlo.reduce(%2563 init: %cst_693) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2565 = stablehlo.reshape %2564 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %2566 = stablehlo.convert %2558 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %cst_694 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2567 = stablehlo.reduce(%2562 init: %cst_694) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_695 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2568 = stablehlo.reduce(%2565 init: %cst_695) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<1x1x1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %2569 = stablehlo.multiply %2567, %61 : tensor<64xf32>
      %2570 = stablehlo.negate %2569 : tensor<64xf32>
      %2571 = stablehlo.multiply %2570, %48 : tensor<64xf32>
      %2572 = stablehlo.add %2568, %2571 : tensor<64xf32>
      %2573 = stablehlo.broadcast_in_dim %2569, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_696 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2574 = stablehlo.pad %2573, %cst_696, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2575 = stablehlo.broadcast_in_dim %2572, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
      %cst_697 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2576 = stablehlo.pad %2575, %cst_697, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<2x64xf32>
      %2577 = stablehlo.add %2574, %2576 : tensor<2x64xf32>
      %cst_698 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2578 = stablehlo.broadcast_in_dim %cst_698, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
      %2579 = stablehlo.divide %2577, %2578 : tensor<2x64xf32>
      %2580 = "stablehlo.all_reduce"(%2579) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<2x64xf32>) -> tensor<2x64xf32>
      %2581 = stablehlo.slice %2580 [0:1, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %2582 = stablehlo.slice %2580 [1:2, 0:64] : (tensor<2x64xf32>) -> tensor<1x64xf32>
      %cst_699 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2583 = stablehlo.reduce(%2582 init: %cst_699) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_700 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2584 = stablehlo.reduce(%2581 init: %cst_700) applies stablehlo.add across dimensions = [0] : (tensor<1x64xf32>, tensor<f32>) -> tensor<64xf32>
      %cst_701 = stablehlo.constant dense<0x4A440000> : tensor<f32>
      %2585 = stablehlo.broadcast_in_dim %cst_701, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2586 = stablehlo.divide %2583, %2585 : tensor<64xf32>
      %2587 = stablehlo.broadcast_in_dim %2586, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %cst_702 = stablehlo.constant dense<0x4A440000> : tensor<f32>
      %2588 = stablehlo.broadcast_in_dim %cst_702, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2589 = stablehlo.divide %2584, %2588 : tensor<64xf32>
      %2590 = stablehlo.broadcast_in_dim %2589, dims = [3] : (tensor<64xf32>) -> tensor<256x112x112x64xf32>
      %2591 = stablehlo.multiply %2587, %29 : tensor<256x112x112x64xf32>
      %2592 = stablehlo.add %2590, %2591 : tensor<256x112x112x64xf32>
      %2593 = stablehlo.convert %2592 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %2594 = stablehlo.add %2566, %2593 : tensor<256x112x112x64xf16>
      %2595 = stablehlo.convolution(%1, %2594) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf16>, tensor<256x112x112x64xf16>) -> tensor<7x7x3x64xf16>
      %2596 = stablehlo.convert %2595 : (tensor<7x7x3x64xf16>) -> tensor<7x7x3x64xf32>
      %2597 = stablehlo.add %1493, %2596 : tensor<7x7x3x64xf32>
      %2598 = stablehlo.convert %arg337 : tensor<f32>
      %2599 = stablehlo.broadcast_in_dim %2598, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %2600 = stablehlo.divide %1547, %2599 : tensor<1000xf32>
      %2601 = stablehlo.convert %arg337 : tensor<f32>
      %2602 = stablehlo.broadcast_in_dim %2601, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %2603 = stablehlo.divide %1549, %2602 : tensor<512x1000xf32>
      %2604 = stablehlo.convert %arg337 : tensor<f32>
      %2605 = stablehlo.broadcast_in_dim %2604, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2606 = stablehlo.divide %2497, %2605 : tensor<64xf32>
      %2607 = stablehlo.convert %arg337 : tensor<f32>
      %2608 = stablehlo.broadcast_in_dim %2607, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2609 = stablehlo.divide %2505, %2608 : tensor<64xf32>
      %2610 = stablehlo.convert %arg337 : tensor<f32>
      %2611 = stablehlo.broadcast_in_dim %2610, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2612 = stablehlo.divide %2445, %2611 : tensor<64xf32>
      %2613 = stablehlo.convert %arg337 : tensor<f32>
      %2614 = stablehlo.broadcast_in_dim %2613, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2615 = stablehlo.divide %2453, %2614 : tensor<64xf32>
      %2616 = stablehlo.convert %arg337 : tensor<f32>
      %2617 = stablehlo.broadcast_in_dim %2616, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2618 = stablehlo.divide %2544, %2617 : tensor<3x3x64x64xf32>
      %2619 = stablehlo.convert %arg337 : tensor<f32>
      %2620 = stablehlo.broadcast_in_dim %2619, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2621 = stablehlo.divide %2491, %2620 : tensor<3x3x64x64xf32>
      %2622 = stablehlo.convert %arg337 : tensor<f32>
      %2623 = stablehlo.broadcast_in_dim %2622, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2624 = stablehlo.divide %2392, %2623 : tensor<64xf32>
      %2625 = stablehlo.convert %arg337 : tensor<f32>
      %2626 = stablehlo.broadcast_in_dim %2625, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2627 = stablehlo.divide %2400, %2626 : tensor<64xf32>
      %2628 = stablehlo.convert %arg337 : tensor<f32>
      %2629 = stablehlo.broadcast_in_dim %2628, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2630 = stablehlo.divide %2340, %2629 : tensor<64xf32>
      %2631 = stablehlo.convert %arg337 : tensor<f32>
      %2632 = stablehlo.broadcast_in_dim %2631, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2633 = stablehlo.divide %2348, %2632 : tensor<64xf32>
      %2634 = stablehlo.convert %arg337 : tensor<f32>
      %2635 = stablehlo.broadcast_in_dim %2634, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2636 = stablehlo.divide %2439, %2635 : tensor<3x3x64x64xf32>
      %2637 = stablehlo.convert %arg337 : tensor<f32>
      %2638 = stablehlo.broadcast_in_dim %2637, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2639 = stablehlo.divide %2386, %2638 : tensor<3x3x64x64xf32>
      %2640 = stablehlo.convert %arg337 : tensor<f32>
      %2641 = stablehlo.broadcast_in_dim %2640, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2642 = stablehlo.divide %2287, %2641 : tensor<128xf32>
      %2643 = stablehlo.convert %arg337 : tensor<f32>
      %2644 = stablehlo.broadcast_in_dim %2643, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2645 = stablehlo.divide %2295, %2644 : tensor<128xf32>
      %2646 = stablehlo.convert %arg337 : tensor<f32>
      %2647 = stablehlo.broadcast_in_dim %2646, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2648 = stablehlo.divide %2235, %2647 : tensor<128xf32>
      %2649 = stablehlo.convert %arg337 : tensor<f32>
      %2650 = stablehlo.broadcast_in_dim %2649, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2651 = stablehlo.divide %2243, %2650 : tensor<128xf32>
      %2652 = stablehlo.convert %arg337 : tensor<f32>
      %2653 = stablehlo.broadcast_in_dim %2652, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %2654 = stablehlo.divide %2334, %2653 : tensor<3x3x64x128xf32>
      %2655 = stablehlo.convert %arg337 : tensor<f32>
      %2656 = stablehlo.broadcast_in_dim %2655, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2657 = stablehlo.divide %2281, %2656 : tensor<3x3x128x128xf32>
      %2658 = stablehlo.convert %arg337 : tensor<f32>
      %2659 = stablehlo.broadcast_in_dim %2658, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %2660 = stablehlo.divide %2231, %2659 : tensor<1x1x64x128xf32>
      %2661 = stablehlo.convert %arg337 : tensor<f32>
      %2662 = stablehlo.broadcast_in_dim %2661, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2663 = stablehlo.divide %2185, %2662 : tensor<128xf32>
      %2664 = stablehlo.convert %arg337 : tensor<f32>
      %2665 = stablehlo.broadcast_in_dim %2664, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2666 = stablehlo.divide %2193, %2665 : tensor<128xf32>
      %2667 = stablehlo.convert %arg337 : tensor<f32>
      %2668 = stablehlo.broadcast_in_dim %2667, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2669 = stablehlo.divide %2132, %2668 : tensor<128xf32>
      %2670 = stablehlo.convert %arg337 : tensor<f32>
      %2671 = stablehlo.broadcast_in_dim %2670, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2672 = stablehlo.divide %2140, %2671 : tensor<128xf32>
      %2673 = stablehlo.convert %arg337 : tensor<f32>
      %2674 = stablehlo.broadcast_in_dim %2673, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2675 = stablehlo.divide %2080, %2674 : tensor<128xf32>
      %2676 = stablehlo.convert %arg337 : tensor<f32>
      %2677 = stablehlo.broadcast_in_dim %2676, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2678 = stablehlo.divide %2088, %2677 : tensor<128xf32>
      %2679 = stablehlo.convert %arg337 : tensor<f32>
      %2680 = stablehlo.broadcast_in_dim %2679, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2681 = stablehlo.divide %2179, %2680 : tensor<3x3x128x128xf32>
      %2682 = stablehlo.convert %arg337 : tensor<f32>
      %2683 = stablehlo.broadcast_in_dim %2682, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2684 = stablehlo.divide %2126, %2683 : tensor<3x3x128x128xf32>
      %2685 = stablehlo.convert %arg337 : tensor<f32>
      %2686 = stablehlo.broadcast_in_dim %2685, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2687 = stablehlo.divide %2027, %2686 : tensor<256xf32>
      %2688 = stablehlo.convert %arg337 : tensor<f32>
      %2689 = stablehlo.broadcast_in_dim %2688, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2690 = stablehlo.divide %2035, %2689 : tensor<256xf32>
      %2691 = stablehlo.convert %arg337 : tensor<f32>
      %2692 = stablehlo.broadcast_in_dim %2691, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2693 = stablehlo.divide %1975, %2692 : tensor<256xf32>
      %2694 = stablehlo.convert %arg337 : tensor<f32>
      %2695 = stablehlo.broadcast_in_dim %2694, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2696 = stablehlo.divide %1983, %2695 : tensor<256xf32>
      %2697 = stablehlo.convert %arg337 : tensor<f32>
      %2698 = stablehlo.broadcast_in_dim %2697, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %2699 = stablehlo.divide %2074, %2698 : tensor<3x3x128x256xf32>
      %2700 = stablehlo.convert %arg337 : tensor<f32>
      %2701 = stablehlo.broadcast_in_dim %2700, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2702 = stablehlo.divide %2021, %2701 : tensor<3x3x256x256xf32>
      %2703 = stablehlo.convert %arg337 : tensor<f32>
      %2704 = stablehlo.broadcast_in_dim %2703, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %2705 = stablehlo.divide %1971, %2704 : tensor<1x1x128x256xf32>
      %2706 = stablehlo.convert %arg337 : tensor<f32>
      %2707 = stablehlo.broadcast_in_dim %2706, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2708 = stablehlo.divide %1925, %2707 : tensor<256xf32>
      %2709 = stablehlo.convert %arg337 : tensor<f32>
      %2710 = stablehlo.broadcast_in_dim %2709, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2711 = stablehlo.divide %1933, %2710 : tensor<256xf32>
      %2712 = stablehlo.convert %arg337 : tensor<f32>
      %2713 = stablehlo.broadcast_in_dim %2712, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2714 = stablehlo.divide %1872, %2713 : tensor<256xf32>
      %2715 = stablehlo.convert %arg337 : tensor<f32>
      %2716 = stablehlo.broadcast_in_dim %2715, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2717 = stablehlo.divide %1880, %2716 : tensor<256xf32>
      %2718 = stablehlo.convert %arg337 : tensor<f32>
      %2719 = stablehlo.broadcast_in_dim %2718, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2720 = stablehlo.divide %1820, %2719 : tensor<256xf32>
      %2721 = stablehlo.convert %arg337 : tensor<f32>
      %2722 = stablehlo.broadcast_in_dim %2721, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2723 = stablehlo.divide %1828, %2722 : tensor<256xf32>
      %2724 = stablehlo.convert %arg337 : tensor<f32>
      %2725 = stablehlo.broadcast_in_dim %2724, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2726 = stablehlo.divide %1919, %2725 : tensor<3x3x256x256xf32>
      %2727 = stablehlo.convert %arg337 : tensor<f32>
      %2728 = stablehlo.broadcast_in_dim %2727, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2729 = stablehlo.divide %1866, %2728 : tensor<3x3x256x256xf32>
      %2730 = stablehlo.convert %arg337 : tensor<f32>
      %2731 = stablehlo.broadcast_in_dim %2730, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2732 = stablehlo.divide %1767, %2731 : tensor<512xf32>
      %2733 = stablehlo.convert %arg337 : tensor<f32>
      %2734 = stablehlo.broadcast_in_dim %2733, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2735 = stablehlo.divide %1775, %2734 : tensor<512xf32>
      %2736 = stablehlo.convert %arg337 : tensor<f32>
      %2737 = stablehlo.broadcast_in_dim %2736, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2738 = stablehlo.divide %1715, %2737 : tensor<512xf32>
      %2739 = stablehlo.convert %arg337 : tensor<f32>
      %2740 = stablehlo.broadcast_in_dim %2739, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2741 = stablehlo.divide %1723, %2740 : tensor<512xf32>
      %2742 = stablehlo.convert %arg337 : tensor<f32>
      %2743 = stablehlo.broadcast_in_dim %2742, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %2744 = stablehlo.divide %1814, %2743 : tensor<3x3x256x512xf32>
      %2745 = stablehlo.convert %arg337 : tensor<f32>
      %2746 = stablehlo.broadcast_in_dim %2745, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2747 = stablehlo.divide %1761, %2746 : tensor<3x3x512x512xf32>
      %2748 = stablehlo.convert %arg337 : tensor<f32>
      %2749 = stablehlo.broadcast_in_dim %2748, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %2750 = stablehlo.divide %1711, %2749 : tensor<1x1x256x512xf32>
      %2751 = stablehlo.convert %arg337 : tensor<f32>
      %2752 = stablehlo.broadcast_in_dim %2751, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2753 = stablehlo.divide %1665, %2752 : tensor<512xf32>
      %2754 = stablehlo.convert %arg337 : tensor<f32>
      %2755 = stablehlo.broadcast_in_dim %2754, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2756 = stablehlo.divide %1673, %2755 : tensor<512xf32>
      %2757 = stablehlo.convert %arg337 : tensor<f32>
      %2758 = stablehlo.broadcast_in_dim %2757, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2759 = stablehlo.divide %1612, %2758 : tensor<512xf32>
      %2760 = stablehlo.convert %arg337 : tensor<f32>
      %2761 = stablehlo.broadcast_in_dim %2760, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2762 = stablehlo.divide %1620, %2761 : tensor<512xf32>
      %2763 = stablehlo.convert %arg337 : tensor<f32>
      %2764 = stablehlo.broadcast_in_dim %2763, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2765 = stablehlo.divide %1560, %2764 : tensor<512xf32>
      %2766 = stablehlo.convert %arg337 : tensor<f32>
      %2767 = stablehlo.broadcast_in_dim %2766, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2768 = stablehlo.divide %1568, %2767 : tensor<512xf32>
      %2769 = stablehlo.convert %arg337 : tensor<f32>
      %2770 = stablehlo.broadcast_in_dim %2769, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2771 = stablehlo.divide %1659, %2770 : tensor<3x3x512x512xf32>
      %2772 = stablehlo.convert %arg337 : tensor<f32>
      %2773 = stablehlo.broadcast_in_dim %2772, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2774 = stablehlo.divide %1606, %2773 : tensor<3x3x512x512xf32>
      %2775 = stablehlo.convert %arg337 : tensor<f32>
      %2776 = stablehlo.broadcast_in_dim %2775, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2777 = stablehlo.divide %2553, %2776 : tensor<64xf32>
      %2778 = stablehlo.convert %arg337 : tensor<f32>
      %2779 = stablehlo.broadcast_in_dim %2778, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2780 = stablehlo.divide %2561, %2779 : tensor<64xf32>
      %2781 = stablehlo.convert %arg337 : tensor<f32>
      %2782 = stablehlo.broadcast_in_dim %2781, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %2783 = stablehlo.divide %2597, %2782 : tensor<7x7x3x64xf32>
      %2784 = "stablehlo.all_reduce"(%2600) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<1000xf32>) -> tensor<1000xf32>
      %2785 = "stablehlo.all_reduce"(%2603) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512x1000xf32>) -> tensor<512x1000xf32>
      %2786 = "stablehlo.all_reduce"(%2606) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2787 = "stablehlo.all_reduce"(%2609) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2788 = "stablehlo.all_reduce"(%2612) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2789 = "stablehlo.all_reduce"(%2615) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2790 = "stablehlo.all_reduce"(%2618) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %2791 = "stablehlo.all_reduce"(%2621) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %2792 = "stablehlo.all_reduce"(%2624) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2793 = "stablehlo.all_reduce"(%2627) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2794 = "stablehlo.all_reduce"(%2630) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2795 = "stablehlo.all_reduce"(%2633) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2796 = "stablehlo.all_reduce"(%2636) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %2797 = "stablehlo.all_reduce"(%2639) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %2798 = "stablehlo.all_reduce"(%2642) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2799 = "stablehlo.all_reduce"(%2645) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2800 = "stablehlo.all_reduce"(%2648) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2801 = "stablehlo.all_reduce"(%2651) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2802 = "stablehlo.all_reduce"(%2654) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
      %2803 = "stablehlo.all_reduce"(%2657) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %2804 = "stablehlo.all_reduce"(%2660) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
      %2805 = "stablehlo.all_reduce"(%2663) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2806 = "stablehlo.all_reduce"(%2666) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2807 = "stablehlo.all_reduce"(%2669) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2808 = "stablehlo.all_reduce"(%2672) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2809 = "stablehlo.all_reduce"(%2675) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2810 = "stablehlo.all_reduce"(%2678) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %2811 = "stablehlo.all_reduce"(%2681) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %2812 = "stablehlo.all_reduce"(%2684) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %2813 = "stablehlo.all_reduce"(%2687) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2814 = "stablehlo.all_reduce"(%2690) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2815 = "stablehlo.all_reduce"(%2693) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2816 = "stablehlo.all_reduce"(%2696) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2817 = "stablehlo.all_reduce"(%2699) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
      %2818 = "stablehlo.all_reduce"(%2702) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %2819 = "stablehlo.all_reduce"(%2705) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
      %2820 = "stablehlo.all_reduce"(%2708) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2821 = "stablehlo.all_reduce"(%2711) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2822 = "stablehlo.all_reduce"(%2714) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2823 = "stablehlo.all_reduce"(%2717) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2824 = "stablehlo.all_reduce"(%2720) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2825 = "stablehlo.all_reduce"(%2723) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %2826 = "stablehlo.all_reduce"(%2726) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %2827 = "stablehlo.all_reduce"(%2729) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %2828 = "stablehlo.all_reduce"(%2732) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2829 = "stablehlo.all_reduce"(%2735) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2830 = "stablehlo.all_reduce"(%2738) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2831 = "stablehlo.all_reduce"(%2741) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2832 = "stablehlo.all_reduce"(%2744) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
      %2833 = "stablehlo.all_reduce"(%2747) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %2834 = "stablehlo.all_reduce"(%2750) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
      %2835 = "stablehlo.all_reduce"(%2753) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2836 = "stablehlo.all_reduce"(%2756) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2837 = "stablehlo.all_reduce"(%2759) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2838 = "stablehlo.all_reduce"(%2762) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2839 = "stablehlo.all_reduce"(%2765) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2840 = "stablehlo.all_reduce"(%2768) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %2841 = "stablehlo.all_reduce"(%2771) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %2842 = "stablehlo.all_reduce"(%2774) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %2843 = "stablehlo.all_reduce"(%2777) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2844 = "stablehlo.all_reduce"(%2780) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %2845 = "stablehlo.all_reduce"(%2783) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
      %cst_703 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2846 = stablehlo.broadcast_in_dim %cst_703, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %2847 = stablehlo.divide %2784, %2846 : tensor<1000xf32>
      %cst_704 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2848 = stablehlo.broadcast_in_dim %cst_704, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %2849 = stablehlo.divide %2785, %2848 : tensor<512x1000xf32>
      %cst_705 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2850 = stablehlo.broadcast_in_dim %cst_705, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2851 = stablehlo.divide %2786, %2850 : tensor<64xf32>
      %cst_706 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2852 = stablehlo.broadcast_in_dim %cst_706, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2853 = stablehlo.divide %2787, %2852 : tensor<64xf32>
      %cst_707 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2854 = stablehlo.broadcast_in_dim %cst_707, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2855 = stablehlo.divide %2788, %2854 : tensor<64xf32>
      %cst_708 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2856 = stablehlo.broadcast_in_dim %cst_708, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2857 = stablehlo.divide %2789, %2856 : tensor<64xf32>
      %cst_709 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2858 = stablehlo.broadcast_in_dim %cst_709, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2859 = stablehlo.divide %2790, %2858 : tensor<3x3x64x64xf32>
      %cst_710 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2860 = stablehlo.broadcast_in_dim %cst_710, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2861 = stablehlo.divide %2791, %2860 : tensor<3x3x64x64xf32>
      %cst_711 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2862 = stablehlo.broadcast_in_dim %cst_711, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2863 = stablehlo.divide %2792, %2862 : tensor<64xf32>
      %cst_712 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2864 = stablehlo.broadcast_in_dim %cst_712, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2865 = stablehlo.divide %2793, %2864 : tensor<64xf32>
      %cst_713 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2866 = stablehlo.broadcast_in_dim %cst_713, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2867 = stablehlo.divide %2794, %2866 : tensor<64xf32>
      %cst_714 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2868 = stablehlo.broadcast_in_dim %cst_714, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2869 = stablehlo.divide %2795, %2868 : tensor<64xf32>
      %cst_715 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2870 = stablehlo.broadcast_in_dim %cst_715, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2871 = stablehlo.divide %2796, %2870 : tensor<3x3x64x64xf32>
      %cst_716 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2872 = stablehlo.broadcast_in_dim %cst_716, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %2873 = stablehlo.divide %2797, %2872 : tensor<3x3x64x64xf32>
      %cst_717 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2874 = stablehlo.broadcast_in_dim %cst_717, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2875 = stablehlo.divide %2798, %2874 : tensor<128xf32>
      %cst_718 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2876 = stablehlo.broadcast_in_dim %cst_718, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2877 = stablehlo.divide %2799, %2876 : tensor<128xf32>
      %cst_719 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2878 = stablehlo.broadcast_in_dim %cst_719, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2879 = stablehlo.divide %2800, %2878 : tensor<128xf32>
      %cst_720 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2880 = stablehlo.broadcast_in_dim %cst_720, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2881 = stablehlo.divide %2801, %2880 : tensor<128xf32>
      %cst_721 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2882 = stablehlo.broadcast_in_dim %cst_721, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %2883 = stablehlo.divide %2802, %2882 : tensor<3x3x64x128xf32>
      %cst_722 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2884 = stablehlo.broadcast_in_dim %cst_722, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2885 = stablehlo.divide %2803, %2884 : tensor<3x3x128x128xf32>
      %cst_723 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2886 = stablehlo.broadcast_in_dim %cst_723, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %2887 = stablehlo.divide %2804, %2886 : tensor<1x1x64x128xf32>
      %cst_724 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2888 = stablehlo.broadcast_in_dim %cst_724, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2889 = stablehlo.divide %2805, %2888 : tensor<128xf32>
      %cst_725 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2890 = stablehlo.broadcast_in_dim %cst_725, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2891 = stablehlo.divide %2806, %2890 : tensor<128xf32>
      %cst_726 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2892 = stablehlo.broadcast_in_dim %cst_726, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2893 = stablehlo.divide %2807, %2892 : tensor<128xf32>
      %cst_727 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2894 = stablehlo.broadcast_in_dim %cst_727, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2895 = stablehlo.divide %2808, %2894 : tensor<128xf32>
      %cst_728 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2896 = stablehlo.broadcast_in_dim %cst_728, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2897 = stablehlo.divide %2809, %2896 : tensor<128xf32>
      %cst_729 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2898 = stablehlo.broadcast_in_dim %cst_729, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %2899 = stablehlo.divide %2810, %2898 : tensor<128xf32>
      %cst_730 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2900 = stablehlo.broadcast_in_dim %cst_730, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2901 = stablehlo.divide %2811, %2900 : tensor<3x3x128x128xf32>
      %cst_731 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2902 = stablehlo.broadcast_in_dim %cst_731, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %2903 = stablehlo.divide %2812, %2902 : tensor<3x3x128x128xf32>
      %cst_732 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2904 = stablehlo.broadcast_in_dim %cst_732, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2905 = stablehlo.divide %2813, %2904 : tensor<256xf32>
      %cst_733 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2906 = stablehlo.broadcast_in_dim %cst_733, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2907 = stablehlo.divide %2814, %2906 : tensor<256xf32>
      %cst_734 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2908 = stablehlo.broadcast_in_dim %cst_734, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2909 = stablehlo.divide %2815, %2908 : tensor<256xf32>
      %cst_735 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2910 = stablehlo.broadcast_in_dim %cst_735, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2911 = stablehlo.divide %2816, %2910 : tensor<256xf32>
      %cst_736 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2912 = stablehlo.broadcast_in_dim %cst_736, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %2913 = stablehlo.divide %2817, %2912 : tensor<3x3x128x256xf32>
      %cst_737 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2914 = stablehlo.broadcast_in_dim %cst_737, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2915 = stablehlo.divide %2818, %2914 : tensor<3x3x256x256xf32>
      %cst_738 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2916 = stablehlo.broadcast_in_dim %cst_738, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %2917 = stablehlo.divide %2819, %2916 : tensor<1x1x128x256xf32>
      %cst_739 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2918 = stablehlo.broadcast_in_dim %cst_739, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2919 = stablehlo.divide %2820, %2918 : tensor<256xf32>
      %cst_740 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2920 = stablehlo.broadcast_in_dim %cst_740, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2921 = stablehlo.divide %2821, %2920 : tensor<256xf32>
      %cst_741 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2922 = stablehlo.broadcast_in_dim %cst_741, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2923 = stablehlo.divide %2822, %2922 : tensor<256xf32>
      %cst_742 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2924 = stablehlo.broadcast_in_dim %cst_742, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2925 = stablehlo.divide %2823, %2924 : tensor<256xf32>
      %cst_743 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2926 = stablehlo.broadcast_in_dim %cst_743, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2927 = stablehlo.divide %2824, %2926 : tensor<256xf32>
      %cst_744 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2928 = stablehlo.broadcast_in_dim %cst_744, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %2929 = stablehlo.divide %2825, %2928 : tensor<256xf32>
      %cst_745 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2930 = stablehlo.broadcast_in_dim %cst_745, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2931 = stablehlo.divide %2826, %2930 : tensor<3x3x256x256xf32>
      %cst_746 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2932 = stablehlo.broadcast_in_dim %cst_746, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %2933 = stablehlo.divide %2827, %2932 : tensor<3x3x256x256xf32>
      %cst_747 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2934 = stablehlo.broadcast_in_dim %cst_747, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2935 = stablehlo.divide %2828, %2934 : tensor<512xf32>
      %cst_748 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2936 = stablehlo.broadcast_in_dim %cst_748, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2937 = stablehlo.divide %2829, %2936 : tensor<512xf32>
      %cst_749 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2938 = stablehlo.broadcast_in_dim %cst_749, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2939 = stablehlo.divide %2830, %2938 : tensor<512xf32>
      %cst_750 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2940 = stablehlo.broadcast_in_dim %cst_750, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2941 = stablehlo.divide %2831, %2940 : tensor<512xf32>
      %cst_751 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2942 = stablehlo.broadcast_in_dim %cst_751, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %2943 = stablehlo.divide %2832, %2942 : tensor<3x3x256x512xf32>
      %cst_752 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2944 = stablehlo.broadcast_in_dim %cst_752, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2945 = stablehlo.divide %2833, %2944 : tensor<3x3x512x512xf32>
      %cst_753 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2946 = stablehlo.broadcast_in_dim %cst_753, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %2947 = stablehlo.divide %2834, %2946 : tensor<1x1x256x512xf32>
      %cst_754 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2948 = stablehlo.broadcast_in_dim %cst_754, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2949 = stablehlo.divide %2835, %2948 : tensor<512xf32>
      %cst_755 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2950 = stablehlo.broadcast_in_dim %cst_755, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2951 = stablehlo.divide %2836, %2950 : tensor<512xf32>
      %cst_756 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2952 = stablehlo.broadcast_in_dim %cst_756, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2953 = stablehlo.divide %2837, %2952 : tensor<512xf32>
      %cst_757 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2954 = stablehlo.broadcast_in_dim %cst_757, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2955 = stablehlo.divide %2838, %2954 : tensor<512xf32>
      %cst_758 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2956 = stablehlo.broadcast_in_dim %cst_758, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2957 = stablehlo.divide %2839, %2956 : tensor<512xf32>
      %cst_759 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2958 = stablehlo.broadcast_in_dim %cst_759, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %2959 = stablehlo.divide %2840, %2958 : tensor<512xf32>
      %cst_760 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2960 = stablehlo.broadcast_in_dim %cst_760, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2961 = stablehlo.divide %2841, %2960 : tensor<3x3x512x512xf32>
      %cst_761 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2962 = stablehlo.broadcast_in_dim %cst_761, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %2963 = stablehlo.divide %2842, %2962 : tensor<3x3x512x512xf32>
      %cst_762 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2964 = stablehlo.broadcast_in_dim %cst_762, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2965 = stablehlo.divide %2843, %2964 : tensor<64xf32>
      %cst_763 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2966 = stablehlo.broadcast_in_dim %cst_763, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %2967 = stablehlo.divide %2844, %2966 : tensor<64xf32>
      %cst_764 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2968 = stablehlo.broadcast_in_dim %cst_764, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %2969 = stablehlo.divide %2845, %2968 : tensor<7x7x3x64xf32>
      %2970 = stablehlo.is_finite %2847 : (tensor<1000xf32>) -> tensor<1000xi1>
      %c_765 = stablehlo.constant dense<true> : tensor<i1>
      %2971 = stablehlo.reduce(%2970 init: %c_765) applies stablehlo.and across dimensions = [0] : (tensor<1000xi1>, tensor<i1>) -> tensor<i1>
      %c_766 = stablehlo.constant dense<true> : tensor<i1>
      %2972 = stablehlo.and %c_766, %2971 : tensor<i1>
      %2973 = stablehlo.is_finite %2849 : (tensor<512x1000xf32>) -> tensor<512x1000xi1>
      %c_767 = stablehlo.constant dense<true> : tensor<i1>
      %2974 = stablehlo.reduce(%2973 init: %c_767) applies stablehlo.and across dimensions = [0, 1] : (tensor<512x1000xi1>, tensor<i1>) -> tensor<i1>
      %2975 = stablehlo.and %2972, %2974 : tensor<i1>
      %2976 = stablehlo.is_finite %2851 : (tensor<64xf32>) -> tensor<64xi1>
      %c_768 = stablehlo.constant dense<true> : tensor<i1>
      %2977 = stablehlo.reduce(%2976 init: %c_768) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %2978 = stablehlo.and %2975, %2977 : tensor<i1>
      %2979 = stablehlo.is_finite %2853 : (tensor<64xf32>) -> tensor<64xi1>
      %c_769 = stablehlo.constant dense<true> : tensor<i1>
      %2980 = stablehlo.reduce(%2979 init: %c_769) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %2981 = stablehlo.and %2978, %2980 : tensor<i1>
      %2982 = stablehlo.is_finite %2855 : (tensor<64xf32>) -> tensor<64xi1>
      %c_770 = stablehlo.constant dense<true> : tensor<i1>
      %2983 = stablehlo.reduce(%2982 init: %c_770) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %2984 = stablehlo.and %2981, %2983 : tensor<i1>
      %2985 = stablehlo.is_finite %2857 : (tensor<64xf32>) -> tensor<64xi1>
      %c_771 = stablehlo.constant dense<true> : tensor<i1>
      %2986 = stablehlo.reduce(%2985 init: %c_771) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %2987 = stablehlo.and %2984, %2986 : tensor<i1>
      %2988 = stablehlo.is_finite %2859 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_772 = stablehlo.constant dense<true> : tensor<i1>
      %2989 = stablehlo.reduce(%2988 init: %c_772) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %2990 = stablehlo.and %2987, %2989 : tensor<i1>
      %2991 = stablehlo.is_finite %2861 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_773 = stablehlo.constant dense<true> : tensor<i1>
      %2992 = stablehlo.reduce(%2991 init: %c_773) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %2993 = stablehlo.and %2990, %2992 : tensor<i1>
      %2994 = stablehlo.is_finite %2863 : (tensor<64xf32>) -> tensor<64xi1>
      %c_774 = stablehlo.constant dense<true> : tensor<i1>
      %2995 = stablehlo.reduce(%2994 init: %c_774) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %2996 = stablehlo.and %2993, %2995 : tensor<i1>
      %2997 = stablehlo.is_finite %2865 : (tensor<64xf32>) -> tensor<64xi1>
      %c_775 = stablehlo.constant dense<true> : tensor<i1>
      %2998 = stablehlo.reduce(%2997 init: %c_775) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %2999 = stablehlo.and %2996, %2998 : tensor<i1>
      %3000 = stablehlo.is_finite %2867 : (tensor<64xf32>) -> tensor<64xi1>
      %c_776 = stablehlo.constant dense<true> : tensor<i1>
      %3001 = stablehlo.reduce(%3000 init: %c_776) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %3002 = stablehlo.and %2999, %3001 : tensor<i1>
      %3003 = stablehlo.is_finite %2869 : (tensor<64xf32>) -> tensor<64xi1>
      %c_777 = stablehlo.constant dense<true> : tensor<i1>
      %3004 = stablehlo.reduce(%3003 init: %c_777) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %3005 = stablehlo.and %3002, %3004 : tensor<i1>
      %3006 = stablehlo.is_finite %2871 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_778 = stablehlo.constant dense<true> : tensor<i1>
      %3007 = stablehlo.reduce(%3006 init: %c_778) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %3008 = stablehlo.and %3005, %3007 : tensor<i1>
      %3009 = stablehlo.is_finite %2873 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xi1>
      %c_779 = stablehlo.constant dense<true> : tensor<i1>
      %3010 = stablehlo.reduce(%3009 init: %c_779) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x64xi1>, tensor<i1>) -> tensor<i1>
      %3011 = stablehlo.and %3008, %3010 : tensor<i1>
      %3012 = stablehlo.is_finite %2875 : (tensor<128xf32>) -> tensor<128xi1>
      %c_780 = stablehlo.constant dense<true> : tensor<i1>
      %3013 = stablehlo.reduce(%3012 init: %c_780) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3014 = stablehlo.and %3011, %3013 : tensor<i1>
      %3015 = stablehlo.is_finite %2877 : (tensor<128xf32>) -> tensor<128xi1>
      %c_781 = stablehlo.constant dense<true> : tensor<i1>
      %3016 = stablehlo.reduce(%3015 init: %c_781) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3017 = stablehlo.and %3014, %3016 : tensor<i1>
      %3018 = stablehlo.is_finite %2879 : (tensor<128xf32>) -> tensor<128xi1>
      %c_782 = stablehlo.constant dense<true> : tensor<i1>
      %3019 = stablehlo.reduce(%3018 init: %c_782) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3020 = stablehlo.and %3017, %3019 : tensor<i1>
      %3021 = stablehlo.is_finite %2881 : (tensor<128xf32>) -> tensor<128xi1>
      %c_783 = stablehlo.constant dense<true> : tensor<i1>
      %3022 = stablehlo.reduce(%3021 init: %c_783) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3023 = stablehlo.and %3020, %3022 : tensor<i1>
      %3024 = stablehlo.is_finite %2883 : (tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xi1>
      %c_784 = stablehlo.constant dense<true> : tensor<i1>
      %3025 = stablehlo.reduce(%3024 init: %c_784) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x64x128xi1>, tensor<i1>) -> tensor<i1>
      %3026 = stablehlo.and %3023, %3025 : tensor<i1>
      %3027 = stablehlo.is_finite %2885 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_785 = stablehlo.constant dense<true> : tensor<i1>
      %3028 = stablehlo.reduce(%3027 init: %c_785) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %3029 = stablehlo.and %3026, %3028 : tensor<i1>
      %3030 = stablehlo.is_finite %2887 : (tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xi1>
      %c_786 = stablehlo.constant dense<true> : tensor<i1>
      %3031 = stablehlo.reduce(%3030 init: %c_786) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x64x128xi1>, tensor<i1>) -> tensor<i1>
      %3032 = stablehlo.and %3029, %3031 : tensor<i1>
      %3033 = stablehlo.is_finite %2889 : (tensor<128xf32>) -> tensor<128xi1>
      %c_787 = stablehlo.constant dense<true> : tensor<i1>
      %3034 = stablehlo.reduce(%3033 init: %c_787) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3035 = stablehlo.and %3032, %3034 : tensor<i1>
      %3036 = stablehlo.is_finite %2891 : (tensor<128xf32>) -> tensor<128xi1>
      %c_788 = stablehlo.constant dense<true> : tensor<i1>
      %3037 = stablehlo.reduce(%3036 init: %c_788) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3038 = stablehlo.and %3035, %3037 : tensor<i1>
      %3039 = stablehlo.is_finite %2893 : (tensor<128xf32>) -> tensor<128xi1>
      %c_789 = stablehlo.constant dense<true> : tensor<i1>
      %3040 = stablehlo.reduce(%3039 init: %c_789) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3041 = stablehlo.and %3038, %3040 : tensor<i1>
      %3042 = stablehlo.is_finite %2895 : (tensor<128xf32>) -> tensor<128xi1>
      %c_790 = stablehlo.constant dense<true> : tensor<i1>
      %3043 = stablehlo.reduce(%3042 init: %c_790) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3044 = stablehlo.and %3041, %3043 : tensor<i1>
      %3045 = stablehlo.is_finite %2897 : (tensor<128xf32>) -> tensor<128xi1>
      %c_791 = stablehlo.constant dense<true> : tensor<i1>
      %3046 = stablehlo.reduce(%3045 init: %c_791) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3047 = stablehlo.and %3044, %3046 : tensor<i1>
      %3048 = stablehlo.is_finite %2899 : (tensor<128xf32>) -> tensor<128xi1>
      %c_792 = stablehlo.constant dense<true> : tensor<i1>
      %3049 = stablehlo.reduce(%3048 init: %c_792) applies stablehlo.and across dimensions = [0] : (tensor<128xi1>, tensor<i1>) -> tensor<i1>
      %3050 = stablehlo.and %3047, %3049 : tensor<i1>
      %3051 = stablehlo.is_finite %2901 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_793 = stablehlo.constant dense<true> : tensor<i1>
      %3052 = stablehlo.reduce(%3051 init: %c_793) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %3053 = stablehlo.and %3050, %3052 : tensor<i1>
      %3054 = stablehlo.is_finite %2903 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xi1>
      %c_794 = stablehlo.constant dense<true> : tensor<i1>
      %3055 = stablehlo.reduce(%3054 init: %c_794) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x128xi1>, tensor<i1>) -> tensor<i1>
      %3056 = stablehlo.and %3053, %3055 : tensor<i1>
      %3057 = stablehlo.is_finite %2905 : (tensor<256xf32>) -> tensor<256xi1>
      %c_795 = stablehlo.constant dense<true> : tensor<i1>
      %3058 = stablehlo.reduce(%3057 init: %c_795) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3059 = stablehlo.and %3056, %3058 : tensor<i1>
      %3060 = stablehlo.is_finite %2907 : (tensor<256xf32>) -> tensor<256xi1>
      %c_796 = stablehlo.constant dense<true> : tensor<i1>
      %3061 = stablehlo.reduce(%3060 init: %c_796) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3062 = stablehlo.and %3059, %3061 : tensor<i1>
      %3063 = stablehlo.is_finite %2909 : (tensor<256xf32>) -> tensor<256xi1>
      %c_797 = stablehlo.constant dense<true> : tensor<i1>
      %3064 = stablehlo.reduce(%3063 init: %c_797) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3065 = stablehlo.and %3062, %3064 : tensor<i1>
      %3066 = stablehlo.is_finite %2911 : (tensor<256xf32>) -> tensor<256xi1>
      %c_798 = stablehlo.constant dense<true> : tensor<i1>
      %3067 = stablehlo.reduce(%3066 init: %c_798) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3068 = stablehlo.and %3065, %3067 : tensor<i1>
      %3069 = stablehlo.is_finite %2913 : (tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xi1>
      %c_799 = stablehlo.constant dense<true> : tensor<i1>
      %3070 = stablehlo.reduce(%3069 init: %c_799) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x128x256xi1>, tensor<i1>) -> tensor<i1>
      %3071 = stablehlo.and %3068, %3070 : tensor<i1>
      %3072 = stablehlo.is_finite %2915 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_800 = stablehlo.constant dense<true> : tensor<i1>
      %3073 = stablehlo.reduce(%3072 init: %c_800) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %3074 = stablehlo.and %3071, %3073 : tensor<i1>
      %3075 = stablehlo.is_finite %2917 : (tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xi1>
      %c_801 = stablehlo.constant dense<true> : tensor<i1>
      %3076 = stablehlo.reduce(%3075 init: %c_801) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x128x256xi1>, tensor<i1>) -> tensor<i1>
      %3077 = stablehlo.and %3074, %3076 : tensor<i1>
      %3078 = stablehlo.is_finite %2919 : (tensor<256xf32>) -> tensor<256xi1>
      %c_802 = stablehlo.constant dense<true> : tensor<i1>
      %3079 = stablehlo.reduce(%3078 init: %c_802) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3080 = stablehlo.and %3077, %3079 : tensor<i1>
      %3081 = stablehlo.is_finite %2921 : (tensor<256xf32>) -> tensor<256xi1>
      %c_803 = stablehlo.constant dense<true> : tensor<i1>
      %3082 = stablehlo.reduce(%3081 init: %c_803) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3083 = stablehlo.and %3080, %3082 : tensor<i1>
      %3084 = stablehlo.is_finite %2923 : (tensor<256xf32>) -> tensor<256xi1>
      %c_804 = stablehlo.constant dense<true> : tensor<i1>
      %3085 = stablehlo.reduce(%3084 init: %c_804) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3086 = stablehlo.and %3083, %3085 : tensor<i1>
      %3087 = stablehlo.is_finite %2925 : (tensor<256xf32>) -> tensor<256xi1>
      %c_805 = stablehlo.constant dense<true> : tensor<i1>
      %3088 = stablehlo.reduce(%3087 init: %c_805) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3089 = stablehlo.and %3086, %3088 : tensor<i1>
      %3090 = stablehlo.is_finite %2927 : (tensor<256xf32>) -> tensor<256xi1>
      %c_806 = stablehlo.constant dense<true> : tensor<i1>
      %3091 = stablehlo.reduce(%3090 init: %c_806) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3092 = stablehlo.and %3089, %3091 : tensor<i1>
      %3093 = stablehlo.is_finite %2929 : (tensor<256xf32>) -> tensor<256xi1>
      %c_807 = stablehlo.constant dense<true> : tensor<i1>
      %3094 = stablehlo.reduce(%3093 init: %c_807) applies stablehlo.and across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
      %3095 = stablehlo.and %3092, %3094 : tensor<i1>
      %3096 = stablehlo.is_finite %2931 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_808 = stablehlo.constant dense<true> : tensor<i1>
      %3097 = stablehlo.reduce(%3096 init: %c_808) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %3098 = stablehlo.and %3095, %3097 : tensor<i1>
      %3099 = stablehlo.is_finite %2933 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xi1>
      %c_809 = stablehlo.constant dense<true> : tensor<i1>
      %3100 = stablehlo.reduce(%3099 init: %c_809) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x256xi1>, tensor<i1>) -> tensor<i1>
      %3101 = stablehlo.and %3098, %3100 : tensor<i1>
      %3102 = stablehlo.is_finite %2935 : (tensor<512xf32>) -> tensor<512xi1>
      %c_810 = stablehlo.constant dense<true> : tensor<i1>
      %3103 = stablehlo.reduce(%3102 init: %c_810) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3104 = stablehlo.and %3101, %3103 : tensor<i1>
      %3105 = stablehlo.is_finite %2937 : (tensor<512xf32>) -> tensor<512xi1>
      %c_811 = stablehlo.constant dense<true> : tensor<i1>
      %3106 = stablehlo.reduce(%3105 init: %c_811) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3107 = stablehlo.and %3104, %3106 : tensor<i1>
      %3108 = stablehlo.is_finite %2939 : (tensor<512xf32>) -> tensor<512xi1>
      %c_812 = stablehlo.constant dense<true> : tensor<i1>
      %3109 = stablehlo.reduce(%3108 init: %c_812) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3110 = stablehlo.and %3107, %3109 : tensor<i1>
      %3111 = stablehlo.is_finite %2941 : (tensor<512xf32>) -> tensor<512xi1>
      %c_813 = stablehlo.constant dense<true> : tensor<i1>
      %3112 = stablehlo.reduce(%3111 init: %c_813) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3113 = stablehlo.and %3110, %3112 : tensor<i1>
      %3114 = stablehlo.is_finite %2943 : (tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xi1>
      %c_814 = stablehlo.constant dense<true> : tensor<i1>
      %3115 = stablehlo.reduce(%3114 init: %c_814) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x256x512xi1>, tensor<i1>) -> tensor<i1>
      %3116 = stablehlo.and %3113, %3115 : tensor<i1>
      %3117 = stablehlo.is_finite %2945 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
      %c_815 = stablehlo.constant dense<true> : tensor<i1>
      %3118 = stablehlo.reduce(%3117 init: %c_815) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
      %3119 = stablehlo.and %3116, %3118 : tensor<i1>
      %3120 = stablehlo.is_finite %2947 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xi1>
      %c_816 = stablehlo.constant dense<true> : tensor<i1>
      %3121 = stablehlo.reduce(%3120 init: %c_816) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<1x1x256x512xi1>, tensor<i1>) -> tensor<i1>
      %3122 = stablehlo.and %3119, %3121 : tensor<i1>
      %3123 = stablehlo.is_finite %2949 : (tensor<512xf32>) -> tensor<512xi1>
      %c_817 = stablehlo.constant dense<true> : tensor<i1>
      %3124 = stablehlo.reduce(%3123 init: %c_817) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3125 = stablehlo.and %3122, %3124 : tensor<i1>
      %3126 = stablehlo.is_finite %2951 : (tensor<512xf32>) -> tensor<512xi1>
      %c_818 = stablehlo.constant dense<true> : tensor<i1>
      %3127 = stablehlo.reduce(%3126 init: %c_818) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3128 = stablehlo.and %3125, %3127 : tensor<i1>
      %3129 = stablehlo.is_finite %2953 : (tensor<512xf32>) -> tensor<512xi1>
      %c_819 = stablehlo.constant dense<true> : tensor<i1>
      %3130 = stablehlo.reduce(%3129 init: %c_819) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3131 = stablehlo.and %3128, %3130 : tensor<i1>
      %3132 = stablehlo.is_finite %2955 : (tensor<512xf32>) -> tensor<512xi1>
      %c_820 = stablehlo.constant dense<true> : tensor<i1>
      %3133 = stablehlo.reduce(%3132 init: %c_820) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3134 = stablehlo.and %3131, %3133 : tensor<i1>
      %3135 = stablehlo.is_finite %2957 : (tensor<512xf32>) -> tensor<512xi1>
      %c_821 = stablehlo.constant dense<true> : tensor<i1>
      %3136 = stablehlo.reduce(%3135 init: %c_821) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3137 = stablehlo.and %3134, %3136 : tensor<i1>
      %3138 = stablehlo.is_finite %2959 : (tensor<512xf32>) -> tensor<512xi1>
      %c_822 = stablehlo.constant dense<true> : tensor<i1>
      %3139 = stablehlo.reduce(%3138 init: %c_822) applies stablehlo.and across dimensions = [0] : (tensor<512xi1>, tensor<i1>) -> tensor<i1>
      %3140 = stablehlo.and %3137, %3139 : tensor<i1>
      %3141 = stablehlo.is_finite %2961 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
      %c_823 = stablehlo.constant dense<true> : tensor<i1>
      %3142 = stablehlo.reduce(%3141 init: %c_823) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
      %3143 = stablehlo.and %3140, %3142 : tensor<i1>
      %3144 = stablehlo.is_finite %2963 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xi1>
      %c_824 = stablehlo.constant dense<true> : tensor<i1>
      %3145 = stablehlo.reduce(%3144 init: %c_824) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<3x3x512x512xi1>, tensor<i1>) -> tensor<i1>
      %3146 = stablehlo.and %3143, %3145 : tensor<i1>
      %3147 = stablehlo.is_finite %2965 : (tensor<64xf32>) -> tensor<64xi1>
      %c_825 = stablehlo.constant dense<true> : tensor<i1>
      %3148 = stablehlo.reduce(%3147 init: %c_825) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %3149 = stablehlo.and %3146, %3148 : tensor<i1>
      %3150 = stablehlo.is_finite %2967 : (tensor<64xf32>) -> tensor<64xi1>
      %c_826 = stablehlo.constant dense<true> : tensor<i1>
      %3151 = stablehlo.reduce(%3150 init: %c_826) applies stablehlo.and across dimensions = [0] : (tensor<64xi1>, tensor<i1>) -> tensor<i1>
      %3152 = stablehlo.and %3149, %3151 : tensor<i1>
      %3153 = stablehlo.is_finite %2969 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xi1>
      %c_827 = stablehlo.constant dense<true> : tensor<i1>
      %3154 = stablehlo.reduce(%3153 init: %c_827) applies stablehlo.and across dimensions = [0, 1, 2, 3] : (tensor<7x7x3x64xi1>, tensor<i1>) -> tensor<i1>
      %3155 = stablehlo.and %3152, %3154 : tensor<i1>
      %c_828 = stablehlo.constant dense<2000> : tensor<i32>
      %3156 = stablehlo.compare  EQ, %arg336, %c_828,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_829 = stablehlo.constant dense<false> : tensor<i1>
      %3157 = stablehlo.broadcast_in_dim %c_829, dims = [] : (tensor<i1>) -> tensor<i1>
      %3158 = stablehlo.compare  NE, %3156, %3157,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %3159 = stablehlo.convert %3158 : tensor<i1>
      %3160 = stablehlo.and %3159, %3155 : tensor<i1>
      %cst_830 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %3161 = stablehlo.multiply %arg337, %cst_830 : tensor<f32>
      %3162 = stablehlo.convert %3161 : tensor<f32>
      %cst_831 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
      %3163 = stablehlo.minimum %3162, %cst_831 : tensor<f32>
      %3164 = func.call @_where_477(%3160, %3163, %arg337) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_832 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %3165 = stablehlo.multiply %arg337, %cst_832 : tensor<f32>
      %3166 = stablehlo.convert %3165 : tensor<f32>
      %cst_833 = stablehlo.constant dense<1.17549435E-38> : tensor<f32>
      %3167 = stablehlo.maximum %3166, %cst_833 : tensor<f32>
      %3168 = func.call @_where_480(%3155, %3164, %3167) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %3169 = stablehlo.not %3155 : tensor<i1>
      %c_834 = stablehlo.constant dense<false> : tensor<i1>
      %3170 = stablehlo.broadcast_in_dim %c_834, dims = [] : (tensor<i1>) -> tensor<i1>
      %3171 = stablehlo.compare  NE, %3156, %3170,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %3172 = stablehlo.convert %3171 : tensor<i1>
      %3173 = stablehlo.or %3172, %3169 : tensor<i1>
      %c_835 = stablehlo.constant dense<1> : tensor<i32>
      %3174 = stablehlo.add %arg336, %c_835 : tensor<i32>
      %c_836 = stablehlo.constant dense<0> : tensor<i32>
      %3175 = func.call @_where_482(%3173, %c_836, %3174) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %3176 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
      %3177 = stablehlo.iota dim = 0 : tensor<1000xi32>
      %3178 = stablehlo.reshape %3177 : (tensor<1000xi32>) -> tensor<1x1000xi32>
      %3179 = stablehlo.broadcast_in_dim %3176, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
      %3180 = stablehlo.broadcast_in_dim %3178, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
      %3181 = stablehlo.compare  EQ, %3179, %3180,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
      %cst_837 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3182 = stablehlo.broadcast_in_dim %cst_837, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %cst_838 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3183 = stablehlo.broadcast_in_dim %cst_838, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %3184 = stablehlo.select %3181, %3182, %3183 : tensor<256x1000xi1>, tensor<256x1000xf32>
      %3185 = stablehlo.convert %3184 : tensor<256x1000xf32>
      %3186 = func.call @log_softmax_484(%1435) : (tensor<256x1000xf16>) -> tensor<256x1000xf16>
      %3187 = stablehlo.convert %3186 : (tensor<256x1000xf16>) -> tensor<256x1000xf32>
      %3188 = stablehlo.multiply %3185, %3187 : tensor<256x1000xf32>
      %cst_839 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3189 = stablehlo.reduce(%3188 init: %cst_839) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
      %3190 = stablehlo.negate %3189 : tensor<256xf32>
      %cst_840 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3191 = stablehlo.reduce(%3190 init: %cst_840) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_841 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %3192 = stablehlo.divide %3191, %cst_841 : tensor<f32>
      %3193 = func.call @argmax(%1435) : (tensor<256x1000xf16>) -> tensor<256xi32>
      %3194 = stablehlo.compare  EQ, %3193, %2,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
      %3195 = stablehlo.convert %3194 : (tensor<256xi1>) -> tensor<256xi32>
      %3196 = stablehlo.convert %3195 : (tensor<256xi32>) -> tensor<256xf32>
      %cst_842 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3197 = stablehlo.reduce(%3196 init: %cst_842) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_843 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %3198 = stablehlo.divide %3197, %cst_843 : tensor<f32>
      %3199 = "stablehlo.all_reduce"(%3198) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %3200 = "stablehlo.all_reduce"(%3192) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_844 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3201 = stablehlo.divide %3199, %cst_844 : tensor<f32>
      %cst_845 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3202 = stablehlo.divide %3200, %cst_845 : tensor<f32>
      %3203 = "stablehlo.all_reduce"(%138) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %3204 = "stablehlo.all_reduce"(%143) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %3205 = "stablehlo.all_reduce"(%208) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %3206 = "stablehlo.all_reduce"(%213) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %3207 = "stablehlo.all_reduce"(%279) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %3208 = "stablehlo.all_reduce"(%284) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %3209 = "stablehlo.all_reduce"(%349) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %3210 = "stablehlo.all_reduce"(%354) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %3211 = "stablehlo.all_reduce"(%420) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3212 = "stablehlo.all_reduce"(%425) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3213 = "stablehlo.all_reduce"(%490) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3214 = "stablehlo.all_reduce"(%495) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3215 = "stablehlo.all_reduce"(%557) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3216 = "stablehlo.all_reduce"(%562) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3217 = "stablehlo.all_reduce"(%628) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3218 = "stablehlo.all_reduce"(%633) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3219 = "stablehlo.all_reduce"(%698) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3220 = "stablehlo.all_reduce"(%703) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<128xf32>) -> tensor<128xf32>
      %3221 = "stablehlo.all_reduce"(%769) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3222 = "stablehlo.all_reduce"(%774) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3223 = "stablehlo.all_reduce"(%839) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3224 = "stablehlo.all_reduce"(%844) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3225 = "stablehlo.all_reduce"(%906) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3226 = "stablehlo.all_reduce"(%911) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3227 = "stablehlo.all_reduce"(%977) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3228 = "stablehlo.all_reduce"(%982) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3229 = "stablehlo.all_reduce"(%1047) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3230 = "stablehlo.all_reduce"(%1052) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<256xf32>) -> tensor<256xf32>
      %3231 = "stablehlo.all_reduce"(%1118) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3232 = "stablehlo.all_reduce"(%1123) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3233 = "stablehlo.all_reduce"(%1188) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3234 = "stablehlo.all_reduce"(%1193) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3235 = "stablehlo.all_reduce"(%1255) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3236 = "stablehlo.all_reduce"(%1260) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3237 = "stablehlo.all_reduce"(%1326) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3238 = "stablehlo.all_reduce"(%1331) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3239 = "stablehlo.all_reduce"(%1396) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3240 = "stablehlo.all_reduce"(%1401) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<512xf32>) -> tensor<512xf32>
      %3241 = "stablehlo.all_reduce"(%66) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %3242 = "stablehlo.all_reduce"(%71) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg340: tensor<f32>, %arg341: tensor<f32>):
        %4098 = stablehlo.add %arg340, %arg341 : tensor<f32>
        stablehlo.return %4098 : tensor<f32>
      }) : (tensor<64xf32>) -> tensor<64xf32>
      %cst_846 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3243 = stablehlo.broadcast_in_dim %cst_846, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3244 = stablehlo.divide %3203, %3243 : tensor<64xf32>
      %cst_847 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3245 = stablehlo.broadcast_in_dim %cst_847, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3246 = stablehlo.divide %3204, %3245 : tensor<64xf32>
      %cst_848 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3247 = stablehlo.broadcast_in_dim %cst_848, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3248 = stablehlo.divide %3205, %3247 : tensor<64xf32>
      %cst_849 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3249 = stablehlo.broadcast_in_dim %cst_849, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3250 = stablehlo.divide %3206, %3249 : tensor<64xf32>
      %cst_850 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3251 = stablehlo.broadcast_in_dim %cst_850, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3252 = stablehlo.divide %3207, %3251 : tensor<64xf32>
      %cst_851 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3253 = stablehlo.broadcast_in_dim %cst_851, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3254 = stablehlo.divide %3208, %3253 : tensor<64xf32>
      %cst_852 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3255 = stablehlo.broadcast_in_dim %cst_852, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3256 = stablehlo.divide %3209, %3255 : tensor<64xf32>
      %cst_853 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3257 = stablehlo.broadcast_in_dim %cst_853, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3258 = stablehlo.divide %3210, %3257 : tensor<64xf32>
      %cst_854 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3259 = stablehlo.broadcast_in_dim %cst_854, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3260 = stablehlo.divide %3211, %3259 : tensor<128xf32>
      %cst_855 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3261 = stablehlo.broadcast_in_dim %cst_855, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3262 = stablehlo.divide %3212, %3261 : tensor<128xf32>
      %cst_856 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3263 = stablehlo.broadcast_in_dim %cst_856, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3264 = stablehlo.divide %3213, %3263 : tensor<128xf32>
      %cst_857 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3265 = stablehlo.broadcast_in_dim %cst_857, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3266 = stablehlo.divide %3214, %3265 : tensor<128xf32>
      %cst_858 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3267 = stablehlo.broadcast_in_dim %cst_858, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3268 = stablehlo.divide %3215, %3267 : tensor<128xf32>
      %cst_859 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3269 = stablehlo.broadcast_in_dim %cst_859, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3270 = stablehlo.divide %3216, %3269 : tensor<128xf32>
      %cst_860 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3271 = stablehlo.broadcast_in_dim %cst_860, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3272 = stablehlo.divide %3217, %3271 : tensor<128xf32>
      %cst_861 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3273 = stablehlo.broadcast_in_dim %cst_861, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3274 = stablehlo.divide %3218, %3273 : tensor<128xf32>
      %cst_862 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3275 = stablehlo.broadcast_in_dim %cst_862, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3276 = stablehlo.divide %3219, %3275 : tensor<128xf32>
      %cst_863 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3277 = stablehlo.broadcast_in_dim %cst_863, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3278 = stablehlo.divide %3220, %3277 : tensor<128xf32>
      %cst_864 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3279 = stablehlo.broadcast_in_dim %cst_864, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3280 = stablehlo.divide %3221, %3279 : tensor<256xf32>
      %cst_865 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3281 = stablehlo.broadcast_in_dim %cst_865, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3282 = stablehlo.divide %3222, %3281 : tensor<256xf32>
      %cst_866 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3283 = stablehlo.broadcast_in_dim %cst_866, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3284 = stablehlo.divide %3223, %3283 : tensor<256xf32>
      %cst_867 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3285 = stablehlo.broadcast_in_dim %cst_867, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3286 = stablehlo.divide %3224, %3285 : tensor<256xf32>
      %cst_868 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3287 = stablehlo.broadcast_in_dim %cst_868, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3288 = stablehlo.divide %3225, %3287 : tensor<256xf32>
      %cst_869 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3289 = stablehlo.broadcast_in_dim %cst_869, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3290 = stablehlo.divide %3226, %3289 : tensor<256xf32>
      %cst_870 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3291 = stablehlo.broadcast_in_dim %cst_870, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3292 = stablehlo.divide %3227, %3291 : tensor<256xf32>
      %cst_871 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3293 = stablehlo.broadcast_in_dim %cst_871, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3294 = stablehlo.divide %3228, %3293 : tensor<256xf32>
      %cst_872 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3295 = stablehlo.broadcast_in_dim %cst_872, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3296 = stablehlo.divide %3229, %3295 : tensor<256xf32>
      %cst_873 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3297 = stablehlo.broadcast_in_dim %cst_873, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3298 = stablehlo.divide %3230, %3297 : tensor<256xf32>
      %cst_874 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3299 = stablehlo.broadcast_in_dim %cst_874, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3300 = stablehlo.divide %3231, %3299 : tensor<512xf32>
      %cst_875 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3301 = stablehlo.broadcast_in_dim %cst_875, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3302 = stablehlo.divide %3232, %3301 : tensor<512xf32>
      %cst_876 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3303 = stablehlo.broadcast_in_dim %cst_876, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3304 = stablehlo.divide %3233, %3303 : tensor<512xf32>
      %cst_877 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3305 = stablehlo.broadcast_in_dim %cst_877, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3306 = stablehlo.divide %3234, %3305 : tensor<512xf32>
      %cst_878 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3307 = stablehlo.broadcast_in_dim %cst_878, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3308 = stablehlo.divide %3235, %3307 : tensor<512xf32>
      %cst_879 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3309 = stablehlo.broadcast_in_dim %cst_879, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3310 = stablehlo.divide %3236, %3309 : tensor<512xf32>
      %cst_880 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3311 = stablehlo.broadcast_in_dim %cst_880, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3312 = stablehlo.divide %3237, %3311 : tensor<512xf32>
      %cst_881 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3313 = stablehlo.broadcast_in_dim %cst_881, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3314 = stablehlo.divide %3238, %3313 : tensor<512xf32>
      %cst_882 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3315 = stablehlo.broadcast_in_dim %cst_882, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3316 = stablehlo.divide %3239, %3315 : tensor<512xf32>
      %cst_883 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3317 = stablehlo.broadcast_in_dim %cst_883, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3318 = stablehlo.divide %3240, %3317 : tensor<512xf32>
      %cst_884 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3319 = stablehlo.broadcast_in_dim %cst_884, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3320 = stablehlo.divide %3241, %3319 : tensor<64xf32>
      %cst_885 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3321 = stablehlo.broadcast_in_dim %cst_885, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3322 = stablehlo.divide %3242, %3321 : tensor<64xf32>
      %cst_886 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3323 = stablehlo.broadcast_in_dim %cst_886, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %3324 = stablehlo.multiply %3323, %arg233 : tensor<1000xf32>
      %3325 = stablehlo.add %2847, %3324 : tensor<1000xf32>
      %cst_887 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3326 = stablehlo.broadcast_in_dim %cst_887, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %3327 = stablehlo.multiply %3326, %arg234 : tensor<512x1000xf32>
      %3328 = stablehlo.add %2849, %3327 : tensor<512x1000xf32>
      %cst_888 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3329 = stablehlo.broadcast_in_dim %cst_888, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3330 = stablehlo.multiply %3329, %arg235 : tensor<64xf32>
      %3331 = stablehlo.add %2851, %3330 : tensor<64xf32>
      %cst_889 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3332 = stablehlo.broadcast_in_dim %cst_889, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3333 = stablehlo.multiply %3332, %arg236 : tensor<64xf32>
      %3334 = stablehlo.add %2853, %3333 : tensor<64xf32>
      %cst_890 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3335 = stablehlo.broadcast_in_dim %cst_890, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3336 = stablehlo.multiply %3335, %arg237 : tensor<64xf32>
      %3337 = stablehlo.add %2855, %3336 : tensor<64xf32>
      %cst_891 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3338 = stablehlo.broadcast_in_dim %cst_891, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3339 = stablehlo.multiply %3338, %arg238 : tensor<64xf32>
      %3340 = stablehlo.add %2857, %3339 : tensor<64xf32>
      %cst_892 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3341 = stablehlo.broadcast_in_dim %cst_892, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3342 = stablehlo.multiply %3341, %arg239 : tensor<3x3x64x64xf32>
      %3343 = stablehlo.add %2859, %3342 : tensor<3x3x64x64xf32>
      %cst_893 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3344 = stablehlo.broadcast_in_dim %cst_893, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3345 = stablehlo.multiply %3344, %arg240 : tensor<3x3x64x64xf32>
      %3346 = stablehlo.add %2861, %3345 : tensor<3x3x64x64xf32>
      %cst_894 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3347 = stablehlo.broadcast_in_dim %cst_894, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3348 = stablehlo.multiply %3347, %arg241 : tensor<64xf32>
      %3349 = stablehlo.add %2863, %3348 : tensor<64xf32>
      %cst_895 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3350 = stablehlo.broadcast_in_dim %cst_895, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3351 = stablehlo.multiply %3350, %arg242 : tensor<64xf32>
      %3352 = stablehlo.add %2865, %3351 : tensor<64xf32>
      %cst_896 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3353 = stablehlo.broadcast_in_dim %cst_896, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3354 = stablehlo.multiply %3353, %arg243 : tensor<64xf32>
      %3355 = stablehlo.add %2867, %3354 : tensor<64xf32>
      %cst_897 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3356 = stablehlo.broadcast_in_dim %cst_897, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3357 = stablehlo.multiply %3356, %arg244 : tensor<64xf32>
      %3358 = stablehlo.add %2869, %3357 : tensor<64xf32>
      %cst_898 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3359 = stablehlo.broadcast_in_dim %cst_898, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3360 = stablehlo.multiply %3359, %arg245 : tensor<3x3x64x64xf32>
      %3361 = stablehlo.add %2871, %3360 : tensor<3x3x64x64xf32>
      %cst_899 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3362 = stablehlo.broadcast_in_dim %cst_899, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3363 = stablehlo.multiply %3362, %arg246 : tensor<3x3x64x64xf32>
      %3364 = stablehlo.add %2873, %3363 : tensor<3x3x64x64xf32>
      %cst_900 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3365 = stablehlo.broadcast_in_dim %cst_900, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3366 = stablehlo.multiply %3365, %arg247 : tensor<128xf32>
      %3367 = stablehlo.add %2875, %3366 : tensor<128xf32>
      %cst_901 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3368 = stablehlo.broadcast_in_dim %cst_901, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3369 = stablehlo.multiply %3368, %arg248 : tensor<128xf32>
      %3370 = stablehlo.add %2877, %3369 : tensor<128xf32>
      %cst_902 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3371 = stablehlo.broadcast_in_dim %cst_902, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3372 = stablehlo.multiply %3371, %arg249 : tensor<128xf32>
      %3373 = stablehlo.add %2879, %3372 : tensor<128xf32>
      %cst_903 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3374 = stablehlo.broadcast_in_dim %cst_903, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3375 = stablehlo.multiply %3374, %arg250 : tensor<128xf32>
      %3376 = stablehlo.add %2881, %3375 : tensor<128xf32>
      %cst_904 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3377 = stablehlo.broadcast_in_dim %cst_904, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %3378 = stablehlo.multiply %3377, %arg251 : tensor<3x3x64x128xf32>
      %3379 = stablehlo.add %2883, %3378 : tensor<3x3x64x128xf32>
      %cst_905 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3380 = stablehlo.broadcast_in_dim %cst_905, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %3381 = stablehlo.multiply %3380, %arg252 : tensor<3x3x128x128xf32>
      %3382 = stablehlo.add %2885, %3381 : tensor<3x3x128x128xf32>
      %cst_906 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3383 = stablehlo.broadcast_in_dim %cst_906, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %3384 = stablehlo.multiply %3383, %arg253 : tensor<1x1x64x128xf32>
      %3385 = stablehlo.add %2887, %3384 : tensor<1x1x64x128xf32>
      %cst_907 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3386 = stablehlo.broadcast_in_dim %cst_907, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3387 = stablehlo.multiply %3386, %arg254 : tensor<128xf32>
      %3388 = stablehlo.add %2889, %3387 : tensor<128xf32>
      %cst_908 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3389 = stablehlo.broadcast_in_dim %cst_908, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3390 = stablehlo.multiply %3389, %arg255 : tensor<128xf32>
      %3391 = stablehlo.add %2891, %3390 : tensor<128xf32>
      %cst_909 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3392 = stablehlo.broadcast_in_dim %cst_909, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3393 = stablehlo.multiply %3392, %arg256 : tensor<128xf32>
      %3394 = stablehlo.add %2893, %3393 : tensor<128xf32>
      %cst_910 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3395 = stablehlo.broadcast_in_dim %cst_910, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3396 = stablehlo.multiply %3395, %arg257 : tensor<128xf32>
      %3397 = stablehlo.add %2895, %3396 : tensor<128xf32>
      %cst_911 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3398 = stablehlo.broadcast_in_dim %cst_911, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3399 = stablehlo.multiply %3398, %arg258 : tensor<128xf32>
      %3400 = stablehlo.add %2897, %3399 : tensor<128xf32>
      %cst_912 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3401 = stablehlo.broadcast_in_dim %cst_912, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3402 = stablehlo.multiply %3401, %arg259 : tensor<128xf32>
      %3403 = stablehlo.add %2899, %3402 : tensor<128xf32>
      %cst_913 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3404 = stablehlo.broadcast_in_dim %cst_913, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %3405 = stablehlo.multiply %3404, %arg260 : tensor<3x3x128x128xf32>
      %3406 = stablehlo.add %2901, %3405 : tensor<3x3x128x128xf32>
      %cst_914 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3407 = stablehlo.broadcast_in_dim %cst_914, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %3408 = stablehlo.multiply %3407, %arg261 : tensor<3x3x128x128xf32>
      %3409 = stablehlo.add %2903, %3408 : tensor<3x3x128x128xf32>
      %cst_915 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3410 = stablehlo.broadcast_in_dim %cst_915, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3411 = stablehlo.multiply %3410, %arg262 : tensor<256xf32>
      %3412 = stablehlo.add %2905, %3411 : tensor<256xf32>
      %cst_916 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3413 = stablehlo.broadcast_in_dim %cst_916, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3414 = stablehlo.multiply %3413, %arg263 : tensor<256xf32>
      %3415 = stablehlo.add %2907, %3414 : tensor<256xf32>
      %cst_917 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3416 = stablehlo.broadcast_in_dim %cst_917, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3417 = stablehlo.multiply %3416, %arg264 : tensor<256xf32>
      %3418 = stablehlo.add %2909, %3417 : tensor<256xf32>
      %cst_918 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3419 = stablehlo.broadcast_in_dim %cst_918, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3420 = stablehlo.multiply %3419, %arg265 : tensor<256xf32>
      %3421 = stablehlo.add %2911, %3420 : tensor<256xf32>
      %cst_919 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3422 = stablehlo.broadcast_in_dim %cst_919, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %3423 = stablehlo.multiply %3422, %arg266 : tensor<3x3x128x256xf32>
      %3424 = stablehlo.add %2913, %3423 : tensor<3x3x128x256xf32>
      %cst_920 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3425 = stablehlo.broadcast_in_dim %cst_920, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %3426 = stablehlo.multiply %3425, %arg267 : tensor<3x3x256x256xf32>
      %3427 = stablehlo.add %2915, %3426 : tensor<3x3x256x256xf32>
      %cst_921 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3428 = stablehlo.broadcast_in_dim %cst_921, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %3429 = stablehlo.multiply %3428, %arg268 : tensor<1x1x128x256xf32>
      %3430 = stablehlo.add %2917, %3429 : tensor<1x1x128x256xf32>
      %cst_922 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3431 = stablehlo.broadcast_in_dim %cst_922, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3432 = stablehlo.multiply %3431, %arg269 : tensor<256xf32>
      %3433 = stablehlo.add %2919, %3432 : tensor<256xf32>
      %cst_923 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3434 = stablehlo.broadcast_in_dim %cst_923, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3435 = stablehlo.multiply %3434, %arg270 : tensor<256xf32>
      %3436 = stablehlo.add %2921, %3435 : tensor<256xf32>
      %cst_924 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3437 = stablehlo.broadcast_in_dim %cst_924, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3438 = stablehlo.multiply %3437, %arg271 : tensor<256xf32>
      %3439 = stablehlo.add %2923, %3438 : tensor<256xf32>
      %cst_925 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3440 = stablehlo.broadcast_in_dim %cst_925, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3441 = stablehlo.multiply %3440, %arg272 : tensor<256xf32>
      %3442 = stablehlo.add %2925, %3441 : tensor<256xf32>
      %cst_926 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3443 = stablehlo.broadcast_in_dim %cst_926, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3444 = stablehlo.multiply %3443, %arg273 : tensor<256xf32>
      %3445 = stablehlo.add %2927, %3444 : tensor<256xf32>
      %cst_927 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3446 = stablehlo.broadcast_in_dim %cst_927, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3447 = stablehlo.multiply %3446, %arg274 : tensor<256xf32>
      %3448 = stablehlo.add %2929, %3447 : tensor<256xf32>
      %cst_928 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3449 = stablehlo.broadcast_in_dim %cst_928, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %3450 = stablehlo.multiply %3449, %arg275 : tensor<3x3x256x256xf32>
      %3451 = stablehlo.add %2931, %3450 : tensor<3x3x256x256xf32>
      %cst_929 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3452 = stablehlo.broadcast_in_dim %cst_929, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %3453 = stablehlo.multiply %3452, %arg276 : tensor<3x3x256x256xf32>
      %3454 = stablehlo.add %2933, %3453 : tensor<3x3x256x256xf32>
      %cst_930 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3455 = stablehlo.broadcast_in_dim %cst_930, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3456 = stablehlo.multiply %3455, %arg277 : tensor<512xf32>
      %3457 = stablehlo.add %2935, %3456 : tensor<512xf32>
      %cst_931 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3458 = stablehlo.broadcast_in_dim %cst_931, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3459 = stablehlo.multiply %3458, %arg278 : tensor<512xf32>
      %3460 = stablehlo.add %2937, %3459 : tensor<512xf32>
      %cst_932 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3461 = stablehlo.broadcast_in_dim %cst_932, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3462 = stablehlo.multiply %3461, %arg279 : tensor<512xf32>
      %3463 = stablehlo.add %2939, %3462 : tensor<512xf32>
      %cst_933 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3464 = stablehlo.broadcast_in_dim %cst_933, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3465 = stablehlo.multiply %3464, %arg280 : tensor<512xf32>
      %3466 = stablehlo.add %2941, %3465 : tensor<512xf32>
      %cst_934 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3467 = stablehlo.broadcast_in_dim %cst_934, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %3468 = stablehlo.multiply %3467, %arg281 : tensor<3x3x256x512xf32>
      %3469 = stablehlo.add %2943, %3468 : tensor<3x3x256x512xf32>
      %cst_935 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3470 = stablehlo.broadcast_in_dim %cst_935, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %3471 = stablehlo.multiply %3470, %arg282 : tensor<3x3x512x512xf32>
      %3472 = stablehlo.add %2945, %3471 : tensor<3x3x512x512xf32>
      %cst_936 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3473 = stablehlo.broadcast_in_dim %cst_936, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %3474 = stablehlo.multiply %3473, %arg283 : tensor<1x1x256x512xf32>
      %3475 = stablehlo.add %2947, %3474 : tensor<1x1x256x512xf32>
      %cst_937 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3476 = stablehlo.broadcast_in_dim %cst_937, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3477 = stablehlo.multiply %3476, %arg284 : tensor<512xf32>
      %3478 = stablehlo.add %2949, %3477 : tensor<512xf32>
      %cst_938 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3479 = stablehlo.broadcast_in_dim %cst_938, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3480 = stablehlo.multiply %3479, %arg285 : tensor<512xf32>
      %3481 = stablehlo.add %2951, %3480 : tensor<512xf32>
      %cst_939 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3482 = stablehlo.broadcast_in_dim %cst_939, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3483 = stablehlo.multiply %3482, %arg286 : tensor<512xf32>
      %3484 = stablehlo.add %2953, %3483 : tensor<512xf32>
      %cst_940 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3485 = stablehlo.broadcast_in_dim %cst_940, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3486 = stablehlo.multiply %3485, %arg287 : tensor<512xf32>
      %3487 = stablehlo.add %2955, %3486 : tensor<512xf32>
      %cst_941 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3488 = stablehlo.broadcast_in_dim %cst_941, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3489 = stablehlo.multiply %3488, %arg288 : tensor<512xf32>
      %3490 = stablehlo.add %2957, %3489 : tensor<512xf32>
      %cst_942 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3491 = stablehlo.broadcast_in_dim %cst_942, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3492 = stablehlo.multiply %3491, %arg289 : tensor<512xf32>
      %3493 = stablehlo.add %2959, %3492 : tensor<512xf32>
      %cst_943 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3494 = stablehlo.broadcast_in_dim %cst_943, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %3495 = stablehlo.multiply %3494, %arg290 : tensor<3x3x512x512xf32>
      %3496 = stablehlo.add %2961, %3495 : tensor<3x3x512x512xf32>
      %cst_944 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3497 = stablehlo.broadcast_in_dim %cst_944, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %3498 = stablehlo.multiply %3497, %arg291 : tensor<3x3x512x512xf32>
      %3499 = stablehlo.add %2963, %3498 : tensor<3x3x512x512xf32>
      %cst_945 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3500 = stablehlo.broadcast_in_dim %cst_945, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3501 = stablehlo.multiply %3500, %arg292 : tensor<64xf32>
      %3502 = stablehlo.add %2965, %3501 : tensor<64xf32>
      %cst_946 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3503 = stablehlo.broadcast_in_dim %cst_946, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3504 = stablehlo.multiply %3503, %arg293 : tensor<64xf32>
      %3505 = stablehlo.add %2967, %3504 : tensor<64xf32>
      %cst_947 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3506 = stablehlo.broadcast_in_dim %cst_947, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %3507 = stablehlo.multiply %3506, %arg294 : tensor<7x7x3x64xf32>
      %3508 = stablehlo.add %2969, %3507 : tensor<7x7x3x64xf32>
      %cst_948 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3509 = stablehlo.broadcast_in_dim %cst_948, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %3510 = stablehlo.multiply %3509, %3325 : tensor<1000xf32>
      %3511 = stablehlo.add %2847, %3510 : tensor<1000xf32>
      %cst_949 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3512 = stablehlo.broadcast_in_dim %cst_949, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %3513 = stablehlo.multiply %3512, %3328 : tensor<512x1000xf32>
      %3514 = stablehlo.add %2849, %3513 : tensor<512x1000xf32>
      %cst_950 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3515 = stablehlo.broadcast_in_dim %cst_950, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3516 = stablehlo.multiply %3515, %3331 : tensor<64xf32>
      %3517 = stablehlo.add %2851, %3516 : tensor<64xf32>
      %cst_951 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3518 = stablehlo.broadcast_in_dim %cst_951, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3519 = stablehlo.multiply %3518, %3334 : tensor<64xf32>
      %3520 = stablehlo.add %2853, %3519 : tensor<64xf32>
      %cst_952 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3521 = stablehlo.broadcast_in_dim %cst_952, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3522 = stablehlo.multiply %3521, %3337 : tensor<64xf32>
      %3523 = stablehlo.add %2855, %3522 : tensor<64xf32>
      %cst_953 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3524 = stablehlo.broadcast_in_dim %cst_953, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3525 = stablehlo.multiply %3524, %3340 : tensor<64xf32>
      %3526 = stablehlo.add %2857, %3525 : tensor<64xf32>
      %cst_954 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3527 = stablehlo.broadcast_in_dim %cst_954, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3528 = stablehlo.multiply %3527, %3343 : tensor<3x3x64x64xf32>
      %3529 = stablehlo.add %2859, %3528 : tensor<3x3x64x64xf32>
      %cst_955 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3530 = stablehlo.broadcast_in_dim %cst_955, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3531 = stablehlo.multiply %3530, %3346 : tensor<3x3x64x64xf32>
      %3532 = stablehlo.add %2861, %3531 : tensor<3x3x64x64xf32>
      %cst_956 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3533 = stablehlo.broadcast_in_dim %cst_956, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3534 = stablehlo.multiply %3533, %3349 : tensor<64xf32>
      %3535 = stablehlo.add %2863, %3534 : tensor<64xf32>
      %cst_957 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3536 = stablehlo.broadcast_in_dim %cst_957, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3537 = stablehlo.multiply %3536, %3352 : tensor<64xf32>
      %3538 = stablehlo.add %2865, %3537 : tensor<64xf32>
      %cst_958 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3539 = stablehlo.broadcast_in_dim %cst_958, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3540 = stablehlo.multiply %3539, %3355 : tensor<64xf32>
      %3541 = stablehlo.add %2867, %3540 : tensor<64xf32>
      %cst_959 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3542 = stablehlo.broadcast_in_dim %cst_959, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3543 = stablehlo.multiply %3542, %3358 : tensor<64xf32>
      %3544 = stablehlo.add %2869, %3543 : tensor<64xf32>
      %cst_960 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3545 = stablehlo.broadcast_in_dim %cst_960, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3546 = stablehlo.multiply %3545, %3361 : tensor<3x3x64x64xf32>
      %3547 = stablehlo.add %2871, %3546 : tensor<3x3x64x64xf32>
      %cst_961 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3548 = stablehlo.broadcast_in_dim %cst_961, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3549 = stablehlo.multiply %3548, %3364 : tensor<3x3x64x64xf32>
      %3550 = stablehlo.add %2873, %3549 : tensor<3x3x64x64xf32>
      %cst_962 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3551 = stablehlo.broadcast_in_dim %cst_962, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3552 = stablehlo.multiply %3551, %3367 : tensor<128xf32>
      %3553 = stablehlo.add %2875, %3552 : tensor<128xf32>
      %cst_963 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3554 = stablehlo.broadcast_in_dim %cst_963, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3555 = stablehlo.multiply %3554, %3370 : tensor<128xf32>
      %3556 = stablehlo.add %2877, %3555 : tensor<128xf32>
      %cst_964 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3557 = stablehlo.broadcast_in_dim %cst_964, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3558 = stablehlo.multiply %3557, %3373 : tensor<128xf32>
      %3559 = stablehlo.add %2879, %3558 : tensor<128xf32>
      %cst_965 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3560 = stablehlo.broadcast_in_dim %cst_965, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3561 = stablehlo.multiply %3560, %3376 : tensor<128xf32>
      %3562 = stablehlo.add %2881, %3561 : tensor<128xf32>
      %cst_966 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3563 = stablehlo.broadcast_in_dim %cst_966, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %3564 = stablehlo.multiply %3563, %3379 : tensor<3x3x64x128xf32>
      %3565 = stablehlo.add %2883, %3564 : tensor<3x3x64x128xf32>
      %cst_967 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3566 = stablehlo.broadcast_in_dim %cst_967, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %3567 = stablehlo.multiply %3566, %3382 : tensor<3x3x128x128xf32>
      %3568 = stablehlo.add %2885, %3567 : tensor<3x3x128x128xf32>
      %cst_968 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3569 = stablehlo.broadcast_in_dim %cst_968, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %3570 = stablehlo.multiply %3569, %3385 : tensor<1x1x64x128xf32>
      %3571 = stablehlo.add %2887, %3570 : tensor<1x1x64x128xf32>
      %cst_969 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3572 = stablehlo.broadcast_in_dim %cst_969, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3573 = stablehlo.multiply %3572, %3388 : tensor<128xf32>
      %3574 = stablehlo.add %2889, %3573 : tensor<128xf32>
      %cst_970 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3575 = stablehlo.broadcast_in_dim %cst_970, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3576 = stablehlo.multiply %3575, %3391 : tensor<128xf32>
      %3577 = stablehlo.add %2891, %3576 : tensor<128xf32>
      %cst_971 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3578 = stablehlo.broadcast_in_dim %cst_971, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3579 = stablehlo.multiply %3578, %3394 : tensor<128xf32>
      %3580 = stablehlo.add %2893, %3579 : tensor<128xf32>
      %cst_972 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3581 = stablehlo.broadcast_in_dim %cst_972, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3582 = stablehlo.multiply %3581, %3397 : tensor<128xf32>
      %3583 = stablehlo.add %2895, %3582 : tensor<128xf32>
      %cst_973 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3584 = stablehlo.broadcast_in_dim %cst_973, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3585 = stablehlo.multiply %3584, %3400 : tensor<128xf32>
      %3586 = stablehlo.add %2897, %3585 : tensor<128xf32>
      %cst_974 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3587 = stablehlo.broadcast_in_dim %cst_974, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3588 = stablehlo.multiply %3587, %3403 : tensor<128xf32>
      %3589 = stablehlo.add %2899, %3588 : tensor<128xf32>
      %cst_975 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3590 = stablehlo.broadcast_in_dim %cst_975, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %3591 = stablehlo.multiply %3590, %3406 : tensor<3x3x128x128xf32>
      %3592 = stablehlo.add %2901, %3591 : tensor<3x3x128x128xf32>
      %cst_976 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3593 = stablehlo.broadcast_in_dim %cst_976, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %3594 = stablehlo.multiply %3593, %3409 : tensor<3x3x128x128xf32>
      %3595 = stablehlo.add %2903, %3594 : tensor<3x3x128x128xf32>
      %cst_977 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3596 = stablehlo.broadcast_in_dim %cst_977, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3597 = stablehlo.multiply %3596, %3412 : tensor<256xf32>
      %3598 = stablehlo.add %2905, %3597 : tensor<256xf32>
      %cst_978 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3599 = stablehlo.broadcast_in_dim %cst_978, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3600 = stablehlo.multiply %3599, %3415 : tensor<256xf32>
      %3601 = stablehlo.add %2907, %3600 : tensor<256xf32>
      %cst_979 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3602 = stablehlo.broadcast_in_dim %cst_979, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3603 = stablehlo.multiply %3602, %3418 : tensor<256xf32>
      %3604 = stablehlo.add %2909, %3603 : tensor<256xf32>
      %cst_980 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3605 = stablehlo.broadcast_in_dim %cst_980, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3606 = stablehlo.multiply %3605, %3421 : tensor<256xf32>
      %3607 = stablehlo.add %2911, %3606 : tensor<256xf32>
      %cst_981 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3608 = stablehlo.broadcast_in_dim %cst_981, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %3609 = stablehlo.multiply %3608, %3424 : tensor<3x3x128x256xf32>
      %3610 = stablehlo.add %2913, %3609 : tensor<3x3x128x256xf32>
      %cst_982 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3611 = stablehlo.broadcast_in_dim %cst_982, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %3612 = stablehlo.multiply %3611, %3427 : tensor<3x3x256x256xf32>
      %3613 = stablehlo.add %2915, %3612 : tensor<3x3x256x256xf32>
      %cst_983 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3614 = stablehlo.broadcast_in_dim %cst_983, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %3615 = stablehlo.multiply %3614, %3430 : tensor<1x1x128x256xf32>
      %3616 = stablehlo.add %2917, %3615 : tensor<1x1x128x256xf32>
      %cst_984 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3617 = stablehlo.broadcast_in_dim %cst_984, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3618 = stablehlo.multiply %3617, %3433 : tensor<256xf32>
      %3619 = stablehlo.add %2919, %3618 : tensor<256xf32>
      %cst_985 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3620 = stablehlo.broadcast_in_dim %cst_985, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3621 = stablehlo.multiply %3620, %3436 : tensor<256xf32>
      %3622 = stablehlo.add %2921, %3621 : tensor<256xf32>
      %cst_986 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3623 = stablehlo.broadcast_in_dim %cst_986, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3624 = stablehlo.multiply %3623, %3439 : tensor<256xf32>
      %3625 = stablehlo.add %2923, %3624 : tensor<256xf32>
      %cst_987 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3626 = stablehlo.broadcast_in_dim %cst_987, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3627 = stablehlo.multiply %3626, %3442 : tensor<256xf32>
      %3628 = stablehlo.add %2925, %3627 : tensor<256xf32>
      %cst_988 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3629 = stablehlo.broadcast_in_dim %cst_988, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3630 = stablehlo.multiply %3629, %3445 : tensor<256xf32>
      %3631 = stablehlo.add %2927, %3630 : tensor<256xf32>
      %cst_989 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3632 = stablehlo.broadcast_in_dim %cst_989, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3633 = stablehlo.multiply %3632, %3448 : tensor<256xf32>
      %3634 = stablehlo.add %2929, %3633 : tensor<256xf32>
      %cst_990 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3635 = stablehlo.broadcast_in_dim %cst_990, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %3636 = stablehlo.multiply %3635, %3451 : tensor<3x3x256x256xf32>
      %3637 = stablehlo.add %2931, %3636 : tensor<3x3x256x256xf32>
      %cst_991 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3638 = stablehlo.broadcast_in_dim %cst_991, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %3639 = stablehlo.multiply %3638, %3454 : tensor<3x3x256x256xf32>
      %3640 = stablehlo.add %2933, %3639 : tensor<3x3x256x256xf32>
      %cst_992 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3641 = stablehlo.broadcast_in_dim %cst_992, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3642 = stablehlo.multiply %3641, %3457 : tensor<512xf32>
      %3643 = stablehlo.add %2935, %3642 : tensor<512xf32>
      %cst_993 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3644 = stablehlo.broadcast_in_dim %cst_993, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3645 = stablehlo.multiply %3644, %3460 : tensor<512xf32>
      %3646 = stablehlo.add %2937, %3645 : tensor<512xf32>
      %cst_994 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3647 = stablehlo.broadcast_in_dim %cst_994, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3648 = stablehlo.multiply %3647, %3463 : tensor<512xf32>
      %3649 = stablehlo.add %2939, %3648 : tensor<512xf32>
      %cst_995 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3650 = stablehlo.broadcast_in_dim %cst_995, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3651 = stablehlo.multiply %3650, %3466 : tensor<512xf32>
      %3652 = stablehlo.add %2941, %3651 : tensor<512xf32>
      %cst_996 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3653 = stablehlo.broadcast_in_dim %cst_996, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %3654 = stablehlo.multiply %3653, %3469 : tensor<3x3x256x512xf32>
      %3655 = stablehlo.add %2943, %3654 : tensor<3x3x256x512xf32>
      %cst_997 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3656 = stablehlo.broadcast_in_dim %cst_997, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %3657 = stablehlo.multiply %3656, %3472 : tensor<3x3x512x512xf32>
      %3658 = stablehlo.add %2945, %3657 : tensor<3x3x512x512xf32>
      %cst_998 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3659 = stablehlo.broadcast_in_dim %cst_998, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %3660 = stablehlo.multiply %3659, %3475 : tensor<1x1x256x512xf32>
      %3661 = stablehlo.add %2947, %3660 : tensor<1x1x256x512xf32>
      %cst_999 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3662 = stablehlo.broadcast_in_dim %cst_999, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3663 = stablehlo.multiply %3662, %3478 : tensor<512xf32>
      %3664 = stablehlo.add %2949, %3663 : tensor<512xf32>
      %cst_1000 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3665 = stablehlo.broadcast_in_dim %cst_1000, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3666 = stablehlo.multiply %3665, %3481 : tensor<512xf32>
      %3667 = stablehlo.add %2951, %3666 : tensor<512xf32>
      %cst_1001 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3668 = stablehlo.broadcast_in_dim %cst_1001, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3669 = stablehlo.multiply %3668, %3484 : tensor<512xf32>
      %3670 = stablehlo.add %2953, %3669 : tensor<512xf32>
      %cst_1002 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3671 = stablehlo.broadcast_in_dim %cst_1002, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3672 = stablehlo.multiply %3671, %3487 : tensor<512xf32>
      %3673 = stablehlo.add %2955, %3672 : tensor<512xf32>
      %cst_1003 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3674 = stablehlo.broadcast_in_dim %cst_1003, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3675 = stablehlo.multiply %3674, %3490 : tensor<512xf32>
      %3676 = stablehlo.add %2957, %3675 : tensor<512xf32>
      %cst_1004 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3677 = stablehlo.broadcast_in_dim %cst_1004, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3678 = stablehlo.multiply %3677, %3493 : tensor<512xf32>
      %3679 = stablehlo.add %2959, %3678 : tensor<512xf32>
      %cst_1005 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3680 = stablehlo.broadcast_in_dim %cst_1005, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %3681 = stablehlo.multiply %3680, %3496 : tensor<3x3x512x512xf32>
      %3682 = stablehlo.add %2961, %3681 : tensor<3x3x512x512xf32>
      %cst_1006 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3683 = stablehlo.broadcast_in_dim %cst_1006, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %3684 = stablehlo.multiply %3683, %3499 : tensor<3x3x512x512xf32>
      %3685 = stablehlo.add %2963, %3684 : tensor<3x3x512x512xf32>
      %cst_1007 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3686 = stablehlo.broadcast_in_dim %cst_1007, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3687 = stablehlo.multiply %3686, %3502 : tensor<64xf32>
      %3688 = stablehlo.add %2965, %3687 : tensor<64xf32>
      %cst_1008 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3689 = stablehlo.broadcast_in_dim %cst_1008, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3690 = stablehlo.multiply %3689, %3505 : tensor<64xf32>
      %3691 = stablehlo.add %2967, %3690 : tensor<64xf32>
      %cst_1009 = stablehlo.constant dense<0.899999976> : tensor<f32>
      %3692 = stablehlo.broadcast_in_dim %cst_1009, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %3693 = stablehlo.multiply %3692, %3508 : tensor<7x7x3x64xf32>
      %3694 = stablehlo.add %2969, %3693 : tensor<7x7x3x64xf32>
      %c_1010 = stablehlo.constant dense<0> : tensor<i32>
      %3695 = stablehlo.subtract %arg295, %c_1010 : tensor<i32>
      %c_1011 = stablehlo.constant dense<0> : tensor<i32>
      %cst_1012 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %3696 = func.call @clip_526(%3695, %c_1011, %cst_1012) : (tensor<i32>, tensor<i32>, tensor<f32>) -> tensor<f32>
      %cst_1013 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %3697 = stablehlo.divide %3696, %cst_1013 : tensor<f32>
      %cst_1014 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3698 = stablehlo.subtract %cst_1014, %3697 : tensor<f32>
      %cst_1015 = stablehlo.constant dense<-4.000000e-01> : tensor<f32>
      %3699 = stablehlo.multiply %cst_1015, %3698 : tensor<f32>
      %cst_1016 = stablehlo.constant dense<4.000000e-01> : tensor<f32>
      %3700 = stablehlo.add %3699, %cst_1016 : tensor<f32>
      %3701 = stablehlo.convert %arg295 : (tensor<i32>) -> tensor<f32>
      %cst_1017 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %3702 = stablehlo.compare  LT, %3701, %cst_1017,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3703 = stablehlo.convert %arg295 : (tensor<i32>) -> tensor<f32>
      %cst_1018 = stablehlo.constant dense<6.255000e+03> : tensor<f32>
      %3704 = stablehlo.subtract %3703, %cst_1018 : tensor<f32>
      %cst_1019 = stablehlo.constant dense<1.188450e+05> : tensor<f32>
      %3705 = stablehlo.minimum %3704, %cst_1019 : tensor<f32>
      %cst_1020 = stablehlo.constant dense<3.14159274> : tensor<f32>
      %3706 = stablehlo.multiply %cst_1020, %3705 : tensor<f32>
      %cst_1021 = stablehlo.constant dense<1.188450e+05> : tensor<f32>
      %3707 = stablehlo.divide %3706, %cst_1021 : tensor<f32>
      %3708 = stablehlo.cosine %3707 : tensor<f32>
      %cst_1022 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3709 = stablehlo.add %cst_1022, %3708 : tensor<f32>
      %cst_1023 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %3710 = stablehlo.multiply %cst_1023, %3709 : tensor<f32>
      %cst_1024 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3711 = stablehlo.power %3710, %cst_1024 : tensor<f32>
      %cst_1025 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3712 = stablehlo.multiply %cst_1025, %3711 : tensor<f32>
      %cst_1026 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3713 = stablehlo.add %3712, %cst_1026 : tensor<f32>
      %cst_1027 = stablehlo.constant dense<4.000000e-01> : tensor<f32>
      %3714 = stablehlo.multiply %cst_1027, %3713 : tensor<f32>
      %3715 = func.call @_where(%3702, %3700, %3714) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_1028 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
      %3716 = stablehlo.multiply %cst_1028, %3715 : tensor<f32>
      %3717 = stablehlo.convert %3716 : tensor<f32>
      %3718 = stablehlo.broadcast_in_dim %3717, dims = [] : (tensor<f32>) -> tensor<1000xf32>
      %3719 = stablehlo.multiply %3718, %3511 : tensor<1000xf32>
      %3720 = stablehlo.convert %3716 : tensor<f32>
      %3721 = stablehlo.broadcast_in_dim %3720, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
      %3722 = stablehlo.multiply %3721, %3514 : tensor<512x1000xf32>
      %3723 = stablehlo.convert %3716 : tensor<f32>
      %3724 = stablehlo.broadcast_in_dim %3723, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3725 = stablehlo.multiply %3724, %3517 : tensor<64xf32>
      %3726 = stablehlo.convert %3716 : tensor<f32>
      %3727 = stablehlo.broadcast_in_dim %3726, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3728 = stablehlo.multiply %3727, %3520 : tensor<64xf32>
      %3729 = stablehlo.convert %3716 : tensor<f32>
      %3730 = stablehlo.broadcast_in_dim %3729, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3731 = stablehlo.multiply %3730, %3523 : tensor<64xf32>
      %3732 = stablehlo.convert %3716 : tensor<f32>
      %3733 = stablehlo.broadcast_in_dim %3732, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3734 = stablehlo.multiply %3733, %3526 : tensor<64xf32>
      %3735 = stablehlo.convert %3716 : tensor<f32>
      %3736 = stablehlo.broadcast_in_dim %3735, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3737 = stablehlo.multiply %3736, %3529 : tensor<3x3x64x64xf32>
      %3738 = stablehlo.convert %3716 : tensor<f32>
      %3739 = stablehlo.broadcast_in_dim %3738, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3740 = stablehlo.multiply %3739, %3532 : tensor<3x3x64x64xf32>
      %3741 = stablehlo.convert %3716 : tensor<f32>
      %3742 = stablehlo.broadcast_in_dim %3741, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3743 = stablehlo.multiply %3742, %3535 : tensor<64xf32>
      %3744 = stablehlo.convert %3716 : tensor<f32>
      %3745 = stablehlo.broadcast_in_dim %3744, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3746 = stablehlo.multiply %3745, %3538 : tensor<64xf32>
      %3747 = stablehlo.convert %3716 : tensor<f32>
      %3748 = stablehlo.broadcast_in_dim %3747, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3749 = stablehlo.multiply %3748, %3541 : tensor<64xf32>
      %3750 = stablehlo.convert %3716 : tensor<f32>
      %3751 = stablehlo.broadcast_in_dim %3750, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3752 = stablehlo.multiply %3751, %3544 : tensor<64xf32>
      %3753 = stablehlo.convert %3716 : tensor<f32>
      %3754 = stablehlo.broadcast_in_dim %3753, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3755 = stablehlo.multiply %3754, %3547 : tensor<3x3x64x64xf32>
      %3756 = stablehlo.convert %3716 : tensor<f32>
      %3757 = stablehlo.broadcast_in_dim %3756, dims = [] : (tensor<f32>) -> tensor<3x3x64x64xf32>
      %3758 = stablehlo.multiply %3757, %3550 : tensor<3x3x64x64xf32>
      %3759 = stablehlo.convert %3716 : tensor<f32>
      %3760 = stablehlo.broadcast_in_dim %3759, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3761 = stablehlo.multiply %3760, %3553 : tensor<128xf32>
      %3762 = stablehlo.convert %3716 : tensor<f32>
      %3763 = stablehlo.broadcast_in_dim %3762, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3764 = stablehlo.multiply %3763, %3556 : tensor<128xf32>
      %3765 = stablehlo.convert %3716 : tensor<f32>
      %3766 = stablehlo.broadcast_in_dim %3765, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3767 = stablehlo.multiply %3766, %3559 : tensor<128xf32>
      %3768 = stablehlo.convert %3716 : tensor<f32>
      %3769 = stablehlo.broadcast_in_dim %3768, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3770 = stablehlo.multiply %3769, %3562 : tensor<128xf32>
      %3771 = stablehlo.convert %3716 : tensor<f32>
      %3772 = stablehlo.broadcast_in_dim %3771, dims = [] : (tensor<f32>) -> tensor<3x3x64x128xf32>
      %3773 = stablehlo.multiply %3772, %3565 : tensor<3x3x64x128xf32>
      %3774 = stablehlo.convert %3716 : tensor<f32>
      %3775 = stablehlo.broadcast_in_dim %3774, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %3776 = stablehlo.multiply %3775, %3568 : tensor<3x3x128x128xf32>
      %3777 = stablehlo.convert %3716 : tensor<f32>
      %3778 = stablehlo.broadcast_in_dim %3777, dims = [] : (tensor<f32>) -> tensor<1x1x64x128xf32>
      %3779 = stablehlo.multiply %3778, %3571 : tensor<1x1x64x128xf32>
      %3780 = stablehlo.convert %3716 : tensor<f32>
      %3781 = stablehlo.broadcast_in_dim %3780, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3782 = stablehlo.multiply %3781, %3574 : tensor<128xf32>
      %3783 = stablehlo.convert %3716 : tensor<f32>
      %3784 = stablehlo.broadcast_in_dim %3783, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3785 = stablehlo.multiply %3784, %3577 : tensor<128xf32>
      %3786 = stablehlo.convert %3716 : tensor<f32>
      %3787 = stablehlo.broadcast_in_dim %3786, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3788 = stablehlo.multiply %3787, %3580 : tensor<128xf32>
      %3789 = stablehlo.convert %3716 : tensor<f32>
      %3790 = stablehlo.broadcast_in_dim %3789, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3791 = stablehlo.multiply %3790, %3583 : tensor<128xf32>
      %3792 = stablehlo.convert %3716 : tensor<f32>
      %3793 = stablehlo.broadcast_in_dim %3792, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3794 = stablehlo.multiply %3793, %3586 : tensor<128xf32>
      %3795 = stablehlo.convert %3716 : tensor<f32>
      %3796 = stablehlo.broadcast_in_dim %3795, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %3797 = stablehlo.multiply %3796, %3589 : tensor<128xf32>
      %3798 = stablehlo.convert %3716 : tensor<f32>
      %3799 = stablehlo.broadcast_in_dim %3798, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %3800 = stablehlo.multiply %3799, %3592 : tensor<3x3x128x128xf32>
      %3801 = stablehlo.convert %3716 : tensor<f32>
      %3802 = stablehlo.broadcast_in_dim %3801, dims = [] : (tensor<f32>) -> tensor<3x3x128x128xf32>
      %3803 = stablehlo.multiply %3802, %3595 : tensor<3x3x128x128xf32>
      %3804 = stablehlo.convert %3716 : tensor<f32>
      %3805 = stablehlo.broadcast_in_dim %3804, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3806 = stablehlo.multiply %3805, %3598 : tensor<256xf32>
      %3807 = stablehlo.convert %3716 : tensor<f32>
      %3808 = stablehlo.broadcast_in_dim %3807, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3809 = stablehlo.multiply %3808, %3601 : tensor<256xf32>
      %3810 = stablehlo.convert %3716 : tensor<f32>
      %3811 = stablehlo.broadcast_in_dim %3810, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3812 = stablehlo.multiply %3811, %3604 : tensor<256xf32>
      %3813 = stablehlo.convert %3716 : tensor<f32>
      %3814 = stablehlo.broadcast_in_dim %3813, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3815 = stablehlo.multiply %3814, %3607 : tensor<256xf32>
      %3816 = stablehlo.convert %3716 : tensor<f32>
      %3817 = stablehlo.broadcast_in_dim %3816, dims = [] : (tensor<f32>) -> tensor<3x3x128x256xf32>
      %3818 = stablehlo.multiply %3817, %3610 : tensor<3x3x128x256xf32>
      %3819 = stablehlo.convert %3716 : tensor<f32>
      %3820 = stablehlo.broadcast_in_dim %3819, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %3821 = stablehlo.multiply %3820, %3613 : tensor<3x3x256x256xf32>
      %3822 = stablehlo.convert %3716 : tensor<f32>
      %3823 = stablehlo.broadcast_in_dim %3822, dims = [] : (tensor<f32>) -> tensor<1x1x128x256xf32>
      %3824 = stablehlo.multiply %3823, %3616 : tensor<1x1x128x256xf32>
      %3825 = stablehlo.convert %3716 : tensor<f32>
      %3826 = stablehlo.broadcast_in_dim %3825, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3827 = stablehlo.multiply %3826, %3619 : tensor<256xf32>
      %3828 = stablehlo.convert %3716 : tensor<f32>
      %3829 = stablehlo.broadcast_in_dim %3828, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3830 = stablehlo.multiply %3829, %3622 : tensor<256xf32>
      %3831 = stablehlo.convert %3716 : tensor<f32>
      %3832 = stablehlo.broadcast_in_dim %3831, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3833 = stablehlo.multiply %3832, %3625 : tensor<256xf32>
      %3834 = stablehlo.convert %3716 : tensor<f32>
      %3835 = stablehlo.broadcast_in_dim %3834, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3836 = stablehlo.multiply %3835, %3628 : tensor<256xf32>
      %3837 = stablehlo.convert %3716 : tensor<f32>
      %3838 = stablehlo.broadcast_in_dim %3837, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3839 = stablehlo.multiply %3838, %3631 : tensor<256xf32>
      %3840 = stablehlo.convert %3716 : tensor<f32>
      %3841 = stablehlo.broadcast_in_dim %3840, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %3842 = stablehlo.multiply %3841, %3634 : tensor<256xf32>
      %3843 = stablehlo.convert %3716 : tensor<f32>
      %3844 = stablehlo.broadcast_in_dim %3843, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %3845 = stablehlo.multiply %3844, %3637 : tensor<3x3x256x256xf32>
      %3846 = stablehlo.convert %3716 : tensor<f32>
      %3847 = stablehlo.broadcast_in_dim %3846, dims = [] : (tensor<f32>) -> tensor<3x3x256x256xf32>
      %3848 = stablehlo.multiply %3847, %3640 : tensor<3x3x256x256xf32>
      %3849 = stablehlo.convert %3716 : tensor<f32>
      %3850 = stablehlo.broadcast_in_dim %3849, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3851 = stablehlo.multiply %3850, %3643 : tensor<512xf32>
      %3852 = stablehlo.convert %3716 : tensor<f32>
      %3853 = stablehlo.broadcast_in_dim %3852, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3854 = stablehlo.multiply %3853, %3646 : tensor<512xf32>
      %3855 = stablehlo.convert %3716 : tensor<f32>
      %3856 = stablehlo.broadcast_in_dim %3855, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3857 = stablehlo.multiply %3856, %3649 : tensor<512xf32>
      %3858 = stablehlo.convert %3716 : tensor<f32>
      %3859 = stablehlo.broadcast_in_dim %3858, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3860 = stablehlo.multiply %3859, %3652 : tensor<512xf32>
      %3861 = stablehlo.convert %3716 : tensor<f32>
      %3862 = stablehlo.broadcast_in_dim %3861, dims = [] : (tensor<f32>) -> tensor<3x3x256x512xf32>
      %3863 = stablehlo.multiply %3862, %3655 : tensor<3x3x256x512xf32>
      %3864 = stablehlo.convert %3716 : tensor<f32>
      %3865 = stablehlo.broadcast_in_dim %3864, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %3866 = stablehlo.multiply %3865, %3658 : tensor<3x3x512x512xf32>
      %3867 = stablehlo.convert %3716 : tensor<f32>
      %3868 = stablehlo.broadcast_in_dim %3867, dims = [] : (tensor<f32>) -> tensor<1x1x256x512xf32>
      %3869 = stablehlo.multiply %3868, %3661 : tensor<1x1x256x512xf32>
      %3870 = stablehlo.convert %3716 : tensor<f32>
      %3871 = stablehlo.broadcast_in_dim %3870, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3872 = stablehlo.multiply %3871, %3664 : tensor<512xf32>
      %3873 = stablehlo.convert %3716 : tensor<f32>
      %3874 = stablehlo.broadcast_in_dim %3873, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3875 = stablehlo.multiply %3874, %3667 : tensor<512xf32>
      %3876 = stablehlo.convert %3716 : tensor<f32>
      %3877 = stablehlo.broadcast_in_dim %3876, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3878 = stablehlo.multiply %3877, %3670 : tensor<512xf32>
      %3879 = stablehlo.convert %3716 : tensor<f32>
      %3880 = stablehlo.broadcast_in_dim %3879, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3881 = stablehlo.multiply %3880, %3673 : tensor<512xf32>
      %3882 = stablehlo.convert %3716 : tensor<f32>
      %3883 = stablehlo.broadcast_in_dim %3882, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3884 = stablehlo.multiply %3883, %3676 : tensor<512xf32>
      %3885 = stablehlo.convert %3716 : tensor<f32>
      %3886 = stablehlo.broadcast_in_dim %3885, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %3887 = stablehlo.multiply %3886, %3679 : tensor<512xf32>
      %3888 = stablehlo.convert %3716 : tensor<f32>
      %3889 = stablehlo.broadcast_in_dim %3888, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %3890 = stablehlo.multiply %3889, %3682 : tensor<3x3x512x512xf32>
      %3891 = stablehlo.convert %3716 : tensor<f32>
      %3892 = stablehlo.broadcast_in_dim %3891, dims = [] : (tensor<f32>) -> tensor<3x3x512x512xf32>
      %3893 = stablehlo.multiply %3892, %3685 : tensor<3x3x512x512xf32>
      %3894 = stablehlo.convert %3716 : tensor<f32>
      %3895 = stablehlo.broadcast_in_dim %3894, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3896 = stablehlo.multiply %3895, %3688 : tensor<64xf32>
      %3897 = stablehlo.convert %3716 : tensor<f32>
      %3898 = stablehlo.broadcast_in_dim %3897, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %3899 = stablehlo.multiply %3898, %3691 : tensor<64xf32>
      %3900 = stablehlo.convert %3716 : tensor<f32>
      %3901 = stablehlo.broadcast_in_dim %3900, dims = [] : (tensor<f32>) -> tensor<7x7x3x64xf32>
      %3902 = stablehlo.multiply %3901, %3694 : tensor<7x7x3x64xf32>
      %c_1029 = stablehlo.constant dense<2147483647> : tensor<i32>
      %3903 = stablehlo.compare  LT, %arg295, %c_1029,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_1030 = stablehlo.constant dense<1> : tensor<i32>
      %3904 = stablehlo.add %arg295, %c_1030 : tensor<i32>
      %c_1031 = stablehlo.constant dense<2147483647> : tensor<i32>
      %3905 = func.call @_where_534(%3903, %3904, %c_1031) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %3906 = stablehlo.add %arg171, %3719 : tensor<1000xf32>
      %3907 = stablehlo.add %arg172, %3722 : tensor<512x1000xf32>
      %3908 = stablehlo.add %arg173, %3725 : tensor<64xf32>
      %3909 = stablehlo.add %arg174, %3728 : tensor<64xf32>
      %3910 = stablehlo.add %arg175, %3731 : tensor<64xf32>
      %3911 = stablehlo.add %arg176, %3734 : tensor<64xf32>
      %3912 = stablehlo.add %arg177, %3737 : tensor<3x3x64x64xf32>
      %3913 = stablehlo.add %arg178, %3740 : tensor<3x3x64x64xf32>
      %3914 = stablehlo.add %arg179, %3743 : tensor<64xf32>
      %3915 = stablehlo.add %arg180, %3746 : tensor<64xf32>
      %3916 = stablehlo.add %arg181, %3749 : tensor<64xf32>
      %3917 = stablehlo.add %arg182, %3752 : tensor<64xf32>
      %3918 = stablehlo.add %arg183, %3755 : tensor<3x3x64x64xf32>
      %3919 = stablehlo.add %arg184, %3758 : tensor<3x3x64x64xf32>
      %3920 = stablehlo.add %arg185, %3761 : tensor<128xf32>
      %3921 = stablehlo.add %arg186, %3764 : tensor<128xf32>
      %3922 = stablehlo.add %arg187, %3767 : tensor<128xf32>
      %3923 = stablehlo.add %arg188, %3770 : tensor<128xf32>
      %3924 = stablehlo.add %arg189, %3773 : tensor<3x3x64x128xf32>
      %3925 = stablehlo.add %arg190, %3776 : tensor<3x3x128x128xf32>
      %3926 = stablehlo.add %arg191, %3779 : tensor<1x1x64x128xf32>
      %3927 = stablehlo.add %arg192, %3782 : tensor<128xf32>
      %3928 = stablehlo.add %arg193, %3785 : tensor<128xf32>
      %3929 = stablehlo.add %arg194, %3788 : tensor<128xf32>
      %3930 = stablehlo.add %arg195, %3791 : tensor<128xf32>
      %3931 = stablehlo.add %arg196, %3794 : tensor<128xf32>
      %3932 = stablehlo.add %arg197, %3797 : tensor<128xf32>
      %3933 = stablehlo.add %arg198, %3800 : tensor<3x3x128x128xf32>
      %3934 = stablehlo.add %arg199, %3803 : tensor<3x3x128x128xf32>
      %3935 = stablehlo.add %arg200, %3806 : tensor<256xf32>
      %3936 = stablehlo.add %arg201, %3809 : tensor<256xf32>
      %3937 = stablehlo.add %arg202, %3812 : tensor<256xf32>
      %3938 = stablehlo.add %arg203, %3815 : tensor<256xf32>
      %3939 = stablehlo.add %arg204, %3818 : tensor<3x3x128x256xf32>
      %3940 = stablehlo.add %arg205, %3821 : tensor<3x3x256x256xf32>
      %3941 = stablehlo.add %arg206, %3824 : tensor<1x1x128x256xf32>
      %3942 = stablehlo.add %arg207, %3827 : tensor<256xf32>
      %3943 = stablehlo.add %arg208, %3830 : tensor<256xf32>
      %3944 = stablehlo.add %arg209, %3833 : tensor<256xf32>
      %3945 = stablehlo.add %arg210, %3836 : tensor<256xf32>
      %3946 = stablehlo.add %arg211, %3839 : tensor<256xf32>
      %3947 = stablehlo.add %arg212, %3842 : tensor<256xf32>
      %3948 = stablehlo.add %arg213, %3845 : tensor<3x3x256x256xf32>
      %3949 = stablehlo.add %arg214, %3848 : tensor<3x3x256x256xf32>
      %3950 = stablehlo.add %arg215, %3851 : tensor<512xf32>
      %3951 = stablehlo.add %arg216, %3854 : tensor<512xf32>
      %3952 = stablehlo.add %arg217, %3857 : tensor<512xf32>
      %3953 = stablehlo.add %arg218, %3860 : tensor<512xf32>
      %3954 = stablehlo.add %arg219, %3863 : tensor<3x3x256x512xf32>
      %3955 = stablehlo.add %arg220, %3866 : tensor<3x3x512x512xf32>
      %3956 = stablehlo.add %arg221, %3869 : tensor<1x1x256x512xf32>
      %3957 = stablehlo.add %arg222, %3872 : tensor<512xf32>
      %3958 = stablehlo.add %arg223, %3875 : tensor<512xf32>
      %3959 = stablehlo.add %arg224, %3878 : tensor<512xf32>
      %3960 = stablehlo.add %arg225, %3881 : tensor<512xf32>
      %3961 = stablehlo.add %arg226, %3884 : tensor<512xf32>
      %3962 = stablehlo.add %arg227, %3887 : tensor<512xf32>
      %3963 = stablehlo.add %arg228, %3890 : tensor<3x3x512x512xf32>
      %3964 = stablehlo.add %arg229, %3893 : tensor<3x3x512x512xf32>
      %3965 = stablehlo.add %arg230, %3896 : tensor<64xf32>
      %3966 = stablehlo.add %arg231, %3899 : tensor<64xf32>
      %3967 = stablehlo.add %arg232, %3902 : tensor<7x7x3x64xf32>
      %c_1032 = stablehlo.constant dense<1> : tensor<i32>
      %3968 = stablehlo.add %arg170, %c_1032 : tensor<i32>
      %3969 = func.call @_where_536(%3155, %3325, %arg233) : (tensor<i1>, tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
      %3970 = func.call @_where_538(%3155, %3328, %arg234) : (tensor<i1>, tensor<512x1000xf32>, tensor<512x1000xf32>) -> tensor<512x1000xf32>
      %3971 = func.call @_where_540(%3155, %3331, %arg235) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %3972 = func.call @_where_540(%3155, %3334, %arg236) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %3973 = func.call @_where_540(%3155, %3337, %arg237) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %3974 = func.call @_where_540(%3155, %3340, %arg238) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %3975 = func.call @_where_542(%3155, %3343, %arg239) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %3976 = func.call @_where_542(%3155, %3346, %arg240) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %3977 = func.call @_where_540(%3155, %3349, %arg241) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %3978 = func.call @_where_540(%3155, %3352, %arg242) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %3979 = func.call @_where_540(%3155, %3355, %arg243) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %3980 = func.call @_where_540(%3155, %3358, %arg244) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %3981 = func.call @_where_542(%3155, %3361, %arg245) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %3982 = func.call @_where_542(%3155, %3364, %arg246) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %3983 = func.call @_where_544(%3155, %3367, %arg247) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3984 = func.call @_where_544(%3155, %3370, %arg248) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3985 = func.call @_where_544(%3155, %3373, %arg249) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3986 = func.call @_where_544(%3155, %3376, %arg250) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3987 = func.call @_where_546(%3155, %3379, %arg251) : (tensor<i1>, tensor<3x3x64x128xf32>, tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
      %3988 = func.call @_where_548(%3155, %3382, %arg252) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %3989 = func.call @_where_550(%3155, %3385, %arg253) : (tensor<i1>, tensor<1x1x64x128xf32>, tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
      %3990 = func.call @_where_544(%3155, %3388, %arg254) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3991 = func.call @_where_544(%3155, %3391, %arg255) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3992 = func.call @_where_544(%3155, %3394, %arg256) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3993 = func.call @_where_544(%3155, %3397, %arg257) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3994 = func.call @_where_544(%3155, %3400, %arg258) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3995 = func.call @_where_544(%3155, %3403, %arg259) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %3996 = func.call @_where_548(%3155, %3406, %arg260) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %3997 = func.call @_where_548(%3155, %3409, %arg261) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %3998 = func.call @_where_552(%3155, %3412, %arg262) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %3999 = func.call @_where_552(%3155, %3415, %arg263) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4000 = func.call @_where_552(%3155, %3418, %arg264) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4001 = func.call @_where_552(%3155, %3421, %arg265) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4002 = func.call @_where_554(%3155, %3424, %arg266) : (tensor<i1>, tensor<3x3x128x256xf32>, tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
      %4003 = func.call @_where_556(%3155, %3427, %arg267) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4004 = func.call @_where_558(%3155, %3430, %arg268) : (tensor<i1>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
      %4005 = func.call @_where_552(%3155, %3433, %arg269) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4006 = func.call @_where_552(%3155, %3436, %arg270) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4007 = func.call @_where_552(%3155, %3439, %arg271) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4008 = func.call @_where_552(%3155, %3442, %arg272) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4009 = func.call @_where_552(%3155, %3445, %arg273) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4010 = func.call @_where_552(%3155, %3448, %arg274) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4011 = func.call @_where_556(%3155, %3451, %arg275) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4012 = func.call @_where_556(%3155, %3454, %arg276) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4013 = func.call @_where_560(%3155, %3457, %arg277) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4014 = func.call @_where_560(%3155, %3460, %arg278) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4015 = func.call @_where_560(%3155, %3463, %arg279) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4016 = func.call @_where_560(%3155, %3466, %arg280) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4017 = func.call @_where_562(%3155, %3469, %arg281) : (tensor<i1>, tensor<3x3x256x512xf32>, tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
      %4018 = func.call @_where_564(%3155, %3472, %arg282) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %4019 = func.call @_where_566(%3155, %3475, %arg283) : (tensor<i1>, tensor<1x1x256x512xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
      %4020 = func.call @_where_560(%3155, %3478, %arg284) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4021 = func.call @_where_560(%3155, %3481, %arg285) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4022 = func.call @_where_560(%3155, %3484, %arg286) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4023 = func.call @_where_560(%3155, %3487, %arg287) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4024 = func.call @_where_560(%3155, %3490, %arg288) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4025 = func.call @_where_560(%3155, %3493, %arg289) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4026 = func.call @_where_564(%3155, %3496, %arg290) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %4027 = func.call @_where_564(%3155, %3499, %arg291) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %4028 = func.call @_where_540(%3155, %3502, %arg292) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4029 = func.call @_where_540(%3155, %3505, %arg293) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4030 = func.call @_where_568(%3155, %3508, %arg294) : (tensor<i1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
      %4031 = func.call @_where_534(%3155, %3905, %arg295) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %4032 = func.call @_where_536(%3155, %3906, %arg171) : (tensor<i1>, tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
      %4033 = func.call @_where_538(%3155, %3907, %arg172) : (tensor<i1>, tensor<512x1000xf32>, tensor<512x1000xf32>) -> tensor<512x1000xf32>
      %4034 = func.call @_where_540(%3155, %3908, %arg173) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4035 = func.call @_where_540(%3155, %3909, %arg174) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4036 = func.call @_where_540(%3155, %3910, %arg175) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4037 = func.call @_where_540(%3155, %3911, %arg176) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4038 = func.call @_where_542(%3155, %3912, %arg177) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %4039 = func.call @_where_542(%3155, %3913, %arg178) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %4040 = func.call @_where_540(%3155, %3914, %arg179) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4041 = func.call @_where_540(%3155, %3915, %arg180) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4042 = func.call @_where_540(%3155, %3916, %arg181) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4043 = func.call @_where_540(%3155, %3917, %arg182) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4044 = func.call @_where_542(%3155, %3918, %arg183) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %4045 = func.call @_where_542(%3155, %3919, %arg184) : (tensor<i1>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32>
      %4046 = func.call @_where_544(%3155, %3920, %arg185) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4047 = func.call @_where_544(%3155, %3921, %arg186) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4048 = func.call @_where_544(%3155, %3922, %arg187) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4049 = func.call @_where_544(%3155, %3923, %arg188) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4050 = func.call @_where_546(%3155, %3924, %arg189) : (tensor<i1>, tensor<3x3x64x128xf32>, tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32>
      %4051 = func.call @_where_548(%3155, %3925, %arg190) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %4052 = func.call @_where_550(%3155, %3926, %arg191) : (tensor<i1>, tensor<1x1x64x128xf32>, tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32>
      %4053 = func.call @_where_544(%3155, %3927, %arg192) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4054 = func.call @_where_544(%3155, %3928, %arg193) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4055 = func.call @_where_544(%3155, %3929, %arg194) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4056 = func.call @_where_544(%3155, %3930, %arg195) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4057 = func.call @_where_544(%3155, %3931, %arg196) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4058 = func.call @_where_544(%3155, %3932, %arg197) : (tensor<i1>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %4059 = func.call @_where_548(%3155, %3933, %arg198) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %4060 = func.call @_where_548(%3155, %3934, %arg199) : (tensor<i1>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32>
      %4061 = func.call @_where_552(%3155, %3935, %arg200) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4062 = func.call @_where_552(%3155, %3936, %arg201) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4063 = func.call @_where_552(%3155, %3937, %arg202) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4064 = func.call @_where_552(%3155, %3938, %arg203) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4065 = func.call @_where_554(%3155, %3939, %arg204) : (tensor<i1>, tensor<3x3x128x256xf32>, tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32>
      %4066 = func.call @_where_556(%3155, %3940, %arg205) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4067 = func.call @_where_558(%3155, %3941, %arg206) : (tensor<i1>, tensor<1x1x128x256xf32>, tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32>
      %4068 = func.call @_where_552(%3155, %3942, %arg207) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4069 = func.call @_where_552(%3155, %3943, %arg208) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4070 = func.call @_where_552(%3155, %3944, %arg209) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4071 = func.call @_where_552(%3155, %3945, %arg210) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4072 = func.call @_where_552(%3155, %3946, %arg211) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4073 = func.call @_where_552(%3155, %3947, %arg212) : (tensor<i1>, tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %4074 = func.call @_where_556(%3155, %3948, %arg213) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4075 = func.call @_where_556(%3155, %3949, %arg214) : (tensor<i1>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
      %4076 = func.call @_where_560(%3155, %3950, %arg215) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4077 = func.call @_where_560(%3155, %3951, %arg216) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4078 = func.call @_where_560(%3155, %3952, %arg217) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4079 = func.call @_where_560(%3155, %3953, %arg218) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4080 = func.call @_where_562(%3155, %3954, %arg219) : (tensor<i1>, tensor<3x3x256x512xf32>, tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32>
      %4081 = func.call @_where_564(%3155, %3955, %arg220) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %4082 = func.call @_where_566(%3155, %3956, %arg221) : (tensor<i1>, tensor<1x1x256x512xf32>, tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
      %4083 = func.call @_where_560(%3155, %3957, %arg222) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4084 = func.call @_where_560(%3155, %3958, %arg223) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4085 = func.call @_where_560(%3155, %3959, %arg224) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4086 = func.call @_where_560(%3155, %3960, %arg225) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4087 = func.call @_where_560(%3155, %3961, %arg226) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4088 = func.call @_where_560(%3155, %3962, %arg227) : (tensor<i1>, tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
      %4089 = func.call @_where_564(%3155, %3963, %arg228) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %4090 = func.call @_where_564(%3155, %3964, %arg229) : (tensor<i1>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32>
      %4091 = func.call @_where_540(%3155, %3965, %arg230) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4092 = func.call @_where_540(%3155, %3966, %arg231) : (tensor<i1>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
      %4093 = func.call @_where_568(%3155, %3967, %arg232) : (tensor<i1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
      %4094 = stablehlo.broadcast_in_dim %3201, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %4095 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %4096 = stablehlo.broadcast_in_dim %3202, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %4097 = stablehlo.broadcast_in_dim %3168, dims = [] : (tensor<f32>) -> tensor<1xf32>
      sdy.return %3968, %4032, %4033, %4034, %4035, %4036, %4037, %4038, %4039, %4040, %4041, %4042, %4043, %4044, %4045, %4046, %4047, %4048, %4049, %4050, %4051, %4052, %4053, %4054, %4055, %4056, %4057, %4058, %4059, %4060, %4061, %4062, %4063, %4064, %4065, %4066, %4067, %4068, %4069, %4070, %4071, %4072, %4073, %4074, %4075, %4076, %4077, %4078, %4079, %4080, %4081, %4082, %4083, %4084, %4085, %4086, %4087, %4088, %4089, %4090, %4091, %4092, %4093, %3969, %3970, %3971, %3972, %3973, %3974, %3975, %3976, %3977, %3978, %3979, %3980, %3981, %3982, %3983, %3984, %3985, %3986, %3987, %3988, %3989, %3990, %3991, %3992, %3993, %3994, %3995, %3996, %3997, %3998, %3999, %4000, %4001, %4002, %4003, %4004, %4005, %4006, %4007, %4008, %4009, %4010, %4011, %4012, %4013, %4014, %4015, %4016, %4017, %4018, %4019, %4020, %4021, %4022, %4023, %4024, %4025, %4026, %4027, %4028, %4029, %4030, %4031, %3244, %3246, %3248, %3250, %3252, %3254, %3256, %3258, %3260, %3262, %3264, %3266, %3268, %3270, %3272, %3274, %3276, %3278, %3280, %3282, %3284, %3286, %3288, %3290, %3292, %3294, %3296, %3298, %3300, %3302, %3304, %3306, %3308, %3310, %3312, %3314, %3316, %3318, %3320, %3322, %3175, %3168, %4094, %4095, %4096, %4097 : tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
    } : (tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<4x256x224x224x3xf16>, tensor<4x256xi32>) -> (tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8, %0#9, %0#10, %0#11, %0#12, %0#13, %0#14, %0#15, %0#16, %0#17, %0#18, %0#19, %0#20, %0#21, %0#22, %0#23, %0#24, %0#25, %0#26, %0#27, %0#28, %0#29, %0#30, %0#31, %0#32, %0#33, %0#34, %0#35, %0#36, %0#37, %0#38, %0#39, %0#40, %0#41, %0#42, %0#43, %0#44, %0#45, %0#46, %0#47, %0#48, %0#49, %0#50, %0#51, %0#52, %0#53, %0#54, %0#55, %0#56, %0#57, %0#58, %0#59, %0#60, %0#61, %0#62, %0#63, %0#64, %0#65, %0#66, %0#67, %0#68, %0#69, %0#70, %0#71, %0#72, %0#73, %0#74, %0#75, %0#76, %0#77, %0#78, %0#79, %0#80, %0#81, %0#82, %0#83, %0#84, %0#85, %0#86, %0#87, %0#88, %0#89, %0#90, %0#91, %0#92, %0#93, %0#94, %0#95, %0#96, %0#97, %0#98, %0#99, %0#100, %0#101, %0#102, %0#103, %0#104, %0#105, %0#106, %0#107, %0#108, %0#109, %0#110, %0#111, %0#112, %0#113, %0#114, %0#115, %0#116, %0#117, %0#118, %0#119, %0#120, %0#121, %0#122, %0#123, %0#124, %0#125, %0#126, %0#127, %0#128, %0#129, %0#130, %0#131, %0#132, %0#133, %0#134, %0#135, %0#136, %0#137, %0#138, %0#139, %0#140, %0#141, %0#142, %0#143, %0#144, %0#145, %0#146, %0#147, %0#148, %0#149, %0#150, %0#151, %0#152, %0#153, %0#154, %0#155, %0#156, %0#157, %0#158, %0#159, %0#160, %0#161, %0#162, %0#163, %0#164, %0#165, %0#166, %0#167, %0#168, %0#169, %0#170, %0#171 : tensor<i32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<1000xf32>, tensor<512x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<3x3x64x64xf32>, tensor<3x3x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x64x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<3x3x128x128xf32>, tensor<3x3x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x128x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x128x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<3x3x256x256xf32>, tensor<3x3x256x256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x256x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<3x3x512x512xf32>, tensor<3x3x512x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<i32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
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
  func.func private @relu_88(%arg0: tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xf16>
    return %1 : tensor<256x28x28x128xf16>
  }
  func.func private @relu_136(%arg0: tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xf16>
    return %1 : tensor<256x14x14x256xf16>
  }
  func.func private @relu_184(%arg0: tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16> {
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
  func.func private @log_softmax_269(%arg0: tensor<256x1000xf16>, %arg1: tensor<256x1xf16>, %arg2: tensor<256x1000xf16>) -> tensor<256x1000xf16> {
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
  func.func private @_where_477(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @_where_480(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @_where_482(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @log_softmax_484(%arg0: tensor<256x1000xf16>) -> tensor<256x1000xf16> {
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
  func.func private @clip_526(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %3 = stablehlo.minimum %arg2, %2 : tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @_where_534(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @_where_536(%arg0: tensor<i1>, %arg1: tensor<1000xf32>, %arg2: tensor<1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
  func.func private @_where_538(%arg0: tensor<i1>, %arg1: tensor<512x1000xf32>, %arg2: tensor<512x1000xf32>) -> tensor<512x1000xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<512x1000xf32>
    return %0 : tensor<512x1000xf32>
  }
  func.func private @_where_540(%arg0: tensor<i1>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @_where_542(%arg0: tensor<i1>, %arg1: tensor<3x3x64x64xf32>, %arg2: tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x64x64xf32>
    return %0 : tensor<3x3x64x64xf32>
  }
  func.func private @_where_544(%arg0: tensor<i1>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  func.func private @_where_546(%arg0: tensor<i1>, %arg1: tensor<3x3x64x128xf32>, %arg2: tensor<3x3x64x128xf32>) -> tensor<3x3x64x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x64x128xf32>
    return %0 : tensor<3x3x64x128xf32>
  }
  func.func private @_where_548(%arg0: tensor<i1>, %arg1: tensor<3x3x128x128xf32>, %arg2: tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x128x128xf32>
    return %0 : tensor<3x3x128x128xf32>
  }
  func.func private @_where_550(%arg0: tensor<i1>, %arg1: tensor<1x1x64x128xf32>, %arg2: tensor<1x1x64x128xf32>) -> tensor<1x1x64x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x64x128xf32>
    return %0 : tensor<1x1x64x128xf32>
  }
  func.func private @_where_552(%arg0: tensor<i1>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  func.func private @_where_554(%arg0: tensor<i1>, %arg1: tensor<3x3x128x256xf32>, %arg2: tensor<3x3x128x256xf32>) -> tensor<3x3x128x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x128x256xf32>
    return %0 : tensor<3x3x128x256xf32>
  }
  func.func private @_where_556(%arg0: tensor<i1>, %arg1: tensor<3x3x256x256xf32>, %arg2: tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x256x256xf32>
    return %0 : tensor<3x3x256x256xf32>
  }
  func.func private @_where_558(%arg0: tensor<i1>, %arg1: tensor<1x1x128x256xf32>, %arg2: tensor<1x1x128x256xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x128x256xf32>
    return %0 : tensor<1x1x128x256xf32>
  }
  func.func private @_where_560(%arg0: tensor<i1>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @_where_562(%arg0: tensor<i1>, %arg1: tensor<3x3x256x512xf32>, %arg2: tensor<3x3x256x512xf32>) -> tensor<3x3x256x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x256x512xf32>
    return %0 : tensor<3x3x256x512xf32>
  }
  func.func private @_where_564(%arg0: tensor<i1>, %arg1: tensor<3x3x512x512xf32>, %arg2: tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<3x3x512x512xf32>
    return %0 : tensor<3x3x512x512xf32>
  }
  func.func private @_where_566(%arg0: tensor<i1>, %arg1: tensor<1x1x256x512xf32>, %arg2: tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<1x1x256x512xf32>
    return %0 : tensor<1x1x256x512xf32>
  }
  func.func private @_where_568(%arg0: tensor<i1>, %arg1: tensor<7x7x3x64xf32>, %arg2: tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<7x7x3x64xf32>
    return %0 : tensor<7x7x3x64xf32>
  }
}
