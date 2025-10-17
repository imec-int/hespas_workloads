module @jit_call_wrapped attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=4]>
  func.func public @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<1x1x64x64xf32>, %arg7: tensor<3x3x64x64xf32>, %arg8: tensor<1x1x64x256xf32>, %arg9: tensor<1x1x64x256xf32>, %arg10: tensor<256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<256xf32>, %arg17: tensor<256xf32>, %arg18: tensor<1x1x256x64xf32>, %arg19: tensor<3x3x64x64xf32>, %arg20: tensor<1x1x64x256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256xf32>, %arg25: tensor<1024xf32>, %arg26: tensor<1024xf32>, %arg27: tensor<1x1x1024x256xf32>, %arg28: tensor<3x3x256x256xf32>, %arg29: tensor<1x1x256x1024xf32>, %arg30: tensor<256xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256xf32>, %arg34: tensor<1024xf32>, %arg35: tensor<1024xf32>, %arg36: tensor<1x1x1024x256xf32>, %arg37: tensor<3x3x256x256xf32>, %arg38: tensor<1x1x256x1024xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256xf32>, %arg43: tensor<1024xf32>, %arg44: tensor<1024xf32>, %arg45: tensor<1x1x1024x256xf32>, %arg46: tensor<3x3x256x256xf32>, %arg47: tensor<1x1x256x1024xf32>, %arg48: tensor<512xf32>, %arg49: tensor<512xf32>, %arg50: tensor<512xf32>, %arg51: tensor<512xf32>, %arg52: tensor<2048xf32>, %arg53: tensor<2048xf32>, %arg54: tensor<1x1x1024x512xf32>, %arg55: tensor<3x3x512x512xf32>, %arg56: tensor<1x1x512x2048xf32>, %arg57: tensor<1x1x1024x2048xf32>, %arg58: tensor<2048xf32>, %arg59: tensor<2048xf32>, %arg60: tensor<512xf32>, %arg61: tensor<512xf32>, %arg62: tensor<512xf32>, %arg63: tensor<512xf32>, %arg64: tensor<2048xf32>, %arg65: tensor<2048xf32>, %arg66: tensor<1x1x2048x512xf32>, %arg67: tensor<3x3x512x512xf32>, %arg68: tensor<1x1x512x2048xf32>, %arg69: tensor<512xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<512xf32>, %arg73: tensor<2048xf32>, %arg74: tensor<2048xf32>, %arg75: tensor<1x1x2048x512xf32>, %arg76: tensor<3x3x512x512xf32>, %arg77: tensor<1x1x512x2048xf32>, %arg78: tensor<64xf32>, %arg79: tensor<64xf32>, %arg80: tensor<64xf32>, %arg81: tensor<64xf32>, %arg82: tensor<256xf32>, %arg83: tensor<256xf32>, %arg84: tensor<1x1x256x64xf32>, %arg85: tensor<3x3x64x64xf32>, %arg86: tensor<1x1x64x256xf32>, %arg87: tensor<128xf32>, %arg88: tensor<128xf32>, %arg89: tensor<128xf32>, %arg90: tensor<128xf32>, %arg91: tensor<512xf32>, %arg92: tensor<512xf32>, %arg93: tensor<1x1x256x128xf32>, %arg94: tensor<3x3x128x128xf32>, %arg95: tensor<1x1x128x512xf32>, %arg96: tensor<1x1x256x512xf32>, %arg97: tensor<512xf32>, %arg98: tensor<512xf32>, %arg99: tensor<128xf32>, %arg100: tensor<128xf32>, %arg101: tensor<128xf32>, %arg102: tensor<128xf32>, %arg103: tensor<512xf32>, %arg104: tensor<512xf32>, %arg105: tensor<1x1x512x128xf32>, %arg106: tensor<3x3x128x128xf32>, %arg107: tensor<1x1x128x512xf32>, %arg108: tensor<128xf32>, %arg109: tensor<128xf32>, %arg110: tensor<128xf32>, %arg111: tensor<128xf32>, %arg112: tensor<512xf32>, %arg113: tensor<512xf32>, %arg114: tensor<1x1x512x128xf32>, %arg115: tensor<3x3x128x128xf32>, %arg116: tensor<1x1x128x512xf32>, %arg117: tensor<128xf32>, %arg118: tensor<128xf32>, %arg119: tensor<128xf32>, %arg120: tensor<128xf32>, %arg121: tensor<512xf32>, %arg122: tensor<512xf32>, %arg123: tensor<1x1x512x128xf32>, %arg124: tensor<3x3x128x128xf32>, %arg125: tensor<1x1x128x512xf32>, %arg126: tensor<256xf32>, %arg127: tensor<256xf32>, %arg128: tensor<256xf32>, %arg129: tensor<256xf32>, %arg130: tensor<1024xf32>, %arg131: tensor<1024xf32>, %arg132: tensor<1x1x512x256xf32>, %arg133: tensor<3x3x256x256xf32>, %arg134: tensor<1x1x256x1024xf32>, %arg135: tensor<1x1x512x1024xf32>, %arg136: tensor<1024xf32>, %arg137: tensor<1024xf32>, %arg138: tensor<256xf32>, %arg139: tensor<256xf32>, %arg140: tensor<256xf32>, %arg141: tensor<256xf32>, %arg142: tensor<1024xf32>, %arg143: tensor<1024xf32>, %arg144: tensor<1x1x1024x256xf32>, %arg145: tensor<3x3x256x256xf32>, %arg146: tensor<1x1x256x1024xf32>, %arg147: tensor<256xf32>, %arg148: tensor<256xf32>, %arg149: tensor<256xf32>, %arg150: tensor<256xf32>, %arg151: tensor<1024xf32>, %arg152: tensor<1024xf32>, %arg153: tensor<1x1x1024x256xf32>, %arg154: tensor<3x3x256x256xf32>, %arg155: tensor<1x1x256x1024xf32>, %arg156: tensor<1000xf32>, %arg157: tensor<2048x1000xf32>, %arg158: tensor<64xf32>, %arg159: tensor<64xf32>, %arg160: tensor<7x7x3x64xf32>, %arg161: tensor<64xf32>, %arg162: tensor<64xf32>, %arg163: tensor<64xf32>, %arg164: tensor<64xf32>, %arg165: tensor<256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256xf32>, %arg168: tensor<256xf32>, %arg169: tensor<64xf32>, %arg170: tensor<64xf32>, %arg171: tensor<64xf32>, %arg172: tensor<64xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<256xf32>, %arg179: tensor<1024xf32>, %arg180: tensor<1024xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<256xf32>, %arg184: tensor<256xf32>, %arg185: tensor<1024xf32>, %arg186: tensor<1024xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<256xf32>, %arg190: tensor<256xf32>, %arg191: tensor<1024xf32>, %arg192: tensor<1024xf32>, %arg193: tensor<512xf32>, %arg194: tensor<512xf32>, %arg195: tensor<512xf32>, %arg196: tensor<512xf32>, %arg197: tensor<2048xf32>, %arg198: tensor<2048xf32>, %arg199: tensor<2048xf32>, %arg200: tensor<2048xf32>, %arg201: tensor<512xf32>, %arg202: tensor<512xf32>, %arg203: tensor<512xf32>, %arg204: tensor<512xf32>, %arg205: tensor<2048xf32>, %arg206: tensor<2048xf32>, %arg207: tensor<512xf32>, %arg208: tensor<512xf32>, %arg209: tensor<512xf32>, %arg210: tensor<512xf32>, %arg211: tensor<2048xf32>, %arg212: tensor<2048xf32>, %arg213: tensor<64xf32>, %arg214: tensor<64xf32>, %arg215: tensor<64xf32>, %arg216: tensor<64xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<128xf32>, %arg220: tensor<128xf32>, %arg221: tensor<128xf32>, %arg222: tensor<128xf32>, %arg223: tensor<512xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<128xf32>, %arg228: tensor<128xf32>, %arg229: tensor<128xf32>, %arg230: tensor<128xf32>, %arg231: tensor<512xf32>, %arg232: tensor<512xf32>, %arg233: tensor<128xf32>, %arg234: tensor<128xf32>, %arg235: tensor<128xf32>, %arg236: tensor<128xf32>, %arg237: tensor<512xf32>, %arg238: tensor<512xf32>, %arg239: tensor<128xf32>, %arg240: tensor<128xf32>, %arg241: tensor<128xf32>, %arg242: tensor<128xf32>, %arg243: tensor<512xf32>, %arg244: tensor<512xf32>, %arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<256xf32>, %arg249: tensor<1024xf32>, %arg250: tensor<1024xf32>, %arg251: tensor<1024xf32>, %arg252: tensor<1024xf32>, %arg253: tensor<256xf32>, %arg254: tensor<256xf32>, %arg255: tensor<256xf32>, %arg256: tensor<256xf32>, %arg257: tensor<1024xf32>, %arg258: tensor<1024xf32>, %arg259: tensor<256xf32>, %arg260: tensor<256xf32>, %arg261: tensor<256xf32>, %arg262: tensor<256xf32>, %arg263: tensor<1024xf32>, %arg264: tensor<1024xf32>, %arg265: tensor<64xf32>, %arg266: tensor<64xf32>, %arg267: tensor<4x256x224x224x3xf16>, %arg268: tensor<4x256xi32>) -> (tensor<4xf32> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %0:2 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg181, %arg182, %arg183, %arg184, %arg185, %arg186, %arg187, %arg188, %arg189, %arg190, %arg191, %arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %arg198, %arg199, %arg200, %arg201, %arg202, %arg203, %arg204, %arg205, %arg206, %arg207, %arg208, %arg209, %arg210, %arg211, %arg212, %arg213, %arg214, %arg215, %arg216, %arg217, %arg218, %arg219, %arg220, %arg221, %arg222, %arg223, %arg224, %arg225, %arg226, %arg227, %arg228, %arg229, %arg230, %arg231, %arg232, %arg233, %arg234, %arg235, %arg236, %arg237, %arg238, %arg239, %arg240, %arg241, %arg242, %arg243, %arg244, %arg245, %arg246, %arg247, %arg248, %arg249, %arg250, %arg251, %arg252, %arg253, %arg254, %arg255, %arg256, %arg257, %arg258, %arg259, %arg260, %arg261, %arg262, %arg263, %arg264, %arg265, %arg266, %arg267, %arg268) in_shardings=[<@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{"batch"}, {}, {}, {}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>] manual_axes={"batch"} (%arg269: tensor<64xf32>, %arg270: tensor<64xf32>, %arg271: tensor<64xf32>, %arg272: tensor<64xf32>, %arg273: tensor<256xf32>, %arg274: tensor<256xf32>, %arg275: tensor<1x1x64x64xf32>, %arg276: tensor<3x3x64x64xf32>, %arg277: tensor<1x1x64x256xf32>, %arg278: tensor<1x1x64x256xf32>, %arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<64xf32>, %arg282: tensor<64xf32>, %arg283: tensor<64xf32>, %arg284: tensor<64xf32>, %arg285: tensor<256xf32>, %arg286: tensor<256xf32>, %arg287: tensor<1x1x256x64xf32>, %arg288: tensor<3x3x64x64xf32>, %arg289: tensor<1x1x64x256xf32>, %arg290: tensor<256xf32>, %arg291: tensor<256xf32>, %arg292: tensor<256xf32>, %arg293: tensor<256xf32>, %arg294: tensor<1024xf32>, %arg295: tensor<1024xf32>, %arg296: tensor<1x1x1024x256xf32>, %arg297: tensor<3x3x256x256xf32>, %arg298: tensor<1x1x256x1024xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256xf32>, %arg301: tensor<256xf32>, %arg302: tensor<256xf32>, %arg303: tensor<1024xf32>, %arg304: tensor<1024xf32>, %arg305: tensor<1x1x1024x256xf32>, %arg306: tensor<3x3x256x256xf32>, %arg307: tensor<1x1x256x1024xf32>, %arg308: tensor<256xf32>, %arg309: tensor<256xf32>, %arg310: tensor<256xf32>, %arg311: tensor<256xf32>, %arg312: tensor<1024xf32>, %arg313: tensor<1024xf32>, %arg314: tensor<1x1x1024x256xf32>, %arg315: tensor<3x3x256x256xf32>, %arg316: tensor<1x1x256x1024xf32>, %arg317: tensor<512xf32>, %arg318: tensor<512xf32>, %arg319: tensor<512xf32>, %arg320: tensor<512xf32>, %arg321: tensor<2048xf32>, %arg322: tensor<2048xf32>, %arg323: tensor<1x1x1024x512xf32>, %arg324: tensor<3x3x512x512xf32>, %arg325: tensor<1x1x512x2048xf32>, %arg326: tensor<1x1x1024x2048xf32>, %arg327: tensor<2048xf32>, %arg328: tensor<2048xf32>, %arg329: tensor<512xf32>, %arg330: tensor<512xf32>, %arg331: tensor<512xf32>, %arg332: tensor<512xf32>, %arg333: tensor<2048xf32>, %arg334: tensor<2048xf32>, %arg335: tensor<1x1x2048x512xf32>, %arg336: tensor<3x3x512x512xf32>, %arg337: tensor<1x1x512x2048xf32>, %arg338: tensor<512xf32>, %arg339: tensor<512xf32>, %arg340: tensor<512xf32>, %arg341: tensor<512xf32>, %arg342: tensor<2048xf32>, %arg343: tensor<2048xf32>, %arg344: tensor<1x1x2048x512xf32>, %arg345: tensor<3x3x512x512xf32>, %arg346: tensor<1x1x512x2048xf32>, %arg347: tensor<64xf32>, %arg348: tensor<64xf32>, %arg349: tensor<64xf32>, %arg350: tensor<64xf32>, %arg351: tensor<256xf32>, %arg352: tensor<256xf32>, %arg353: tensor<1x1x256x64xf32>, %arg354: tensor<3x3x64x64xf32>, %arg355: tensor<1x1x64x256xf32>, %arg356: tensor<128xf32>, %arg357: tensor<128xf32>, %arg358: tensor<128xf32>, %arg359: tensor<128xf32>, %arg360: tensor<512xf32>, %arg361: tensor<512xf32>, %arg362: tensor<1x1x256x128xf32>, %arg363: tensor<3x3x128x128xf32>, %arg364: tensor<1x1x128x512xf32>, %arg365: tensor<1x1x256x512xf32>, %arg366: tensor<512xf32>, %arg367: tensor<512xf32>, %arg368: tensor<128xf32>, %arg369: tensor<128xf32>, %arg370: tensor<128xf32>, %arg371: tensor<128xf32>, %arg372: tensor<512xf32>, %arg373: tensor<512xf32>, %arg374: tensor<1x1x512x128xf32>, %arg375: tensor<3x3x128x128xf32>, %arg376: tensor<1x1x128x512xf32>, %arg377: tensor<128xf32>, %arg378: tensor<128xf32>, %arg379: tensor<128xf32>, %arg380: tensor<128xf32>, %arg381: tensor<512xf32>, %arg382: tensor<512xf32>, %arg383: tensor<1x1x512x128xf32>, %arg384: tensor<3x3x128x128xf32>, %arg385: tensor<1x1x128x512xf32>, %arg386: tensor<128xf32>, %arg387: tensor<128xf32>, %arg388: tensor<128xf32>, %arg389: tensor<128xf32>, %arg390: tensor<512xf32>, %arg391: tensor<512xf32>, %arg392: tensor<1x1x512x128xf32>, %arg393: tensor<3x3x128x128xf32>, %arg394: tensor<1x1x128x512xf32>, %arg395: tensor<256xf32>, %arg396: tensor<256xf32>, %arg397: tensor<256xf32>, %arg398: tensor<256xf32>, %arg399: tensor<1024xf32>, %arg400: tensor<1024xf32>, %arg401: tensor<1x1x512x256xf32>, %arg402: tensor<3x3x256x256xf32>, %arg403: tensor<1x1x256x1024xf32>, %arg404: tensor<1x1x512x1024xf32>, %arg405: tensor<1024xf32>, %arg406: tensor<1024xf32>, %arg407: tensor<256xf32>, %arg408: tensor<256xf32>, %arg409: tensor<256xf32>, %arg410: tensor<256xf32>, %arg411: tensor<1024xf32>, %arg412: tensor<1024xf32>, %arg413: tensor<1x1x1024x256xf32>, %arg414: tensor<3x3x256x256xf32>, %arg415: tensor<1x1x256x1024xf32>, %arg416: tensor<256xf32>, %arg417: tensor<256xf32>, %arg418: tensor<256xf32>, %arg419: tensor<256xf32>, %arg420: tensor<1024xf32>, %arg421: tensor<1024xf32>, %arg422: tensor<1x1x1024x256xf32>, %arg423: tensor<3x3x256x256xf32>, %arg424: tensor<1x1x256x1024xf32>, %arg425: tensor<1000xf32>, %arg426: tensor<2048x1000xf32>, %arg427: tensor<64xf32>, %arg428: tensor<64xf32>, %arg429: tensor<7x7x3x64xf32>, %arg430: tensor<64xf32>, %arg431: tensor<64xf32>, %arg432: tensor<64xf32>, %arg433: tensor<64xf32>, %arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<256xf32>, %arg437: tensor<256xf32>, %arg438: tensor<64xf32>, %arg439: tensor<64xf32>, %arg440: tensor<64xf32>, %arg441: tensor<64xf32>, %arg442: tensor<256xf32>, %arg443: tensor<256xf32>, %arg444: tensor<256xf32>, %arg445: tensor<256xf32>, %arg446: tensor<256xf32>, %arg447: tensor<256xf32>, %arg448: tensor<1024xf32>, %arg449: tensor<1024xf32>, %arg450: tensor<256xf32>, %arg451: tensor<256xf32>, %arg452: tensor<256xf32>, %arg453: tensor<256xf32>, %arg454: tensor<1024xf32>, %arg455: tensor<1024xf32>, %arg456: tensor<256xf32>, %arg457: tensor<256xf32>, %arg458: tensor<256xf32>, %arg459: tensor<256xf32>, %arg460: tensor<1024xf32>, %arg461: tensor<1024xf32>, %arg462: tensor<512xf32>, %arg463: tensor<512xf32>, %arg464: tensor<512xf32>, %arg465: tensor<512xf32>, %arg466: tensor<2048xf32>, %arg467: tensor<2048xf32>, %arg468: tensor<2048xf32>, %arg469: tensor<2048xf32>, %arg470: tensor<512xf32>, %arg471: tensor<512xf32>, %arg472: tensor<512xf32>, %arg473: tensor<512xf32>, %arg474: tensor<2048xf32>, %arg475: tensor<2048xf32>, %arg476: tensor<512xf32>, %arg477: tensor<512xf32>, %arg478: tensor<512xf32>, %arg479: tensor<512xf32>, %arg480: tensor<2048xf32>, %arg481: tensor<2048xf32>, %arg482: tensor<64xf32>, %arg483: tensor<64xf32>, %arg484: tensor<64xf32>, %arg485: tensor<64xf32>, %arg486: tensor<256xf32>, %arg487: tensor<256xf32>, %arg488: tensor<128xf32>, %arg489: tensor<128xf32>, %arg490: tensor<128xf32>, %arg491: tensor<128xf32>, %arg492: tensor<512xf32>, %arg493: tensor<512xf32>, %arg494: tensor<512xf32>, %arg495: tensor<512xf32>, %arg496: tensor<128xf32>, %arg497: tensor<128xf32>, %arg498: tensor<128xf32>, %arg499: tensor<128xf32>, %arg500: tensor<512xf32>, %arg501: tensor<512xf32>, %arg502: tensor<128xf32>, %arg503: tensor<128xf32>, %arg504: tensor<128xf32>, %arg505: tensor<128xf32>, %arg506: tensor<512xf32>, %arg507: tensor<512xf32>, %arg508: tensor<128xf32>, %arg509: tensor<128xf32>, %arg510: tensor<128xf32>, %arg511: tensor<128xf32>, %arg512: tensor<512xf32>, %arg513: tensor<512xf32>, %arg514: tensor<256xf32>, %arg515: tensor<256xf32>, %arg516: tensor<256xf32>, %arg517: tensor<256xf32>, %arg518: tensor<1024xf32>, %arg519: tensor<1024xf32>, %arg520: tensor<1024xf32>, %arg521: tensor<1024xf32>, %arg522: tensor<256xf32>, %arg523: tensor<256xf32>, %arg524: tensor<256xf32>, %arg525: tensor<256xf32>, %arg526: tensor<1024xf32>, %arg527: tensor<1024xf32>, %arg528: tensor<256xf32>, %arg529: tensor<256xf32>, %arg530: tensor<256xf32>, %arg531: tensor<256xf32>, %arg532: tensor<1024xf32>, %arg533: tensor<1024xf32>, %arg534: tensor<64xf32>, %arg535: tensor<64xf32>, %arg536: tensor<1x256x224x224x3xf16>, %arg537: tensor<1x256xi32>) {
      %1 = stablehlo.reshape %arg536 : (tensor<1x256x224x224x3xf16>) -> tensor<256x224x224x3xf16>
      %2 = stablehlo.reshape %arg537 : (tensor<1x256xi32>) -> tensor<256xi32>
      %3 = stablehlo.convert %arg429 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf16>
      %4 = stablehlo.convolution(%1, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf16>, tensor<7x7x3x64xf16>) -> tensor<256x112x112x64xf16>
      %5 = stablehlo.broadcast_in_dim %arg534, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %6 = stablehlo.broadcast_in_dim %arg535, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %7 = stablehlo.convert %4 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
      %8 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %9 = stablehlo.subtract %7, %8 : tensor<256x112x112x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %11 = stablehlo.add %6, %10 : tensor<1x1x1x64xf32>
      %12 = stablehlo.rsqrt %11 : tensor<1x1x1x64xf32>
      %13 = stablehlo.reshape %arg428 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %14 = stablehlo.multiply %12, %13 : tensor<1x1x1x64xf32>
      %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %16 = stablehlo.multiply %9, %15 : tensor<256x112x112x64xf32>
      %17 = stablehlo.reshape %arg427 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %19 = stablehlo.add %16, %18 : tensor<256x112x112x64xf32>
      %20 = stablehlo.convert %19 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %21 = func.call @relu(%20) : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16>
      %cst_0 = stablehlo.constant dense<0xFC00> : tensor<f16>
      %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f16>) -> tensor<f16>
      %23 = "stablehlo.reduce_window"(%21, %22) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg538: tensor<f16>, %arg539: tensor<f16>):
        %1064 = stablehlo.maximum %arg538, %arg539 : tensor<f16>
        stablehlo.return %1064 : tensor<f16>
      }) : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x56x56x64xf16>
      %24 = stablehlo.convert %arg275 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xf16>
      %25 = stablehlo.convolution(%23, %24) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x64xf16>) -> tensor<256x56x56x64xf16>
      %26 = stablehlo.broadcast_in_dim %arg430, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %27 = stablehlo.broadcast_in_dim %arg431, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %28 = stablehlo.convert %25 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %29 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %30 = stablehlo.subtract %28, %29 : tensor<256x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %31 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %32 = stablehlo.add %27, %31 : tensor<1x1x1x64xf32>
      %33 = stablehlo.rsqrt %32 : tensor<1x1x1x64xf32>
      %34 = stablehlo.reshape %arg270 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %35 = stablehlo.multiply %33, %34 : tensor<1x1x1x64xf32>
      %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %37 = stablehlo.multiply %30, %36 : tensor<256x56x56x64xf32>
      %38 = stablehlo.reshape %arg269 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %40 = stablehlo.add %37, %39 : tensor<256x56x56x64xf32>
      %41 = stablehlo.convert %40 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %42 = func.call @relu_13(%41) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %43 = stablehlo.convert %arg276 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %44 = stablehlo.convolution(%42, %43) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %45 = stablehlo.broadcast_in_dim %arg432, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %46 = stablehlo.broadcast_in_dim %arg433, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %47 = stablehlo.convert %44 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %48 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %49 = stablehlo.subtract %47, %48 : tensor<256x56x56x64xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %50 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %51 = stablehlo.add %46, %50 : tensor<1x1x1x64xf32>
      %52 = stablehlo.rsqrt %51 : tensor<1x1x1x64xf32>
      %53 = stablehlo.reshape %arg272 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %54 = stablehlo.multiply %52, %53 : tensor<1x1x1x64xf32>
      %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %56 = stablehlo.multiply %49, %55 : tensor<256x56x56x64xf32>
      %57 = stablehlo.reshape %arg271 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %59 = stablehlo.add %56, %58 : tensor<256x56x56x64xf32>
      %60 = stablehlo.convert %59 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %61 = func.call @relu_13(%60) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %62 = stablehlo.convert %arg277 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %63 = stablehlo.convolution(%61, %62) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %64 = stablehlo.broadcast_in_dim %arg434, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %65 = stablehlo.broadcast_in_dim %arg435, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %66 = stablehlo.convert %63 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %67 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %68 = stablehlo.subtract %66, %67 : tensor<256x56x56x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %69 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %70 = stablehlo.add %65, %69 : tensor<1x1x1x256xf32>
      %71 = stablehlo.rsqrt %70 : tensor<1x1x1x256xf32>
      %72 = stablehlo.reshape %arg274 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %73 = stablehlo.multiply %71, %72 : tensor<1x1x1x256xf32>
      %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %75 = stablehlo.multiply %68, %74 : tensor<256x56x56x256xf32>
      %76 = stablehlo.reshape %arg273 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %78 = stablehlo.add %75, %77 : tensor<256x56x56x256xf32>
      %79 = stablehlo.convert %78 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %80 = stablehlo.convert %arg278 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %81 = stablehlo.convolution(%23, %80) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %82 = stablehlo.broadcast_in_dim %arg436, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %83 = stablehlo.broadcast_in_dim %arg437, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %84 = stablehlo.convert %81 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %85 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %86 = stablehlo.subtract %84, %85 : tensor<256x56x56x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %87 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %88 = stablehlo.add %83, %87 : tensor<1x1x1x256xf32>
      %89 = stablehlo.rsqrt %88 : tensor<1x1x1x256xf32>
      %90 = stablehlo.reshape %arg280 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %91 = stablehlo.multiply %89, %90 : tensor<1x1x1x256xf32>
      %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %93 = stablehlo.multiply %86, %92 : tensor<256x56x56x256xf32>
      %94 = stablehlo.reshape %arg279 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %96 = stablehlo.add %93, %95 : tensor<256x56x56x256xf32>
      %97 = stablehlo.convert %96 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %98 = stablehlo.add %97, %79 : tensor<256x56x56x256xf16>
      %99 = func.call @relu_32(%98) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
      %100 = stablehlo.convert %arg287 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
      %101 = stablehlo.convolution(%99, %100) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<256x56x56x64xf16>
      %102 = stablehlo.broadcast_in_dim %arg438, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %103 = stablehlo.broadcast_in_dim %arg439, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %104 = stablehlo.convert %101 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %105 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %106 = stablehlo.subtract %104, %105 : tensor<256x56x56x64xf32>
      %cst_5 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %107 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %108 = stablehlo.add %103, %107 : tensor<1x1x1x64xf32>
      %109 = stablehlo.rsqrt %108 : tensor<1x1x1x64xf32>
      %110 = stablehlo.reshape %arg282 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %111 = stablehlo.multiply %109, %110 : tensor<1x1x1x64xf32>
      %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %113 = stablehlo.multiply %106, %112 : tensor<256x56x56x64xf32>
      %114 = stablehlo.reshape %arg281 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %115 = stablehlo.broadcast_in_dim %114, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %116 = stablehlo.add %113, %115 : tensor<256x56x56x64xf32>
      %117 = stablehlo.convert %116 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %118 = func.call @relu_13(%117) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %119 = stablehlo.convert %arg288 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %120 = stablehlo.convolution(%118, %119) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %121 = stablehlo.broadcast_in_dim %arg440, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %122 = stablehlo.broadcast_in_dim %arg441, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %123 = stablehlo.convert %120 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %124 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %125 = stablehlo.subtract %123, %124 : tensor<256x56x56x64xf32>
      %cst_6 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %126 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %127 = stablehlo.add %122, %126 : tensor<1x1x1x64xf32>
      %128 = stablehlo.rsqrt %127 : tensor<1x1x1x64xf32>
      %129 = stablehlo.reshape %arg284 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %130 = stablehlo.multiply %128, %129 : tensor<1x1x1x64xf32>
      %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %132 = stablehlo.multiply %125, %131 : tensor<256x56x56x64xf32>
      %133 = stablehlo.reshape %arg283 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %134 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %135 = stablehlo.add %132, %134 : tensor<256x56x56x64xf32>
      %136 = stablehlo.convert %135 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %137 = func.call @relu_13(%136) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %138 = stablehlo.convert %arg289 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %139 = stablehlo.convolution(%137, %138) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %140 = stablehlo.broadcast_in_dim %arg442, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %141 = stablehlo.broadcast_in_dim %arg443, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %142 = stablehlo.convert %139 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %143 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %144 = stablehlo.subtract %142, %143 : tensor<256x56x56x256xf32>
      %cst_7 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %145 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %146 = stablehlo.add %141, %145 : tensor<1x1x1x256xf32>
      %147 = stablehlo.rsqrt %146 : tensor<1x1x1x256xf32>
      %148 = stablehlo.reshape %arg286 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %149 = stablehlo.multiply %147, %148 : tensor<1x1x1x256xf32>
      %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %151 = stablehlo.multiply %144, %150 : tensor<256x56x56x256xf32>
      %152 = stablehlo.reshape %arg285 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %154 = stablehlo.add %151, %153 : tensor<256x56x56x256xf32>
      %155 = stablehlo.convert %154 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %156 = stablehlo.add %99, %155 : tensor<256x56x56x256xf16>
      %157 = func.call @relu_32(%156) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
      %158 = stablehlo.convert %arg353 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
      %159 = stablehlo.convolution(%157, %158) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<256x56x56x64xf16>
      %160 = stablehlo.broadcast_in_dim %arg482, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %161 = stablehlo.broadcast_in_dim %arg483, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %162 = stablehlo.convert %159 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %163 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %164 = stablehlo.subtract %162, %163 : tensor<256x56x56x64xf32>
      %cst_8 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %165 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %166 = stablehlo.add %161, %165 : tensor<1x1x1x64xf32>
      %167 = stablehlo.rsqrt %166 : tensor<1x1x1x64xf32>
      %168 = stablehlo.reshape %arg348 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %169 = stablehlo.multiply %167, %168 : tensor<1x1x1x64xf32>
      %170 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %171 = stablehlo.multiply %164, %170 : tensor<256x56x56x64xf32>
      %172 = stablehlo.reshape %arg347 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %173 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %174 = stablehlo.add %171, %173 : tensor<256x56x56x64xf32>
      %175 = stablehlo.convert %174 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %176 = func.call @relu_13(%175) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %177 = stablehlo.convert %arg354 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %178 = stablehlo.convolution(%176, %177) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %179 = stablehlo.broadcast_in_dim %arg484, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %180 = stablehlo.broadcast_in_dim %arg485, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %181 = stablehlo.convert %178 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %182 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %183 = stablehlo.subtract %181, %182 : tensor<256x56x56x64xf32>
      %cst_9 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %184 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %185 = stablehlo.add %180, %184 : tensor<1x1x1x64xf32>
      %186 = stablehlo.rsqrt %185 : tensor<1x1x1x64xf32>
      %187 = stablehlo.reshape %arg350 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %188 = stablehlo.multiply %186, %187 : tensor<1x1x1x64xf32>
      %189 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %190 = stablehlo.multiply %183, %189 : tensor<256x56x56x64xf32>
      %191 = stablehlo.reshape %arg349 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %193 = stablehlo.add %190, %192 : tensor<256x56x56x64xf32>
      %194 = stablehlo.convert %193 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %195 = func.call @relu_13(%194) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %196 = stablehlo.convert %arg355 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %197 = stablehlo.convolution(%195, %196) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %198 = stablehlo.broadcast_in_dim %arg486, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %199 = stablehlo.broadcast_in_dim %arg487, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %200 = stablehlo.convert %197 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %201 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %202 = stablehlo.subtract %200, %201 : tensor<256x56x56x256xf32>
      %cst_10 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %204 = stablehlo.add %199, %203 : tensor<1x1x1x256xf32>
      %205 = stablehlo.rsqrt %204 : tensor<1x1x1x256xf32>
      %206 = stablehlo.reshape %arg352 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %207 = stablehlo.multiply %205, %206 : tensor<1x1x1x256xf32>
      %208 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %209 = stablehlo.multiply %202, %208 : tensor<256x56x56x256xf32>
      %210 = stablehlo.reshape %arg351 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %212 = stablehlo.add %209, %211 : tensor<256x56x56x256xf32>
      %213 = stablehlo.convert %212 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %214 = stablehlo.add %157, %213 : tensor<256x56x56x256xf16>
      %215 = func.call @relu_32(%214) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
      %216 = stablehlo.convert %arg362 : (tensor<1x1x256x128xf32>) -> tensor<1x1x256x128xf16>
      %217 = stablehlo.convolution(%215, %216) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x128xf16>) -> tensor<256x56x56x128xf16>
      %218 = stablehlo.broadcast_in_dim %arg488, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %219 = stablehlo.broadcast_in_dim %arg489, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %220 = stablehlo.convert %217 : (tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf32>
      %221 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
      %222 = stablehlo.subtract %220, %221 : tensor<256x56x56x128xf32>
      %cst_11 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %223 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %224 = stablehlo.add %219, %223 : tensor<1x1x1x128xf32>
      %225 = stablehlo.rsqrt %224 : tensor<1x1x1x128xf32>
      %226 = stablehlo.reshape %arg357 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<1x1x1x128xf32>
      %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
      %229 = stablehlo.multiply %222, %228 : tensor<256x56x56x128xf32>
      %230 = stablehlo.reshape %arg356 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %231 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
      %232 = stablehlo.add %229, %231 : tensor<256x56x56x128xf32>
      %233 = stablehlo.convert %232 : (tensor<256x56x56x128xf32>) -> tensor<256x56x56x128xf16>
      %234 = func.call @relu_55(%233) : (tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf16>
      %235 = stablehlo.convert %arg363 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %236 = stablehlo.convolution(%234, %235) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %237 = stablehlo.broadcast_in_dim %arg490, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %238 = stablehlo.broadcast_in_dim %arg491, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %239 = stablehlo.convert %236 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %240 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %241 = stablehlo.subtract %239, %240 : tensor<256x28x28x128xf32>
      %cst_12 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %242 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %243 = stablehlo.add %238, %242 : tensor<1x1x1x128xf32>
      %244 = stablehlo.rsqrt %243 : tensor<1x1x1x128xf32>
      %245 = stablehlo.reshape %arg359 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %246 = stablehlo.multiply %244, %245 : tensor<1x1x1x128xf32>
      %247 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %248 = stablehlo.multiply %241, %247 : tensor<256x28x28x128xf32>
      %249 = stablehlo.reshape %arg358 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %251 = stablehlo.add %248, %250 : tensor<256x28x28x128xf32>
      %252 = stablehlo.convert %251 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %253 = func.call @relu_65(%252) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %254 = stablehlo.convert %arg364 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %255 = stablehlo.convolution(%253, %254) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %256 = stablehlo.broadcast_in_dim %arg492, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %257 = stablehlo.broadcast_in_dim %arg493, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %258 = stablehlo.convert %255 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %259 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %260 = stablehlo.subtract %258, %259 : tensor<256x28x28x512xf32>
      %cst_13 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %261 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %262 = stablehlo.add %257, %261 : tensor<1x1x1x512xf32>
      %263 = stablehlo.rsqrt %262 : tensor<1x1x1x512xf32>
      %264 = stablehlo.reshape %arg361 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<1x1x1x512xf32>
      %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %267 = stablehlo.multiply %260, %266 : tensor<256x28x28x512xf32>
      %268 = stablehlo.reshape %arg360 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %270 = stablehlo.add %267, %269 : tensor<256x28x28x512xf32>
      %271 = stablehlo.convert %270 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %272 = stablehlo.convert %arg365 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf16>
      %273 = stablehlo.convolution(%215, %272) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x512xf16>) -> tensor<256x28x28x512xf16>
      %274 = stablehlo.broadcast_in_dim %arg494, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %275 = stablehlo.broadcast_in_dim %arg495, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %276 = stablehlo.convert %273 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %277 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %278 = stablehlo.subtract %276, %277 : tensor<256x28x28x512xf32>
      %cst_14 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %280 = stablehlo.add %275, %279 : tensor<1x1x1x512xf32>
      %281 = stablehlo.rsqrt %280 : tensor<1x1x1x512xf32>
      %282 = stablehlo.reshape %arg367 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %283 = stablehlo.multiply %281, %282 : tensor<1x1x1x512xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %285 = stablehlo.multiply %278, %284 : tensor<256x28x28x512xf32>
      %286 = stablehlo.reshape %arg366 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %288 = stablehlo.add %285, %287 : tensor<256x28x28x512xf32>
      %289 = stablehlo.convert %288 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %290 = stablehlo.add %289, %271 : tensor<256x28x28x512xf16>
      %291 = func.call @relu_83(%290) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %292 = stablehlo.convert %arg374 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %293 = stablehlo.convolution(%291, %292) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %294 = stablehlo.broadcast_in_dim %arg496, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg497, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %296 = stablehlo.convert %293 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %297 = stablehlo.broadcast_in_dim %294, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %298 = stablehlo.subtract %296, %297 : tensor<256x28x28x128xf32>
      %cst_15 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %300 = stablehlo.add %295, %299 : tensor<1x1x1x128xf32>
      %301 = stablehlo.rsqrt %300 : tensor<1x1x1x128xf32>
      %302 = stablehlo.reshape %arg369 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<1x1x1x128xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %305 = stablehlo.multiply %298, %304 : tensor<256x28x28x128xf32>
      %306 = stablehlo.reshape %arg368 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %308 = stablehlo.add %305, %307 : tensor<256x28x28x128xf32>
      %309 = stablehlo.convert %308 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %310 = func.call @relu_65(%309) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %311 = stablehlo.convert %arg375 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %312 = stablehlo.convolution(%310, %311) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %313 = stablehlo.broadcast_in_dim %arg498, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %314 = stablehlo.broadcast_in_dim %arg499, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %315 = stablehlo.convert %312 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %316 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %317 = stablehlo.subtract %315, %316 : tensor<256x28x28x128xf32>
      %cst_16 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %319 = stablehlo.add %314, %318 : tensor<1x1x1x128xf32>
      %320 = stablehlo.rsqrt %319 : tensor<1x1x1x128xf32>
      %321 = stablehlo.reshape %arg371 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<1x1x1x128xf32>
      %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %324 = stablehlo.multiply %317, %323 : tensor<256x28x28x128xf32>
      %325 = stablehlo.reshape %arg370 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %327 = stablehlo.add %324, %326 : tensor<256x28x28x128xf32>
      %328 = stablehlo.convert %327 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %329 = func.call @relu_65(%328) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %330 = stablehlo.convert %arg376 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %331 = stablehlo.convolution(%329, %330) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %332 = stablehlo.broadcast_in_dim %arg500, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %333 = stablehlo.broadcast_in_dim %arg501, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %334 = stablehlo.convert %331 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %335 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %336 = stablehlo.subtract %334, %335 : tensor<256x28x28x512xf32>
      %cst_17 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %338 = stablehlo.add %333, %337 : tensor<1x1x1x512xf32>
      %339 = stablehlo.rsqrt %338 : tensor<1x1x1x512xf32>
      %340 = stablehlo.reshape %arg373 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<1x1x1x512xf32>
      %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %343 = stablehlo.multiply %336, %342 : tensor<256x28x28x512xf32>
      %344 = stablehlo.reshape %arg372 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %346 = stablehlo.add %343, %345 : tensor<256x28x28x512xf32>
      %347 = stablehlo.convert %346 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %348 = stablehlo.add %291, %347 : tensor<256x28x28x512xf16>
      %349 = func.call @relu_83(%348) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %350 = stablehlo.convert %arg383 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %351 = stablehlo.convolution(%349, %350) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %352 = stablehlo.broadcast_in_dim %arg502, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %353 = stablehlo.broadcast_in_dim %arg503, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %354 = stablehlo.convert %351 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %355 = stablehlo.broadcast_in_dim %352, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %356 = stablehlo.subtract %354, %355 : tensor<256x28x28x128xf32>
      %cst_18 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %358 = stablehlo.add %353, %357 : tensor<1x1x1x128xf32>
      %359 = stablehlo.rsqrt %358 : tensor<1x1x1x128xf32>
      %360 = stablehlo.reshape %arg378 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %361 = stablehlo.multiply %359, %360 : tensor<1x1x1x128xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %363 = stablehlo.multiply %356, %362 : tensor<256x28x28x128xf32>
      %364 = stablehlo.reshape %arg377 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %366 = stablehlo.add %363, %365 : tensor<256x28x28x128xf32>
      %367 = stablehlo.convert %366 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %368 = func.call @relu_65(%367) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %369 = stablehlo.convert %arg384 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %370 = stablehlo.convolution(%368, %369) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %371 = stablehlo.broadcast_in_dim %arg504, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %372 = stablehlo.broadcast_in_dim %arg505, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %373 = stablehlo.convert %370 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %374 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %375 = stablehlo.subtract %373, %374 : tensor<256x28x28x128xf32>
      %cst_19 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %377 = stablehlo.add %372, %376 : tensor<1x1x1x128xf32>
      %378 = stablehlo.rsqrt %377 : tensor<1x1x1x128xf32>
      %379 = stablehlo.reshape %arg380 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %380 = stablehlo.multiply %378, %379 : tensor<1x1x1x128xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %382 = stablehlo.multiply %375, %381 : tensor<256x28x28x128xf32>
      %383 = stablehlo.reshape %arg379 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %385 = stablehlo.add %382, %384 : tensor<256x28x28x128xf32>
      %386 = stablehlo.convert %385 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %387 = func.call @relu_65(%386) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %388 = stablehlo.convert %arg385 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %389 = stablehlo.convolution(%387, %388) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %390 = stablehlo.broadcast_in_dim %arg506, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %391 = stablehlo.broadcast_in_dim %arg507, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %392 = stablehlo.convert %389 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %393 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %394 = stablehlo.subtract %392, %393 : tensor<256x28x28x512xf32>
      %cst_20 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %396 = stablehlo.add %391, %395 : tensor<1x1x1x512xf32>
      %397 = stablehlo.rsqrt %396 : tensor<1x1x1x512xf32>
      %398 = stablehlo.reshape %arg382 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %399 = stablehlo.multiply %397, %398 : tensor<1x1x1x512xf32>
      %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %401 = stablehlo.multiply %394, %400 : tensor<256x28x28x512xf32>
      %402 = stablehlo.reshape %arg381 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %403 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %404 = stablehlo.add %401, %403 : tensor<256x28x28x512xf32>
      %405 = stablehlo.convert %404 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %406 = stablehlo.add %349, %405 : tensor<256x28x28x512xf16>
      %407 = func.call @relu_83(%406) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %408 = stablehlo.convert %arg392 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %409 = stablehlo.convolution(%407, %408) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %410 = stablehlo.broadcast_in_dim %arg508, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %411 = stablehlo.broadcast_in_dim %arg509, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %412 = stablehlo.convert %409 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %413 = stablehlo.broadcast_in_dim %410, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %414 = stablehlo.subtract %412, %413 : tensor<256x28x28x128xf32>
      %cst_21 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %416 = stablehlo.add %411, %415 : tensor<1x1x1x128xf32>
      %417 = stablehlo.rsqrt %416 : tensor<1x1x1x128xf32>
      %418 = stablehlo.reshape %arg387 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x1x1x128xf32>
      %420 = stablehlo.broadcast_in_dim %419, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %421 = stablehlo.multiply %414, %420 : tensor<256x28x28x128xf32>
      %422 = stablehlo.reshape %arg386 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %424 = stablehlo.add %421, %423 : tensor<256x28x28x128xf32>
      %425 = stablehlo.convert %424 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %426 = func.call @relu_65(%425) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %427 = stablehlo.convert %arg393 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %428 = stablehlo.convolution(%426, %427) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %429 = stablehlo.broadcast_in_dim %arg510, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %430 = stablehlo.broadcast_in_dim %arg511, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %431 = stablehlo.convert %428 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %432 = stablehlo.broadcast_in_dim %429, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %433 = stablehlo.subtract %431, %432 : tensor<256x28x28x128xf32>
      %cst_22 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %435 = stablehlo.add %430, %434 : tensor<1x1x1x128xf32>
      %436 = stablehlo.rsqrt %435 : tensor<1x1x1x128xf32>
      %437 = stablehlo.reshape %arg389 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<1x1x1x128xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %440 = stablehlo.multiply %433, %439 : tensor<256x28x28x128xf32>
      %441 = stablehlo.reshape %arg388 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %443 = stablehlo.add %440, %442 : tensor<256x28x28x128xf32>
      %444 = stablehlo.convert %443 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %445 = func.call @relu_65(%444) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %446 = stablehlo.convert %arg394 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %447 = stablehlo.convolution(%445, %446) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %448 = stablehlo.broadcast_in_dim %arg512, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %449 = stablehlo.broadcast_in_dim %arg513, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %450 = stablehlo.convert %447 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %451 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %452 = stablehlo.subtract %450, %451 : tensor<256x28x28x512xf32>
      %cst_23 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %454 = stablehlo.add %449, %453 : tensor<1x1x1x512xf32>
      %455 = stablehlo.rsqrt %454 : tensor<1x1x1x512xf32>
      %456 = stablehlo.reshape %arg391 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<1x1x1x512xf32>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %459 = stablehlo.multiply %452, %458 : tensor<256x28x28x512xf32>
      %460 = stablehlo.reshape %arg390 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %462 = stablehlo.add %459, %461 : tensor<256x28x28x512xf32>
      %463 = stablehlo.convert %462 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %464 = stablehlo.add %407, %463 : tensor<256x28x28x512xf16>
      %465 = func.call @relu_83(%464) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %466 = stablehlo.convert %arg401 : (tensor<1x1x512x256xf32>) -> tensor<1x1x512x256xf16>
      %467 = stablehlo.convolution(%465, %466) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x256xf16>) -> tensor<256x28x28x256xf16>
      %468 = stablehlo.broadcast_in_dim %arg514, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %469 = stablehlo.broadcast_in_dim %arg515, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %470 = stablehlo.convert %467 : (tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf32>
      %471 = stablehlo.broadcast_in_dim %468, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
      %472 = stablehlo.subtract %470, %471 : tensor<256x28x28x256xf32>
      %cst_24 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %474 = stablehlo.add %469, %473 : tensor<1x1x1x256xf32>
      %475 = stablehlo.rsqrt %474 : tensor<1x1x1x256xf32>
      %476 = stablehlo.reshape %arg396 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %477 = stablehlo.multiply %475, %476 : tensor<1x1x1x256xf32>
      %478 = stablehlo.broadcast_in_dim %477, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
      %479 = stablehlo.multiply %472, %478 : tensor<256x28x28x256xf32>
      %480 = stablehlo.reshape %arg395 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
      %482 = stablehlo.add %479, %481 : tensor<256x28x28x256xf32>
      %483 = stablehlo.convert %482 : (tensor<256x28x28x256xf32>) -> tensor<256x28x28x256xf16>
      %484 = func.call @relu_105(%483) : (tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf16>
      %485 = stablehlo.convert %arg402 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %486 = stablehlo.convolution(%484, %485) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %487 = stablehlo.broadcast_in_dim %arg516, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %488 = stablehlo.broadcast_in_dim %arg517, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %489 = stablehlo.convert %486 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %490 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %491 = stablehlo.subtract %489, %490 : tensor<256x14x14x256xf32>
      %cst_25 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %493 = stablehlo.add %488, %492 : tensor<1x1x1x256xf32>
      %494 = stablehlo.rsqrt %493 : tensor<1x1x1x256xf32>
      %495 = stablehlo.reshape %arg398 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %496 = stablehlo.multiply %494, %495 : tensor<1x1x1x256xf32>
      %497 = stablehlo.broadcast_in_dim %496, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %498 = stablehlo.multiply %491, %497 : tensor<256x14x14x256xf32>
      %499 = stablehlo.reshape %arg397 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %500 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %501 = stablehlo.add %498, %500 : tensor<256x14x14x256xf32>
      %502 = stablehlo.convert %501 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %503 = func.call @relu_115(%502) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %504 = stablehlo.convert %arg403 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %505 = stablehlo.convolution(%503, %504) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %506 = stablehlo.broadcast_in_dim %arg518, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %507 = stablehlo.broadcast_in_dim %arg519, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %508 = stablehlo.convert %505 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %509 = stablehlo.broadcast_in_dim %506, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %510 = stablehlo.subtract %508, %509 : tensor<256x14x14x1024xf32>
      %cst_26 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %512 = stablehlo.add %507, %511 : tensor<1x1x1x1024xf32>
      %513 = stablehlo.rsqrt %512 : tensor<1x1x1x1024xf32>
      %514 = stablehlo.reshape %arg400 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %515 = stablehlo.multiply %513, %514 : tensor<1x1x1x1024xf32>
      %516 = stablehlo.broadcast_in_dim %515, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %517 = stablehlo.multiply %510, %516 : tensor<256x14x14x1024xf32>
      %518 = stablehlo.reshape %arg399 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %519 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %520 = stablehlo.add %517, %519 : tensor<256x14x14x1024xf32>
      %521 = stablehlo.convert %520 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %522 = stablehlo.convert %arg404 : (tensor<1x1x512x1024xf32>) -> tensor<1x1x512x1024xf16>
      %523 = stablehlo.convolution(%465, %522) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x1024xf16>) -> tensor<256x14x14x1024xf16>
      %524 = stablehlo.broadcast_in_dim %arg520, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %525 = stablehlo.broadcast_in_dim %arg521, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %526 = stablehlo.convert %523 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %527 = stablehlo.broadcast_in_dim %524, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %528 = stablehlo.subtract %526, %527 : tensor<256x14x14x1024xf32>
      %cst_27 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %530 = stablehlo.add %525, %529 : tensor<1x1x1x1024xf32>
      %531 = stablehlo.rsqrt %530 : tensor<1x1x1x1024xf32>
      %532 = stablehlo.reshape %arg406 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %533 = stablehlo.multiply %531, %532 : tensor<1x1x1x1024xf32>
      %534 = stablehlo.broadcast_in_dim %533, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %535 = stablehlo.multiply %528, %534 : tensor<256x14x14x1024xf32>
      %536 = stablehlo.reshape %arg405 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %537 = stablehlo.broadcast_in_dim %536, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %538 = stablehlo.add %535, %537 : tensor<256x14x14x1024xf32>
      %539 = stablehlo.convert %538 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %540 = stablehlo.add %539, %521 : tensor<256x14x14x1024xf16>
      %541 = func.call @relu_133(%540) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %542 = stablehlo.convert %arg413 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %543 = stablehlo.convolution(%541, %542) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %544 = stablehlo.broadcast_in_dim %arg522, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %545 = stablehlo.broadcast_in_dim %arg523, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %546 = stablehlo.convert %543 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %547 = stablehlo.broadcast_in_dim %544, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %548 = stablehlo.subtract %546, %547 : tensor<256x14x14x256xf32>
      %cst_28 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %549 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %550 = stablehlo.add %545, %549 : tensor<1x1x1x256xf32>
      %551 = stablehlo.rsqrt %550 : tensor<1x1x1x256xf32>
      %552 = stablehlo.reshape %arg408 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %553 = stablehlo.multiply %551, %552 : tensor<1x1x1x256xf32>
      %554 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %555 = stablehlo.multiply %548, %554 : tensor<256x14x14x256xf32>
      %556 = stablehlo.reshape %arg407 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %557 = stablehlo.broadcast_in_dim %556, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %558 = stablehlo.add %555, %557 : tensor<256x14x14x256xf32>
      %559 = stablehlo.convert %558 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %560 = func.call @relu_115(%559) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %561 = stablehlo.convert %arg414 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %562 = stablehlo.convolution(%560, %561) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %563 = stablehlo.broadcast_in_dim %arg524, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %564 = stablehlo.broadcast_in_dim %arg525, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %565 = stablehlo.convert %562 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %566 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %567 = stablehlo.subtract %565, %566 : tensor<256x14x14x256xf32>
      %cst_29 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %568 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %569 = stablehlo.add %564, %568 : tensor<1x1x1x256xf32>
      %570 = stablehlo.rsqrt %569 : tensor<1x1x1x256xf32>
      %571 = stablehlo.reshape %arg410 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %572 = stablehlo.multiply %570, %571 : tensor<1x1x1x256xf32>
      %573 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %574 = stablehlo.multiply %567, %573 : tensor<256x14x14x256xf32>
      %575 = stablehlo.reshape %arg409 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %577 = stablehlo.add %574, %576 : tensor<256x14x14x256xf32>
      %578 = stablehlo.convert %577 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %579 = func.call @relu_115(%578) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %580 = stablehlo.convert %arg415 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %581 = stablehlo.convolution(%579, %580) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %582 = stablehlo.broadcast_in_dim %arg526, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %583 = stablehlo.broadcast_in_dim %arg527, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %584 = stablehlo.convert %581 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %585 = stablehlo.broadcast_in_dim %582, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %586 = stablehlo.subtract %584, %585 : tensor<256x14x14x1024xf32>
      %cst_30 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %587 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %588 = stablehlo.add %583, %587 : tensor<1x1x1x1024xf32>
      %589 = stablehlo.rsqrt %588 : tensor<1x1x1x1024xf32>
      %590 = stablehlo.reshape %arg412 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %591 = stablehlo.multiply %589, %590 : tensor<1x1x1x1024xf32>
      %592 = stablehlo.broadcast_in_dim %591, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %593 = stablehlo.multiply %586, %592 : tensor<256x14x14x1024xf32>
      %594 = stablehlo.reshape %arg411 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %595 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %596 = stablehlo.add %593, %595 : tensor<256x14x14x1024xf32>
      %597 = stablehlo.convert %596 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %598 = stablehlo.add %541, %597 : tensor<256x14x14x1024xf16>
      %599 = func.call @relu_133(%598) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %600 = stablehlo.convert %arg422 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %601 = stablehlo.convolution(%599, %600) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %602 = stablehlo.broadcast_in_dim %arg528, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %603 = stablehlo.broadcast_in_dim %arg529, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %604 = stablehlo.convert %601 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %605 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %606 = stablehlo.subtract %604, %605 : tensor<256x14x14x256xf32>
      %cst_31 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %607 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %608 = stablehlo.add %603, %607 : tensor<1x1x1x256xf32>
      %609 = stablehlo.rsqrt %608 : tensor<1x1x1x256xf32>
      %610 = stablehlo.reshape %arg417 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %611 = stablehlo.multiply %609, %610 : tensor<1x1x1x256xf32>
      %612 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %613 = stablehlo.multiply %606, %612 : tensor<256x14x14x256xf32>
      %614 = stablehlo.reshape %arg416 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %615 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %616 = stablehlo.add %613, %615 : tensor<256x14x14x256xf32>
      %617 = stablehlo.convert %616 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %618 = func.call @relu_115(%617) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %619 = stablehlo.convert %arg423 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %620 = stablehlo.convolution(%618, %619) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %621 = stablehlo.broadcast_in_dim %arg530, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %622 = stablehlo.broadcast_in_dim %arg531, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %623 = stablehlo.convert %620 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %624 = stablehlo.broadcast_in_dim %621, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %625 = stablehlo.subtract %623, %624 : tensor<256x14x14x256xf32>
      %cst_32 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %626 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %627 = stablehlo.add %622, %626 : tensor<1x1x1x256xf32>
      %628 = stablehlo.rsqrt %627 : tensor<1x1x1x256xf32>
      %629 = stablehlo.reshape %arg419 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %630 = stablehlo.multiply %628, %629 : tensor<1x1x1x256xf32>
      %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %632 = stablehlo.multiply %625, %631 : tensor<256x14x14x256xf32>
      %633 = stablehlo.reshape %arg418 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %634 = stablehlo.broadcast_in_dim %633, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %635 = stablehlo.add %632, %634 : tensor<256x14x14x256xf32>
      %636 = stablehlo.convert %635 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %637 = func.call @relu_115(%636) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %638 = stablehlo.convert %arg424 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %639 = stablehlo.convolution(%637, %638) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %640 = stablehlo.broadcast_in_dim %arg532, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %641 = stablehlo.broadcast_in_dim %arg533, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %642 = stablehlo.convert %639 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %643 = stablehlo.broadcast_in_dim %640, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %644 = stablehlo.subtract %642, %643 : tensor<256x14x14x1024xf32>
      %cst_33 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %646 = stablehlo.add %641, %645 : tensor<1x1x1x1024xf32>
      %647 = stablehlo.rsqrt %646 : tensor<1x1x1x1024xf32>
      %648 = stablehlo.reshape %arg421 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<1x1x1x1024xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %651 = stablehlo.multiply %644, %650 : tensor<256x14x14x1024xf32>
      %652 = stablehlo.reshape %arg420 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %654 = stablehlo.add %651, %653 : tensor<256x14x14x1024xf32>
      %655 = stablehlo.convert %654 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %656 = stablehlo.add %599, %655 : tensor<256x14x14x1024xf16>
      %657 = func.call @relu_133(%656) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %658 = stablehlo.convert %arg296 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %659 = stablehlo.convolution(%657, %658) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %660 = stablehlo.broadcast_in_dim %arg444, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %661 = stablehlo.broadcast_in_dim %arg445, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %662 = stablehlo.convert %659 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %663 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %664 = stablehlo.subtract %662, %663 : tensor<256x14x14x256xf32>
      %cst_34 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %666 = stablehlo.add %661, %665 : tensor<1x1x1x256xf32>
      %667 = stablehlo.rsqrt %666 : tensor<1x1x1x256xf32>
      %668 = stablehlo.reshape %arg291 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<1x1x1x256xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %671 = stablehlo.multiply %664, %670 : tensor<256x14x14x256xf32>
      %672 = stablehlo.reshape %arg290 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %674 = stablehlo.add %671, %673 : tensor<256x14x14x256xf32>
      %675 = stablehlo.convert %674 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %676 = func.call @relu_115(%675) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %677 = stablehlo.convert %arg297 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %678 = stablehlo.convolution(%676, %677) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %679 = stablehlo.broadcast_in_dim %arg446, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %680 = stablehlo.broadcast_in_dim %arg447, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %681 = stablehlo.convert %678 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %682 = stablehlo.broadcast_in_dim %679, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %683 = stablehlo.subtract %681, %682 : tensor<256x14x14x256xf32>
      %cst_35 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %684 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %685 = stablehlo.add %680, %684 : tensor<1x1x1x256xf32>
      %686 = stablehlo.rsqrt %685 : tensor<1x1x1x256xf32>
      %687 = stablehlo.reshape %arg293 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %688 = stablehlo.multiply %686, %687 : tensor<1x1x1x256xf32>
      %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %690 = stablehlo.multiply %683, %689 : tensor<256x14x14x256xf32>
      %691 = stablehlo.reshape %arg292 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %692 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %693 = stablehlo.add %690, %692 : tensor<256x14x14x256xf32>
      %694 = stablehlo.convert %693 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %695 = func.call @relu_115(%694) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %696 = stablehlo.convert %arg298 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %697 = stablehlo.convolution(%695, %696) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %698 = stablehlo.broadcast_in_dim %arg448, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %699 = stablehlo.broadcast_in_dim %arg449, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %700 = stablehlo.convert %697 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %701 = stablehlo.broadcast_in_dim %698, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %702 = stablehlo.subtract %700, %701 : tensor<256x14x14x1024xf32>
      %cst_36 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %703 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %704 = stablehlo.add %699, %703 : tensor<1x1x1x1024xf32>
      %705 = stablehlo.rsqrt %704 : tensor<1x1x1x1024xf32>
      %706 = stablehlo.reshape %arg295 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %707 = stablehlo.multiply %705, %706 : tensor<1x1x1x1024xf32>
      %708 = stablehlo.broadcast_in_dim %707, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %709 = stablehlo.multiply %702, %708 : tensor<256x14x14x1024xf32>
      %710 = stablehlo.reshape %arg294 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %711 = stablehlo.broadcast_in_dim %710, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %712 = stablehlo.add %709, %711 : tensor<256x14x14x1024xf32>
      %713 = stablehlo.convert %712 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %714 = stablehlo.add %657, %713 : tensor<256x14x14x1024xf16>
      %715 = func.call @relu_133(%714) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %716 = stablehlo.convert %arg305 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %717 = stablehlo.convolution(%715, %716) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %718 = stablehlo.broadcast_in_dim %arg450, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %719 = stablehlo.broadcast_in_dim %arg451, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %720 = stablehlo.convert %717 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %721 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %722 = stablehlo.subtract %720, %721 : tensor<256x14x14x256xf32>
      %cst_37 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %723 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %724 = stablehlo.add %719, %723 : tensor<1x1x1x256xf32>
      %725 = stablehlo.rsqrt %724 : tensor<1x1x1x256xf32>
      %726 = stablehlo.reshape %arg300 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %727 = stablehlo.multiply %725, %726 : tensor<1x1x1x256xf32>
      %728 = stablehlo.broadcast_in_dim %727, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %729 = stablehlo.multiply %722, %728 : tensor<256x14x14x256xf32>
      %730 = stablehlo.reshape %arg299 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %731 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %732 = stablehlo.add %729, %731 : tensor<256x14x14x256xf32>
      %733 = stablehlo.convert %732 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %734 = func.call @relu_115(%733) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %735 = stablehlo.convert %arg306 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %736 = stablehlo.convolution(%734, %735) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %737 = stablehlo.broadcast_in_dim %arg452, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %738 = stablehlo.broadcast_in_dim %arg453, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %739 = stablehlo.convert %736 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %740 = stablehlo.broadcast_in_dim %737, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %741 = stablehlo.subtract %739, %740 : tensor<256x14x14x256xf32>
      %cst_38 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %742 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %743 = stablehlo.add %738, %742 : tensor<1x1x1x256xf32>
      %744 = stablehlo.rsqrt %743 : tensor<1x1x1x256xf32>
      %745 = stablehlo.reshape %arg302 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %746 = stablehlo.multiply %744, %745 : tensor<1x1x1x256xf32>
      %747 = stablehlo.broadcast_in_dim %746, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %748 = stablehlo.multiply %741, %747 : tensor<256x14x14x256xf32>
      %749 = stablehlo.reshape %arg301 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %750 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %751 = stablehlo.add %748, %750 : tensor<256x14x14x256xf32>
      %752 = stablehlo.convert %751 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %753 = func.call @relu_115(%752) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %754 = stablehlo.convert %arg307 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %755 = stablehlo.convolution(%753, %754) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %756 = stablehlo.broadcast_in_dim %arg454, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %757 = stablehlo.broadcast_in_dim %arg455, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %758 = stablehlo.convert %755 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %759 = stablehlo.broadcast_in_dim %756, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %760 = stablehlo.subtract %758, %759 : tensor<256x14x14x1024xf32>
      %cst_39 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %761 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %762 = stablehlo.add %757, %761 : tensor<1x1x1x1024xf32>
      %763 = stablehlo.rsqrt %762 : tensor<1x1x1x1024xf32>
      %764 = stablehlo.reshape %arg304 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %765 = stablehlo.multiply %763, %764 : tensor<1x1x1x1024xf32>
      %766 = stablehlo.broadcast_in_dim %765, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %767 = stablehlo.multiply %760, %766 : tensor<256x14x14x1024xf32>
      %768 = stablehlo.reshape %arg303 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %769 = stablehlo.broadcast_in_dim %768, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %770 = stablehlo.add %767, %769 : tensor<256x14x14x1024xf32>
      %771 = stablehlo.convert %770 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %772 = stablehlo.add %715, %771 : tensor<256x14x14x1024xf16>
      %773 = func.call @relu_133(%772) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %774 = stablehlo.convert %arg314 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %775 = stablehlo.convolution(%773, %774) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %776 = stablehlo.broadcast_in_dim %arg456, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %777 = stablehlo.broadcast_in_dim %arg457, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %778 = stablehlo.convert %775 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %779 = stablehlo.broadcast_in_dim %776, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %780 = stablehlo.subtract %778, %779 : tensor<256x14x14x256xf32>
      %cst_40 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %781 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %782 = stablehlo.add %777, %781 : tensor<1x1x1x256xf32>
      %783 = stablehlo.rsqrt %782 : tensor<1x1x1x256xf32>
      %784 = stablehlo.reshape %arg309 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %785 = stablehlo.multiply %783, %784 : tensor<1x1x1x256xf32>
      %786 = stablehlo.broadcast_in_dim %785, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %787 = stablehlo.multiply %780, %786 : tensor<256x14x14x256xf32>
      %788 = stablehlo.reshape %arg308 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %789 = stablehlo.broadcast_in_dim %788, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %790 = stablehlo.add %787, %789 : tensor<256x14x14x256xf32>
      %791 = stablehlo.convert %790 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %792 = func.call @relu_115(%791) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %793 = stablehlo.convert %arg315 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %794 = stablehlo.convolution(%792, %793) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %795 = stablehlo.broadcast_in_dim %arg458, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %796 = stablehlo.broadcast_in_dim %arg459, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %797 = stablehlo.convert %794 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %798 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %799 = stablehlo.subtract %797, %798 : tensor<256x14x14x256xf32>
      %cst_41 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %800 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %801 = stablehlo.add %796, %800 : tensor<1x1x1x256xf32>
      %802 = stablehlo.rsqrt %801 : tensor<1x1x1x256xf32>
      %803 = stablehlo.reshape %arg311 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %804 = stablehlo.multiply %802, %803 : tensor<1x1x1x256xf32>
      %805 = stablehlo.broadcast_in_dim %804, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %806 = stablehlo.multiply %799, %805 : tensor<256x14x14x256xf32>
      %807 = stablehlo.reshape %arg310 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %808 = stablehlo.broadcast_in_dim %807, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %809 = stablehlo.add %806, %808 : tensor<256x14x14x256xf32>
      %810 = stablehlo.convert %809 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %811 = func.call @relu_115(%810) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %812 = stablehlo.convert %arg316 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %813 = stablehlo.convolution(%811, %812) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %814 = stablehlo.broadcast_in_dim %arg460, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %815 = stablehlo.broadcast_in_dim %arg461, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %816 = stablehlo.convert %813 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %817 = stablehlo.broadcast_in_dim %814, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %818 = stablehlo.subtract %816, %817 : tensor<256x14x14x1024xf32>
      %cst_42 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %819 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %820 = stablehlo.add %815, %819 : tensor<1x1x1x1024xf32>
      %821 = stablehlo.rsqrt %820 : tensor<1x1x1x1024xf32>
      %822 = stablehlo.reshape %arg313 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %823 = stablehlo.multiply %821, %822 : tensor<1x1x1x1024xf32>
      %824 = stablehlo.broadcast_in_dim %823, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %825 = stablehlo.multiply %818, %824 : tensor<256x14x14x1024xf32>
      %826 = stablehlo.reshape %arg312 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %827 = stablehlo.broadcast_in_dim %826, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %828 = stablehlo.add %825, %827 : tensor<256x14x14x1024xf32>
      %829 = stablehlo.convert %828 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %830 = stablehlo.add %773, %829 : tensor<256x14x14x1024xf16>
      %831 = func.call @relu_133(%830) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %832 = stablehlo.convert %arg323 : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf16>
      %833 = stablehlo.convolution(%831, %832) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x512xf16>) -> tensor<256x14x14x512xf16>
      %834 = stablehlo.broadcast_in_dim %arg462, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %835 = stablehlo.broadcast_in_dim %arg463, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %836 = stablehlo.convert %833 : (tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf32>
      %837 = stablehlo.broadcast_in_dim %834, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
      %838 = stablehlo.subtract %836, %837 : tensor<256x14x14x512xf32>
      %cst_43 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %839 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %840 = stablehlo.add %835, %839 : tensor<1x1x1x512xf32>
      %841 = stablehlo.rsqrt %840 : tensor<1x1x1x512xf32>
      %842 = stablehlo.reshape %arg318 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %843 = stablehlo.multiply %841, %842 : tensor<1x1x1x512xf32>
      %844 = stablehlo.broadcast_in_dim %843, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
      %845 = stablehlo.multiply %838, %844 : tensor<256x14x14x512xf32>
      %846 = stablehlo.reshape %arg317 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %847 = stablehlo.broadcast_in_dim %846, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
      %848 = stablehlo.add %845, %847 : tensor<256x14x14x512xf32>
      %849 = stablehlo.convert %848 : (tensor<256x14x14x512xf32>) -> tensor<256x14x14x512xf16>
      %850 = func.call @relu_161(%849) : (tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf16>
      %851 = stablehlo.convert %arg324 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %852 = stablehlo.convolution(%850, %851) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %853 = stablehlo.broadcast_in_dim %arg464, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %854 = stablehlo.broadcast_in_dim %arg465, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %855 = stablehlo.convert %852 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %856 = stablehlo.broadcast_in_dim %853, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %857 = stablehlo.subtract %855, %856 : tensor<256x7x7x512xf32>
      %cst_44 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %858 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %859 = stablehlo.add %854, %858 : tensor<1x1x1x512xf32>
      %860 = stablehlo.rsqrt %859 : tensor<1x1x1x512xf32>
      %861 = stablehlo.reshape %arg320 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %862 = stablehlo.multiply %860, %861 : tensor<1x1x1x512xf32>
      %863 = stablehlo.broadcast_in_dim %862, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %864 = stablehlo.multiply %857, %863 : tensor<256x7x7x512xf32>
      %865 = stablehlo.reshape %arg319 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %866 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %867 = stablehlo.add %864, %866 : tensor<256x7x7x512xf32>
      %868 = stablehlo.convert %867 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %869 = func.call @relu_171(%868) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %870 = stablehlo.convert %arg325 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
      %871 = stablehlo.convolution(%869, %870) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
      %872 = stablehlo.broadcast_in_dim %arg466, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %873 = stablehlo.broadcast_in_dim %arg467, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %874 = stablehlo.convert %871 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %875 = stablehlo.broadcast_in_dim %872, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %876 = stablehlo.subtract %874, %875 : tensor<256x7x7x2048xf32>
      %cst_45 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %877 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %878 = stablehlo.add %873, %877 : tensor<1x1x1x2048xf32>
      %879 = stablehlo.rsqrt %878 : tensor<1x1x1x2048xf32>
      %880 = stablehlo.reshape %arg322 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %881 = stablehlo.multiply %879, %880 : tensor<1x1x1x2048xf32>
      %882 = stablehlo.broadcast_in_dim %881, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %883 = stablehlo.multiply %876, %882 : tensor<256x7x7x2048xf32>
      %884 = stablehlo.reshape %arg321 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %885 = stablehlo.broadcast_in_dim %884, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %886 = stablehlo.add %883, %885 : tensor<256x7x7x2048xf32>
      %887 = stablehlo.convert %886 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %888 = stablehlo.convert %arg326 : (tensor<1x1x1024x2048xf32>) -> tensor<1x1x1024x2048xf16>
      %889 = stablehlo.convolution(%831, %888) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x2048xf16>) -> tensor<256x7x7x2048xf16>
      %890 = stablehlo.broadcast_in_dim %arg468, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %891 = stablehlo.broadcast_in_dim %arg469, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %892 = stablehlo.convert %889 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %893 = stablehlo.broadcast_in_dim %890, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %894 = stablehlo.subtract %892, %893 : tensor<256x7x7x2048xf32>
      %cst_46 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %895 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %896 = stablehlo.add %891, %895 : tensor<1x1x1x2048xf32>
      %897 = stablehlo.rsqrt %896 : tensor<1x1x1x2048xf32>
      %898 = stablehlo.reshape %arg328 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %899 = stablehlo.multiply %897, %898 : tensor<1x1x1x2048xf32>
      %900 = stablehlo.broadcast_in_dim %899, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %901 = stablehlo.multiply %894, %900 : tensor<256x7x7x2048xf32>
      %902 = stablehlo.reshape %arg327 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %903 = stablehlo.broadcast_in_dim %902, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %904 = stablehlo.add %901, %903 : tensor<256x7x7x2048xf32>
      %905 = stablehlo.convert %904 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %906 = stablehlo.add %905, %887 : tensor<256x7x7x2048xf16>
      %907 = func.call @relu_189(%906) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
      %908 = stablehlo.convert %arg335 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
      %909 = stablehlo.convolution(%907, %908) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<256x7x7x512xf16>
      %910 = stablehlo.broadcast_in_dim %arg470, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %911 = stablehlo.broadcast_in_dim %arg471, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %912 = stablehlo.convert %909 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %913 = stablehlo.broadcast_in_dim %910, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %914 = stablehlo.subtract %912, %913 : tensor<256x7x7x512xf32>
      %cst_47 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %915 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %916 = stablehlo.add %911, %915 : tensor<1x1x1x512xf32>
      %917 = stablehlo.rsqrt %916 : tensor<1x1x1x512xf32>
      %918 = stablehlo.reshape %arg330 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %919 = stablehlo.multiply %917, %918 : tensor<1x1x1x512xf32>
      %920 = stablehlo.broadcast_in_dim %919, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %921 = stablehlo.multiply %914, %920 : tensor<256x7x7x512xf32>
      %922 = stablehlo.reshape %arg329 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %923 = stablehlo.broadcast_in_dim %922, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %924 = stablehlo.add %921, %923 : tensor<256x7x7x512xf32>
      %925 = stablehlo.convert %924 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %926 = func.call @relu_171(%925) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %927 = stablehlo.convert %arg336 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %928 = stablehlo.convolution(%926, %927) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %929 = stablehlo.broadcast_in_dim %arg472, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %930 = stablehlo.broadcast_in_dim %arg473, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %931 = stablehlo.convert %928 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %932 = stablehlo.broadcast_in_dim %929, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %933 = stablehlo.subtract %931, %932 : tensor<256x7x7x512xf32>
      %cst_48 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %934 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %935 = stablehlo.add %930, %934 : tensor<1x1x1x512xf32>
      %936 = stablehlo.rsqrt %935 : tensor<1x1x1x512xf32>
      %937 = stablehlo.reshape %arg332 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %938 = stablehlo.multiply %936, %937 : tensor<1x1x1x512xf32>
      %939 = stablehlo.broadcast_in_dim %938, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %940 = stablehlo.multiply %933, %939 : tensor<256x7x7x512xf32>
      %941 = stablehlo.reshape %arg331 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %942 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %943 = stablehlo.add %940, %942 : tensor<256x7x7x512xf32>
      %944 = stablehlo.convert %943 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %945 = func.call @relu_171(%944) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %946 = stablehlo.convert %arg337 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
      %947 = stablehlo.convolution(%945, %946) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
      %948 = stablehlo.broadcast_in_dim %arg474, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %949 = stablehlo.broadcast_in_dim %arg475, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %950 = stablehlo.convert %947 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %951 = stablehlo.broadcast_in_dim %948, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %952 = stablehlo.subtract %950, %951 : tensor<256x7x7x2048xf32>
      %cst_49 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %953 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %954 = stablehlo.add %949, %953 : tensor<1x1x1x2048xf32>
      %955 = stablehlo.rsqrt %954 : tensor<1x1x1x2048xf32>
      %956 = stablehlo.reshape %arg334 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %957 = stablehlo.multiply %955, %956 : tensor<1x1x1x2048xf32>
      %958 = stablehlo.broadcast_in_dim %957, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %959 = stablehlo.multiply %952, %958 : tensor<256x7x7x2048xf32>
      %960 = stablehlo.reshape %arg333 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %961 = stablehlo.broadcast_in_dim %960, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %962 = stablehlo.add %959, %961 : tensor<256x7x7x2048xf32>
      %963 = stablehlo.convert %962 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %964 = stablehlo.add %907, %963 : tensor<256x7x7x2048xf16>
      %965 = func.call @relu_189(%964) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
      %966 = stablehlo.convert %arg344 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
      %967 = stablehlo.convolution(%965, %966) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<256x7x7x512xf16>
      %968 = stablehlo.broadcast_in_dim %arg476, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %969 = stablehlo.broadcast_in_dim %arg477, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %970 = stablehlo.convert %967 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %971 = stablehlo.broadcast_in_dim %968, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %972 = stablehlo.subtract %970, %971 : tensor<256x7x7x512xf32>
      %cst_50 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %973 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %974 = stablehlo.add %969, %973 : tensor<1x1x1x512xf32>
      %975 = stablehlo.rsqrt %974 : tensor<1x1x1x512xf32>
      %976 = stablehlo.reshape %arg339 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %977 = stablehlo.multiply %975, %976 : tensor<1x1x1x512xf32>
      %978 = stablehlo.broadcast_in_dim %977, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %979 = stablehlo.multiply %972, %978 : tensor<256x7x7x512xf32>
      %980 = stablehlo.reshape %arg338 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %981 = stablehlo.broadcast_in_dim %980, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %982 = stablehlo.add %979, %981 : tensor<256x7x7x512xf32>
      %983 = stablehlo.convert %982 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %984 = func.call @relu_171(%983) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %985 = stablehlo.convert %arg345 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %986 = stablehlo.convolution(%984, %985) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %987 = stablehlo.broadcast_in_dim %arg478, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %988 = stablehlo.broadcast_in_dim %arg479, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %989 = stablehlo.convert %986 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %990 = stablehlo.broadcast_in_dim %987, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %991 = stablehlo.subtract %989, %990 : tensor<256x7x7x512xf32>
      %cst_51 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %992 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %993 = stablehlo.add %988, %992 : tensor<1x1x1x512xf32>
      %994 = stablehlo.rsqrt %993 : tensor<1x1x1x512xf32>
      %995 = stablehlo.reshape %arg341 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %996 = stablehlo.multiply %994, %995 : tensor<1x1x1x512xf32>
      %997 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %998 = stablehlo.multiply %991, %997 : tensor<256x7x7x512xf32>
      %999 = stablehlo.reshape %arg340 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1000 = stablehlo.broadcast_in_dim %999, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1001 = stablehlo.add %998, %1000 : tensor<256x7x7x512xf32>
      %1002 = stablehlo.convert %1001 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1003 = func.call @relu_171(%1002) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %1004 = stablehlo.convert %arg346 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
      %1005 = stablehlo.convolution(%1003, %1004) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
      %1006 = stablehlo.broadcast_in_dim %arg480, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1007 = stablehlo.broadcast_in_dim %arg481, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1008 = stablehlo.convert %1005 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %1009 = stablehlo.broadcast_in_dim %1006, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1010 = stablehlo.subtract %1008, %1009 : tensor<256x7x7x2048xf32>
      %cst_52 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1011 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %1012 = stablehlo.add %1007, %1011 : tensor<1x1x1x2048xf32>
      %1013 = stablehlo.rsqrt %1012 : tensor<1x1x1x2048xf32>
      %1014 = stablehlo.reshape %arg343 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1015 = stablehlo.multiply %1013, %1014 : tensor<1x1x1x2048xf32>
      %1016 = stablehlo.broadcast_in_dim %1015, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1017 = stablehlo.multiply %1010, %1016 : tensor<256x7x7x2048xf32>
      %1018 = stablehlo.reshape %arg342 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1019 = stablehlo.broadcast_in_dim %1018, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1020 = stablehlo.add %1017, %1019 : tensor<256x7x7x2048xf32>
      %1021 = stablehlo.convert %1020 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %1022 = stablehlo.add %965, %1021 : tensor<256x7x7x2048xf16>
      %1023 = func.call @relu_189(%1022) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
      %1024 = stablehlo.convert %1023 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1025 = stablehlo.reduce(%1024 init: %cst_53) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x2048xf32>, tensor<f32>) -> tensor<256x2048xf32>
      %cst_54 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
      %1026 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
      %1027 = stablehlo.divide %1025, %1026 : tensor<256x2048xf32>
      %1028 = stablehlo.convert %1027 : (tensor<256x2048xf32>) -> tensor<256x2048xf16>
      %1029 = stablehlo.convert %arg426 : (tensor<2048x1000xf32>) -> tensor<2048x1000xf16>
      %1030 = stablehlo.convert %arg425 : (tensor<1000xf32>) -> tensor<1000xf16>
      %1031 = stablehlo.dot_general %1028, %1029, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf16>, tensor<2048x1000xf16>) -> tensor<256x1000xf16>
      %1032 = stablehlo.reshape %1030 : (tensor<1000xf16>) -> tensor<1x1000xf16>
      %1033 = stablehlo.broadcast_in_dim %1032, dims = [0, 1] : (tensor<1x1000xf16>) -> tensor<256x1000xf16>
      %1034 = stablehlo.add %1031, %1033 : tensor<256x1000xf16>
      %1035 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
      %1036 = stablehlo.iota dim = 0 : tensor<1000xi32>
      %1037 = stablehlo.reshape %1036 : (tensor<1000xi32>) -> tensor<1x1000xi32>
      %1038 = stablehlo.broadcast_in_dim %1035, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
      %1039 = stablehlo.broadcast_in_dim %1037, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
      %1040 = stablehlo.compare  EQ, %1038, %1039,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
      %cst_55 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1041 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %cst_56 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1042 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %1043 = stablehlo.select %1040, %1041, %1042 : tensor<256x1000xi1>, tensor<256x1000xf32>
      %1044 = stablehlo.convert %1043 : tensor<256x1000xf32>
      %1045 = func.call @log_softmax(%1034) : (tensor<256x1000xf16>) -> tensor<256x1000xf16>
      %1046 = stablehlo.convert %1045 : (tensor<256x1000xf16>) -> tensor<256x1000xf32>
      %1047 = stablehlo.multiply %1044, %1046 : tensor<256x1000xf32>
      %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1048 = stablehlo.reduce(%1047 init: %cst_57) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
      %1049 = stablehlo.negate %1048 : tensor<256xf32>
      %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1050 = stablehlo.reduce(%1049 init: %cst_58) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_59 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %1051 = stablehlo.divide %1050, %cst_59 : tensor<f32>
      %1052 = func.call @argmax(%1034) : (tensor<256x1000xf16>) -> tensor<256xi32>
      %1053 = stablehlo.compare  EQ, %1052, %2,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
      %1054 = stablehlo.convert %1053 : (tensor<256xi1>) -> tensor<256xi32>
      %1055 = stablehlo.convert %1054 : (tensor<256xi32>) -> tensor<256xf32>
      %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1056 = stablehlo.reduce(%1055 init: %cst_60) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_61 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %1057 = stablehlo.divide %1056, %cst_61 : tensor<f32>
      %1058 = "stablehlo.all_reduce"(%1057) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg538: tensor<f32>, %arg539: tensor<f32>):
        %1064 = stablehlo.add %arg538, %arg539 : tensor<f32>
        stablehlo.return %1064 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %1059 = "stablehlo.all_reduce"(%1051) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg538: tensor<f32>, %arg539: tensor<f32>):
        %1064 = stablehlo.add %arg538, %arg539 : tensor<f32>
        stablehlo.return %1064 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_62 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1060 = stablehlo.divide %1058, %cst_62 : tensor<f32>
      %cst_63 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %1061 = stablehlo.divide %1059, %cst_63 : tensor<f32>
      %1062 = stablehlo.broadcast_in_dim %1060, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %1063 = stablehlo.broadcast_in_dim %1061, dims = [] : (tensor<f32>) -> tensor<1xf32>
      sdy.return %1062, %1063 : tensor<1xf32>, tensor<1xf32>
    } : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x1x64x64xf32>, tensor<3x3x64x64xf32>, tensor<1x1x64x256xf32>, tensor<1x1x64x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x1x256x64xf32>, tensor<3x3x64x64xf32>, tensor<1x1x64x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x1x1024x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x512x2048xf32>, tensor<1x1x1024x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x1x2048x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x512x2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x1x2048x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x512x2048xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x1x256x64xf32>, tensor<3x3x64x64xf32>, tensor<1x1x64x256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x256x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x512x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<1x1x512x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<1000xf32>, tensor<2048x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x256x224x224x3xf16>, tensor<4x256xi32>) -> (tensor<4xf32>, tensor<4xf32>)
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
  func.func private @relu_32(%arg0: tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x56x56x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x256xf16>
    return %1 : tensor<256x56x56x256xf16>
  }
  func.func private @relu_55(%arg0: tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x56x56x128xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x56x56x128xf16>
    return %1 : tensor<256x56x56x128xf16>
  }
  func.func private @relu_65(%arg0: tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x128xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x128xf16>
    return %1 : tensor<256x28x28x128xf16>
  }
  func.func private @relu_83(%arg0: tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x512xf16>
    return %1 : tensor<256x28x28x512xf16>
  }
  func.func private @relu_105(%arg0: tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x256xf16>
    return %1 : tensor<256x28x28x256xf16>
  }
  func.func private @relu_115(%arg0: tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xf16>
    return %1 : tensor<256x14x14x256xf16>
  }
  func.func private @relu_133(%arg0: tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x1024xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x1024xf16>
    return %1 : tensor<256x14x14x1024xf16>
  }
  func.func private @relu_161(%arg0: tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x512xf16>
    return %1 : tensor<256x14x14x512xf16>
  }
  func.func private @relu_171(%arg0: tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xf16>
    return %1 : tensor<256x7x7x512xf16>
  }
  func.func private @relu_189(%arg0: tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x7x7x2048xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x2048xf16>
    return %1 : tensor<256x7x7x2048xf16>
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
