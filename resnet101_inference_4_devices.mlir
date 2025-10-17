module @jit_call_wrapped attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=4]>
  func.func public @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<1x1x64x64xf32>, %arg7: tensor<3x3x64x64xf32>, %arg8: tensor<1x1x64x256xf32>, %arg9: tensor<1x1x64x256xf32>, %arg10: tensor<256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<256xf32>, %arg17: tensor<256xf32>, %arg18: tensor<1x1x256x64xf32>, %arg19: tensor<3x3x64x64xf32>, %arg20: tensor<1x1x64x256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256xf32>, %arg25: tensor<1024xf32>, %arg26: tensor<1024xf32>, %arg27: tensor<1x1x1024x256xf32>, %arg28: tensor<3x3x256x256xf32>, %arg29: tensor<1x1x256x1024xf32>, %arg30: tensor<256xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256xf32>, %arg34: tensor<1024xf32>, %arg35: tensor<1024xf32>, %arg36: tensor<1x1x1024x256xf32>, %arg37: tensor<3x3x256x256xf32>, %arg38: tensor<1x1x256x1024xf32>, %arg39: tensor<256xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<256xf32>, %arg43: tensor<1024xf32>, %arg44: tensor<1024xf32>, %arg45: tensor<1x1x1024x256xf32>, %arg46: tensor<3x3x256x256xf32>, %arg47: tensor<1x1x256x1024xf32>, %arg48: tensor<256xf32>, %arg49: tensor<256xf32>, %arg50: tensor<256xf32>, %arg51: tensor<256xf32>, %arg52: tensor<1024xf32>, %arg53: tensor<1024xf32>, %arg54: tensor<1x1x1024x256xf32>, %arg55: tensor<3x3x256x256xf32>, %arg56: tensor<1x1x256x1024xf32>, %arg57: tensor<256xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<1024xf32>, %arg62: tensor<1024xf32>, %arg63: tensor<1x1x1024x256xf32>, %arg64: tensor<3x3x256x256xf32>, %arg65: tensor<1x1x256x1024xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<1024xf32>, %arg71: tensor<1024xf32>, %arg72: tensor<1x1x1024x256xf32>, %arg73: tensor<3x3x256x256xf32>, %arg74: tensor<1x1x256x1024xf32>, %arg75: tensor<256xf32>, %arg76: tensor<256xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256xf32>, %arg79: tensor<1024xf32>, %arg80: tensor<1024xf32>, %arg81: tensor<1x1x1024x256xf32>, %arg82: tensor<3x3x256x256xf32>, %arg83: tensor<1x1x256x1024xf32>, %arg84: tensor<256xf32>, %arg85: tensor<256xf32>, %arg86: tensor<256xf32>, %arg87: tensor<256xf32>, %arg88: tensor<1024xf32>, %arg89: tensor<1024xf32>, %arg90: tensor<1x1x1024x256xf32>, %arg91: tensor<3x3x256x256xf32>, %arg92: tensor<1x1x256x1024xf32>, %arg93: tensor<256xf32>, %arg94: tensor<256xf32>, %arg95: tensor<256xf32>, %arg96: tensor<256xf32>, %arg97: tensor<1024xf32>, %arg98: tensor<1024xf32>, %arg99: tensor<1x1x1024x256xf32>, %arg100: tensor<3x3x256x256xf32>, %arg101: tensor<1x1x256x1024xf32>, %arg102: tensor<256xf32>, %arg103: tensor<256xf32>, %arg104: tensor<256xf32>, %arg105: tensor<256xf32>, %arg106: tensor<1024xf32>, %arg107: tensor<1024xf32>, %arg108: tensor<1x1x1024x256xf32>, %arg109: tensor<3x3x256x256xf32>, %arg110: tensor<1x1x256x1024xf32>, %arg111: tensor<64xf32>, %arg112: tensor<64xf32>, %arg113: tensor<64xf32>, %arg114: tensor<64xf32>, %arg115: tensor<256xf32>, %arg116: tensor<256xf32>, %arg117: tensor<1x1x256x64xf32>, %arg118: tensor<3x3x64x64xf32>, %arg119: tensor<1x1x64x256xf32>, %arg120: tensor<256xf32>, %arg121: tensor<256xf32>, %arg122: tensor<256xf32>, %arg123: tensor<256xf32>, %arg124: tensor<1024xf32>, %arg125: tensor<1024xf32>, %arg126: tensor<1x1x1024x256xf32>, %arg127: tensor<3x3x256x256xf32>, %arg128: tensor<1x1x256x1024xf32>, %arg129: tensor<256xf32>, %arg130: tensor<256xf32>, %arg131: tensor<256xf32>, %arg132: tensor<256xf32>, %arg133: tensor<1024xf32>, %arg134: tensor<1024xf32>, %arg135: tensor<1x1x1024x256xf32>, %arg136: tensor<3x3x256x256xf32>, %arg137: tensor<1x1x256x1024xf32>, %arg138: tensor<256xf32>, %arg139: tensor<256xf32>, %arg140: tensor<256xf32>, %arg141: tensor<256xf32>, %arg142: tensor<1024xf32>, %arg143: tensor<1024xf32>, %arg144: tensor<1x1x1024x256xf32>, %arg145: tensor<3x3x256x256xf32>, %arg146: tensor<1x1x256x1024xf32>, %arg147: tensor<256xf32>, %arg148: tensor<256xf32>, %arg149: tensor<256xf32>, %arg150: tensor<256xf32>, %arg151: tensor<1024xf32>, %arg152: tensor<1024xf32>, %arg153: tensor<1x1x1024x256xf32>, %arg154: tensor<3x3x256x256xf32>, %arg155: tensor<1x1x256x1024xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<256xf32>, %arg160: tensor<1024xf32>, %arg161: tensor<1024xf32>, %arg162: tensor<1x1x1024x256xf32>, %arg163: tensor<3x3x256x256xf32>, %arg164: tensor<1x1x256x1024xf32>, %arg165: tensor<256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256xf32>, %arg168: tensor<256xf32>, %arg169: tensor<1024xf32>, %arg170: tensor<1024xf32>, %arg171: tensor<1x1x1024x256xf32>, %arg172: tensor<3x3x256x256xf32>, %arg173: tensor<1x1x256x1024xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<1024xf32>, %arg179: tensor<1024xf32>, %arg180: tensor<1x1x1024x256xf32>, %arg181: tensor<3x3x256x256xf32>, %arg182: tensor<1x1x256x1024xf32>, %arg183: tensor<256xf32>, %arg184: tensor<256xf32>, %arg185: tensor<256xf32>, %arg186: tensor<256xf32>, %arg187: tensor<1024xf32>, %arg188: tensor<1024xf32>, %arg189: tensor<1x1x1024x256xf32>, %arg190: tensor<3x3x256x256xf32>, %arg191: tensor<1x1x256x1024xf32>, %arg192: tensor<256xf32>, %arg193: tensor<256xf32>, %arg194: tensor<256xf32>, %arg195: tensor<256xf32>, %arg196: tensor<1024xf32>, %arg197: tensor<1024xf32>, %arg198: tensor<1x1x1024x256xf32>, %arg199: tensor<3x3x256x256xf32>, %arg200: tensor<1x1x256x1024xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256xf32>, %arg203: tensor<256xf32>, %arg204: tensor<256xf32>, %arg205: tensor<1024xf32>, %arg206: tensor<1024xf32>, %arg207: tensor<1x1x1024x256xf32>, %arg208: tensor<3x3x256x256xf32>, %arg209: tensor<1x1x256x1024xf32>, %arg210: tensor<128xf32>, %arg211: tensor<128xf32>, %arg212: tensor<128xf32>, %arg213: tensor<128xf32>, %arg214: tensor<512xf32>, %arg215: tensor<512xf32>, %arg216: tensor<1x1x256x128xf32>, %arg217: tensor<3x3x128x128xf32>, %arg218: tensor<1x1x128x512xf32>, %arg219: tensor<1x1x256x512xf32>, %arg220: tensor<512xf32>, %arg221: tensor<512xf32>, %arg222: tensor<512xf32>, %arg223: tensor<512xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<2048xf32>, %arg227: tensor<2048xf32>, %arg228: tensor<1x1x1024x512xf32>, %arg229: tensor<3x3x512x512xf32>, %arg230: tensor<1x1x512x2048xf32>, %arg231: tensor<1x1x1024x2048xf32>, %arg232: tensor<2048xf32>, %arg233: tensor<2048xf32>, %arg234: tensor<512xf32>, %arg235: tensor<512xf32>, %arg236: tensor<512xf32>, %arg237: tensor<512xf32>, %arg238: tensor<2048xf32>, %arg239: tensor<2048xf32>, %arg240: tensor<1x1x2048x512xf32>, %arg241: tensor<3x3x512x512xf32>, %arg242: tensor<1x1x512x2048xf32>, %arg243: tensor<512xf32>, %arg244: tensor<512xf32>, %arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<2048xf32>, %arg248: tensor<2048xf32>, %arg249: tensor<1x1x2048x512xf32>, %arg250: tensor<3x3x512x512xf32>, %arg251: tensor<1x1x512x2048xf32>, %arg252: tensor<128xf32>, %arg253: tensor<128xf32>, %arg254: tensor<128xf32>, %arg255: tensor<128xf32>, %arg256: tensor<512xf32>, %arg257: tensor<512xf32>, %arg258: tensor<1x1x512x128xf32>, %arg259: tensor<3x3x128x128xf32>, %arg260: tensor<1x1x128x512xf32>, %arg261: tensor<128xf32>, %arg262: tensor<128xf32>, %arg263: tensor<128xf32>, %arg264: tensor<128xf32>, %arg265: tensor<512xf32>, %arg266: tensor<512xf32>, %arg267: tensor<1x1x512x128xf32>, %arg268: tensor<3x3x128x128xf32>, %arg269: tensor<1x1x128x512xf32>, %arg270: tensor<128xf32>, %arg271: tensor<128xf32>, %arg272: tensor<128xf32>, %arg273: tensor<128xf32>, %arg274: tensor<512xf32>, %arg275: tensor<512xf32>, %arg276: tensor<1x1x512x128xf32>, %arg277: tensor<3x3x128x128xf32>, %arg278: tensor<1x1x128x512xf32>, %arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<1024xf32>, %arg284: tensor<1024xf32>, %arg285: tensor<1x1x512x256xf32>, %arg286: tensor<3x3x256x256xf32>, %arg287: tensor<1x1x256x1024xf32>, %arg288: tensor<1x1x512x1024xf32>, %arg289: tensor<1024xf32>, %arg290: tensor<1024xf32>, %arg291: tensor<256xf32>, %arg292: tensor<256xf32>, %arg293: tensor<256xf32>, %arg294: tensor<256xf32>, %arg295: tensor<1024xf32>, %arg296: tensor<1024xf32>, %arg297: tensor<1x1x1024x256xf32>, %arg298: tensor<3x3x256x256xf32>, %arg299: tensor<1x1x256x1024xf32>, %arg300: tensor<256xf32>, %arg301: tensor<256xf32>, %arg302: tensor<256xf32>, %arg303: tensor<256xf32>, %arg304: tensor<1024xf32>, %arg305: tensor<1024xf32>, %arg306: tensor<1x1x1024x256xf32>, %arg307: tensor<3x3x256x256xf32>, %arg308: tensor<1x1x256x1024xf32>, %arg309: tensor<1000xf32>, %arg310: tensor<2048x1000xf32>, %arg311: tensor<64xf32>, %arg312: tensor<64xf32>, %arg313: tensor<7x7x3x64xf32>, %arg314: tensor<64xf32>, %arg315: tensor<64xf32>, %arg316: tensor<64xf32>, %arg317: tensor<64xf32>, %arg318: tensor<256xf32>, %arg319: tensor<256xf32>, %arg320: tensor<256xf32>, %arg321: tensor<256xf32>, %arg322: tensor<64xf32>, %arg323: tensor<64xf32>, %arg324: tensor<64xf32>, %arg325: tensor<64xf32>, %arg326: tensor<256xf32>, %arg327: tensor<256xf32>, %arg328: tensor<256xf32>, %arg329: tensor<256xf32>, %arg330: tensor<256xf32>, %arg331: tensor<256xf32>, %arg332: tensor<1024xf32>, %arg333: tensor<1024xf32>, %arg334: tensor<256xf32>, %arg335: tensor<256xf32>, %arg336: tensor<256xf32>, %arg337: tensor<256xf32>, %arg338: tensor<1024xf32>, %arg339: tensor<1024xf32>, %arg340: tensor<256xf32>, %arg341: tensor<256xf32>, %arg342: tensor<256xf32>, %arg343: tensor<256xf32>, %arg344: tensor<1024xf32>, %arg345: tensor<1024xf32>, %arg346: tensor<256xf32>, %arg347: tensor<256xf32>, %arg348: tensor<256xf32>, %arg349: tensor<256xf32>, %arg350: tensor<1024xf32>, %arg351: tensor<1024xf32>, %arg352: tensor<256xf32>, %arg353: tensor<256xf32>, %arg354: tensor<256xf32>, %arg355: tensor<256xf32>, %arg356: tensor<1024xf32>, %arg357: tensor<1024xf32>, %arg358: tensor<256xf32>, %arg359: tensor<256xf32>, %arg360: tensor<256xf32>, %arg361: tensor<256xf32>, %arg362: tensor<1024xf32>, %arg363: tensor<1024xf32>, %arg364: tensor<256xf32>, %arg365: tensor<256xf32>, %arg366: tensor<256xf32>, %arg367: tensor<256xf32>, %arg368: tensor<1024xf32>, %arg369: tensor<1024xf32>, %arg370: tensor<256xf32>, %arg371: tensor<256xf32>, %arg372: tensor<256xf32>, %arg373: tensor<256xf32>, %arg374: tensor<1024xf32>, %arg375: tensor<1024xf32>, %arg376: tensor<256xf32>, %arg377: tensor<256xf32>, %arg378: tensor<256xf32>, %arg379: tensor<256xf32>, %arg380: tensor<1024xf32>, %arg381: tensor<1024xf32>, %arg382: tensor<256xf32>, %arg383: tensor<256xf32>, %arg384: tensor<256xf32>, %arg385: tensor<256xf32>, %arg386: tensor<1024xf32>, %arg387: tensor<1024xf32>, %arg388: tensor<64xf32>, %arg389: tensor<64xf32>, %arg390: tensor<64xf32>, %arg391: tensor<64xf32>, %arg392: tensor<256xf32>, %arg393: tensor<256xf32>, %arg394: tensor<256xf32>, %arg395: tensor<256xf32>, %arg396: tensor<256xf32>, %arg397: tensor<256xf32>, %arg398: tensor<1024xf32>, %arg399: tensor<1024xf32>, %arg400: tensor<256xf32>, %arg401: tensor<256xf32>, %arg402: tensor<256xf32>, %arg403: tensor<256xf32>, %arg404: tensor<1024xf32>, %arg405: tensor<1024xf32>, %arg406: tensor<256xf32>, %arg407: tensor<256xf32>, %arg408: tensor<256xf32>, %arg409: tensor<256xf32>, %arg410: tensor<1024xf32>, %arg411: tensor<1024xf32>, %arg412: tensor<256xf32>, %arg413: tensor<256xf32>, %arg414: tensor<256xf32>, %arg415: tensor<256xf32>, %arg416: tensor<1024xf32>, %arg417: tensor<1024xf32>, %arg418: tensor<256xf32>, %arg419: tensor<256xf32>, %arg420: tensor<256xf32>, %arg421: tensor<256xf32>, %arg422: tensor<1024xf32>, %arg423: tensor<1024xf32>, %arg424: tensor<256xf32>, %arg425: tensor<256xf32>, %arg426: tensor<256xf32>, %arg427: tensor<256xf32>, %arg428: tensor<1024xf32>, %arg429: tensor<1024xf32>, %arg430: tensor<256xf32>, %arg431: tensor<256xf32>, %arg432: tensor<256xf32>, %arg433: tensor<256xf32>, %arg434: tensor<1024xf32>, %arg435: tensor<1024xf32>, %arg436: tensor<256xf32>, %arg437: tensor<256xf32>, %arg438: tensor<256xf32>, %arg439: tensor<256xf32>, %arg440: tensor<1024xf32>, %arg441: tensor<1024xf32>, %arg442: tensor<256xf32>, %arg443: tensor<256xf32>, %arg444: tensor<256xf32>, %arg445: tensor<256xf32>, %arg446: tensor<1024xf32>, %arg447: tensor<1024xf32>, %arg448: tensor<256xf32>, %arg449: tensor<256xf32>, %arg450: tensor<256xf32>, %arg451: tensor<256xf32>, %arg452: tensor<1024xf32>, %arg453: tensor<1024xf32>, %arg454: tensor<128xf32>, %arg455: tensor<128xf32>, %arg456: tensor<128xf32>, %arg457: tensor<128xf32>, %arg458: tensor<512xf32>, %arg459: tensor<512xf32>, %arg460: tensor<512xf32>, %arg461: tensor<512xf32>, %arg462: tensor<512xf32>, %arg463: tensor<512xf32>, %arg464: tensor<512xf32>, %arg465: tensor<512xf32>, %arg466: tensor<2048xf32>, %arg467: tensor<2048xf32>, %arg468: tensor<2048xf32>, %arg469: tensor<2048xf32>, %arg470: tensor<512xf32>, %arg471: tensor<512xf32>, %arg472: tensor<512xf32>, %arg473: tensor<512xf32>, %arg474: tensor<2048xf32>, %arg475: tensor<2048xf32>, %arg476: tensor<512xf32>, %arg477: tensor<512xf32>, %arg478: tensor<512xf32>, %arg479: tensor<512xf32>, %arg480: tensor<2048xf32>, %arg481: tensor<2048xf32>, %arg482: tensor<128xf32>, %arg483: tensor<128xf32>, %arg484: tensor<128xf32>, %arg485: tensor<128xf32>, %arg486: tensor<512xf32>, %arg487: tensor<512xf32>, %arg488: tensor<128xf32>, %arg489: tensor<128xf32>, %arg490: tensor<128xf32>, %arg491: tensor<128xf32>, %arg492: tensor<512xf32>, %arg493: tensor<512xf32>, %arg494: tensor<128xf32>, %arg495: tensor<128xf32>, %arg496: tensor<128xf32>, %arg497: tensor<128xf32>, %arg498: tensor<512xf32>, %arg499: tensor<512xf32>, %arg500: tensor<256xf32>, %arg501: tensor<256xf32>, %arg502: tensor<256xf32>, %arg503: tensor<256xf32>, %arg504: tensor<1024xf32>, %arg505: tensor<1024xf32>, %arg506: tensor<1024xf32>, %arg507: tensor<1024xf32>, %arg508: tensor<256xf32>, %arg509: tensor<256xf32>, %arg510: tensor<256xf32>, %arg511: tensor<256xf32>, %arg512: tensor<1024xf32>, %arg513: tensor<1024xf32>, %arg514: tensor<256xf32>, %arg515: tensor<256xf32>, %arg516: tensor<256xf32>, %arg517: tensor<256xf32>, %arg518: tensor<1024xf32>, %arg519: tensor<1024xf32>, %arg520: tensor<64xf32>, %arg521: tensor<64xf32>, %arg522: tensor<4x256x224x224x3xf16>, %arg523: tensor<4x256xi32>) -> (tensor<4xf32> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %0:2 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg181, %arg182, %arg183, %arg184, %arg185, %arg186, %arg187, %arg188, %arg189, %arg190, %arg191, %arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %arg198, %arg199, %arg200, %arg201, %arg202, %arg203, %arg204, %arg205, %arg206, %arg207, %arg208, %arg209, %arg210, %arg211, %arg212, %arg213, %arg214, %arg215, %arg216, %arg217, %arg218, %arg219, %arg220, %arg221, %arg222, %arg223, %arg224, %arg225, %arg226, %arg227, %arg228, %arg229, %arg230, %arg231, %arg232, %arg233, %arg234, %arg235, %arg236, %arg237, %arg238, %arg239, %arg240, %arg241, %arg242, %arg243, %arg244, %arg245, %arg246, %arg247, %arg248, %arg249, %arg250, %arg251, %arg252, %arg253, %arg254, %arg255, %arg256, %arg257, %arg258, %arg259, %arg260, %arg261, %arg262, %arg263, %arg264, %arg265, %arg266, %arg267, %arg268, %arg269, %arg270, %arg271, %arg272, %arg273, %arg274, %arg275, %arg276, %arg277, %arg278, %arg279, %arg280, %arg281, %arg282, %arg283, %arg284, %arg285, %arg286, %arg287, %arg288, %arg289, %arg290, %arg291, %arg292, %arg293, %arg294, %arg295, %arg296, %arg297, %arg298, %arg299, %arg300, %arg301, %arg302, %arg303, %arg304, %arg305, %arg306, %arg307, %arg308, %arg309, %arg310, %arg311, %arg312, %arg313, %arg314, %arg315, %arg316, %arg317, %arg318, %arg319, %arg320, %arg321, %arg322, %arg323, %arg324, %arg325, %arg326, %arg327, %arg328, %arg329, %arg330, %arg331, %arg332, %arg333, %arg334, %arg335, %arg336, %arg337, %arg338, %arg339, %arg340, %arg341, %arg342, %arg343, %arg344, %arg345, %arg346, %arg347, %arg348, %arg349, %arg350, %arg351, %arg352, %arg353, %arg354, %arg355, %arg356, %arg357, %arg358, %arg359, %arg360, %arg361, %arg362, %arg363, %arg364, %arg365, %arg366, %arg367, %arg368, %arg369, %arg370, %arg371, %arg372, %arg373, %arg374, %arg375, %arg376, %arg377, %arg378, %arg379, %arg380, %arg381, %arg382, %arg383, %arg384, %arg385, %arg386, %arg387, %arg388, %arg389, %arg390, %arg391, %arg392, %arg393, %arg394, %arg395, %arg396, %arg397, %arg398, %arg399, %arg400, %arg401, %arg402, %arg403, %arg404, %arg405, %arg406, %arg407, %arg408, %arg409, %arg410, %arg411, %arg412, %arg413, %arg414, %arg415, %arg416, %arg417, %arg418, %arg419, %arg420, %arg421, %arg422, %arg423, %arg424, %arg425, %arg426, %arg427, %arg428, %arg429, %arg430, %arg431, %arg432, %arg433, %arg434, %arg435, %arg436, %arg437, %arg438, %arg439, %arg440, %arg441, %arg442, %arg443, %arg444, %arg445, %arg446, %arg447, %arg448, %arg449, %arg450, %arg451, %arg452, %arg453, %arg454, %arg455, %arg456, %arg457, %arg458, %arg459, %arg460, %arg461, %arg462, %arg463, %arg464, %arg465, %arg466, %arg467, %arg468, %arg469, %arg470, %arg471, %arg472, %arg473, %arg474, %arg475, %arg476, %arg477, %arg478, %arg479, %arg480, %arg481, %arg482, %arg483, %arg484, %arg485, %arg486, %arg487, %arg488, %arg489, %arg490, %arg491, %arg492, %arg493, %arg494, %arg495, %arg496, %arg497, %arg498, %arg499, %arg500, %arg501, %arg502, %arg503, %arg504, %arg505, %arg506, %arg507, %arg508, %arg509, %arg510, %arg511, %arg512, %arg513, %arg514, %arg515, %arg516, %arg517, %arg518, %arg519, %arg520, %arg521, %arg522, %arg523) in_shardings=[<@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{"batch"}, {}, {}, {}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>] manual_axes={"batch"} (%arg524: tensor<64xf32>, %arg525: tensor<64xf32>, %arg526: tensor<64xf32>, %arg527: tensor<64xf32>, %arg528: tensor<256xf32>, %arg529: tensor<256xf32>, %arg530: tensor<1x1x64x64xf32>, %arg531: tensor<3x3x64x64xf32>, %arg532: tensor<1x1x64x256xf32>, %arg533: tensor<1x1x64x256xf32>, %arg534: tensor<256xf32>, %arg535: tensor<256xf32>, %arg536: tensor<64xf32>, %arg537: tensor<64xf32>, %arg538: tensor<64xf32>, %arg539: tensor<64xf32>, %arg540: tensor<256xf32>, %arg541: tensor<256xf32>, %arg542: tensor<1x1x256x64xf32>, %arg543: tensor<3x3x64x64xf32>, %arg544: tensor<1x1x64x256xf32>, %arg545: tensor<256xf32>, %arg546: tensor<256xf32>, %arg547: tensor<256xf32>, %arg548: tensor<256xf32>, %arg549: tensor<1024xf32>, %arg550: tensor<1024xf32>, %arg551: tensor<1x1x1024x256xf32>, %arg552: tensor<3x3x256x256xf32>, %arg553: tensor<1x1x256x1024xf32>, %arg554: tensor<256xf32>, %arg555: tensor<256xf32>, %arg556: tensor<256xf32>, %arg557: tensor<256xf32>, %arg558: tensor<1024xf32>, %arg559: tensor<1024xf32>, %arg560: tensor<1x1x1024x256xf32>, %arg561: tensor<3x3x256x256xf32>, %arg562: tensor<1x1x256x1024xf32>, %arg563: tensor<256xf32>, %arg564: tensor<256xf32>, %arg565: tensor<256xf32>, %arg566: tensor<256xf32>, %arg567: tensor<1024xf32>, %arg568: tensor<1024xf32>, %arg569: tensor<1x1x1024x256xf32>, %arg570: tensor<3x3x256x256xf32>, %arg571: tensor<1x1x256x1024xf32>, %arg572: tensor<256xf32>, %arg573: tensor<256xf32>, %arg574: tensor<256xf32>, %arg575: tensor<256xf32>, %arg576: tensor<1024xf32>, %arg577: tensor<1024xf32>, %arg578: tensor<1x1x1024x256xf32>, %arg579: tensor<3x3x256x256xf32>, %arg580: tensor<1x1x256x1024xf32>, %arg581: tensor<256xf32>, %arg582: tensor<256xf32>, %arg583: tensor<256xf32>, %arg584: tensor<256xf32>, %arg585: tensor<1024xf32>, %arg586: tensor<1024xf32>, %arg587: tensor<1x1x1024x256xf32>, %arg588: tensor<3x3x256x256xf32>, %arg589: tensor<1x1x256x1024xf32>, %arg590: tensor<256xf32>, %arg591: tensor<256xf32>, %arg592: tensor<256xf32>, %arg593: tensor<256xf32>, %arg594: tensor<1024xf32>, %arg595: tensor<1024xf32>, %arg596: tensor<1x1x1024x256xf32>, %arg597: tensor<3x3x256x256xf32>, %arg598: tensor<1x1x256x1024xf32>, %arg599: tensor<256xf32>, %arg600: tensor<256xf32>, %arg601: tensor<256xf32>, %arg602: tensor<256xf32>, %arg603: tensor<1024xf32>, %arg604: tensor<1024xf32>, %arg605: tensor<1x1x1024x256xf32>, %arg606: tensor<3x3x256x256xf32>, %arg607: tensor<1x1x256x1024xf32>, %arg608: tensor<256xf32>, %arg609: tensor<256xf32>, %arg610: tensor<256xf32>, %arg611: tensor<256xf32>, %arg612: tensor<1024xf32>, %arg613: tensor<1024xf32>, %arg614: tensor<1x1x1024x256xf32>, %arg615: tensor<3x3x256x256xf32>, %arg616: tensor<1x1x256x1024xf32>, %arg617: tensor<256xf32>, %arg618: tensor<256xf32>, %arg619: tensor<256xf32>, %arg620: tensor<256xf32>, %arg621: tensor<1024xf32>, %arg622: tensor<1024xf32>, %arg623: tensor<1x1x1024x256xf32>, %arg624: tensor<3x3x256x256xf32>, %arg625: tensor<1x1x256x1024xf32>, %arg626: tensor<256xf32>, %arg627: tensor<256xf32>, %arg628: tensor<256xf32>, %arg629: tensor<256xf32>, %arg630: tensor<1024xf32>, %arg631: tensor<1024xf32>, %arg632: tensor<1x1x1024x256xf32>, %arg633: tensor<3x3x256x256xf32>, %arg634: tensor<1x1x256x1024xf32>, %arg635: tensor<64xf32>, %arg636: tensor<64xf32>, %arg637: tensor<64xf32>, %arg638: tensor<64xf32>, %arg639: tensor<256xf32>, %arg640: tensor<256xf32>, %arg641: tensor<1x1x256x64xf32>, %arg642: tensor<3x3x64x64xf32>, %arg643: tensor<1x1x64x256xf32>, %arg644: tensor<256xf32>, %arg645: tensor<256xf32>, %arg646: tensor<256xf32>, %arg647: tensor<256xf32>, %arg648: tensor<1024xf32>, %arg649: tensor<1024xf32>, %arg650: tensor<1x1x1024x256xf32>, %arg651: tensor<3x3x256x256xf32>, %arg652: tensor<1x1x256x1024xf32>, %arg653: tensor<256xf32>, %arg654: tensor<256xf32>, %arg655: tensor<256xf32>, %arg656: tensor<256xf32>, %arg657: tensor<1024xf32>, %arg658: tensor<1024xf32>, %arg659: tensor<1x1x1024x256xf32>, %arg660: tensor<3x3x256x256xf32>, %arg661: tensor<1x1x256x1024xf32>, %arg662: tensor<256xf32>, %arg663: tensor<256xf32>, %arg664: tensor<256xf32>, %arg665: tensor<256xf32>, %arg666: tensor<1024xf32>, %arg667: tensor<1024xf32>, %arg668: tensor<1x1x1024x256xf32>, %arg669: tensor<3x3x256x256xf32>, %arg670: tensor<1x1x256x1024xf32>, %arg671: tensor<256xf32>, %arg672: tensor<256xf32>, %arg673: tensor<256xf32>, %arg674: tensor<256xf32>, %arg675: tensor<1024xf32>, %arg676: tensor<1024xf32>, %arg677: tensor<1x1x1024x256xf32>, %arg678: tensor<3x3x256x256xf32>, %arg679: tensor<1x1x256x1024xf32>, %arg680: tensor<256xf32>, %arg681: tensor<256xf32>, %arg682: tensor<256xf32>, %arg683: tensor<256xf32>, %arg684: tensor<1024xf32>, %arg685: tensor<1024xf32>, %arg686: tensor<1x1x1024x256xf32>, %arg687: tensor<3x3x256x256xf32>, %arg688: tensor<1x1x256x1024xf32>, %arg689: tensor<256xf32>, %arg690: tensor<256xf32>, %arg691: tensor<256xf32>, %arg692: tensor<256xf32>, %arg693: tensor<1024xf32>, %arg694: tensor<1024xf32>, %arg695: tensor<1x1x1024x256xf32>, %arg696: tensor<3x3x256x256xf32>, %arg697: tensor<1x1x256x1024xf32>, %arg698: tensor<256xf32>, %arg699: tensor<256xf32>, %arg700: tensor<256xf32>, %arg701: tensor<256xf32>, %arg702: tensor<1024xf32>, %arg703: tensor<1024xf32>, %arg704: tensor<1x1x1024x256xf32>, %arg705: tensor<3x3x256x256xf32>, %arg706: tensor<1x1x256x1024xf32>, %arg707: tensor<256xf32>, %arg708: tensor<256xf32>, %arg709: tensor<256xf32>, %arg710: tensor<256xf32>, %arg711: tensor<1024xf32>, %arg712: tensor<1024xf32>, %arg713: tensor<1x1x1024x256xf32>, %arg714: tensor<3x3x256x256xf32>, %arg715: tensor<1x1x256x1024xf32>, %arg716: tensor<256xf32>, %arg717: tensor<256xf32>, %arg718: tensor<256xf32>, %arg719: tensor<256xf32>, %arg720: tensor<1024xf32>, %arg721: tensor<1024xf32>, %arg722: tensor<1x1x1024x256xf32>, %arg723: tensor<3x3x256x256xf32>, %arg724: tensor<1x1x256x1024xf32>, %arg725: tensor<256xf32>, %arg726: tensor<256xf32>, %arg727: tensor<256xf32>, %arg728: tensor<256xf32>, %arg729: tensor<1024xf32>, %arg730: tensor<1024xf32>, %arg731: tensor<1x1x1024x256xf32>, %arg732: tensor<3x3x256x256xf32>, %arg733: tensor<1x1x256x1024xf32>, %arg734: tensor<128xf32>, %arg735: tensor<128xf32>, %arg736: tensor<128xf32>, %arg737: tensor<128xf32>, %arg738: tensor<512xf32>, %arg739: tensor<512xf32>, %arg740: tensor<1x1x256x128xf32>, %arg741: tensor<3x3x128x128xf32>, %arg742: tensor<1x1x128x512xf32>, %arg743: tensor<1x1x256x512xf32>, %arg744: tensor<512xf32>, %arg745: tensor<512xf32>, %arg746: tensor<512xf32>, %arg747: tensor<512xf32>, %arg748: tensor<512xf32>, %arg749: tensor<512xf32>, %arg750: tensor<2048xf32>, %arg751: tensor<2048xf32>, %arg752: tensor<1x1x1024x512xf32>, %arg753: tensor<3x3x512x512xf32>, %arg754: tensor<1x1x512x2048xf32>, %arg755: tensor<1x1x1024x2048xf32>, %arg756: tensor<2048xf32>, %arg757: tensor<2048xf32>, %arg758: tensor<512xf32>, %arg759: tensor<512xf32>, %arg760: tensor<512xf32>, %arg761: tensor<512xf32>, %arg762: tensor<2048xf32>, %arg763: tensor<2048xf32>, %arg764: tensor<1x1x2048x512xf32>, %arg765: tensor<3x3x512x512xf32>, %arg766: tensor<1x1x512x2048xf32>, %arg767: tensor<512xf32>, %arg768: tensor<512xf32>, %arg769: tensor<512xf32>, %arg770: tensor<512xf32>, %arg771: tensor<2048xf32>, %arg772: tensor<2048xf32>, %arg773: tensor<1x1x2048x512xf32>, %arg774: tensor<3x3x512x512xf32>, %arg775: tensor<1x1x512x2048xf32>, %arg776: tensor<128xf32>, %arg777: tensor<128xf32>, %arg778: tensor<128xf32>, %arg779: tensor<128xf32>, %arg780: tensor<512xf32>, %arg781: tensor<512xf32>, %arg782: tensor<1x1x512x128xf32>, %arg783: tensor<3x3x128x128xf32>, %arg784: tensor<1x1x128x512xf32>, %arg785: tensor<128xf32>, %arg786: tensor<128xf32>, %arg787: tensor<128xf32>, %arg788: tensor<128xf32>, %arg789: tensor<512xf32>, %arg790: tensor<512xf32>, %arg791: tensor<1x1x512x128xf32>, %arg792: tensor<3x3x128x128xf32>, %arg793: tensor<1x1x128x512xf32>, %arg794: tensor<128xf32>, %arg795: tensor<128xf32>, %arg796: tensor<128xf32>, %arg797: tensor<128xf32>, %arg798: tensor<512xf32>, %arg799: tensor<512xf32>, %arg800: tensor<1x1x512x128xf32>, %arg801: tensor<3x3x128x128xf32>, %arg802: tensor<1x1x128x512xf32>, %arg803: tensor<256xf32>, %arg804: tensor<256xf32>, %arg805: tensor<256xf32>, %arg806: tensor<256xf32>, %arg807: tensor<1024xf32>, %arg808: tensor<1024xf32>, %arg809: tensor<1x1x512x256xf32>, %arg810: tensor<3x3x256x256xf32>, %arg811: tensor<1x1x256x1024xf32>, %arg812: tensor<1x1x512x1024xf32>, %arg813: tensor<1024xf32>, %arg814: tensor<1024xf32>, %arg815: tensor<256xf32>, %arg816: tensor<256xf32>, %arg817: tensor<256xf32>, %arg818: tensor<256xf32>, %arg819: tensor<1024xf32>, %arg820: tensor<1024xf32>, %arg821: tensor<1x1x1024x256xf32>, %arg822: tensor<3x3x256x256xf32>, %arg823: tensor<1x1x256x1024xf32>, %arg824: tensor<256xf32>, %arg825: tensor<256xf32>, %arg826: tensor<256xf32>, %arg827: tensor<256xf32>, %arg828: tensor<1024xf32>, %arg829: tensor<1024xf32>, %arg830: tensor<1x1x1024x256xf32>, %arg831: tensor<3x3x256x256xf32>, %arg832: tensor<1x1x256x1024xf32>, %arg833: tensor<1000xf32>, %arg834: tensor<2048x1000xf32>, %arg835: tensor<64xf32>, %arg836: tensor<64xf32>, %arg837: tensor<7x7x3x64xf32>, %arg838: tensor<64xf32>, %arg839: tensor<64xf32>, %arg840: tensor<64xf32>, %arg841: tensor<64xf32>, %arg842: tensor<256xf32>, %arg843: tensor<256xf32>, %arg844: tensor<256xf32>, %arg845: tensor<256xf32>, %arg846: tensor<64xf32>, %arg847: tensor<64xf32>, %arg848: tensor<64xf32>, %arg849: tensor<64xf32>, %arg850: tensor<256xf32>, %arg851: tensor<256xf32>, %arg852: tensor<256xf32>, %arg853: tensor<256xf32>, %arg854: tensor<256xf32>, %arg855: tensor<256xf32>, %arg856: tensor<1024xf32>, %arg857: tensor<1024xf32>, %arg858: tensor<256xf32>, %arg859: tensor<256xf32>, %arg860: tensor<256xf32>, %arg861: tensor<256xf32>, %arg862: tensor<1024xf32>, %arg863: tensor<1024xf32>, %arg864: tensor<256xf32>, %arg865: tensor<256xf32>, %arg866: tensor<256xf32>, %arg867: tensor<256xf32>, %arg868: tensor<1024xf32>, %arg869: tensor<1024xf32>, %arg870: tensor<256xf32>, %arg871: tensor<256xf32>, %arg872: tensor<256xf32>, %arg873: tensor<256xf32>, %arg874: tensor<1024xf32>, %arg875: tensor<1024xf32>, %arg876: tensor<256xf32>, %arg877: tensor<256xf32>, %arg878: tensor<256xf32>, %arg879: tensor<256xf32>, %arg880: tensor<1024xf32>, %arg881: tensor<1024xf32>, %arg882: tensor<256xf32>, %arg883: tensor<256xf32>, %arg884: tensor<256xf32>, %arg885: tensor<256xf32>, %arg886: tensor<1024xf32>, %arg887: tensor<1024xf32>, %arg888: tensor<256xf32>, %arg889: tensor<256xf32>, %arg890: tensor<256xf32>, %arg891: tensor<256xf32>, %arg892: tensor<1024xf32>, %arg893: tensor<1024xf32>, %arg894: tensor<256xf32>, %arg895: tensor<256xf32>, %arg896: tensor<256xf32>, %arg897: tensor<256xf32>, %arg898: tensor<1024xf32>, %arg899: tensor<1024xf32>, %arg900: tensor<256xf32>, %arg901: tensor<256xf32>, %arg902: tensor<256xf32>, %arg903: tensor<256xf32>, %arg904: tensor<1024xf32>, %arg905: tensor<1024xf32>, %arg906: tensor<256xf32>, %arg907: tensor<256xf32>, %arg908: tensor<256xf32>, %arg909: tensor<256xf32>, %arg910: tensor<1024xf32>, %arg911: tensor<1024xf32>, %arg912: tensor<64xf32>, %arg913: tensor<64xf32>, %arg914: tensor<64xf32>, %arg915: tensor<64xf32>, %arg916: tensor<256xf32>, %arg917: tensor<256xf32>, %arg918: tensor<256xf32>, %arg919: tensor<256xf32>, %arg920: tensor<256xf32>, %arg921: tensor<256xf32>, %arg922: tensor<1024xf32>, %arg923: tensor<1024xf32>, %arg924: tensor<256xf32>, %arg925: tensor<256xf32>, %arg926: tensor<256xf32>, %arg927: tensor<256xf32>, %arg928: tensor<1024xf32>, %arg929: tensor<1024xf32>, %arg930: tensor<256xf32>, %arg931: tensor<256xf32>, %arg932: tensor<256xf32>, %arg933: tensor<256xf32>, %arg934: tensor<1024xf32>, %arg935: tensor<1024xf32>, %arg936: tensor<256xf32>, %arg937: tensor<256xf32>, %arg938: tensor<256xf32>, %arg939: tensor<256xf32>, %arg940: tensor<1024xf32>, %arg941: tensor<1024xf32>, %arg942: tensor<256xf32>, %arg943: tensor<256xf32>, %arg944: tensor<256xf32>, %arg945: tensor<256xf32>, %arg946: tensor<1024xf32>, %arg947: tensor<1024xf32>, %arg948: tensor<256xf32>, %arg949: tensor<256xf32>, %arg950: tensor<256xf32>, %arg951: tensor<256xf32>, %arg952: tensor<1024xf32>, %arg953: tensor<1024xf32>, %arg954: tensor<256xf32>, %arg955: tensor<256xf32>, %arg956: tensor<256xf32>, %arg957: tensor<256xf32>, %arg958: tensor<1024xf32>, %arg959: tensor<1024xf32>, %arg960: tensor<256xf32>, %arg961: tensor<256xf32>, %arg962: tensor<256xf32>, %arg963: tensor<256xf32>, %arg964: tensor<1024xf32>, %arg965: tensor<1024xf32>, %arg966: tensor<256xf32>, %arg967: tensor<256xf32>, %arg968: tensor<256xf32>, %arg969: tensor<256xf32>, %arg970: tensor<1024xf32>, %arg971: tensor<1024xf32>, %arg972: tensor<256xf32>, %arg973: tensor<256xf32>, %arg974: tensor<256xf32>, %arg975: tensor<256xf32>, %arg976: tensor<1024xf32>, %arg977: tensor<1024xf32>, %arg978: tensor<128xf32>, %arg979: tensor<128xf32>, %arg980: tensor<128xf32>, %arg981: tensor<128xf32>, %arg982: tensor<512xf32>, %arg983: tensor<512xf32>, %arg984: tensor<512xf32>, %arg985: tensor<512xf32>, %arg986: tensor<512xf32>, %arg987: tensor<512xf32>, %arg988: tensor<512xf32>, %arg989: tensor<512xf32>, %arg990: tensor<2048xf32>, %arg991: tensor<2048xf32>, %arg992: tensor<2048xf32>, %arg993: tensor<2048xf32>, %arg994: tensor<512xf32>, %arg995: tensor<512xf32>, %arg996: tensor<512xf32>, %arg997: tensor<512xf32>, %arg998: tensor<2048xf32>, %arg999: tensor<2048xf32>, %arg1000: tensor<512xf32>, %arg1001: tensor<512xf32>, %arg1002: tensor<512xf32>, %arg1003: tensor<512xf32>, %arg1004: tensor<2048xf32>, %arg1005: tensor<2048xf32>, %arg1006: tensor<128xf32>, %arg1007: tensor<128xf32>, %arg1008: tensor<128xf32>, %arg1009: tensor<128xf32>, %arg1010: tensor<512xf32>, %arg1011: tensor<512xf32>, %arg1012: tensor<128xf32>, %arg1013: tensor<128xf32>, %arg1014: tensor<128xf32>, %arg1015: tensor<128xf32>, %arg1016: tensor<512xf32>, %arg1017: tensor<512xf32>, %arg1018: tensor<128xf32>, %arg1019: tensor<128xf32>, %arg1020: tensor<128xf32>, %arg1021: tensor<128xf32>, %arg1022: tensor<512xf32>, %arg1023: tensor<512xf32>, %arg1024: tensor<256xf32>, %arg1025: tensor<256xf32>, %arg1026: tensor<256xf32>, %arg1027: tensor<256xf32>, %arg1028: tensor<1024xf32>, %arg1029: tensor<1024xf32>, %arg1030: tensor<1024xf32>, %arg1031: tensor<1024xf32>, %arg1032: tensor<256xf32>, %arg1033: tensor<256xf32>, %arg1034: tensor<256xf32>, %arg1035: tensor<256xf32>, %arg1036: tensor<1024xf32>, %arg1037: tensor<1024xf32>, %arg1038: tensor<256xf32>, %arg1039: tensor<256xf32>, %arg1040: tensor<256xf32>, %arg1041: tensor<256xf32>, %arg1042: tensor<1024xf32>, %arg1043: tensor<1024xf32>, %arg1044: tensor<64xf32>, %arg1045: tensor<64xf32>, %arg1046: tensor<1x256x224x224x3xf16>, %arg1047: tensor<1x256xi32>) {
      %1 = stablehlo.reshape %arg1046 : (tensor<1x256x224x224x3xf16>) -> tensor<256x224x224x3xf16>
      %2 = stablehlo.reshape %arg1047 : (tensor<1x256xi32>) -> tensor<256xi32>
      %3 = stablehlo.convert %arg837 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf16>
      %4 = stablehlo.convolution(%1, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf16>, tensor<7x7x3x64xf16>) -> tensor<256x112x112x64xf16>
      %5 = stablehlo.broadcast_in_dim %arg1044, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %6 = stablehlo.broadcast_in_dim %arg1045, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %7 = stablehlo.convert %4 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
      %8 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %9 = stablehlo.subtract %7, %8 : tensor<256x112x112x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %11 = stablehlo.add %6, %10 : tensor<1x1x1x64xf32>
      %12 = stablehlo.rsqrt %11 : tensor<1x1x1x64xf32>
      %13 = stablehlo.reshape %arg836 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %14 = stablehlo.multiply %12, %13 : tensor<1x1x1x64xf32>
      %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %16 = stablehlo.multiply %9, %15 : tensor<256x112x112x64xf32>
      %17 = stablehlo.reshape %arg835 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %19 = stablehlo.add %16, %18 : tensor<256x112x112x64xf32>
      %20 = stablehlo.convert %19 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %21 = func.call @relu(%20) : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16>
      %cst_0 = stablehlo.constant dense<0xFC00> : tensor<f16>
      %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f16>) -> tensor<f16>
      %23 = "stablehlo.reduce_window"(%21, %22) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg1048: tensor<f16>, %arg1049: tensor<f16>):
        %2050 = stablehlo.maximum %arg1048, %arg1049 : tensor<f16>
        stablehlo.return %2050 : tensor<f16>
      }) : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x56x56x64xf16>
      %24 = stablehlo.convert %arg530 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xf16>
      %25 = stablehlo.convolution(%23, %24) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x64xf16>) -> tensor<256x56x56x64xf16>
      %26 = stablehlo.broadcast_in_dim %arg838, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %27 = stablehlo.broadcast_in_dim %arg839, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %28 = stablehlo.convert %25 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %29 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %30 = stablehlo.subtract %28, %29 : tensor<256x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %31 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %32 = stablehlo.add %27, %31 : tensor<1x1x1x64xf32>
      %33 = stablehlo.rsqrt %32 : tensor<1x1x1x64xf32>
      %34 = stablehlo.reshape %arg525 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %35 = stablehlo.multiply %33, %34 : tensor<1x1x1x64xf32>
      %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %37 = stablehlo.multiply %30, %36 : tensor<256x56x56x64xf32>
      %38 = stablehlo.reshape %arg524 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %40 = stablehlo.add %37, %39 : tensor<256x56x56x64xf32>
      %41 = stablehlo.convert %40 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %42 = func.call @relu_13(%41) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %43 = stablehlo.convert %arg531 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %44 = stablehlo.convolution(%42, %43) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %45 = stablehlo.broadcast_in_dim %arg840, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %46 = stablehlo.broadcast_in_dim %arg841, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %47 = stablehlo.convert %44 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %48 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %49 = stablehlo.subtract %47, %48 : tensor<256x56x56x64xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %50 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %51 = stablehlo.add %46, %50 : tensor<1x1x1x64xf32>
      %52 = stablehlo.rsqrt %51 : tensor<1x1x1x64xf32>
      %53 = stablehlo.reshape %arg527 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %54 = stablehlo.multiply %52, %53 : tensor<1x1x1x64xf32>
      %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %56 = stablehlo.multiply %49, %55 : tensor<256x56x56x64xf32>
      %57 = stablehlo.reshape %arg526 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %59 = stablehlo.add %56, %58 : tensor<256x56x56x64xf32>
      %60 = stablehlo.convert %59 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %61 = func.call @relu_13(%60) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %62 = stablehlo.convert %arg532 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %63 = stablehlo.convolution(%61, %62) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %64 = stablehlo.broadcast_in_dim %arg842, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %65 = stablehlo.broadcast_in_dim %arg843, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %66 = stablehlo.convert %63 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %67 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %68 = stablehlo.subtract %66, %67 : tensor<256x56x56x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %69 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %70 = stablehlo.add %65, %69 : tensor<1x1x1x256xf32>
      %71 = stablehlo.rsqrt %70 : tensor<1x1x1x256xf32>
      %72 = stablehlo.reshape %arg529 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %73 = stablehlo.multiply %71, %72 : tensor<1x1x1x256xf32>
      %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %75 = stablehlo.multiply %68, %74 : tensor<256x56x56x256xf32>
      %76 = stablehlo.reshape %arg528 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %78 = stablehlo.add %75, %77 : tensor<256x56x56x256xf32>
      %79 = stablehlo.convert %78 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %80 = stablehlo.convert %arg533 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %81 = stablehlo.convolution(%23, %80) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %82 = stablehlo.broadcast_in_dim %arg844, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %83 = stablehlo.broadcast_in_dim %arg845, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %84 = stablehlo.convert %81 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %85 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %86 = stablehlo.subtract %84, %85 : tensor<256x56x56x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %87 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %88 = stablehlo.add %83, %87 : tensor<1x1x1x256xf32>
      %89 = stablehlo.rsqrt %88 : tensor<1x1x1x256xf32>
      %90 = stablehlo.reshape %arg535 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %91 = stablehlo.multiply %89, %90 : tensor<1x1x1x256xf32>
      %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %93 = stablehlo.multiply %86, %92 : tensor<256x56x56x256xf32>
      %94 = stablehlo.reshape %arg534 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %96 = stablehlo.add %93, %95 : tensor<256x56x56x256xf32>
      %97 = stablehlo.convert %96 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %98 = stablehlo.add %97, %79 : tensor<256x56x56x256xf16>
      %99 = func.call @relu_32(%98) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
      %100 = stablehlo.convert %arg542 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
      %101 = stablehlo.convolution(%99, %100) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<256x56x56x64xf16>
      %102 = stablehlo.broadcast_in_dim %arg846, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %103 = stablehlo.broadcast_in_dim %arg847, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %104 = stablehlo.convert %101 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %105 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %106 = stablehlo.subtract %104, %105 : tensor<256x56x56x64xf32>
      %cst_5 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %107 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %108 = stablehlo.add %103, %107 : tensor<1x1x1x64xf32>
      %109 = stablehlo.rsqrt %108 : tensor<1x1x1x64xf32>
      %110 = stablehlo.reshape %arg537 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %111 = stablehlo.multiply %109, %110 : tensor<1x1x1x64xf32>
      %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %113 = stablehlo.multiply %106, %112 : tensor<256x56x56x64xf32>
      %114 = stablehlo.reshape %arg536 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %115 = stablehlo.broadcast_in_dim %114, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %116 = stablehlo.add %113, %115 : tensor<256x56x56x64xf32>
      %117 = stablehlo.convert %116 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %118 = func.call @relu_13(%117) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %119 = stablehlo.convert %arg543 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %120 = stablehlo.convolution(%118, %119) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %121 = stablehlo.broadcast_in_dim %arg848, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %122 = stablehlo.broadcast_in_dim %arg849, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %123 = stablehlo.convert %120 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %124 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %125 = stablehlo.subtract %123, %124 : tensor<256x56x56x64xf32>
      %cst_6 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %126 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %127 = stablehlo.add %122, %126 : tensor<1x1x1x64xf32>
      %128 = stablehlo.rsqrt %127 : tensor<1x1x1x64xf32>
      %129 = stablehlo.reshape %arg539 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %130 = stablehlo.multiply %128, %129 : tensor<1x1x1x64xf32>
      %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %132 = stablehlo.multiply %125, %131 : tensor<256x56x56x64xf32>
      %133 = stablehlo.reshape %arg538 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %134 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %135 = stablehlo.add %132, %134 : tensor<256x56x56x64xf32>
      %136 = stablehlo.convert %135 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %137 = func.call @relu_13(%136) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %138 = stablehlo.convert %arg544 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %139 = stablehlo.convolution(%137, %138) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %140 = stablehlo.broadcast_in_dim %arg850, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %141 = stablehlo.broadcast_in_dim %arg851, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %142 = stablehlo.convert %139 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %143 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %144 = stablehlo.subtract %142, %143 : tensor<256x56x56x256xf32>
      %cst_7 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %145 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %146 = stablehlo.add %141, %145 : tensor<1x1x1x256xf32>
      %147 = stablehlo.rsqrt %146 : tensor<1x1x1x256xf32>
      %148 = stablehlo.reshape %arg541 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %149 = stablehlo.multiply %147, %148 : tensor<1x1x1x256xf32>
      %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %151 = stablehlo.multiply %144, %150 : tensor<256x56x56x256xf32>
      %152 = stablehlo.reshape %arg540 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %154 = stablehlo.add %151, %153 : tensor<256x56x56x256xf32>
      %155 = stablehlo.convert %154 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %156 = stablehlo.add %99, %155 : tensor<256x56x56x256xf16>
      %157 = func.call @relu_32(%156) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
      %158 = stablehlo.convert %arg641 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
      %159 = stablehlo.convolution(%157, %158) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<256x56x56x64xf16>
      %160 = stablehlo.broadcast_in_dim %arg912, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %161 = stablehlo.broadcast_in_dim %arg913, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %162 = stablehlo.convert %159 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %163 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %164 = stablehlo.subtract %162, %163 : tensor<256x56x56x64xf32>
      %cst_8 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %165 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %166 = stablehlo.add %161, %165 : tensor<1x1x1x64xf32>
      %167 = stablehlo.rsqrt %166 : tensor<1x1x1x64xf32>
      %168 = stablehlo.reshape %arg636 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %169 = stablehlo.multiply %167, %168 : tensor<1x1x1x64xf32>
      %170 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %171 = stablehlo.multiply %164, %170 : tensor<256x56x56x64xf32>
      %172 = stablehlo.reshape %arg635 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %173 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %174 = stablehlo.add %171, %173 : tensor<256x56x56x64xf32>
      %175 = stablehlo.convert %174 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %176 = func.call @relu_13(%175) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %177 = stablehlo.convert %arg642 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %178 = stablehlo.convolution(%176, %177) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %179 = stablehlo.broadcast_in_dim %arg914, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %180 = stablehlo.broadcast_in_dim %arg915, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %181 = stablehlo.convert %178 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %182 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %183 = stablehlo.subtract %181, %182 : tensor<256x56x56x64xf32>
      %cst_9 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %184 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %185 = stablehlo.add %180, %184 : tensor<1x1x1x64xf32>
      %186 = stablehlo.rsqrt %185 : tensor<1x1x1x64xf32>
      %187 = stablehlo.reshape %arg638 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %188 = stablehlo.multiply %186, %187 : tensor<1x1x1x64xf32>
      %189 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %190 = stablehlo.multiply %183, %189 : tensor<256x56x56x64xf32>
      %191 = stablehlo.reshape %arg637 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %193 = stablehlo.add %190, %192 : tensor<256x56x56x64xf32>
      %194 = stablehlo.convert %193 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %195 = func.call @relu_13(%194) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %196 = stablehlo.convert %arg643 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %197 = stablehlo.convolution(%195, %196) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %198 = stablehlo.broadcast_in_dim %arg916, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %199 = stablehlo.broadcast_in_dim %arg917, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %200 = stablehlo.convert %197 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %201 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %202 = stablehlo.subtract %200, %201 : tensor<256x56x56x256xf32>
      %cst_10 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %204 = stablehlo.add %199, %203 : tensor<1x1x1x256xf32>
      %205 = stablehlo.rsqrt %204 : tensor<1x1x1x256xf32>
      %206 = stablehlo.reshape %arg640 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %207 = stablehlo.multiply %205, %206 : tensor<1x1x1x256xf32>
      %208 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %209 = stablehlo.multiply %202, %208 : tensor<256x56x56x256xf32>
      %210 = stablehlo.reshape %arg639 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %212 = stablehlo.add %209, %211 : tensor<256x56x56x256xf32>
      %213 = stablehlo.convert %212 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %214 = stablehlo.add %157, %213 : tensor<256x56x56x256xf16>
      %215 = func.call @relu_32(%214) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
      %216 = stablehlo.convert %arg740 : (tensor<1x1x256x128xf32>) -> tensor<1x1x256x128xf16>
      %217 = stablehlo.convolution(%215, %216) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x128xf16>) -> tensor<256x56x56x128xf16>
      %218 = stablehlo.broadcast_in_dim %arg978, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %219 = stablehlo.broadcast_in_dim %arg979, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %220 = stablehlo.convert %217 : (tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf32>
      %221 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
      %222 = stablehlo.subtract %220, %221 : tensor<256x56x56x128xf32>
      %cst_11 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %223 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %224 = stablehlo.add %219, %223 : tensor<1x1x1x128xf32>
      %225 = stablehlo.rsqrt %224 : tensor<1x1x1x128xf32>
      %226 = stablehlo.reshape %arg735 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<1x1x1x128xf32>
      %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
      %229 = stablehlo.multiply %222, %228 : tensor<256x56x56x128xf32>
      %230 = stablehlo.reshape %arg734 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %231 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
      %232 = stablehlo.add %229, %231 : tensor<256x56x56x128xf32>
      %233 = stablehlo.convert %232 : (tensor<256x56x56x128xf32>) -> tensor<256x56x56x128xf16>
      %234 = func.call @relu_55(%233) : (tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf16>
      %235 = stablehlo.convert %arg741 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %236 = stablehlo.convolution(%234, %235) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %237 = stablehlo.broadcast_in_dim %arg980, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %238 = stablehlo.broadcast_in_dim %arg981, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %239 = stablehlo.convert %236 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %240 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %241 = stablehlo.subtract %239, %240 : tensor<256x28x28x128xf32>
      %cst_12 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %242 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %243 = stablehlo.add %238, %242 : tensor<1x1x1x128xf32>
      %244 = stablehlo.rsqrt %243 : tensor<1x1x1x128xf32>
      %245 = stablehlo.reshape %arg737 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %246 = stablehlo.multiply %244, %245 : tensor<1x1x1x128xf32>
      %247 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %248 = stablehlo.multiply %241, %247 : tensor<256x28x28x128xf32>
      %249 = stablehlo.reshape %arg736 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %251 = stablehlo.add %248, %250 : tensor<256x28x28x128xf32>
      %252 = stablehlo.convert %251 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %253 = func.call @relu_65(%252) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %254 = stablehlo.convert %arg742 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %255 = stablehlo.convolution(%253, %254) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %256 = stablehlo.broadcast_in_dim %arg982, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %257 = stablehlo.broadcast_in_dim %arg983, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %258 = stablehlo.convert %255 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %259 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %260 = stablehlo.subtract %258, %259 : tensor<256x28x28x512xf32>
      %cst_13 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %261 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %262 = stablehlo.add %257, %261 : tensor<1x1x1x512xf32>
      %263 = stablehlo.rsqrt %262 : tensor<1x1x1x512xf32>
      %264 = stablehlo.reshape %arg739 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<1x1x1x512xf32>
      %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %267 = stablehlo.multiply %260, %266 : tensor<256x28x28x512xf32>
      %268 = stablehlo.reshape %arg738 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %270 = stablehlo.add %267, %269 : tensor<256x28x28x512xf32>
      %271 = stablehlo.convert %270 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %272 = stablehlo.convert %arg743 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf16>
      %273 = stablehlo.convolution(%215, %272) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x512xf16>) -> tensor<256x28x28x512xf16>
      %274 = stablehlo.broadcast_in_dim %arg984, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %275 = stablehlo.broadcast_in_dim %arg985, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %276 = stablehlo.convert %273 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %277 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %278 = stablehlo.subtract %276, %277 : tensor<256x28x28x512xf32>
      %cst_14 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %280 = stablehlo.add %275, %279 : tensor<1x1x1x512xf32>
      %281 = stablehlo.rsqrt %280 : tensor<1x1x1x512xf32>
      %282 = stablehlo.reshape %arg745 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %283 = stablehlo.multiply %281, %282 : tensor<1x1x1x512xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %285 = stablehlo.multiply %278, %284 : tensor<256x28x28x512xf32>
      %286 = stablehlo.reshape %arg744 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %288 = stablehlo.add %285, %287 : tensor<256x28x28x512xf32>
      %289 = stablehlo.convert %288 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %290 = stablehlo.add %289, %271 : tensor<256x28x28x512xf16>
      %291 = func.call @relu_83(%290) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %292 = stablehlo.convert %arg782 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %293 = stablehlo.convolution(%291, %292) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %294 = stablehlo.broadcast_in_dim %arg1006, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg1007, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %296 = stablehlo.convert %293 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %297 = stablehlo.broadcast_in_dim %294, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %298 = stablehlo.subtract %296, %297 : tensor<256x28x28x128xf32>
      %cst_15 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %300 = stablehlo.add %295, %299 : tensor<1x1x1x128xf32>
      %301 = stablehlo.rsqrt %300 : tensor<1x1x1x128xf32>
      %302 = stablehlo.reshape %arg777 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<1x1x1x128xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %305 = stablehlo.multiply %298, %304 : tensor<256x28x28x128xf32>
      %306 = stablehlo.reshape %arg776 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %308 = stablehlo.add %305, %307 : tensor<256x28x28x128xf32>
      %309 = stablehlo.convert %308 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %310 = func.call @relu_65(%309) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %311 = stablehlo.convert %arg783 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %312 = stablehlo.convolution(%310, %311) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %313 = stablehlo.broadcast_in_dim %arg1008, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %314 = stablehlo.broadcast_in_dim %arg1009, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %315 = stablehlo.convert %312 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %316 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %317 = stablehlo.subtract %315, %316 : tensor<256x28x28x128xf32>
      %cst_16 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %319 = stablehlo.add %314, %318 : tensor<1x1x1x128xf32>
      %320 = stablehlo.rsqrt %319 : tensor<1x1x1x128xf32>
      %321 = stablehlo.reshape %arg779 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<1x1x1x128xf32>
      %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %324 = stablehlo.multiply %317, %323 : tensor<256x28x28x128xf32>
      %325 = stablehlo.reshape %arg778 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %327 = stablehlo.add %324, %326 : tensor<256x28x28x128xf32>
      %328 = stablehlo.convert %327 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %329 = func.call @relu_65(%328) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %330 = stablehlo.convert %arg784 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %331 = stablehlo.convolution(%329, %330) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %332 = stablehlo.broadcast_in_dim %arg1010, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %333 = stablehlo.broadcast_in_dim %arg1011, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %334 = stablehlo.convert %331 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %335 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %336 = stablehlo.subtract %334, %335 : tensor<256x28x28x512xf32>
      %cst_17 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %338 = stablehlo.add %333, %337 : tensor<1x1x1x512xf32>
      %339 = stablehlo.rsqrt %338 : tensor<1x1x1x512xf32>
      %340 = stablehlo.reshape %arg781 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<1x1x1x512xf32>
      %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %343 = stablehlo.multiply %336, %342 : tensor<256x28x28x512xf32>
      %344 = stablehlo.reshape %arg780 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %346 = stablehlo.add %343, %345 : tensor<256x28x28x512xf32>
      %347 = stablehlo.convert %346 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %348 = stablehlo.add %291, %347 : tensor<256x28x28x512xf16>
      %349 = func.call @relu_83(%348) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %350 = stablehlo.convert %arg791 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %351 = stablehlo.convolution(%349, %350) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %352 = stablehlo.broadcast_in_dim %arg1012, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %353 = stablehlo.broadcast_in_dim %arg1013, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %354 = stablehlo.convert %351 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %355 = stablehlo.broadcast_in_dim %352, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %356 = stablehlo.subtract %354, %355 : tensor<256x28x28x128xf32>
      %cst_18 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %358 = stablehlo.add %353, %357 : tensor<1x1x1x128xf32>
      %359 = stablehlo.rsqrt %358 : tensor<1x1x1x128xf32>
      %360 = stablehlo.reshape %arg786 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %361 = stablehlo.multiply %359, %360 : tensor<1x1x1x128xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %363 = stablehlo.multiply %356, %362 : tensor<256x28x28x128xf32>
      %364 = stablehlo.reshape %arg785 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %366 = stablehlo.add %363, %365 : tensor<256x28x28x128xf32>
      %367 = stablehlo.convert %366 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %368 = func.call @relu_65(%367) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %369 = stablehlo.convert %arg792 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %370 = stablehlo.convolution(%368, %369) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %371 = stablehlo.broadcast_in_dim %arg1014, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %372 = stablehlo.broadcast_in_dim %arg1015, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %373 = stablehlo.convert %370 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %374 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %375 = stablehlo.subtract %373, %374 : tensor<256x28x28x128xf32>
      %cst_19 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %377 = stablehlo.add %372, %376 : tensor<1x1x1x128xf32>
      %378 = stablehlo.rsqrt %377 : tensor<1x1x1x128xf32>
      %379 = stablehlo.reshape %arg788 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %380 = stablehlo.multiply %378, %379 : tensor<1x1x1x128xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %382 = stablehlo.multiply %375, %381 : tensor<256x28x28x128xf32>
      %383 = stablehlo.reshape %arg787 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %385 = stablehlo.add %382, %384 : tensor<256x28x28x128xf32>
      %386 = stablehlo.convert %385 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %387 = func.call @relu_65(%386) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %388 = stablehlo.convert %arg793 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %389 = stablehlo.convolution(%387, %388) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %390 = stablehlo.broadcast_in_dim %arg1016, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %391 = stablehlo.broadcast_in_dim %arg1017, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %392 = stablehlo.convert %389 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %393 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %394 = stablehlo.subtract %392, %393 : tensor<256x28x28x512xf32>
      %cst_20 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %396 = stablehlo.add %391, %395 : tensor<1x1x1x512xf32>
      %397 = stablehlo.rsqrt %396 : tensor<1x1x1x512xf32>
      %398 = stablehlo.reshape %arg790 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %399 = stablehlo.multiply %397, %398 : tensor<1x1x1x512xf32>
      %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %401 = stablehlo.multiply %394, %400 : tensor<256x28x28x512xf32>
      %402 = stablehlo.reshape %arg789 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %403 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %404 = stablehlo.add %401, %403 : tensor<256x28x28x512xf32>
      %405 = stablehlo.convert %404 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %406 = stablehlo.add %349, %405 : tensor<256x28x28x512xf16>
      %407 = func.call @relu_83(%406) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %408 = stablehlo.convert %arg800 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %409 = stablehlo.convolution(%407, %408) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %410 = stablehlo.broadcast_in_dim %arg1018, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %411 = stablehlo.broadcast_in_dim %arg1019, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %412 = stablehlo.convert %409 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %413 = stablehlo.broadcast_in_dim %410, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %414 = stablehlo.subtract %412, %413 : tensor<256x28x28x128xf32>
      %cst_21 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %416 = stablehlo.add %411, %415 : tensor<1x1x1x128xf32>
      %417 = stablehlo.rsqrt %416 : tensor<1x1x1x128xf32>
      %418 = stablehlo.reshape %arg795 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x1x1x128xf32>
      %420 = stablehlo.broadcast_in_dim %419, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %421 = stablehlo.multiply %414, %420 : tensor<256x28x28x128xf32>
      %422 = stablehlo.reshape %arg794 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %424 = stablehlo.add %421, %423 : tensor<256x28x28x128xf32>
      %425 = stablehlo.convert %424 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %426 = func.call @relu_65(%425) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %427 = stablehlo.convert %arg801 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %428 = stablehlo.convolution(%426, %427) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %429 = stablehlo.broadcast_in_dim %arg1020, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %430 = stablehlo.broadcast_in_dim %arg1021, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %431 = stablehlo.convert %428 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %432 = stablehlo.broadcast_in_dim %429, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %433 = stablehlo.subtract %431, %432 : tensor<256x28x28x128xf32>
      %cst_22 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %435 = stablehlo.add %430, %434 : tensor<1x1x1x128xf32>
      %436 = stablehlo.rsqrt %435 : tensor<1x1x1x128xf32>
      %437 = stablehlo.reshape %arg797 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<1x1x1x128xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %440 = stablehlo.multiply %433, %439 : tensor<256x28x28x128xf32>
      %441 = stablehlo.reshape %arg796 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %443 = stablehlo.add %440, %442 : tensor<256x28x28x128xf32>
      %444 = stablehlo.convert %443 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %445 = func.call @relu_65(%444) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %446 = stablehlo.convert %arg802 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %447 = stablehlo.convolution(%445, %446) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %448 = stablehlo.broadcast_in_dim %arg1022, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %449 = stablehlo.broadcast_in_dim %arg1023, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %450 = stablehlo.convert %447 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %451 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %452 = stablehlo.subtract %450, %451 : tensor<256x28x28x512xf32>
      %cst_23 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %454 = stablehlo.add %449, %453 : tensor<1x1x1x512xf32>
      %455 = stablehlo.rsqrt %454 : tensor<1x1x1x512xf32>
      %456 = stablehlo.reshape %arg799 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<1x1x1x512xf32>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %459 = stablehlo.multiply %452, %458 : tensor<256x28x28x512xf32>
      %460 = stablehlo.reshape %arg798 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %462 = stablehlo.add %459, %461 : tensor<256x28x28x512xf32>
      %463 = stablehlo.convert %462 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %464 = stablehlo.add %407, %463 : tensor<256x28x28x512xf16>
      %465 = func.call @relu_83(%464) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %466 = stablehlo.convert %arg809 : (tensor<1x1x512x256xf32>) -> tensor<1x1x512x256xf16>
      %467 = stablehlo.convolution(%465, %466) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x256xf16>) -> tensor<256x28x28x256xf16>
      %468 = stablehlo.broadcast_in_dim %arg1024, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %469 = stablehlo.broadcast_in_dim %arg1025, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %470 = stablehlo.convert %467 : (tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf32>
      %471 = stablehlo.broadcast_in_dim %468, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
      %472 = stablehlo.subtract %470, %471 : tensor<256x28x28x256xf32>
      %cst_24 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %474 = stablehlo.add %469, %473 : tensor<1x1x1x256xf32>
      %475 = stablehlo.rsqrt %474 : tensor<1x1x1x256xf32>
      %476 = stablehlo.reshape %arg804 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %477 = stablehlo.multiply %475, %476 : tensor<1x1x1x256xf32>
      %478 = stablehlo.broadcast_in_dim %477, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
      %479 = stablehlo.multiply %472, %478 : tensor<256x28x28x256xf32>
      %480 = stablehlo.reshape %arg803 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
      %482 = stablehlo.add %479, %481 : tensor<256x28x28x256xf32>
      %483 = stablehlo.convert %482 : (tensor<256x28x28x256xf32>) -> tensor<256x28x28x256xf16>
      %484 = func.call @relu_105(%483) : (tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf16>
      %485 = stablehlo.convert %arg810 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %486 = stablehlo.convolution(%484, %485) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %487 = stablehlo.broadcast_in_dim %arg1026, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %488 = stablehlo.broadcast_in_dim %arg1027, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %489 = stablehlo.convert %486 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %490 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %491 = stablehlo.subtract %489, %490 : tensor<256x14x14x256xf32>
      %cst_25 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %493 = stablehlo.add %488, %492 : tensor<1x1x1x256xf32>
      %494 = stablehlo.rsqrt %493 : tensor<1x1x1x256xf32>
      %495 = stablehlo.reshape %arg806 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %496 = stablehlo.multiply %494, %495 : tensor<1x1x1x256xf32>
      %497 = stablehlo.broadcast_in_dim %496, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %498 = stablehlo.multiply %491, %497 : tensor<256x14x14x256xf32>
      %499 = stablehlo.reshape %arg805 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %500 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %501 = stablehlo.add %498, %500 : tensor<256x14x14x256xf32>
      %502 = stablehlo.convert %501 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %503 = func.call @relu_115(%502) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %504 = stablehlo.convert %arg811 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %505 = stablehlo.convolution(%503, %504) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %506 = stablehlo.broadcast_in_dim %arg1028, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %507 = stablehlo.broadcast_in_dim %arg1029, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %508 = stablehlo.convert %505 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %509 = stablehlo.broadcast_in_dim %506, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %510 = stablehlo.subtract %508, %509 : tensor<256x14x14x1024xf32>
      %cst_26 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %512 = stablehlo.add %507, %511 : tensor<1x1x1x1024xf32>
      %513 = stablehlo.rsqrt %512 : tensor<1x1x1x1024xf32>
      %514 = stablehlo.reshape %arg808 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %515 = stablehlo.multiply %513, %514 : tensor<1x1x1x1024xf32>
      %516 = stablehlo.broadcast_in_dim %515, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %517 = stablehlo.multiply %510, %516 : tensor<256x14x14x1024xf32>
      %518 = stablehlo.reshape %arg807 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %519 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %520 = stablehlo.add %517, %519 : tensor<256x14x14x1024xf32>
      %521 = stablehlo.convert %520 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %522 = stablehlo.convert %arg812 : (tensor<1x1x512x1024xf32>) -> tensor<1x1x512x1024xf16>
      %523 = stablehlo.convolution(%465, %522) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x1024xf16>) -> tensor<256x14x14x1024xf16>
      %524 = stablehlo.broadcast_in_dim %arg1030, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %525 = stablehlo.broadcast_in_dim %arg1031, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %526 = stablehlo.convert %523 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %527 = stablehlo.broadcast_in_dim %524, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %528 = stablehlo.subtract %526, %527 : tensor<256x14x14x1024xf32>
      %cst_27 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %530 = stablehlo.add %525, %529 : tensor<1x1x1x1024xf32>
      %531 = stablehlo.rsqrt %530 : tensor<1x1x1x1024xf32>
      %532 = stablehlo.reshape %arg814 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %533 = stablehlo.multiply %531, %532 : tensor<1x1x1x1024xf32>
      %534 = stablehlo.broadcast_in_dim %533, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %535 = stablehlo.multiply %528, %534 : tensor<256x14x14x1024xf32>
      %536 = stablehlo.reshape %arg813 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %537 = stablehlo.broadcast_in_dim %536, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %538 = stablehlo.add %535, %537 : tensor<256x14x14x1024xf32>
      %539 = stablehlo.convert %538 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %540 = stablehlo.add %539, %521 : tensor<256x14x14x1024xf16>
      %541 = func.call @relu_133(%540) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %542 = stablehlo.convert %arg821 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %543 = stablehlo.convolution(%541, %542) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %544 = stablehlo.broadcast_in_dim %arg1032, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %545 = stablehlo.broadcast_in_dim %arg1033, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %546 = stablehlo.convert %543 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %547 = stablehlo.broadcast_in_dim %544, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %548 = stablehlo.subtract %546, %547 : tensor<256x14x14x256xf32>
      %cst_28 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %549 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %550 = stablehlo.add %545, %549 : tensor<1x1x1x256xf32>
      %551 = stablehlo.rsqrt %550 : tensor<1x1x1x256xf32>
      %552 = stablehlo.reshape %arg816 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %553 = stablehlo.multiply %551, %552 : tensor<1x1x1x256xf32>
      %554 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %555 = stablehlo.multiply %548, %554 : tensor<256x14x14x256xf32>
      %556 = stablehlo.reshape %arg815 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %557 = stablehlo.broadcast_in_dim %556, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %558 = stablehlo.add %555, %557 : tensor<256x14x14x256xf32>
      %559 = stablehlo.convert %558 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %560 = func.call @relu_115(%559) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %561 = stablehlo.convert %arg822 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %562 = stablehlo.convolution(%560, %561) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %563 = stablehlo.broadcast_in_dim %arg1034, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %564 = stablehlo.broadcast_in_dim %arg1035, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %565 = stablehlo.convert %562 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %566 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %567 = stablehlo.subtract %565, %566 : tensor<256x14x14x256xf32>
      %cst_29 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %568 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %569 = stablehlo.add %564, %568 : tensor<1x1x1x256xf32>
      %570 = stablehlo.rsqrt %569 : tensor<1x1x1x256xf32>
      %571 = stablehlo.reshape %arg818 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %572 = stablehlo.multiply %570, %571 : tensor<1x1x1x256xf32>
      %573 = stablehlo.broadcast_in_dim %572, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %574 = stablehlo.multiply %567, %573 : tensor<256x14x14x256xf32>
      %575 = stablehlo.reshape %arg817 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %577 = stablehlo.add %574, %576 : tensor<256x14x14x256xf32>
      %578 = stablehlo.convert %577 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %579 = func.call @relu_115(%578) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %580 = stablehlo.convert %arg823 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %581 = stablehlo.convolution(%579, %580) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %582 = stablehlo.broadcast_in_dim %arg1036, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %583 = stablehlo.broadcast_in_dim %arg1037, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %584 = stablehlo.convert %581 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %585 = stablehlo.broadcast_in_dim %582, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %586 = stablehlo.subtract %584, %585 : tensor<256x14x14x1024xf32>
      %cst_30 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %587 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %588 = stablehlo.add %583, %587 : tensor<1x1x1x1024xf32>
      %589 = stablehlo.rsqrt %588 : tensor<1x1x1x1024xf32>
      %590 = stablehlo.reshape %arg820 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %591 = stablehlo.multiply %589, %590 : tensor<1x1x1x1024xf32>
      %592 = stablehlo.broadcast_in_dim %591, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %593 = stablehlo.multiply %586, %592 : tensor<256x14x14x1024xf32>
      %594 = stablehlo.reshape %arg819 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %595 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %596 = stablehlo.add %593, %595 : tensor<256x14x14x1024xf32>
      %597 = stablehlo.convert %596 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %598 = stablehlo.add %541, %597 : tensor<256x14x14x1024xf16>
      %599 = func.call @relu_133(%598) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %600 = stablehlo.convert %arg830 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %601 = stablehlo.convolution(%599, %600) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %602 = stablehlo.broadcast_in_dim %arg1038, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %603 = stablehlo.broadcast_in_dim %arg1039, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %604 = stablehlo.convert %601 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %605 = stablehlo.broadcast_in_dim %602, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %606 = stablehlo.subtract %604, %605 : tensor<256x14x14x256xf32>
      %cst_31 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %607 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %608 = stablehlo.add %603, %607 : tensor<1x1x1x256xf32>
      %609 = stablehlo.rsqrt %608 : tensor<1x1x1x256xf32>
      %610 = stablehlo.reshape %arg825 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %611 = stablehlo.multiply %609, %610 : tensor<1x1x1x256xf32>
      %612 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %613 = stablehlo.multiply %606, %612 : tensor<256x14x14x256xf32>
      %614 = stablehlo.reshape %arg824 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %615 = stablehlo.broadcast_in_dim %614, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %616 = stablehlo.add %613, %615 : tensor<256x14x14x256xf32>
      %617 = stablehlo.convert %616 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %618 = func.call @relu_115(%617) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %619 = stablehlo.convert %arg831 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %620 = stablehlo.convolution(%618, %619) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %621 = stablehlo.broadcast_in_dim %arg1040, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %622 = stablehlo.broadcast_in_dim %arg1041, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %623 = stablehlo.convert %620 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %624 = stablehlo.broadcast_in_dim %621, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %625 = stablehlo.subtract %623, %624 : tensor<256x14x14x256xf32>
      %cst_32 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %626 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %627 = stablehlo.add %622, %626 : tensor<1x1x1x256xf32>
      %628 = stablehlo.rsqrt %627 : tensor<1x1x1x256xf32>
      %629 = stablehlo.reshape %arg827 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %630 = stablehlo.multiply %628, %629 : tensor<1x1x1x256xf32>
      %631 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %632 = stablehlo.multiply %625, %631 : tensor<256x14x14x256xf32>
      %633 = stablehlo.reshape %arg826 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %634 = stablehlo.broadcast_in_dim %633, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %635 = stablehlo.add %632, %634 : tensor<256x14x14x256xf32>
      %636 = stablehlo.convert %635 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %637 = func.call @relu_115(%636) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %638 = stablehlo.convert %arg832 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %639 = stablehlo.convolution(%637, %638) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %640 = stablehlo.broadcast_in_dim %arg1042, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %641 = stablehlo.broadcast_in_dim %arg1043, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %642 = stablehlo.convert %639 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %643 = stablehlo.broadcast_in_dim %640, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %644 = stablehlo.subtract %642, %643 : tensor<256x14x14x1024xf32>
      %cst_33 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %646 = stablehlo.add %641, %645 : tensor<1x1x1x1024xf32>
      %647 = stablehlo.rsqrt %646 : tensor<1x1x1x1024xf32>
      %648 = stablehlo.reshape %arg829 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<1x1x1x1024xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %651 = stablehlo.multiply %644, %650 : tensor<256x14x14x1024xf32>
      %652 = stablehlo.reshape %arg828 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %654 = stablehlo.add %651, %653 : tensor<256x14x14x1024xf32>
      %655 = stablehlo.convert %654 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %656 = stablehlo.add %599, %655 : tensor<256x14x14x1024xf16>
      %657 = func.call @relu_133(%656) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %658 = stablehlo.convert %arg551 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %659 = stablehlo.convolution(%657, %658) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %660 = stablehlo.broadcast_in_dim %arg852, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %661 = stablehlo.broadcast_in_dim %arg853, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %662 = stablehlo.convert %659 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %663 = stablehlo.broadcast_in_dim %660, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %664 = stablehlo.subtract %662, %663 : tensor<256x14x14x256xf32>
      %cst_34 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %665 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %666 = stablehlo.add %661, %665 : tensor<1x1x1x256xf32>
      %667 = stablehlo.rsqrt %666 : tensor<1x1x1x256xf32>
      %668 = stablehlo.reshape %arg546 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %669 = stablehlo.multiply %667, %668 : tensor<1x1x1x256xf32>
      %670 = stablehlo.broadcast_in_dim %669, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %671 = stablehlo.multiply %664, %670 : tensor<256x14x14x256xf32>
      %672 = stablehlo.reshape %arg545 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %673 = stablehlo.broadcast_in_dim %672, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %674 = stablehlo.add %671, %673 : tensor<256x14x14x256xf32>
      %675 = stablehlo.convert %674 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %676 = func.call @relu_115(%675) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %677 = stablehlo.convert %arg552 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %678 = stablehlo.convolution(%676, %677) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %679 = stablehlo.broadcast_in_dim %arg854, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %680 = stablehlo.broadcast_in_dim %arg855, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %681 = stablehlo.convert %678 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %682 = stablehlo.broadcast_in_dim %679, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %683 = stablehlo.subtract %681, %682 : tensor<256x14x14x256xf32>
      %cst_35 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %684 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %685 = stablehlo.add %680, %684 : tensor<1x1x1x256xf32>
      %686 = stablehlo.rsqrt %685 : tensor<1x1x1x256xf32>
      %687 = stablehlo.reshape %arg548 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %688 = stablehlo.multiply %686, %687 : tensor<1x1x1x256xf32>
      %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %690 = stablehlo.multiply %683, %689 : tensor<256x14x14x256xf32>
      %691 = stablehlo.reshape %arg547 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %692 = stablehlo.broadcast_in_dim %691, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %693 = stablehlo.add %690, %692 : tensor<256x14x14x256xf32>
      %694 = stablehlo.convert %693 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %695 = func.call @relu_115(%694) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %696 = stablehlo.convert %arg553 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %697 = stablehlo.convolution(%695, %696) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %698 = stablehlo.broadcast_in_dim %arg856, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %699 = stablehlo.broadcast_in_dim %arg857, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %700 = stablehlo.convert %697 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %701 = stablehlo.broadcast_in_dim %698, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %702 = stablehlo.subtract %700, %701 : tensor<256x14x14x1024xf32>
      %cst_36 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %703 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %704 = stablehlo.add %699, %703 : tensor<1x1x1x1024xf32>
      %705 = stablehlo.rsqrt %704 : tensor<1x1x1x1024xf32>
      %706 = stablehlo.reshape %arg550 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %707 = stablehlo.multiply %705, %706 : tensor<1x1x1x1024xf32>
      %708 = stablehlo.broadcast_in_dim %707, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %709 = stablehlo.multiply %702, %708 : tensor<256x14x14x1024xf32>
      %710 = stablehlo.reshape %arg549 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %711 = stablehlo.broadcast_in_dim %710, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %712 = stablehlo.add %709, %711 : tensor<256x14x14x1024xf32>
      %713 = stablehlo.convert %712 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %714 = stablehlo.add %657, %713 : tensor<256x14x14x1024xf16>
      %715 = func.call @relu_133(%714) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %716 = stablehlo.convert %arg560 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %717 = stablehlo.convolution(%715, %716) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %718 = stablehlo.broadcast_in_dim %arg858, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %719 = stablehlo.broadcast_in_dim %arg859, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %720 = stablehlo.convert %717 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %721 = stablehlo.broadcast_in_dim %718, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %722 = stablehlo.subtract %720, %721 : tensor<256x14x14x256xf32>
      %cst_37 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %723 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %724 = stablehlo.add %719, %723 : tensor<1x1x1x256xf32>
      %725 = stablehlo.rsqrt %724 : tensor<1x1x1x256xf32>
      %726 = stablehlo.reshape %arg555 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %727 = stablehlo.multiply %725, %726 : tensor<1x1x1x256xf32>
      %728 = stablehlo.broadcast_in_dim %727, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %729 = stablehlo.multiply %722, %728 : tensor<256x14x14x256xf32>
      %730 = stablehlo.reshape %arg554 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %731 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %732 = stablehlo.add %729, %731 : tensor<256x14x14x256xf32>
      %733 = stablehlo.convert %732 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %734 = func.call @relu_115(%733) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %735 = stablehlo.convert %arg561 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %736 = stablehlo.convolution(%734, %735) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %737 = stablehlo.broadcast_in_dim %arg860, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %738 = stablehlo.broadcast_in_dim %arg861, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %739 = stablehlo.convert %736 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %740 = stablehlo.broadcast_in_dim %737, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %741 = stablehlo.subtract %739, %740 : tensor<256x14x14x256xf32>
      %cst_38 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %742 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %743 = stablehlo.add %738, %742 : tensor<1x1x1x256xf32>
      %744 = stablehlo.rsqrt %743 : tensor<1x1x1x256xf32>
      %745 = stablehlo.reshape %arg557 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %746 = stablehlo.multiply %744, %745 : tensor<1x1x1x256xf32>
      %747 = stablehlo.broadcast_in_dim %746, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %748 = stablehlo.multiply %741, %747 : tensor<256x14x14x256xf32>
      %749 = stablehlo.reshape %arg556 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %750 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %751 = stablehlo.add %748, %750 : tensor<256x14x14x256xf32>
      %752 = stablehlo.convert %751 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %753 = func.call @relu_115(%752) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %754 = stablehlo.convert %arg562 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %755 = stablehlo.convolution(%753, %754) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %756 = stablehlo.broadcast_in_dim %arg862, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %757 = stablehlo.broadcast_in_dim %arg863, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %758 = stablehlo.convert %755 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %759 = stablehlo.broadcast_in_dim %756, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %760 = stablehlo.subtract %758, %759 : tensor<256x14x14x1024xf32>
      %cst_39 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %761 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %762 = stablehlo.add %757, %761 : tensor<1x1x1x1024xf32>
      %763 = stablehlo.rsqrt %762 : tensor<1x1x1x1024xf32>
      %764 = stablehlo.reshape %arg559 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %765 = stablehlo.multiply %763, %764 : tensor<1x1x1x1024xf32>
      %766 = stablehlo.broadcast_in_dim %765, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %767 = stablehlo.multiply %760, %766 : tensor<256x14x14x1024xf32>
      %768 = stablehlo.reshape %arg558 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %769 = stablehlo.broadcast_in_dim %768, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %770 = stablehlo.add %767, %769 : tensor<256x14x14x1024xf32>
      %771 = stablehlo.convert %770 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %772 = stablehlo.add %715, %771 : tensor<256x14x14x1024xf16>
      %773 = func.call @relu_133(%772) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %774 = stablehlo.convert %arg569 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %775 = stablehlo.convolution(%773, %774) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %776 = stablehlo.broadcast_in_dim %arg864, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %777 = stablehlo.broadcast_in_dim %arg865, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %778 = stablehlo.convert %775 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %779 = stablehlo.broadcast_in_dim %776, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %780 = stablehlo.subtract %778, %779 : tensor<256x14x14x256xf32>
      %cst_40 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %781 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %782 = stablehlo.add %777, %781 : tensor<1x1x1x256xf32>
      %783 = stablehlo.rsqrt %782 : tensor<1x1x1x256xf32>
      %784 = stablehlo.reshape %arg564 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %785 = stablehlo.multiply %783, %784 : tensor<1x1x1x256xf32>
      %786 = stablehlo.broadcast_in_dim %785, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %787 = stablehlo.multiply %780, %786 : tensor<256x14x14x256xf32>
      %788 = stablehlo.reshape %arg563 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %789 = stablehlo.broadcast_in_dim %788, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %790 = stablehlo.add %787, %789 : tensor<256x14x14x256xf32>
      %791 = stablehlo.convert %790 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %792 = func.call @relu_115(%791) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %793 = stablehlo.convert %arg570 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %794 = stablehlo.convolution(%792, %793) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %795 = stablehlo.broadcast_in_dim %arg866, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %796 = stablehlo.broadcast_in_dim %arg867, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %797 = stablehlo.convert %794 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %798 = stablehlo.broadcast_in_dim %795, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %799 = stablehlo.subtract %797, %798 : tensor<256x14x14x256xf32>
      %cst_41 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %800 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %801 = stablehlo.add %796, %800 : tensor<1x1x1x256xf32>
      %802 = stablehlo.rsqrt %801 : tensor<1x1x1x256xf32>
      %803 = stablehlo.reshape %arg566 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %804 = stablehlo.multiply %802, %803 : tensor<1x1x1x256xf32>
      %805 = stablehlo.broadcast_in_dim %804, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %806 = stablehlo.multiply %799, %805 : tensor<256x14x14x256xf32>
      %807 = stablehlo.reshape %arg565 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %808 = stablehlo.broadcast_in_dim %807, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %809 = stablehlo.add %806, %808 : tensor<256x14x14x256xf32>
      %810 = stablehlo.convert %809 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %811 = func.call @relu_115(%810) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %812 = stablehlo.convert %arg571 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %813 = stablehlo.convolution(%811, %812) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %814 = stablehlo.broadcast_in_dim %arg868, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %815 = stablehlo.broadcast_in_dim %arg869, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %816 = stablehlo.convert %813 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %817 = stablehlo.broadcast_in_dim %814, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %818 = stablehlo.subtract %816, %817 : tensor<256x14x14x1024xf32>
      %cst_42 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %819 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %820 = stablehlo.add %815, %819 : tensor<1x1x1x1024xf32>
      %821 = stablehlo.rsqrt %820 : tensor<1x1x1x1024xf32>
      %822 = stablehlo.reshape %arg568 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %823 = stablehlo.multiply %821, %822 : tensor<1x1x1x1024xf32>
      %824 = stablehlo.broadcast_in_dim %823, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %825 = stablehlo.multiply %818, %824 : tensor<256x14x14x1024xf32>
      %826 = stablehlo.reshape %arg567 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %827 = stablehlo.broadcast_in_dim %826, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %828 = stablehlo.add %825, %827 : tensor<256x14x14x1024xf32>
      %829 = stablehlo.convert %828 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %830 = stablehlo.add %773, %829 : tensor<256x14x14x1024xf16>
      %831 = func.call @relu_133(%830) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %832 = stablehlo.convert %arg578 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %833 = stablehlo.convolution(%831, %832) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %834 = stablehlo.broadcast_in_dim %arg870, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %835 = stablehlo.broadcast_in_dim %arg871, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %836 = stablehlo.convert %833 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %837 = stablehlo.broadcast_in_dim %834, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %838 = stablehlo.subtract %836, %837 : tensor<256x14x14x256xf32>
      %cst_43 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %839 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %840 = stablehlo.add %835, %839 : tensor<1x1x1x256xf32>
      %841 = stablehlo.rsqrt %840 : tensor<1x1x1x256xf32>
      %842 = stablehlo.reshape %arg573 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %843 = stablehlo.multiply %841, %842 : tensor<1x1x1x256xf32>
      %844 = stablehlo.broadcast_in_dim %843, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %845 = stablehlo.multiply %838, %844 : tensor<256x14x14x256xf32>
      %846 = stablehlo.reshape %arg572 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %847 = stablehlo.broadcast_in_dim %846, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %848 = stablehlo.add %845, %847 : tensor<256x14x14x256xf32>
      %849 = stablehlo.convert %848 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %850 = func.call @relu_115(%849) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %851 = stablehlo.convert %arg579 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %852 = stablehlo.convolution(%850, %851) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %853 = stablehlo.broadcast_in_dim %arg872, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %854 = stablehlo.broadcast_in_dim %arg873, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %855 = stablehlo.convert %852 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %856 = stablehlo.broadcast_in_dim %853, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %857 = stablehlo.subtract %855, %856 : tensor<256x14x14x256xf32>
      %cst_44 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %858 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %859 = stablehlo.add %854, %858 : tensor<1x1x1x256xf32>
      %860 = stablehlo.rsqrt %859 : tensor<1x1x1x256xf32>
      %861 = stablehlo.reshape %arg575 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %862 = stablehlo.multiply %860, %861 : tensor<1x1x1x256xf32>
      %863 = stablehlo.broadcast_in_dim %862, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %864 = stablehlo.multiply %857, %863 : tensor<256x14x14x256xf32>
      %865 = stablehlo.reshape %arg574 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %866 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %867 = stablehlo.add %864, %866 : tensor<256x14x14x256xf32>
      %868 = stablehlo.convert %867 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %869 = func.call @relu_115(%868) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %870 = stablehlo.convert %arg580 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %871 = stablehlo.convolution(%869, %870) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %872 = stablehlo.broadcast_in_dim %arg874, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %873 = stablehlo.broadcast_in_dim %arg875, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %874 = stablehlo.convert %871 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %875 = stablehlo.broadcast_in_dim %872, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %876 = stablehlo.subtract %874, %875 : tensor<256x14x14x1024xf32>
      %cst_45 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %877 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %878 = stablehlo.add %873, %877 : tensor<1x1x1x1024xf32>
      %879 = stablehlo.rsqrt %878 : tensor<1x1x1x1024xf32>
      %880 = stablehlo.reshape %arg577 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %881 = stablehlo.multiply %879, %880 : tensor<1x1x1x1024xf32>
      %882 = stablehlo.broadcast_in_dim %881, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %883 = stablehlo.multiply %876, %882 : tensor<256x14x14x1024xf32>
      %884 = stablehlo.reshape %arg576 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %885 = stablehlo.broadcast_in_dim %884, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %886 = stablehlo.add %883, %885 : tensor<256x14x14x1024xf32>
      %887 = stablehlo.convert %886 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %888 = stablehlo.add %831, %887 : tensor<256x14x14x1024xf16>
      %889 = func.call @relu_133(%888) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %890 = stablehlo.convert %arg587 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %891 = stablehlo.convolution(%889, %890) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %892 = stablehlo.broadcast_in_dim %arg876, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %893 = stablehlo.broadcast_in_dim %arg877, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %894 = stablehlo.convert %891 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %895 = stablehlo.broadcast_in_dim %892, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %896 = stablehlo.subtract %894, %895 : tensor<256x14x14x256xf32>
      %cst_46 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %897 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %898 = stablehlo.add %893, %897 : tensor<1x1x1x256xf32>
      %899 = stablehlo.rsqrt %898 : tensor<1x1x1x256xf32>
      %900 = stablehlo.reshape %arg582 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %901 = stablehlo.multiply %899, %900 : tensor<1x1x1x256xf32>
      %902 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %903 = stablehlo.multiply %896, %902 : tensor<256x14x14x256xf32>
      %904 = stablehlo.reshape %arg581 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %905 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %906 = stablehlo.add %903, %905 : tensor<256x14x14x256xf32>
      %907 = stablehlo.convert %906 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %908 = func.call @relu_115(%907) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %909 = stablehlo.convert %arg588 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %910 = stablehlo.convolution(%908, %909) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %911 = stablehlo.broadcast_in_dim %arg878, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %912 = stablehlo.broadcast_in_dim %arg879, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %913 = stablehlo.convert %910 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %914 = stablehlo.broadcast_in_dim %911, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %915 = stablehlo.subtract %913, %914 : tensor<256x14x14x256xf32>
      %cst_47 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %916 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %917 = stablehlo.add %912, %916 : tensor<1x1x1x256xf32>
      %918 = stablehlo.rsqrt %917 : tensor<1x1x1x256xf32>
      %919 = stablehlo.reshape %arg584 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %920 = stablehlo.multiply %918, %919 : tensor<1x1x1x256xf32>
      %921 = stablehlo.broadcast_in_dim %920, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %922 = stablehlo.multiply %915, %921 : tensor<256x14x14x256xf32>
      %923 = stablehlo.reshape %arg583 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %924 = stablehlo.broadcast_in_dim %923, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %925 = stablehlo.add %922, %924 : tensor<256x14x14x256xf32>
      %926 = stablehlo.convert %925 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %927 = func.call @relu_115(%926) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %928 = stablehlo.convert %arg589 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %929 = stablehlo.convolution(%927, %928) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %930 = stablehlo.broadcast_in_dim %arg880, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %931 = stablehlo.broadcast_in_dim %arg881, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %932 = stablehlo.convert %929 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %933 = stablehlo.broadcast_in_dim %930, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %934 = stablehlo.subtract %932, %933 : tensor<256x14x14x1024xf32>
      %cst_48 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %935 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %936 = stablehlo.add %931, %935 : tensor<1x1x1x1024xf32>
      %937 = stablehlo.rsqrt %936 : tensor<1x1x1x1024xf32>
      %938 = stablehlo.reshape %arg586 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %939 = stablehlo.multiply %937, %938 : tensor<1x1x1x1024xf32>
      %940 = stablehlo.broadcast_in_dim %939, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %941 = stablehlo.multiply %934, %940 : tensor<256x14x14x1024xf32>
      %942 = stablehlo.reshape %arg585 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %943 = stablehlo.broadcast_in_dim %942, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %944 = stablehlo.add %941, %943 : tensor<256x14x14x1024xf32>
      %945 = stablehlo.convert %944 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %946 = stablehlo.add %889, %945 : tensor<256x14x14x1024xf16>
      %947 = func.call @relu_133(%946) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %948 = stablehlo.convert %arg596 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %949 = stablehlo.convolution(%947, %948) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %950 = stablehlo.broadcast_in_dim %arg882, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %951 = stablehlo.broadcast_in_dim %arg883, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %952 = stablehlo.convert %949 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %953 = stablehlo.broadcast_in_dim %950, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %954 = stablehlo.subtract %952, %953 : tensor<256x14x14x256xf32>
      %cst_49 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %955 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %956 = stablehlo.add %951, %955 : tensor<1x1x1x256xf32>
      %957 = stablehlo.rsqrt %956 : tensor<1x1x1x256xf32>
      %958 = stablehlo.reshape %arg591 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %959 = stablehlo.multiply %957, %958 : tensor<1x1x1x256xf32>
      %960 = stablehlo.broadcast_in_dim %959, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %961 = stablehlo.multiply %954, %960 : tensor<256x14x14x256xf32>
      %962 = stablehlo.reshape %arg590 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %963 = stablehlo.broadcast_in_dim %962, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %964 = stablehlo.add %961, %963 : tensor<256x14x14x256xf32>
      %965 = stablehlo.convert %964 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %966 = func.call @relu_115(%965) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %967 = stablehlo.convert %arg597 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %968 = stablehlo.convolution(%966, %967) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %969 = stablehlo.broadcast_in_dim %arg884, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %970 = stablehlo.broadcast_in_dim %arg885, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %971 = stablehlo.convert %968 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %972 = stablehlo.broadcast_in_dim %969, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %973 = stablehlo.subtract %971, %972 : tensor<256x14x14x256xf32>
      %cst_50 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %974 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %975 = stablehlo.add %970, %974 : tensor<1x1x1x256xf32>
      %976 = stablehlo.rsqrt %975 : tensor<1x1x1x256xf32>
      %977 = stablehlo.reshape %arg593 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %978 = stablehlo.multiply %976, %977 : tensor<1x1x1x256xf32>
      %979 = stablehlo.broadcast_in_dim %978, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %980 = stablehlo.multiply %973, %979 : tensor<256x14x14x256xf32>
      %981 = stablehlo.reshape %arg592 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %982 = stablehlo.broadcast_in_dim %981, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %983 = stablehlo.add %980, %982 : tensor<256x14x14x256xf32>
      %984 = stablehlo.convert %983 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %985 = func.call @relu_115(%984) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %986 = stablehlo.convert %arg598 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %987 = stablehlo.convolution(%985, %986) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %988 = stablehlo.broadcast_in_dim %arg886, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %989 = stablehlo.broadcast_in_dim %arg887, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %990 = stablehlo.convert %987 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %991 = stablehlo.broadcast_in_dim %988, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %992 = stablehlo.subtract %990, %991 : tensor<256x14x14x1024xf32>
      %cst_51 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %993 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %994 = stablehlo.add %989, %993 : tensor<1x1x1x1024xf32>
      %995 = stablehlo.rsqrt %994 : tensor<1x1x1x1024xf32>
      %996 = stablehlo.reshape %arg595 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %997 = stablehlo.multiply %995, %996 : tensor<1x1x1x1024xf32>
      %998 = stablehlo.broadcast_in_dim %997, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %999 = stablehlo.multiply %992, %998 : tensor<256x14x14x1024xf32>
      %1000 = stablehlo.reshape %arg594 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1001 = stablehlo.broadcast_in_dim %1000, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1002 = stablehlo.add %999, %1001 : tensor<256x14x14x1024xf32>
      %1003 = stablehlo.convert %1002 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1004 = stablehlo.add %947, %1003 : tensor<256x14x14x1024xf16>
      %1005 = func.call @relu_133(%1004) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1006 = stablehlo.convert %arg605 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1007 = stablehlo.convolution(%1005, %1006) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1008 = stablehlo.broadcast_in_dim %arg888, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1009 = stablehlo.broadcast_in_dim %arg889, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1010 = stablehlo.convert %1007 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1011 = stablehlo.broadcast_in_dim %1008, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1012 = stablehlo.subtract %1010, %1011 : tensor<256x14x14x256xf32>
      %cst_52 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1013 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1014 = stablehlo.add %1009, %1013 : tensor<1x1x1x256xf32>
      %1015 = stablehlo.rsqrt %1014 : tensor<1x1x1x256xf32>
      %1016 = stablehlo.reshape %arg600 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1017 = stablehlo.multiply %1015, %1016 : tensor<1x1x1x256xf32>
      %1018 = stablehlo.broadcast_in_dim %1017, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1019 = stablehlo.multiply %1012, %1018 : tensor<256x14x14x256xf32>
      %1020 = stablehlo.reshape %arg599 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1021 = stablehlo.broadcast_in_dim %1020, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1022 = stablehlo.add %1019, %1021 : tensor<256x14x14x256xf32>
      %1023 = stablehlo.convert %1022 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1024 = func.call @relu_115(%1023) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1025 = stablehlo.convert %arg606 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1026 = stablehlo.convolution(%1024, %1025) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1027 = stablehlo.broadcast_in_dim %arg890, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1028 = stablehlo.broadcast_in_dim %arg891, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1029 = stablehlo.convert %1026 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1030 = stablehlo.broadcast_in_dim %1027, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1031 = stablehlo.subtract %1029, %1030 : tensor<256x14x14x256xf32>
      %cst_53 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1032 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1033 = stablehlo.add %1028, %1032 : tensor<1x1x1x256xf32>
      %1034 = stablehlo.rsqrt %1033 : tensor<1x1x1x256xf32>
      %1035 = stablehlo.reshape %arg602 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1036 = stablehlo.multiply %1034, %1035 : tensor<1x1x1x256xf32>
      %1037 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1038 = stablehlo.multiply %1031, %1037 : tensor<256x14x14x256xf32>
      %1039 = stablehlo.reshape %arg601 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1040 = stablehlo.broadcast_in_dim %1039, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1041 = stablehlo.add %1038, %1040 : tensor<256x14x14x256xf32>
      %1042 = stablehlo.convert %1041 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1043 = func.call @relu_115(%1042) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1044 = stablehlo.convert %arg607 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1045 = stablehlo.convolution(%1043, %1044) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1046 = stablehlo.broadcast_in_dim %arg892, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1047 = stablehlo.broadcast_in_dim %arg893, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1048 = stablehlo.convert %1045 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1049 = stablehlo.broadcast_in_dim %1046, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1050 = stablehlo.subtract %1048, %1049 : tensor<256x14x14x1024xf32>
      %cst_54 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1051 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1052 = stablehlo.add %1047, %1051 : tensor<1x1x1x1024xf32>
      %1053 = stablehlo.rsqrt %1052 : tensor<1x1x1x1024xf32>
      %1054 = stablehlo.reshape %arg604 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1055 = stablehlo.multiply %1053, %1054 : tensor<1x1x1x1024xf32>
      %1056 = stablehlo.broadcast_in_dim %1055, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1057 = stablehlo.multiply %1050, %1056 : tensor<256x14x14x1024xf32>
      %1058 = stablehlo.reshape %arg603 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1059 = stablehlo.broadcast_in_dim %1058, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1060 = stablehlo.add %1057, %1059 : tensor<256x14x14x1024xf32>
      %1061 = stablehlo.convert %1060 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1062 = stablehlo.add %1005, %1061 : tensor<256x14x14x1024xf16>
      %1063 = func.call @relu_133(%1062) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1064 = stablehlo.convert %arg614 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1065 = stablehlo.convolution(%1063, %1064) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1066 = stablehlo.broadcast_in_dim %arg894, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1067 = stablehlo.broadcast_in_dim %arg895, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1068 = stablehlo.convert %1065 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1069 = stablehlo.broadcast_in_dim %1066, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1070 = stablehlo.subtract %1068, %1069 : tensor<256x14x14x256xf32>
      %cst_55 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1071 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1072 = stablehlo.add %1067, %1071 : tensor<1x1x1x256xf32>
      %1073 = stablehlo.rsqrt %1072 : tensor<1x1x1x256xf32>
      %1074 = stablehlo.reshape %arg609 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1075 = stablehlo.multiply %1073, %1074 : tensor<1x1x1x256xf32>
      %1076 = stablehlo.broadcast_in_dim %1075, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1077 = stablehlo.multiply %1070, %1076 : tensor<256x14x14x256xf32>
      %1078 = stablehlo.reshape %arg608 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1079 = stablehlo.broadcast_in_dim %1078, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1080 = stablehlo.add %1077, %1079 : tensor<256x14x14x256xf32>
      %1081 = stablehlo.convert %1080 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1082 = func.call @relu_115(%1081) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1083 = stablehlo.convert %arg615 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1084 = stablehlo.convolution(%1082, %1083) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1085 = stablehlo.broadcast_in_dim %arg896, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1086 = stablehlo.broadcast_in_dim %arg897, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1087 = stablehlo.convert %1084 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1088 = stablehlo.broadcast_in_dim %1085, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1089 = stablehlo.subtract %1087, %1088 : tensor<256x14x14x256xf32>
      %cst_56 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1090 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1091 = stablehlo.add %1086, %1090 : tensor<1x1x1x256xf32>
      %1092 = stablehlo.rsqrt %1091 : tensor<1x1x1x256xf32>
      %1093 = stablehlo.reshape %arg611 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1094 = stablehlo.multiply %1092, %1093 : tensor<1x1x1x256xf32>
      %1095 = stablehlo.broadcast_in_dim %1094, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1096 = stablehlo.multiply %1089, %1095 : tensor<256x14x14x256xf32>
      %1097 = stablehlo.reshape %arg610 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1098 = stablehlo.broadcast_in_dim %1097, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1099 = stablehlo.add %1096, %1098 : tensor<256x14x14x256xf32>
      %1100 = stablehlo.convert %1099 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1101 = func.call @relu_115(%1100) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1102 = stablehlo.convert %arg616 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1103 = stablehlo.convolution(%1101, %1102) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1104 = stablehlo.broadcast_in_dim %arg898, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1105 = stablehlo.broadcast_in_dim %arg899, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1106 = stablehlo.convert %1103 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1107 = stablehlo.broadcast_in_dim %1104, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1108 = stablehlo.subtract %1106, %1107 : tensor<256x14x14x1024xf32>
      %cst_57 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1109 = stablehlo.broadcast_in_dim %cst_57, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1110 = stablehlo.add %1105, %1109 : tensor<1x1x1x1024xf32>
      %1111 = stablehlo.rsqrt %1110 : tensor<1x1x1x1024xf32>
      %1112 = stablehlo.reshape %arg613 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1113 = stablehlo.multiply %1111, %1112 : tensor<1x1x1x1024xf32>
      %1114 = stablehlo.broadcast_in_dim %1113, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1115 = stablehlo.multiply %1108, %1114 : tensor<256x14x14x1024xf32>
      %1116 = stablehlo.reshape %arg612 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1117 = stablehlo.broadcast_in_dim %1116, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1118 = stablehlo.add %1115, %1117 : tensor<256x14x14x1024xf32>
      %1119 = stablehlo.convert %1118 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1120 = stablehlo.add %1063, %1119 : tensor<256x14x14x1024xf16>
      %1121 = func.call @relu_133(%1120) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1122 = stablehlo.convert %arg623 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1123 = stablehlo.convolution(%1121, %1122) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1124 = stablehlo.broadcast_in_dim %arg900, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1125 = stablehlo.broadcast_in_dim %arg901, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1126 = stablehlo.convert %1123 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1127 = stablehlo.broadcast_in_dim %1124, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1128 = stablehlo.subtract %1126, %1127 : tensor<256x14x14x256xf32>
      %cst_58 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1129 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1130 = stablehlo.add %1125, %1129 : tensor<1x1x1x256xf32>
      %1131 = stablehlo.rsqrt %1130 : tensor<1x1x1x256xf32>
      %1132 = stablehlo.reshape %arg618 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1133 = stablehlo.multiply %1131, %1132 : tensor<1x1x1x256xf32>
      %1134 = stablehlo.broadcast_in_dim %1133, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1135 = stablehlo.multiply %1128, %1134 : tensor<256x14x14x256xf32>
      %1136 = stablehlo.reshape %arg617 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1137 = stablehlo.broadcast_in_dim %1136, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1138 = stablehlo.add %1135, %1137 : tensor<256x14x14x256xf32>
      %1139 = stablehlo.convert %1138 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1140 = func.call @relu_115(%1139) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1141 = stablehlo.convert %arg624 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1142 = stablehlo.convolution(%1140, %1141) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1143 = stablehlo.broadcast_in_dim %arg902, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1144 = stablehlo.broadcast_in_dim %arg903, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1145 = stablehlo.convert %1142 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1146 = stablehlo.broadcast_in_dim %1143, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1147 = stablehlo.subtract %1145, %1146 : tensor<256x14x14x256xf32>
      %cst_59 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1148 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1149 = stablehlo.add %1144, %1148 : tensor<1x1x1x256xf32>
      %1150 = stablehlo.rsqrt %1149 : tensor<1x1x1x256xf32>
      %1151 = stablehlo.reshape %arg620 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1152 = stablehlo.multiply %1150, %1151 : tensor<1x1x1x256xf32>
      %1153 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1154 = stablehlo.multiply %1147, %1153 : tensor<256x14x14x256xf32>
      %1155 = stablehlo.reshape %arg619 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1156 = stablehlo.broadcast_in_dim %1155, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1157 = stablehlo.add %1154, %1156 : tensor<256x14x14x256xf32>
      %1158 = stablehlo.convert %1157 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1159 = func.call @relu_115(%1158) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1160 = stablehlo.convert %arg625 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1161 = stablehlo.convolution(%1159, %1160) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1162 = stablehlo.broadcast_in_dim %arg904, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1163 = stablehlo.broadcast_in_dim %arg905, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1164 = stablehlo.convert %1161 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1165 = stablehlo.broadcast_in_dim %1162, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1166 = stablehlo.subtract %1164, %1165 : tensor<256x14x14x1024xf32>
      %cst_60 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1167 = stablehlo.broadcast_in_dim %cst_60, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1168 = stablehlo.add %1163, %1167 : tensor<1x1x1x1024xf32>
      %1169 = stablehlo.rsqrt %1168 : tensor<1x1x1x1024xf32>
      %1170 = stablehlo.reshape %arg622 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1171 = stablehlo.multiply %1169, %1170 : tensor<1x1x1x1024xf32>
      %1172 = stablehlo.broadcast_in_dim %1171, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1173 = stablehlo.multiply %1166, %1172 : tensor<256x14x14x1024xf32>
      %1174 = stablehlo.reshape %arg621 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1175 = stablehlo.broadcast_in_dim %1174, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1176 = stablehlo.add %1173, %1175 : tensor<256x14x14x1024xf32>
      %1177 = stablehlo.convert %1176 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1178 = stablehlo.add %1121, %1177 : tensor<256x14x14x1024xf16>
      %1179 = func.call @relu_133(%1178) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1180 = stablehlo.convert %arg632 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1181 = stablehlo.convolution(%1179, %1180) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1182 = stablehlo.broadcast_in_dim %arg906, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1183 = stablehlo.broadcast_in_dim %arg907, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1184 = stablehlo.convert %1181 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1185 = stablehlo.broadcast_in_dim %1182, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1186 = stablehlo.subtract %1184, %1185 : tensor<256x14x14x256xf32>
      %cst_61 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1187 = stablehlo.broadcast_in_dim %cst_61, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1188 = stablehlo.add %1183, %1187 : tensor<1x1x1x256xf32>
      %1189 = stablehlo.rsqrt %1188 : tensor<1x1x1x256xf32>
      %1190 = stablehlo.reshape %arg627 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1191 = stablehlo.multiply %1189, %1190 : tensor<1x1x1x256xf32>
      %1192 = stablehlo.broadcast_in_dim %1191, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1193 = stablehlo.multiply %1186, %1192 : tensor<256x14x14x256xf32>
      %1194 = stablehlo.reshape %arg626 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1195 = stablehlo.broadcast_in_dim %1194, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1196 = stablehlo.add %1193, %1195 : tensor<256x14x14x256xf32>
      %1197 = stablehlo.convert %1196 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1198 = func.call @relu_115(%1197) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1199 = stablehlo.convert %arg633 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1200 = stablehlo.convolution(%1198, %1199) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1201 = stablehlo.broadcast_in_dim %arg908, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1202 = stablehlo.broadcast_in_dim %arg909, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1203 = stablehlo.convert %1200 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1204 = stablehlo.broadcast_in_dim %1201, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1205 = stablehlo.subtract %1203, %1204 : tensor<256x14x14x256xf32>
      %cst_62 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1206 = stablehlo.broadcast_in_dim %cst_62, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1207 = stablehlo.add %1202, %1206 : tensor<1x1x1x256xf32>
      %1208 = stablehlo.rsqrt %1207 : tensor<1x1x1x256xf32>
      %1209 = stablehlo.reshape %arg629 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1210 = stablehlo.multiply %1208, %1209 : tensor<1x1x1x256xf32>
      %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1212 = stablehlo.multiply %1205, %1211 : tensor<256x14x14x256xf32>
      %1213 = stablehlo.reshape %arg628 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1214 = stablehlo.broadcast_in_dim %1213, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1215 = stablehlo.add %1212, %1214 : tensor<256x14x14x256xf32>
      %1216 = stablehlo.convert %1215 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1217 = func.call @relu_115(%1216) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1218 = stablehlo.convert %arg634 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1219 = stablehlo.convolution(%1217, %1218) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1220 = stablehlo.broadcast_in_dim %arg910, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1221 = stablehlo.broadcast_in_dim %arg911, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1222 = stablehlo.convert %1219 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1223 = stablehlo.broadcast_in_dim %1220, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1224 = stablehlo.subtract %1222, %1223 : tensor<256x14x14x1024xf32>
      %cst_63 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1225 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1226 = stablehlo.add %1221, %1225 : tensor<1x1x1x1024xf32>
      %1227 = stablehlo.rsqrt %1226 : tensor<1x1x1x1024xf32>
      %1228 = stablehlo.reshape %arg631 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1229 = stablehlo.multiply %1227, %1228 : tensor<1x1x1x1024xf32>
      %1230 = stablehlo.broadcast_in_dim %1229, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1231 = stablehlo.multiply %1224, %1230 : tensor<256x14x14x1024xf32>
      %1232 = stablehlo.reshape %arg630 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1233 = stablehlo.broadcast_in_dim %1232, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1234 = stablehlo.add %1231, %1233 : tensor<256x14x14x1024xf32>
      %1235 = stablehlo.convert %1234 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1236 = stablehlo.add %1179, %1235 : tensor<256x14x14x1024xf16>
      %1237 = func.call @relu_133(%1236) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1238 = stablehlo.convert %arg650 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1239 = stablehlo.convolution(%1237, %1238) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1240 = stablehlo.broadcast_in_dim %arg918, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1241 = stablehlo.broadcast_in_dim %arg919, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1242 = stablehlo.convert %1239 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1243 = stablehlo.broadcast_in_dim %1240, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1244 = stablehlo.subtract %1242, %1243 : tensor<256x14x14x256xf32>
      %cst_64 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1245 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1246 = stablehlo.add %1241, %1245 : tensor<1x1x1x256xf32>
      %1247 = stablehlo.rsqrt %1246 : tensor<1x1x1x256xf32>
      %1248 = stablehlo.reshape %arg645 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1249 = stablehlo.multiply %1247, %1248 : tensor<1x1x1x256xf32>
      %1250 = stablehlo.broadcast_in_dim %1249, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1251 = stablehlo.multiply %1244, %1250 : tensor<256x14x14x256xf32>
      %1252 = stablehlo.reshape %arg644 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1253 = stablehlo.broadcast_in_dim %1252, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1254 = stablehlo.add %1251, %1253 : tensor<256x14x14x256xf32>
      %1255 = stablehlo.convert %1254 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1256 = func.call @relu_115(%1255) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1257 = stablehlo.convert %arg651 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1258 = stablehlo.convolution(%1256, %1257) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1259 = stablehlo.broadcast_in_dim %arg920, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1260 = stablehlo.broadcast_in_dim %arg921, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1261 = stablehlo.convert %1258 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1262 = stablehlo.broadcast_in_dim %1259, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1263 = stablehlo.subtract %1261, %1262 : tensor<256x14x14x256xf32>
      %cst_65 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1264 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1265 = stablehlo.add %1260, %1264 : tensor<1x1x1x256xf32>
      %1266 = stablehlo.rsqrt %1265 : tensor<1x1x1x256xf32>
      %1267 = stablehlo.reshape %arg647 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1268 = stablehlo.multiply %1266, %1267 : tensor<1x1x1x256xf32>
      %1269 = stablehlo.broadcast_in_dim %1268, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1270 = stablehlo.multiply %1263, %1269 : tensor<256x14x14x256xf32>
      %1271 = stablehlo.reshape %arg646 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1272 = stablehlo.broadcast_in_dim %1271, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1273 = stablehlo.add %1270, %1272 : tensor<256x14x14x256xf32>
      %1274 = stablehlo.convert %1273 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1275 = func.call @relu_115(%1274) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1276 = stablehlo.convert %arg652 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1277 = stablehlo.convolution(%1275, %1276) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1278 = stablehlo.broadcast_in_dim %arg922, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1279 = stablehlo.broadcast_in_dim %arg923, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1280 = stablehlo.convert %1277 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1281 = stablehlo.broadcast_in_dim %1278, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1282 = stablehlo.subtract %1280, %1281 : tensor<256x14x14x1024xf32>
      %cst_66 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1283 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1284 = stablehlo.add %1279, %1283 : tensor<1x1x1x1024xf32>
      %1285 = stablehlo.rsqrt %1284 : tensor<1x1x1x1024xf32>
      %1286 = stablehlo.reshape %arg649 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1287 = stablehlo.multiply %1285, %1286 : tensor<1x1x1x1024xf32>
      %1288 = stablehlo.broadcast_in_dim %1287, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1289 = stablehlo.multiply %1282, %1288 : tensor<256x14x14x1024xf32>
      %1290 = stablehlo.reshape %arg648 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1291 = stablehlo.broadcast_in_dim %1290, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1292 = stablehlo.add %1289, %1291 : tensor<256x14x14x1024xf32>
      %1293 = stablehlo.convert %1292 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1294 = stablehlo.add %1237, %1293 : tensor<256x14x14x1024xf16>
      %1295 = func.call @relu_133(%1294) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1296 = stablehlo.convert %arg659 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1297 = stablehlo.convolution(%1295, %1296) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1298 = stablehlo.broadcast_in_dim %arg924, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1299 = stablehlo.broadcast_in_dim %arg925, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1300 = stablehlo.convert %1297 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1301 = stablehlo.broadcast_in_dim %1298, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1302 = stablehlo.subtract %1300, %1301 : tensor<256x14x14x256xf32>
      %cst_67 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1303 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1304 = stablehlo.add %1299, %1303 : tensor<1x1x1x256xf32>
      %1305 = stablehlo.rsqrt %1304 : tensor<1x1x1x256xf32>
      %1306 = stablehlo.reshape %arg654 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1307 = stablehlo.multiply %1305, %1306 : tensor<1x1x1x256xf32>
      %1308 = stablehlo.broadcast_in_dim %1307, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1309 = stablehlo.multiply %1302, %1308 : tensor<256x14x14x256xf32>
      %1310 = stablehlo.reshape %arg653 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1311 = stablehlo.broadcast_in_dim %1310, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1312 = stablehlo.add %1309, %1311 : tensor<256x14x14x256xf32>
      %1313 = stablehlo.convert %1312 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1314 = func.call @relu_115(%1313) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1315 = stablehlo.convert %arg660 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1316 = stablehlo.convolution(%1314, %1315) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1317 = stablehlo.broadcast_in_dim %arg926, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1318 = stablehlo.broadcast_in_dim %arg927, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1319 = stablehlo.convert %1316 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1320 = stablehlo.broadcast_in_dim %1317, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1321 = stablehlo.subtract %1319, %1320 : tensor<256x14x14x256xf32>
      %cst_68 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1322 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1323 = stablehlo.add %1318, %1322 : tensor<1x1x1x256xf32>
      %1324 = stablehlo.rsqrt %1323 : tensor<1x1x1x256xf32>
      %1325 = stablehlo.reshape %arg656 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1326 = stablehlo.multiply %1324, %1325 : tensor<1x1x1x256xf32>
      %1327 = stablehlo.broadcast_in_dim %1326, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1328 = stablehlo.multiply %1321, %1327 : tensor<256x14x14x256xf32>
      %1329 = stablehlo.reshape %arg655 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1330 = stablehlo.broadcast_in_dim %1329, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1331 = stablehlo.add %1328, %1330 : tensor<256x14x14x256xf32>
      %1332 = stablehlo.convert %1331 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1333 = func.call @relu_115(%1332) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1334 = stablehlo.convert %arg661 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1335 = stablehlo.convolution(%1333, %1334) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1336 = stablehlo.broadcast_in_dim %arg928, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1337 = stablehlo.broadcast_in_dim %arg929, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1338 = stablehlo.convert %1335 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1339 = stablehlo.broadcast_in_dim %1336, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1340 = stablehlo.subtract %1338, %1339 : tensor<256x14x14x1024xf32>
      %cst_69 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1341 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1342 = stablehlo.add %1337, %1341 : tensor<1x1x1x1024xf32>
      %1343 = stablehlo.rsqrt %1342 : tensor<1x1x1x1024xf32>
      %1344 = stablehlo.reshape %arg658 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1345 = stablehlo.multiply %1343, %1344 : tensor<1x1x1x1024xf32>
      %1346 = stablehlo.broadcast_in_dim %1345, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1347 = stablehlo.multiply %1340, %1346 : tensor<256x14x14x1024xf32>
      %1348 = stablehlo.reshape %arg657 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1349 = stablehlo.broadcast_in_dim %1348, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1350 = stablehlo.add %1347, %1349 : tensor<256x14x14x1024xf32>
      %1351 = stablehlo.convert %1350 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1352 = stablehlo.add %1295, %1351 : tensor<256x14x14x1024xf16>
      %1353 = func.call @relu_133(%1352) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1354 = stablehlo.convert %arg668 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1355 = stablehlo.convolution(%1353, %1354) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1356 = stablehlo.broadcast_in_dim %arg930, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1357 = stablehlo.broadcast_in_dim %arg931, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1358 = stablehlo.convert %1355 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1359 = stablehlo.broadcast_in_dim %1356, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1360 = stablehlo.subtract %1358, %1359 : tensor<256x14x14x256xf32>
      %cst_70 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1361 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1362 = stablehlo.add %1357, %1361 : tensor<1x1x1x256xf32>
      %1363 = stablehlo.rsqrt %1362 : tensor<1x1x1x256xf32>
      %1364 = stablehlo.reshape %arg663 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1365 = stablehlo.multiply %1363, %1364 : tensor<1x1x1x256xf32>
      %1366 = stablehlo.broadcast_in_dim %1365, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1367 = stablehlo.multiply %1360, %1366 : tensor<256x14x14x256xf32>
      %1368 = stablehlo.reshape %arg662 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1369 = stablehlo.broadcast_in_dim %1368, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1370 = stablehlo.add %1367, %1369 : tensor<256x14x14x256xf32>
      %1371 = stablehlo.convert %1370 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1372 = func.call @relu_115(%1371) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1373 = stablehlo.convert %arg669 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1374 = stablehlo.convolution(%1372, %1373) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1375 = stablehlo.broadcast_in_dim %arg932, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1376 = stablehlo.broadcast_in_dim %arg933, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1377 = stablehlo.convert %1374 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1378 = stablehlo.broadcast_in_dim %1375, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1379 = stablehlo.subtract %1377, %1378 : tensor<256x14x14x256xf32>
      %cst_71 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1380 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1381 = stablehlo.add %1376, %1380 : tensor<1x1x1x256xf32>
      %1382 = stablehlo.rsqrt %1381 : tensor<1x1x1x256xf32>
      %1383 = stablehlo.reshape %arg665 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1384 = stablehlo.multiply %1382, %1383 : tensor<1x1x1x256xf32>
      %1385 = stablehlo.broadcast_in_dim %1384, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1386 = stablehlo.multiply %1379, %1385 : tensor<256x14x14x256xf32>
      %1387 = stablehlo.reshape %arg664 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1388 = stablehlo.broadcast_in_dim %1387, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1389 = stablehlo.add %1386, %1388 : tensor<256x14x14x256xf32>
      %1390 = stablehlo.convert %1389 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1391 = func.call @relu_115(%1390) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1392 = stablehlo.convert %arg670 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1393 = stablehlo.convolution(%1391, %1392) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1394 = stablehlo.broadcast_in_dim %arg934, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1395 = stablehlo.broadcast_in_dim %arg935, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1396 = stablehlo.convert %1393 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1397 = stablehlo.broadcast_in_dim %1394, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1398 = stablehlo.subtract %1396, %1397 : tensor<256x14x14x1024xf32>
      %cst_72 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1399 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1400 = stablehlo.add %1395, %1399 : tensor<1x1x1x1024xf32>
      %1401 = stablehlo.rsqrt %1400 : tensor<1x1x1x1024xf32>
      %1402 = stablehlo.reshape %arg667 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1403 = stablehlo.multiply %1401, %1402 : tensor<1x1x1x1024xf32>
      %1404 = stablehlo.broadcast_in_dim %1403, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1405 = stablehlo.multiply %1398, %1404 : tensor<256x14x14x1024xf32>
      %1406 = stablehlo.reshape %arg666 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1407 = stablehlo.broadcast_in_dim %1406, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1408 = stablehlo.add %1405, %1407 : tensor<256x14x14x1024xf32>
      %1409 = stablehlo.convert %1408 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1410 = stablehlo.add %1353, %1409 : tensor<256x14x14x1024xf16>
      %1411 = func.call @relu_133(%1410) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1412 = stablehlo.convert %arg677 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1413 = stablehlo.convolution(%1411, %1412) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1414 = stablehlo.broadcast_in_dim %arg936, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1415 = stablehlo.broadcast_in_dim %arg937, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1416 = stablehlo.convert %1413 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1417 = stablehlo.broadcast_in_dim %1414, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1418 = stablehlo.subtract %1416, %1417 : tensor<256x14x14x256xf32>
      %cst_73 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1419 = stablehlo.broadcast_in_dim %cst_73, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1420 = stablehlo.add %1415, %1419 : tensor<1x1x1x256xf32>
      %1421 = stablehlo.rsqrt %1420 : tensor<1x1x1x256xf32>
      %1422 = stablehlo.reshape %arg672 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1423 = stablehlo.multiply %1421, %1422 : tensor<1x1x1x256xf32>
      %1424 = stablehlo.broadcast_in_dim %1423, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1425 = stablehlo.multiply %1418, %1424 : tensor<256x14x14x256xf32>
      %1426 = stablehlo.reshape %arg671 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1427 = stablehlo.broadcast_in_dim %1426, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1428 = stablehlo.add %1425, %1427 : tensor<256x14x14x256xf32>
      %1429 = stablehlo.convert %1428 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1430 = func.call @relu_115(%1429) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1431 = stablehlo.convert %arg678 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1432 = stablehlo.convolution(%1430, %1431) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1433 = stablehlo.broadcast_in_dim %arg938, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1434 = stablehlo.broadcast_in_dim %arg939, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1435 = stablehlo.convert %1432 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1436 = stablehlo.broadcast_in_dim %1433, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1437 = stablehlo.subtract %1435, %1436 : tensor<256x14x14x256xf32>
      %cst_74 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1438 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1439 = stablehlo.add %1434, %1438 : tensor<1x1x1x256xf32>
      %1440 = stablehlo.rsqrt %1439 : tensor<1x1x1x256xf32>
      %1441 = stablehlo.reshape %arg674 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1442 = stablehlo.multiply %1440, %1441 : tensor<1x1x1x256xf32>
      %1443 = stablehlo.broadcast_in_dim %1442, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1444 = stablehlo.multiply %1437, %1443 : tensor<256x14x14x256xf32>
      %1445 = stablehlo.reshape %arg673 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1446 = stablehlo.broadcast_in_dim %1445, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1447 = stablehlo.add %1444, %1446 : tensor<256x14x14x256xf32>
      %1448 = stablehlo.convert %1447 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1449 = func.call @relu_115(%1448) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1450 = stablehlo.convert %arg679 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1451 = stablehlo.convolution(%1449, %1450) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1452 = stablehlo.broadcast_in_dim %arg940, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1453 = stablehlo.broadcast_in_dim %arg941, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1454 = stablehlo.convert %1451 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1455 = stablehlo.broadcast_in_dim %1452, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1456 = stablehlo.subtract %1454, %1455 : tensor<256x14x14x1024xf32>
      %cst_75 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1457 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1458 = stablehlo.add %1453, %1457 : tensor<1x1x1x1024xf32>
      %1459 = stablehlo.rsqrt %1458 : tensor<1x1x1x1024xf32>
      %1460 = stablehlo.reshape %arg676 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1461 = stablehlo.multiply %1459, %1460 : tensor<1x1x1x1024xf32>
      %1462 = stablehlo.broadcast_in_dim %1461, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1463 = stablehlo.multiply %1456, %1462 : tensor<256x14x14x1024xf32>
      %1464 = stablehlo.reshape %arg675 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1465 = stablehlo.broadcast_in_dim %1464, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1466 = stablehlo.add %1463, %1465 : tensor<256x14x14x1024xf32>
      %1467 = stablehlo.convert %1466 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1468 = stablehlo.add %1411, %1467 : tensor<256x14x14x1024xf16>
      %1469 = func.call @relu_133(%1468) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1470 = stablehlo.convert %arg686 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1471 = stablehlo.convolution(%1469, %1470) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1472 = stablehlo.broadcast_in_dim %arg942, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1473 = stablehlo.broadcast_in_dim %arg943, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1474 = stablehlo.convert %1471 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1475 = stablehlo.broadcast_in_dim %1472, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1476 = stablehlo.subtract %1474, %1475 : tensor<256x14x14x256xf32>
      %cst_76 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1477 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1478 = stablehlo.add %1473, %1477 : tensor<1x1x1x256xf32>
      %1479 = stablehlo.rsqrt %1478 : tensor<1x1x1x256xf32>
      %1480 = stablehlo.reshape %arg681 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1481 = stablehlo.multiply %1479, %1480 : tensor<1x1x1x256xf32>
      %1482 = stablehlo.broadcast_in_dim %1481, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1483 = stablehlo.multiply %1476, %1482 : tensor<256x14x14x256xf32>
      %1484 = stablehlo.reshape %arg680 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1485 = stablehlo.broadcast_in_dim %1484, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1486 = stablehlo.add %1483, %1485 : tensor<256x14x14x256xf32>
      %1487 = stablehlo.convert %1486 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1488 = func.call @relu_115(%1487) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1489 = stablehlo.convert %arg687 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1490 = stablehlo.convolution(%1488, %1489) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1491 = stablehlo.broadcast_in_dim %arg944, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1492 = stablehlo.broadcast_in_dim %arg945, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1493 = stablehlo.convert %1490 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1494 = stablehlo.broadcast_in_dim %1491, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1495 = stablehlo.subtract %1493, %1494 : tensor<256x14x14x256xf32>
      %cst_77 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1496 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1497 = stablehlo.add %1492, %1496 : tensor<1x1x1x256xf32>
      %1498 = stablehlo.rsqrt %1497 : tensor<1x1x1x256xf32>
      %1499 = stablehlo.reshape %arg683 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1500 = stablehlo.multiply %1498, %1499 : tensor<1x1x1x256xf32>
      %1501 = stablehlo.broadcast_in_dim %1500, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1502 = stablehlo.multiply %1495, %1501 : tensor<256x14x14x256xf32>
      %1503 = stablehlo.reshape %arg682 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1504 = stablehlo.broadcast_in_dim %1503, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1505 = stablehlo.add %1502, %1504 : tensor<256x14x14x256xf32>
      %1506 = stablehlo.convert %1505 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1507 = func.call @relu_115(%1506) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1508 = stablehlo.convert %arg688 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1509 = stablehlo.convolution(%1507, %1508) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1510 = stablehlo.broadcast_in_dim %arg946, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1511 = stablehlo.broadcast_in_dim %arg947, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1512 = stablehlo.convert %1509 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1513 = stablehlo.broadcast_in_dim %1510, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1514 = stablehlo.subtract %1512, %1513 : tensor<256x14x14x1024xf32>
      %cst_78 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1515 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1516 = stablehlo.add %1511, %1515 : tensor<1x1x1x1024xf32>
      %1517 = stablehlo.rsqrt %1516 : tensor<1x1x1x1024xf32>
      %1518 = stablehlo.reshape %arg685 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1519 = stablehlo.multiply %1517, %1518 : tensor<1x1x1x1024xf32>
      %1520 = stablehlo.broadcast_in_dim %1519, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1521 = stablehlo.multiply %1514, %1520 : tensor<256x14x14x1024xf32>
      %1522 = stablehlo.reshape %arg684 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1523 = stablehlo.broadcast_in_dim %1522, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1524 = stablehlo.add %1521, %1523 : tensor<256x14x14x1024xf32>
      %1525 = stablehlo.convert %1524 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1526 = stablehlo.add %1469, %1525 : tensor<256x14x14x1024xf16>
      %1527 = func.call @relu_133(%1526) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1528 = stablehlo.convert %arg695 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1529 = stablehlo.convolution(%1527, %1528) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1530 = stablehlo.broadcast_in_dim %arg948, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1531 = stablehlo.broadcast_in_dim %arg949, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1532 = stablehlo.convert %1529 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1533 = stablehlo.broadcast_in_dim %1530, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1534 = stablehlo.subtract %1532, %1533 : tensor<256x14x14x256xf32>
      %cst_79 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1535 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1536 = stablehlo.add %1531, %1535 : tensor<1x1x1x256xf32>
      %1537 = stablehlo.rsqrt %1536 : tensor<1x1x1x256xf32>
      %1538 = stablehlo.reshape %arg690 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1539 = stablehlo.multiply %1537, %1538 : tensor<1x1x1x256xf32>
      %1540 = stablehlo.broadcast_in_dim %1539, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1541 = stablehlo.multiply %1534, %1540 : tensor<256x14x14x256xf32>
      %1542 = stablehlo.reshape %arg689 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1543 = stablehlo.broadcast_in_dim %1542, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1544 = stablehlo.add %1541, %1543 : tensor<256x14x14x256xf32>
      %1545 = stablehlo.convert %1544 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1546 = func.call @relu_115(%1545) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1547 = stablehlo.convert %arg696 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1548 = stablehlo.convolution(%1546, %1547) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1549 = stablehlo.broadcast_in_dim %arg950, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1550 = stablehlo.broadcast_in_dim %arg951, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1551 = stablehlo.convert %1548 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1552 = stablehlo.broadcast_in_dim %1549, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1553 = stablehlo.subtract %1551, %1552 : tensor<256x14x14x256xf32>
      %cst_80 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1554 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1555 = stablehlo.add %1550, %1554 : tensor<1x1x1x256xf32>
      %1556 = stablehlo.rsqrt %1555 : tensor<1x1x1x256xf32>
      %1557 = stablehlo.reshape %arg692 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1558 = stablehlo.multiply %1556, %1557 : tensor<1x1x1x256xf32>
      %1559 = stablehlo.broadcast_in_dim %1558, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1560 = stablehlo.multiply %1553, %1559 : tensor<256x14x14x256xf32>
      %1561 = stablehlo.reshape %arg691 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1562 = stablehlo.broadcast_in_dim %1561, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1563 = stablehlo.add %1560, %1562 : tensor<256x14x14x256xf32>
      %1564 = stablehlo.convert %1563 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1565 = func.call @relu_115(%1564) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1566 = stablehlo.convert %arg697 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1567 = stablehlo.convolution(%1565, %1566) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1568 = stablehlo.broadcast_in_dim %arg952, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1569 = stablehlo.broadcast_in_dim %arg953, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1570 = stablehlo.convert %1567 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1571 = stablehlo.broadcast_in_dim %1568, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1572 = stablehlo.subtract %1570, %1571 : tensor<256x14x14x1024xf32>
      %cst_81 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1573 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1574 = stablehlo.add %1569, %1573 : tensor<1x1x1x1024xf32>
      %1575 = stablehlo.rsqrt %1574 : tensor<1x1x1x1024xf32>
      %1576 = stablehlo.reshape %arg694 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1577 = stablehlo.multiply %1575, %1576 : tensor<1x1x1x1024xf32>
      %1578 = stablehlo.broadcast_in_dim %1577, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1579 = stablehlo.multiply %1572, %1578 : tensor<256x14x14x1024xf32>
      %1580 = stablehlo.reshape %arg693 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1581 = stablehlo.broadcast_in_dim %1580, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1582 = stablehlo.add %1579, %1581 : tensor<256x14x14x1024xf32>
      %1583 = stablehlo.convert %1582 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1584 = stablehlo.add %1527, %1583 : tensor<256x14x14x1024xf16>
      %1585 = func.call @relu_133(%1584) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1586 = stablehlo.convert %arg704 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1587 = stablehlo.convolution(%1585, %1586) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1588 = stablehlo.broadcast_in_dim %arg954, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1589 = stablehlo.broadcast_in_dim %arg955, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1590 = stablehlo.convert %1587 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1591 = stablehlo.broadcast_in_dim %1588, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1592 = stablehlo.subtract %1590, %1591 : tensor<256x14x14x256xf32>
      %cst_82 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1593 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1594 = stablehlo.add %1589, %1593 : tensor<1x1x1x256xf32>
      %1595 = stablehlo.rsqrt %1594 : tensor<1x1x1x256xf32>
      %1596 = stablehlo.reshape %arg699 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1597 = stablehlo.multiply %1595, %1596 : tensor<1x1x1x256xf32>
      %1598 = stablehlo.broadcast_in_dim %1597, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1599 = stablehlo.multiply %1592, %1598 : tensor<256x14x14x256xf32>
      %1600 = stablehlo.reshape %arg698 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1601 = stablehlo.broadcast_in_dim %1600, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1602 = stablehlo.add %1599, %1601 : tensor<256x14x14x256xf32>
      %1603 = stablehlo.convert %1602 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1604 = func.call @relu_115(%1603) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1605 = stablehlo.convert %arg705 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1606 = stablehlo.convolution(%1604, %1605) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1607 = stablehlo.broadcast_in_dim %arg956, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1608 = stablehlo.broadcast_in_dim %arg957, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1609 = stablehlo.convert %1606 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1610 = stablehlo.broadcast_in_dim %1607, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1611 = stablehlo.subtract %1609, %1610 : tensor<256x14x14x256xf32>
      %cst_83 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1612 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1613 = stablehlo.add %1608, %1612 : tensor<1x1x1x256xf32>
      %1614 = stablehlo.rsqrt %1613 : tensor<1x1x1x256xf32>
      %1615 = stablehlo.reshape %arg701 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1616 = stablehlo.multiply %1614, %1615 : tensor<1x1x1x256xf32>
      %1617 = stablehlo.broadcast_in_dim %1616, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1618 = stablehlo.multiply %1611, %1617 : tensor<256x14x14x256xf32>
      %1619 = stablehlo.reshape %arg700 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1620 = stablehlo.broadcast_in_dim %1619, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1621 = stablehlo.add %1618, %1620 : tensor<256x14x14x256xf32>
      %1622 = stablehlo.convert %1621 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1623 = func.call @relu_115(%1622) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1624 = stablehlo.convert %arg706 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1625 = stablehlo.convolution(%1623, %1624) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1626 = stablehlo.broadcast_in_dim %arg958, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1627 = stablehlo.broadcast_in_dim %arg959, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1628 = stablehlo.convert %1625 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1629 = stablehlo.broadcast_in_dim %1626, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1630 = stablehlo.subtract %1628, %1629 : tensor<256x14x14x1024xf32>
      %cst_84 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1631 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1632 = stablehlo.add %1627, %1631 : tensor<1x1x1x1024xf32>
      %1633 = stablehlo.rsqrt %1632 : tensor<1x1x1x1024xf32>
      %1634 = stablehlo.reshape %arg703 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1635 = stablehlo.multiply %1633, %1634 : tensor<1x1x1x1024xf32>
      %1636 = stablehlo.broadcast_in_dim %1635, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1637 = stablehlo.multiply %1630, %1636 : tensor<256x14x14x1024xf32>
      %1638 = stablehlo.reshape %arg702 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1639 = stablehlo.broadcast_in_dim %1638, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1640 = stablehlo.add %1637, %1639 : tensor<256x14x14x1024xf32>
      %1641 = stablehlo.convert %1640 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1642 = stablehlo.add %1585, %1641 : tensor<256x14x14x1024xf16>
      %1643 = func.call @relu_133(%1642) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1644 = stablehlo.convert %arg713 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1645 = stablehlo.convolution(%1643, %1644) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1646 = stablehlo.broadcast_in_dim %arg960, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1647 = stablehlo.broadcast_in_dim %arg961, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1648 = stablehlo.convert %1645 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1649 = stablehlo.broadcast_in_dim %1646, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1650 = stablehlo.subtract %1648, %1649 : tensor<256x14x14x256xf32>
      %cst_85 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1651 = stablehlo.broadcast_in_dim %cst_85, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1652 = stablehlo.add %1647, %1651 : tensor<1x1x1x256xf32>
      %1653 = stablehlo.rsqrt %1652 : tensor<1x1x1x256xf32>
      %1654 = stablehlo.reshape %arg708 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1655 = stablehlo.multiply %1653, %1654 : tensor<1x1x1x256xf32>
      %1656 = stablehlo.broadcast_in_dim %1655, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1657 = stablehlo.multiply %1650, %1656 : tensor<256x14x14x256xf32>
      %1658 = stablehlo.reshape %arg707 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1659 = stablehlo.broadcast_in_dim %1658, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1660 = stablehlo.add %1657, %1659 : tensor<256x14x14x256xf32>
      %1661 = stablehlo.convert %1660 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1662 = func.call @relu_115(%1661) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1663 = stablehlo.convert %arg714 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1664 = stablehlo.convolution(%1662, %1663) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1665 = stablehlo.broadcast_in_dim %arg962, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1666 = stablehlo.broadcast_in_dim %arg963, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1667 = stablehlo.convert %1664 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1668 = stablehlo.broadcast_in_dim %1665, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1669 = stablehlo.subtract %1667, %1668 : tensor<256x14x14x256xf32>
      %cst_86 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1670 = stablehlo.broadcast_in_dim %cst_86, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1671 = stablehlo.add %1666, %1670 : tensor<1x1x1x256xf32>
      %1672 = stablehlo.rsqrt %1671 : tensor<1x1x1x256xf32>
      %1673 = stablehlo.reshape %arg710 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1674 = stablehlo.multiply %1672, %1673 : tensor<1x1x1x256xf32>
      %1675 = stablehlo.broadcast_in_dim %1674, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1676 = stablehlo.multiply %1669, %1675 : tensor<256x14x14x256xf32>
      %1677 = stablehlo.reshape %arg709 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1678 = stablehlo.broadcast_in_dim %1677, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1679 = stablehlo.add %1676, %1678 : tensor<256x14x14x256xf32>
      %1680 = stablehlo.convert %1679 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1681 = func.call @relu_115(%1680) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1682 = stablehlo.convert %arg715 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1683 = stablehlo.convolution(%1681, %1682) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1684 = stablehlo.broadcast_in_dim %arg964, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1685 = stablehlo.broadcast_in_dim %arg965, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1686 = stablehlo.convert %1683 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1687 = stablehlo.broadcast_in_dim %1684, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1688 = stablehlo.subtract %1686, %1687 : tensor<256x14x14x1024xf32>
      %cst_87 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1689 = stablehlo.broadcast_in_dim %cst_87, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1690 = stablehlo.add %1685, %1689 : tensor<1x1x1x1024xf32>
      %1691 = stablehlo.rsqrt %1690 : tensor<1x1x1x1024xf32>
      %1692 = stablehlo.reshape %arg712 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1693 = stablehlo.multiply %1691, %1692 : tensor<1x1x1x1024xf32>
      %1694 = stablehlo.broadcast_in_dim %1693, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1695 = stablehlo.multiply %1688, %1694 : tensor<256x14x14x1024xf32>
      %1696 = stablehlo.reshape %arg711 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1697 = stablehlo.broadcast_in_dim %1696, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1698 = stablehlo.add %1695, %1697 : tensor<256x14x14x1024xf32>
      %1699 = stablehlo.convert %1698 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1700 = stablehlo.add %1643, %1699 : tensor<256x14x14x1024xf16>
      %1701 = func.call @relu_133(%1700) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1702 = stablehlo.convert %arg722 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1703 = stablehlo.convolution(%1701, %1702) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1704 = stablehlo.broadcast_in_dim %arg966, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1705 = stablehlo.broadcast_in_dim %arg967, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1706 = stablehlo.convert %1703 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1707 = stablehlo.broadcast_in_dim %1704, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1708 = stablehlo.subtract %1706, %1707 : tensor<256x14x14x256xf32>
      %cst_88 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1709 = stablehlo.broadcast_in_dim %cst_88, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1710 = stablehlo.add %1705, %1709 : tensor<1x1x1x256xf32>
      %1711 = stablehlo.rsqrt %1710 : tensor<1x1x1x256xf32>
      %1712 = stablehlo.reshape %arg717 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1713 = stablehlo.multiply %1711, %1712 : tensor<1x1x1x256xf32>
      %1714 = stablehlo.broadcast_in_dim %1713, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1715 = stablehlo.multiply %1708, %1714 : tensor<256x14x14x256xf32>
      %1716 = stablehlo.reshape %arg716 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1717 = stablehlo.broadcast_in_dim %1716, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1718 = stablehlo.add %1715, %1717 : tensor<256x14x14x256xf32>
      %1719 = stablehlo.convert %1718 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1720 = func.call @relu_115(%1719) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1721 = stablehlo.convert %arg723 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1722 = stablehlo.convolution(%1720, %1721) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1723 = stablehlo.broadcast_in_dim %arg968, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1724 = stablehlo.broadcast_in_dim %arg969, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1725 = stablehlo.convert %1722 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1726 = stablehlo.broadcast_in_dim %1723, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1727 = stablehlo.subtract %1725, %1726 : tensor<256x14x14x256xf32>
      %cst_89 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1728 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1729 = stablehlo.add %1724, %1728 : tensor<1x1x1x256xf32>
      %1730 = stablehlo.rsqrt %1729 : tensor<1x1x1x256xf32>
      %1731 = stablehlo.reshape %arg719 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1732 = stablehlo.multiply %1730, %1731 : tensor<1x1x1x256xf32>
      %1733 = stablehlo.broadcast_in_dim %1732, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1734 = stablehlo.multiply %1727, %1733 : tensor<256x14x14x256xf32>
      %1735 = stablehlo.reshape %arg718 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1736 = stablehlo.broadcast_in_dim %1735, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1737 = stablehlo.add %1734, %1736 : tensor<256x14x14x256xf32>
      %1738 = stablehlo.convert %1737 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1739 = func.call @relu_115(%1738) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1740 = stablehlo.convert %arg724 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1741 = stablehlo.convolution(%1739, %1740) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1742 = stablehlo.broadcast_in_dim %arg970, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1743 = stablehlo.broadcast_in_dim %arg971, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1744 = stablehlo.convert %1741 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1745 = stablehlo.broadcast_in_dim %1742, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1746 = stablehlo.subtract %1744, %1745 : tensor<256x14x14x1024xf32>
      %cst_90 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1747 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1748 = stablehlo.add %1743, %1747 : tensor<1x1x1x1024xf32>
      %1749 = stablehlo.rsqrt %1748 : tensor<1x1x1x1024xf32>
      %1750 = stablehlo.reshape %arg721 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1751 = stablehlo.multiply %1749, %1750 : tensor<1x1x1x1024xf32>
      %1752 = stablehlo.broadcast_in_dim %1751, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1753 = stablehlo.multiply %1746, %1752 : tensor<256x14x14x1024xf32>
      %1754 = stablehlo.reshape %arg720 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1755 = stablehlo.broadcast_in_dim %1754, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1756 = stablehlo.add %1753, %1755 : tensor<256x14x14x1024xf32>
      %1757 = stablehlo.convert %1756 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1758 = stablehlo.add %1701, %1757 : tensor<256x14x14x1024xf16>
      %1759 = func.call @relu_133(%1758) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1760 = stablehlo.convert %arg731 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1761 = stablehlo.convolution(%1759, %1760) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1762 = stablehlo.broadcast_in_dim %arg972, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1763 = stablehlo.broadcast_in_dim %arg973, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1764 = stablehlo.convert %1761 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1765 = stablehlo.broadcast_in_dim %1762, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1766 = stablehlo.subtract %1764, %1765 : tensor<256x14x14x256xf32>
      %cst_91 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1767 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1768 = stablehlo.add %1763, %1767 : tensor<1x1x1x256xf32>
      %1769 = stablehlo.rsqrt %1768 : tensor<1x1x1x256xf32>
      %1770 = stablehlo.reshape %arg726 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1771 = stablehlo.multiply %1769, %1770 : tensor<1x1x1x256xf32>
      %1772 = stablehlo.broadcast_in_dim %1771, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1773 = stablehlo.multiply %1766, %1772 : tensor<256x14x14x256xf32>
      %1774 = stablehlo.reshape %arg725 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1775 = stablehlo.broadcast_in_dim %1774, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1776 = stablehlo.add %1773, %1775 : tensor<256x14x14x256xf32>
      %1777 = stablehlo.convert %1776 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1778 = func.call @relu_115(%1777) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1779 = stablehlo.convert %arg732 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1780 = stablehlo.convolution(%1778, %1779) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1781 = stablehlo.broadcast_in_dim %arg974, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1782 = stablehlo.broadcast_in_dim %arg975, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1783 = stablehlo.convert %1780 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1784 = stablehlo.broadcast_in_dim %1781, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1785 = stablehlo.subtract %1783, %1784 : tensor<256x14x14x256xf32>
      %cst_92 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1786 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1787 = stablehlo.add %1782, %1786 : tensor<1x1x1x256xf32>
      %1788 = stablehlo.rsqrt %1787 : tensor<1x1x1x256xf32>
      %1789 = stablehlo.reshape %arg728 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1790 = stablehlo.multiply %1788, %1789 : tensor<1x1x1x256xf32>
      %1791 = stablehlo.broadcast_in_dim %1790, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1792 = stablehlo.multiply %1785, %1791 : tensor<256x14x14x256xf32>
      %1793 = stablehlo.reshape %arg727 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1794 = stablehlo.broadcast_in_dim %1793, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1795 = stablehlo.add %1792, %1794 : tensor<256x14x14x256xf32>
      %1796 = stablehlo.convert %1795 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1797 = func.call @relu_115(%1796) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1798 = stablehlo.convert %arg733 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1799 = stablehlo.convolution(%1797, %1798) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1800 = stablehlo.broadcast_in_dim %arg976, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1801 = stablehlo.broadcast_in_dim %arg977, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1802 = stablehlo.convert %1799 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1803 = stablehlo.broadcast_in_dim %1800, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1804 = stablehlo.subtract %1802, %1803 : tensor<256x14x14x1024xf32>
      %cst_93 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1805 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1806 = stablehlo.add %1801, %1805 : tensor<1x1x1x1024xf32>
      %1807 = stablehlo.rsqrt %1806 : tensor<1x1x1x1024xf32>
      %1808 = stablehlo.reshape %arg730 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1809 = stablehlo.multiply %1807, %1808 : tensor<1x1x1x1024xf32>
      %1810 = stablehlo.broadcast_in_dim %1809, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1811 = stablehlo.multiply %1804, %1810 : tensor<256x14x14x1024xf32>
      %1812 = stablehlo.reshape %arg729 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1813 = stablehlo.broadcast_in_dim %1812, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1814 = stablehlo.add %1811, %1813 : tensor<256x14x14x1024xf32>
      %1815 = stablehlo.convert %1814 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1816 = stablehlo.add %1759, %1815 : tensor<256x14x14x1024xf16>
      %1817 = func.call @relu_133(%1816) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1818 = stablehlo.convert %arg752 : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf16>
      %1819 = stablehlo.convolution(%1817, %1818) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x512xf16>) -> tensor<256x14x14x512xf16>
      %1820 = stablehlo.broadcast_in_dim %arg986, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1821 = stablehlo.broadcast_in_dim %arg987, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1822 = stablehlo.convert %1819 : (tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf32>
      %1823 = stablehlo.broadcast_in_dim %1820, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
      %1824 = stablehlo.subtract %1822, %1823 : tensor<256x14x14x512xf32>
      %cst_94 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1825 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1826 = stablehlo.add %1821, %1825 : tensor<1x1x1x512xf32>
      %1827 = stablehlo.rsqrt %1826 : tensor<1x1x1x512xf32>
      %1828 = stablehlo.reshape %arg747 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1829 = stablehlo.multiply %1827, %1828 : tensor<1x1x1x512xf32>
      %1830 = stablehlo.broadcast_in_dim %1829, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
      %1831 = stablehlo.multiply %1824, %1830 : tensor<256x14x14x512xf32>
      %1832 = stablehlo.reshape %arg746 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1833 = stablehlo.broadcast_in_dim %1832, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
      %1834 = stablehlo.add %1831, %1833 : tensor<256x14x14x512xf32>
      %1835 = stablehlo.convert %1834 : (tensor<256x14x14x512xf32>) -> tensor<256x14x14x512xf16>
      %1836 = func.call @relu_212(%1835) : (tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf16>
      %1837 = stablehlo.convert %arg753 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %1838 = stablehlo.convolution(%1836, %1837) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %1839 = stablehlo.broadcast_in_dim %arg988, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1840 = stablehlo.broadcast_in_dim %arg989, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1841 = stablehlo.convert %1838 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1842 = stablehlo.broadcast_in_dim %1839, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1843 = stablehlo.subtract %1841, %1842 : tensor<256x7x7x512xf32>
      %cst_95 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1844 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1845 = stablehlo.add %1840, %1844 : tensor<1x1x1x512xf32>
      %1846 = stablehlo.rsqrt %1845 : tensor<1x1x1x512xf32>
      %1847 = stablehlo.reshape %arg749 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1848 = stablehlo.multiply %1846, %1847 : tensor<1x1x1x512xf32>
      %1849 = stablehlo.broadcast_in_dim %1848, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1850 = stablehlo.multiply %1843, %1849 : tensor<256x7x7x512xf32>
      %1851 = stablehlo.reshape %arg748 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1852 = stablehlo.broadcast_in_dim %1851, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1853 = stablehlo.add %1850, %1852 : tensor<256x7x7x512xf32>
      %1854 = stablehlo.convert %1853 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1855 = func.call @relu_222(%1854) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %1856 = stablehlo.convert %arg754 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
      %1857 = stablehlo.convolution(%1855, %1856) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
      %1858 = stablehlo.broadcast_in_dim %arg990, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1859 = stablehlo.broadcast_in_dim %arg991, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1860 = stablehlo.convert %1857 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %1861 = stablehlo.broadcast_in_dim %1858, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1862 = stablehlo.subtract %1860, %1861 : tensor<256x7x7x2048xf32>
      %cst_96 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1863 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %1864 = stablehlo.add %1859, %1863 : tensor<1x1x1x2048xf32>
      %1865 = stablehlo.rsqrt %1864 : tensor<1x1x1x2048xf32>
      %1866 = stablehlo.reshape %arg751 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1867 = stablehlo.multiply %1865, %1866 : tensor<1x1x1x2048xf32>
      %1868 = stablehlo.broadcast_in_dim %1867, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1869 = stablehlo.multiply %1862, %1868 : tensor<256x7x7x2048xf32>
      %1870 = stablehlo.reshape %arg750 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1871 = stablehlo.broadcast_in_dim %1870, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1872 = stablehlo.add %1869, %1871 : tensor<256x7x7x2048xf32>
      %1873 = stablehlo.convert %1872 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %1874 = stablehlo.convert %arg755 : (tensor<1x1x1024x2048xf32>) -> tensor<1x1x1024x2048xf16>
      %1875 = stablehlo.convolution(%1817, %1874) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x2048xf16>) -> tensor<256x7x7x2048xf16>
      %1876 = stablehlo.broadcast_in_dim %arg992, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1877 = stablehlo.broadcast_in_dim %arg993, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1878 = stablehlo.convert %1875 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %1879 = stablehlo.broadcast_in_dim %1876, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1880 = stablehlo.subtract %1878, %1879 : tensor<256x7x7x2048xf32>
      %cst_97 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1881 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %1882 = stablehlo.add %1877, %1881 : tensor<1x1x1x2048xf32>
      %1883 = stablehlo.rsqrt %1882 : tensor<1x1x1x2048xf32>
      %1884 = stablehlo.reshape %arg757 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1885 = stablehlo.multiply %1883, %1884 : tensor<1x1x1x2048xf32>
      %1886 = stablehlo.broadcast_in_dim %1885, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1887 = stablehlo.multiply %1880, %1886 : tensor<256x7x7x2048xf32>
      %1888 = stablehlo.reshape %arg756 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1889 = stablehlo.broadcast_in_dim %1888, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1890 = stablehlo.add %1887, %1889 : tensor<256x7x7x2048xf32>
      %1891 = stablehlo.convert %1890 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %1892 = stablehlo.add %1891, %1873 : tensor<256x7x7x2048xf16>
      %1893 = func.call @relu_240(%1892) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
      %1894 = stablehlo.convert %arg764 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
      %1895 = stablehlo.convolution(%1893, %1894) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<256x7x7x512xf16>
      %1896 = stablehlo.broadcast_in_dim %arg994, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1897 = stablehlo.broadcast_in_dim %arg995, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1898 = stablehlo.convert %1895 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1899 = stablehlo.broadcast_in_dim %1896, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1900 = stablehlo.subtract %1898, %1899 : tensor<256x7x7x512xf32>
      %cst_98 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1901 = stablehlo.broadcast_in_dim %cst_98, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1902 = stablehlo.add %1897, %1901 : tensor<1x1x1x512xf32>
      %1903 = stablehlo.rsqrt %1902 : tensor<1x1x1x512xf32>
      %1904 = stablehlo.reshape %arg759 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1905 = stablehlo.multiply %1903, %1904 : tensor<1x1x1x512xf32>
      %1906 = stablehlo.broadcast_in_dim %1905, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1907 = stablehlo.multiply %1900, %1906 : tensor<256x7x7x512xf32>
      %1908 = stablehlo.reshape %arg758 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1909 = stablehlo.broadcast_in_dim %1908, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1910 = stablehlo.add %1907, %1909 : tensor<256x7x7x512xf32>
      %1911 = stablehlo.convert %1910 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1912 = func.call @relu_222(%1911) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %1913 = stablehlo.convert %arg765 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %1914 = stablehlo.convolution(%1912, %1913) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %1915 = stablehlo.broadcast_in_dim %arg996, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1916 = stablehlo.broadcast_in_dim %arg997, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1917 = stablehlo.convert %1914 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1918 = stablehlo.broadcast_in_dim %1915, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1919 = stablehlo.subtract %1917, %1918 : tensor<256x7x7x512xf32>
      %cst_99 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1920 = stablehlo.broadcast_in_dim %cst_99, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1921 = stablehlo.add %1916, %1920 : tensor<1x1x1x512xf32>
      %1922 = stablehlo.rsqrt %1921 : tensor<1x1x1x512xf32>
      %1923 = stablehlo.reshape %arg761 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1924 = stablehlo.multiply %1922, %1923 : tensor<1x1x1x512xf32>
      %1925 = stablehlo.broadcast_in_dim %1924, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1926 = stablehlo.multiply %1919, %1925 : tensor<256x7x7x512xf32>
      %1927 = stablehlo.reshape %arg760 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1928 = stablehlo.broadcast_in_dim %1927, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1929 = stablehlo.add %1926, %1928 : tensor<256x7x7x512xf32>
      %1930 = stablehlo.convert %1929 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1931 = func.call @relu_222(%1930) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %1932 = stablehlo.convert %arg766 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
      %1933 = stablehlo.convolution(%1931, %1932) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
      %1934 = stablehlo.broadcast_in_dim %arg998, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1935 = stablehlo.broadcast_in_dim %arg999, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1936 = stablehlo.convert %1933 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %1937 = stablehlo.broadcast_in_dim %1934, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1938 = stablehlo.subtract %1936, %1937 : tensor<256x7x7x2048xf32>
      %cst_100 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1939 = stablehlo.broadcast_in_dim %cst_100, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %1940 = stablehlo.add %1935, %1939 : tensor<1x1x1x2048xf32>
      %1941 = stablehlo.rsqrt %1940 : tensor<1x1x1x2048xf32>
      %1942 = stablehlo.reshape %arg763 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1943 = stablehlo.multiply %1941, %1942 : tensor<1x1x1x2048xf32>
      %1944 = stablehlo.broadcast_in_dim %1943, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1945 = stablehlo.multiply %1938, %1944 : tensor<256x7x7x2048xf32>
      %1946 = stablehlo.reshape %arg762 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1947 = stablehlo.broadcast_in_dim %1946, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1948 = stablehlo.add %1945, %1947 : tensor<256x7x7x2048xf32>
      %1949 = stablehlo.convert %1948 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %1950 = stablehlo.add %1893, %1949 : tensor<256x7x7x2048xf16>
      %1951 = func.call @relu_240(%1950) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
      %1952 = stablehlo.convert %arg773 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
      %1953 = stablehlo.convolution(%1951, %1952) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<256x7x7x512xf16>
      %1954 = stablehlo.broadcast_in_dim %arg1000, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1955 = stablehlo.broadcast_in_dim %arg1001, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1956 = stablehlo.convert %1953 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1957 = stablehlo.broadcast_in_dim %1954, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1958 = stablehlo.subtract %1956, %1957 : tensor<256x7x7x512xf32>
      %cst_101 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1959 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1960 = stablehlo.add %1955, %1959 : tensor<1x1x1x512xf32>
      %1961 = stablehlo.rsqrt %1960 : tensor<1x1x1x512xf32>
      %1962 = stablehlo.reshape %arg768 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1963 = stablehlo.multiply %1961, %1962 : tensor<1x1x1x512xf32>
      %1964 = stablehlo.broadcast_in_dim %1963, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1965 = stablehlo.multiply %1958, %1964 : tensor<256x7x7x512xf32>
      %1966 = stablehlo.reshape %arg767 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1967 = stablehlo.broadcast_in_dim %1966, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1968 = stablehlo.add %1965, %1967 : tensor<256x7x7x512xf32>
      %1969 = stablehlo.convert %1968 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1970 = func.call @relu_222(%1969) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %1971 = stablehlo.convert %arg774 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %1972 = stablehlo.convolution(%1970, %1971) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %1973 = stablehlo.broadcast_in_dim %arg1002, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1974 = stablehlo.broadcast_in_dim %arg1003, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1975 = stablehlo.convert %1972 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %1976 = stablehlo.broadcast_in_dim %1973, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1977 = stablehlo.subtract %1975, %1976 : tensor<256x7x7x512xf32>
      %cst_102 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1978 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1979 = stablehlo.add %1974, %1978 : tensor<1x1x1x512xf32>
      %1980 = stablehlo.rsqrt %1979 : tensor<1x1x1x512xf32>
      %1981 = stablehlo.reshape %arg770 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1982 = stablehlo.multiply %1980, %1981 : tensor<1x1x1x512xf32>
      %1983 = stablehlo.broadcast_in_dim %1982, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1984 = stablehlo.multiply %1977, %1983 : tensor<256x7x7x512xf32>
      %1985 = stablehlo.reshape %arg769 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1986 = stablehlo.broadcast_in_dim %1985, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %1987 = stablehlo.add %1984, %1986 : tensor<256x7x7x512xf32>
      %1988 = stablehlo.convert %1987 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %1989 = func.call @relu_222(%1988) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %1990 = stablehlo.convert %arg775 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
      %1991 = stablehlo.convolution(%1989, %1990) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
      %1992 = stablehlo.broadcast_in_dim %arg1004, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1993 = stablehlo.broadcast_in_dim %arg1005, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %1994 = stablehlo.convert %1991 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %1995 = stablehlo.broadcast_in_dim %1992, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %1996 = stablehlo.subtract %1994, %1995 : tensor<256x7x7x2048xf32>
      %cst_103 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1997 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %1998 = stablehlo.add %1993, %1997 : tensor<1x1x1x2048xf32>
      %1999 = stablehlo.rsqrt %1998 : tensor<1x1x1x2048xf32>
      %2000 = stablehlo.reshape %arg772 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %2001 = stablehlo.multiply %1999, %2000 : tensor<1x1x1x2048xf32>
      %2002 = stablehlo.broadcast_in_dim %2001, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %2003 = stablehlo.multiply %1996, %2002 : tensor<256x7x7x2048xf32>
      %2004 = stablehlo.reshape %arg771 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %2005 = stablehlo.broadcast_in_dim %2004, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %2006 = stablehlo.add %2003, %2005 : tensor<256x7x7x2048xf32>
      %2007 = stablehlo.convert %2006 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %2008 = stablehlo.add %1951, %2007 : tensor<256x7x7x2048xf16>
      %2009 = func.call @relu_240(%2008) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
      %2010 = stablehlo.convert %2009 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2011 = stablehlo.reduce(%2010 init: %cst_104) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x2048xf32>, tensor<f32>) -> tensor<256x2048xf32>
      %cst_105 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
      %2012 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
      %2013 = stablehlo.divide %2011, %2012 : tensor<256x2048xf32>
      %2014 = stablehlo.convert %2013 : (tensor<256x2048xf32>) -> tensor<256x2048xf16>
      %2015 = stablehlo.convert %arg834 : (tensor<2048x1000xf32>) -> tensor<2048x1000xf16>
      %2016 = stablehlo.convert %arg833 : (tensor<1000xf32>) -> tensor<1000xf16>
      %2017 = stablehlo.dot_general %2014, %2015, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf16>, tensor<2048x1000xf16>) -> tensor<256x1000xf16>
      %2018 = stablehlo.reshape %2016 : (tensor<1000xf16>) -> tensor<1x1000xf16>
      %2019 = stablehlo.broadcast_in_dim %2018, dims = [0, 1] : (tensor<1x1000xf16>) -> tensor<256x1000xf16>
      %2020 = stablehlo.add %2017, %2019 : tensor<256x1000xf16>
      %2021 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
      %2022 = stablehlo.iota dim = 0 : tensor<1000xi32>
      %2023 = stablehlo.reshape %2022 : (tensor<1000xi32>) -> tensor<1x1000xi32>
      %2024 = stablehlo.broadcast_in_dim %2021, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
      %2025 = stablehlo.broadcast_in_dim %2023, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
      %2026 = stablehlo.compare  EQ, %2024, %2025,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
      %cst_106 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2027 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2028 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %2029 = stablehlo.select %2026, %2027, %2028 : tensor<256x1000xi1>, tensor<256x1000xf32>
      %2030 = stablehlo.convert %2029 : tensor<256x1000xf32>
      %2031 = func.call @log_softmax(%2020) : (tensor<256x1000xf16>) -> tensor<256x1000xf16>
      %2032 = stablehlo.convert %2031 : (tensor<256x1000xf16>) -> tensor<256x1000xf32>
      %2033 = stablehlo.multiply %2030, %2032 : tensor<256x1000xf32>
      %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2034 = stablehlo.reduce(%2033 init: %cst_108) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
      %2035 = stablehlo.negate %2034 : tensor<256xf32>
      %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2036 = stablehlo.reduce(%2035 init: %cst_109) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_110 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %2037 = stablehlo.divide %2036, %cst_110 : tensor<f32>
      %2038 = func.call @argmax(%2020) : (tensor<256x1000xf16>) -> tensor<256xi32>
      %2039 = stablehlo.compare  EQ, %2038, %2,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
      %2040 = stablehlo.convert %2039 : (tensor<256xi1>) -> tensor<256xi32>
      %2041 = stablehlo.convert %2040 : (tensor<256xi32>) -> tensor<256xf32>
      %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %2042 = stablehlo.reduce(%2041 init: %cst_111) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_112 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %2043 = stablehlo.divide %2042, %cst_112 : tensor<f32>
      %2044 = "stablehlo.all_reduce"(%2043) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg1048: tensor<f32>, %arg1049: tensor<f32>):
        %2050 = stablehlo.add %arg1048, %arg1049 : tensor<f32>
        stablehlo.return %2050 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %2045 = "stablehlo.all_reduce"(%2037) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg1048: tensor<f32>, %arg1049: tensor<f32>):
        %2050 = stablehlo.add %arg1048, %arg1049 : tensor<f32>
        stablehlo.return %2050 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_113 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2046 = stablehlo.divide %2044, %cst_113 : tensor<f32>
      %cst_114 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %2047 = stablehlo.divide %2045, %cst_114 : tensor<f32>
      %2048 = stablehlo.broadcast_in_dim %2046, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %2049 = stablehlo.broadcast_in_dim %2047, dims = [] : (tensor<f32>) -> tensor<1xf32>
      sdy.return %2048, %2049 : tensor<1xf32>, tensor<1xf32>
    } : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x1x64x64xf32>, tensor<3x3x64x64xf32>, tensor<1x1x64x256xf32>, tensor<1x1x64x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x1x256x64xf32>, tensor<3x3x64x64xf32>, tensor<1x1x64x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x1x256x64xf32>, tensor<3x3x64x64xf32>, tensor<1x1x64x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x256x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x1x1024x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x512x2048xf32>, tensor<1x1x1024x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x1x2048x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x512x2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x1x2048x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x512x2048xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x512x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<1x1x512x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<1000xf32>, tensor<2048x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x256x224x224x3xf16>, tensor<4x256xi32>) -> (tensor<4xf32>, tensor<4xf32>)
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
  func.func private @relu_212(%arg0: tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x512xf16>
    return %1 : tensor<256x14x14x512xf16>
  }
  func.func private @relu_222(%arg0: tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xf16>
    return %1 : tensor<256x7x7x512xf16>
  }
  func.func private @relu_240(%arg0: tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16> {
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
