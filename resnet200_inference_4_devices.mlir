module @jit_call_wrapped attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["batch"=4]>
  func.func public @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<1x1x64x64xf32>, %arg7: tensor<3x3x64x64xf32>, %arg8: tensor<1x1x64x256xf32>, %arg9: tensor<1x1x64x256xf32>, %arg10: tensor<256xf32>, %arg11: tensor<256xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<256xf32>, %arg17: tensor<256xf32>, %arg18: tensor<1x1x256x64xf32>, %arg19: tensor<3x3x64x64xf32>, %arg20: tensor<1x1x64x256xf32>, %arg21: tensor<128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<512xf32>, %arg26: tensor<512xf32>, %arg27: tensor<1x1x512x128xf32>, %arg28: tensor<3x3x128x128xf32>, %arg29: tensor<1x1x128x512xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<1x1x512x128xf32>, %arg37: tensor<3x3x128x128xf32>, %arg38: tensor<1x1x128x512xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<1x1x512x128xf32>, %arg46: tensor<3x3x128x128xf32>, %arg47: tensor<1x1x128x512xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<512xf32>, %arg53: tensor<512xf32>, %arg54: tensor<1x1x512x128xf32>, %arg55: tensor<3x3x128x128xf32>, %arg56: tensor<1x1x128x512xf32>, %arg57: tensor<128xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<128xf32>, %arg61: tensor<512xf32>, %arg62: tensor<512xf32>, %arg63: tensor<1x1x512x128xf32>, %arg64: tensor<3x3x128x128xf32>, %arg65: tensor<1x1x128x512xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<128xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<1x1x512x128xf32>, %arg73: tensor<3x3x128x128xf32>, %arg74: tensor<1x1x128x512xf32>, %arg75: tensor<128xf32>, %arg76: tensor<128xf32>, %arg77: tensor<128xf32>, %arg78: tensor<128xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<1x1x512x128xf32>, %arg82: tensor<3x3x128x128xf32>, %arg83: tensor<1x1x128x512xf32>, %arg84: tensor<128xf32>, %arg85: tensor<128xf32>, %arg86: tensor<128xf32>, %arg87: tensor<128xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<1x1x512x128xf32>, %arg91: tensor<3x3x128x128xf32>, %arg92: tensor<1x1x128x512xf32>, %arg93: tensor<128xf32>, %arg94: tensor<128xf32>, %arg95: tensor<128xf32>, %arg96: tensor<128xf32>, %arg97: tensor<512xf32>, %arg98: tensor<512xf32>, %arg99: tensor<1x1x512x128xf32>, %arg100: tensor<3x3x128x128xf32>, %arg101: tensor<1x1x128x512xf32>, %arg102: tensor<128xf32>, %arg103: tensor<128xf32>, %arg104: tensor<128xf32>, %arg105: tensor<128xf32>, %arg106: tensor<512xf32>, %arg107: tensor<512xf32>, %arg108: tensor<1x1x512x128xf32>, %arg109: tensor<3x3x128x128xf32>, %arg110: tensor<1x1x128x512xf32>, %arg111: tensor<64xf32>, %arg112: tensor<64xf32>, %arg113: tensor<64xf32>, %arg114: tensor<64xf32>, %arg115: tensor<256xf32>, %arg116: tensor<256xf32>, %arg117: tensor<1x1x256x64xf32>, %arg118: tensor<3x3x64x64xf32>, %arg119: tensor<1x1x64x256xf32>, %arg120: tensor<128xf32>, %arg121: tensor<128xf32>, %arg122: tensor<128xf32>, %arg123: tensor<128xf32>, %arg124: tensor<512xf32>, %arg125: tensor<512xf32>, %arg126: tensor<1x1x512x128xf32>, %arg127: tensor<3x3x128x128xf32>, %arg128: tensor<1x1x128x512xf32>, %arg129: tensor<128xf32>, %arg130: tensor<128xf32>, %arg131: tensor<128xf32>, %arg132: tensor<128xf32>, %arg133: tensor<512xf32>, %arg134: tensor<512xf32>, %arg135: tensor<1x1x512x128xf32>, %arg136: tensor<3x3x128x128xf32>, %arg137: tensor<1x1x128x512xf32>, %arg138: tensor<128xf32>, %arg139: tensor<128xf32>, %arg140: tensor<128xf32>, %arg141: tensor<128xf32>, %arg142: tensor<512xf32>, %arg143: tensor<512xf32>, %arg144: tensor<1x1x512x128xf32>, %arg145: tensor<3x3x128x128xf32>, %arg146: tensor<1x1x128x512xf32>, %arg147: tensor<128xf32>, %arg148: tensor<128xf32>, %arg149: tensor<128xf32>, %arg150: tensor<128xf32>, %arg151: tensor<512xf32>, %arg152: tensor<512xf32>, %arg153: tensor<1x1x512x128xf32>, %arg154: tensor<3x3x128x128xf32>, %arg155: tensor<1x1x128x512xf32>, %arg156: tensor<128xf32>, %arg157: tensor<128xf32>, %arg158: tensor<128xf32>, %arg159: tensor<128xf32>, %arg160: tensor<512xf32>, %arg161: tensor<512xf32>, %arg162: tensor<1x1x512x128xf32>, %arg163: tensor<3x3x128x128xf32>, %arg164: tensor<1x1x128x512xf32>, %arg165: tensor<128xf32>, %arg166: tensor<128xf32>, %arg167: tensor<128xf32>, %arg168: tensor<128xf32>, %arg169: tensor<512xf32>, %arg170: tensor<512xf32>, %arg171: tensor<1x1x512x128xf32>, %arg172: tensor<3x3x128x128xf32>, %arg173: tensor<1x1x128x512xf32>, %arg174: tensor<128xf32>, %arg175: tensor<128xf32>, %arg176: tensor<128xf32>, %arg177: tensor<128xf32>, %arg178: tensor<512xf32>, %arg179: tensor<512xf32>, %arg180: tensor<1x1x512x128xf32>, %arg181: tensor<3x3x128x128xf32>, %arg182: tensor<1x1x128x512xf32>, %arg183: tensor<256xf32>, %arg184: tensor<256xf32>, %arg185: tensor<256xf32>, %arg186: tensor<256xf32>, %arg187: tensor<1024xf32>, %arg188: tensor<1024xf32>, %arg189: tensor<1x1x512x256xf32>, %arg190: tensor<3x3x256x256xf32>, %arg191: tensor<1x1x256x1024xf32>, %arg192: tensor<1x1x512x1024xf32>, %arg193: tensor<1024xf32>, %arg194: tensor<1024xf32>, %arg195: tensor<256xf32>, %arg196: tensor<256xf32>, %arg197: tensor<256xf32>, %arg198: tensor<256xf32>, %arg199: tensor<1024xf32>, %arg200: tensor<1024xf32>, %arg201: tensor<1x1x1024x256xf32>, %arg202: tensor<3x3x256x256xf32>, %arg203: tensor<1x1x256x1024xf32>, %arg204: tensor<256xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<256xf32>, %arg208: tensor<1024xf32>, %arg209: tensor<1024xf32>, %arg210: tensor<1x1x1024x256xf32>, %arg211: tensor<3x3x256x256xf32>, %arg212: tensor<1x1x256x1024xf32>, %arg213: tensor<128xf32>, %arg214: tensor<128xf32>, %arg215: tensor<128xf32>, %arg216: tensor<128xf32>, %arg217: tensor<512xf32>, %arg218: tensor<512xf32>, %arg219: tensor<1x1x256x128xf32>, %arg220: tensor<3x3x128x128xf32>, %arg221: tensor<1x1x128x512xf32>, %arg222: tensor<1x1x256x512xf32>, %arg223: tensor<512xf32>, %arg224: tensor<512xf32>, %arg225: tensor<256xf32>, %arg226: tensor<256xf32>, %arg227: tensor<256xf32>, %arg228: tensor<256xf32>, %arg229: tensor<1024xf32>, %arg230: tensor<1024xf32>, %arg231: tensor<1x1x1024x256xf32>, %arg232: tensor<3x3x256x256xf32>, %arg233: tensor<1x1x256x1024xf32>, %arg234: tensor<256xf32>, %arg235: tensor<256xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256xf32>, %arg238: tensor<1024xf32>, %arg239: tensor<1024xf32>, %arg240: tensor<1x1x1024x256xf32>, %arg241: tensor<3x3x256x256xf32>, %arg242: tensor<1x1x256x1024xf32>, %arg243: tensor<256xf32>, %arg244: tensor<256xf32>, %arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<1024xf32>, %arg248: tensor<1024xf32>, %arg249: tensor<1x1x1024x256xf32>, %arg250: tensor<3x3x256x256xf32>, %arg251: tensor<1x1x256x1024xf32>, %arg252: tensor<256xf32>, %arg253: tensor<256xf32>, %arg254: tensor<256xf32>, %arg255: tensor<256xf32>, %arg256: tensor<1024xf32>, %arg257: tensor<1024xf32>, %arg258: tensor<1x1x1024x256xf32>, %arg259: tensor<3x3x256x256xf32>, %arg260: tensor<1x1x256x1024xf32>, %arg261: tensor<256xf32>, %arg262: tensor<256xf32>, %arg263: tensor<256xf32>, %arg264: tensor<256xf32>, %arg265: tensor<1024xf32>, %arg266: tensor<1024xf32>, %arg267: tensor<1x1x1024x256xf32>, %arg268: tensor<3x3x256x256xf32>, %arg269: tensor<1x1x256x1024xf32>, %arg270: tensor<256xf32>, %arg271: tensor<256xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256xf32>, %arg274: tensor<1024xf32>, %arg275: tensor<1024xf32>, %arg276: tensor<1x1x1024x256xf32>, %arg277: tensor<3x3x256x256xf32>, %arg278: tensor<1x1x256x1024xf32>, %arg279: tensor<256xf32>, %arg280: tensor<256xf32>, %arg281: tensor<256xf32>, %arg282: tensor<256xf32>, %arg283: tensor<1024xf32>, %arg284: tensor<1024xf32>, %arg285: tensor<1x1x1024x256xf32>, %arg286: tensor<3x3x256x256xf32>, %arg287: tensor<1x1x256x1024xf32>, %arg288: tensor<256xf32>, %arg289: tensor<256xf32>, %arg290: tensor<256xf32>, %arg291: tensor<256xf32>, %arg292: tensor<1024xf32>, %arg293: tensor<1024xf32>, %arg294: tensor<1x1x1024x256xf32>, %arg295: tensor<3x3x256x256xf32>, %arg296: tensor<1x1x256x1024xf32>, %arg297: tensor<256xf32>, %arg298: tensor<256xf32>, %arg299: tensor<256xf32>, %arg300: tensor<256xf32>, %arg301: tensor<1024xf32>, %arg302: tensor<1024xf32>, %arg303: tensor<1x1x1024x256xf32>, %arg304: tensor<3x3x256x256xf32>, %arg305: tensor<1x1x256x1024xf32>, %arg306: tensor<256xf32>, %arg307: tensor<256xf32>, %arg308: tensor<256xf32>, %arg309: tensor<256xf32>, %arg310: tensor<1024xf32>, %arg311: tensor<1024xf32>, %arg312: tensor<1x1x1024x256xf32>, %arg313: tensor<3x3x256x256xf32>, %arg314: tensor<1x1x256x1024xf32>, %arg315: tensor<128xf32>, %arg316: tensor<128xf32>, %arg317: tensor<128xf32>, %arg318: tensor<128xf32>, %arg319: tensor<512xf32>, %arg320: tensor<512xf32>, %arg321: tensor<1x1x512x128xf32>, %arg322: tensor<3x3x128x128xf32>, %arg323: tensor<1x1x128x512xf32>, %arg324: tensor<256xf32>, %arg325: tensor<256xf32>, %arg326: tensor<256xf32>, %arg327: tensor<256xf32>, %arg328: tensor<1024xf32>, %arg329: tensor<1024xf32>, %arg330: tensor<1x1x1024x256xf32>, %arg331: tensor<3x3x256x256xf32>, %arg332: tensor<1x1x256x1024xf32>, %arg333: tensor<256xf32>, %arg334: tensor<256xf32>, %arg335: tensor<256xf32>, %arg336: tensor<256xf32>, %arg337: tensor<1024xf32>, %arg338: tensor<1024xf32>, %arg339: tensor<1x1x1024x256xf32>, %arg340: tensor<3x3x256x256xf32>, %arg341: tensor<1x1x256x1024xf32>, %arg342: tensor<256xf32>, %arg343: tensor<256xf32>, %arg344: tensor<256xf32>, %arg345: tensor<256xf32>, %arg346: tensor<1024xf32>, %arg347: tensor<1024xf32>, %arg348: tensor<1x1x1024x256xf32>, %arg349: tensor<3x3x256x256xf32>, %arg350: tensor<1x1x256x1024xf32>, %arg351: tensor<256xf32>, %arg352: tensor<256xf32>, %arg353: tensor<256xf32>, %arg354: tensor<256xf32>, %arg355: tensor<1024xf32>, %arg356: tensor<1024xf32>, %arg357: tensor<1x1x1024x256xf32>, %arg358: tensor<3x3x256x256xf32>, %arg359: tensor<1x1x256x1024xf32>, %arg360: tensor<256xf32>, %arg361: tensor<256xf32>, %arg362: tensor<256xf32>, %arg363: tensor<256xf32>, %arg364: tensor<1024xf32>, %arg365: tensor<1024xf32>, %arg366: tensor<1x1x1024x256xf32>, %arg367: tensor<3x3x256x256xf32>, %arg368: tensor<1x1x256x1024xf32>, %arg369: tensor<256xf32>, %arg370: tensor<256xf32>, %arg371: tensor<256xf32>, %arg372: tensor<256xf32>, %arg373: tensor<1024xf32>, %arg374: tensor<1024xf32>, %arg375: tensor<1x1x1024x256xf32>, %arg376: tensor<3x3x256x256xf32>, %arg377: tensor<1x1x256x1024xf32>, %arg378: tensor<256xf32>, %arg379: tensor<256xf32>, %arg380: tensor<256xf32>, %arg381: tensor<256xf32>, %arg382: tensor<1024xf32>, %arg383: tensor<1024xf32>, %arg384: tensor<1x1x1024x256xf32>, %arg385: tensor<3x3x256x256xf32>, %arg386: tensor<1x1x256x1024xf32>, %arg387: tensor<256xf32>, %arg388: tensor<256xf32>, %arg389: tensor<256xf32>, %arg390: tensor<256xf32>, %arg391: tensor<1024xf32>, %arg392: tensor<1024xf32>, %arg393: tensor<1x1x1024x256xf32>, %arg394: tensor<3x3x256x256xf32>, %arg395: tensor<1x1x256x1024xf32>, %arg396: tensor<256xf32>, %arg397: tensor<256xf32>, %arg398: tensor<256xf32>, %arg399: tensor<256xf32>, %arg400: tensor<1024xf32>, %arg401: tensor<1024xf32>, %arg402: tensor<1x1x1024x256xf32>, %arg403: tensor<3x3x256x256xf32>, %arg404: tensor<1x1x256x1024xf32>, %arg405: tensor<256xf32>, %arg406: tensor<256xf32>, %arg407: tensor<256xf32>, %arg408: tensor<256xf32>, %arg409: tensor<1024xf32>, %arg410: tensor<1024xf32>, %arg411: tensor<1x1x1024x256xf32>, %arg412: tensor<3x3x256x256xf32>, %arg413: tensor<1x1x256x1024xf32>, %arg414: tensor<128xf32>, %arg415: tensor<128xf32>, %arg416: tensor<128xf32>, %arg417: tensor<128xf32>, %arg418: tensor<512xf32>, %arg419: tensor<512xf32>, %arg420: tensor<1x1x512x128xf32>, %arg421: tensor<3x3x128x128xf32>, %arg422: tensor<1x1x128x512xf32>, %arg423: tensor<256xf32>, %arg424: tensor<256xf32>, %arg425: tensor<256xf32>, %arg426: tensor<256xf32>, %arg427: tensor<1024xf32>, %arg428: tensor<1024xf32>, %arg429: tensor<1x1x1024x256xf32>, %arg430: tensor<3x3x256x256xf32>, %arg431: tensor<1x1x256x1024xf32>, %arg432: tensor<256xf32>, %arg433: tensor<256xf32>, %arg434: tensor<256xf32>, %arg435: tensor<256xf32>, %arg436: tensor<1024xf32>, %arg437: tensor<1024xf32>, %arg438: tensor<1x1x1024x256xf32>, %arg439: tensor<3x3x256x256xf32>, %arg440: tensor<1x1x256x1024xf32>, %arg441: tensor<256xf32>, %arg442: tensor<256xf32>, %arg443: tensor<256xf32>, %arg444: tensor<256xf32>, %arg445: tensor<1024xf32>, %arg446: tensor<1024xf32>, %arg447: tensor<1x1x1024x256xf32>, %arg448: tensor<3x3x256x256xf32>, %arg449: tensor<1x1x256x1024xf32>, %arg450: tensor<256xf32>, %arg451: tensor<256xf32>, %arg452: tensor<256xf32>, %arg453: tensor<256xf32>, %arg454: tensor<1024xf32>, %arg455: tensor<1024xf32>, %arg456: tensor<1x1x1024x256xf32>, %arg457: tensor<3x3x256x256xf32>, %arg458: tensor<1x1x256x1024xf32>, %arg459: tensor<256xf32>, %arg460: tensor<256xf32>, %arg461: tensor<256xf32>, %arg462: tensor<256xf32>, %arg463: tensor<1024xf32>, %arg464: tensor<1024xf32>, %arg465: tensor<1x1x1024x256xf32>, %arg466: tensor<3x3x256x256xf32>, %arg467: tensor<1x1x256x1024xf32>, %arg468: tensor<256xf32>, %arg469: tensor<256xf32>, %arg470: tensor<256xf32>, %arg471: tensor<256xf32>, %arg472: tensor<1024xf32>, %arg473: tensor<1024xf32>, %arg474: tensor<1x1x1024x256xf32>, %arg475: tensor<3x3x256x256xf32>, %arg476: tensor<1x1x256x1024xf32>, %arg477: tensor<256xf32>, %arg478: tensor<256xf32>, %arg479: tensor<256xf32>, %arg480: tensor<256xf32>, %arg481: tensor<1024xf32>, %arg482: tensor<1024xf32>, %arg483: tensor<1x1x1024x256xf32>, %arg484: tensor<3x3x256x256xf32>, %arg485: tensor<1x1x256x1024xf32>, %arg486: tensor<256xf32>, %arg487: tensor<256xf32>, %arg488: tensor<256xf32>, %arg489: tensor<256xf32>, %arg490: tensor<1024xf32>, %arg491: tensor<1024xf32>, %arg492: tensor<1x1x1024x256xf32>, %arg493: tensor<3x3x256x256xf32>, %arg494: tensor<1x1x256x1024xf32>, %arg495: tensor<256xf32>, %arg496: tensor<256xf32>, %arg497: tensor<256xf32>, %arg498: tensor<256xf32>, %arg499: tensor<1024xf32>, %arg500: tensor<1024xf32>, %arg501: tensor<1x1x1024x256xf32>, %arg502: tensor<3x3x256x256xf32>, %arg503: tensor<1x1x256x1024xf32>, %arg504: tensor<256xf32>, %arg505: tensor<256xf32>, %arg506: tensor<256xf32>, %arg507: tensor<256xf32>, %arg508: tensor<1024xf32>, %arg509: tensor<1024xf32>, %arg510: tensor<1x1x1024x256xf32>, %arg511: tensor<3x3x256x256xf32>, %arg512: tensor<1x1x256x1024xf32>, %arg513: tensor<128xf32>, %arg514: tensor<128xf32>, %arg515: tensor<128xf32>, %arg516: tensor<128xf32>, %arg517: tensor<512xf32>, %arg518: tensor<512xf32>, %arg519: tensor<1x1x512x128xf32>, %arg520: tensor<3x3x128x128xf32>, %arg521: tensor<1x1x128x512xf32>, %arg522: tensor<256xf32>, %arg523: tensor<256xf32>, %arg524: tensor<256xf32>, %arg525: tensor<256xf32>, %arg526: tensor<1024xf32>, %arg527: tensor<1024xf32>, %arg528: tensor<1x1x1024x256xf32>, %arg529: tensor<3x3x256x256xf32>, %arg530: tensor<1x1x256x1024xf32>, %arg531: tensor<256xf32>, %arg532: tensor<256xf32>, %arg533: tensor<256xf32>, %arg534: tensor<256xf32>, %arg535: tensor<1024xf32>, %arg536: tensor<1024xf32>, %arg537: tensor<1x1x1024x256xf32>, %arg538: tensor<3x3x256x256xf32>, %arg539: tensor<1x1x256x1024xf32>, %arg540: tensor<256xf32>, %arg541: tensor<256xf32>, %arg542: tensor<256xf32>, %arg543: tensor<256xf32>, %arg544: tensor<1024xf32>, %arg545: tensor<1024xf32>, %arg546: tensor<1x1x1024x256xf32>, %arg547: tensor<3x3x256x256xf32>, %arg548: tensor<1x1x256x1024xf32>, %arg549: tensor<512xf32>, %arg550: tensor<512xf32>, %arg551: tensor<512xf32>, %arg552: tensor<512xf32>, %arg553: tensor<2048xf32>, %arg554: tensor<2048xf32>, %arg555: tensor<1x1x1024x512xf32>, %arg556: tensor<3x3x512x512xf32>, %arg557: tensor<1x1x512x2048xf32>, %arg558: tensor<1x1x1024x2048xf32>, %arg559: tensor<2048xf32>, %arg560: tensor<2048xf32>, %arg561: tensor<512xf32>, %arg562: tensor<512xf32>, %arg563: tensor<512xf32>, %arg564: tensor<512xf32>, %arg565: tensor<2048xf32>, %arg566: tensor<2048xf32>, %arg567: tensor<1x1x2048x512xf32>, %arg568: tensor<3x3x512x512xf32>, %arg569: tensor<1x1x512x2048xf32>, %arg570: tensor<512xf32>, %arg571: tensor<512xf32>, %arg572: tensor<512xf32>, %arg573: tensor<512xf32>, %arg574: tensor<2048xf32>, %arg575: tensor<2048xf32>, %arg576: tensor<1x1x2048x512xf32>, %arg577: tensor<3x3x512x512xf32>, %arg578: tensor<1x1x512x2048xf32>, %arg579: tensor<128xf32>, %arg580: tensor<128xf32>, %arg581: tensor<128xf32>, %arg582: tensor<128xf32>, %arg583: tensor<512xf32>, %arg584: tensor<512xf32>, %arg585: tensor<1x1x512x128xf32>, %arg586: tensor<3x3x128x128xf32>, %arg587: tensor<1x1x128x512xf32>, %arg588: tensor<128xf32>, %arg589: tensor<128xf32>, %arg590: tensor<128xf32>, %arg591: tensor<128xf32>, %arg592: tensor<512xf32>, %arg593: tensor<512xf32>, %arg594: tensor<1x1x512x128xf32>, %arg595: tensor<3x3x128x128xf32>, %arg596: tensor<1x1x128x512xf32>, %arg597: tensor<128xf32>, %arg598: tensor<128xf32>, %arg599: tensor<128xf32>, %arg600: tensor<128xf32>, %arg601: tensor<512xf32>, %arg602: tensor<512xf32>, %arg603: tensor<1x1x512x128xf32>, %arg604: tensor<3x3x128x128xf32>, %arg605: tensor<1x1x128x512xf32>, %arg606: tensor<1000xf32>, %arg607: tensor<2048x1000xf32>, %arg608: tensor<64xf32>, %arg609: tensor<64xf32>, %arg610: tensor<7x7x3x64xf32>, %arg611: tensor<64xf32>, %arg612: tensor<64xf32>, %arg613: tensor<64xf32>, %arg614: tensor<64xf32>, %arg615: tensor<256xf32>, %arg616: tensor<256xf32>, %arg617: tensor<256xf32>, %arg618: tensor<256xf32>, %arg619: tensor<64xf32>, %arg620: tensor<64xf32>, %arg621: tensor<64xf32>, %arg622: tensor<64xf32>, %arg623: tensor<256xf32>, %arg624: tensor<256xf32>, %arg625: tensor<128xf32>, %arg626: tensor<128xf32>, %arg627: tensor<128xf32>, %arg628: tensor<128xf32>, %arg629: tensor<512xf32>, %arg630: tensor<512xf32>, %arg631: tensor<128xf32>, %arg632: tensor<128xf32>, %arg633: tensor<128xf32>, %arg634: tensor<128xf32>, %arg635: tensor<512xf32>, %arg636: tensor<512xf32>, %arg637: tensor<128xf32>, %arg638: tensor<128xf32>, %arg639: tensor<128xf32>, %arg640: tensor<128xf32>, %arg641: tensor<512xf32>, %arg642: tensor<512xf32>, %arg643: tensor<128xf32>, %arg644: tensor<128xf32>, %arg645: tensor<128xf32>, %arg646: tensor<128xf32>, %arg647: tensor<512xf32>, %arg648: tensor<512xf32>, %arg649: tensor<128xf32>, %arg650: tensor<128xf32>, %arg651: tensor<128xf32>, %arg652: tensor<128xf32>, %arg653: tensor<512xf32>, %arg654: tensor<512xf32>, %arg655: tensor<128xf32>, %arg656: tensor<128xf32>, %arg657: tensor<128xf32>, %arg658: tensor<128xf32>, %arg659: tensor<512xf32>, %arg660: tensor<512xf32>, %arg661: tensor<128xf32>, %arg662: tensor<128xf32>, %arg663: tensor<128xf32>, %arg664: tensor<128xf32>, %arg665: tensor<512xf32>, %arg666: tensor<512xf32>, %arg667: tensor<128xf32>, %arg668: tensor<128xf32>, %arg669: tensor<128xf32>, %arg670: tensor<128xf32>, %arg671: tensor<512xf32>, %arg672: tensor<512xf32>, %arg673: tensor<128xf32>, %arg674: tensor<128xf32>, %arg675: tensor<128xf32>, %arg676: tensor<128xf32>, %arg677: tensor<512xf32>, %arg678: tensor<512xf32>, %arg679: tensor<128xf32>, %arg680: tensor<128xf32>, %arg681: tensor<128xf32>, %arg682: tensor<128xf32>, %arg683: tensor<512xf32>, %arg684: tensor<512xf32>, %arg685: tensor<64xf32>, %arg686: tensor<64xf32>, %arg687: tensor<64xf32>, %arg688: tensor<64xf32>, %arg689: tensor<256xf32>, %arg690: tensor<256xf32>, %arg691: tensor<128xf32>, %arg692: tensor<128xf32>, %arg693: tensor<128xf32>, %arg694: tensor<128xf32>, %arg695: tensor<512xf32>, %arg696: tensor<512xf32>, %arg697: tensor<128xf32>, %arg698: tensor<128xf32>, %arg699: tensor<128xf32>, %arg700: tensor<128xf32>, %arg701: tensor<512xf32>, %arg702: tensor<512xf32>, %arg703: tensor<128xf32>, %arg704: tensor<128xf32>, %arg705: tensor<128xf32>, %arg706: tensor<128xf32>, %arg707: tensor<512xf32>, %arg708: tensor<512xf32>, %arg709: tensor<128xf32>, %arg710: tensor<128xf32>, %arg711: tensor<128xf32>, %arg712: tensor<128xf32>, %arg713: tensor<512xf32>, %arg714: tensor<512xf32>, %arg715: tensor<128xf32>, %arg716: tensor<128xf32>, %arg717: tensor<128xf32>, %arg718: tensor<128xf32>, %arg719: tensor<512xf32>, %arg720: tensor<512xf32>, %arg721: tensor<128xf32>, %arg722: tensor<128xf32>, %arg723: tensor<128xf32>, %arg724: tensor<128xf32>, %arg725: tensor<512xf32>, %arg726: tensor<512xf32>, %arg727: tensor<128xf32>, %arg728: tensor<128xf32>, %arg729: tensor<128xf32>, %arg730: tensor<128xf32>, %arg731: tensor<512xf32>, %arg732: tensor<512xf32>, %arg733: tensor<256xf32>, %arg734: tensor<256xf32>, %arg735: tensor<256xf32>, %arg736: tensor<256xf32>, %arg737: tensor<1024xf32>, %arg738: tensor<1024xf32>, %arg739: tensor<1024xf32>, %arg740: tensor<1024xf32>, %arg741: tensor<256xf32>, %arg742: tensor<256xf32>, %arg743: tensor<256xf32>, %arg744: tensor<256xf32>, %arg745: tensor<1024xf32>, %arg746: tensor<1024xf32>, %arg747: tensor<256xf32>, %arg748: tensor<256xf32>, %arg749: tensor<256xf32>, %arg750: tensor<256xf32>, %arg751: tensor<1024xf32>, %arg752: tensor<1024xf32>, %arg753: tensor<128xf32>, %arg754: tensor<128xf32>, %arg755: tensor<128xf32>, %arg756: tensor<128xf32>, %arg757: tensor<512xf32>, %arg758: tensor<512xf32>, %arg759: tensor<512xf32>, %arg760: tensor<512xf32>, %arg761: tensor<256xf32>, %arg762: tensor<256xf32>, %arg763: tensor<256xf32>, %arg764: tensor<256xf32>, %arg765: tensor<1024xf32>, %arg766: tensor<1024xf32>, %arg767: tensor<256xf32>, %arg768: tensor<256xf32>, %arg769: tensor<256xf32>, %arg770: tensor<256xf32>, %arg771: tensor<1024xf32>, %arg772: tensor<1024xf32>, %arg773: tensor<256xf32>, %arg774: tensor<256xf32>, %arg775: tensor<256xf32>, %arg776: tensor<256xf32>, %arg777: tensor<1024xf32>, %arg778: tensor<1024xf32>, %arg779: tensor<256xf32>, %arg780: tensor<256xf32>, %arg781: tensor<256xf32>, %arg782: tensor<256xf32>, %arg783: tensor<1024xf32>, %arg784: tensor<1024xf32>, %arg785: tensor<256xf32>, %arg786: tensor<256xf32>, %arg787: tensor<256xf32>, %arg788: tensor<256xf32>, %arg789: tensor<1024xf32>, %arg790: tensor<1024xf32>, %arg791: tensor<256xf32>, %arg792: tensor<256xf32>, %arg793: tensor<256xf32>, %arg794: tensor<256xf32>, %arg795: tensor<1024xf32>, %arg796: tensor<1024xf32>, %arg797: tensor<256xf32>, %arg798: tensor<256xf32>, %arg799: tensor<256xf32>, %arg800: tensor<256xf32>, %arg801: tensor<1024xf32>, %arg802: tensor<1024xf32>, %arg803: tensor<256xf32>, %arg804: tensor<256xf32>, %arg805: tensor<256xf32>, %arg806: tensor<256xf32>, %arg807: tensor<1024xf32>, %arg808: tensor<1024xf32>, %arg809: tensor<256xf32>, %arg810: tensor<256xf32>, %arg811: tensor<256xf32>, %arg812: tensor<256xf32>, %arg813: tensor<1024xf32>, %arg814: tensor<1024xf32>, %arg815: tensor<256xf32>, %arg816: tensor<256xf32>, %arg817: tensor<256xf32>, %arg818: tensor<256xf32>, %arg819: tensor<1024xf32>, %arg820: tensor<1024xf32>, %arg821: tensor<128xf32>, %arg822: tensor<128xf32>, %arg823: tensor<128xf32>, %arg824: tensor<128xf32>, %arg825: tensor<512xf32>, %arg826: tensor<512xf32>, %arg827: tensor<256xf32>, %arg828: tensor<256xf32>, %arg829: tensor<256xf32>, %arg830: tensor<256xf32>, %arg831: tensor<1024xf32>, %arg832: tensor<1024xf32>, %arg833: tensor<256xf32>, %arg834: tensor<256xf32>, %arg835: tensor<256xf32>, %arg836: tensor<256xf32>, %arg837: tensor<1024xf32>, %arg838: tensor<1024xf32>, %arg839: tensor<256xf32>, %arg840: tensor<256xf32>, %arg841: tensor<256xf32>, %arg842: tensor<256xf32>, %arg843: tensor<1024xf32>, %arg844: tensor<1024xf32>, %arg845: tensor<256xf32>, %arg846: tensor<256xf32>, %arg847: tensor<256xf32>, %arg848: tensor<256xf32>, %arg849: tensor<1024xf32>, %arg850: tensor<1024xf32>, %arg851: tensor<256xf32>, %arg852: tensor<256xf32>, %arg853: tensor<256xf32>, %arg854: tensor<256xf32>, %arg855: tensor<1024xf32>, %arg856: tensor<1024xf32>, %arg857: tensor<256xf32>, %arg858: tensor<256xf32>, %arg859: tensor<256xf32>, %arg860: tensor<256xf32>, %arg861: tensor<1024xf32>, %arg862: tensor<1024xf32>, %arg863: tensor<256xf32>, %arg864: tensor<256xf32>, %arg865: tensor<256xf32>, %arg866: tensor<256xf32>, %arg867: tensor<1024xf32>, %arg868: tensor<1024xf32>, %arg869: tensor<256xf32>, %arg870: tensor<256xf32>, %arg871: tensor<256xf32>, %arg872: tensor<256xf32>, %arg873: tensor<1024xf32>, %arg874: tensor<1024xf32>, %arg875: tensor<256xf32>, %arg876: tensor<256xf32>, %arg877: tensor<256xf32>, %arg878: tensor<256xf32>, %arg879: tensor<1024xf32>, %arg880: tensor<1024xf32>, %arg881: tensor<256xf32>, %arg882: tensor<256xf32>, %arg883: tensor<256xf32>, %arg884: tensor<256xf32>, %arg885: tensor<1024xf32>, %arg886: tensor<1024xf32>, %arg887: tensor<128xf32>, %arg888: tensor<128xf32>, %arg889: tensor<128xf32>, %arg890: tensor<128xf32>, %arg891: tensor<512xf32>, %arg892: tensor<512xf32>, %arg893: tensor<256xf32>, %arg894: tensor<256xf32>, %arg895: tensor<256xf32>, %arg896: tensor<256xf32>, %arg897: tensor<1024xf32>, %arg898: tensor<1024xf32>, %arg899: tensor<256xf32>, %arg900: tensor<256xf32>, %arg901: tensor<256xf32>, %arg902: tensor<256xf32>, %arg903: tensor<1024xf32>, %arg904: tensor<1024xf32>, %arg905: tensor<256xf32>, %arg906: tensor<256xf32>, %arg907: tensor<256xf32>, %arg908: tensor<256xf32>, %arg909: tensor<1024xf32>, %arg910: tensor<1024xf32>, %arg911: tensor<256xf32>, %arg912: tensor<256xf32>, %arg913: tensor<256xf32>, %arg914: tensor<256xf32>, %arg915: tensor<1024xf32>, %arg916: tensor<1024xf32>, %arg917: tensor<256xf32>, %arg918: tensor<256xf32>, %arg919: tensor<256xf32>, %arg920: tensor<256xf32>, %arg921: tensor<1024xf32>, %arg922: tensor<1024xf32>, %arg923: tensor<256xf32>, %arg924: tensor<256xf32>, %arg925: tensor<256xf32>, %arg926: tensor<256xf32>, %arg927: tensor<1024xf32>, %arg928: tensor<1024xf32>, %arg929: tensor<256xf32>, %arg930: tensor<256xf32>, %arg931: tensor<256xf32>, %arg932: tensor<256xf32>, %arg933: tensor<1024xf32>, %arg934: tensor<1024xf32>, %arg935: tensor<256xf32>, %arg936: tensor<256xf32>, %arg937: tensor<256xf32>, %arg938: tensor<256xf32>, %arg939: tensor<1024xf32>, %arg940: tensor<1024xf32>, %arg941: tensor<256xf32>, %arg942: tensor<256xf32>, %arg943: tensor<256xf32>, %arg944: tensor<256xf32>, %arg945: tensor<1024xf32>, %arg946: tensor<1024xf32>, %arg947: tensor<256xf32>, %arg948: tensor<256xf32>, %arg949: tensor<256xf32>, %arg950: tensor<256xf32>, %arg951: tensor<1024xf32>, %arg952: tensor<1024xf32>, %arg953: tensor<128xf32>, %arg954: tensor<128xf32>, %arg955: tensor<128xf32>, %arg956: tensor<128xf32>, %arg957: tensor<512xf32>, %arg958: tensor<512xf32>, %arg959: tensor<256xf32>, %arg960: tensor<256xf32>, %arg961: tensor<256xf32>, %arg962: tensor<256xf32>, %arg963: tensor<1024xf32>, %arg964: tensor<1024xf32>, %arg965: tensor<256xf32>, %arg966: tensor<256xf32>, %arg967: tensor<256xf32>, %arg968: tensor<256xf32>, %arg969: tensor<1024xf32>, %arg970: tensor<1024xf32>, %arg971: tensor<256xf32>, %arg972: tensor<256xf32>, %arg973: tensor<256xf32>, %arg974: tensor<256xf32>, %arg975: tensor<1024xf32>, %arg976: tensor<1024xf32>, %arg977: tensor<512xf32>, %arg978: tensor<512xf32>, %arg979: tensor<512xf32>, %arg980: tensor<512xf32>, %arg981: tensor<2048xf32>, %arg982: tensor<2048xf32>, %arg983: tensor<2048xf32>, %arg984: tensor<2048xf32>, %arg985: tensor<512xf32>, %arg986: tensor<512xf32>, %arg987: tensor<512xf32>, %arg988: tensor<512xf32>, %arg989: tensor<2048xf32>, %arg990: tensor<2048xf32>, %arg991: tensor<512xf32>, %arg992: tensor<512xf32>, %arg993: tensor<512xf32>, %arg994: tensor<512xf32>, %arg995: tensor<2048xf32>, %arg996: tensor<2048xf32>, %arg997: tensor<128xf32>, %arg998: tensor<128xf32>, %arg999: tensor<128xf32>, %arg1000: tensor<128xf32>, %arg1001: tensor<512xf32>, %arg1002: tensor<512xf32>, %arg1003: tensor<128xf32>, %arg1004: tensor<128xf32>, %arg1005: tensor<128xf32>, %arg1006: tensor<128xf32>, %arg1007: tensor<512xf32>, %arg1008: tensor<512xf32>, %arg1009: tensor<128xf32>, %arg1010: tensor<128xf32>, %arg1011: tensor<128xf32>, %arg1012: tensor<128xf32>, %arg1013: tensor<512xf32>, %arg1014: tensor<512xf32>, %arg1015: tensor<64xf32>, %arg1016: tensor<64xf32>, %arg1017: tensor<4x256x224x224x3xf16>, %arg1018: tensor<4x256xi32>) -> (tensor<4xf32> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %0:2 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg181, %arg182, %arg183, %arg184, %arg185, %arg186, %arg187, %arg188, %arg189, %arg190, %arg191, %arg192, %arg193, %arg194, %arg195, %arg196, %arg197, %arg198, %arg199, %arg200, %arg201, %arg202, %arg203, %arg204, %arg205, %arg206, %arg207, %arg208, %arg209, %arg210, %arg211, %arg212, %arg213, %arg214, %arg215, %arg216, %arg217, %arg218, %arg219, %arg220, %arg221, %arg222, %arg223, %arg224, %arg225, %arg226, %arg227, %arg228, %arg229, %arg230, %arg231, %arg232, %arg233, %arg234, %arg235, %arg236, %arg237, %arg238, %arg239, %arg240, %arg241, %arg242, %arg243, %arg244, %arg245, %arg246, %arg247, %arg248, %arg249, %arg250, %arg251, %arg252, %arg253, %arg254, %arg255, %arg256, %arg257, %arg258, %arg259, %arg260, %arg261, %arg262, %arg263, %arg264, %arg265, %arg266, %arg267, %arg268, %arg269, %arg270, %arg271, %arg272, %arg273, %arg274, %arg275, %arg276, %arg277, %arg278, %arg279, %arg280, %arg281, %arg282, %arg283, %arg284, %arg285, %arg286, %arg287, %arg288, %arg289, %arg290, %arg291, %arg292, %arg293, %arg294, %arg295, %arg296, %arg297, %arg298, %arg299, %arg300, %arg301, %arg302, %arg303, %arg304, %arg305, %arg306, %arg307, %arg308, %arg309, %arg310, %arg311, %arg312, %arg313, %arg314, %arg315, %arg316, %arg317, %arg318, %arg319, %arg320, %arg321, %arg322, %arg323, %arg324, %arg325, %arg326, %arg327, %arg328, %arg329, %arg330, %arg331, %arg332, %arg333, %arg334, %arg335, %arg336, %arg337, %arg338, %arg339, %arg340, %arg341, %arg342, %arg343, %arg344, %arg345, %arg346, %arg347, %arg348, %arg349, %arg350, %arg351, %arg352, %arg353, %arg354, %arg355, %arg356, %arg357, %arg358, %arg359, %arg360, %arg361, %arg362, %arg363, %arg364, %arg365, %arg366, %arg367, %arg368, %arg369, %arg370, %arg371, %arg372, %arg373, %arg374, %arg375, %arg376, %arg377, %arg378, %arg379, %arg380, %arg381, %arg382, %arg383, %arg384, %arg385, %arg386, %arg387, %arg388, %arg389, %arg390, %arg391, %arg392, %arg393, %arg394, %arg395, %arg396, %arg397, %arg398, %arg399, %arg400, %arg401, %arg402, %arg403, %arg404, %arg405, %arg406, %arg407, %arg408, %arg409, %arg410, %arg411, %arg412, %arg413, %arg414, %arg415, %arg416, %arg417, %arg418, %arg419, %arg420, %arg421, %arg422, %arg423, %arg424, %arg425, %arg426, %arg427, %arg428, %arg429, %arg430, %arg431, %arg432, %arg433, %arg434, %arg435, %arg436, %arg437, %arg438, %arg439, %arg440, %arg441, %arg442, %arg443, %arg444, %arg445, %arg446, %arg447, %arg448, %arg449, %arg450, %arg451, %arg452, %arg453, %arg454, %arg455, %arg456, %arg457, %arg458, %arg459, %arg460, %arg461, %arg462, %arg463, %arg464, %arg465, %arg466, %arg467, %arg468, %arg469, %arg470, %arg471, %arg472, %arg473, %arg474, %arg475, %arg476, %arg477, %arg478, %arg479, %arg480, %arg481, %arg482, %arg483, %arg484, %arg485, %arg486, %arg487, %arg488, %arg489, %arg490, %arg491, %arg492, %arg493, %arg494, %arg495, %arg496, %arg497, %arg498, %arg499, %arg500, %arg501, %arg502, %arg503, %arg504, %arg505, %arg506, %arg507, %arg508, %arg509, %arg510, %arg511, %arg512, %arg513, %arg514, %arg515, %arg516, %arg517, %arg518, %arg519, %arg520, %arg521, %arg522, %arg523, %arg524, %arg525, %arg526, %arg527, %arg528, %arg529, %arg530, %arg531, %arg532, %arg533, %arg534, %arg535, %arg536, %arg537, %arg538, %arg539, %arg540, %arg541, %arg542, %arg543, %arg544, %arg545, %arg546, %arg547, %arg548, %arg549, %arg550, %arg551, %arg552, %arg553, %arg554, %arg555, %arg556, %arg557, %arg558, %arg559, %arg560, %arg561, %arg562, %arg563, %arg564, %arg565, %arg566, %arg567, %arg568, %arg569, %arg570, %arg571, %arg572, %arg573, %arg574, %arg575, %arg576, %arg577, %arg578, %arg579, %arg580, %arg581, %arg582, %arg583, %arg584, %arg585, %arg586, %arg587, %arg588, %arg589, %arg590, %arg591, %arg592, %arg593, %arg594, %arg595, %arg596, %arg597, %arg598, %arg599, %arg600, %arg601, %arg602, %arg603, %arg604, %arg605, %arg606, %arg607, %arg608, %arg609, %arg610, %arg611, %arg612, %arg613, %arg614, %arg615, %arg616, %arg617, %arg618, %arg619, %arg620, %arg621, %arg622, %arg623, %arg624, %arg625, %arg626, %arg627, %arg628, %arg629, %arg630, %arg631, %arg632, %arg633, %arg634, %arg635, %arg636, %arg637, %arg638, %arg639, %arg640, %arg641, %arg642, %arg643, %arg644, %arg645, %arg646, %arg647, %arg648, %arg649, %arg650, %arg651, %arg652, %arg653, %arg654, %arg655, %arg656, %arg657, %arg658, %arg659, %arg660, %arg661, %arg662, %arg663, %arg664, %arg665, %arg666, %arg667, %arg668, %arg669, %arg670, %arg671, %arg672, %arg673, %arg674, %arg675, %arg676, %arg677, %arg678, %arg679, %arg680, %arg681, %arg682, %arg683, %arg684, %arg685, %arg686, %arg687, %arg688, %arg689, %arg690, %arg691, %arg692, %arg693, %arg694, %arg695, %arg696, %arg697, %arg698, %arg699, %arg700, %arg701, %arg702, %arg703, %arg704, %arg705, %arg706, %arg707, %arg708, %arg709, %arg710, %arg711, %arg712, %arg713, %arg714, %arg715, %arg716, %arg717, %arg718, %arg719, %arg720, %arg721, %arg722, %arg723, %arg724, %arg725, %arg726, %arg727, %arg728, %arg729, %arg730, %arg731, %arg732, %arg733, %arg734, %arg735, %arg736, %arg737, %arg738, %arg739, %arg740, %arg741, %arg742, %arg743, %arg744, %arg745, %arg746, %arg747, %arg748, %arg749, %arg750, %arg751, %arg752, %arg753, %arg754, %arg755, %arg756, %arg757, %arg758, %arg759, %arg760, %arg761, %arg762, %arg763, %arg764, %arg765, %arg766, %arg767, %arg768, %arg769, %arg770, %arg771, %arg772, %arg773, %arg774, %arg775, %arg776, %arg777, %arg778, %arg779, %arg780, %arg781, %arg782, %arg783, %arg784, %arg785, %arg786, %arg787, %arg788, %arg789, %arg790, %arg791, %arg792, %arg793, %arg794, %arg795, %arg796, %arg797, %arg798, %arg799, %arg800, %arg801, %arg802, %arg803, %arg804, %arg805, %arg806, %arg807, %arg808, %arg809, %arg810, %arg811, %arg812, %arg813, %arg814, %arg815, %arg816, %arg817, %arg818, %arg819, %arg820, %arg821, %arg822, %arg823, %arg824, %arg825, %arg826, %arg827, %arg828, %arg829, %arg830, %arg831, %arg832, %arg833, %arg834, %arg835, %arg836, %arg837, %arg838, %arg839, %arg840, %arg841, %arg842, %arg843, %arg844, %arg845, %arg846, %arg847, %arg848, %arg849, %arg850, %arg851, %arg852, %arg853, %arg854, %arg855, %arg856, %arg857, %arg858, %arg859, %arg860, %arg861, %arg862, %arg863, %arg864, %arg865, %arg866, %arg867, %arg868, %arg869, %arg870, %arg871, %arg872, %arg873, %arg874, %arg875, %arg876, %arg877, %arg878, %arg879, %arg880, %arg881, %arg882, %arg883, %arg884, %arg885, %arg886, %arg887, %arg888, %arg889, %arg890, %arg891, %arg892, %arg893, %arg894, %arg895, %arg896, %arg897, %arg898, %arg899, %arg900, %arg901, %arg902, %arg903, %arg904, %arg905, %arg906, %arg907, %arg908, %arg909, %arg910, %arg911, %arg912, %arg913, %arg914, %arg915, %arg916, %arg917, %arg918, %arg919, %arg920, %arg921, %arg922, %arg923, %arg924, %arg925, %arg926, %arg927, %arg928, %arg929, %arg930, %arg931, %arg932, %arg933, %arg934, %arg935, %arg936, %arg937, %arg938, %arg939, %arg940, %arg941, %arg942, %arg943, %arg944, %arg945, %arg946, %arg947, %arg948, %arg949, %arg950, %arg951, %arg952, %arg953, %arg954, %arg955, %arg956, %arg957, %arg958, %arg959, %arg960, %arg961, %arg962, %arg963, %arg964, %arg965, %arg966, %arg967, %arg968, %arg969, %arg970, %arg971, %arg972, %arg973, %arg974, %arg975, %arg976, %arg977, %arg978, %arg979, %arg980, %arg981, %arg982, %arg983, %arg984, %arg985, %arg986, %arg987, %arg988, %arg989, %arg990, %arg991, %arg992, %arg993, %arg994, %arg995, %arg996, %arg997, %arg998, %arg999, %arg1000, %arg1001, %arg1002, %arg1003, %arg1004, %arg1005, %arg1006, %arg1007, %arg1008, %arg1009, %arg1010, %arg1011, %arg1012, %arg1013, %arg1014, %arg1015, %arg1016, %arg1017, %arg1018) in_shardings=[<@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{"batch"}, {}, {}, {}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}]>, <@mesh, [{"batch"}]>] manual_axes={"batch"} (%arg1019: tensor<64xf32>, %arg1020: tensor<64xf32>, %arg1021: tensor<64xf32>, %arg1022: tensor<64xf32>, %arg1023: tensor<256xf32>, %arg1024: tensor<256xf32>, %arg1025: tensor<1x1x64x64xf32>, %arg1026: tensor<3x3x64x64xf32>, %arg1027: tensor<1x1x64x256xf32>, %arg1028: tensor<1x1x64x256xf32>, %arg1029: tensor<256xf32>, %arg1030: tensor<256xf32>, %arg1031: tensor<64xf32>, %arg1032: tensor<64xf32>, %arg1033: tensor<64xf32>, %arg1034: tensor<64xf32>, %arg1035: tensor<256xf32>, %arg1036: tensor<256xf32>, %arg1037: tensor<1x1x256x64xf32>, %arg1038: tensor<3x3x64x64xf32>, %arg1039: tensor<1x1x64x256xf32>, %arg1040: tensor<128xf32>, %arg1041: tensor<128xf32>, %arg1042: tensor<128xf32>, %arg1043: tensor<128xf32>, %arg1044: tensor<512xf32>, %arg1045: tensor<512xf32>, %arg1046: tensor<1x1x512x128xf32>, %arg1047: tensor<3x3x128x128xf32>, %arg1048: tensor<1x1x128x512xf32>, %arg1049: tensor<128xf32>, %arg1050: tensor<128xf32>, %arg1051: tensor<128xf32>, %arg1052: tensor<128xf32>, %arg1053: tensor<512xf32>, %arg1054: tensor<512xf32>, %arg1055: tensor<1x1x512x128xf32>, %arg1056: tensor<3x3x128x128xf32>, %arg1057: tensor<1x1x128x512xf32>, %arg1058: tensor<128xf32>, %arg1059: tensor<128xf32>, %arg1060: tensor<128xf32>, %arg1061: tensor<128xf32>, %arg1062: tensor<512xf32>, %arg1063: tensor<512xf32>, %arg1064: tensor<1x1x512x128xf32>, %arg1065: tensor<3x3x128x128xf32>, %arg1066: tensor<1x1x128x512xf32>, %arg1067: tensor<128xf32>, %arg1068: tensor<128xf32>, %arg1069: tensor<128xf32>, %arg1070: tensor<128xf32>, %arg1071: tensor<512xf32>, %arg1072: tensor<512xf32>, %arg1073: tensor<1x1x512x128xf32>, %arg1074: tensor<3x3x128x128xf32>, %arg1075: tensor<1x1x128x512xf32>, %arg1076: tensor<128xf32>, %arg1077: tensor<128xf32>, %arg1078: tensor<128xf32>, %arg1079: tensor<128xf32>, %arg1080: tensor<512xf32>, %arg1081: tensor<512xf32>, %arg1082: tensor<1x1x512x128xf32>, %arg1083: tensor<3x3x128x128xf32>, %arg1084: tensor<1x1x128x512xf32>, %arg1085: tensor<128xf32>, %arg1086: tensor<128xf32>, %arg1087: tensor<128xf32>, %arg1088: tensor<128xf32>, %arg1089: tensor<512xf32>, %arg1090: tensor<512xf32>, %arg1091: tensor<1x1x512x128xf32>, %arg1092: tensor<3x3x128x128xf32>, %arg1093: tensor<1x1x128x512xf32>, %arg1094: tensor<128xf32>, %arg1095: tensor<128xf32>, %arg1096: tensor<128xf32>, %arg1097: tensor<128xf32>, %arg1098: tensor<512xf32>, %arg1099: tensor<512xf32>, %arg1100: tensor<1x1x512x128xf32>, %arg1101: tensor<3x3x128x128xf32>, %arg1102: tensor<1x1x128x512xf32>, %arg1103: tensor<128xf32>, %arg1104: tensor<128xf32>, %arg1105: tensor<128xf32>, %arg1106: tensor<128xf32>, %arg1107: tensor<512xf32>, %arg1108: tensor<512xf32>, %arg1109: tensor<1x1x512x128xf32>, %arg1110: tensor<3x3x128x128xf32>, %arg1111: tensor<1x1x128x512xf32>, %arg1112: tensor<128xf32>, %arg1113: tensor<128xf32>, %arg1114: tensor<128xf32>, %arg1115: tensor<128xf32>, %arg1116: tensor<512xf32>, %arg1117: tensor<512xf32>, %arg1118: tensor<1x1x512x128xf32>, %arg1119: tensor<3x3x128x128xf32>, %arg1120: tensor<1x1x128x512xf32>, %arg1121: tensor<128xf32>, %arg1122: tensor<128xf32>, %arg1123: tensor<128xf32>, %arg1124: tensor<128xf32>, %arg1125: tensor<512xf32>, %arg1126: tensor<512xf32>, %arg1127: tensor<1x1x512x128xf32>, %arg1128: tensor<3x3x128x128xf32>, %arg1129: tensor<1x1x128x512xf32>, %arg1130: tensor<64xf32>, %arg1131: tensor<64xf32>, %arg1132: tensor<64xf32>, %arg1133: tensor<64xf32>, %arg1134: tensor<256xf32>, %arg1135: tensor<256xf32>, %arg1136: tensor<1x1x256x64xf32>, %arg1137: tensor<3x3x64x64xf32>, %arg1138: tensor<1x1x64x256xf32>, %arg1139: tensor<128xf32>, %arg1140: tensor<128xf32>, %arg1141: tensor<128xf32>, %arg1142: tensor<128xf32>, %arg1143: tensor<512xf32>, %arg1144: tensor<512xf32>, %arg1145: tensor<1x1x512x128xf32>, %arg1146: tensor<3x3x128x128xf32>, %arg1147: tensor<1x1x128x512xf32>, %arg1148: tensor<128xf32>, %arg1149: tensor<128xf32>, %arg1150: tensor<128xf32>, %arg1151: tensor<128xf32>, %arg1152: tensor<512xf32>, %arg1153: tensor<512xf32>, %arg1154: tensor<1x1x512x128xf32>, %arg1155: tensor<3x3x128x128xf32>, %arg1156: tensor<1x1x128x512xf32>, %arg1157: tensor<128xf32>, %arg1158: tensor<128xf32>, %arg1159: tensor<128xf32>, %arg1160: tensor<128xf32>, %arg1161: tensor<512xf32>, %arg1162: tensor<512xf32>, %arg1163: tensor<1x1x512x128xf32>, %arg1164: tensor<3x3x128x128xf32>, %arg1165: tensor<1x1x128x512xf32>, %arg1166: tensor<128xf32>, %arg1167: tensor<128xf32>, %arg1168: tensor<128xf32>, %arg1169: tensor<128xf32>, %arg1170: tensor<512xf32>, %arg1171: tensor<512xf32>, %arg1172: tensor<1x1x512x128xf32>, %arg1173: tensor<3x3x128x128xf32>, %arg1174: tensor<1x1x128x512xf32>, %arg1175: tensor<128xf32>, %arg1176: tensor<128xf32>, %arg1177: tensor<128xf32>, %arg1178: tensor<128xf32>, %arg1179: tensor<512xf32>, %arg1180: tensor<512xf32>, %arg1181: tensor<1x1x512x128xf32>, %arg1182: tensor<3x3x128x128xf32>, %arg1183: tensor<1x1x128x512xf32>, %arg1184: tensor<128xf32>, %arg1185: tensor<128xf32>, %arg1186: tensor<128xf32>, %arg1187: tensor<128xf32>, %arg1188: tensor<512xf32>, %arg1189: tensor<512xf32>, %arg1190: tensor<1x1x512x128xf32>, %arg1191: tensor<3x3x128x128xf32>, %arg1192: tensor<1x1x128x512xf32>, %arg1193: tensor<128xf32>, %arg1194: tensor<128xf32>, %arg1195: tensor<128xf32>, %arg1196: tensor<128xf32>, %arg1197: tensor<512xf32>, %arg1198: tensor<512xf32>, %arg1199: tensor<1x1x512x128xf32>, %arg1200: tensor<3x3x128x128xf32>, %arg1201: tensor<1x1x128x512xf32>, %arg1202: tensor<256xf32>, %arg1203: tensor<256xf32>, %arg1204: tensor<256xf32>, %arg1205: tensor<256xf32>, %arg1206: tensor<1024xf32>, %arg1207: tensor<1024xf32>, %arg1208: tensor<1x1x512x256xf32>, %arg1209: tensor<3x3x256x256xf32>, %arg1210: tensor<1x1x256x1024xf32>, %arg1211: tensor<1x1x512x1024xf32>, %arg1212: tensor<1024xf32>, %arg1213: tensor<1024xf32>, %arg1214: tensor<256xf32>, %arg1215: tensor<256xf32>, %arg1216: tensor<256xf32>, %arg1217: tensor<256xf32>, %arg1218: tensor<1024xf32>, %arg1219: tensor<1024xf32>, %arg1220: tensor<1x1x1024x256xf32>, %arg1221: tensor<3x3x256x256xf32>, %arg1222: tensor<1x1x256x1024xf32>, %arg1223: tensor<256xf32>, %arg1224: tensor<256xf32>, %arg1225: tensor<256xf32>, %arg1226: tensor<256xf32>, %arg1227: tensor<1024xf32>, %arg1228: tensor<1024xf32>, %arg1229: tensor<1x1x1024x256xf32>, %arg1230: tensor<3x3x256x256xf32>, %arg1231: tensor<1x1x256x1024xf32>, %arg1232: tensor<128xf32>, %arg1233: tensor<128xf32>, %arg1234: tensor<128xf32>, %arg1235: tensor<128xf32>, %arg1236: tensor<512xf32>, %arg1237: tensor<512xf32>, %arg1238: tensor<1x1x256x128xf32>, %arg1239: tensor<3x3x128x128xf32>, %arg1240: tensor<1x1x128x512xf32>, %arg1241: tensor<1x1x256x512xf32>, %arg1242: tensor<512xf32>, %arg1243: tensor<512xf32>, %arg1244: tensor<256xf32>, %arg1245: tensor<256xf32>, %arg1246: tensor<256xf32>, %arg1247: tensor<256xf32>, %arg1248: tensor<1024xf32>, %arg1249: tensor<1024xf32>, %arg1250: tensor<1x1x1024x256xf32>, %arg1251: tensor<3x3x256x256xf32>, %arg1252: tensor<1x1x256x1024xf32>, %arg1253: tensor<256xf32>, %arg1254: tensor<256xf32>, %arg1255: tensor<256xf32>, %arg1256: tensor<256xf32>, %arg1257: tensor<1024xf32>, %arg1258: tensor<1024xf32>, %arg1259: tensor<1x1x1024x256xf32>, %arg1260: tensor<3x3x256x256xf32>, %arg1261: tensor<1x1x256x1024xf32>, %arg1262: tensor<256xf32>, %arg1263: tensor<256xf32>, %arg1264: tensor<256xf32>, %arg1265: tensor<256xf32>, %arg1266: tensor<1024xf32>, %arg1267: tensor<1024xf32>, %arg1268: tensor<1x1x1024x256xf32>, %arg1269: tensor<3x3x256x256xf32>, %arg1270: tensor<1x1x256x1024xf32>, %arg1271: tensor<256xf32>, %arg1272: tensor<256xf32>, %arg1273: tensor<256xf32>, %arg1274: tensor<256xf32>, %arg1275: tensor<1024xf32>, %arg1276: tensor<1024xf32>, %arg1277: tensor<1x1x1024x256xf32>, %arg1278: tensor<3x3x256x256xf32>, %arg1279: tensor<1x1x256x1024xf32>, %arg1280: tensor<256xf32>, %arg1281: tensor<256xf32>, %arg1282: tensor<256xf32>, %arg1283: tensor<256xf32>, %arg1284: tensor<1024xf32>, %arg1285: tensor<1024xf32>, %arg1286: tensor<1x1x1024x256xf32>, %arg1287: tensor<3x3x256x256xf32>, %arg1288: tensor<1x1x256x1024xf32>, %arg1289: tensor<256xf32>, %arg1290: tensor<256xf32>, %arg1291: tensor<256xf32>, %arg1292: tensor<256xf32>, %arg1293: tensor<1024xf32>, %arg1294: tensor<1024xf32>, %arg1295: tensor<1x1x1024x256xf32>, %arg1296: tensor<3x3x256x256xf32>, %arg1297: tensor<1x1x256x1024xf32>, %arg1298: tensor<256xf32>, %arg1299: tensor<256xf32>, %arg1300: tensor<256xf32>, %arg1301: tensor<256xf32>, %arg1302: tensor<1024xf32>, %arg1303: tensor<1024xf32>, %arg1304: tensor<1x1x1024x256xf32>, %arg1305: tensor<3x3x256x256xf32>, %arg1306: tensor<1x1x256x1024xf32>, %arg1307: tensor<256xf32>, %arg1308: tensor<256xf32>, %arg1309: tensor<256xf32>, %arg1310: tensor<256xf32>, %arg1311: tensor<1024xf32>, %arg1312: tensor<1024xf32>, %arg1313: tensor<1x1x1024x256xf32>, %arg1314: tensor<3x3x256x256xf32>, %arg1315: tensor<1x1x256x1024xf32>, %arg1316: tensor<256xf32>, %arg1317: tensor<256xf32>, %arg1318: tensor<256xf32>, %arg1319: tensor<256xf32>, %arg1320: tensor<1024xf32>, %arg1321: tensor<1024xf32>, %arg1322: tensor<1x1x1024x256xf32>, %arg1323: tensor<3x3x256x256xf32>, %arg1324: tensor<1x1x256x1024xf32>, %arg1325: tensor<256xf32>, %arg1326: tensor<256xf32>, %arg1327: tensor<256xf32>, %arg1328: tensor<256xf32>, %arg1329: tensor<1024xf32>, %arg1330: tensor<1024xf32>, %arg1331: tensor<1x1x1024x256xf32>, %arg1332: tensor<3x3x256x256xf32>, %arg1333: tensor<1x1x256x1024xf32>, %arg1334: tensor<128xf32>, %arg1335: tensor<128xf32>, %arg1336: tensor<128xf32>, %arg1337: tensor<128xf32>, %arg1338: tensor<512xf32>, %arg1339: tensor<512xf32>, %arg1340: tensor<1x1x512x128xf32>, %arg1341: tensor<3x3x128x128xf32>, %arg1342: tensor<1x1x128x512xf32>, %arg1343: tensor<256xf32>, %arg1344: tensor<256xf32>, %arg1345: tensor<256xf32>, %arg1346: tensor<256xf32>, %arg1347: tensor<1024xf32>, %arg1348: tensor<1024xf32>, %arg1349: tensor<1x1x1024x256xf32>, %arg1350: tensor<3x3x256x256xf32>, %arg1351: tensor<1x1x256x1024xf32>, %arg1352: tensor<256xf32>, %arg1353: tensor<256xf32>, %arg1354: tensor<256xf32>, %arg1355: tensor<256xf32>, %arg1356: tensor<1024xf32>, %arg1357: tensor<1024xf32>, %arg1358: tensor<1x1x1024x256xf32>, %arg1359: tensor<3x3x256x256xf32>, %arg1360: tensor<1x1x256x1024xf32>, %arg1361: tensor<256xf32>, %arg1362: tensor<256xf32>, %arg1363: tensor<256xf32>, %arg1364: tensor<256xf32>, %arg1365: tensor<1024xf32>, %arg1366: tensor<1024xf32>, %arg1367: tensor<1x1x1024x256xf32>, %arg1368: tensor<3x3x256x256xf32>, %arg1369: tensor<1x1x256x1024xf32>, %arg1370: tensor<256xf32>, %arg1371: tensor<256xf32>, %arg1372: tensor<256xf32>, %arg1373: tensor<256xf32>, %arg1374: tensor<1024xf32>, %arg1375: tensor<1024xf32>, %arg1376: tensor<1x1x1024x256xf32>, %arg1377: tensor<3x3x256x256xf32>, %arg1378: tensor<1x1x256x1024xf32>, %arg1379: tensor<256xf32>, %arg1380: tensor<256xf32>, %arg1381: tensor<256xf32>, %arg1382: tensor<256xf32>, %arg1383: tensor<1024xf32>, %arg1384: tensor<1024xf32>, %arg1385: tensor<1x1x1024x256xf32>, %arg1386: tensor<3x3x256x256xf32>, %arg1387: tensor<1x1x256x1024xf32>, %arg1388: tensor<256xf32>, %arg1389: tensor<256xf32>, %arg1390: tensor<256xf32>, %arg1391: tensor<256xf32>, %arg1392: tensor<1024xf32>, %arg1393: tensor<1024xf32>, %arg1394: tensor<1x1x1024x256xf32>, %arg1395: tensor<3x3x256x256xf32>, %arg1396: tensor<1x1x256x1024xf32>, %arg1397: tensor<256xf32>, %arg1398: tensor<256xf32>, %arg1399: tensor<256xf32>, %arg1400: tensor<256xf32>, %arg1401: tensor<1024xf32>, %arg1402: tensor<1024xf32>, %arg1403: tensor<1x1x1024x256xf32>, %arg1404: tensor<3x3x256x256xf32>, %arg1405: tensor<1x1x256x1024xf32>, %arg1406: tensor<256xf32>, %arg1407: tensor<256xf32>, %arg1408: tensor<256xf32>, %arg1409: tensor<256xf32>, %arg1410: tensor<1024xf32>, %arg1411: tensor<1024xf32>, %arg1412: tensor<1x1x1024x256xf32>, %arg1413: tensor<3x3x256x256xf32>, %arg1414: tensor<1x1x256x1024xf32>, %arg1415: tensor<256xf32>, %arg1416: tensor<256xf32>, %arg1417: tensor<256xf32>, %arg1418: tensor<256xf32>, %arg1419: tensor<1024xf32>, %arg1420: tensor<1024xf32>, %arg1421: tensor<1x1x1024x256xf32>, %arg1422: tensor<3x3x256x256xf32>, %arg1423: tensor<1x1x256x1024xf32>, %arg1424: tensor<256xf32>, %arg1425: tensor<256xf32>, %arg1426: tensor<256xf32>, %arg1427: tensor<256xf32>, %arg1428: tensor<1024xf32>, %arg1429: tensor<1024xf32>, %arg1430: tensor<1x1x1024x256xf32>, %arg1431: tensor<3x3x256x256xf32>, %arg1432: tensor<1x1x256x1024xf32>, %arg1433: tensor<128xf32>, %arg1434: tensor<128xf32>, %arg1435: tensor<128xf32>, %arg1436: tensor<128xf32>, %arg1437: tensor<512xf32>, %arg1438: tensor<512xf32>, %arg1439: tensor<1x1x512x128xf32>, %arg1440: tensor<3x3x128x128xf32>, %arg1441: tensor<1x1x128x512xf32>, %arg1442: tensor<256xf32>, %arg1443: tensor<256xf32>, %arg1444: tensor<256xf32>, %arg1445: tensor<256xf32>, %arg1446: tensor<1024xf32>, %arg1447: tensor<1024xf32>, %arg1448: tensor<1x1x1024x256xf32>, %arg1449: tensor<3x3x256x256xf32>, %arg1450: tensor<1x1x256x1024xf32>, %arg1451: tensor<256xf32>, %arg1452: tensor<256xf32>, %arg1453: tensor<256xf32>, %arg1454: tensor<256xf32>, %arg1455: tensor<1024xf32>, %arg1456: tensor<1024xf32>, %arg1457: tensor<1x1x1024x256xf32>, %arg1458: tensor<3x3x256x256xf32>, %arg1459: tensor<1x1x256x1024xf32>, %arg1460: tensor<256xf32>, %arg1461: tensor<256xf32>, %arg1462: tensor<256xf32>, %arg1463: tensor<256xf32>, %arg1464: tensor<1024xf32>, %arg1465: tensor<1024xf32>, %arg1466: tensor<1x1x1024x256xf32>, %arg1467: tensor<3x3x256x256xf32>, %arg1468: tensor<1x1x256x1024xf32>, %arg1469: tensor<256xf32>, %arg1470: tensor<256xf32>, %arg1471: tensor<256xf32>, %arg1472: tensor<256xf32>, %arg1473: tensor<1024xf32>, %arg1474: tensor<1024xf32>, %arg1475: tensor<1x1x1024x256xf32>, %arg1476: tensor<3x3x256x256xf32>, %arg1477: tensor<1x1x256x1024xf32>, %arg1478: tensor<256xf32>, %arg1479: tensor<256xf32>, %arg1480: tensor<256xf32>, %arg1481: tensor<256xf32>, %arg1482: tensor<1024xf32>, %arg1483: tensor<1024xf32>, %arg1484: tensor<1x1x1024x256xf32>, %arg1485: tensor<3x3x256x256xf32>, %arg1486: tensor<1x1x256x1024xf32>, %arg1487: tensor<256xf32>, %arg1488: tensor<256xf32>, %arg1489: tensor<256xf32>, %arg1490: tensor<256xf32>, %arg1491: tensor<1024xf32>, %arg1492: tensor<1024xf32>, %arg1493: tensor<1x1x1024x256xf32>, %arg1494: tensor<3x3x256x256xf32>, %arg1495: tensor<1x1x256x1024xf32>, %arg1496: tensor<256xf32>, %arg1497: tensor<256xf32>, %arg1498: tensor<256xf32>, %arg1499: tensor<256xf32>, %arg1500: tensor<1024xf32>, %arg1501: tensor<1024xf32>, %arg1502: tensor<1x1x1024x256xf32>, %arg1503: tensor<3x3x256x256xf32>, %arg1504: tensor<1x1x256x1024xf32>, %arg1505: tensor<256xf32>, %arg1506: tensor<256xf32>, %arg1507: tensor<256xf32>, %arg1508: tensor<256xf32>, %arg1509: tensor<1024xf32>, %arg1510: tensor<1024xf32>, %arg1511: tensor<1x1x1024x256xf32>, %arg1512: tensor<3x3x256x256xf32>, %arg1513: tensor<1x1x256x1024xf32>, %arg1514: tensor<256xf32>, %arg1515: tensor<256xf32>, %arg1516: tensor<256xf32>, %arg1517: tensor<256xf32>, %arg1518: tensor<1024xf32>, %arg1519: tensor<1024xf32>, %arg1520: tensor<1x1x1024x256xf32>, %arg1521: tensor<3x3x256x256xf32>, %arg1522: tensor<1x1x256x1024xf32>, %arg1523: tensor<256xf32>, %arg1524: tensor<256xf32>, %arg1525: tensor<256xf32>, %arg1526: tensor<256xf32>, %arg1527: tensor<1024xf32>, %arg1528: tensor<1024xf32>, %arg1529: tensor<1x1x1024x256xf32>, %arg1530: tensor<3x3x256x256xf32>, %arg1531: tensor<1x1x256x1024xf32>, %arg1532: tensor<128xf32>, %arg1533: tensor<128xf32>, %arg1534: tensor<128xf32>, %arg1535: tensor<128xf32>, %arg1536: tensor<512xf32>, %arg1537: tensor<512xf32>, %arg1538: tensor<1x1x512x128xf32>, %arg1539: tensor<3x3x128x128xf32>, %arg1540: tensor<1x1x128x512xf32>, %arg1541: tensor<256xf32>, %arg1542: tensor<256xf32>, %arg1543: tensor<256xf32>, %arg1544: tensor<256xf32>, %arg1545: tensor<1024xf32>, %arg1546: tensor<1024xf32>, %arg1547: tensor<1x1x1024x256xf32>, %arg1548: tensor<3x3x256x256xf32>, %arg1549: tensor<1x1x256x1024xf32>, %arg1550: tensor<256xf32>, %arg1551: tensor<256xf32>, %arg1552: tensor<256xf32>, %arg1553: tensor<256xf32>, %arg1554: tensor<1024xf32>, %arg1555: tensor<1024xf32>, %arg1556: tensor<1x1x1024x256xf32>, %arg1557: tensor<3x3x256x256xf32>, %arg1558: tensor<1x1x256x1024xf32>, %arg1559: tensor<256xf32>, %arg1560: tensor<256xf32>, %arg1561: tensor<256xf32>, %arg1562: tensor<256xf32>, %arg1563: tensor<1024xf32>, %arg1564: tensor<1024xf32>, %arg1565: tensor<1x1x1024x256xf32>, %arg1566: tensor<3x3x256x256xf32>, %arg1567: tensor<1x1x256x1024xf32>, %arg1568: tensor<512xf32>, %arg1569: tensor<512xf32>, %arg1570: tensor<512xf32>, %arg1571: tensor<512xf32>, %arg1572: tensor<2048xf32>, %arg1573: tensor<2048xf32>, %arg1574: tensor<1x1x1024x512xf32>, %arg1575: tensor<3x3x512x512xf32>, %arg1576: tensor<1x1x512x2048xf32>, %arg1577: tensor<1x1x1024x2048xf32>, %arg1578: tensor<2048xf32>, %arg1579: tensor<2048xf32>, %arg1580: tensor<512xf32>, %arg1581: tensor<512xf32>, %arg1582: tensor<512xf32>, %arg1583: tensor<512xf32>, %arg1584: tensor<2048xf32>, %arg1585: tensor<2048xf32>, %arg1586: tensor<1x1x2048x512xf32>, %arg1587: tensor<3x3x512x512xf32>, %arg1588: tensor<1x1x512x2048xf32>, %arg1589: tensor<512xf32>, %arg1590: tensor<512xf32>, %arg1591: tensor<512xf32>, %arg1592: tensor<512xf32>, %arg1593: tensor<2048xf32>, %arg1594: tensor<2048xf32>, %arg1595: tensor<1x1x2048x512xf32>, %arg1596: tensor<3x3x512x512xf32>, %arg1597: tensor<1x1x512x2048xf32>, %arg1598: tensor<128xf32>, %arg1599: tensor<128xf32>, %arg1600: tensor<128xf32>, %arg1601: tensor<128xf32>, %arg1602: tensor<512xf32>, %arg1603: tensor<512xf32>, %arg1604: tensor<1x1x512x128xf32>, %arg1605: tensor<3x3x128x128xf32>, %arg1606: tensor<1x1x128x512xf32>, %arg1607: tensor<128xf32>, %arg1608: tensor<128xf32>, %arg1609: tensor<128xf32>, %arg1610: tensor<128xf32>, %arg1611: tensor<512xf32>, %arg1612: tensor<512xf32>, %arg1613: tensor<1x1x512x128xf32>, %arg1614: tensor<3x3x128x128xf32>, %arg1615: tensor<1x1x128x512xf32>, %arg1616: tensor<128xf32>, %arg1617: tensor<128xf32>, %arg1618: tensor<128xf32>, %arg1619: tensor<128xf32>, %arg1620: tensor<512xf32>, %arg1621: tensor<512xf32>, %arg1622: tensor<1x1x512x128xf32>, %arg1623: tensor<3x3x128x128xf32>, %arg1624: tensor<1x1x128x512xf32>, %arg1625: tensor<1000xf32>, %arg1626: tensor<2048x1000xf32>, %arg1627: tensor<64xf32>, %arg1628: tensor<64xf32>, %arg1629: tensor<7x7x3x64xf32>, %arg1630: tensor<64xf32>, %arg1631: tensor<64xf32>, %arg1632: tensor<64xf32>, %arg1633: tensor<64xf32>, %arg1634: tensor<256xf32>, %arg1635: tensor<256xf32>, %arg1636: tensor<256xf32>, %arg1637: tensor<256xf32>, %arg1638: tensor<64xf32>, %arg1639: tensor<64xf32>, %arg1640: tensor<64xf32>, %arg1641: tensor<64xf32>, %arg1642: tensor<256xf32>, %arg1643: tensor<256xf32>, %arg1644: tensor<128xf32>, %arg1645: tensor<128xf32>, %arg1646: tensor<128xf32>, %arg1647: tensor<128xf32>, %arg1648: tensor<512xf32>, %arg1649: tensor<512xf32>, %arg1650: tensor<128xf32>, %arg1651: tensor<128xf32>, %arg1652: tensor<128xf32>, %arg1653: tensor<128xf32>, %arg1654: tensor<512xf32>, %arg1655: tensor<512xf32>, %arg1656: tensor<128xf32>, %arg1657: tensor<128xf32>, %arg1658: tensor<128xf32>, %arg1659: tensor<128xf32>, %arg1660: tensor<512xf32>, %arg1661: tensor<512xf32>, %arg1662: tensor<128xf32>, %arg1663: tensor<128xf32>, %arg1664: tensor<128xf32>, %arg1665: tensor<128xf32>, %arg1666: tensor<512xf32>, %arg1667: tensor<512xf32>, %arg1668: tensor<128xf32>, %arg1669: tensor<128xf32>, %arg1670: tensor<128xf32>, %arg1671: tensor<128xf32>, %arg1672: tensor<512xf32>, %arg1673: tensor<512xf32>, %arg1674: tensor<128xf32>, %arg1675: tensor<128xf32>, %arg1676: tensor<128xf32>, %arg1677: tensor<128xf32>, %arg1678: tensor<512xf32>, %arg1679: tensor<512xf32>, %arg1680: tensor<128xf32>, %arg1681: tensor<128xf32>, %arg1682: tensor<128xf32>, %arg1683: tensor<128xf32>, %arg1684: tensor<512xf32>, %arg1685: tensor<512xf32>, %arg1686: tensor<128xf32>, %arg1687: tensor<128xf32>, %arg1688: tensor<128xf32>, %arg1689: tensor<128xf32>, %arg1690: tensor<512xf32>, %arg1691: tensor<512xf32>, %arg1692: tensor<128xf32>, %arg1693: tensor<128xf32>, %arg1694: tensor<128xf32>, %arg1695: tensor<128xf32>, %arg1696: tensor<512xf32>, %arg1697: tensor<512xf32>, %arg1698: tensor<128xf32>, %arg1699: tensor<128xf32>, %arg1700: tensor<128xf32>, %arg1701: tensor<128xf32>, %arg1702: tensor<512xf32>, %arg1703: tensor<512xf32>, %arg1704: tensor<64xf32>, %arg1705: tensor<64xf32>, %arg1706: tensor<64xf32>, %arg1707: tensor<64xf32>, %arg1708: tensor<256xf32>, %arg1709: tensor<256xf32>, %arg1710: tensor<128xf32>, %arg1711: tensor<128xf32>, %arg1712: tensor<128xf32>, %arg1713: tensor<128xf32>, %arg1714: tensor<512xf32>, %arg1715: tensor<512xf32>, %arg1716: tensor<128xf32>, %arg1717: tensor<128xf32>, %arg1718: tensor<128xf32>, %arg1719: tensor<128xf32>, %arg1720: tensor<512xf32>, %arg1721: tensor<512xf32>, %arg1722: tensor<128xf32>, %arg1723: tensor<128xf32>, %arg1724: tensor<128xf32>, %arg1725: tensor<128xf32>, %arg1726: tensor<512xf32>, %arg1727: tensor<512xf32>, %arg1728: tensor<128xf32>, %arg1729: tensor<128xf32>, %arg1730: tensor<128xf32>, %arg1731: tensor<128xf32>, %arg1732: tensor<512xf32>, %arg1733: tensor<512xf32>, %arg1734: tensor<128xf32>, %arg1735: tensor<128xf32>, %arg1736: tensor<128xf32>, %arg1737: tensor<128xf32>, %arg1738: tensor<512xf32>, %arg1739: tensor<512xf32>, %arg1740: tensor<128xf32>, %arg1741: tensor<128xf32>, %arg1742: tensor<128xf32>, %arg1743: tensor<128xf32>, %arg1744: tensor<512xf32>, %arg1745: tensor<512xf32>, %arg1746: tensor<128xf32>, %arg1747: tensor<128xf32>, %arg1748: tensor<128xf32>, %arg1749: tensor<128xf32>, %arg1750: tensor<512xf32>, %arg1751: tensor<512xf32>, %arg1752: tensor<256xf32>, %arg1753: tensor<256xf32>, %arg1754: tensor<256xf32>, %arg1755: tensor<256xf32>, %arg1756: tensor<1024xf32>, %arg1757: tensor<1024xf32>, %arg1758: tensor<1024xf32>, %arg1759: tensor<1024xf32>, %arg1760: tensor<256xf32>, %arg1761: tensor<256xf32>, %arg1762: tensor<256xf32>, %arg1763: tensor<256xf32>, %arg1764: tensor<1024xf32>, %arg1765: tensor<1024xf32>, %arg1766: tensor<256xf32>, %arg1767: tensor<256xf32>, %arg1768: tensor<256xf32>, %arg1769: tensor<256xf32>, %arg1770: tensor<1024xf32>, %arg1771: tensor<1024xf32>, %arg1772: tensor<128xf32>, %arg1773: tensor<128xf32>, %arg1774: tensor<128xf32>, %arg1775: tensor<128xf32>, %arg1776: tensor<512xf32>, %arg1777: tensor<512xf32>, %arg1778: tensor<512xf32>, %arg1779: tensor<512xf32>, %arg1780: tensor<256xf32>, %arg1781: tensor<256xf32>, %arg1782: tensor<256xf32>, %arg1783: tensor<256xf32>, %arg1784: tensor<1024xf32>, %arg1785: tensor<1024xf32>, %arg1786: tensor<256xf32>, %arg1787: tensor<256xf32>, %arg1788: tensor<256xf32>, %arg1789: tensor<256xf32>, %arg1790: tensor<1024xf32>, %arg1791: tensor<1024xf32>, %arg1792: tensor<256xf32>, %arg1793: tensor<256xf32>, %arg1794: tensor<256xf32>, %arg1795: tensor<256xf32>, %arg1796: tensor<1024xf32>, %arg1797: tensor<1024xf32>, %arg1798: tensor<256xf32>, %arg1799: tensor<256xf32>, %arg1800: tensor<256xf32>, %arg1801: tensor<256xf32>, %arg1802: tensor<1024xf32>, %arg1803: tensor<1024xf32>, %arg1804: tensor<256xf32>, %arg1805: tensor<256xf32>, %arg1806: tensor<256xf32>, %arg1807: tensor<256xf32>, %arg1808: tensor<1024xf32>, %arg1809: tensor<1024xf32>, %arg1810: tensor<256xf32>, %arg1811: tensor<256xf32>, %arg1812: tensor<256xf32>, %arg1813: tensor<256xf32>, %arg1814: tensor<1024xf32>, %arg1815: tensor<1024xf32>, %arg1816: tensor<256xf32>, %arg1817: tensor<256xf32>, %arg1818: tensor<256xf32>, %arg1819: tensor<256xf32>, %arg1820: tensor<1024xf32>, %arg1821: tensor<1024xf32>, %arg1822: tensor<256xf32>, %arg1823: tensor<256xf32>, %arg1824: tensor<256xf32>, %arg1825: tensor<256xf32>, %arg1826: tensor<1024xf32>, %arg1827: tensor<1024xf32>, %arg1828: tensor<256xf32>, %arg1829: tensor<256xf32>, %arg1830: tensor<256xf32>, %arg1831: tensor<256xf32>, %arg1832: tensor<1024xf32>, %arg1833: tensor<1024xf32>, %arg1834: tensor<256xf32>, %arg1835: tensor<256xf32>, %arg1836: tensor<256xf32>, %arg1837: tensor<256xf32>, %arg1838: tensor<1024xf32>, %arg1839: tensor<1024xf32>, %arg1840: tensor<128xf32>, %arg1841: tensor<128xf32>, %arg1842: tensor<128xf32>, %arg1843: tensor<128xf32>, %arg1844: tensor<512xf32>, %arg1845: tensor<512xf32>, %arg1846: tensor<256xf32>, %arg1847: tensor<256xf32>, %arg1848: tensor<256xf32>, %arg1849: tensor<256xf32>, %arg1850: tensor<1024xf32>, %arg1851: tensor<1024xf32>, %arg1852: tensor<256xf32>, %arg1853: tensor<256xf32>, %arg1854: tensor<256xf32>, %arg1855: tensor<256xf32>, %arg1856: tensor<1024xf32>, %arg1857: tensor<1024xf32>, %arg1858: tensor<256xf32>, %arg1859: tensor<256xf32>, %arg1860: tensor<256xf32>, %arg1861: tensor<256xf32>, %arg1862: tensor<1024xf32>, %arg1863: tensor<1024xf32>, %arg1864: tensor<256xf32>, %arg1865: tensor<256xf32>, %arg1866: tensor<256xf32>, %arg1867: tensor<256xf32>, %arg1868: tensor<1024xf32>, %arg1869: tensor<1024xf32>, %arg1870: tensor<256xf32>, %arg1871: tensor<256xf32>, %arg1872: tensor<256xf32>, %arg1873: tensor<256xf32>, %arg1874: tensor<1024xf32>, %arg1875: tensor<1024xf32>, %arg1876: tensor<256xf32>, %arg1877: tensor<256xf32>, %arg1878: tensor<256xf32>, %arg1879: tensor<256xf32>, %arg1880: tensor<1024xf32>, %arg1881: tensor<1024xf32>, %arg1882: tensor<256xf32>, %arg1883: tensor<256xf32>, %arg1884: tensor<256xf32>, %arg1885: tensor<256xf32>, %arg1886: tensor<1024xf32>, %arg1887: tensor<1024xf32>, %arg1888: tensor<256xf32>, %arg1889: tensor<256xf32>, %arg1890: tensor<256xf32>, %arg1891: tensor<256xf32>, %arg1892: tensor<1024xf32>, %arg1893: tensor<1024xf32>, %arg1894: tensor<256xf32>, %arg1895: tensor<256xf32>, %arg1896: tensor<256xf32>, %arg1897: tensor<256xf32>, %arg1898: tensor<1024xf32>, %arg1899: tensor<1024xf32>, %arg1900: tensor<256xf32>, %arg1901: tensor<256xf32>, %arg1902: tensor<256xf32>, %arg1903: tensor<256xf32>, %arg1904: tensor<1024xf32>, %arg1905: tensor<1024xf32>, %arg1906: tensor<128xf32>, %arg1907: tensor<128xf32>, %arg1908: tensor<128xf32>, %arg1909: tensor<128xf32>, %arg1910: tensor<512xf32>, %arg1911: tensor<512xf32>, %arg1912: tensor<256xf32>, %arg1913: tensor<256xf32>, %arg1914: tensor<256xf32>, %arg1915: tensor<256xf32>, %arg1916: tensor<1024xf32>, %arg1917: tensor<1024xf32>, %arg1918: tensor<256xf32>, %arg1919: tensor<256xf32>, %arg1920: tensor<256xf32>, %arg1921: tensor<256xf32>, %arg1922: tensor<1024xf32>, %arg1923: tensor<1024xf32>, %arg1924: tensor<256xf32>, %arg1925: tensor<256xf32>, %arg1926: tensor<256xf32>, %arg1927: tensor<256xf32>, %arg1928: tensor<1024xf32>, %arg1929: tensor<1024xf32>, %arg1930: tensor<256xf32>, %arg1931: tensor<256xf32>, %arg1932: tensor<256xf32>, %arg1933: tensor<256xf32>, %arg1934: tensor<1024xf32>, %arg1935: tensor<1024xf32>, %arg1936: tensor<256xf32>, %arg1937: tensor<256xf32>, %arg1938: tensor<256xf32>, %arg1939: tensor<256xf32>, %arg1940: tensor<1024xf32>, %arg1941: tensor<1024xf32>, %arg1942: tensor<256xf32>, %arg1943: tensor<256xf32>, %arg1944: tensor<256xf32>, %arg1945: tensor<256xf32>, %arg1946: tensor<1024xf32>, %arg1947: tensor<1024xf32>, %arg1948: tensor<256xf32>, %arg1949: tensor<256xf32>, %arg1950: tensor<256xf32>, %arg1951: tensor<256xf32>, %arg1952: tensor<1024xf32>, %arg1953: tensor<1024xf32>, %arg1954: tensor<256xf32>, %arg1955: tensor<256xf32>, %arg1956: tensor<256xf32>, %arg1957: tensor<256xf32>, %arg1958: tensor<1024xf32>, %arg1959: tensor<1024xf32>, %arg1960: tensor<256xf32>, %arg1961: tensor<256xf32>, %arg1962: tensor<256xf32>, %arg1963: tensor<256xf32>, %arg1964: tensor<1024xf32>, %arg1965: tensor<1024xf32>, %arg1966: tensor<256xf32>, %arg1967: tensor<256xf32>, %arg1968: tensor<256xf32>, %arg1969: tensor<256xf32>, %arg1970: tensor<1024xf32>, %arg1971: tensor<1024xf32>, %arg1972: tensor<128xf32>, %arg1973: tensor<128xf32>, %arg1974: tensor<128xf32>, %arg1975: tensor<128xf32>, %arg1976: tensor<512xf32>, %arg1977: tensor<512xf32>, %arg1978: tensor<256xf32>, %arg1979: tensor<256xf32>, %arg1980: tensor<256xf32>, %arg1981: tensor<256xf32>, %arg1982: tensor<1024xf32>, %arg1983: tensor<1024xf32>, %arg1984: tensor<256xf32>, %arg1985: tensor<256xf32>, %arg1986: tensor<256xf32>, %arg1987: tensor<256xf32>, %arg1988: tensor<1024xf32>, %arg1989: tensor<1024xf32>, %arg1990: tensor<256xf32>, %arg1991: tensor<256xf32>, %arg1992: tensor<256xf32>, %arg1993: tensor<256xf32>, %arg1994: tensor<1024xf32>, %arg1995: tensor<1024xf32>, %arg1996: tensor<512xf32>, %arg1997: tensor<512xf32>, %arg1998: tensor<512xf32>, %arg1999: tensor<512xf32>, %arg2000: tensor<2048xf32>, %arg2001: tensor<2048xf32>, %arg2002: tensor<2048xf32>, %arg2003: tensor<2048xf32>, %arg2004: tensor<512xf32>, %arg2005: tensor<512xf32>, %arg2006: tensor<512xf32>, %arg2007: tensor<512xf32>, %arg2008: tensor<2048xf32>, %arg2009: tensor<2048xf32>, %arg2010: tensor<512xf32>, %arg2011: tensor<512xf32>, %arg2012: tensor<512xf32>, %arg2013: tensor<512xf32>, %arg2014: tensor<2048xf32>, %arg2015: tensor<2048xf32>, %arg2016: tensor<128xf32>, %arg2017: tensor<128xf32>, %arg2018: tensor<128xf32>, %arg2019: tensor<128xf32>, %arg2020: tensor<512xf32>, %arg2021: tensor<512xf32>, %arg2022: tensor<128xf32>, %arg2023: tensor<128xf32>, %arg2024: tensor<128xf32>, %arg2025: tensor<128xf32>, %arg2026: tensor<512xf32>, %arg2027: tensor<512xf32>, %arg2028: tensor<128xf32>, %arg2029: tensor<128xf32>, %arg2030: tensor<128xf32>, %arg2031: tensor<128xf32>, %arg2032: tensor<512xf32>, %arg2033: tensor<512xf32>, %arg2034: tensor<64xf32>, %arg2035: tensor<64xf32>, %arg2036: tensor<1x256x224x224x3xf16>, %arg2037: tensor<1x256xi32>) {
      %1 = stablehlo.reshape %arg2036 : (tensor<1x256x224x224x3xf16>) -> tensor<256x224x224x3xf16>
      %2 = stablehlo.reshape %arg2037 : (tensor<1x256xi32>) -> tensor<256xi32>
      %3 = stablehlo.convert %arg1629 : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf16>
      %4 = stablehlo.convolution(%1, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x224x224x3xf16>, tensor<7x7x3x64xf16>) -> tensor<256x112x112x64xf16>
      %5 = stablehlo.broadcast_in_dim %arg2034, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %6 = stablehlo.broadcast_in_dim %arg2035, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %7 = stablehlo.convert %4 : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf32>
      %8 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %9 = stablehlo.subtract %7, %8 : tensor<256x112x112x64xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %11 = stablehlo.add %6, %10 : tensor<1x1x1x64xf32>
      %12 = stablehlo.rsqrt %11 : tensor<1x1x1x64xf32>
      %13 = stablehlo.reshape %arg1628 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %14 = stablehlo.multiply %12, %13 : tensor<1x1x1x64xf32>
      %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %16 = stablehlo.multiply %9, %15 : tensor<256x112x112x64xf32>
      %17 = stablehlo.reshape %arg1627 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x112x112x64xf32>
      %19 = stablehlo.add %16, %18 : tensor<256x112x112x64xf32>
      %20 = stablehlo.convert %19 : (tensor<256x112x112x64xf32>) -> tensor<256x112x112x64xf16>
      %21 = func.call @relu(%20) : (tensor<256x112x112x64xf16>) -> tensor<256x112x112x64xf16>
      %cst_0 = stablehlo.constant dense<0xFC00> : tensor<f16>
      %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f16>) -> tensor<f16>
      %23 = "stablehlo.reduce_window"(%21, %22) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
      ^bb0(%arg2038: tensor<f16>, %arg2039: tensor<f16>):
        %3964 = stablehlo.maximum %arg2038, %arg2039 : tensor<f16>
        stablehlo.return %3964 : tensor<f16>
      }) : (tensor<256x112x112x64xf16>, tensor<f16>) -> tensor<256x56x56x64xf16>
      %24 = stablehlo.convert %arg1025 : (tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xf16>
      %25 = stablehlo.convolution(%23, %24) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x64xf16>) -> tensor<256x56x56x64xf16>
      %26 = stablehlo.broadcast_in_dim %arg1630, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %27 = stablehlo.broadcast_in_dim %arg1631, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %28 = stablehlo.convert %25 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %29 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %30 = stablehlo.subtract %28, %29 : tensor<256x56x56x64xf32>
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %31 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %32 = stablehlo.add %27, %31 : tensor<1x1x1x64xf32>
      %33 = stablehlo.rsqrt %32 : tensor<1x1x1x64xf32>
      %34 = stablehlo.reshape %arg1020 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %35 = stablehlo.multiply %33, %34 : tensor<1x1x1x64xf32>
      %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %37 = stablehlo.multiply %30, %36 : tensor<256x56x56x64xf32>
      %38 = stablehlo.reshape %arg1019 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %40 = stablehlo.add %37, %39 : tensor<256x56x56x64xf32>
      %41 = stablehlo.convert %40 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %42 = func.call @relu_13(%41) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %43 = stablehlo.convert %arg1026 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %44 = stablehlo.convolution(%42, %43) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %45 = stablehlo.broadcast_in_dim %arg1632, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %46 = stablehlo.broadcast_in_dim %arg1633, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %47 = stablehlo.convert %44 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %48 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %49 = stablehlo.subtract %47, %48 : tensor<256x56x56x64xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %50 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %51 = stablehlo.add %46, %50 : tensor<1x1x1x64xf32>
      %52 = stablehlo.rsqrt %51 : tensor<1x1x1x64xf32>
      %53 = stablehlo.reshape %arg1022 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %54 = stablehlo.multiply %52, %53 : tensor<1x1x1x64xf32>
      %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %56 = stablehlo.multiply %49, %55 : tensor<256x56x56x64xf32>
      %57 = stablehlo.reshape %arg1021 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %59 = stablehlo.add %56, %58 : tensor<256x56x56x64xf32>
      %60 = stablehlo.convert %59 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %61 = func.call @relu_13(%60) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %62 = stablehlo.convert %arg1027 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %63 = stablehlo.convolution(%61, %62) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %64 = stablehlo.broadcast_in_dim %arg1634, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %65 = stablehlo.broadcast_in_dim %arg1635, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %66 = stablehlo.convert %63 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %67 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %68 = stablehlo.subtract %66, %67 : tensor<256x56x56x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %69 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %70 = stablehlo.add %65, %69 : tensor<1x1x1x256xf32>
      %71 = stablehlo.rsqrt %70 : tensor<1x1x1x256xf32>
      %72 = stablehlo.reshape %arg1024 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %73 = stablehlo.multiply %71, %72 : tensor<1x1x1x256xf32>
      %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %75 = stablehlo.multiply %68, %74 : tensor<256x56x56x256xf32>
      %76 = stablehlo.reshape %arg1023 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %78 = stablehlo.add %75, %77 : tensor<256x56x56x256xf32>
      %79 = stablehlo.convert %78 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %80 = stablehlo.convert %arg1028 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %81 = stablehlo.convolution(%23, %80) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %82 = stablehlo.broadcast_in_dim %arg1636, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %83 = stablehlo.broadcast_in_dim %arg1637, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %84 = stablehlo.convert %81 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %85 = stablehlo.broadcast_in_dim %82, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %86 = stablehlo.subtract %84, %85 : tensor<256x56x56x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %87 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %88 = stablehlo.add %83, %87 : tensor<1x1x1x256xf32>
      %89 = stablehlo.rsqrt %88 : tensor<1x1x1x256xf32>
      %90 = stablehlo.reshape %arg1030 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %91 = stablehlo.multiply %89, %90 : tensor<1x1x1x256xf32>
      %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %93 = stablehlo.multiply %86, %92 : tensor<256x56x56x256xf32>
      %94 = stablehlo.reshape %arg1029 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %96 = stablehlo.add %93, %95 : tensor<256x56x56x256xf32>
      %97 = stablehlo.convert %96 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %98 = stablehlo.add %97, %79 : tensor<256x56x56x256xf16>
      %99 = func.call @relu_32(%98) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
      %100 = stablehlo.convert %arg1037 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
      %101 = stablehlo.convolution(%99, %100) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<256x56x56x64xf16>
      %102 = stablehlo.broadcast_in_dim %arg1638, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %103 = stablehlo.broadcast_in_dim %arg1639, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %104 = stablehlo.convert %101 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %105 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %106 = stablehlo.subtract %104, %105 : tensor<256x56x56x64xf32>
      %cst_5 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %107 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %108 = stablehlo.add %103, %107 : tensor<1x1x1x64xf32>
      %109 = stablehlo.rsqrt %108 : tensor<1x1x1x64xf32>
      %110 = stablehlo.reshape %arg1032 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %111 = stablehlo.multiply %109, %110 : tensor<1x1x1x64xf32>
      %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %113 = stablehlo.multiply %106, %112 : tensor<256x56x56x64xf32>
      %114 = stablehlo.reshape %arg1031 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %115 = stablehlo.broadcast_in_dim %114, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %116 = stablehlo.add %113, %115 : tensor<256x56x56x64xf32>
      %117 = stablehlo.convert %116 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %118 = func.call @relu_13(%117) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %119 = stablehlo.convert %arg1038 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %120 = stablehlo.convolution(%118, %119) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %121 = stablehlo.broadcast_in_dim %arg1640, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %122 = stablehlo.broadcast_in_dim %arg1641, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %123 = stablehlo.convert %120 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %124 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %125 = stablehlo.subtract %123, %124 : tensor<256x56x56x64xf32>
      %cst_6 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %126 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %127 = stablehlo.add %122, %126 : tensor<1x1x1x64xf32>
      %128 = stablehlo.rsqrt %127 : tensor<1x1x1x64xf32>
      %129 = stablehlo.reshape %arg1034 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %130 = stablehlo.multiply %128, %129 : tensor<1x1x1x64xf32>
      %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %132 = stablehlo.multiply %125, %131 : tensor<256x56x56x64xf32>
      %133 = stablehlo.reshape %arg1033 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %134 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %135 = stablehlo.add %132, %134 : tensor<256x56x56x64xf32>
      %136 = stablehlo.convert %135 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %137 = func.call @relu_13(%136) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %138 = stablehlo.convert %arg1039 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %139 = stablehlo.convolution(%137, %138) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %140 = stablehlo.broadcast_in_dim %arg1642, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %141 = stablehlo.broadcast_in_dim %arg1643, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %142 = stablehlo.convert %139 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %143 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %144 = stablehlo.subtract %142, %143 : tensor<256x56x56x256xf32>
      %cst_7 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %145 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %146 = stablehlo.add %141, %145 : tensor<1x1x1x256xf32>
      %147 = stablehlo.rsqrt %146 : tensor<1x1x1x256xf32>
      %148 = stablehlo.reshape %arg1036 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %149 = stablehlo.multiply %147, %148 : tensor<1x1x1x256xf32>
      %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %151 = stablehlo.multiply %144, %150 : tensor<256x56x56x256xf32>
      %152 = stablehlo.reshape %arg1035 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %154 = stablehlo.add %151, %153 : tensor<256x56x56x256xf32>
      %155 = stablehlo.convert %154 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %156 = stablehlo.add %99, %155 : tensor<256x56x56x256xf16>
      %157 = func.call @relu_32(%156) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
      %158 = stablehlo.convert %arg1136 : (tensor<1x1x256x64xf32>) -> tensor<1x1x256x64xf16>
      %159 = stablehlo.convolution(%157, %158) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x64xf16>) -> tensor<256x56x56x64xf16>
      %160 = stablehlo.broadcast_in_dim %arg1704, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %161 = stablehlo.broadcast_in_dim %arg1705, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %162 = stablehlo.convert %159 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %163 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %164 = stablehlo.subtract %162, %163 : tensor<256x56x56x64xf32>
      %cst_8 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %165 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %166 = stablehlo.add %161, %165 : tensor<1x1x1x64xf32>
      %167 = stablehlo.rsqrt %166 : tensor<1x1x1x64xf32>
      %168 = stablehlo.reshape %arg1131 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %169 = stablehlo.multiply %167, %168 : tensor<1x1x1x64xf32>
      %170 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %171 = stablehlo.multiply %164, %170 : tensor<256x56x56x64xf32>
      %172 = stablehlo.reshape %arg1130 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %173 = stablehlo.broadcast_in_dim %172, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %174 = stablehlo.add %171, %173 : tensor<256x56x56x64xf32>
      %175 = stablehlo.convert %174 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %176 = func.call @relu_13(%175) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %177 = stablehlo.convert %arg1137 : (tensor<3x3x64x64xf32>) -> tensor<3x3x64x64xf16>
      %178 = stablehlo.convolution(%176, %177) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<3x3x64x64xf16>) -> tensor<256x56x56x64xf16>
      %179 = stablehlo.broadcast_in_dim %arg1706, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %180 = stablehlo.broadcast_in_dim %arg1707, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %181 = stablehlo.convert %178 : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf32>
      %182 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %183 = stablehlo.subtract %181, %182 : tensor<256x56x56x64xf32>
      %cst_9 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %184 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
      %185 = stablehlo.add %180, %184 : tensor<1x1x1x64xf32>
      %186 = stablehlo.rsqrt %185 : tensor<1x1x1x64xf32>
      %187 = stablehlo.reshape %arg1133 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %188 = stablehlo.multiply %186, %187 : tensor<1x1x1x64xf32>
      %189 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %190 = stablehlo.multiply %183, %189 : tensor<256x56x56x64xf32>
      %191 = stablehlo.reshape %arg1132 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
      %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<256x56x56x64xf32>
      %193 = stablehlo.add %190, %192 : tensor<256x56x56x64xf32>
      %194 = stablehlo.convert %193 : (tensor<256x56x56x64xf32>) -> tensor<256x56x56x64xf16>
      %195 = func.call @relu_13(%194) : (tensor<256x56x56x64xf16>) -> tensor<256x56x56x64xf16>
      %196 = stablehlo.convert %arg1138 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xf16>
      %197 = stablehlo.convolution(%195, %196) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xf16>, tensor<1x1x64x256xf16>) -> tensor<256x56x56x256xf16>
      %198 = stablehlo.broadcast_in_dim %arg1708, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %199 = stablehlo.broadcast_in_dim %arg1709, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %200 = stablehlo.convert %197 : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf32>
      %201 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %202 = stablehlo.subtract %200, %201 : tensor<256x56x56x256xf32>
      %cst_10 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %203 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %204 = stablehlo.add %199, %203 : tensor<1x1x1x256xf32>
      %205 = stablehlo.rsqrt %204 : tensor<1x1x1x256xf32>
      %206 = stablehlo.reshape %arg1135 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %207 = stablehlo.multiply %205, %206 : tensor<1x1x1x256xf32>
      %208 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %209 = stablehlo.multiply %202, %208 : tensor<256x56x56x256xf32>
      %210 = stablehlo.reshape %arg1134 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %211 = stablehlo.broadcast_in_dim %210, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
      %212 = stablehlo.add %209, %211 : tensor<256x56x56x256xf32>
      %213 = stablehlo.convert %212 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xf16>
      %214 = stablehlo.add %157, %213 : tensor<256x56x56x256xf16>
      %215 = func.call @relu_32(%214) : (tensor<256x56x56x256xf16>) -> tensor<256x56x56x256xf16>
      %216 = stablehlo.convert %arg1238 : (tensor<1x1x256x128xf32>) -> tensor<1x1x256x128xf16>
      %217 = stablehlo.convolution(%215, %216) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x128xf16>) -> tensor<256x56x56x128xf16>
      %218 = stablehlo.broadcast_in_dim %arg1772, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %219 = stablehlo.broadcast_in_dim %arg1773, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %220 = stablehlo.convert %217 : (tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf32>
      %221 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
      %222 = stablehlo.subtract %220, %221 : tensor<256x56x56x128xf32>
      %cst_11 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %223 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %224 = stablehlo.add %219, %223 : tensor<1x1x1x128xf32>
      %225 = stablehlo.rsqrt %224 : tensor<1x1x1x128xf32>
      %226 = stablehlo.reshape %arg1233 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %227 = stablehlo.multiply %225, %226 : tensor<1x1x1x128xf32>
      %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
      %229 = stablehlo.multiply %222, %228 : tensor<256x56x56x128xf32>
      %230 = stablehlo.reshape %arg1232 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %231 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x56x56x128xf32>
      %232 = stablehlo.add %229, %231 : tensor<256x56x56x128xf32>
      %233 = stablehlo.convert %232 : (tensor<256x56x56x128xf32>) -> tensor<256x56x56x128xf16>
      %234 = func.call @relu_55(%233) : (tensor<256x56x56x128xf16>) -> tensor<256x56x56x128xf16>
      %235 = stablehlo.convert %arg1239 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %236 = stablehlo.convolution(%234, %235) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %237 = stablehlo.broadcast_in_dim %arg1774, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %238 = stablehlo.broadcast_in_dim %arg1775, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %239 = stablehlo.convert %236 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %240 = stablehlo.broadcast_in_dim %237, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %241 = stablehlo.subtract %239, %240 : tensor<256x28x28x128xf32>
      %cst_12 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %242 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %243 = stablehlo.add %238, %242 : tensor<1x1x1x128xf32>
      %244 = stablehlo.rsqrt %243 : tensor<1x1x1x128xf32>
      %245 = stablehlo.reshape %arg1235 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %246 = stablehlo.multiply %244, %245 : tensor<1x1x1x128xf32>
      %247 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %248 = stablehlo.multiply %241, %247 : tensor<256x28x28x128xf32>
      %249 = stablehlo.reshape %arg1234 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %251 = stablehlo.add %248, %250 : tensor<256x28x28x128xf32>
      %252 = stablehlo.convert %251 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %253 = func.call @relu_65(%252) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %254 = stablehlo.convert %arg1240 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %255 = stablehlo.convolution(%253, %254) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %256 = stablehlo.broadcast_in_dim %arg1776, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %257 = stablehlo.broadcast_in_dim %arg1777, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %258 = stablehlo.convert %255 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %259 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %260 = stablehlo.subtract %258, %259 : tensor<256x28x28x512xf32>
      %cst_13 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %261 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %262 = stablehlo.add %257, %261 : tensor<1x1x1x512xf32>
      %263 = stablehlo.rsqrt %262 : tensor<1x1x1x512xf32>
      %264 = stablehlo.reshape %arg1237 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %265 = stablehlo.multiply %263, %264 : tensor<1x1x1x512xf32>
      %266 = stablehlo.broadcast_in_dim %265, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %267 = stablehlo.multiply %260, %266 : tensor<256x28x28x512xf32>
      %268 = stablehlo.reshape %arg1236 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %270 = stablehlo.add %267, %269 : tensor<256x28x28x512xf32>
      %271 = stablehlo.convert %270 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %272 = stablehlo.convert %arg1241 : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf16>
      %273 = stablehlo.convolution(%215, %272) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x256xf16>, tensor<1x1x256x512xf16>) -> tensor<256x28x28x512xf16>
      %274 = stablehlo.broadcast_in_dim %arg1778, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %275 = stablehlo.broadcast_in_dim %arg1779, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %276 = stablehlo.convert %273 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %277 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %278 = stablehlo.subtract %276, %277 : tensor<256x28x28x512xf32>
      %cst_14 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %279 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %280 = stablehlo.add %275, %279 : tensor<1x1x1x512xf32>
      %281 = stablehlo.rsqrt %280 : tensor<1x1x1x512xf32>
      %282 = stablehlo.reshape %arg1243 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %283 = stablehlo.multiply %281, %282 : tensor<1x1x1x512xf32>
      %284 = stablehlo.broadcast_in_dim %283, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %285 = stablehlo.multiply %278, %284 : tensor<256x28x28x512xf32>
      %286 = stablehlo.reshape %arg1242 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %287 = stablehlo.broadcast_in_dim %286, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %288 = stablehlo.add %285, %287 : tensor<256x28x28x512xf32>
      %289 = stablehlo.convert %288 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %290 = stablehlo.add %289, %271 : tensor<256x28x28x512xf16>
      %291 = func.call @relu_83(%290) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %292 = stablehlo.convert %arg1340 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %293 = stablehlo.convolution(%291, %292) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %294 = stablehlo.broadcast_in_dim %arg1840, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %295 = stablehlo.broadcast_in_dim %arg1841, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %296 = stablehlo.convert %293 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %297 = stablehlo.broadcast_in_dim %294, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %298 = stablehlo.subtract %296, %297 : tensor<256x28x28x128xf32>
      %cst_15 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %300 = stablehlo.add %295, %299 : tensor<1x1x1x128xf32>
      %301 = stablehlo.rsqrt %300 : tensor<1x1x1x128xf32>
      %302 = stablehlo.reshape %arg1335 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %303 = stablehlo.multiply %301, %302 : tensor<1x1x1x128xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %305 = stablehlo.multiply %298, %304 : tensor<256x28x28x128xf32>
      %306 = stablehlo.reshape %arg1334 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %308 = stablehlo.add %305, %307 : tensor<256x28x28x128xf32>
      %309 = stablehlo.convert %308 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %310 = func.call @relu_65(%309) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %311 = stablehlo.convert %arg1341 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %312 = stablehlo.convolution(%310, %311) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %313 = stablehlo.broadcast_in_dim %arg1842, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %314 = stablehlo.broadcast_in_dim %arg1843, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %315 = stablehlo.convert %312 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %316 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %317 = stablehlo.subtract %315, %316 : tensor<256x28x28x128xf32>
      %cst_16 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %319 = stablehlo.add %314, %318 : tensor<1x1x1x128xf32>
      %320 = stablehlo.rsqrt %319 : tensor<1x1x1x128xf32>
      %321 = stablehlo.reshape %arg1337 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %322 = stablehlo.multiply %320, %321 : tensor<1x1x1x128xf32>
      %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %324 = stablehlo.multiply %317, %323 : tensor<256x28x28x128xf32>
      %325 = stablehlo.reshape %arg1336 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %327 = stablehlo.add %324, %326 : tensor<256x28x28x128xf32>
      %328 = stablehlo.convert %327 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %329 = func.call @relu_65(%328) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %330 = stablehlo.convert %arg1342 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %331 = stablehlo.convolution(%329, %330) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %332 = stablehlo.broadcast_in_dim %arg1844, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %333 = stablehlo.broadcast_in_dim %arg1845, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %334 = stablehlo.convert %331 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %335 = stablehlo.broadcast_in_dim %332, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %336 = stablehlo.subtract %334, %335 : tensor<256x28x28x512xf32>
      %cst_17 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %338 = stablehlo.add %333, %337 : tensor<1x1x1x512xf32>
      %339 = stablehlo.rsqrt %338 : tensor<1x1x1x512xf32>
      %340 = stablehlo.reshape %arg1339 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<1x1x1x512xf32>
      %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %343 = stablehlo.multiply %336, %342 : tensor<256x28x28x512xf32>
      %344 = stablehlo.reshape %arg1338 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %346 = stablehlo.add %343, %345 : tensor<256x28x28x512xf32>
      %347 = stablehlo.convert %346 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %348 = stablehlo.add %291, %347 : tensor<256x28x28x512xf16>
      %349 = func.call @relu_83(%348) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %350 = stablehlo.convert %arg1439 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %351 = stablehlo.convolution(%349, %350) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %352 = stablehlo.broadcast_in_dim %arg1906, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %353 = stablehlo.broadcast_in_dim %arg1907, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %354 = stablehlo.convert %351 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %355 = stablehlo.broadcast_in_dim %352, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %356 = stablehlo.subtract %354, %355 : tensor<256x28x28x128xf32>
      %cst_18 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %358 = stablehlo.add %353, %357 : tensor<1x1x1x128xf32>
      %359 = stablehlo.rsqrt %358 : tensor<1x1x1x128xf32>
      %360 = stablehlo.reshape %arg1434 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %361 = stablehlo.multiply %359, %360 : tensor<1x1x1x128xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %363 = stablehlo.multiply %356, %362 : tensor<256x28x28x128xf32>
      %364 = stablehlo.reshape %arg1433 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %366 = stablehlo.add %363, %365 : tensor<256x28x28x128xf32>
      %367 = stablehlo.convert %366 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %368 = func.call @relu_65(%367) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %369 = stablehlo.convert %arg1440 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %370 = stablehlo.convolution(%368, %369) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %371 = stablehlo.broadcast_in_dim %arg1908, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %372 = stablehlo.broadcast_in_dim %arg1909, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %373 = stablehlo.convert %370 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %374 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %375 = stablehlo.subtract %373, %374 : tensor<256x28x28x128xf32>
      %cst_19 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %377 = stablehlo.add %372, %376 : tensor<1x1x1x128xf32>
      %378 = stablehlo.rsqrt %377 : tensor<1x1x1x128xf32>
      %379 = stablehlo.reshape %arg1436 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %380 = stablehlo.multiply %378, %379 : tensor<1x1x1x128xf32>
      %381 = stablehlo.broadcast_in_dim %380, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %382 = stablehlo.multiply %375, %381 : tensor<256x28x28x128xf32>
      %383 = stablehlo.reshape %arg1435 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %384 = stablehlo.broadcast_in_dim %383, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %385 = stablehlo.add %382, %384 : tensor<256x28x28x128xf32>
      %386 = stablehlo.convert %385 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %387 = func.call @relu_65(%386) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %388 = stablehlo.convert %arg1441 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %389 = stablehlo.convolution(%387, %388) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %390 = stablehlo.broadcast_in_dim %arg1910, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %391 = stablehlo.broadcast_in_dim %arg1911, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %392 = stablehlo.convert %389 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %393 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %394 = stablehlo.subtract %392, %393 : tensor<256x28x28x512xf32>
      %cst_20 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %395 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %396 = stablehlo.add %391, %395 : tensor<1x1x1x512xf32>
      %397 = stablehlo.rsqrt %396 : tensor<1x1x1x512xf32>
      %398 = stablehlo.reshape %arg1438 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %399 = stablehlo.multiply %397, %398 : tensor<1x1x1x512xf32>
      %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %401 = stablehlo.multiply %394, %400 : tensor<256x28x28x512xf32>
      %402 = stablehlo.reshape %arg1437 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %403 = stablehlo.broadcast_in_dim %402, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %404 = stablehlo.add %401, %403 : tensor<256x28x28x512xf32>
      %405 = stablehlo.convert %404 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %406 = stablehlo.add %349, %405 : tensor<256x28x28x512xf16>
      %407 = func.call @relu_83(%406) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %408 = stablehlo.convert %arg1538 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %409 = stablehlo.convolution(%407, %408) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %410 = stablehlo.broadcast_in_dim %arg1972, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %411 = stablehlo.broadcast_in_dim %arg1973, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %412 = stablehlo.convert %409 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %413 = stablehlo.broadcast_in_dim %410, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %414 = stablehlo.subtract %412, %413 : tensor<256x28x28x128xf32>
      %cst_21 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %415 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %416 = stablehlo.add %411, %415 : tensor<1x1x1x128xf32>
      %417 = stablehlo.rsqrt %416 : tensor<1x1x1x128xf32>
      %418 = stablehlo.reshape %arg1533 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %419 = stablehlo.multiply %417, %418 : tensor<1x1x1x128xf32>
      %420 = stablehlo.broadcast_in_dim %419, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %421 = stablehlo.multiply %414, %420 : tensor<256x28x28x128xf32>
      %422 = stablehlo.reshape %arg1532 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %424 = stablehlo.add %421, %423 : tensor<256x28x28x128xf32>
      %425 = stablehlo.convert %424 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %426 = func.call @relu_65(%425) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %427 = stablehlo.convert %arg1539 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %428 = stablehlo.convolution(%426, %427) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %429 = stablehlo.broadcast_in_dim %arg1974, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %430 = stablehlo.broadcast_in_dim %arg1975, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %431 = stablehlo.convert %428 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %432 = stablehlo.broadcast_in_dim %429, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %433 = stablehlo.subtract %431, %432 : tensor<256x28x28x128xf32>
      %cst_22 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %435 = stablehlo.add %430, %434 : tensor<1x1x1x128xf32>
      %436 = stablehlo.rsqrt %435 : tensor<1x1x1x128xf32>
      %437 = stablehlo.reshape %arg1535 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<1x1x1x128xf32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %440 = stablehlo.multiply %433, %439 : tensor<256x28x28x128xf32>
      %441 = stablehlo.reshape %arg1534 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %443 = stablehlo.add %440, %442 : tensor<256x28x28x128xf32>
      %444 = stablehlo.convert %443 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %445 = func.call @relu_65(%444) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %446 = stablehlo.convert %arg1540 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %447 = stablehlo.convolution(%445, %446) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %448 = stablehlo.broadcast_in_dim %arg1976, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %449 = stablehlo.broadcast_in_dim %arg1977, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %450 = stablehlo.convert %447 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %451 = stablehlo.broadcast_in_dim %448, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %452 = stablehlo.subtract %450, %451 : tensor<256x28x28x512xf32>
      %cst_23 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %453 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %454 = stablehlo.add %449, %453 : tensor<1x1x1x512xf32>
      %455 = stablehlo.rsqrt %454 : tensor<1x1x1x512xf32>
      %456 = stablehlo.reshape %arg1537 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<1x1x1x512xf32>
      %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %459 = stablehlo.multiply %452, %458 : tensor<256x28x28x512xf32>
      %460 = stablehlo.reshape %arg1536 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %462 = stablehlo.add %459, %461 : tensor<256x28x28x512xf32>
      %463 = stablehlo.convert %462 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %464 = stablehlo.add %407, %463 : tensor<256x28x28x512xf16>
      %465 = func.call @relu_83(%464) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %466 = stablehlo.convert %arg1604 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %467 = stablehlo.convolution(%465, %466) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %468 = stablehlo.broadcast_in_dim %arg2016, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %469 = stablehlo.broadcast_in_dim %arg2017, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %470 = stablehlo.convert %467 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %471 = stablehlo.broadcast_in_dim %468, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %472 = stablehlo.subtract %470, %471 : tensor<256x28x28x128xf32>
      %cst_24 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %474 = stablehlo.add %469, %473 : tensor<1x1x1x128xf32>
      %475 = stablehlo.rsqrt %474 : tensor<1x1x1x128xf32>
      %476 = stablehlo.reshape %arg1599 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %477 = stablehlo.multiply %475, %476 : tensor<1x1x1x128xf32>
      %478 = stablehlo.broadcast_in_dim %477, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %479 = stablehlo.multiply %472, %478 : tensor<256x28x28x128xf32>
      %480 = stablehlo.reshape %arg1598 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %482 = stablehlo.add %479, %481 : tensor<256x28x28x128xf32>
      %483 = stablehlo.convert %482 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %484 = func.call @relu_65(%483) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %485 = stablehlo.convert %arg1605 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %486 = stablehlo.convolution(%484, %485) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %487 = stablehlo.broadcast_in_dim %arg2018, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %488 = stablehlo.broadcast_in_dim %arg2019, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %489 = stablehlo.convert %486 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %490 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %491 = stablehlo.subtract %489, %490 : tensor<256x28x28x128xf32>
      %cst_25 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %493 = stablehlo.add %488, %492 : tensor<1x1x1x128xf32>
      %494 = stablehlo.rsqrt %493 : tensor<1x1x1x128xf32>
      %495 = stablehlo.reshape %arg1601 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %496 = stablehlo.multiply %494, %495 : tensor<1x1x1x128xf32>
      %497 = stablehlo.broadcast_in_dim %496, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %498 = stablehlo.multiply %491, %497 : tensor<256x28x28x128xf32>
      %499 = stablehlo.reshape %arg1600 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %500 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %501 = stablehlo.add %498, %500 : tensor<256x28x28x128xf32>
      %502 = stablehlo.convert %501 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %503 = func.call @relu_65(%502) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %504 = stablehlo.convert %arg1606 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %505 = stablehlo.convolution(%503, %504) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %506 = stablehlo.broadcast_in_dim %arg2020, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %507 = stablehlo.broadcast_in_dim %arg2021, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %508 = stablehlo.convert %505 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %509 = stablehlo.broadcast_in_dim %506, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %510 = stablehlo.subtract %508, %509 : tensor<256x28x28x512xf32>
      %cst_26 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %511 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %512 = stablehlo.add %507, %511 : tensor<1x1x1x512xf32>
      %513 = stablehlo.rsqrt %512 : tensor<1x1x1x512xf32>
      %514 = stablehlo.reshape %arg1603 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %515 = stablehlo.multiply %513, %514 : tensor<1x1x1x512xf32>
      %516 = stablehlo.broadcast_in_dim %515, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %517 = stablehlo.multiply %510, %516 : tensor<256x28x28x512xf32>
      %518 = stablehlo.reshape %arg1602 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %519 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %520 = stablehlo.add %517, %519 : tensor<256x28x28x512xf32>
      %521 = stablehlo.convert %520 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %522 = stablehlo.add %465, %521 : tensor<256x28x28x512xf16>
      %523 = func.call @relu_83(%522) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %524 = stablehlo.convert %arg1613 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %525 = stablehlo.convolution(%523, %524) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %526 = stablehlo.broadcast_in_dim %arg2022, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %527 = stablehlo.broadcast_in_dim %arg2023, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %528 = stablehlo.convert %525 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %529 = stablehlo.broadcast_in_dim %526, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %530 = stablehlo.subtract %528, %529 : tensor<256x28x28x128xf32>
      %cst_27 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %531 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %532 = stablehlo.add %527, %531 : tensor<1x1x1x128xf32>
      %533 = stablehlo.rsqrt %532 : tensor<1x1x1x128xf32>
      %534 = stablehlo.reshape %arg1608 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %535 = stablehlo.multiply %533, %534 : tensor<1x1x1x128xf32>
      %536 = stablehlo.broadcast_in_dim %535, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %537 = stablehlo.multiply %530, %536 : tensor<256x28x28x128xf32>
      %538 = stablehlo.reshape %arg1607 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %539 = stablehlo.broadcast_in_dim %538, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %540 = stablehlo.add %537, %539 : tensor<256x28x28x128xf32>
      %541 = stablehlo.convert %540 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %542 = func.call @relu_65(%541) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %543 = stablehlo.convert %arg1614 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %544 = stablehlo.convolution(%542, %543) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %545 = stablehlo.broadcast_in_dim %arg2024, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %546 = stablehlo.broadcast_in_dim %arg2025, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %547 = stablehlo.convert %544 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %548 = stablehlo.broadcast_in_dim %545, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %549 = stablehlo.subtract %547, %548 : tensor<256x28x28x128xf32>
      %cst_28 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %550 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %551 = stablehlo.add %546, %550 : tensor<1x1x1x128xf32>
      %552 = stablehlo.rsqrt %551 : tensor<1x1x1x128xf32>
      %553 = stablehlo.reshape %arg1610 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %554 = stablehlo.multiply %552, %553 : tensor<1x1x1x128xf32>
      %555 = stablehlo.broadcast_in_dim %554, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %556 = stablehlo.multiply %549, %555 : tensor<256x28x28x128xf32>
      %557 = stablehlo.reshape %arg1609 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %558 = stablehlo.broadcast_in_dim %557, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %559 = stablehlo.add %556, %558 : tensor<256x28x28x128xf32>
      %560 = stablehlo.convert %559 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %561 = func.call @relu_65(%560) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %562 = stablehlo.convert %arg1615 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %563 = stablehlo.convolution(%561, %562) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %564 = stablehlo.broadcast_in_dim %arg2026, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %565 = stablehlo.broadcast_in_dim %arg2027, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %566 = stablehlo.convert %563 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %567 = stablehlo.broadcast_in_dim %564, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %568 = stablehlo.subtract %566, %567 : tensor<256x28x28x512xf32>
      %cst_29 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %569 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %570 = stablehlo.add %565, %569 : tensor<1x1x1x512xf32>
      %571 = stablehlo.rsqrt %570 : tensor<1x1x1x512xf32>
      %572 = stablehlo.reshape %arg1612 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %573 = stablehlo.multiply %571, %572 : tensor<1x1x1x512xf32>
      %574 = stablehlo.broadcast_in_dim %573, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %575 = stablehlo.multiply %568, %574 : tensor<256x28x28x512xf32>
      %576 = stablehlo.reshape %arg1611 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %577 = stablehlo.broadcast_in_dim %576, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %578 = stablehlo.add %575, %577 : tensor<256x28x28x512xf32>
      %579 = stablehlo.convert %578 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %580 = stablehlo.add %523, %579 : tensor<256x28x28x512xf16>
      %581 = func.call @relu_83(%580) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %582 = stablehlo.convert %arg1622 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %583 = stablehlo.convolution(%581, %582) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %584 = stablehlo.broadcast_in_dim %arg2028, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %585 = stablehlo.broadcast_in_dim %arg2029, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %586 = stablehlo.convert %583 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %587 = stablehlo.broadcast_in_dim %584, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %588 = stablehlo.subtract %586, %587 : tensor<256x28x28x128xf32>
      %cst_30 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %589 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %590 = stablehlo.add %585, %589 : tensor<1x1x1x128xf32>
      %591 = stablehlo.rsqrt %590 : tensor<1x1x1x128xf32>
      %592 = stablehlo.reshape %arg1617 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %593 = stablehlo.multiply %591, %592 : tensor<1x1x1x128xf32>
      %594 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %595 = stablehlo.multiply %588, %594 : tensor<256x28x28x128xf32>
      %596 = stablehlo.reshape %arg1616 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %597 = stablehlo.broadcast_in_dim %596, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %598 = stablehlo.add %595, %597 : tensor<256x28x28x128xf32>
      %599 = stablehlo.convert %598 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %600 = func.call @relu_65(%599) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %601 = stablehlo.convert %arg1623 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %602 = stablehlo.convolution(%600, %601) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %603 = stablehlo.broadcast_in_dim %arg2030, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %604 = stablehlo.broadcast_in_dim %arg2031, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %605 = stablehlo.convert %602 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %606 = stablehlo.broadcast_in_dim %603, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %607 = stablehlo.subtract %605, %606 : tensor<256x28x28x128xf32>
      %cst_31 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %608 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %609 = stablehlo.add %604, %608 : tensor<1x1x1x128xf32>
      %610 = stablehlo.rsqrt %609 : tensor<1x1x1x128xf32>
      %611 = stablehlo.reshape %arg1619 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %612 = stablehlo.multiply %610, %611 : tensor<1x1x1x128xf32>
      %613 = stablehlo.broadcast_in_dim %612, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %614 = stablehlo.multiply %607, %613 : tensor<256x28x28x128xf32>
      %615 = stablehlo.reshape %arg1618 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %616 = stablehlo.broadcast_in_dim %615, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %617 = stablehlo.add %614, %616 : tensor<256x28x28x128xf32>
      %618 = stablehlo.convert %617 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %619 = func.call @relu_65(%618) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %620 = stablehlo.convert %arg1624 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %621 = stablehlo.convolution(%619, %620) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %622 = stablehlo.broadcast_in_dim %arg2032, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %623 = stablehlo.broadcast_in_dim %arg2033, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %624 = stablehlo.convert %621 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %625 = stablehlo.broadcast_in_dim %622, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %626 = stablehlo.subtract %624, %625 : tensor<256x28x28x512xf32>
      %cst_32 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %627 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %628 = stablehlo.add %623, %627 : tensor<1x1x1x512xf32>
      %629 = stablehlo.rsqrt %628 : tensor<1x1x1x512xf32>
      %630 = stablehlo.reshape %arg1621 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %631 = stablehlo.multiply %629, %630 : tensor<1x1x1x512xf32>
      %632 = stablehlo.broadcast_in_dim %631, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %633 = stablehlo.multiply %626, %632 : tensor<256x28x28x512xf32>
      %634 = stablehlo.reshape %arg1620 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %635 = stablehlo.broadcast_in_dim %634, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %636 = stablehlo.add %633, %635 : tensor<256x28x28x512xf32>
      %637 = stablehlo.convert %636 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %638 = stablehlo.add %581, %637 : tensor<256x28x28x512xf16>
      %639 = func.call @relu_83(%638) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %640 = stablehlo.convert %arg1046 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %641 = stablehlo.convolution(%639, %640) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %642 = stablehlo.broadcast_in_dim %arg1644, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %643 = stablehlo.broadcast_in_dim %arg1645, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %644 = stablehlo.convert %641 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %645 = stablehlo.broadcast_in_dim %642, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %646 = stablehlo.subtract %644, %645 : tensor<256x28x28x128xf32>
      %cst_33 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %648 = stablehlo.add %643, %647 : tensor<1x1x1x128xf32>
      %649 = stablehlo.rsqrt %648 : tensor<1x1x1x128xf32>
      %650 = stablehlo.reshape %arg1041 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %651 = stablehlo.multiply %649, %650 : tensor<1x1x1x128xf32>
      %652 = stablehlo.broadcast_in_dim %651, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %653 = stablehlo.multiply %646, %652 : tensor<256x28x28x128xf32>
      %654 = stablehlo.reshape %arg1040 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %655 = stablehlo.broadcast_in_dim %654, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %656 = stablehlo.add %653, %655 : tensor<256x28x28x128xf32>
      %657 = stablehlo.convert %656 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %658 = func.call @relu_65(%657) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %659 = stablehlo.convert %arg1047 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %660 = stablehlo.convolution(%658, %659) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %661 = stablehlo.broadcast_in_dim %arg1646, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %662 = stablehlo.broadcast_in_dim %arg1647, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %663 = stablehlo.convert %660 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %664 = stablehlo.broadcast_in_dim %661, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %665 = stablehlo.subtract %663, %664 : tensor<256x28x28x128xf32>
      %cst_34 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %666 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %667 = stablehlo.add %662, %666 : tensor<1x1x1x128xf32>
      %668 = stablehlo.rsqrt %667 : tensor<1x1x1x128xf32>
      %669 = stablehlo.reshape %arg1043 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %670 = stablehlo.multiply %668, %669 : tensor<1x1x1x128xf32>
      %671 = stablehlo.broadcast_in_dim %670, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %672 = stablehlo.multiply %665, %671 : tensor<256x28x28x128xf32>
      %673 = stablehlo.reshape %arg1042 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %674 = stablehlo.broadcast_in_dim %673, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %675 = stablehlo.add %672, %674 : tensor<256x28x28x128xf32>
      %676 = stablehlo.convert %675 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %677 = func.call @relu_65(%676) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %678 = stablehlo.convert %arg1048 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %679 = stablehlo.convolution(%677, %678) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %680 = stablehlo.broadcast_in_dim %arg1648, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %681 = stablehlo.broadcast_in_dim %arg1649, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %682 = stablehlo.convert %679 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %683 = stablehlo.broadcast_in_dim %680, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %684 = stablehlo.subtract %682, %683 : tensor<256x28x28x512xf32>
      %cst_35 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %686 = stablehlo.add %681, %685 : tensor<1x1x1x512xf32>
      %687 = stablehlo.rsqrt %686 : tensor<1x1x1x512xf32>
      %688 = stablehlo.reshape %arg1045 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %689 = stablehlo.multiply %687, %688 : tensor<1x1x1x512xf32>
      %690 = stablehlo.broadcast_in_dim %689, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %691 = stablehlo.multiply %684, %690 : tensor<256x28x28x512xf32>
      %692 = stablehlo.reshape %arg1044 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %693 = stablehlo.broadcast_in_dim %692, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %694 = stablehlo.add %691, %693 : tensor<256x28x28x512xf32>
      %695 = stablehlo.convert %694 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %696 = stablehlo.add %639, %695 : tensor<256x28x28x512xf16>
      %697 = func.call @relu_83(%696) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %698 = stablehlo.convert %arg1055 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %699 = stablehlo.convolution(%697, %698) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %700 = stablehlo.broadcast_in_dim %arg1650, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %701 = stablehlo.broadcast_in_dim %arg1651, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %702 = stablehlo.convert %699 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %703 = stablehlo.broadcast_in_dim %700, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %704 = stablehlo.subtract %702, %703 : tensor<256x28x28x128xf32>
      %cst_36 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %705 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %706 = stablehlo.add %701, %705 : tensor<1x1x1x128xf32>
      %707 = stablehlo.rsqrt %706 : tensor<1x1x1x128xf32>
      %708 = stablehlo.reshape %arg1050 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %709 = stablehlo.multiply %707, %708 : tensor<1x1x1x128xf32>
      %710 = stablehlo.broadcast_in_dim %709, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %711 = stablehlo.multiply %704, %710 : tensor<256x28x28x128xf32>
      %712 = stablehlo.reshape %arg1049 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %713 = stablehlo.broadcast_in_dim %712, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %714 = stablehlo.add %711, %713 : tensor<256x28x28x128xf32>
      %715 = stablehlo.convert %714 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %716 = func.call @relu_65(%715) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %717 = stablehlo.convert %arg1056 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %718 = stablehlo.convolution(%716, %717) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %719 = stablehlo.broadcast_in_dim %arg1652, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %720 = stablehlo.broadcast_in_dim %arg1653, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %721 = stablehlo.convert %718 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %722 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %723 = stablehlo.subtract %721, %722 : tensor<256x28x28x128xf32>
      %cst_37 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %724 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %725 = stablehlo.add %720, %724 : tensor<1x1x1x128xf32>
      %726 = stablehlo.rsqrt %725 : tensor<1x1x1x128xf32>
      %727 = stablehlo.reshape %arg1052 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %728 = stablehlo.multiply %726, %727 : tensor<1x1x1x128xf32>
      %729 = stablehlo.broadcast_in_dim %728, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %730 = stablehlo.multiply %723, %729 : tensor<256x28x28x128xf32>
      %731 = stablehlo.reshape %arg1051 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %732 = stablehlo.broadcast_in_dim %731, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %733 = stablehlo.add %730, %732 : tensor<256x28x28x128xf32>
      %734 = stablehlo.convert %733 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %735 = func.call @relu_65(%734) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %736 = stablehlo.convert %arg1057 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %737 = stablehlo.convolution(%735, %736) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %738 = stablehlo.broadcast_in_dim %arg1654, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %739 = stablehlo.broadcast_in_dim %arg1655, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %740 = stablehlo.convert %737 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %741 = stablehlo.broadcast_in_dim %738, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %742 = stablehlo.subtract %740, %741 : tensor<256x28x28x512xf32>
      %cst_38 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %743 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %744 = stablehlo.add %739, %743 : tensor<1x1x1x512xf32>
      %745 = stablehlo.rsqrt %744 : tensor<1x1x1x512xf32>
      %746 = stablehlo.reshape %arg1054 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %747 = stablehlo.multiply %745, %746 : tensor<1x1x1x512xf32>
      %748 = stablehlo.broadcast_in_dim %747, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %749 = stablehlo.multiply %742, %748 : tensor<256x28x28x512xf32>
      %750 = stablehlo.reshape %arg1053 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %751 = stablehlo.broadcast_in_dim %750, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %752 = stablehlo.add %749, %751 : tensor<256x28x28x512xf32>
      %753 = stablehlo.convert %752 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %754 = stablehlo.add %697, %753 : tensor<256x28x28x512xf16>
      %755 = func.call @relu_83(%754) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %756 = stablehlo.convert %arg1064 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %757 = stablehlo.convolution(%755, %756) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %758 = stablehlo.broadcast_in_dim %arg1656, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %759 = stablehlo.broadcast_in_dim %arg1657, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %760 = stablehlo.convert %757 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %761 = stablehlo.broadcast_in_dim %758, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %762 = stablehlo.subtract %760, %761 : tensor<256x28x28x128xf32>
      %cst_39 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %763 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %764 = stablehlo.add %759, %763 : tensor<1x1x1x128xf32>
      %765 = stablehlo.rsqrt %764 : tensor<1x1x1x128xf32>
      %766 = stablehlo.reshape %arg1059 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %767 = stablehlo.multiply %765, %766 : tensor<1x1x1x128xf32>
      %768 = stablehlo.broadcast_in_dim %767, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %769 = stablehlo.multiply %762, %768 : tensor<256x28x28x128xf32>
      %770 = stablehlo.reshape %arg1058 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %771 = stablehlo.broadcast_in_dim %770, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %772 = stablehlo.add %769, %771 : tensor<256x28x28x128xf32>
      %773 = stablehlo.convert %772 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %774 = func.call @relu_65(%773) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %775 = stablehlo.convert %arg1065 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %776 = stablehlo.convolution(%774, %775) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %777 = stablehlo.broadcast_in_dim %arg1658, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %778 = stablehlo.broadcast_in_dim %arg1659, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %779 = stablehlo.convert %776 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %780 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %781 = stablehlo.subtract %779, %780 : tensor<256x28x28x128xf32>
      %cst_40 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %782 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %783 = stablehlo.add %778, %782 : tensor<1x1x1x128xf32>
      %784 = stablehlo.rsqrt %783 : tensor<1x1x1x128xf32>
      %785 = stablehlo.reshape %arg1061 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %786 = stablehlo.multiply %784, %785 : tensor<1x1x1x128xf32>
      %787 = stablehlo.broadcast_in_dim %786, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %788 = stablehlo.multiply %781, %787 : tensor<256x28x28x128xf32>
      %789 = stablehlo.reshape %arg1060 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %790 = stablehlo.broadcast_in_dim %789, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %791 = stablehlo.add %788, %790 : tensor<256x28x28x128xf32>
      %792 = stablehlo.convert %791 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %793 = func.call @relu_65(%792) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %794 = stablehlo.convert %arg1066 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %795 = stablehlo.convolution(%793, %794) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %796 = stablehlo.broadcast_in_dim %arg1660, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %797 = stablehlo.broadcast_in_dim %arg1661, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %798 = stablehlo.convert %795 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %799 = stablehlo.broadcast_in_dim %796, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %800 = stablehlo.subtract %798, %799 : tensor<256x28x28x512xf32>
      %cst_41 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %801 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %802 = stablehlo.add %797, %801 : tensor<1x1x1x512xf32>
      %803 = stablehlo.rsqrt %802 : tensor<1x1x1x512xf32>
      %804 = stablehlo.reshape %arg1063 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %805 = stablehlo.multiply %803, %804 : tensor<1x1x1x512xf32>
      %806 = stablehlo.broadcast_in_dim %805, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %807 = stablehlo.multiply %800, %806 : tensor<256x28x28x512xf32>
      %808 = stablehlo.reshape %arg1062 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %809 = stablehlo.broadcast_in_dim %808, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %810 = stablehlo.add %807, %809 : tensor<256x28x28x512xf32>
      %811 = stablehlo.convert %810 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %812 = stablehlo.add %755, %811 : tensor<256x28x28x512xf16>
      %813 = func.call @relu_83(%812) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %814 = stablehlo.convert %arg1073 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %815 = stablehlo.convolution(%813, %814) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %816 = stablehlo.broadcast_in_dim %arg1662, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %817 = stablehlo.broadcast_in_dim %arg1663, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %818 = stablehlo.convert %815 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %819 = stablehlo.broadcast_in_dim %816, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %820 = stablehlo.subtract %818, %819 : tensor<256x28x28x128xf32>
      %cst_42 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %821 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %822 = stablehlo.add %817, %821 : tensor<1x1x1x128xf32>
      %823 = stablehlo.rsqrt %822 : tensor<1x1x1x128xf32>
      %824 = stablehlo.reshape %arg1068 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %825 = stablehlo.multiply %823, %824 : tensor<1x1x1x128xf32>
      %826 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %827 = stablehlo.multiply %820, %826 : tensor<256x28x28x128xf32>
      %828 = stablehlo.reshape %arg1067 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %829 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %830 = stablehlo.add %827, %829 : tensor<256x28x28x128xf32>
      %831 = stablehlo.convert %830 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %832 = func.call @relu_65(%831) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %833 = stablehlo.convert %arg1074 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %834 = stablehlo.convolution(%832, %833) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %835 = stablehlo.broadcast_in_dim %arg1664, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %836 = stablehlo.broadcast_in_dim %arg1665, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %837 = stablehlo.convert %834 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %838 = stablehlo.broadcast_in_dim %835, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %839 = stablehlo.subtract %837, %838 : tensor<256x28x28x128xf32>
      %cst_43 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %840 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %841 = stablehlo.add %836, %840 : tensor<1x1x1x128xf32>
      %842 = stablehlo.rsqrt %841 : tensor<1x1x1x128xf32>
      %843 = stablehlo.reshape %arg1070 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %844 = stablehlo.multiply %842, %843 : tensor<1x1x1x128xf32>
      %845 = stablehlo.broadcast_in_dim %844, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %846 = stablehlo.multiply %839, %845 : tensor<256x28x28x128xf32>
      %847 = stablehlo.reshape %arg1069 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %848 = stablehlo.broadcast_in_dim %847, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %849 = stablehlo.add %846, %848 : tensor<256x28x28x128xf32>
      %850 = stablehlo.convert %849 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %851 = func.call @relu_65(%850) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %852 = stablehlo.convert %arg1075 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %853 = stablehlo.convolution(%851, %852) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %854 = stablehlo.broadcast_in_dim %arg1666, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %855 = stablehlo.broadcast_in_dim %arg1667, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %856 = stablehlo.convert %853 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %857 = stablehlo.broadcast_in_dim %854, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %858 = stablehlo.subtract %856, %857 : tensor<256x28x28x512xf32>
      %cst_44 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %859 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %860 = stablehlo.add %855, %859 : tensor<1x1x1x512xf32>
      %861 = stablehlo.rsqrt %860 : tensor<1x1x1x512xf32>
      %862 = stablehlo.reshape %arg1072 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %863 = stablehlo.multiply %861, %862 : tensor<1x1x1x512xf32>
      %864 = stablehlo.broadcast_in_dim %863, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %865 = stablehlo.multiply %858, %864 : tensor<256x28x28x512xf32>
      %866 = stablehlo.reshape %arg1071 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %867 = stablehlo.broadcast_in_dim %866, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %868 = stablehlo.add %865, %867 : tensor<256x28x28x512xf32>
      %869 = stablehlo.convert %868 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %870 = stablehlo.add %813, %869 : tensor<256x28x28x512xf16>
      %871 = func.call @relu_83(%870) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %872 = stablehlo.convert %arg1082 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %873 = stablehlo.convolution(%871, %872) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %874 = stablehlo.broadcast_in_dim %arg1668, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %875 = stablehlo.broadcast_in_dim %arg1669, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %876 = stablehlo.convert %873 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %877 = stablehlo.broadcast_in_dim %874, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %878 = stablehlo.subtract %876, %877 : tensor<256x28x28x128xf32>
      %cst_45 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %879 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %880 = stablehlo.add %875, %879 : tensor<1x1x1x128xf32>
      %881 = stablehlo.rsqrt %880 : tensor<1x1x1x128xf32>
      %882 = stablehlo.reshape %arg1077 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %883 = stablehlo.multiply %881, %882 : tensor<1x1x1x128xf32>
      %884 = stablehlo.broadcast_in_dim %883, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %885 = stablehlo.multiply %878, %884 : tensor<256x28x28x128xf32>
      %886 = stablehlo.reshape %arg1076 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %887 = stablehlo.broadcast_in_dim %886, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %888 = stablehlo.add %885, %887 : tensor<256x28x28x128xf32>
      %889 = stablehlo.convert %888 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %890 = func.call @relu_65(%889) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %891 = stablehlo.convert %arg1083 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %892 = stablehlo.convolution(%890, %891) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %893 = stablehlo.broadcast_in_dim %arg1670, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %894 = stablehlo.broadcast_in_dim %arg1671, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %895 = stablehlo.convert %892 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %896 = stablehlo.broadcast_in_dim %893, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %897 = stablehlo.subtract %895, %896 : tensor<256x28x28x128xf32>
      %cst_46 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %898 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %899 = stablehlo.add %894, %898 : tensor<1x1x1x128xf32>
      %900 = stablehlo.rsqrt %899 : tensor<1x1x1x128xf32>
      %901 = stablehlo.reshape %arg1079 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %902 = stablehlo.multiply %900, %901 : tensor<1x1x1x128xf32>
      %903 = stablehlo.broadcast_in_dim %902, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %904 = stablehlo.multiply %897, %903 : tensor<256x28x28x128xf32>
      %905 = stablehlo.reshape %arg1078 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %906 = stablehlo.broadcast_in_dim %905, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %907 = stablehlo.add %904, %906 : tensor<256x28x28x128xf32>
      %908 = stablehlo.convert %907 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %909 = func.call @relu_65(%908) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %910 = stablehlo.convert %arg1084 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %911 = stablehlo.convolution(%909, %910) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %912 = stablehlo.broadcast_in_dim %arg1672, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %913 = stablehlo.broadcast_in_dim %arg1673, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %914 = stablehlo.convert %911 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %915 = stablehlo.broadcast_in_dim %912, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %916 = stablehlo.subtract %914, %915 : tensor<256x28x28x512xf32>
      %cst_47 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %917 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %918 = stablehlo.add %913, %917 : tensor<1x1x1x512xf32>
      %919 = stablehlo.rsqrt %918 : tensor<1x1x1x512xf32>
      %920 = stablehlo.reshape %arg1081 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %921 = stablehlo.multiply %919, %920 : tensor<1x1x1x512xf32>
      %922 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %923 = stablehlo.multiply %916, %922 : tensor<256x28x28x512xf32>
      %924 = stablehlo.reshape %arg1080 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %925 = stablehlo.broadcast_in_dim %924, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %926 = stablehlo.add %923, %925 : tensor<256x28x28x512xf32>
      %927 = stablehlo.convert %926 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %928 = stablehlo.add %871, %927 : tensor<256x28x28x512xf16>
      %929 = func.call @relu_83(%928) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %930 = stablehlo.convert %arg1091 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %931 = stablehlo.convolution(%929, %930) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %932 = stablehlo.broadcast_in_dim %arg1674, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %933 = stablehlo.broadcast_in_dim %arg1675, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %934 = stablehlo.convert %931 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %935 = stablehlo.broadcast_in_dim %932, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %936 = stablehlo.subtract %934, %935 : tensor<256x28x28x128xf32>
      %cst_48 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %937 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %938 = stablehlo.add %933, %937 : tensor<1x1x1x128xf32>
      %939 = stablehlo.rsqrt %938 : tensor<1x1x1x128xf32>
      %940 = stablehlo.reshape %arg1086 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %941 = stablehlo.multiply %939, %940 : tensor<1x1x1x128xf32>
      %942 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %943 = stablehlo.multiply %936, %942 : tensor<256x28x28x128xf32>
      %944 = stablehlo.reshape %arg1085 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %945 = stablehlo.broadcast_in_dim %944, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %946 = stablehlo.add %943, %945 : tensor<256x28x28x128xf32>
      %947 = stablehlo.convert %946 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %948 = func.call @relu_65(%947) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %949 = stablehlo.convert %arg1092 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %950 = stablehlo.convolution(%948, %949) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %951 = stablehlo.broadcast_in_dim %arg1676, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %952 = stablehlo.broadcast_in_dim %arg1677, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %953 = stablehlo.convert %950 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %954 = stablehlo.broadcast_in_dim %951, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %955 = stablehlo.subtract %953, %954 : tensor<256x28x28x128xf32>
      %cst_49 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %956 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %957 = stablehlo.add %952, %956 : tensor<1x1x1x128xf32>
      %958 = stablehlo.rsqrt %957 : tensor<1x1x1x128xf32>
      %959 = stablehlo.reshape %arg1088 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %960 = stablehlo.multiply %958, %959 : tensor<1x1x1x128xf32>
      %961 = stablehlo.broadcast_in_dim %960, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %962 = stablehlo.multiply %955, %961 : tensor<256x28x28x128xf32>
      %963 = stablehlo.reshape %arg1087 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %964 = stablehlo.broadcast_in_dim %963, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %965 = stablehlo.add %962, %964 : tensor<256x28x28x128xf32>
      %966 = stablehlo.convert %965 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %967 = func.call @relu_65(%966) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %968 = stablehlo.convert %arg1093 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %969 = stablehlo.convolution(%967, %968) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %970 = stablehlo.broadcast_in_dim %arg1678, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %971 = stablehlo.broadcast_in_dim %arg1679, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %972 = stablehlo.convert %969 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %973 = stablehlo.broadcast_in_dim %970, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %974 = stablehlo.subtract %972, %973 : tensor<256x28x28x512xf32>
      %cst_50 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %975 = stablehlo.broadcast_in_dim %cst_50, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %976 = stablehlo.add %971, %975 : tensor<1x1x1x512xf32>
      %977 = stablehlo.rsqrt %976 : tensor<1x1x1x512xf32>
      %978 = stablehlo.reshape %arg1090 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %979 = stablehlo.multiply %977, %978 : tensor<1x1x1x512xf32>
      %980 = stablehlo.broadcast_in_dim %979, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %981 = stablehlo.multiply %974, %980 : tensor<256x28x28x512xf32>
      %982 = stablehlo.reshape %arg1089 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %983 = stablehlo.broadcast_in_dim %982, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %984 = stablehlo.add %981, %983 : tensor<256x28x28x512xf32>
      %985 = stablehlo.convert %984 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %986 = stablehlo.add %929, %985 : tensor<256x28x28x512xf16>
      %987 = func.call @relu_83(%986) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %988 = stablehlo.convert %arg1100 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %989 = stablehlo.convolution(%987, %988) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %990 = stablehlo.broadcast_in_dim %arg1680, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %991 = stablehlo.broadcast_in_dim %arg1681, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %992 = stablehlo.convert %989 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %993 = stablehlo.broadcast_in_dim %990, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %994 = stablehlo.subtract %992, %993 : tensor<256x28x28x128xf32>
      %cst_51 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %995 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %996 = stablehlo.add %991, %995 : tensor<1x1x1x128xf32>
      %997 = stablehlo.rsqrt %996 : tensor<1x1x1x128xf32>
      %998 = stablehlo.reshape %arg1095 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %999 = stablehlo.multiply %997, %998 : tensor<1x1x1x128xf32>
      %1000 = stablehlo.broadcast_in_dim %999, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1001 = stablehlo.multiply %994, %1000 : tensor<256x28x28x128xf32>
      %1002 = stablehlo.reshape %arg1094 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1003 = stablehlo.broadcast_in_dim %1002, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1004 = stablehlo.add %1001, %1003 : tensor<256x28x28x128xf32>
      %1005 = stablehlo.convert %1004 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1006 = func.call @relu_65(%1005) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1007 = stablehlo.convert %arg1101 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1008 = stablehlo.convolution(%1006, %1007) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1009 = stablehlo.broadcast_in_dim %arg1682, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1010 = stablehlo.broadcast_in_dim %arg1683, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1011 = stablehlo.convert %1008 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1012 = stablehlo.broadcast_in_dim %1009, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1013 = stablehlo.subtract %1011, %1012 : tensor<256x28x28x128xf32>
      %cst_52 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1014 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1015 = stablehlo.add %1010, %1014 : tensor<1x1x1x128xf32>
      %1016 = stablehlo.rsqrt %1015 : tensor<1x1x1x128xf32>
      %1017 = stablehlo.reshape %arg1097 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1018 = stablehlo.multiply %1016, %1017 : tensor<1x1x1x128xf32>
      %1019 = stablehlo.broadcast_in_dim %1018, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1020 = stablehlo.multiply %1013, %1019 : tensor<256x28x28x128xf32>
      %1021 = stablehlo.reshape %arg1096 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1022 = stablehlo.broadcast_in_dim %1021, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1023 = stablehlo.add %1020, %1022 : tensor<256x28x28x128xf32>
      %1024 = stablehlo.convert %1023 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1025 = func.call @relu_65(%1024) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1026 = stablehlo.convert %arg1102 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1027 = stablehlo.convolution(%1025, %1026) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1028 = stablehlo.broadcast_in_dim %arg1684, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1029 = stablehlo.broadcast_in_dim %arg1685, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1030 = stablehlo.convert %1027 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1031 = stablehlo.broadcast_in_dim %1028, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1032 = stablehlo.subtract %1030, %1031 : tensor<256x28x28x512xf32>
      %cst_53 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1033 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1034 = stablehlo.add %1029, %1033 : tensor<1x1x1x512xf32>
      %1035 = stablehlo.rsqrt %1034 : tensor<1x1x1x512xf32>
      %1036 = stablehlo.reshape %arg1099 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1037 = stablehlo.multiply %1035, %1036 : tensor<1x1x1x512xf32>
      %1038 = stablehlo.broadcast_in_dim %1037, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1039 = stablehlo.multiply %1032, %1038 : tensor<256x28x28x512xf32>
      %1040 = stablehlo.reshape %arg1098 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1041 = stablehlo.broadcast_in_dim %1040, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1042 = stablehlo.add %1039, %1041 : tensor<256x28x28x512xf32>
      %1043 = stablehlo.convert %1042 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1044 = stablehlo.add %987, %1043 : tensor<256x28x28x512xf16>
      %1045 = func.call @relu_83(%1044) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1046 = stablehlo.convert %arg1109 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1047 = stablehlo.convolution(%1045, %1046) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1048 = stablehlo.broadcast_in_dim %arg1686, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1049 = stablehlo.broadcast_in_dim %arg1687, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1050 = stablehlo.convert %1047 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1051 = stablehlo.broadcast_in_dim %1048, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1052 = stablehlo.subtract %1050, %1051 : tensor<256x28x28x128xf32>
      %cst_54 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1053 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1054 = stablehlo.add %1049, %1053 : tensor<1x1x1x128xf32>
      %1055 = stablehlo.rsqrt %1054 : tensor<1x1x1x128xf32>
      %1056 = stablehlo.reshape %arg1104 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1057 = stablehlo.multiply %1055, %1056 : tensor<1x1x1x128xf32>
      %1058 = stablehlo.broadcast_in_dim %1057, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1059 = stablehlo.multiply %1052, %1058 : tensor<256x28x28x128xf32>
      %1060 = stablehlo.reshape %arg1103 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1061 = stablehlo.broadcast_in_dim %1060, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1062 = stablehlo.add %1059, %1061 : tensor<256x28x28x128xf32>
      %1063 = stablehlo.convert %1062 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1064 = func.call @relu_65(%1063) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1065 = stablehlo.convert %arg1110 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1066 = stablehlo.convolution(%1064, %1065) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1067 = stablehlo.broadcast_in_dim %arg1688, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1068 = stablehlo.broadcast_in_dim %arg1689, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1069 = stablehlo.convert %1066 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1070 = stablehlo.broadcast_in_dim %1067, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1071 = stablehlo.subtract %1069, %1070 : tensor<256x28x28x128xf32>
      %cst_55 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1072 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1073 = stablehlo.add %1068, %1072 : tensor<1x1x1x128xf32>
      %1074 = stablehlo.rsqrt %1073 : tensor<1x1x1x128xf32>
      %1075 = stablehlo.reshape %arg1106 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1076 = stablehlo.multiply %1074, %1075 : tensor<1x1x1x128xf32>
      %1077 = stablehlo.broadcast_in_dim %1076, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1078 = stablehlo.multiply %1071, %1077 : tensor<256x28x28x128xf32>
      %1079 = stablehlo.reshape %arg1105 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1080 = stablehlo.broadcast_in_dim %1079, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1081 = stablehlo.add %1078, %1080 : tensor<256x28x28x128xf32>
      %1082 = stablehlo.convert %1081 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1083 = func.call @relu_65(%1082) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1084 = stablehlo.convert %arg1111 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1085 = stablehlo.convolution(%1083, %1084) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1086 = stablehlo.broadcast_in_dim %arg1690, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1087 = stablehlo.broadcast_in_dim %arg1691, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1088 = stablehlo.convert %1085 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1089 = stablehlo.broadcast_in_dim %1086, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1090 = stablehlo.subtract %1088, %1089 : tensor<256x28x28x512xf32>
      %cst_56 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1091 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1092 = stablehlo.add %1087, %1091 : tensor<1x1x1x512xf32>
      %1093 = stablehlo.rsqrt %1092 : tensor<1x1x1x512xf32>
      %1094 = stablehlo.reshape %arg1108 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1095 = stablehlo.multiply %1093, %1094 : tensor<1x1x1x512xf32>
      %1096 = stablehlo.broadcast_in_dim %1095, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1097 = stablehlo.multiply %1090, %1096 : tensor<256x28x28x512xf32>
      %1098 = stablehlo.reshape %arg1107 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1099 = stablehlo.broadcast_in_dim %1098, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1100 = stablehlo.add %1097, %1099 : tensor<256x28x28x512xf32>
      %1101 = stablehlo.convert %1100 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1102 = stablehlo.add %1045, %1101 : tensor<256x28x28x512xf16>
      %1103 = func.call @relu_83(%1102) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1104 = stablehlo.convert %arg1118 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1105 = stablehlo.convolution(%1103, %1104) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1106 = stablehlo.broadcast_in_dim %arg1692, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1107 = stablehlo.broadcast_in_dim %arg1693, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1108 = stablehlo.convert %1105 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1109 = stablehlo.broadcast_in_dim %1106, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1110 = stablehlo.subtract %1108, %1109 : tensor<256x28x28x128xf32>
      %cst_57 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1111 = stablehlo.broadcast_in_dim %cst_57, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1112 = stablehlo.add %1107, %1111 : tensor<1x1x1x128xf32>
      %1113 = stablehlo.rsqrt %1112 : tensor<1x1x1x128xf32>
      %1114 = stablehlo.reshape %arg1113 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1115 = stablehlo.multiply %1113, %1114 : tensor<1x1x1x128xf32>
      %1116 = stablehlo.broadcast_in_dim %1115, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1117 = stablehlo.multiply %1110, %1116 : tensor<256x28x28x128xf32>
      %1118 = stablehlo.reshape %arg1112 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1119 = stablehlo.broadcast_in_dim %1118, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1120 = stablehlo.add %1117, %1119 : tensor<256x28x28x128xf32>
      %1121 = stablehlo.convert %1120 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1122 = func.call @relu_65(%1121) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1123 = stablehlo.convert %arg1119 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1124 = stablehlo.convolution(%1122, %1123) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1125 = stablehlo.broadcast_in_dim %arg1694, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1126 = stablehlo.broadcast_in_dim %arg1695, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1127 = stablehlo.convert %1124 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1128 = stablehlo.broadcast_in_dim %1125, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1129 = stablehlo.subtract %1127, %1128 : tensor<256x28x28x128xf32>
      %cst_58 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1130 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1131 = stablehlo.add %1126, %1130 : tensor<1x1x1x128xf32>
      %1132 = stablehlo.rsqrt %1131 : tensor<1x1x1x128xf32>
      %1133 = stablehlo.reshape %arg1115 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1134 = stablehlo.multiply %1132, %1133 : tensor<1x1x1x128xf32>
      %1135 = stablehlo.broadcast_in_dim %1134, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1136 = stablehlo.multiply %1129, %1135 : tensor<256x28x28x128xf32>
      %1137 = stablehlo.reshape %arg1114 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1138 = stablehlo.broadcast_in_dim %1137, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1139 = stablehlo.add %1136, %1138 : tensor<256x28x28x128xf32>
      %1140 = stablehlo.convert %1139 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1141 = func.call @relu_65(%1140) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1142 = stablehlo.convert %arg1120 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1143 = stablehlo.convolution(%1141, %1142) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1144 = stablehlo.broadcast_in_dim %arg1696, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1145 = stablehlo.broadcast_in_dim %arg1697, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1146 = stablehlo.convert %1143 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1147 = stablehlo.broadcast_in_dim %1144, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1148 = stablehlo.subtract %1146, %1147 : tensor<256x28x28x512xf32>
      %cst_59 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1149 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1150 = stablehlo.add %1145, %1149 : tensor<1x1x1x512xf32>
      %1151 = stablehlo.rsqrt %1150 : tensor<1x1x1x512xf32>
      %1152 = stablehlo.reshape %arg1117 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1153 = stablehlo.multiply %1151, %1152 : tensor<1x1x1x512xf32>
      %1154 = stablehlo.broadcast_in_dim %1153, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1155 = stablehlo.multiply %1148, %1154 : tensor<256x28x28x512xf32>
      %1156 = stablehlo.reshape %arg1116 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1157 = stablehlo.broadcast_in_dim %1156, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1158 = stablehlo.add %1155, %1157 : tensor<256x28x28x512xf32>
      %1159 = stablehlo.convert %1158 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1160 = stablehlo.add %1103, %1159 : tensor<256x28x28x512xf16>
      %1161 = func.call @relu_83(%1160) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1162 = stablehlo.convert %arg1127 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1163 = stablehlo.convolution(%1161, %1162) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1164 = stablehlo.broadcast_in_dim %arg1698, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1165 = stablehlo.broadcast_in_dim %arg1699, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1166 = stablehlo.convert %1163 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1167 = stablehlo.broadcast_in_dim %1164, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1168 = stablehlo.subtract %1166, %1167 : tensor<256x28x28x128xf32>
      %cst_60 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1169 = stablehlo.broadcast_in_dim %cst_60, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1170 = stablehlo.add %1165, %1169 : tensor<1x1x1x128xf32>
      %1171 = stablehlo.rsqrt %1170 : tensor<1x1x1x128xf32>
      %1172 = stablehlo.reshape %arg1122 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1173 = stablehlo.multiply %1171, %1172 : tensor<1x1x1x128xf32>
      %1174 = stablehlo.broadcast_in_dim %1173, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1175 = stablehlo.multiply %1168, %1174 : tensor<256x28x28x128xf32>
      %1176 = stablehlo.reshape %arg1121 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1177 = stablehlo.broadcast_in_dim %1176, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1178 = stablehlo.add %1175, %1177 : tensor<256x28x28x128xf32>
      %1179 = stablehlo.convert %1178 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1180 = func.call @relu_65(%1179) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1181 = stablehlo.convert %arg1128 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1182 = stablehlo.convolution(%1180, %1181) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1183 = stablehlo.broadcast_in_dim %arg1700, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1184 = stablehlo.broadcast_in_dim %arg1701, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1185 = stablehlo.convert %1182 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1186 = stablehlo.broadcast_in_dim %1183, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1187 = stablehlo.subtract %1185, %1186 : tensor<256x28x28x128xf32>
      %cst_61 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1188 = stablehlo.broadcast_in_dim %cst_61, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1189 = stablehlo.add %1184, %1188 : tensor<1x1x1x128xf32>
      %1190 = stablehlo.rsqrt %1189 : tensor<1x1x1x128xf32>
      %1191 = stablehlo.reshape %arg1124 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1192 = stablehlo.multiply %1190, %1191 : tensor<1x1x1x128xf32>
      %1193 = stablehlo.broadcast_in_dim %1192, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1194 = stablehlo.multiply %1187, %1193 : tensor<256x28x28x128xf32>
      %1195 = stablehlo.reshape %arg1123 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1196 = stablehlo.broadcast_in_dim %1195, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1197 = stablehlo.add %1194, %1196 : tensor<256x28x28x128xf32>
      %1198 = stablehlo.convert %1197 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1199 = func.call @relu_65(%1198) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1200 = stablehlo.convert %arg1129 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1201 = stablehlo.convolution(%1199, %1200) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1202 = stablehlo.broadcast_in_dim %arg1702, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1203 = stablehlo.broadcast_in_dim %arg1703, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1204 = stablehlo.convert %1201 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1205 = stablehlo.broadcast_in_dim %1202, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1206 = stablehlo.subtract %1204, %1205 : tensor<256x28x28x512xf32>
      %cst_62 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1207 = stablehlo.broadcast_in_dim %cst_62, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1208 = stablehlo.add %1203, %1207 : tensor<1x1x1x512xf32>
      %1209 = stablehlo.rsqrt %1208 : tensor<1x1x1x512xf32>
      %1210 = stablehlo.reshape %arg1126 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1211 = stablehlo.multiply %1209, %1210 : tensor<1x1x1x512xf32>
      %1212 = stablehlo.broadcast_in_dim %1211, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1213 = stablehlo.multiply %1206, %1212 : tensor<256x28x28x512xf32>
      %1214 = stablehlo.reshape %arg1125 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1215 = stablehlo.broadcast_in_dim %1214, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1216 = stablehlo.add %1213, %1215 : tensor<256x28x28x512xf32>
      %1217 = stablehlo.convert %1216 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1218 = stablehlo.add %1161, %1217 : tensor<256x28x28x512xf16>
      %1219 = func.call @relu_83(%1218) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1220 = stablehlo.convert %arg1145 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1221 = stablehlo.convolution(%1219, %1220) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1222 = stablehlo.broadcast_in_dim %arg1710, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1223 = stablehlo.broadcast_in_dim %arg1711, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1224 = stablehlo.convert %1221 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1225 = stablehlo.broadcast_in_dim %1222, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1226 = stablehlo.subtract %1224, %1225 : tensor<256x28x28x128xf32>
      %cst_63 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1227 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1228 = stablehlo.add %1223, %1227 : tensor<1x1x1x128xf32>
      %1229 = stablehlo.rsqrt %1228 : tensor<1x1x1x128xf32>
      %1230 = stablehlo.reshape %arg1140 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1231 = stablehlo.multiply %1229, %1230 : tensor<1x1x1x128xf32>
      %1232 = stablehlo.broadcast_in_dim %1231, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1233 = stablehlo.multiply %1226, %1232 : tensor<256x28x28x128xf32>
      %1234 = stablehlo.reshape %arg1139 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1235 = stablehlo.broadcast_in_dim %1234, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1236 = stablehlo.add %1233, %1235 : tensor<256x28x28x128xf32>
      %1237 = stablehlo.convert %1236 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1238 = func.call @relu_65(%1237) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1239 = stablehlo.convert %arg1146 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1240 = stablehlo.convolution(%1238, %1239) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1241 = stablehlo.broadcast_in_dim %arg1712, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1242 = stablehlo.broadcast_in_dim %arg1713, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1243 = stablehlo.convert %1240 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1244 = stablehlo.broadcast_in_dim %1241, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1245 = stablehlo.subtract %1243, %1244 : tensor<256x28x28x128xf32>
      %cst_64 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1246 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1247 = stablehlo.add %1242, %1246 : tensor<1x1x1x128xf32>
      %1248 = stablehlo.rsqrt %1247 : tensor<1x1x1x128xf32>
      %1249 = stablehlo.reshape %arg1142 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1250 = stablehlo.multiply %1248, %1249 : tensor<1x1x1x128xf32>
      %1251 = stablehlo.broadcast_in_dim %1250, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1252 = stablehlo.multiply %1245, %1251 : tensor<256x28x28x128xf32>
      %1253 = stablehlo.reshape %arg1141 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1254 = stablehlo.broadcast_in_dim %1253, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1255 = stablehlo.add %1252, %1254 : tensor<256x28x28x128xf32>
      %1256 = stablehlo.convert %1255 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1257 = func.call @relu_65(%1256) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1258 = stablehlo.convert %arg1147 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1259 = stablehlo.convolution(%1257, %1258) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1260 = stablehlo.broadcast_in_dim %arg1714, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1261 = stablehlo.broadcast_in_dim %arg1715, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1262 = stablehlo.convert %1259 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1263 = stablehlo.broadcast_in_dim %1260, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1264 = stablehlo.subtract %1262, %1263 : tensor<256x28x28x512xf32>
      %cst_65 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1265 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1266 = stablehlo.add %1261, %1265 : tensor<1x1x1x512xf32>
      %1267 = stablehlo.rsqrt %1266 : tensor<1x1x1x512xf32>
      %1268 = stablehlo.reshape %arg1144 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1269 = stablehlo.multiply %1267, %1268 : tensor<1x1x1x512xf32>
      %1270 = stablehlo.broadcast_in_dim %1269, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1271 = stablehlo.multiply %1264, %1270 : tensor<256x28x28x512xf32>
      %1272 = stablehlo.reshape %arg1143 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1273 = stablehlo.broadcast_in_dim %1272, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1274 = stablehlo.add %1271, %1273 : tensor<256x28x28x512xf32>
      %1275 = stablehlo.convert %1274 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1276 = stablehlo.add %1219, %1275 : tensor<256x28x28x512xf16>
      %1277 = func.call @relu_83(%1276) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1278 = stablehlo.convert %arg1154 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1279 = stablehlo.convolution(%1277, %1278) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1280 = stablehlo.broadcast_in_dim %arg1716, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1281 = stablehlo.broadcast_in_dim %arg1717, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1282 = stablehlo.convert %1279 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1283 = stablehlo.broadcast_in_dim %1280, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1284 = stablehlo.subtract %1282, %1283 : tensor<256x28x28x128xf32>
      %cst_66 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1285 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1286 = stablehlo.add %1281, %1285 : tensor<1x1x1x128xf32>
      %1287 = stablehlo.rsqrt %1286 : tensor<1x1x1x128xf32>
      %1288 = stablehlo.reshape %arg1149 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1289 = stablehlo.multiply %1287, %1288 : tensor<1x1x1x128xf32>
      %1290 = stablehlo.broadcast_in_dim %1289, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1291 = stablehlo.multiply %1284, %1290 : tensor<256x28x28x128xf32>
      %1292 = stablehlo.reshape %arg1148 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1293 = stablehlo.broadcast_in_dim %1292, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1294 = stablehlo.add %1291, %1293 : tensor<256x28x28x128xf32>
      %1295 = stablehlo.convert %1294 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1296 = func.call @relu_65(%1295) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1297 = stablehlo.convert %arg1155 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1298 = stablehlo.convolution(%1296, %1297) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1299 = stablehlo.broadcast_in_dim %arg1718, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1300 = stablehlo.broadcast_in_dim %arg1719, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1301 = stablehlo.convert %1298 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1302 = stablehlo.broadcast_in_dim %1299, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1303 = stablehlo.subtract %1301, %1302 : tensor<256x28x28x128xf32>
      %cst_67 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1304 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1305 = stablehlo.add %1300, %1304 : tensor<1x1x1x128xf32>
      %1306 = stablehlo.rsqrt %1305 : tensor<1x1x1x128xf32>
      %1307 = stablehlo.reshape %arg1151 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1308 = stablehlo.multiply %1306, %1307 : tensor<1x1x1x128xf32>
      %1309 = stablehlo.broadcast_in_dim %1308, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1310 = stablehlo.multiply %1303, %1309 : tensor<256x28x28x128xf32>
      %1311 = stablehlo.reshape %arg1150 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1312 = stablehlo.broadcast_in_dim %1311, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1313 = stablehlo.add %1310, %1312 : tensor<256x28x28x128xf32>
      %1314 = stablehlo.convert %1313 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1315 = func.call @relu_65(%1314) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1316 = stablehlo.convert %arg1156 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1317 = stablehlo.convolution(%1315, %1316) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1318 = stablehlo.broadcast_in_dim %arg1720, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1319 = stablehlo.broadcast_in_dim %arg1721, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1320 = stablehlo.convert %1317 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1321 = stablehlo.broadcast_in_dim %1318, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1322 = stablehlo.subtract %1320, %1321 : tensor<256x28x28x512xf32>
      %cst_68 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1323 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1324 = stablehlo.add %1319, %1323 : tensor<1x1x1x512xf32>
      %1325 = stablehlo.rsqrt %1324 : tensor<1x1x1x512xf32>
      %1326 = stablehlo.reshape %arg1153 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1327 = stablehlo.multiply %1325, %1326 : tensor<1x1x1x512xf32>
      %1328 = stablehlo.broadcast_in_dim %1327, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1329 = stablehlo.multiply %1322, %1328 : tensor<256x28x28x512xf32>
      %1330 = stablehlo.reshape %arg1152 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1331 = stablehlo.broadcast_in_dim %1330, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1332 = stablehlo.add %1329, %1331 : tensor<256x28x28x512xf32>
      %1333 = stablehlo.convert %1332 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1334 = stablehlo.add %1277, %1333 : tensor<256x28x28x512xf16>
      %1335 = func.call @relu_83(%1334) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1336 = stablehlo.convert %arg1163 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1337 = stablehlo.convolution(%1335, %1336) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1338 = stablehlo.broadcast_in_dim %arg1722, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1339 = stablehlo.broadcast_in_dim %arg1723, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1340 = stablehlo.convert %1337 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1341 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1342 = stablehlo.subtract %1340, %1341 : tensor<256x28x28x128xf32>
      %cst_69 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1343 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1344 = stablehlo.add %1339, %1343 : tensor<1x1x1x128xf32>
      %1345 = stablehlo.rsqrt %1344 : tensor<1x1x1x128xf32>
      %1346 = stablehlo.reshape %arg1158 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1347 = stablehlo.multiply %1345, %1346 : tensor<1x1x1x128xf32>
      %1348 = stablehlo.broadcast_in_dim %1347, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1349 = stablehlo.multiply %1342, %1348 : tensor<256x28x28x128xf32>
      %1350 = stablehlo.reshape %arg1157 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1351 = stablehlo.broadcast_in_dim %1350, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1352 = stablehlo.add %1349, %1351 : tensor<256x28x28x128xf32>
      %1353 = stablehlo.convert %1352 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1354 = func.call @relu_65(%1353) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1355 = stablehlo.convert %arg1164 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1356 = stablehlo.convolution(%1354, %1355) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1357 = stablehlo.broadcast_in_dim %arg1724, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1358 = stablehlo.broadcast_in_dim %arg1725, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1359 = stablehlo.convert %1356 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1360 = stablehlo.broadcast_in_dim %1357, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1361 = stablehlo.subtract %1359, %1360 : tensor<256x28x28x128xf32>
      %cst_70 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1362 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1363 = stablehlo.add %1358, %1362 : tensor<1x1x1x128xf32>
      %1364 = stablehlo.rsqrt %1363 : tensor<1x1x1x128xf32>
      %1365 = stablehlo.reshape %arg1160 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1366 = stablehlo.multiply %1364, %1365 : tensor<1x1x1x128xf32>
      %1367 = stablehlo.broadcast_in_dim %1366, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1368 = stablehlo.multiply %1361, %1367 : tensor<256x28x28x128xf32>
      %1369 = stablehlo.reshape %arg1159 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1370 = stablehlo.broadcast_in_dim %1369, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1371 = stablehlo.add %1368, %1370 : tensor<256x28x28x128xf32>
      %1372 = stablehlo.convert %1371 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1373 = func.call @relu_65(%1372) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1374 = stablehlo.convert %arg1165 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1375 = stablehlo.convolution(%1373, %1374) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1376 = stablehlo.broadcast_in_dim %arg1726, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1377 = stablehlo.broadcast_in_dim %arg1727, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1378 = stablehlo.convert %1375 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1379 = stablehlo.broadcast_in_dim %1376, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1380 = stablehlo.subtract %1378, %1379 : tensor<256x28x28x512xf32>
      %cst_71 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1381 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1382 = stablehlo.add %1377, %1381 : tensor<1x1x1x512xf32>
      %1383 = stablehlo.rsqrt %1382 : tensor<1x1x1x512xf32>
      %1384 = stablehlo.reshape %arg1162 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1385 = stablehlo.multiply %1383, %1384 : tensor<1x1x1x512xf32>
      %1386 = stablehlo.broadcast_in_dim %1385, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1387 = stablehlo.multiply %1380, %1386 : tensor<256x28x28x512xf32>
      %1388 = stablehlo.reshape %arg1161 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1389 = stablehlo.broadcast_in_dim %1388, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1390 = stablehlo.add %1387, %1389 : tensor<256x28x28x512xf32>
      %1391 = stablehlo.convert %1390 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1392 = stablehlo.add %1335, %1391 : tensor<256x28x28x512xf16>
      %1393 = func.call @relu_83(%1392) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1394 = stablehlo.convert %arg1172 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1395 = stablehlo.convolution(%1393, %1394) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1396 = stablehlo.broadcast_in_dim %arg1728, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1397 = stablehlo.broadcast_in_dim %arg1729, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1398 = stablehlo.convert %1395 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1399 = stablehlo.broadcast_in_dim %1396, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1400 = stablehlo.subtract %1398, %1399 : tensor<256x28x28x128xf32>
      %cst_72 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1401 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1402 = stablehlo.add %1397, %1401 : tensor<1x1x1x128xf32>
      %1403 = stablehlo.rsqrt %1402 : tensor<1x1x1x128xf32>
      %1404 = stablehlo.reshape %arg1167 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1405 = stablehlo.multiply %1403, %1404 : tensor<1x1x1x128xf32>
      %1406 = stablehlo.broadcast_in_dim %1405, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1407 = stablehlo.multiply %1400, %1406 : tensor<256x28x28x128xf32>
      %1408 = stablehlo.reshape %arg1166 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1409 = stablehlo.broadcast_in_dim %1408, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1410 = stablehlo.add %1407, %1409 : tensor<256x28x28x128xf32>
      %1411 = stablehlo.convert %1410 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1412 = func.call @relu_65(%1411) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1413 = stablehlo.convert %arg1173 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1414 = stablehlo.convolution(%1412, %1413) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1415 = stablehlo.broadcast_in_dim %arg1730, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1416 = stablehlo.broadcast_in_dim %arg1731, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1417 = stablehlo.convert %1414 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1418 = stablehlo.broadcast_in_dim %1415, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1419 = stablehlo.subtract %1417, %1418 : tensor<256x28x28x128xf32>
      %cst_73 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1420 = stablehlo.broadcast_in_dim %cst_73, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1421 = stablehlo.add %1416, %1420 : tensor<1x1x1x128xf32>
      %1422 = stablehlo.rsqrt %1421 : tensor<1x1x1x128xf32>
      %1423 = stablehlo.reshape %arg1169 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1424 = stablehlo.multiply %1422, %1423 : tensor<1x1x1x128xf32>
      %1425 = stablehlo.broadcast_in_dim %1424, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1426 = stablehlo.multiply %1419, %1425 : tensor<256x28x28x128xf32>
      %1427 = stablehlo.reshape %arg1168 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1428 = stablehlo.broadcast_in_dim %1427, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1429 = stablehlo.add %1426, %1428 : tensor<256x28x28x128xf32>
      %1430 = stablehlo.convert %1429 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1431 = func.call @relu_65(%1430) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1432 = stablehlo.convert %arg1174 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1433 = stablehlo.convolution(%1431, %1432) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1434 = stablehlo.broadcast_in_dim %arg1732, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1435 = stablehlo.broadcast_in_dim %arg1733, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1436 = stablehlo.convert %1433 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1437 = stablehlo.broadcast_in_dim %1434, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1438 = stablehlo.subtract %1436, %1437 : tensor<256x28x28x512xf32>
      %cst_74 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1439 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1440 = stablehlo.add %1435, %1439 : tensor<1x1x1x512xf32>
      %1441 = stablehlo.rsqrt %1440 : tensor<1x1x1x512xf32>
      %1442 = stablehlo.reshape %arg1171 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1443 = stablehlo.multiply %1441, %1442 : tensor<1x1x1x512xf32>
      %1444 = stablehlo.broadcast_in_dim %1443, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1445 = stablehlo.multiply %1438, %1444 : tensor<256x28x28x512xf32>
      %1446 = stablehlo.reshape %arg1170 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1447 = stablehlo.broadcast_in_dim %1446, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1448 = stablehlo.add %1445, %1447 : tensor<256x28x28x512xf32>
      %1449 = stablehlo.convert %1448 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1450 = stablehlo.add %1393, %1449 : tensor<256x28x28x512xf16>
      %1451 = func.call @relu_83(%1450) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1452 = stablehlo.convert %arg1181 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1453 = stablehlo.convolution(%1451, %1452) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1454 = stablehlo.broadcast_in_dim %arg1734, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1455 = stablehlo.broadcast_in_dim %arg1735, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1456 = stablehlo.convert %1453 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1457 = stablehlo.broadcast_in_dim %1454, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1458 = stablehlo.subtract %1456, %1457 : tensor<256x28x28x128xf32>
      %cst_75 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1459 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1460 = stablehlo.add %1455, %1459 : tensor<1x1x1x128xf32>
      %1461 = stablehlo.rsqrt %1460 : tensor<1x1x1x128xf32>
      %1462 = stablehlo.reshape %arg1176 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1463 = stablehlo.multiply %1461, %1462 : tensor<1x1x1x128xf32>
      %1464 = stablehlo.broadcast_in_dim %1463, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1465 = stablehlo.multiply %1458, %1464 : tensor<256x28x28x128xf32>
      %1466 = stablehlo.reshape %arg1175 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1467 = stablehlo.broadcast_in_dim %1466, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1468 = stablehlo.add %1465, %1467 : tensor<256x28x28x128xf32>
      %1469 = stablehlo.convert %1468 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1470 = func.call @relu_65(%1469) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1471 = stablehlo.convert %arg1182 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1472 = stablehlo.convolution(%1470, %1471) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1473 = stablehlo.broadcast_in_dim %arg1736, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1474 = stablehlo.broadcast_in_dim %arg1737, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1475 = stablehlo.convert %1472 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1476 = stablehlo.broadcast_in_dim %1473, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1477 = stablehlo.subtract %1475, %1476 : tensor<256x28x28x128xf32>
      %cst_76 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1478 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1479 = stablehlo.add %1474, %1478 : tensor<1x1x1x128xf32>
      %1480 = stablehlo.rsqrt %1479 : tensor<1x1x1x128xf32>
      %1481 = stablehlo.reshape %arg1178 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1482 = stablehlo.multiply %1480, %1481 : tensor<1x1x1x128xf32>
      %1483 = stablehlo.broadcast_in_dim %1482, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1484 = stablehlo.multiply %1477, %1483 : tensor<256x28x28x128xf32>
      %1485 = stablehlo.reshape %arg1177 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1486 = stablehlo.broadcast_in_dim %1485, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1487 = stablehlo.add %1484, %1486 : tensor<256x28x28x128xf32>
      %1488 = stablehlo.convert %1487 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1489 = func.call @relu_65(%1488) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1490 = stablehlo.convert %arg1183 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1491 = stablehlo.convolution(%1489, %1490) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1492 = stablehlo.broadcast_in_dim %arg1738, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1493 = stablehlo.broadcast_in_dim %arg1739, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1494 = stablehlo.convert %1491 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1495 = stablehlo.broadcast_in_dim %1492, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1496 = stablehlo.subtract %1494, %1495 : tensor<256x28x28x512xf32>
      %cst_77 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1497 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1498 = stablehlo.add %1493, %1497 : tensor<1x1x1x512xf32>
      %1499 = stablehlo.rsqrt %1498 : tensor<1x1x1x512xf32>
      %1500 = stablehlo.reshape %arg1180 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1501 = stablehlo.multiply %1499, %1500 : tensor<1x1x1x512xf32>
      %1502 = stablehlo.broadcast_in_dim %1501, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1503 = stablehlo.multiply %1496, %1502 : tensor<256x28x28x512xf32>
      %1504 = stablehlo.reshape %arg1179 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1505 = stablehlo.broadcast_in_dim %1504, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1506 = stablehlo.add %1503, %1505 : tensor<256x28x28x512xf32>
      %1507 = stablehlo.convert %1506 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1508 = stablehlo.add %1451, %1507 : tensor<256x28x28x512xf16>
      %1509 = func.call @relu_83(%1508) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1510 = stablehlo.convert %arg1190 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1511 = stablehlo.convolution(%1509, %1510) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1512 = stablehlo.broadcast_in_dim %arg1740, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1513 = stablehlo.broadcast_in_dim %arg1741, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1514 = stablehlo.convert %1511 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1515 = stablehlo.broadcast_in_dim %1512, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1516 = stablehlo.subtract %1514, %1515 : tensor<256x28x28x128xf32>
      %cst_78 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1517 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1518 = stablehlo.add %1513, %1517 : tensor<1x1x1x128xf32>
      %1519 = stablehlo.rsqrt %1518 : tensor<1x1x1x128xf32>
      %1520 = stablehlo.reshape %arg1185 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1521 = stablehlo.multiply %1519, %1520 : tensor<1x1x1x128xf32>
      %1522 = stablehlo.broadcast_in_dim %1521, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1523 = stablehlo.multiply %1516, %1522 : tensor<256x28x28x128xf32>
      %1524 = stablehlo.reshape %arg1184 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1525 = stablehlo.broadcast_in_dim %1524, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1526 = stablehlo.add %1523, %1525 : tensor<256x28x28x128xf32>
      %1527 = stablehlo.convert %1526 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1528 = func.call @relu_65(%1527) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1529 = stablehlo.convert %arg1191 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1530 = stablehlo.convolution(%1528, %1529) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1531 = stablehlo.broadcast_in_dim %arg1742, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1532 = stablehlo.broadcast_in_dim %arg1743, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1533 = stablehlo.convert %1530 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1534 = stablehlo.broadcast_in_dim %1531, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1535 = stablehlo.subtract %1533, %1534 : tensor<256x28x28x128xf32>
      %cst_79 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1536 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1537 = stablehlo.add %1532, %1536 : tensor<1x1x1x128xf32>
      %1538 = stablehlo.rsqrt %1537 : tensor<1x1x1x128xf32>
      %1539 = stablehlo.reshape %arg1187 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1540 = stablehlo.multiply %1538, %1539 : tensor<1x1x1x128xf32>
      %1541 = stablehlo.broadcast_in_dim %1540, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1542 = stablehlo.multiply %1535, %1541 : tensor<256x28x28x128xf32>
      %1543 = stablehlo.reshape %arg1186 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1544 = stablehlo.broadcast_in_dim %1543, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1545 = stablehlo.add %1542, %1544 : tensor<256x28x28x128xf32>
      %1546 = stablehlo.convert %1545 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1547 = func.call @relu_65(%1546) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1548 = stablehlo.convert %arg1192 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1549 = stablehlo.convolution(%1547, %1548) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1550 = stablehlo.broadcast_in_dim %arg1744, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1551 = stablehlo.broadcast_in_dim %arg1745, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1552 = stablehlo.convert %1549 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1553 = stablehlo.broadcast_in_dim %1550, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1554 = stablehlo.subtract %1552, %1553 : tensor<256x28x28x512xf32>
      %cst_80 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1555 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1556 = stablehlo.add %1551, %1555 : tensor<1x1x1x512xf32>
      %1557 = stablehlo.rsqrt %1556 : tensor<1x1x1x512xf32>
      %1558 = stablehlo.reshape %arg1189 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1559 = stablehlo.multiply %1557, %1558 : tensor<1x1x1x512xf32>
      %1560 = stablehlo.broadcast_in_dim %1559, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1561 = stablehlo.multiply %1554, %1560 : tensor<256x28x28x512xf32>
      %1562 = stablehlo.reshape %arg1188 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1563 = stablehlo.broadcast_in_dim %1562, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1564 = stablehlo.add %1561, %1563 : tensor<256x28x28x512xf32>
      %1565 = stablehlo.convert %1564 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1566 = stablehlo.add %1509, %1565 : tensor<256x28x28x512xf16>
      %1567 = func.call @relu_83(%1566) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1568 = stablehlo.convert %arg1199 : (tensor<1x1x512x128xf32>) -> tensor<1x1x512x128xf16>
      %1569 = stablehlo.convolution(%1567, %1568) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x128xf16>) -> tensor<256x28x28x128xf16>
      %1570 = stablehlo.broadcast_in_dim %arg1746, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1571 = stablehlo.broadcast_in_dim %arg1747, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1572 = stablehlo.convert %1569 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1573 = stablehlo.broadcast_in_dim %1570, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1574 = stablehlo.subtract %1572, %1573 : tensor<256x28x28x128xf32>
      %cst_81 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1575 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1576 = stablehlo.add %1571, %1575 : tensor<1x1x1x128xf32>
      %1577 = stablehlo.rsqrt %1576 : tensor<1x1x1x128xf32>
      %1578 = stablehlo.reshape %arg1194 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1579 = stablehlo.multiply %1577, %1578 : tensor<1x1x1x128xf32>
      %1580 = stablehlo.broadcast_in_dim %1579, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1581 = stablehlo.multiply %1574, %1580 : tensor<256x28x28x128xf32>
      %1582 = stablehlo.reshape %arg1193 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1583 = stablehlo.broadcast_in_dim %1582, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1584 = stablehlo.add %1581, %1583 : tensor<256x28x28x128xf32>
      %1585 = stablehlo.convert %1584 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1586 = func.call @relu_65(%1585) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1587 = stablehlo.convert %arg1200 : (tensor<3x3x128x128xf32>) -> tensor<3x3x128x128xf16>
      %1588 = stablehlo.convolution(%1586, %1587) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<3x3x128x128xf16>) -> tensor<256x28x28x128xf16>
      %1589 = stablehlo.broadcast_in_dim %arg1748, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1590 = stablehlo.broadcast_in_dim %arg1749, dims = [3] : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1591 = stablehlo.convert %1588 : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf32>
      %1592 = stablehlo.broadcast_in_dim %1589, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1593 = stablehlo.subtract %1591, %1592 : tensor<256x28x28x128xf32>
      %cst_82 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1594 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f32>) -> tensor<1x1x1x128xf32>
      %1595 = stablehlo.add %1590, %1594 : tensor<1x1x1x128xf32>
      %1596 = stablehlo.rsqrt %1595 : tensor<1x1x1x128xf32>
      %1597 = stablehlo.reshape %arg1196 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1598 = stablehlo.multiply %1596, %1597 : tensor<1x1x1x128xf32>
      %1599 = stablehlo.broadcast_in_dim %1598, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1600 = stablehlo.multiply %1593, %1599 : tensor<256x28x28x128xf32>
      %1601 = stablehlo.reshape %arg1195 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
      %1602 = stablehlo.broadcast_in_dim %1601, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<256x28x28x128xf32>
      %1603 = stablehlo.add %1600, %1602 : tensor<256x28x28x128xf32>
      %1604 = stablehlo.convert %1603 : (tensor<256x28x28x128xf32>) -> tensor<256x28x28x128xf16>
      %1605 = func.call @relu_65(%1604) : (tensor<256x28x28x128xf16>) -> tensor<256x28x28x128xf16>
      %1606 = stablehlo.convert %arg1201 : (tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf16>
      %1607 = stablehlo.convolution(%1605, %1606) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x128xf16>, tensor<1x1x128x512xf16>) -> tensor<256x28x28x512xf16>
      %1608 = stablehlo.broadcast_in_dim %arg1750, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1609 = stablehlo.broadcast_in_dim %arg1751, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1610 = stablehlo.convert %1607 : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf32>
      %1611 = stablehlo.broadcast_in_dim %1608, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1612 = stablehlo.subtract %1610, %1611 : tensor<256x28x28x512xf32>
      %cst_83 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1613 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %1614 = stablehlo.add %1609, %1613 : tensor<1x1x1x512xf32>
      %1615 = stablehlo.rsqrt %1614 : tensor<1x1x1x512xf32>
      %1616 = stablehlo.reshape %arg1198 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1617 = stablehlo.multiply %1615, %1616 : tensor<1x1x1x512xf32>
      %1618 = stablehlo.broadcast_in_dim %1617, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1619 = stablehlo.multiply %1612, %1618 : tensor<256x28x28x512xf32>
      %1620 = stablehlo.reshape %arg1197 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %1621 = stablehlo.broadcast_in_dim %1620, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x28x28x512xf32>
      %1622 = stablehlo.add %1619, %1621 : tensor<256x28x28x512xf32>
      %1623 = stablehlo.convert %1622 : (tensor<256x28x28x512xf32>) -> tensor<256x28x28x512xf16>
      %1624 = stablehlo.add %1567, %1623 : tensor<256x28x28x512xf16>
      %1625 = func.call @relu_83(%1624) : (tensor<256x28x28x512xf16>) -> tensor<256x28x28x512xf16>
      %1626 = stablehlo.convert %arg1208 : (tensor<1x1x512x256xf32>) -> tensor<1x1x512x256xf16>
      %1627 = stablehlo.convolution(%1625, %1626) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x256xf16>) -> tensor<256x28x28x256xf16>
      %1628 = stablehlo.broadcast_in_dim %arg1752, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1629 = stablehlo.broadcast_in_dim %arg1753, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1630 = stablehlo.convert %1627 : (tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf32>
      %1631 = stablehlo.broadcast_in_dim %1628, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
      %1632 = stablehlo.subtract %1630, %1631 : tensor<256x28x28x256xf32>
      %cst_84 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1633 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1634 = stablehlo.add %1629, %1633 : tensor<1x1x1x256xf32>
      %1635 = stablehlo.rsqrt %1634 : tensor<1x1x1x256xf32>
      %1636 = stablehlo.reshape %arg1203 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1637 = stablehlo.multiply %1635, %1636 : tensor<1x1x1x256xf32>
      %1638 = stablehlo.broadcast_in_dim %1637, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
      %1639 = stablehlo.multiply %1632, %1638 : tensor<256x28x28x256xf32>
      %1640 = stablehlo.reshape %arg1202 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1641 = stablehlo.broadcast_in_dim %1640, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x28x28x256xf32>
      %1642 = stablehlo.add %1639, %1641 : tensor<256x28x28x256xf32>
      %1643 = stablehlo.convert %1642 : (tensor<256x28x28x256xf32>) -> tensor<256x28x28x256xf16>
      %1644 = func.call @relu_165(%1643) : (tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf16>
      %1645 = stablehlo.convert %arg1209 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1646 = stablehlo.convolution(%1644, %1645) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1647 = stablehlo.broadcast_in_dim %arg1754, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1648 = stablehlo.broadcast_in_dim %arg1755, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1649 = stablehlo.convert %1646 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1650 = stablehlo.broadcast_in_dim %1647, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1651 = stablehlo.subtract %1649, %1650 : tensor<256x14x14x256xf32>
      %cst_85 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1652 = stablehlo.broadcast_in_dim %cst_85, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1653 = stablehlo.add %1648, %1652 : tensor<1x1x1x256xf32>
      %1654 = stablehlo.rsqrt %1653 : tensor<1x1x1x256xf32>
      %1655 = stablehlo.reshape %arg1205 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1656 = stablehlo.multiply %1654, %1655 : tensor<1x1x1x256xf32>
      %1657 = stablehlo.broadcast_in_dim %1656, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1658 = stablehlo.multiply %1651, %1657 : tensor<256x14x14x256xf32>
      %1659 = stablehlo.reshape %arg1204 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1660 = stablehlo.broadcast_in_dim %1659, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1661 = stablehlo.add %1658, %1660 : tensor<256x14x14x256xf32>
      %1662 = stablehlo.convert %1661 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1663 = func.call @relu_175(%1662) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1664 = stablehlo.convert %arg1210 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1665 = stablehlo.convolution(%1663, %1664) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1666 = stablehlo.broadcast_in_dim %arg1756, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1667 = stablehlo.broadcast_in_dim %arg1757, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1668 = stablehlo.convert %1665 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1669 = stablehlo.broadcast_in_dim %1666, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1670 = stablehlo.subtract %1668, %1669 : tensor<256x14x14x1024xf32>
      %cst_86 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1671 = stablehlo.broadcast_in_dim %cst_86, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1672 = stablehlo.add %1667, %1671 : tensor<1x1x1x1024xf32>
      %1673 = stablehlo.rsqrt %1672 : tensor<1x1x1x1024xf32>
      %1674 = stablehlo.reshape %arg1207 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1675 = stablehlo.multiply %1673, %1674 : tensor<1x1x1x1024xf32>
      %1676 = stablehlo.broadcast_in_dim %1675, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1677 = stablehlo.multiply %1670, %1676 : tensor<256x14x14x1024xf32>
      %1678 = stablehlo.reshape %arg1206 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1679 = stablehlo.broadcast_in_dim %1678, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1680 = stablehlo.add %1677, %1679 : tensor<256x14x14x1024xf32>
      %1681 = stablehlo.convert %1680 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1682 = stablehlo.convert %arg1211 : (tensor<1x1x512x1024xf32>) -> tensor<1x1x512x1024xf16>
      %1683 = stablehlo.convolution(%1625, %1682) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x28x28x512xf16>, tensor<1x1x512x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1684 = stablehlo.broadcast_in_dim %arg1758, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1685 = stablehlo.broadcast_in_dim %arg1759, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1686 = stablehlo.convert %1683 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1687 = stablehlo.broadcast_in_dim %1684, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1688 = stablehlo.subtract %1686, %1687 : tensor<256x14x14x1024xf32>
      %cst_87 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1689 = stablehlo.broadcast_in_dim %cst_87, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1690 = stablehlo.add %1685, %1689 : tensor<1x1x1x1024xf32>
      %1691 = stablehlo.rsqrt %1690 : tensor<1x1x1x1024xf32>
      %1692 = stablehlo.reshape %arg1213 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1693 = stablehlo.multiply %1691, %1692 : tensor<1x1x1x1024xf32>
      %1694 = stablehlo.broadcast_in_dim %1693, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1695 = stablehlo.multiply %1688, %1694 : tensor<256x14x14x1024xf32>
      %1696 = stablehlo.reshape %arg1212 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1697 = stablehlo.broadcast_in_dim %1696, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1698 = stablehlo.add %1695, %1697 : tensor<256x14x14x1024xf32>
      %1699 = stablehlo.convert %1698 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1700 = stablehlo.add %1699, %1681 : tensor<256x14x14x1024xf16>
      %1701 = func.call @relu_193(%1700) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1702 = stablehlo.convert %arg1220 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1703 = stablehlo.convolution(%1701, %1702) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1704 = stablehlo.broadcast_in_dim %arg1760, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1705 = stablehlo.broadcast_in_dim %arg1761, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1706 = stablehlo.convert %1703 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1707 = stablehlo.broadcast_in_dim %1704, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1708 = stablehlo.subtract %1706, %1707 : tensor<256x14x14x256xf32>
      %cst_88 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1709 = stablehlo.broadcast_in_dim %cst_88, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1710 = stablehlo.add %1705, %1709 : tensor<1x1x1x256xf32>
      %1711 = stablehlo.rsqrt %1710 : tensor<1x1x1x256xf32>
      %1712 = stablehlo.reshape %arg1215 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1713 = stablehlo.multiply %1711, %1712 : tensor<1x1x1x256xf32>
      %1714 = stablehlo.broadcast_in_dim %1713, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1715 = stablehlo.multiply %1708, %1714 : tensor<256x14x14x256xf32>
      %1716 = stablehlo.reshape %arg1214 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1717 = stablehlo.broadcast_in_dim %1716, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1718 = stablehlo.add %1715, %1717 : tensor<256x14x14x256xf32>
      %1719 = stablehlo.convert %1718 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1720 = func.call @relu_175(%1719) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1721 = stablehlo.convert %arg1221 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1722 = stablehlo.convolution(%1720, %1721) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1723 = stablehlo.broadcast_in_dim %arg1762, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1724 = stablehlo.broadcast_in_dim %arg1763, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1725 = stablehlo.convert %1722 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1726 = stablehlo.broadcast_in_dim %1723, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1727 = stablehlo.subtract %1725, %1726 : tensor<256x14x14x256xf32>
      %cst_89 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1728 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1729 = stablehlo.add %1724, %1728 : tensor<1x1x1x256xf32>
      %1730 = stablehlo.rsqrt %1729 : tensor<1x1x1x256xf32>
      %1731 = stablehlo.reshape %arg1217 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1732 = stablehlo.multiply %1730, %1731 : tensor<1x1x1x256xf32>
      %1733 = stablehlo.broadcast_in_dim %1732, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1734 = stablehlo.multiply %1727, %1733 : tensor<256x14x14x256xf32>
      %1735 = stablehlo.reshape %arg1216 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1736 = stablehlo.broadcast_in_dim %1735, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1737 = stablehlo.add %1734, %1736 : tensor<256x14x14x256xf32>
      %1738 = stablehlo.convert %1737 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1739 = func.call @relu_175(%1738) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1740 = stablehlo.convert %arg1222 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1741 = stablehlo.convolution(%1739, %1740) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1742 = stablehlo.broadcast_in_dim %arg1764, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1743 = stablehlo.broadcast_in_dim %arg1765, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1744 = stablehlo.convert %1741 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1745 = stablehlo.broadcast_in_dim %1742, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1746 = stablehlo.subtract %1744, %1745 : tensor<256x14x14x1024xf32>
      %cst_90 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1747 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1748 = stablehlo.add %1743, %1747 : tensor<1x1x1x1024xf32>
      %1749 = stablehlo.rsqrt %1748 : tensor<1x1x1x1024xf32>
      %1750 = stablehlo.reshape %arg1219 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1751 = stablehlo.multiply %1749, %1750 : tensor<1x1x1x1024xf32>
      %1752 = stablehlo.broadcast_in_dim %1751, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1753 = stablehlo.multiply %1746, %1752 : tensor<256x14x14x1024xf32>
      %1754 = stablehlo.reshape %arg1218 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1755 = stablehlo.broadcast_in_dim %1754, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1756 = stablehlo.add %1753, %1755 : tensor<256x14x14x1024xf32>
      %1757 = stablehlo.convert %1756 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1758 = stablehlo.add %1701, %1757 : tensor<256x14x14x1024xf16>
      %1759 = func.call @relu_193(%1758) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1760 = stablehlo.convert %arg1229 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1761 = stablehlo.convolution(%1759, %1760) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1762 = stablehlo.broadcast_in_dim %arg1766, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1763 = stablehlo.broadcast_in_dim %arg1767, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1764 = stablehlo.convert %1761 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1765 = stablehlo.broadcast_in_dim %1762, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1766 = stablehlo.subtract %1764, %1765 : tensor<256x14x14x256xf32>
      %cst_91 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1767 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1768 = stablehlo.add %1763, %1767 : tensor<1x1x1x256xf32>
      %1769 = stablehlo.rsqrt %1768 : tensor<1x1x1x256xf32>
      %1770 = stablehlo.reshape %arg1224 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1771 = stablehlo.multiply %1769, %1770 : tensor<1x1x1x256xf32>
      %1772 = stablehlo.broadcast_in_dim %1771, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1773 = stablehlo.multiply %1766, %1772 : tensor<256x14x14x256xf32>
      %1774 = stablehlo.reshape %arg1223 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1775 = stablehlo.broadcast_in_dim %1774, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1776 = stablehlo.add %1773, %1775 : tensor<256x14x14x256xf32>
      %1777 = stablehlo.convert %1776 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1778 = func.call @relu_175(%1777) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1779 = stablehlo.convert %arg1230 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1780 = stablehlo.convolution(%1778, %1779) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1781 = stablehlo.broadcast_in_dim %arg1768, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1782 = stablehlo.broadcast_in_dim %arg1769, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1783 = stablehlo.convert %1780 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1784 = stablehlo.broadcast_in_dim %1781, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1785 = stablehlo.subtract %1783, %1784 : tensor<256x14x14x256xf32>
      %cst_92 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1786 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1787 = stablehlo.add %1782, %1786 : tensor<1x1x1x256xf32>
      %1788 = stablehlo.rsqrt %1787 : tensor<1x1x1x256xf32>
      %1789 = stablehlo.reshape %arg1226 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1790 = stablehlo.multiply %1788, %1789 : tensor<1x1x1x256xf32>
      %1791 = stablehlo.broadcast_in_dim %1790, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1792 = stablehlo.multiply %1785, %1791 : tensor<256x14x14x256xf32>
      %1793 = stablehlo.reshape %arg1225 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1794 = stablehlo.broadcast_in_dim %1793, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1795 = stablehlo.add %1792, %1794 : tensor<256x14x14x256xf32>
      %1796 = stablehlo.convert %1795 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1797 = func.call @relu_175(%1796) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1798 = stablehlo.convert %arg1231 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1799 = stablehlo.convolution(%1797, %1798) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1800 = stablehlo.broadcast_in_dim %arg1770, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1801 = stablehlo.broadcast_in_dim %arg1771, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1802 = stablehlo.convert %1799 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1803 = stablehlo.broadcast_in_dim %1800, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1804 = stablehlo.subtract %1802, %1803 : tensor<256x14x14x1024xf32>
      %cst_93 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1805 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1806 = stablehlo.add %1801, %1805 : tensor<1x1x1x1024xf32>
      %1807 = stablehlo.rsqrt %1806 : tensor<1x1x1x1024xf32>
      %1808 = stablehlo.reshape %arg1228 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1809 = stablehlo.multiply %1807, %1808 : tensor<1x1x1x1024xf32>
      %1810 = stablehlo.broadcast_in_dim %1809, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1811 = stablehlo.multiply %1804, %1810 : tensor<256x14x14x1024xf32>
      %1812 = stablehlo.reshape %arg1227 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1813 = stablehlo.broadcast_in_dim %1812, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1814 = stablehlo.add %1811, %1813 : tensor<256x14x14x1024xf32>
      %1815 = stablehlo.convert %1814 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1816 = stablehlo.add %1759, %1815 : tensor<256x14x14x1024xf16>
      %1817 = func.call @relu_193(%1816) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1818 = stablehlo.convert %arg1250 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1819 = stablehlo.convolution(%1817, %1818) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1820 = stablehlo.broadcast_in_dim %arg1780, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1821 = stablehlo.broadcast_in_dim %arg1781, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1822 = stablehlo.convert %1819 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1823 = stablehlo.broadcast_in_dim %1820, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1824 = stablehlo.subtract %1822, %1823 : tensor<256x14x14x256xf32>
      %cst_94 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1825 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1826 = stablehlo.add %1821, %1825 : tensor<1x1x1x256xf32>
      %1827 = stablehlo.rsqrt %1826 : tensor<1x1x1x256xf32>
      %1828 = stablehlo.reshape %arg1245 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1829 = stablehlo.multiply %1827, %1828 : tensor<1x1x1x256xf32>
      %1830 = stablehlo.broadcast_in_dim %1829, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1831 = stablehlo.multiply %1824, %1830 : tensor<256x14x14x256xf32>
      %1832 = stablehlo.reshape %arg1244 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1833 = stablehlo.broadcast_in_dim %1832, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1834 = stablehlo.add %1831, %1833 : tensor<256x14x14x256xf32>
      %1835 = stablehlo.convert %1834 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1836 = func.call @relu_175(%1835) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1837 = stablehlo.convert %arg1251 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1838 = stablehlo.convolution(%1836, %1837) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1839 = stablehlo.broadcast_in_dim %arg1782, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1840 = stablehlo.broadcast_in_dim %arg1783, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1841 = stablehlo.convert %1838 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1842 = stablehlo.broadcast_in_dim %1839, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1843 = stablehlo.subtract %1841, %1842 : tensor<256x14x14x256xf32>
      %cst_95 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1844 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1845 = stablehlo.add %1840, %1844 : tensor<1x1x1x256xf32>
      %1846 = stablehlo.rsqrt %1845 : tensor<1x1x1x256xf32>
      %1847 = stablehlo.reshape %arg1247 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1848 = stablehlo.multiply %1846, %1847 : tensor<1x1x1x256xf32>
      %1849 = stablehlo.broadcast_in_dim %1848, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1850 = stablehlo.multiply %1843, %1849 : tensor<256x14x14x256xf32>
      %1851 = stablehlo.reshape %arg1246 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1852 = stablehlo.broadcast_in_dim %1851, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1853 = stablehlo.add %1850, %1852 : tensor<256x14x14x256xf32>
      %1854 = stablehlo.convert %1853 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1855 = func.call @relu_175(%1854) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1856 = stablehlo.convert %arg1252 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1857 = stablehlo.convolution(%1855, %1856) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1858 = stablehlo.broadcast_in_dim %arg1784, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1859 = stablehlo.broadcast_in_dim %arg1785, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1860 = stablehlo.convert %1857 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1861 = stablehlo.broadcast_in_dim %1858, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1862 = stablehlo.subtract %1860, %1861 : tensor<256x14x14x1024xf32>
      %cst_96 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1863 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1864 = stablehlo.add %1859, %1863 : tensor<1x1x1x1024xf32>
      %1865 = stablehlo.rsqrt %1864 : tensor<1x1x1x1024xf32>
      %1866 = stablehlo.reshape %arg1249 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1867 = stablehlo.multiply %1865, %1866 : tensor<1x1x1x1024xf32>
      %1868 = stablehlo.broadcast_in_dim %1867, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1869 = stablehlo.multiply %1862, %1868 : tensor<256x14x14x1024xf32>
      %1870 = stablehlo.reshape %arg1248 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1871 = stablehlo.broadcast_in_dim %1870, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1872 = stablehlo.add %1869, %1871 : tensor<256x14x14x1024xf32>
      %1873 = stablehlo.convert %1872 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1874 = stablehlo.add %1817, %1873 : tensor<256x14x14x1024xf16>
      %1875 = func.call @relu_193(%1874) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1876 = stablehlo.convert %arg1259 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1877 = stablehlo.convolution(%1875, %1876) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1878 = stablehlo.broadcast_in_dim %arg1786, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1879 = stablehlo.broadcast_in_dim %arg1787, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1880 = stablehlo.convert %1877 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1881 = stablehlo.broadcast_in_dim %1878, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1882 = stablehlo.subtract %1880, %1881 : tensor<256x14x14x256xf32>
      %cst_97 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1883 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1884 = stablehlo.add %1879, %1883 : tensor<1x1x1x256xf32>
      %1885 = stablehlo.rsqrt %1884 : tensor<1x1x1x256xf32>
      %1886 = stablehlo.reshape %arg1254 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1887 = stablehlo.multiply %1885, %1886 : tensor<1x1x1x256xf32>
      %1888 = stablehlo.broadcast_in_dim %1887, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1889 = stablehlo.multiply %1882, %1888 : tensor<256x14x14x256xf32>
      %1890 = stablehlo.reshape %arg1253 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1891 = stablehlo.broadcast_in_dim %1890, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1892 = stablehlo.add %1889, %1891 : tensor<256x14x14x256xf32>
      %1893 = stablehlo.convert %1892 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1894 = func.call @relu_175(%1893) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1895 = stablehlo.convert %arg1260 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1896 = stablehlo.convolution(%1894, %1895) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1897 = stablehlo.broadcast_in_dim %arg1788, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1898 = stablehlo.broadcast_in_dim %arg1789, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1899 = stablehlo.convert %1896 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1900 = stablehlo.broadcast_in_dim %1897, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1901 = stablehlo.subtract %1899, %1900 : tensor<256x14x14x256xf32>
      %cst_98 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1902 = stablehlo.broadcast_in_dim %cst_98, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1903 = stablehlo.add %1898, %1902 : tensor<1x1x1x256xf32>
      %1904 = stablehlo.rsqrt %1903 : tensor<1x1x1x256xf32>
      %1905 = stablehlo.reshape %arg1256 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1906 = stablehlo.multiply %1904, %1905 : tensor<1x1x1x256xf32>
      %1907 = stablehlo.broadcast_in_dim %1906, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1908 = stablehlo.multiply %1901, %1907 : tensor<256x14x14x256xf32>
      %1909 = stablehlo.reshape %arg1255 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1910 = stablehlo.broadcast_in_dim %1909, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1911 = stablehlo.add %1908, %1910 : tensor<256x14x14x256xf32>
      %1912 = stablehlo.convert %1911 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1913 = func.call @relu_175(%1912) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1914 = stablehlo.convert %arg1261 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1915 = stablehlo.convolution(%1913, %1914) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1916 = stablehlo.broadcast_in_dim %arg1790, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1917 = stablehlo.broadcast_in_dim %arg1791, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1918 = stablehlo.convert %1915 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1919 = stablehlo.broadcast_in_dim %1916, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1920 = stablehlo.subtract %1918, %1919 : tensor<256x14x14x1024xf32>
      %cst_99 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1921 = stablehlo.broadcast_in_dim %cst_99, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1922 = stablehlo.add %1917, %1921 : tensor<1x1x1x1024xf32>
      %1923 = stablehlo.rsqrt %1922 : tensor<1x1x1x1024xf32>
      %1924 = stablehlo.reshape %arg1258 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1925 = stablehlo.multiply %1923, %1924 : tensor<1x1x1x1024xf32>
      %1926 = stablehlo.broadcast_in_dim %1925, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1927 = stablehlo.multiply %1920, %1926 : tensor<256x14x14x1024xf32>
      %1928 = stablehlo.reshape %arg1257 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1929 = stablehlo.broadcast_in_dim %1928, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1930 = stablehlo.add %1927, %1929 : tensor<256x14x14x1024xf32>
      %1931 = stablehlo.convert %1930 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1932 = stablehlo.add %1875, %1931 : tensor<256x14x14x1024xf16>
      %1933 = func.call @relu_193(%1932) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1934 = stablehlo.convert %arg1268 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1935 = stablehlo.convolution(%1933, %1934) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1936 = stablehlo.broadcast_in_dim %arg1792, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1937 = stablehlo.broadcast_in_dim %arg1793, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1938 = stablehlo.convert %1935 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1939 = stablehlo.broadcast_in_dim %1936, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1940 = stablehlo.subtract %1938, %1939 : tensor<256x14x14x256xf32>
      %cst_100 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1941 = stablehlo.broadcast_in_dim %cst_100, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1942 = stablehlo.add %1937, %1941 : tensor<1x1x1x256xf32>
      %1943 = stablehlo.rsqrt %1942 : tensor<1x1x1x256xf32>
      %1944 = stablehlo.reshape %arg1263 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1945 = stablehlo.multiply %1943, %1944 : tensor<1x1x1x256xf32>
      %1946 = stablehlo.broadcast_in_dim %1945, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1947 = stablehlo.multiply %1940, %1946 : tensor<256x14x14x256xf32>
      %1948 = stablehlo.reshape %arg1262 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1949 = stablehlo.broadcast_in_dim %1948, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1950 = stablehlo.add %1947, %1949 : tensor<256x14x14x256xf32>
      %1951 = stablehlo.convert %1950 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1952 = func.call @relu_175(%1951) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1953 = stablehlo.convert %arg1269 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %1954 = stablehlo.convolution(%1952, %1953) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %1955 = stablehlo.broadcast_in_dim %arg1794, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1956 = stablehlo.broadcast_in_dim %arg1795, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1957 = stablehlo.convert %1954 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1958 = stablehlo.broadcast_in_dim %1955, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1959 = stablehlo.subtract %1957, %1958 : tensor<256x14x14x256xf32>
      %cst_101 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1960 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %1961 = stablehlo.add %1956, %1960 : tensor<1x1x1x256xf32>
      %1962 = stablehlo.rsqrt %1961 : tensor<1x1x1x256xf32>
      %1963 = stablehlo.reshape %arg1265 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1964 = stablehlo.multiply %1962, %1963 : tensor<1x1x1x256xf32>
      %1965 = stablehlo.broadcast_in_dim %1964, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1966 = stablehlo.multiply %1959, %1965 : tensor<256x14x14x256xf32>
      %1967 = stablehlo.reshape %arg1264 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1968 = stablehlo.broadcast_in_dim %1967, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1969 = stablehlo.add %1966, %1968 : tensor<256x14x14x256xf32>
      %1970 = stablehlo.convert %1969 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %1971 = func.call @relu_175(%1970) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %1972 = stablehlo.convert %arg1270 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %1973 = stablehlo.convolution(%1971, %1972) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1974 = stablehlo.broadcast_in_dim %arg1796, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1975 = stablehlo.broadcast_in_dim %arg1797, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1976 = stablehlo.convert %1973 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %1977 = stablehlo.broadcast_in_dim %1974, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1978 = stablehlo.subtract %1976, %1977 : tensor<256x14x14x1024xf32>
      %cst_102 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1979 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %1980 = stablehlo.add %1975, %1979 : tensor<1x1x1x1024xf32>
      %1981 = stablehlo.rsqrt %1980 : tensor<1x1x1x1024xf32>
      %1982 = stablehlo.reshape %arg1267 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1983 = stablehlo.multiply %1981, %1982 : tensor<1x1x1x1024xf32>
      %1984 = stablehlo.broadcast_in_dim %1983, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1985 = stablehlo.multiply %1978, %1984 : tensor<256x14x14x1024xf32>
      %1986 = stablehlo.reshape %arg1266 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %1987 = stablehlo.broadcast_in_dim %1986, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %1988 = stablehlo.add %1985, %1987 : tensor<256x14x14x1024xf32>
      %1989 = stablehlo.convert %1988 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %1990 = stablehlo.add %1933, %1989 : tensor<256x14x14x1024xf16>
      %1991 = func.call @relu_193(%1990) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %1992 = stablehlo.convert %arg1277 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %1993 = stablehlo.convolution(%1991, %1992) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %1994 = stablehlo.broadcast_in_dim %arg1798, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1995 = stablehlo.broadcast_in_dim %arg1799, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %1996 = stablehlo.convert %1993 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %1997 = stablehlo.broadcast_in_dim %1994, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %1998 = stablehlo.subtract %1996, %1997 : tensor<256x14x14x256xf32>
      %cst_103 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %1999 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2000 = stablehlo.add %1995, %1999 : tensor<1x1x1x256xf32>
      %2001 = stablehlo.rsqrt %2000 : tensor<1x1x1x256xf32>
      %2002 = stablehlo.reshape %arg1272 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2003 = stablehlo.multiply %2001, %2002 : tensor<1x1x1x256xf32>
      %2004 = stablehlo.broadcast_in_dim %2003, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2005 = stablehlo.multiply %1998, %2004 : tensor<256x14x14x256xf32>
      %2006 = stablehlo.reshape %arg1271 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2007 = stablehlo.broadcast_in_dim %2006, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2008 = stablehlo.add %2005, %2007 : tensor<256x14x14x256xf32>
      %2009 = stablehlo.convert %2008 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2010 = func.call @relu_175(%2009) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2011 = stablehlo.convert %arg1278 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2012 = stablehlo.convolution(%2010, %2011) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2013 = stablehlo.broadcast_in_dim %arg1800, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2014 = stablehlo.broadcast_in_dim %arg1801, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2015 = stablehlo.convert %2012 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2016 = stablehlo.broadcast_in_dim %2013, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2017 = stablehlo.subtract %2015, %2016 : tensor<256x14x14x256xf32>
      %cst_104 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2018 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2019 = stablehlo.add %2014, %2018 : tensor<1x1x1x256xf32>
      %2020 = stablehlo.rsqrt %2019 : tensor<1x1x1x256xf32>
      %2021 = stablehlo.reshape %arg1274 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2022 = stablehlo.multiply %2020, %2021 : tensor<1x1x1x256xf32>
      %2023 = stablehlo.broadcast_in_dim %2022, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2024 = stablehlo.multiply %2017, %2023 : tensor<256x14x14x256xf32>
      %2025 = stablehlo.reshape %arg1273 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2026 = stablehlo.broadcast_in_dim %2025, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2027 = stablehlo.add %2024, %2026 : tensor<256x14x14x256xf32>
      %2028 = stablehlo.convert %2027 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2029 = func.call @relu_175(%2028) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2030 = stablehlo.convert %arg1279 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2031 = stablehlo.convolution(%2029, %2030) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2032 = stablehlo.broadcast_in_dim %arg1802, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2033 = stablehlo.broadcast_in_dim %arg1803, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2034 = stablehlo.convert %2031 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2035 = stablehlo.broadcast_in_dim %2032, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2036 = stablehlo.subtract %2034, %2035 : tensor<256x14x14x1024xf32>
      %cst_105 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2037 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2038 = stablehlo.add %2033, %2037 : tensor<1x1x1x1024xf32>
      %2039 = stablehlo.rsqrt %2038 : tensor<1x1x1x1024xf32>
      %2040 = stablehlo.reshape %arg1276 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2041 = stablehlo.multiply %2039, %2040 : tensor<1x1x1x1024xf32>
      %2042 = stablehlo.broadcast_in_dim %2041, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2043 = stablehlo.multiply %2036, %2042 : tensor<256x14x14x1024xf32>
      %2044 = stablehlo.reshape %arg1275 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2045 = stablehlo.broadcast_in_dim %2044, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2046 = stablehlo.add %2043, %2045 : tensor<256x14x14x1024xf32>
      %2047 = stablehlo.convert %2046 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2048 = stablehlo.add %1991, %2047 : tensor<256x14x14x1024xf16>
      %2049 = func.call @relu_193(%2048) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2050 = stablehlo.convert %arg1286 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2051 = stablehlo.convolution(%2049, %2050) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2052 = stablehlo.broadcast_in_dim %arg1804, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2053 = stablehlo.broadcast_in_dim %arg1805, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2054 = stablehlo.convert %2051 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2055 = stablehlo.broadcast_in_dim %2052, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2056 = stablehlo.subtract %2054, %2055 : tensor<256x14x14x256xf32>
      %cst_106 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2057 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2058 = stablehlo.add %2053, %2057 : tensor<1x1x1x256xf32>
      %2059 = stablehlo.rsqrt %2058 : tensor<1x1x1x256xf32>
      %2060 = stablehlo.reshape %arg1281 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2061 = stablehlo.multiply %2059, %2060 : tensor<1x1x1x256xf32>
      %2062 = stablehlo.broadcast_in_dim %2061, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2063 = stablehlo.multiply %2056, %2062 : tensor<256x14x14x256xf32>
      %2064 = stablehlo.reshape %arg1280 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2065 = stablehlo.broadcast_in_dim %2064, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2066 = stablehlo.add %2063, %2065 : tensor<256x14x14x256xf32>
      %2067 = stablehlo.convert %2066 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2068 = func.call @relu_175(%2067) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2069 = stablehlo.convert %arg1287 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2070 = stablehlo.convolution(%2068, %2069) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2071 = stablehlo.broadcast_in_dim %arg1806, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2072 = stablehlo.broadcast_in_dim %arg1807, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2073 = stablehlo.convert %2070 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2074 = stablehlo.broadcast_in_dim %2071, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2075 = stablehlo.subtract %2073, %2074 : tensor<256x14x14x256xf32>
      %cst_107 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2076 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2077 = stablehlo.add %2072, %2076 : tensor<1x1x1x256xf32>
      %2078 = stablehlo.rsqrt %2077 : tensor<1x1x1x256xf32>
      %2079 = stablehlo.reshape %arg1283 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2080 = stablehlo.multiply %2078, %2079 : tensor<1x1x1x256xf32>
      %2081 = stablehlo.broadcast_in_dim %2080, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2082 = stablehlo.multiply %2075, %2081 : tensor<256x14x14x256xf32>
      %2083 = stablehlo.reshape %arg1282 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2084 = stablehlo.broadcast_in_dim %2083, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2085 = stablehlo.add %2082, %2084 : tensor<256x14x14x256xf32>
      %2086 = stablehlo.convert %2085 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2087 = func.call @relu_175(%2086) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2088 = stablehlo.convert %arg1288 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2089 = stablehlo.convolution(%2087, %2088) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2090 = stablehlo.broadcast_in_dim %arg1808, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2091 = stablehlo.broadcast_in_dim %arg1809, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2092 = stablehlo.convert %2089 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2093 = stablehlo.broadcast_in_dim %2090, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2094 = stablehlo.subtract %2092, %2093 : tensor<256x14x14x1024xf32>
      %cst_108 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2095 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2096 = stablehlo.add %2091, %2095 : tensor<1x1x1x1024xf32>
      %2097 = stablehlo.rsqrt %2096 : tensor<1x1x1x1024xf32>
      %2098 = stablehlo.reshape %arg1285 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2099 = stablehlo.multiply %2097, %2098 : tensor<1x1x1x1024xf32>
      %2100 = stablehlo.broadcast_in_dim %2099, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2101 = stablehlo.multiply %2094, %2100 : tensor<256x14x14x1024xf32>
      %2102 = stablehlo.reshape %arg1284 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2103 = stablehlo.broadcast_in_dim %2102, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2104 = stablehlo.add %2101, %2103 : tensor<256x14x14x1024xf32>
      %2105 = stablehlo.convert %2104 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2106 = stablehlo.add %2049, %2105 : tensor<256x14x14x1024xf16>
      %2107 = func.call @relu_193(%2106) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2108 = stablehlo.convert %arg1295 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2109 = stablehlo.convolution(%2107, %2108) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2110 = stablehlo.broadcast_in_dim %arg1810, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2111 = stablehlo.broadcast_in_dim %arg1811, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2112 = stablehlo.convert %2109 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2113 = stablehlo.broadcast_in_dim %2110, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2114 = stablehlo.subtract %2112, %2113 : tensor<256x14x14x256xf32>
      %cst_109 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2115 = stablehlo.broadcast_in_dim %cst_109, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2116 = stablehlo.add %2111, %2115 : tensor<1x1x1x256xf32>
      %2117 = stablehlo.rsqrt %2116 : tensor<1x1x1x256xf32>
      %2118 = stablehlo.reshape %arg1290 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2119 = stablehlo.multiply %2117, %2118 : tensor<1x1x1x256xf32>
      %2120 = stablehlo.broadcast_in_dim %2119, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2121 = stablehlo.multiply %2114, %2120 : tensor<256x14x14x256xf32>
      %2122 = stablehlo.reshape %arg1289 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2123 = stablehlo.broadcast_in_dim %2122, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2124 = stablehlo.add %2121, %2123 : tensor<256x14x14x256xf32>
      %2125 = stablehlo.convert %2124 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2126 = func.call @relu_175(%2125) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2127 = stablehlo.convert %arg1296 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2128 = stablehlo.convolution(%2126, %2127) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2129 = stablehlo.broadcast_in_dim %arg1812, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2130 = stablehlo.broadcast_in_dim %arg1813, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2131 = stablehlo.convert %2128 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2132 = stablehlo.broadcast_in_dim %2129, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2133 = stablehlo.subtract %2131, %2132 : tensor<256x14x14x256xf32>
      %cst_110 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2134 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2135 = stablehlo.add %2130, %2134 : tensor<1x1x1x256xf32>
      %2136 = stablehlo.rsqrt %2135 : tensor<1x1x1x256xf32>
      %2137 = stablehlo.reshape %arg1292 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2138 = stablehlo.multiply %2136, %2137 : tensor<1x1x1x256xf32>
      %2139 = stablehlo.broadcast_in_dim %2138, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2140 = stablehlo.multiply %2133, %2139 : tensor<256x14x14x256xf32>
      %2141 = stablehlo.reshape %arg1291 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2142 = stablehlo.broadcast_in_dim %2141, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2143 = stablehlo.add %2140, %2142 : tensor<256x14x14x256xf32>
      %2144 = stablehlo.convert %2143 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2145 = func.call @relu_175(%2144) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2146 = stablehlo.convert %arg1297 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2147 = stablehlo.convolution(%2145, %2146) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2148 = stablehlo.broadcast_in_dim %arg1814, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2149 = stablehlo.broadcast_in_dim %arg1815, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2150 = stablehlo.convert %2147 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2151 = stablehlo.broadcast_in_dim %2148, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2152 = stablehlo.subtract %2150, %2151 : tensor<256x14x14x1024xf32>
      %cst_111 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2153 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2154 = stablehlo.add %2149, %2153 : tensor<1x1x1x1024xf32>
      %2155 = stablehlo.rsqrt %2154 : tensor<1x1x1x1024xf32>
      %2156 = stablehlo.reshape %arg1294 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2157 = stablehlo.multiply %2155, %2156 : tensor<1x1x1x1024xf32>
      %2158 = stablehlo.broadcast_in_dim %2157, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2159 = stablehlo.multiply %2152, %2158 : tensor<256x14x14x1024xf32>
      %2160 = stablehlo.reshape %arg1293 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2161 = stablehlo.broadcast_in_dim %2160, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2162 = stablehlo.add %2159, %2161 : tensor<256x14x14x1024xf32>
      %2163 = stablehlo.convert %2162 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2164 = stablehlo.add %2107, %2163 : tensor<256x14x14x1024xf16>
      %2165 = func.call @relu_193(%2164) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2166 = stablehlo.convert %arg1304 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2167 = stablehlo.convolution(%2165, %2166) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2168 = stablehlo.broadcast_in_dim %arg1816, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2169 = stablehlo.broadcast_in_dim %arg1817, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2170 = stablehlo.convert %2167 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2171 = stablehlo.broadcast_in_dim %2168, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2172 = stablehlo.subtract %2170, %2171 : tensor<256x14x14x256xf32>
      %cst_112 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2173 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2174 = stablehlo.add %2169, %2173 : tensor<1x1x1x256xf32>
      %2175 = stablehlo.rsqrt %2174 : tensor<1x1x1x256xf32>
      %2176 = stablehlo.reshape %arg1299 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2177 = stablehlo.multiply %2175, %2176 : tensor<1x1x1x256xf32>
      %2178 = stablehlo.broadcast_in_dim %2177, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2179 = stablehlo.multiply %2172, %2178 : tensor<256x14x14x256xf32>
      %2180 = stablehlo.reshape %arg1298 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2181 = stablehlo.broadcast_in_dim %2180, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2182 = stablehlo.add %2179, %2181 : tensor<256x14x14x256xf32>
      %2183 = stablehlo.convert %2182 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2184 = func.call @relu_175(%2183) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2185 = stablehlo.convert %arg1305 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2186 = stablehlo.convolution(%2184, %2185) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2187 = stablehlo.broadcast_in_dim %arg1818, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2188 = stablehlo.broadcast_in_dim %arg1819, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2189 = stablehlo.convert %2186 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2190 = stablehlo.broadcast_in_dim %2187, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2191 = stablehlo.subtract %2189, %2190 : tensor<256x14x14x256xf32>
      %cst_113 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2192 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2193 = stablehlo.add %2188, %2192 : tensor<1x1x1x256xf32>
      %2194 = stablehlo.rsqrt %2193 : tensor<1x1x1x256xf32>
      %2195 = stablehlo.reshape %arg1301 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2196 = stablehlo.multiply %2194, %2195 : tensor<1x1x1x256xf32>
      %2197 = stablehlo.broadcast_in_dim %2196, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2198 = stablehlo.multiply %2191, %2197 : tensor<256x14x14x256xf32>
      %2199 = stablehlo.reshape %arg1300 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2200 = stablehlo.broadcast_in_dim %2199, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2201 = stablehlo.add %2198, %2200 : tensor<256x14x14x256xf32>
      %2202 = stablehlo.convert %2201 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2203 = func.call @relu_175(%2202) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2204 = stablehlo.convert %arg1306 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2205 = stablehlo.convolution(%2203, %2204) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2206 = stablehlo.broadcast_in_dim %arg1820, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2207 = stablehlo.broadcast_in_dim %arg1821, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2208 = stablehlo.convert %2205 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2209 = stablehlo.broadcast_in_dim %2206, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2210 = stablehlo.subtract %2208, %2209 : tensor<256x14x14x1024xf32>
      %cst_114 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2211 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2212 = stablehlo.add %2207, %2211 : tensor<1x1x1x1024xf32>
      %2213 = stablehlo.rsqrt %2212 : tensor<1x1x1x1024xf32>
      %2214 = stablehlo.reshape %arg1303 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2215 = stablehlo.multiply %2213, %2214 : tensor<1x1x1x1024xf32>
      %2216 = stablehlo.broadcast_in_dim %2215, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2217 = stablehlo.multiply %2210, %2216 : tensor<256x14x14x1024xf32>
      %2218 = stablehlo.reshape %arg1302 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2219 = stablehlo.broadcast_in_dim %2218, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2220 = stablehlo.add %2217, %2219 : tensor<256x14x14x1024xf32>
      %2221 = stablehlo.convert %2220 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2222 = stablehlo.add %2165, %2221 : tensor<256x14x14x1024xf16>
      %2223 = func.call @relu_193(%2222) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2224 = stablehlo.convert %arg1313 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2225 = stablehlo.convolution(%2223, %2224) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2226 = stablehlo.broadcast_in_dim %arg1822, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2227 = stablehlo.broadcast_in_dim %arg1823, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2228 = stablehlo.convert %2225 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2229 = stablehlo.broadcast_in_dim %2226, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2230 = stablehlo.subtract %2228, %2229 : tensor<256x14x14x256xf32>
      %cst_115 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2231 = stablehlo.broadcast_in_dim %cst_115, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2232 = stablehlo.add %2227, %2231 : tensor<1x1x1x256xf32>
      %2233 = stablehlo.rsqrt %2232 : tensor<1x1x1x256xf32>
      %2234 = stablehlo.reshape %arg1308 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2235 = stablehlo.multiply %2233, %2234 : tensor<1x1x1x256xf32>
      %2236 = stablehlo.broadcast_in_dim %2235, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2237 = stablehlo.multiply %2230, %2236 : tensor<256x14x14x256xf32>
      %2238 = stablehlo.reshape %arg1307 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2239 = stablehlo.broadcast_in_dim %2238, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2240 = stablehlo.add %2237, %2239 : tensor<256x14x14x256xf32>
      %2241 = stablehlo.convert %2240 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2242 = func.call @relu_175(%2241) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2243 = stablehlo.convert %arg1314 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2244 = stablehlo.convolution(%2242, %2243) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2245 = stablehlo.broadcast_in_dim %arg1824, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2246 = stablehlo.broadcast_in_dim %arg1825, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2247 = stablehlo.convert %2244 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2248 = stablehlo.broadcast_in_dim %2245, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2249 = stablehlo.subtract %2247, %2248 : tensor<256x14x14x256xf32>
      %cst_116 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2250 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2251 = stablehlo.add %2246, %2250 : tensor<1x1x1x256xf32>
      %2252 = stablehlo.rsqrt %2251 : tensor<1x1x1x256xf32>
      %2253 = stablehlo.reshape %arg1310 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2254 = stablehlo.multiply %2252, %2253 : tensor<1x1x1x256xf32>
      %2255 = stablehlo.broadcast_in_dim %2254, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2256 = stablehlo.multiply %2249, %2255 : tensor<256x14x14x256xf32>
      %2257 = stablehlo.reshape %arg1309 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2258 = stablehlo.broadcast_in_dim %2257, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2259 = stablehlo.add %2256, %2258 : tensor<256x14x14x256xf32>
      %2260 = stablehlo.convert %2259 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2261 = func.call @relu_175(%2260) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2262 = stablehlo.convert %arg1315 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2263 = stablehlo.convolution(%2261, %2262) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2264 = stablehlo.broadcast_in_dim %arg1826, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2265 = stablehlo.broadcast_in_dim %arg1827, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2266 = stablehlo.convert %2263 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2267 = stablehlo.broadcast_in_dim %2264, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2268 = stablehlo.subtract %2266, %2267 : tensor<256x14x14x1024xf32>
      %cst_117 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2269 = stablehlo.broadcast_in_dim %cst_117, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2270 = stablehlo.add %2265, %2269 : tensor<1x1x1x1024xf32>
      %2271 = stablehlo.rsqrt %2270 : tensor<1x1x1x1024xf32>
      %2272 = stablehlo.reshape %arg1312 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2273 = stablehlo.multiply %2271, %2272 : tensor<1x1x1x1024xf32>
      %2274 = stablehlo.broadcast_in_dim %2273, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2275 = stablehlo.multiply %2268, %2274 : tensor<256x14x14x1024xf32>
      %2276 = stablehlo.reshape %arg1311 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2277 = stablehlo.broadcast_in_dim %2276, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2278 = stablehlo.add %2275, %2277 : tensor<256x14x14x1024xf32>
      %2279 = stablehlo.convert %2278 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2280 = stablehlo.add %2223, %2279 : tensor<256x14x14x1024xf16>
      %2281 = func.call @relu_193(%2280) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2282 = stablehlo.convert %arg1322 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2283 = stablehlo.convolution(%2281, %2282) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2284 = stablehlo.broadcast_in_dim %arg1828, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2285 = stablehlo.broadcast_in_dim %arg1829, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2286 = stablehlo.convert %2283 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2287 = stablehlo.broadcast_in_dim %2284, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2288 = stablehlo.subtract %2286, %2287 : tensor<256x14x14x256xf32>
      %cst_118 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2289 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2290 = stablehlo.add %2285, %2289 : tensor<1x1x1x256xf32>
      %2291 = stablehlo.rsqrt %2290 : tensor<1x1x1x256xf32>
      %2292 = stablehlo.reshape %arg1317 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2293 = stablehlo.multiply %2291, %2292 : tensor<1x1x1x256xf32>
      %2294 = stablehlo.broadcast_in_dim %2293, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2295 = stablehlo.multiply %2288, %2294 : tensor<256x14x14x256xf32>
      %2296 = stablehlo.reshape %arg1316 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2297 = stablehlo.broadcast_in_dim %2296, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2298 = stablehlo.add %2295, %2297 : tensor<256x14x14x256xf32>
      %2299 = stablehlo.convert %2298 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2300 = func.call @relu_175(%2299) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2301 = stablehlo.convert %arg1323 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2302 = stablehlo.convolution(%2300, %2301) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2303 = stablehlo.broadcast_in_dim %arg1830, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2304 = stablehlo.broadcast_in_dim %arg1831, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2305 = stablehlo.convert %2302 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2306 = stablehlo.broadcast_in_dim %2303, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2307 = stablehlo.subtract %2305, %2306 : tensor<256x14x14x256xf32>
      %cst_119 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2308 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2309 = stablehlo.add %2304, %2308 : tensor<1x1x1x256xf32>
      %2310 = stablehlo.rsqrt %2309 : tensor<1x1x1x256xf32>
      %2311 = stablehlo.reshape %arg1319 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2312 = stablehlo.multiply %2310, %2311 : tensor<1x1x1x256xf32>
      %2313 = stablehlo.broadcast_in_dim %2312, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2314 = stablehlo.multiply %2307, %2313 : tensor<256x14x14x256xf32>
      %2315 = stablehlo.reshape %arg1318 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2316 = stablehlo.broadcast_in_dim %2315, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2317 = stablehlo.add %2314, %2316 : tensor<256x14x14x256xf32>
      %2318 = stablehlo.convert %2317 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2319 = func.call @relu_175(%2318) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2320 = stablehlo.convert %arg1324 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2321 = stablehlo.convolution(%2319, %2320) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2322 = stablehlo.broadcast_in_dim %arg1832, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2323 = stablehlo.broadcast_in_dim %arg1833, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2324 = stablehlo.convert %2321 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2325 = stablehlo.broadcast_in_dim %2322, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2326 = stablehlo.subtract %2324, %2325 : tensor<256x14x14x1024xf32>
      %cst_120 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2327 = stablehlo.broadcast_in_dim %cst_120, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2328 = stablehlo.add %2323, %2327 : tensor<1x1x1x1024xf32>
      %2329 = stablehlo.rsqrt %2328 : tensor<1x1x1x1024xf32>
      %2330 = stablehlo.reshape %arg1321 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2331 = stablehlo.multiply %2329, %2330 : tensor<1x1x1x1024xf32>
      %2332 = stablehlo.broadcast_in_dim %2331, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2333 = stablehlo.multiply %2326, %2332 : tensor<256x14x14x1024xf32>
      %2334 = stablehlo.reshape %arg1320 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2335 = stablehlo.broadcast_in_dim %2334, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2336 = stablehlo.add %2333, %2335 : tensor<256x14x14x1024xf32>
      %2337 = stablehlo.convert %2336 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2338 = stablehlo.add %2281, %2337 : tensor<256x14x14x1024xf16>
      %2339 = func.call @relu_193(%2338) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2340 = stablehlo.convert %arg1331 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2341 = stablehlo.convolution(%2339, %2340) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2342 = stablehlo.broadcast_in_dim %arg1834, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2343 = stablehlo.broadcast_in_dim %arg1835, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2344 = stablehlo.convert %2341 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2345 = stablehlo.broadcast_in_dim %2342, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2346 = stablehlo.subtract %2344, %2345 : tensor<256x14x14x256xf32>
      %cst_121 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2347 = stablehlo.broadcast_in_dim %cst_121, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2348 = stablehlo.add %2343, %2347 : tensor<1x1x1x256xf32>
      %2349 = stablehlo.rsqrt %2348 : tensor<1x1x1x256xf32>
      %2350 = stablehlo.reshape %arg1326 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2351 = stablehlo.multiply %2349, %2350 : tensor<1x1x1x256xf32>
      %2352 = stablehlo.broadcast_in_dim %2351, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2353 = stablehlo.multiply %2346, %2352 : tensor<256x14x14x256xf32>
      %2354 = stablehlo.reshape %arg1325 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2355 = stablehlo.broadcast_in_dim %2354, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2356 = stablehlo.add %2353, %2355 : tensor<256x14x14x256xf32>
      %2357 = stablehlo.convert %2356 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2358 = func.call @relu_175(%2357) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2359 = stablehlo.convert %arg1332 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2360 = stablehlo.convolution(%2358, %2359) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2361 = stablehlo.broadcast_in_dim %arg1836, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2362 = stablehlo.broadcast_in_dim %arg1837, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2363 = stablehlo.convert %2360 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2364 = stablehlo.broadcast_in_dim %2361, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2365 = stablehlo.subtract %2363, %2364 : tensor<256x14x14x256xf32>
      %cst_122 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2366 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2367 = stablehlo.add %2362, %2366 : tensor<1x1x1x256xf32>
      %2368 = stablehlo.rsqrt %2367 : tensor<1x1x1x256xf32>
      %2369 = stablehlo.reshape %arg1328 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2370 = stablehlo.multiply %2368, %2369 : tensor<1x1x1x256xf32>
      %2371 = stablehlo.broadcast_in_dim %2370, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2372 = stablehlo.multiply %2365, %2371 : tensor<256x14x14x256xf32>
      %2373 = stablehlo.reshape %arg1327 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2374 = stablehlo.broadcast_in_dim %2373, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2375 = stablehlo.add %2372, %2374 : tensor<256x14x14x256xf32>
      %2376 = stablehlo.convert %2375 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2377 = func.call @relu_175(%2376) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2378 = stablehlo.convert %arg1333 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2379 = stablehlo.convolution(%2377, %2378) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2380 = stablehlo.broadcast_in_dim %arg1838, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2381 = stablehlo.broadcast_in_dim %arg1839, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2382 = stablehlo.convert %2379 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2383 = stablehlo.broadcast_in_dim %2380, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2384 = stablehlo.subtract %2382, %2383 : tensor<256x14x14x1024xf32>
      %cst_123 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2385 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2386 = stablehlo.add %2381, %2385 : tensor<1x1x1x1024xf32>
      %2387 = stablehlo.rsqrt %2386 : tensor<1x1x1x1024xf32>
      %2388 = stablehlo.reshape %arg1330 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2389 = stablehlo.multiply %2387, %2388 : tensor<1x1x1x1024xf32>
      %2390 = stablehlo.broadcast_in_dim %2389, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2391 = stablehlo.multiply %2384, %2390 : tensor<256x14x14x1024xf32>
      %2392 = stablehlo.reshape %arg1329 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2393 = stablehlo.broadcast_in_dim %2392, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2394 = stablehlo.add %2391, %2393 : tensor<256x14x14x1024xf32>
      %2395 = stablehlo.convert %2394 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2396 = stablehlo.add %2339, %2395 : tensor<256x14x14x1024xf16>
      %2397 = func.call @relu_193(%2396) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2398 = stablehlo.convert %arg1349 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2399 = stablehlo.convolution(%2397, %2398) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2400 = stablehlo.broadcast_in_dim %arg1846, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2401 = stablehlo.broadcast_in_dim %arg1847, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2402 = stablehlo.convert %2399 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2403 = stablehlo.broadcast_in_dim %2400, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2404 = stablehlo.subtract %2402, %2403 : tensor<256x14x14x256xf32>
      %cst_124 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2405 = stablehlo.broadcast_in_dim %cst_124, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2406 = stablehlo.add %2401, %2405 : tensor<1x1x1x256xf32>
      %2407 = stablehlo.rsqrt %2406 : tensor<1x1x1x256xf32>
      %2408 = stablehlo.reshape %arg1344 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2409 = stablehlo.multiply %2407, %2408 : tensor<1x1x1x256xf32>
      %2410 = stablehlo.broadcast_in_dim %2409, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2411 = stablehlo.multiply %2404, %2410 : tensor<256x14x14x256xf32>
      %2412 = stablehlo.reshape %arg1343 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2413 = stablehlo.broadcast_in_dim %2412, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2414 = stablehlo.add %2411, %2413 : tensor<256x14x14x256xf32>
      %2415 = stablehlo.convert %2414 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2416 = func.call @relu_175(%2415) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2417 = stablehlo.convert %arg1350 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2418 = stablehlo.convolution(%2416, %2417) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2419 = stablehlo.broadcast_in_dim %arg1848, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2420 = stablehlo.broadcast_in_dim %arg1849, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2421 = stablehlo.convert %2418 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2422 = stablehlo.broadcast_in_dim %2419, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2423 = stablehlo.subtract %2421, %2422 : tensor<256x14x14x256xf32>
      %cst_125 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2424 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2425 = stablehlo.add %2420, %2424 : tensor<1x1x1x256xf32>
      %2426 = stablehlo.rsqrt %2425 : tensor<1x1x1x256xf32>
      %2427 = stablehlo.reshape %arg1346 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2428 = stablehlo.multiply %2426, %2427 : tensor<1x1x1x256xf32>
      %2429 = stablehlo.broadcast_in_dim %2428, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2430 = stablehlo.multiply %2423, %2429 : tensor<256x14x14x256xf32>
      %2431 = stablehlo.reshape %arg1345 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2432 = stablehlo.broadcast_in_dim %2431, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2433 = stablehlo.add %2430, %2432 : tensor<256x14x14x256xf32>
      %2434 = stablehlo.convert %2433 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2435 = func.call @relu_175(%2434) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2436 = stablehlo.convert %arg1351 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2437 = stablehlo.convolution(%2435, %2436) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2438 = stablehlo.broadcast_in_dim %arg1850, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2439 = stablehlo.broadcast_in_dim %arg1851, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2440 = stablehlo.convert %2437 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2441 = stablehlo.broadcast_in_dim %2438, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2442 = stablehlo.subtract %2440, %2441 : tensor<256x14x14x1024xf32>
      %cst_126 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2443 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2444 = stablehlo.add %2439, %2443 : tensor<1x1x1x1024xf32>
      %2445 = stablehlo.rsqrt %2444 : tensor<1x1x1x1024xf32>
      %2446 = stablehlo.reshape %arg1348 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2447 = stablehlo.multiply %2445, %2446 : tensor<1x1x1x1024xf32>
      %2448 = stablehlo.broadcast_in_dim %2447, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2449 = stablehlo.multiply %2442, %2448 : tensor<256x14x14x1024xf32>
      %2450 = stablehlo.reshape %arg1347 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2451 = stablehlo.broadcast_in_dim %2450, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2452 = stablehlo.add %2449, %2451 : tensor<256x14x14x1024xf32>
      %2453 = stablehlo.convert %2452 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2454 = stablehlo.add %2397, %2453 : tensor<256x14x14x1024xf16>
      %2455 = func.call @relu_193(%2454) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2456 = stablehlo.convert %arg1358 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2457 = stablehlo.convolution(%2455, %2456) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2458 = stablehlo.broadcast_in_dim %arg1852, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2459 = stablehlo.broadcast_in_dim %arg1853, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2460 = stablehlo.convert %2457 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2461 = stablehlo.broadcast_in_dim %2458, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2462 = stablehlo.subtract %2460, %2461 : tensor<256x14x14x256xf32>
      %cst_127 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2463 = stablehlo.broadcast_in_dim %cst_127, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2464 = stablehlo.add %2459, %2463 : tensor<1x1x1x256xf32>
      %2465 = stablehlo.rsqrt %2464 : tensor<1x1x1x256xf32>
      %2466 = stablehlo.reshape %arg1353 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2467 = stablehlo.multiply %2465, %2466 : tensor<1x1x1x256xf32>
      %2468 = stablehlo.broadcast_in_dim %2467, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2469 = stablehlo.multiply %2462, %2468 : tensor<256x14x14x256xf32>
      %2470 = stablehlo.reshape %arg1352 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2471 = stablehlo.broadcast_in_dim %2470, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2472 = stablehlo.add %2469, %2471 : tensor<256x14x14x256xf32>
      %2473 = stablehlo.convert %2472 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2474 = func.call @relu_175(%2473) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2475 = stablehlo.convert %arg1359 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2476 = stablehlo.convolution(%2474, %2475) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2477 = stablehlo.broadcast_in_dim %arg1854, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2478 = stablehlo.broadcast_in_dim %arg1855, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2479 = stablehlo.convert %2476 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2480 = stablehlo.broadcast_in_dim %2477, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2481 = stablehlo.subtract %2479, %2480 : tensor<256x14x14x256xf32>
      %cst_128 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2482 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2483 = stablehlo.add %2478, %2482 : tensor<1x1x1x256xf32>
      %2484 = stablehlo.rsqrt %2483 : tensor<1x1x1x256xf32>
      %2485 = stablehlo.reshape %arg1355 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2486 = stablehlo.multiply %2484, %2485 : tensor<1x1x1x256xf32>
      %2487 = stablehlo.broadcast_in_dim %2486, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2488 = stablehlo.multiply %2481, %2487 : tensor<256x14x14x256xf32>
      %2489 = stablehlo.reshape %arg1354 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2490 = stablehlo.broadcast_in_dim %2489, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2491 = stablehlo.add %2488, %2490 : tensor<256x14x14x256xf32>
      %2492 = stablehlo.convert %2491 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2493 = func.call @relu_175(%2492) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2494 = stablehlo.convert %arg1360 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2495 = stablehlo.convolution(%2493, %2494) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2496 = stablehlo.broadcast_in_dim %arg1856, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2497 = stablehlo.broadcast_in_dim %arg1857, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2498 = stablehlo.convert %2495 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2499 = stablehlo.broadcast_in_dim %2496, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2500 = stablehlo.subtract %2498, %2499 : tensor<256x14x14x1024xf32>
      %cst_129 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2501 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2502 = stablehlo.add %2497, %2501 : tensor<1x1x1x1024xf32>
      %2503 = stablehlo.rsqrt %2502 : tensor<1x1x1x1024xf32>
      %2504 = stablehlo.reshape %arg1357 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2505 = stablehlo.multiply %2503, %2504 : tensor<1x1x1x1024xf32>
      %2506 = stablehlo.broadcast_in_dim %2505, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2507 = stablehlo.multiply %2500, %2506 : tensor<256x14x14x1024xf32>
      %2508 = stablehlo.reshape %arg1356 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2509 = stablehlo.broadcast_in_dim %2508, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2510 = stablehlo.add %2507, %2509 : tensor<256x14x14x1024xf32>
      %2511 = stablehlo.convert %2510 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2512 = stablehlo.add %2455, %2511 : tensor<256x14x14x1024xf16>
      %2513 = func.call @relu_193(%2512) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2514 = stablehlo.convert %arg1367 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2515 = stablehlo.convolution(%2513, %2514) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2516 = stablehlo.broadcast_in_dim %arg1858, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2517 = stablehlo.broadcast_in_dim %arg1859, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2518 = stablehlo.convert %2515 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2519 = stablehlo.broadcast_in_dim %2516, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2520 = stablehlo.subtract %2518, %2519 : tensor<256x14x14x256xf32>
      %cst_130 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2521 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2522 = stablehlo.add %2517, %2521 : tensor<1x1x1x256xf32>
      %2523 = stablehlo.rsqrt %2522 : tensor<1x1x1x256xf32>
      %2524 = stablehlo.reshape %arg1362 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2525 = stablehlo.multiply %2523, %2524 : tensor<1x1x1x256xf32>
      %2526 = stablehlo.broadcast_in_dim %2525, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2527 = stablehlo.multiply %2520, %2526 : tensor<256x14x14x256xf32>
      %2528 = stablehlo.reshape %arg1361 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2529 = stablehlo.broadcast_in_dim %2528, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2530 = stablehlo.add %2527, %2529 : tensor<256x14x14x256xf32>
      %2531 = stablehlo.convert %2530 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2532 = func.call @relu_175(%2531) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2533 = stablehlo.convert %arg1368 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2534 = stablehlo.convolution(%2532, %2533) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2535 = stablehlo.broadcast_in_dim %arg1860, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2536 = stablehlo.broadcast_in_dim %arg1861, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2537 = stablehlo.convert %2534 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2538 = stablehlo.broadcast_in_dim %2535, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2539 = stablehlo.subtract %2537, %2538 : tensor<256x14x14x256xf32>
      %cst_131 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2540 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2541 = stablehlo.add %2536, %2540 : tensor<1x1x1x256xf32>
      %2542 = stablehlo.rsqrt %2541 : tensor<1x1x1x256xf32>
      %2543 = stablehlo.reshape %arg1364 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2544 = stablehlo.multiply %2542, %2543 : tensor<1x1x1x256xf32>
      %2545 = stablehlo.broadcast_in_dim %2544, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2546 = stablehlo.multiply %2539, %2545 : tensor<256x14x14x256xf32>
      %2547 = stablehlo.reshape %arg1363 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2548 = stablehlo.broadcast_in_dim %2547, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2549 = stablehlo.add %2546, %2548 : tensor<256x14x14x256xf32>
      %2550 = stablehlo.convert %2549 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2551 = func.call @relu_175(%2550) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2552 = stablehlo.convert %arg1369 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2553 = stablehlo.convolution(%2551, %2552) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2554 = stablehlo.broadcast_in_dim %arg1862, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2555 = stablehlo.broadcast_in_dim %arg1863, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2556 = stablehlo.convert %2553 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2557 = stablehlo.broadcast_in_dim %2554, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2558 = stablehlo.subtract %2556, %2557 : tensor<256x14x14x1024xf32>
      %cst_132 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2559 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2560 = stablehlo.add %2555, %2559 : tensor<1x1x1x1024xf32>
      %2561 = stablehlo.rsqrt %2560 : tensor<1x1x1x1024xf32>
      %2562 = stablehlo.reshape %arg1366 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2563 = stablehlo.multiply %2561, %2562 : tensor<1x1x1x1024xf32>
      %2564 = stablehlo.broadcast_in_dim %2563, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2565 = stablehlo.multiply %2558, %2564 : tensor<256x14x14x1024xf32>
      %2566 = stablehlo.reshape %arg1365 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2567 = stablehlo.broadcast_in_dim %2566, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2568 = stablehlo.add %2565, %2567 : tensor<256x14x14x1024xf32>
      %2569 = stablehlo.convert %2568 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2570 = stablehlo.add %2513, %2569 : tensor<256x14x14x1024xf16>
      %2571 = func.call @relu_193(%2570) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2572 = stablehlo.convert %arg1376 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2573 = stablehlo.convolution(%2571, %2572) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2574 = stablehlo.broadcast_in_dim %arg1864, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2575 = stablehlo.broadcast_in_dim %arg1865, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2576 = stablehlo.convert %2573 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2577 = stablehlo.broadcast_in_dim %2574, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2578 = stablehlo.subtract %2576, %2577 : tensor<256x14x14x256xf32>
      %cst_133 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2579 = stablehlo.broadcast_in_dim %cst_133, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2580 = stablehlo.add %2575, %2579 : tensor<1x1x1x256xf32>
      %2581 = stablehlo.rsqrt %2580 : tensor<1x1x1x256xf32>
      %2582 = stablehlo.reshape %arg1371 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2583 = stablehlo.multiply %2581, %2582 : tensor<1x1x1x256xf32>
      %2584 = stablehlo.broadcast_in_dim %2583, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2585 = stablehlo.multiply %2578, %2584 : tensor<256x14x14x256xf32>
      %2586 = stablehlo.reshape %arg1370 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2587 = stablehlo.broadcast_in_dim %2586, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2588 = stablehlo.add %2585, %2587 : tensor<256x14x14x256xf32>
      %2589 = stablehlo.convert %2588 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2590 = func.call @relu_175(%2589) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2591 = stablehlo.convert %arg1377 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2592 = stablehlo.convolution(%2590, %2591) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2593 = stablehlo.broadcast_in_dim %arg1866, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2594 = stablehlo.broadcast_in_dim %arg1867, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2595 = stablehlo.convert %2592 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2596 = stablehlo.broadcast_in_dim %2593, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2597 = stablehlo.subtract %2595, %2596 : tensor<256x14x14x256xf32>
      %cst_134 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2598 = stablehlo.broadcast_in_dim %cst_134, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2599 = stablehlo.add %2594, %2598 : tensor<1x1x1x256xf32>
      %2600 = stablehlo.rsqrt %2599 : tensor<1x1x1x256xf32>
      %2601 = stablehlo.reshape %arg1373 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2602 = stablehlo.multiply %2600, %2601 : tensor<1x1x1x256xf32>
      %2603 = stablehlo.broadcast_in_dim %2602, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2604 = stablehlo.multiply %2597, %2603 : tensor<256x14x14x256xf32>
      %2605 = stablehlo.reshape %arg1372 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2606 = stablehlo.broadcast_in_dim %2605, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2607 = stablehlo.add %2604, %2606 : tensor<256x14x14x256xf32>
      %2608 = stablehlo.convert %2607 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2609 = func.call @relu_175(%2608) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2610 = stablehlo.convert %arg1378 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2611 = stablehlo.convolution(%2609, %2610) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2612 = stablehlo.broadcast_in_dim %arg1868, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2613 = stablehlo.broadcast_in_dim %arg1869, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2614 = stablehlo.convert %2611 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2615 = stablehlo.broadcast_in_dim %2612, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2616 = stablehlo.subtract %2614, %2615 : tensor<256x14x14x1024xf32>
      %cst_135 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2617 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2618 = stablehlo.add %2613, %2617 : tensor<1x1x1x1024xf32>
      %2619 = stablehlo.rsqrt %2618 : tensor<1x1x1x1024xf32>
      %2620 = stablehlo.reshape %arg1375 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2621 = stablehlo.multiply %2619, %2620 : tensor<1x1x1x1024xf32>
      %2622 = stablehlo.broadcast_in_dim %2621, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2623 = stablehlo.multiply %2616, %2622 : tensor<256x14x14x1024xf32>
      %2624 = stablehlo.reshape %arg1374 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2625 = stablehlo.broadcast_in_dim %2624, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2626 = stablehlo.add %2623, %2625 : tensor<256x14x14x1024xf32>
      %2627 = stablehlo.convert %2626 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2628 = stablehlo.add %2571, %2627 : tensor<256x14x14x1024xf16>
      %2629 = func.call @relu_193(%2628) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2630 = stablehlo.convert %arg1385 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2631 = stablehlo.convolution(%2629, %2630) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2632 = stablehlo.broadcast_in_dim %arg1870, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2633 = stablehlo.broadcast_in_dim %arg1871, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2634 = stablehlo.convert %2631 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2635 = stablehlo.broadcast_in_dim %2632, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2636 = stablehlo.subtract %2634, %2635 : tensor<256x14x14x256xf32>
      %cst_136 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2637 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2638 = stablehlo.add %2633, %2637 : tensor<1x1x1x256xf32>
      %2639 = stablehlo.rsqrt %2638 : tensor<1x1x1x256xf32>
      %2640 = stablehlo.reshape %arg1380 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2641 = stablehlo.multiply %2639, %2640 : tensor<1x1x1x256xf32>
      %2642 = stablehlo.broadcast_in_dim %2641, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2643 = stablehlo.multiply %2636, %2642 : tensor<256x14x14x256xf32>
      %2644 = stablehlo.reshape %arg1379 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2645 = stablehlo.broadcast_in_dim %2644, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2646 = stablehlo.add %2643, %2645 : tensor<256x14x14x256xf32>
      %2647 = stablehlo.convert %2646 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2648 = func.call @relu_175(%2647) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2649 = stablehlo.convert %arg1386 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2650 = stablehlo.convolution(%2648, %2649) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2651 = stablehlo.broadcast_in_dim %arg1872, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2652 = stablehlo.broadcast_in_dim %arg1873, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2653 = stablehlo.convert %2650 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2654 = stablehlo.broadcast_in_dim %2651, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2655 = stablehlo.subtract %2653, %2654 : tensor<256x14x14x256xf32>
      %cst_137 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2656 = stablehlo.broadcast_in_dim %cst_137, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2657 = stablehlo.add %2652, %2656 : tensor<1x1x1x256xf32>
      %2658 = stablehlo.rsqrt %2657 : tensor<1x1x1x256xf32>
      %2659 = stablehlo.reshape %arg1382 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2660 = stablehlo.multiply %2658, %2659 : tensor<1x1x1x256xf32>
      %2661 = stablehlo.broadcast_in_dim %2660, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2662 = stablehlo.multiply %2655, %2661 : tensor<256x14x14x256xf32>
      %2663 = stablehlo.reshape %arg1381 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2664 = stablehlo.broadcast_in_dim %2663, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2665 = stablehlo.add %2662, %2664 : tensor<256x14x14x256xf32>
      %2666 = stablehlo.convert %2665 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2667 = func.call @relu_175(%2666) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2668 = stablehlo.convert %arg1387 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2669 = stablehlo.convolution(%2667, %2668) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2670 = stablehlo.broadcast_in_dim %arg1874, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2671 = stablehlo.broadcast_in_dim %arg1875, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2672 = stablehlo.convert %2669 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2673 = stablehlo.broadcast_in_dim %2670, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2674 = stablehlo.subtract %2672, %2673 : tensor<256x14x14x1024xf32>
      %cst_138 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2675 = stablehlo.broadcast_in_dim %cst_138, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2676 = stablehlo.add %2671, %2675 : tensor<1x1x1x1024xf32>
      %2677 = stablehlo.rsqrt %2676 : tensor<1x1x1x1024xf32>
      %2678 = stablehlo.reshape %arg1384 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2679 = stablehlo.multiply %2677, %2678 : tensor<1x1x1x1024xf32>
      %2680 = stablehlo.broadcast_in_dim %2679, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2681 = stablehlo.multiply %2674, %2680 : tensor<256x14x14x1024xf32>
      %2682 = stablehlo.reshape %arg1383 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2683 = stablehlo.broadcast_in_dim %2682, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2684 = stablehlo.add %2681, %2683 : tensor<256x14x14x1024xf32>
      %2685 = stablehlo.convert %2684 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2686 = stablehlo.add %2629, %2685 : tensor<256x14x14x1024xf16>
      %2687 = func.call @relu_193(%2686) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2688 = stablehlo.convert %arg1394 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2689 = stablehlo.convolution(%2687, %2688) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2690 = stablehlo.broadcast_in_dim %arg1876, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2691 = stablehlo.broadcast_in_dim %arg1877, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2692 = stablehlo.convert %2689 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2693 = stablehlo.broadcast_in_dim %2690, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2694 = stablehlo.subtract %2692, %2693 : tensor<256x14x14x256xf32>
      %cst_139 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2695 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2696 = stablehlo.add %2691, %2695 : tensor<1x1x1x256xf32>
      %2697 = stablehlo.rsqrt %2696 : tensor<1x1x1x256xf32>
      %2698 = stablehlo.reshape %arg1389 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2699 = stablehlo.multiply %2697, %2698 : tensor<1x1x1x256xf32>
      %2700 = stablehlo.broadcast_in_dim %2699, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2701 = stablehlo.multiply %2694, %2700 : tensor<256x14x14x256xf32>
      %2702 = stablehlo.reshape %arg1388 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2703 = stablehlo.broadcast_in_dim %2702, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2704 = stablehlo.add %2701, %2703 : tensor<256x14x14x256xf32>
      %2705 = stablehlo.convert %2704 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2706 = func.call @relu_175(%2705) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2707 = stablehlo.convert %arg1395 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2708 = stablehlo.convolution(%2706, %2707) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2709 = stablehlo.broadcast_in_dim %arg1878, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2710 = stablehlo.broadcast_in_dim %arg1879, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2711 = stablehlo.convert %2708 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2712 = stablehlo.broadcast_in_dim %2709, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2713 = stablehlo.subtract %2711, %2712 : tensor<256x14x14x256xf32>
      %cst_140 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2714 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2715 = stablehlo.add %2710, %2714 : tensor<1x1x1x256xf32>
      %2716 = stablehlo.rsqrt %2715 : tensor<1x1x1x256xf32>
      %2717 = stablehlo.reshape %arg1391 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2718 = stablehlo.multiply %2716, %2717 : tensor<1x1x1x256xf32>
      %2719 = stablehlo.broadcast_in_dim %2718, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2720 = stablehlo.multiply %2713, %2719 : tensor<256x14x14x256xf32>
      %2721 = stablehlo.reshape %arg1390 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2722 = stablehlo.broadcast_in_dim %2721, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2723 = stablehlo.add %2720, %2722 : tensor<256x14x14x256xf32>
      %2724 = stablehlo.convert %2723 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2725 = func.call @relu_175(%2724) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2726 = stablehlo.convert %arg1396 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2727 = stablehlo.convolution(%2725, %2726) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2728 = stablehlo.broadcast_in_dim %arg1880, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2729 = stablehlo.broadcast_in_dim %arg1881, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2730 = stablehlo.convert %2727 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2731 = stablehlo.broadcast_in_dim %2728, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2732 = stablehlo.subtract %2730, %2731 : tensor<256x14x14x1024xf32>
      %cst_141 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2733 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2734 = stablehlo.add %2729, %2733 : tensor<1x1x1x1024xf32>
      %2735 = stablehlo.rsqrt %2734 : tensor<1x1x1x1024xf32>
      %2736 = stablehlo.reshape %arg1393 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2737 = stablehlo.multiply %2735, %2736 : tensor<1x1x1x1024xf32>
      %2738 = stablehlo.broadcast_in_dim %2737, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2739 = stablehlo.multiply %2732, %2738 : tensor<256x14x14x1024xf32>
      %2740 = stablehlo.reshape %arg1392 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2741 = stablehlo.broadcast_in_dim %2740, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2742 = stablehlo.add %2739, %2741 : tensor<256x14x14x1024xf32>
      %2743 = stablehlo.convert %2742 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2744 = stablehlo.add %2687, %2743 : tensor<256x14x14x1024xf16>
      %2745 = func.call @relu_193(%2744) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2746 = stablehlo.convert %arg1403 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2747 = stablehlo.convolution(%2745, %2746) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2748 = stablehlo.broadcast_in_dim %arg1882, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2749 = stablehlo.broadcast_in_dim %arg1883, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2750 = stablehlo.convert %2747 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2751 = stablehlo.broadcast_in_dim %2748, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2752 = stablehlo.subtract %2750, %2751 : tensor<256x14x14x256xf32>
      %cst_142 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2753 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2754 = stablehlo.add %2749, %2753 : tensor<1x1x1x256xf32>
      %2755 = stablehlo.rsqrt %2754 : tensor<1x1x1x256xf32>
      %2756 = stablehlo.reshape %arg1398 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2757 = stablehlo.multiply %2755, %2756 : tensor<1x1x1x256xf32>
      %2758 = stablehlo.broadcast_in_dim %2757, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2759 = stablehlo.multiply %2752, %2758 : tensor<256x14x14x256xf32>
      %2760 = stablehlo.reshape %arg1397 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2761 = stablehlo.broadcast_in_dim %2760, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2762 = stablehlo.add %2759, %2761 : tensor<256x14x14x256xf32>
      %2763 = stablehlo.convert %2762 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2764 = func.call @relu_175(%2763) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2765 = stablehlo.convert %arg1404 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2766 = stablehlo.convolution(%2764, %2765) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2767 = stablehlo.broadcast_in_dim %arg1884, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2768 = stablehlo.broadcast_in_dim %arg1885, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2769 = stablehlo.convert %2766 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2770 = stablehlo.broadcast_in_dim %2767, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2771 = stablehlo.subtract %2769, %2770 : tensor<256x14x14x256xf32>
      %cst_143 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2772 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2773 = stablehlo.add %2768, %2772 : tensor<1x1x1x256xf32>
      %2774 = stablehlo.rsqrt %2773 : tensor<1x1x1x256xf32>
      %2775 = stablehlo.reshape %arg1400 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2776 = stablehlo.multiply %2774, %2775 : tensor<1x1x1x256xf32>
      %2777 = stablehlo.broadcast_in_dim %2776, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2778 = stablehlo.multiply %2771, %2777 : tensor<256x14x14x256xf32>
      %2779 = stablehlo.reshape %arg1399 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2780 = stablehlo.broadcast_in_dim %2779, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2781 = stablehlo.add %2778, %2780 : tensor<256x14x14x256xf32>
      %2782 = stablehlo.convert %2781 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2783 = func.call @relu_175(%2782) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2784 = stablehlo.convert %arg1405 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2785 = stablehlo.convolution(%2783, %2784) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2786 = stablehlo.broadcast_in_dim %arg1886, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2787 = stablehlo.broadcast_in_dim %arg1887, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2788 = stablehlo.convert %2785 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2789 = stablehlo.broadcast_in_dim %2786, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2790 = stablehlo.subtract %2788, %2789 : tensor<256x14x14x1024xf32>
      %cst_144 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2791 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2792 = stablehlo.add %2787, %2791 : tensor<1x1x1x1024xf32>
      %2793 = stablehlo.rsqrt %2792 : tensor<1x1x1x1024xf32>
      %2794 = stablehlo.reshape %arg1402 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2795 = stablehlo.multiply %2793, %2794 : tensor<1x1x1x1024xf32>
      %2796 = stablehlo.broadcast_in_dim %2795, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2797 = stablehlo.multiply %2790, %2796 : tensor<256x14x14x1024xf32>
      %2798 = stablehlo.reshape %arg1401 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2799 = stablehlo.broadcast_in_dim %2798, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2800 = stablehlo.add %2797, %2799 : tensor<256x14x14x1024xf32>
      %2801 = stablehlo.convert %2800 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2802 = stablehlo.add %2745, %2801 : tensor<256x14x14x1024xf16>
      %2803 = func.call @relu_193(%2802) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2804 = stablehlo.convert %arg1412 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2805 = stablehlo.convolution(%2803, %2804) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2806 = stablehlo.broadcast_in_dim %arg1888, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2807 = stablehlo.broadcast_in_dim %arg1889, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2808 = stablehlo.convert %2805 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2809 = stablehlo.broadcast_in_dim %2806, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2810 = stablehlo.subtract %2808, %2809 : tensor<256x14x14x256xf32>
      %cst_145 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2811 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2812 = stablehlo.add %2807, %2811 : tensor<1x1x1x256xf32>
      %2813 = stablehlo.rsqrt %2812 : tensor<1x1x1x256xf32>
      %2814 = stablehlo.reshape %arg1407 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2815 = stablehlo.multiply %2813, %2814 : tensor<1x1x1x256xf32>
      %2816 = stablehlo.broadcast_in_dim %2815, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2817 = stablehlo.multiply %2810, %2816 : tensor<256x14x14x256xf32>
      %2818 = stablehlo.reshape %arg1406 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2819 = stablehlo.broadcast_in_dim %2818, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2820 = stablehlo.add %2817, %2819 : tensor<256x14x14x256xf32>
      %2821 = stablehlo.convert %2820 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2822 = func.call @relu_175(%2821) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2823 = stablehlo.convert %arg1413 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2824 = stablehlo.convolution(%2822, %2823) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2825 = stablehlo.broadcast_in_dim %arg1890, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2826 = stablehlo.broadcast_in_dim %arg1891, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2827 = stablehlo.convert %2824 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2828 = stablehlo.broadcast_in_dim %2825, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2829 = stablehlo.subtract %2827, %2828 : tensor<256x14x14x256xf32>
      %cst_146 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2830 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2831 = stablehlo.add %2826, %2830 : tensor<1x1x1x256xf32>
      %2832 = stablehlo.rsqrt %2831 : tensor<1x1x1x256xf32>
      %2833 = stablehlo.reshape %arg1409 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2834 = stablehlo.multiply %2832, %2833 : tensor<1x1x1x256xf32>
      %2835 = stablehlo.broadcast_in_dim %2834, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2836 = stablehlo.multiply %2829, %2835 : tensor<256x14x14x256xf32>
      %2837 = stablehlo.reshape %arg1408 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2838 = stablehlo.broadcast_in_dim %2837, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2839 = stablehlo.add %2836, %2838 : tensor<256x14x14x256xf32>
      %2840 = stablehlo.convert %2839 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2841 = func.call @relu_175(%2840) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2842 = stablehlo.convert %arg1414 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2843 = stablehlo.convolution(%2841, %2842) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2844 = stablehlo.broadcast_in_dim %arg1892, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2845 = stablehlo.broadcast_in_dim %arg1893, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2846 = stablehlo.convert %2843 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2847 = stablehlo.broadcast_in_dim %2844, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2848 = stablehlo.subtract %2846, %2847 : tensor<256x14x14x1024xf32>
      %cst_147 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2849 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2850 = stablehlo.add %2845, %2849 : tensor<1x1x1x1024xf32>
      %2851 = stablehlo.rsqrt %2850 : tensor<1x1x1x1024xf32>
      %2852 = stablehlo.reshape %arg1411 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2853 = stablehlo.multiply %2851, %2852 : tensor<1x1x1x1024xf32>
      %2854 = stablehlo.broadcast_in_dim %2853, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2855 = stablehlo.multiply %2848, %2854 : tensor<256x14x14x1024xf32>
      %2856 = stablehlo.reshape %arg1410 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2857 = stablehlo.broadcast_in_dim %2856, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2858 = stablehlo.add %2855, %2857 : tensor<256x14x14x1024xf32>
      %2859 = stablehlo.convert %2858 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2860 = stablehlo.add %2803, %2859 : tensor<256x14x14x1024xf16>
      %2861 = func.call @relu_193(%2860) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2862 = stablehlo.convert %arg1421 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2863 = stablehlo.convolution(%2861, %2862) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2864 = stablehlo.broadcast_in_dim %arg1894, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2865 = stablehlo.broadcast_in_dim %arg1895, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2866 = stablehlo.convert %2863 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2867 = stablehlo.broadcast_in_dim %2864, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2868 = stablehlo.subtract %2866, %2867 : tensor<256x14x14x256xf32>
      %cst_148 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2869 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2870 = stablehlo.add %2865, %2869 : tensor<1x1x1x256xf32>
      %2871 = stablehlo.rsqrt %2870 : tensor<1x1x1x256xf32>
      %2872 = stablehlo.reshape %arg1416 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2873 = stablehlo.multiply %2871, %2872 : tensor<1x1x1x256xf32>
      %2874 = stablehlo.broadcast_in_dim %2873, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2875 = stablehlo.multiply %2868, %2874 : tensor<256x14x14x256xf32>
      %2876 = stablehlo.reshape %arg1415 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2877 = stablehlo.broadcast_in_dim %2876, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2878 = stablehlo.add %2875, %2877 : tensor<256x14x14x256xf32>
      %2879 = stablehlo.convert %2878 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2880 = func.call @relu_175(%2879) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2881 = stablehlo.convert %arg1422 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2882 = stablehlo.convolution(%2880, %2881) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2883 = stablehlo.broadcast_in_dim %arg1896, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2884 = stablehlo.broadcast_in_dim %arg1897, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2885 = stablehlo.convert %2882 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2886 = stablehlo.broadcast_in_dim %2883, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2887 = stablehlo.subtract %2885, %2886 : tensor<256x14x14x256xf32>
      %cst_149 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2888 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2889 = stablehlo.add %2884, %2888 : tensor<1x1x1x256xf32>
      %2890 = stablehlo.rsqrt %2889 : tensor<1x1x1x256xf32>
      %2891 = stablehlo.reshape %arg1418 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2892 = stablehlo.multiply %2890, %2891 : tensor<1x1x1x256xf32>
      %2893 = stablehlo.broadcast_in_dim %2892, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2894 = stablehlo.multiply %2887, %2893 : tensor<256x14x14x256xf32>
      %2895 = stablehlo.reshape %arg1417 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2896 = stablehlo.broadcast_in_dim %2895, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2897 = stablehlo.add %2894, %2896 : tensor<256x14x14x256xf32>
      %2898 = stablehlo.convert %2897 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2899 = func.call @relu_175(%2898) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2900 = stablehlo.convert %arg1423 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2901 = stablehlo.convolution(%2899, %2900) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2902 = stablehlo.broadcast_in_dim %arg1898, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2903 = stablehlo.broadcast_in_dim %arg1899, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2904 = stablehlo.convert %2901 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2905 = stablehlo.broadcast_in_dim %2902, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2906 = stablehlo.subtract %2904, %2905 : tensor<256x14x14x1024xf32>
      %cst_150 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2907 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2908 = stablehlo.add %2903, %2907 : tensor<1x1x1x1024xf32>
      %2909 = stablehlo.rsqrt %2908 : tensor<1x1x1x1024xf32>
      %2910 = stablehlo.reshape %arg1420 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2911 = stablehlo.multiply %2909, %2910 : tensor<1x1x1x1024xf32>
      %2912 = stablehlo.broadcast_in_dim %2911, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2913 = stablehlo.multiply %2906, %2912 : tensor<256x14x14x1024xf32>
      %2914 = stablehlo.reshape %arg1419 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2915 = stablehlo.broadcast_in_dim %2914, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2916 = stablehlo.add %2913, %2915 : tensor<256x14x14x1024xf32>
      %2917 = stablehlo.convert %2916 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2918 = stablehlo.add %2861, %2917 : tensor<256x14x14x1024xf16>
      %2919 = func.call @relu_193(%2918) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2920 = stablehlo.convert %arg1430 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2921 = stablehlo.convolution(%2919, %2920) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2922 = stablehlo.broadcast_in_dim %arg1900, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2923 = stablehlo.broadcast_in_dim %arg1901, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2924 = stablehlo.convert %2921 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2925 = stablehlo.broadcast_in_dim %2922, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2926 = stablehlo.subtract %2924, %2925 : tensor<256x14x14x256xf32>
      %cst_151 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2927 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2928 = stablehlo.add %2923, %2927 : tensor<1x1x1x256xf32>
      %2929 = stablehlo.rsqrt %2928 : tensor<1x1x1x256xf32>
      %2930 = stablehlo.reshape %arg1425 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2931 = stablehlo.multiply %2929, %2930 : tensor<1x1x1x256xf32>
      %2932 = stablehlo.broadcast_in_dim %2931, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2933 = stablehlo.multiply %2926, %2932 : tensor<256x14x14x256xf32>
      %2934 = stablehlo.reshape %arg1424 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2935 = stablehlo.broadcast_in_dim %2934, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2936 = stablehlo.add %2933, %2935 : tensor<256x14x14x256xf32>
      %2937 = stablehlo.convert %2936 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2938 = func.call @relu_175(%2937) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2939 = stablehlo.convert %arg1431 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2940 = stablehlo.convolution(%2938, %2939) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2941 = stablehlo.broadcast_in_dim %arg1902, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2942 = stablehlo.broadcast_in_dim %arg1903, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2943 = stablehlo.convert %2940 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2944 = stablehlo.broadcast_in_dim %2941, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2945 = stablehlo.subtract %2943, %2944 : tensor<256x14x14x256xf32>
      %cst_152 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2946 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2947 = stablehlo.add %2942, %2946 : tensor<1x1x1x256xf32>
      %2948 = stablehlo.rsqrt %2947 : tensor<1x1x1x256xf32>
      %2949 = stablehlo.reshape %arg1427 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2950 = stablehlo.multiply %2948, %2949 : tensor<1x1x1x256xf32>
      %2951 = stablehlo.broadcast_in_dim %2950, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2952 = stablehlo.multiply %2945, %2951 : tensor<256x14x14x256xf32>
      %2953 = stablehlo.reshape %arg1426 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2954 = stablehlo.broadcast_in_dim %2953, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2955 = stablehlo.add %2952, %2954 : tensor<256x14x14x256xf32>
      %2956 = stablehlo.convert %2955 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2957 = func.call @relu_175(%2956) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2958 = stablehlo.convert %arg1432 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %2959 = stablehlo.convolution(%2957, %2958) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2960 = stablehlo.broadcast_in_dim %arg1904, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2961 = stablehlo.broadcast_in_dim %arg1905, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2962 = stablehlo.convert %2959 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %2963 = stablehlo.broadcast_in_dim %2960, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2964 = stablehlo.subtract %2962, %2963 : tensor<256x14x14x1024xf32>
      %cst_153 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2965 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %2966 = stablehlo.add %2961, %2965 : tensor<1x1x1x1024xf32>
      %2967 = stablehlo.rsqrt %2966 : tensor<1x1x1x1024xf32>
      %2968 = stablehlo.reshape %arg1429 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2969 = stablehlo.multiply %2967, %2968 : tensor<1x1x1x1024xf32>
      %2970 = stablehlo.broadcast_in_dim %2969, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2971 = stablehlo.multiply %2964, %2970 : tensor<256x14x14x1024xf32>
      %2972 = stablehlo.reshape %arg1428 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %2973 = stablehlo.broadcast_in_dim %2972, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %2974 = stablehlo.add %2971, %2973 : tensor<256x14x14x1024xf32>
      %2975 = stablehlo.convert %2974 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %2976 = stablehlo.add %2919, %2975 : tensor<256x14x14x1024xf16>
      %2977 = func.call @relu_193(%2976) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %2978 = stablehlo.convert %arg1448 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %2979 = stablehlo.convolution(%2977, %2978) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %2980 = stablehlo.broadcast_in_dim %arg1912, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2981 = stablehlo.broadcast_in_dim %arg1913, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2982 = stablehlo.convert %2979 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %2983 = stablehlo.broadcast_in_dim %2980, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2984 = stablehlo.subtract %2982, %2983 : tensor<256x14x14x256xf32>
      %cst_154 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %2985 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %2986 = stablehlo.add %2981, %2985 : tensor<1x1x1x256xf32>
      %2987 = stablehlo.rsqrt %2986 : tensor<1x1x1x256xf32>
      %2988 = stablehlo.reshape %arg1443 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2989 = stablehlo.multiply %2987, %2988 : tensor<1x1x1x256xf32>
      %2990 = stablehlo.broadcast_in_dim %2989, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2991 = stablehlo.multiply %2984, %2990 : tensor<256x14x14x256xf32>
      %2992 = stablehlo.reshape %arg1442 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %2993 = stablehlo.broadcast_in_dim %2992, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %2994 = stablehlo.add %2991, %2993 : tensor<256x14x14x256xf32>
      %2995 = stablehlo.convert %2994 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %2996 = func.call @relu_175(%2995) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %2997 = stablehlo.convert %arg1449 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %2998 = stablehlo.convolution(%2996, %2997) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %2999 = stablehlo.broadcast_in_dim %arg1914, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3000 = stablehlo.broadcast_in_dim %arg1915, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3001 = stablehlo.convert %2998 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3002 = stablehlo.broadcast_in_dim %2999, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3003 = stablehlo.subtract %3001, %3002 : tensor<256x14x14x256xf32>
      %cst_155 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3004 = stablehlo.broadcast_in_dim %cst_155, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3005 = stablehlo.add %3000, %3004 : tensor<1x1x1x256xf32>
      %3006 = stablehlo.rsqrt %3005 : tensor<1x1x1x256xf32>
      %3007 = stablehlo.reshape %arg1445 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3008 = stablehlo.multiply %3006, %3007 : tensor<1x1x1x256xf32>
      %3009 = stablehlo.broadcast_in_dim %3008, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3010 = stablehlo.multiply %3003, %3009 : tensor<256x14x14x256xf32>
      %3011 = stablehlo.reshape %arg1444 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3012 = stablehlo.broadcast_in_dim %3011, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3013 = stablehlo.add %3010, %3012 : tensor<256x14x14x256xf32>
      %3014 = stablehlo.convert %3013 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3015 = func.call @relu_175(%3014) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3016 = stablehlo.convert %arg1450 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3017 = stablehlo.convolution(%3015, %3016) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3018 = stablehlo.broadcast_in_dim %arg1916, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3019 = stablehlo.broadcast_in_dim %arg1917, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3020 = stablehlo.convert %3017 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3021 = stablehlo.broadcast_in_dim %3018, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3022 = stablehlo.subtract %3020, %3021 : tensor<256x14x14x1024xf32>
      %cst_156 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3023 = stablehlo.broadcast_in_dim %cst_156, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3024 = stablehlo.add %3019, %3023 : tensor<1x1x1x1024xf32>
      %3025 = stablehlo.rsqrt %3024 : tensor<1x1x1x1024xf32>
      %3026 = stablehlo.reshape %arg1447 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3027 = stablehlo.multiply %3025, %3026 : tensor<1x1x1x1024xf32>
      %3028 = stablehlo.broadcast_in_dim %3027, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3029 = stablehlo.multiply %3022, %3028 : tensor<256x14x14x1024xf32>
      %3030 = stablehlo.reshape %arg1446 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3031 = stablehlo.broadcast_in_dim %3030, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3032 = stablehlo.add %3029, %3031 : tensor<256x14x14x1024xf32>
      %3033 = stablehlo.convert %3032 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3034 = stablehlo.add %2977, %3033 : tensor<256x14x14x1024xf16>
      %3035 = func.call @relu_193(%3034) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3036 = stablehlo.convert %arg1457 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3037 = stablehlo.convolution(%3035, %3036) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3038 = stablehlo.broadcast_in_dim %arg1918, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3039 = stablehlo.broadcast_in_dim %arg1919, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3040 = stablehlo.convert %3037 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3041 = stablehlo.broadcast_in_dim %3038, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3042 = stablehlo.subtract %3040, %3041 : tensor<256x14x14x256xf32>
      %cst_157 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3043 = stablehlo.broadcast_in_dim %cst_157, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3044 = stablehlo.add %3039, %3043 : tensor<1x1x1x256xf32>
      %3045 = stablehlo.rsqrt %3044 : tensor<1x1x1x256xf32>
      %3046 = stablehlo.reshape %arg1452 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3047 = stablehlo.multiply %3045, %3046 : tensor<1x1x1x256xf32>
      %3048 = stablehlo.broadcast_in_dim %3047, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3049 = stablehlo.multiply %3042, %3048 : tensor<256x14x14x256xf32>
      %3050 = stablehlo.reshape %arg1451 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3051 = stablehlo.broadcast_in_dim %3050, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3052 = stablehlo.add %3049, %3051 : tensor<256x14x14x256xf32>
      %3053 = stablehlo.convert %3052 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3054 = func.call @relu_175(%3053) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3055 = stablehlo.convert %arg1458 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3056 = stablehlo.convolution(%3054, %3055) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3057 = stablehlo.broadcast_in_dim %arg1920, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3058 = stablehlo.broadcast_in_dim %arg1921, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3059 = stablehlo.convert %3056 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3060 = stablehlo.broadcast_in_dim %3057, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3061 = stablehlo.subtract %3059, %3060 : tensor<256x14x14x256xf32>
      %cst_158 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3062 = stablehlo.broadcast_in_dim %cst_158, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3063 = stablehlo.add %3058, %3062 : tensor<1x1x1x256xf32>
      %3064 = stablehlo.rsqrt %3063 : tensor<1x1x1x256xf32>
      %3065 = stablehlo.reshape %arg1454 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3066 = stablehlo.multiply %3064, %3065 : tensor<1x1x1x256xf32>
      %3067 = stablehlo.broadcast_in_dim %3066, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3068 = stablehlo.multiply %3061, %3067 : tensor<256x14x14x256xf32>
      %3069 = stablehlo.reshape %arg1453 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3070 = stablehlo.broadcast_in_dim %3069, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3071 = stablehlo.add %3068, %3070 : tensor<256x14x14x256xf32>
      %3072 = stablehlo.convert %3071 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3073 = func.call @relu_175(%3072) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3074 = stablehlo.convert %arg1459 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3075 = stablehlo.convolution(%3073, %3074) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3076 = stablehlo.broadcast_in_dim %arg1922, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3077 = stablehlo.broadcast_in_dim %arg1923, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3078 = stablehlo.convert %3075 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3079 = stablehlo.broadcast_in_dim %3076, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3080 = stablehlo.subtract %3078, %3079 : tensor<256x14x14x1024xf32>
      %cst_159 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3081 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3082 = stablehlo.add %3077, %3081 : tensor<1x1x1x1024xf32>
      %3083 = stablehlo.rsqrt %3082 : tensor<1x1x1x1024xf32>
      %3084 = stablehlo.reshape %arg1456 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3085 = stablehlo.multiply %3083, %3084 : tensor<1x1x1x1024xf32>
      %3086 = stablehlo.broadcast_in_dim %3085, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3087 = stablehlo.multiply %3080, %3086 : tensor<256x14x14x1024xf32>
      %3088 = stablehlo.reshape %arg1455 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3089 = stablehlo.broadcast_in_dim %3088, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3090 = stablehlo.add %3087, %3089 : tensor<256x14x14x1024xf32>
      %3091 = stablehlo.convert %3090 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3092 = stablehlo.add %3035, %3091 : tensor<256x14x14x1024xf16>
      %3093 = func.call @relu_193(%3092) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3094 = stablehlo.convert %arg1466 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3095 = stablehlo.convolution(%3093, %3094) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3096 = stablehlo.broadcast_in_dim %arg1924, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3097 = stablehlo.broadcast_in_dim %arg1925, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3098 = stablehlo.convert %3095 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3099 = stablehlo.broadcast_in_dim %3096, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3100 = stablehlo.subtract %3098, %3099 : tensor<256x14x14x256xf32>
      %cst_160 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3101 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3102 = stablehlo.add %3097, %3101 : tensor<1x1x1x256xf32>
      %3103 = stablehlo.rsqrt %3102 : tensor<1x1x1x256xf32>
      %3104 = stablehlo.reshape %arg1461 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3105 = stablehlo.multiply %3103, %3104 : tensor<1x1x1x256xf32>
      %3106 = stablehlo.broadcast_in_dim %3105, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3107 = stablehlo.multiply %3100, %3106 : tensor<256x14x14x256xf32>
      %3108 = stablehlo.reshape %arg1460 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3109 = stablehlo.broadcast_in_dim %3108, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3110 = stablehlo.add %3107, %3109 : tensor<256x14x14x256xf32>
      %3111 = stablehlo.convert %3110 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3112 = func.call @relu_175(%3111) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3113 = stablehlo.convert %arg1467 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3114 = stablehlo.convolution(%3112, %3113) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3115 = stablehlo.broadcast_in_dim %arg1926, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3116 = stablehlo.broadcast_in_dim %arg1927, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3117 = stablehlo.convert %3114 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3118 = stablehlo.broadcast_in_dim %3115, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3119 = stablehlo.subtract %3117, %3118 : tensor<256x14x14x256xf32>
      %cst_161 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3120 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3121 = stablehlo.add %3116, %3120 : tensor<1x1x1x256xf32>
      %3122 = stablehlo.rsqrt %3121 : tensor<1x1x1x256xf32>
      %3123 = stablehlo.reshape %arg1463 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3124 = stablehlo.multiply %3122, %3123 : tensor<1x1x1x256xf32>
      %3125 = stablehlo.broadcast_in_dim %3124, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3126 = stablehlo.multiply %3119, %3125 : tensor<256x14x14x256xf32>
      %3127 = stablehlo.reshape %arg1462 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3128 = stablehlo.broadcast_in_dim %3127, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3129 = stablehlo.add %3126, %3128 : tensor<256x14x14x256xf32>
      %3130 = stablehlo.convert %3129 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3131 = func.call @relu_175(%3130) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3132 = stablehlo.convert %arg1468 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3133 = stablehlo.convolution(%3131, %3132) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3134 = stablehlo.broadcast_in_dim %arg1928, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3135 = stablehlo.broadcast_in_dim %arg1929, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3136 = stablehlo.convert %3133 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3137 = stablehlo.broadcast_in_dim %3134, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3138 = stablehlo.subtract %3136, %3137 : tensor<256x14x14x1024xf32>
      %cst_162 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3139 = stablehlo.broadcast_in_dim %cst_162, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3140 = stablehlo.add %3135, %3139 : tensor<1x1x1x1024xf32>
      %3141 = stablehlo.rsqrt %3140 : tensor<1x1x1x1024xf32>
      %3142 = stablehlo.reshape %arg1465 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3143 = stablehlo.multiply %3141, %3142 : tensor<1x1x1x1024xf32>
      %3144 = stablehlo.broadcast_in_dim %3143, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3145 = stablehlo.multiply %3138, %3144 : tensor<256x14x14x1024xf32>
      %3146 = stablehlo.reshape %arg1464 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3147 = stablehlo.broadcast_in_dim %3146, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3148 = stablehlo.add %3145, %3147 : tensor<256x14x14x1024xf32>
      %3149 = stablehlo.convert %3148 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3150 = stablehlo.add %3093, %3149 : tensor<256x14x14x1024xf16>
      %3151 = func.call @relu_193(%3150) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3152 = stablehlo.convert %arg1475 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3153 = stablehlo.convolution(%3151, %3152) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3154 = stablehlo.broadcast_in_dim %arg1930, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3155 = stablehlo.broadcast_in_dim %arg1931, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3156 = stablehlo.convert %3153 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3157 = stablehlo.broadcast_in_dim %3154, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3158 = stablehlo.subtract %3156, %3157 : tensor<256x14x14x256xf32>
      %cst_163 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3159 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3160 = stablehlo.add %3155, %3159 : tensor<1x1x1x256xf32>
      %3161 = stablehlo.rsqrt %3160 : tensor<1x1x1x256xf32>
      %3162 = stablehlo.reshape %arg1470 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3163 = stablehlo.multiply %3161, %3162 : tensor<1x1x1x256xf32>
      %3164 = stablehlo.broadcast_in_dim %3163, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3165 = stablehlo.multiply %3158, %3164 : tensor<256x14x14x256xf32>
      %3166 = stablehlo.reshape %arg1469 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3167 = stablehlo.broadcast_in_dim %3166, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3168 = stablehlo.add %3165, %3167 : tensor<256x14x14x256xf32>
      %3169 = stablehlo.convert %3168 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3170 = func.call @relu_175(%3169) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3171 = stablehlo.convert %arg1476 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3172 = stablehlo.convolution(%3170, %3171) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3173 = stablehlo.broadcast_in_dim %arg1932, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3174 = stablehlo.broadcast_in_dim %arg1933, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3175 = stablehlo.convert %3172 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3176 = stablehlo.broadcast_in_dim %3173, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3177 = stablehlo.subtract %3175, %3176 : tensor<256x14x14x256xf32>
      %cst_164 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3178 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3179 = stablehlo.add %3174, %3178 : tensor<1x1x1x256xf32>
      %3180 = stablehlo.rsqrt %3179 : tensor<1x1x1x256xf32>
      %3181 = stablehlo.reshape %arg1472 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3182 = stablehlo.multiply %3180, %3181 : tensor<1x1x1x256xf32>
      %3183 = stablehlo.broadcast_in_dim %3182, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3184 = stablehlo.multiply %3177, %3183 : tensor<256x14x14x256xf32>
      %3185 = stablehlo.reshape %arg1471 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3186 = stablehlo.broadcast_in_dim %3185, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3187 = stablehlo.add %3184, %3186 : tensor<256x14x14x256xf32>
      %3188 = stablehlo.convert %3187 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3189 = func.call @relu_175(%3188) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3190 = stablehlo.convert %arg1477 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3191 = stablehlo.convolution(%3189, %3190) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3192 = stablehlo.broadcast_in_dim %arg1934, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3193 = stablehlo.broadcast_in_dim %arg1935, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3194 = stablehlo.convert %3191 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3195 = stablehlo.broadcast_in_dim %3192, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3196 = stablehlo.subtract %3194, %3195 : tensor<256x14x14x1024xf32>
      %cst_165 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3197 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3198 = stablehlo.add %3193, %3197 : tensor<1x1x1x1024xf32>
      %3199 = stablehlo.rsqrt %3198 : tensor<1x1x1x1024xf32>
      %3200 = stablehlo.reshape %arg1474 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3201 = stablehlo.multiply %3199, %3200 : tensor<1x1x1x1024xf32>
      %3202 = stablehlo.broadcast_in_dim %3201, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3203 = stablehlo.multiply %3196, %3202 : tensor<256x14x14x1024xf32>
      %3204 = stablehlo.reshape %arg1473 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3205 = stablehlo.broadcast_in_dim %3204, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3206 = stablehlo.add %3203, %3205 : tensor<256x14x14x1024xf32>
      %3207 = stablehlo.convert %3206 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3208 = stablehlo.add %3151, %3207 : tensor<256x14x14x1024xf16>
      %3209 = func.call @relu_193(%3208) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3210 = stablehlo.convert %arg1484 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3211 = stablehlo.convolution(%3209, %3210) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3212 = stablehlo.broadcast_in_dim %arg1936, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3213 = stablehlo.broadcast_in_dim %arg1937, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3214 = stablehlo.convert %3211 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3215 = stablehlo.broadcast_in_dim %3212, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3216 = stablehlo.subtract %3214, %3215 : tensor<256x14x14x256xf32>
      %cst_166 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3217 = stablehlo.broadcast_in_dim %cst_166, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3218 = stablehlo.add %3213, %3217 : tensor<1x1x1x256xf32>
      %3219 = stablehlo.rsqrt %3218 : tensor<1x1x1x256xf32>
      %3220 = stablehlo.reshape %arg1479 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3221 = stablehlo.multiply %3219, %3220 : tensor<1x1x1x256xf32>
      %3222 = stablehlo.broadcast_in_dim %3221, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3223 = stablehlo.multiply %3216, %3222 : tensor<256x14x14x256xf32>
      %3224 = stablehlo.reshape %arg1478 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3225 = stablehlo.broadcast_in_dim %3224, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3226 = stablehlo.add %3223, %3225 : tensor<256x14x14x256xf32>
      %3227 = stablehlo.convert %3226 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3228 = func.call @relu_175(%3227) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3229 = stablehlo.convert %arg1485 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3230 = stablehlo.convolution(%3228, %3229) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3231 = stablehlo.broadcast_in_dim %arg1938, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3232 = stablehlo.broadcast_in_dim %arg1939, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3233 = stablehlo.convert %3230 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3234 = stablehlo.broadcast_in_dim %3231, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3235 = stablehlo.subtract %3233, %3234 : tensor<256x14x14x256xf32>
      %cst_167 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3236 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3237 = stablehlo.add %3232, %3236 : tensor<1x1x1x256xf32>
      %3238 = stablehlo.rsqrt %3237 : tensor<1x1x1x256xf32>
      %3239 = stablehlo.reshape %arg1481 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3240 = stablehlo.multiply %3238, %3239 : tensor<1x1x1x256xf32>
      %3241 = stablehlo.broadcast_in_dim %3240, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3242 = stablehlo.multiply %3235, %3241 : tensor<256x14x14x256xf32>
      %3243 = stablehlo.reshape %arg1480 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3244 = stablehlo.broadcast_in_dim %3243, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3245 = stablehlo.add %3242, %3244 : tensor<256x14x14x256xf32>
      %3246 = stablehlo.convert %3245 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3247 = func.call @relu_175(%3246) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3248 = stablehlo.convert %arg1486 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3249 = stablehlo.convolution(%3247, %3248) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3250 = stablehlo.broadcast_in_dim %arg1940, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3251 = stablehlo.broadcast_in_dim %arg1941, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3252 = stablehlo.convert %3249 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3253 = stablehlo.broadcast_in_dim %3250, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3254 = stablehlo.subtract %3252, %3253 : tensor<256x14x14x1024xf32>
      %cst_168 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3255 = stablehlo.broadcast_in_dim %cst_168, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3256 = stablehlo.add %3251, %3255 : tensor<1x1x1x1024xf32>
      %3257 = stablehlo.rsqrt %3256 : tensor<1x1x1x1024xf32>
      %3258 = stablehlo.reshape %arg1483 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3259 = stablehlo.multiply %3257, %3258 : tensor<1x1x1x1024xf32>
      %3260 = stablehlo.broadcast_in_dim %3259, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3261 = stablehlo.multiply %3254, %3260 : tensor<256x14x14x1024xf32>
      %3262 = stablehlo.reshape %arg1482 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3263 = stablehlo.broadcast_in_dim %3262, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3264 = stablehlo.add %3261, %3263 : tensor<256x14x14x1024xf32>
      %3265 = stablehlo.convert %3264 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3266 = stablehlo.add %3209, %3265 : tensor<256x14x14x1024xf16>
      %3267 = func.call @relu_193(%3266) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3268 = stablehlo.convert %arg1493 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3269 = stablehlo.convolution(%3267, %3268) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3270 = stablehlo.broadcast_in_dim %arg1942, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3271 = stablehlo.broadcast_in_dim %arg1943, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3272 = stablehlo.convert %3269 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3273 = stablehlo.broadcast_in_dim %3270, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3274 = stablehlo.subtract %3272, %3273 : tensor<256x14x14x256xf32>
      %cst_169 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3275 = stablehlo.broadcast_in_dim %cst_169, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3276 = stablehlo.add %3271, %3275 : tensor<1x1x1x256xf32>
      %3277 = stablehlo.rsqrt %3276 : tensor<1x1x1x256xf32>
      %3278 = stablehlo.reshape %arg1488 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3279 = stablehlo.multiply %3277, %3278 : tensor<1x1x1x256xf32>
      %3280 = stablehlo.broadcast_in_dim %3279, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3281 = stablehlo.multiply %3274, %3280 : tensor<256x14x14x256xf32>
      %3282 = stablehlo.reshape %arg1487 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3283 = stablehlo.broadcast_in_dim %3282, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3284 = stablehlo.add %3281, %3283 : tensor<256x14x14x256xf32>
      %3285 = stablehlo.convert %3284 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3286 = func.call @relu_175(%3285) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3287 = stablehlo.convert %arg1494 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3288 = stablehlo.convolution(%3286, %3287) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3289 = stablehlo.broadcast_in_dim %arg1944, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3290 = stablehlo.broadcast_in_dim %arg1945, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3291 = stablehlo.convert %3288 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3292 = stablehlo.broadcast_in_dim %3289, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3293 = stablehlo.subtract %3291, %3292 : tensor<256x14x14x256xf32>
      %cst_170 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3294 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3295 = stablehlo.add %3290, %3294 : tensor<1x1x1x256xf32>
      %3296 = stablehlo.rsqrt %3295 : tensor<1x1x1x256xf32>
      %3297 = stablehlo.reshape %arg1490 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3298 = stablehlo.multiply %3296, %3297 : tensor<1x1x1x256xf32>
      %3299 = stablehlo.broadcast_in_dim %3298, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3300 = stablehlo.multiply %3293, %3299 : tensor<256x14x14x256xf32>
      %3301 = stablehlo.reshape %arg1489 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3302 = stablehlo.broadcast_in_dim %3301, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3303 = stablehlo.add %3300, %3302 : tensor<256x14x14x256xf32>
      %3304 = stablehlo.convert %3303 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3305 = func.call @relu_175(%3304) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3306 = stablehlo.convert %arg1495 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3307 = stablehlo.convolution(%3305, %3306) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3308 = stablehlo.broadcast_in_dim %arg1946, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3309 = stablehlo.broadcast_in_dim %arg1947, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3310 = stablehlo.convert %3307 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3311 = stablehlo.broadcast_in_dim %3308, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3312 = stablehlo.subtract %3310, %3311 : tensor<256x14x14x1024xf32>
      %cst_171 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3313 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3314 = stablehlo.add %3309, %3313 : tensor<1x1x1x1024xf32>
      %3315 = stablehlo.rsqrt %3314 : tensor<1x1x1x1024xf32>
      %3316 = stablehlo.reshape %arg1492 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3317 = stablehlo.multiply %3315, %3316 : tensor<1x1x1x1024xf32>
      %3318 = stablehlo.broadcast_in_dim %3317, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3319 = stablehlo.multiply %3312, %3318 : tensor<256x14x14x1024xf32>
      %3320 = stablehlo.reshape %arg1491 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3321 = stablehlo.broadcast_in_dim %3320, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3322 = stablehlo.add %3319, %3321 : tensor<256x14x14x1024xf32>
      %3323 = stablehlo.convert %3322 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3324 = stablehlo.add %3267, %3323 : tensor<256x14x14x1024xf16>
      %3325 = func.call @relu_193(%3324) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3326 = stablehlo.convert %arg1502 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3327 = stablehlo.convolution(%3325, %3326) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3328 = stablehlo.broadcast_in_dim %arg1948, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3329 = stablehlo.broadcast_in_dim %arg1949, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3330 = stablehlo.convert %3327 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3331 = stablehlo.broadcast_in_dim %3328, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3332 = stablehlo.subtract %3330, %3331 : tensor<256x14x14x256xf32>
      %cst_172 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3333 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3334 = stablehlo.add %3329, %3333 : tensor<1x1x1x256xf32>
      %3335 = stablehlo.rsqrt %3334 : tensor<1x1x1x256xf32>
      %3336 = stablehlo.reshape %arg1497 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3337 = stablehlo.multiply %3335, %3336 : tensor<1x1x1x256xf32>
      %3338 = stablehlo.broadcast_in_dim %3337, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3339 = stablehlo.multiply %3332, %3338 : tensor<256x14x14x256xf32>
      %3340 = stablehlo.reshape %arg1496 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3341 = stablehlo.broadcast_in_dim %3340, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3342 = stablehlo.add %3339, %3341 : tensor<256x14x14x256xf32>
      %3343 = stablehlo.convert %3342 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3344 = func.call @relu_175(%3343) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3345 = stablehlo.convert %arg1503 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3346 = stablehlo.convolution(%3344, %3345) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3347 = stablehlo.broadcast_in_dim %arg1950, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3348 = stablehlo.broadcast_in_dim %arg1951, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3349 = stablehlo.convert %3346 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3350 = stablehlo.broadcast_in_dim %3347, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3351 = stablehlo.subtract %3349, %3350 : tensor<256x14x14x256xf32>
      %cst_173 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3352 = stablehlo.broadcast_in_dim %cst_173, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3353 = stablehlo.add %3348, %3352 : tensor<1x1x1x256xf32>
      %3354 = stablehlo.rsqrt %3353 : tensor<1x1x1x256xf32>
      %3355 = stablehlo.reshape %arg1499 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3356 = stablehlo.multiply %3354, %3355 : tensor<1x1x1x256xf32>
      %3357 = stablehlo.broadcast_in_dim %3356, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3358 = stablehlo.multiply %3351, %3357 : tensor<256x14x14x256xf32>
      %3359 = stablehlo.reshape %arg1498 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3360 = stablehlo.broadcast_in_dim %3359, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3361 = stablehlo.add %3358, %3360 : tensor<256x14x14x256xf32>
      %3362 = stablehlo.convert %3361 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3363 = func.call @relu_175(%3362) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3364 = stablehlo.convert %arg1504 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3365 = stablehlo.convolution(%3363, %3364) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3366 = stablehlo.broadcast_in_dim %arg1952, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3367 = stablehlo.broadcast_in_dim %arg1953, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3368 = stablehlo.convert %3365 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3369 = stablehlo.broadcast_in_dim %3366, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3370 = stablehlo.subtract %3368, %3369 : tensor<256x14x14x1024xf32>
      %cst_174 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3371 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3372 = stablehlo.add %3367, %3371 : tensor<1x1x1x1024xf32>
      %3373 = stablehlo.rsqrt %3372 : tensor<1x1x1x1024xf32>
      %3374 = stablehlo.reshape %arg1501 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3375 = stablehlo.multiply %3373, %3374 : tensor<1x1x1x1024xf32>
      %3376 = stablehlo.broadcast_in_dim %3375, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3377 = stablehlo.multiply %3370, %3376 : tensor<256x14x14x1024xf32>
      %3378 = stablehlo.reshape %arg1500 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3379 = stablehlo.broadcast_in_dim %3378, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3380 = stablehlo.add %3377, %3379 : tensor<256x14x14x1024xf32>
      %3381 = stablehlo.convert %3380 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3382 = stablehlo.add %3325, %3381 : tensor<256x14x14x1024xf16>
      %3383 = func.call @relu_193(%3382) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3384 = stablehlo.convert %arg1511 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3385 = stablehlo.convolution(%3383, %3384) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3386 = stablehlo.broadcast_in_dim %arg1954, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3387 = stablehlo.broadcast_in_dim %arg1955, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3388 = stablehlo.convert %3385 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3389 = stablehlo.broadcast_in_dim %3386, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3390 = stablehlo.subtract %3388, %3389 : tensor<256x14x14x256xf32>
      %cst_175 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3391 = stablehlo.broadcast_in_dim %cst_175, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3392 = stablehlo.add %3387, %3391 : tensor<1x1x1x256xf32>
      %3393 = stablehlo.rsqrt %3392 : tensor<1x1x1x256xf32>
      %3394 = stablehlo.reshape %arg1506 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3395 = stablehlo.multiply %3393, %3394 : tensor<1x1x1x256xf32>
      %3396 = stablehlo.broadcast_in_dim %3395, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3397 = stablehlo.multiply %3390, %3396 : tensor<256x14x14x256xf32>
      %3398 = stablehlo.reshape %arg1505 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3399 = stablehlo.broadcast_in_dim %3398, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3400 = stablehlo.add %3397, %3399 : tensor<256x14x14x256xf32>
      %3401 = stablehlo.convert %3400 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3402 = func.call @relu_175(%3401) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3403 = stablehlo.convert %arg1512 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3404 = stablehlo.convolution(%3402, %3403) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3405 = stablehlo.broadcast_in_dim %arg1956, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3406 = stablehlo.broadcast_in_dim %arg1957, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3407 = stablehlo.convert %3404 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3408 = stablehlo.broadcast_in_dim %3405, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3409 = stablehlo.subtract %3407, %3408 : tensor<256x14x14x256xf32>
      %cst_176 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3410 = stablehlo.broadcast_in_dim %cst_176, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3411 = stablehlo.add %3406, %3410 : tensor<1x1x1x256xf32>
      %3412 = stablehlo.rsqrt %3411 : tensor<1x1x1x256xf32>
      %3413 = stablehlo.reshape %arg1508 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3414 = stablehlo.multiply %3412, %3413 : tensor<1x1x1x256xf32>
      %3415 = stablehlo.broadcast_in_dim %3414, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3416 = stablehlo.multiply %3409, %3415 : tensor<256x14x14x256xf32>
      %3417 = stablehlo.reshape %arg1507 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3418 = stablehlo.broadcast_in_dim %3417, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3419 = stablehlo.add %3416, %3418 : tensor<256x14x14x256xf32>
      %3420 = stablehlo.convert %3419 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3421 = func.call @relu_175(%3420) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3422 = stablehlo.convert %arg1513 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3423 = stablehlo.convolution(%3421, %3422) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3424 = stablehlo.broadcast_in_dim %arg1958, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3425 = stablehlo.broadcast_in_dim %arg1959, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3426 = stablehlo.convert %3423 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3427 = stablehlo.broadcast_in_dim %3424, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3428 = stablehlo.subtract %3426, %3427 : tensor<256x14x14x1024xf32>
      %cst_177 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3429 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3430 = stablehlo.add %3425, %3429 : tensor<1x1x1x1024xf32>
      %3431 = stablehlo.rsqrt %3430 : tensor<1x1x1x1024xf32>
      %3432 = stablehlo.reshape %arg1510 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3433 = stablehlo.multiply %3431, %3432 : tensor<1x1x1x1024xf32>
      %3434 = stablehlo.broadcast_in_dim %3433, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3435 = stablehlo.multiply %3428, %3434 : tensor<256x14x14x1024xf32>
      %3436 = stablehlo.reshape %arg1509 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3437 = stablehlo.broadcast_in_dim %3436, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3438 = stablehlo.add %3435, %3437 : tensor<256x14x14x1024xf32>
      %3439 = stablehlo.convert %3438 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3440 = stablehlo.add %3383, %3439 : tensor<256x14x14x1024xf16>
      %3441 = func.call @relu_193(%3440) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3442 = stablehlo.convert %arg1520 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3443 = stablehlo.convolution(%3441, %3442) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3444 = stablehlo.broadcast_in_dim %arg1960, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3445 = stablehlo.broadcast_in_dim %arg1961, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3446 = stablehlo.convert %3443 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3447 = stablehlo.broadcast_in_dim %3444, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3448 = stablehlo.subtract %3446, %3447 : tensor<256x14x14x256xf32>
      %cst_178 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3449 = stablehlo.broadcast_in_dim %cst_178, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3450 = stablehlo.add %3445, %3449 : tensor<1x1x1x256xf32>
      %3451 = stablehlo.rsqrt %3450 : tensor<1x1x1x256xf32>
      %3452 = stablehlo.reshape %arg1515 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3453 = stablehlo.multiply %3451, %3452 : tensor<1x1x1x256xf32>
      %3454 = stablehlo.broadcast_in_dim %3453, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3455 = stablehlo.multiply %3448, %3454 : tensor<256x14x14x256xf32>
      %3456 = stablehlo.reshape %arg1514 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3457 = stablehlo.broadcast_in_dim %3456, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3458 = stablehlo.add %3455, %3457 : tensor<256x14x14x256xf32>
      %3459 = stablehlo.convert %3458 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3460 = func.call @relu_175(%3459) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3461 = stablehlo.convert %arg1521 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3462 = stablehlo.convolution(%3460, %3461) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3463 = stablehlo.broadcast_in_dim %arg1962, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3464 = stablehlo.broadcast_in_dim %arg1963, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3465 = stablehlo.convert %3462 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3466 = stablehlo.broadcast_in_dim %3463, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3467 = stablehlo.subtract %3465, %3466 : tensor<256x14x14x256xf32>
      %cst_179 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3468 = stablehlo.broadcast_in_dim %cst_179, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3469 = stablehlo.add %3464, %3468 : tensor<1x1x1x256xf32>
      %3470 = stablehlo.rsqrt %3469 : tensor<1x1x1x256xf32>
      %3471 = stablehlo.reshape %arg1517 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3472 = stablehlo.multiply %3470, %3471 : tensor<1x1x1x256xf32>
      %3473 = stablehlo.broadcast_in_dim %3472, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3474 = stablehlo.multiply %3467, %3473 : tensor<256x14x14x256xf32>
      %3475 = stablehlo.reshape %arg1516 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3476 = stablehlo.broadcast_in_dim %3475, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3477 = stablehlo.add %3474, %3476 : tensor<256x14x14x256xf32>
      %3478 = stablehlo.convert %3477 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3479 = func.call @relu_175(%3478) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3480 = stablehlo.convert %arg1522 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3481 = stablehlo.convolution(%3479, %3480) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3482 = stablehlo.broadcast_in_dim %arg1964, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3483 = stablehlo.broadcast_in_dim %arg1965, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3484 = stablehlo.convert %3481 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3485 = stablehlo.broadcast_in_dim %3482, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3486 = stablehlo.subtract %3484, %3485 : tensor<256x14x14x1024xf32>
      %cst_180 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3487 = stablehlo.broadcast_in_dim %cst_180, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3488 = stablehlo.add %3483, %3487 : tensor<1x1x1x1024xf32>
      %3489 = stablehlo.rsqrt %3488 : tensor<1x1x1x1024xf32>
      %3490 = stablehlo.reshape %arg1519 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3491 = stablehlo.multiply %3489, %3490 : tensor<1x1x1x1024xf32>
      %3492 = stablehlo.broadcast_in_dim %3491, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3493 = stablehlo.multiply %3486, %3492 : tensor<256x14x14x1024xf32>
      %3494 = stablehlo.reshape %arg1518 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3495 = stablehlo.broadcast_in_dim %3494, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3496 = stablehlo.add %3493, %3495 : tensor<256x14x14x1024xf32>
      %3497 = stablehlo.convert %3496 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3498 = stablehlo.add %3441, %3497 : tensor<256x14x14x1024xf16>
      %3499 = func.call @relu_193(%3498) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3500 = stablehlo.convert %arg1529 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3501 = stablehlo.convolution(%3499, %3500) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3502 = stablehlo.broadcast_in_dim %arg1966, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3503 = stablehlo.broadcast_in_dim %arg1967, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3504 = stablehlo.convert %3501 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3505 = stablehlo.broadcast_in_dim %3502, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3506 = stablehlo.subtract %3504, %3505 : tensor<256x14x14x256xf32>
      %cst_181 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3507 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3508 = stablehlo.add %3503, %3507 : tensor<1x1x1x256xf32>
      %3509 = stablehlo.rsqrt %3508 : tensor<1x1x1x256xf32>
      %3510 = stablehlo.reshape %arg1524 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3511 = stablehlo.multiply %3509, %3510 : tensor<1x1x1x256xf32>
      %3512 = stablehlo.broadcast_in_dim %3511, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3513 = stablehlo.multiply %3506, %3512 : tensor<256x14x14x256xf32>
      %3514 = stablehlo.reshape %arg1523 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3515 = stablehlo.broadcast_in_dim %3514, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3516 = stablehlo.add %3513, %3515 : tensor<256x14x14x256xf32>
      %3517 = stablehlo.convert %3516 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3518 = func.call @relu_175(%3517) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3519 = stablehlo.convert %arg1530 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3520 = stablehlo.convolution(%3518, %3519) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3521 = stablehlo.broadcast_in_dim %arg1968, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3522 = stablehlo.broadcast_in_dim %arg1969, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3523 = stablehlo.convert %3520 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3524 = stablehlo.broadcast_in_dim %3521, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3525 = stablehlo.subtract %3523, %3524 : tensor<256x14x14x256xf32>
      %cst_182 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3526 = stablehlo.broadcast_in_dim %cst_182, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3527 = stablehlo.add %3522, %3526 : tensor<1x1x1x256xf32>
      %3528 = stablehlo.rsqrt %3527 : tensor<1x1x1x256xf32>
      %3529 = stablehlo.reshape %arg1526 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3530 = stablehlo.multiply %3528, %3529 : tensor<1x1x1x256xf32>
      %3531 = stablehlo.broadcast_in_dim %3530, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3532 = stablehlo.multiply %3525, %3531 : tensor<256x14x14x256xf32>
      %3533 = stablehlo.reshape %arg1525 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3534 = stablehlo.broadcast_in_dim %3533, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3535 = stablehlo.add %3532, %3534 : tensor<256x14x14x256xf32>
      %3536 = stablehlo.convert %3535 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3537 = func.call @relu_175(%3536) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3538 = stablehlo.convert %arg1531 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3539 = stablehlo.convolution(%3537, %3538) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3540 = stablehlo.broadcast_in_dim %arg1970, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3541 = stablehlo.broadcast_in_dim %arg1971, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3542 = stablehlo.convert %3539 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3543 = stablehlo.broadcast_in_dim %3540, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3544 = stablehlo.subtract %3542, %3543 : tensor<256x14x14x1024xf32>
      %cst_183 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3545 = stablehlo.broadcast_in_dim %cst_183, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3546 = stablehlo.add %3541, %3545 : tensor<1x1x1x1024xf32>
      %3547 = stablehlo.rsqrt %3546 : tensor<1x1x1x1024xf32>
      %3548 = stablehlo.reshape %arg1528 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3549 = stablehlo.multiply %3547, %3548 : tensor<1x1x1x1024xf32>
      %3550 = stablehlo.broadcast_in_dim %3549, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3551 = stablehlo.multiply %3544, %3550 : tensor<256x14x14x1024xf32>
      %3552 = stablehlo.reshape %arg1527 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3553 = stablehlo.broadcast_in_dim %3552, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3554 = stablehlo.add %3551, %3553 : tensor<256x14x14x1024xf32>
      %3555 = stablehlo.convert %3554 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3556 = stablehlo.add %3499, %3555 : tensor<256x14x14x1024xf16>
      %3557 = func.call @relu_193(%3556) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3558 = stablehlo.convert %arg1547 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3559 = stablehlo.convolution(%3557, %3558) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3560 = stablehlo.broadcast_in_dim %arg1978, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3561 = stablehlo.broadcast_in_dim %arg1979, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3562 = stablehlo.convert %3559 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3563 = stablehlo.broadcast_in_dim %3560, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3564 = stablehlo.subtract %3562, %3563 : tensor<256x14x14x256xf32>
      %cst_184 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3565 = stablehlo.broadcast_in_dim %cst_184, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3566 = stablehlo.add %3561, %3565 : tensor<1x1x1x256xf32>
      %3567 = stablehlo.rsqrt %3566 : tensor<1x1x1x256xf32>
      %3568 = stablehlo.reshape %arg1542 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3569 = stablehlo.multiply %3567, %3568 : tensor<1x1x1x256xf32>
      %3570 = stablehlo.broadcast_in_dim %3569, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3571 = stablehlo.multiply %3564, %3570 : tensor<256x14x14x256xf32>
      %3572 = stablehlo.reshape %arg1541 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3573 = stablehlo.broadcast_in_dim %3572, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3574 = stablehlo.add %3571, %3573 : tensor<256x14x14x256xf32>
      %3575 = stablehlo.convert %3574 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3576 = func.call @relu_175(%3575) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3577 = stablehlo.convert %arg1548 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3578 = stablehlo.convolution(%3576, %3577) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3579 = stablehlo.broadcast_in_dim %arg1980, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3580 = stablehlo.broadcast_in_dim %arg1981, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3581 = stablehlo.convert %3578 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3582 = stablehlo.broadcast_in_dim %3579, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3583 = stablehlo.subtract %3581, %3582 : tensor<256x14x14x256xf32>
      %cst_185 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3584 = stablehlo.broadcast_in_dim %cst_185, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3585 = stablehlo.add %3580, %3584 : tensor<1x1x1x256xf32>
      %3586 = stablehlo.rsqrt %3585 : tensor<1x1x1x256xf32>
      %3587 = stablehlo.reshape %arg1544 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3588 = stablehlo.multiply %3586, %3587 : tensor<1x1x1x256xf32>
      %3589 = stablehlo.broadcast_in_dim %3588, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3590 = stablehlo.multiply %3583, %3589 : tensor<256x14x14x256xf32>
      %3591 = stablehlo.reshape %arg1543 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3592 = stablehlo.broadcast_in_dim %3591, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3593 = stablehlo.add %3590, %3592 : tensor<256x14x14x256xf32>
      %3594 = stablehlo.convert %3593 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3595 = func.call @relu_175(%3594) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3596 = stablehlo.convert %arg1549 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3597 = stablehlo.convolution(%3595, %3596) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3598 = stablehlo.broadcast_in_dim %arg1982, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3599 = stablehlo.broadcast_in_dim %arg1983, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3600 = stablehlo.convert %3597 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3601 = stablehlo.broadcast_in_dim %3598, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3602 = stablehlo.subtract %3600, %3601 : tensor<256x14x14x1024xf32>
      %cst_186 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3603 = stablehlo.broadcast_in_dim %cst_186, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3604 = stablehlo.add %3599, %3603 : tensor<1x1x1x1024xf32>
      %3605 = stablehlo.rsqrt %3604 : tensor<1x1x1x1024xf32>
      %3606 = stablehlo.reshape %arg1546 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3607 = stablehlo.multiply %3605, %3606 : tensor<1x1x1x1024xf32>
      %3608 = stablehlo.broadcast_in_dim %3607, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3609 = stablehlo.multiply %3602, %3608 : tensor<256x14x14x1024xf32>
      %3610 = stablehlo.reshape %arg1545 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3611 = stablehlo.broadcast_in_dim %3610, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3612 = stablehlo.add %3609, %3611 : tensor<256x14x14x1024xf32>
      %3613 = stablehlo.convert %3612 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3614 = stablehlo.add %3557, %3613 : tensor<256x14x14x1024xf16>
      %3615 = func.call @relu_193(%3614) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3616 = stablehlo.convert %arg1556 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3617 = stablehlo.convolution(%3615, %3616) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3618 = stablehlo.broadcast_in_dim %arg1984, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3619 = stablehlo.broadcast_in_dim %arg1985, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3620 = stablehlo.convert %3617 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3621 = stablehlo.broadcast_in_dim %3618, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3622 = stablehlo.subtract %3620, %3621 : tensor<256x14x14x256xf32>
      %cst_187 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3623 = stablehlo.broadcast_in_dim %cst_187, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3624 = stablehlo.add %3619, %3623 : tensor<1x1x1x256xf32>
      %3625 = stablehlo.rsqrt %3624 : tensor<1x1x1x256xf32>
      %3626 = stablehlo.reshape %arg1551 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3627 = stablehlo.multiply %3625, %3626 : tensor<1x1x1x256xf32>
      %3628 = stablehlo.broadcast_in_dim %3627, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3629 = stablehlo.multiply %3622, %3628 : tensor<256x14x14x256xf32>
      %3630 = stablehlo.reshape %arg1550 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3631 = stablehlo.broadcast_in_dim %3630, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3632 = stablehlo.add %3629, %3631 : tensor<256x14x14x256xf32>
      %3633 = stablehlo.convert %3632 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3634 = func.call @relu_175(%3633) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3635 = stablehlo.convert %arg1557 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3636 = stablehlo.convolution(%3634, %3635) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3637 = stablehlo.broadcast_in_dim %arg1986, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3638 = stablehlo.broadcast_in_dim %arg1987, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3639 = stablehlo.convert %3636 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3640 = stablehlo.broadcast_in_dim %3637, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3641 = stablehlo.subtract %3639, %3640 : tensor<256x14x14x256xf32>
      %cst_188 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3642 = stablehlo.broadcast_in_dim %cst_188, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3643 = stablehlo.add %3638, %3642 : tensor<1x1x1x256xf32>
      %3644 = stablehlo.rsqrt %3643 : tensor<1x1x1x256xf32>
      %3645 = stablehlo.reshape %arg1553 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3646 = stablehlo.multiply %3644, %3645 : tensor<1x1x1x256xf32>
      %3647 = stablehlo.broadcast_in_dim %3646, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3648 = stablehlo.multiply %3641, %3647 : tensor<256x14x14x256xf32>
      %3649 = stablehlo.reshape %arg1552 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3650 = stablehlo.broadcast_in_dim %3649, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3651 = stablehlo.add %3648, %3650 : tensor<256x14x14x256xf32>
      %3652 = stablehlo.convert %3651 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3653 = func.call @relu_175(%3652) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3654 = stablehlo.convert %arg1558 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3655 = stablehlo.convolution(%3653, %3654) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3656 = stablehlo.broadcast_in_dim %arg1988, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3657 = stablehlo.broadcast_in_dim %arg1989, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3658 = stablehlo.convert %3655 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3659 = stablehlo.broadcast_in_dim %3656, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3660 = stablehlo.subtract %3658, %3659 : tensor<256x14x14x1024xf32>
      %cst_189 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3661 = stablehlo.broadcast_in_dim %cst_189, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3662 = stablehlo.add %3657, %3661 : tensor<1x1x1x1024xf32>
      %3663 = stablehlo.rsqrt %3662 : tensor<1x1x1x1024xf32>
      %3664 = stablehlo.reshape %arg1555 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3665 = stablehlo.multiply %3663, %3664 : tensor<1x1x1x1024xf32>
      %3666 = stablehlo.broadcast_in_dim %3665, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3667 = stablehlo.multiply %3660, %3666 : tensor<256x14x14x1024xf32>
      %3668 = stablehlo.reshape %arg1554 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3669 = stablehlo.broadcast_in_dim %3668, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3670 = stablehlo.add %3667, %3669 : tensor<256x14x14x1024xf32>
      %3671 = stablehlo.convert %3670 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3672 = stablehlo.add %3615, %3671 : tensor<256x14x14x1024xf16>
      %3673 = func.call @relu_193(%3672) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3674 = stablehlo.convert %arg1565 : (tensor<1x1x1024x256xf32>) -> tensor<1x1x1024x256xf16>
      %3675 = stablehlo.convolution(%3673, %3674) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x256xf16>) -> tensor<256x14x14x256xf16>
      %3676 = stablehlo.broadcast_in_dim %arg1990, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3677 = stablehlo.broadcast_in_dim %arg1991, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3678 = stablehlo.convert %3675 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3679 = stablehlo.broadcast_in_dim %3676, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3680 = stablehlo.subtract %3678, %3679 : tensor<256x14x14x256xf32>
      %cst_190 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3681 = stablehlo.broadcast_in_dim %cst_190, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3682 = stablehlo.add %3677, %3681 : tensor<1x1x1x256xf32>
      %3683 = stablehlo.rsqrt %3682 : tensor<1x1x1x256xf32>
      %3684 = stablehlo.reshape %arg1560 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3685 = stablehlo.multiply %3683, %3684 : tensor<1x1x1x256xf32>
      %3686 = stablehlo.broadcast_in_dim %3685, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3687 = stablehlo.multiply %3680, %3686 : tensor<256x14x14x256xf32>
      %3688 = stablehlo.reshape %arg1559 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3689 = stablehlo.broadcast_in_dim %3688, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3690 = stablehlo.add %3687, %3689 : tensor<256x14x14x256xf32>
      %3691 = stablehlo.convert %3690 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3692 = func.call @relu_175(%3691) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3693 = stablehlo.convert %arg1566 : (tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf16>
      %3694 = stablehlo.convolution(%3692, %3693) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<3x3x256x256xf16>) -> tensor<256x14x14x256xf16>
      %3695 = stablehlo.broadcast_in_dim %arg1992, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3696 = stablehlo.broadcast_in_dim %arg1993, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3697 = stablehlo.convert %3694 : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf32>
      %3698 = stablehlo.broadcast_in_dim %3695, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3699 = stablehlo.subtract %3697, %3698 : tensor<256x14x14x256xf32>
      %cst_191 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3700 = stablehlo.broadcast_in_dim %cst_191, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
      %3701 = stablehlo.add %3696, %3700 : tensor<1x1x1x256xf32>
      %3702 = stablehlo.rsqrt %3701 : tensor<1x1x1x256xf32>
      %3703 = stablehlo.reshape %arg1562 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3704 = stablehlo.multiply %3702, %3703 : tensor<1x1x1x256xf32>
      %3705 = stablehlo.broadcast_in_dim %3704, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3706 = stablehlo.multiply %3699, %3705 : tensor<256x14x14x256xf32>
      %3707 = stablehlo.reshape %arg1561 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
      %3708 = stablehlo.broadcast_in_dim %3707, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x14x14x256xf32>
      %3709 = stablehlo.add %3706, %3708 : tensor<256x14x14x256xf32>
      %3710 = stablehlo.convert %3709 : (tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf16>
      %3711 = func.call @relu_175(%3710) : (tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16>
      %3712 = stablehlo.convert %arg1567 : (tensor<1x1x256x1024xf32>) -> tensor<1x1x256x1024xf16>
      %3713 = stablehlo.convolution(%3711, %3712) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x256xf16>, tensor<1x1x256x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3714 = stablehlo.broadcast_in_dim %arg1994, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3715 = stablehlo.broadcast_in_dim %arg1995, dims = [3] : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3716 = stablehlo.convert %3713 : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf32>
      %3717 = stablehlo.broadcast_in_dim %3714, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3718 = stablehlo.subtract %3716, %3717 : tensor<256x14x14x1024xf32>
      %cst_192 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3719 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<f32>) -> tensor<1x1x1x1024xf32>
      %3720 = stablehlo.add %3715, %3719 : tensor<1x1x1x1024xf32>
      %3721 = stablehlo.rsqrt %3720 : tensor<1x1x1x1024xf32>
      %3722 = stablehlo.reshape %arg1564 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3723 = stablehlo.multiply %3721, %3722 : tensor<1x1x1x1024xf32>
      %3724 = stablehlo.broadcast_in_dim %3723, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3725 = stablehlo.multiply %3718, %3724 : tensor<256x14x14x1024xf32>
      %3726 = stablehlo.reshape %arg1563 : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
      %3727 = stablehlo.broadcast_in_dim %3726, dims = [0, 1, 2, 3] : (tensor<1x1x1x1024xf32>) -> tensor<256x14x14x1024xf32>
      %3728 = stablehlo.add %3725, %3727 : tensor<256x14x14x1024xf32>
      %3729 = stablehlo.convert %3728 : (tensor<256x14x14x1024xf32>) -> tensor<256x14x14x1024xf16>
      %3730 = stablehlo.add %3673, %3729 : tensor<256x14x14x1024xf16>
      %3731 = func.call @relu_193(%3730) : (tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16>
      %3732 = stablehlo.convert %arg1574 : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf16>
      %3733 = stablehlo.convolution(%3731, %3732) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x512xf16>) -> tensor<256x14x14x512xf16>
      %3734 = stablehlo.broadcast_in_dim %arg1996, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3735 = stablehlo.broadcast_in_dim %arg1997, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3736 = stablehlo.convert %3733 : (tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf32>
      %3737 = stablehlo.broadcast_in_dim %3734, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
      %3738 = stablehlo.subtract %3736, %3737 : tensor<256x14x14x512xf32>
      %cst_193 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3739 = stablehlo.broadcast_in_dim %cst_193, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %3740 = stablehlo.add %3735, %3739 : tensor<1x1x1x512xf32>
      %3741 = stablehlo.rsqrt %3740 : tensor<1x1x1x512xf32>
      %3742 = stablehlo.reshape %arg1569 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3743 = stablehlo.multiply %3741, %3742 : tensor<1x1x1x512xf32>
      %3744 = stablehlo.broadcast_in_dim %3743, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
      %3745 = stablehlo.multiply %3738, %3744 : tensor<256x14x14x512xf32>
      %3746 = stablehlo.reshape %arg1568 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3747 = stablehlo.broadcast_in_dim %3746, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x14x14x512xf32>
      %3748 = stablehlo.add %3745, %3747 : tensor<256x14x14x512xf32>
      %3749 = stablehlo.convert %3748 : (tensor<256x14x14x512xf32>) -> tensor<256x14x14x512xf16>
      %3750 = func.call @relu_311(%3749) : (tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf16>
      %3751 = stablehlo.convert %arg1575 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %3752 = stablehlo.convolution(%3750, %3751) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %3753 = stablehlo.broadcast_in_dim %arg1998, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3754 = stablehlo.broadcast_in_dim %arg1999, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3755 = stablehlo.convert %3752 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %3756 = stablehlo.broadcast_in_dim %3753, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3757 = stablehlo.subtract %3755, %3756 : tensor<256x7x7x512xf32>
      %cst_194 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3758 = stablehlo.broadcast_in_dim %cst_194, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %3759 = stablehlo.add %3754, %3758 : tensor<1x1x1x512xf32>
      %3760 = stablehlo.rsqrt %3759 : tensor<1x1x1x512xf32>
      %3761 = stablehlo.reshape %arg1571 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3762 = stablehlo.multiply %3760, %3761 : tensor<1x1x1x512xf32>
      %3763 = stablehlo.broadcast_in_dim %3762, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3764 = stablehlo.multiply %3757, %3763 : tensor<256x7x7x512xf32>
      %3765 = stablehlo.reshape %arg1570 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3766 = stablehlo.broadcast_in_dim %3765, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3767 = stablehlo.add %3764, %3766 : tensor<256x7x7x512xf32>
      %3768 = stablehlo.convert %3767 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %3769 = func.call @relu_321(%3768) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %3770 = stablehlo.convert %arg1576 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
      %3771 = stablehlo.convolution(%3769, %3770) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
      %3772 = stablehlo.broadcast_in_dim %arg2000, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3773 = stablehlo.broadcast_in_dim %arg2001, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3774 = stablehlo.convert %3771 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %3775 = stablehlo.broadcast_in_dim %3772, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3776 = stablehlo.subtract %3774, %3775 : tensor<256x7x7x2048xf32>
      %cst_195 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3777 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %3778 = stablehlo.add %3773, %3777 : tensor<1x1x1x2048xf32>
      %3779 = stablehlo.rsqrt %3778 : tensor<1x1x1x2048xf32>
      %3780 = stablehlo.reshape %arg1573 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3781 = stablehlo.multiply %3779, %3780 : tensor<1x1x1x2048xf32>
      %3782 = stablehlo.broadcast_in_dim %3781, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3783 = stablehlo.multiply %3776, %3782 : tensor<256x7x7x2048xf32>
      %3784 = stablehlo.reshape %arg1572 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3785 = stablehlo.broadcast_in_dim %3784, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3786 = stablehlo.add %3783, %3785 : tensor<256x7x7x2048xf32>
      %3787 = stablehlo.convert %3786 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %3788 = stablehlo.convert %arg1577 : (tensor<1x1x1024x2048xf32>) -> tensor<1x1x1024x2048xf16>
      %3789 = stablehlo.convolution(%3731, %3788) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x14x14x1024xf16>, tensor<1x1x1024x2048xf16>) -> tensor<256x7x7x2048xf16>
      %3790 = stablehlo.broadcast_in_dim %arg2002, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3791 = stablehlo.broadcast_in_dim %arg2003, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3792 = stablehlo.convert %3789 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %3793 = stablehlo.broadcast_in_dim %3790, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3794 = stablehlo.subtract %3792, %3793 : tensor<256x7x7x2048xf32>
      %cst_196 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3795 = stablehlo.broadcast_in_dim %cst_196, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %3796 = stablehlo.add %3791, %3795 : tensor<1x1x1x2048xf32>
      %3797 = stablehlo.rsqrt %3796 : tensor<1x1x1x2048xf32>
      %3798 = stablehlo.reshape %arg1579 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3799 = stablehlo.multiply %3797, %3798 : tensor<1x1x1x2048xf32>
      %3800 = stablehlo.broadcast_in_dim %3799, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3801 = stablehlo.multiply %3794, %3800 : tensor<256x7x7x2048xf32>
      %3802 = stablehlo.reshape %arg1578 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3803 = stablehlo.broadcast_in_dim %3802, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3804 = stablehlo.add %3801, %3803 : tensor<256x7x7x2048xf32>
      %3805 = stablehlo.convert %3804 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %3806 = stablehlo.add %3805, %3787 : tensor<256x7x7x2048xf16>
      %3807 = func.call @relu_339(%3806) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
      %3808 = stablehlo.convert %arg1586 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
      %3809 = stablehlo.convolution(%3807, %3808) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<256x7x7x512xf16>
      %3810 = stablehlo.broadcast_in_dim %arg2004, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3811 = stablehlo.broadcast_in_dim %arg2005, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3812 = stablehlo.convert %3809 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %3813 = stablehlo.broadcast_in_dim %3810, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3814 = stablehlo.subtract %3812, %3813 : tensor<256x7x7x512xf32>
      %cst_197 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3815 = stablehlo.broadcast_in_dim %cst_197, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %3816 = stablehlo.add %3811, %3815 : tensor<1x1x1x512xf32>
      %3817 = stablehlo.rsqrt %3816 : tensor<1x1x1x512xf32>
      %3818 = stablehlo.reshape %arg1581 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3819 = stablehlo.multiply %3817, %3818 : tensor<1x1x1x512xf32>
      %3820 = stablehlo.broadcast_in_dim %3819, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3821 = stablehlo.multiply %3814, %3820 : tensor<256x7x7x512xf32>
      %3822 = stablehlo.reshape %arg1580 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3823 = stablehlo.broadcast_in_dim %3822, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3824 = stablehlo.add %3821, %3823 : tensor<256x7x7x512xf32>
      %3825 = stablehlo.convert %3824 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %3826 = func.call @relu_321(%3825) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %3827 = stablehlo.convert %arg1587 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %3828 = stablehlo.convolution(%3826, %3827) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %3829 = stablehlo.broadcast_in_dim %arg2006, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3830 = stablehlo.broadcast_in_dim %arg2007, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3831 = stablehlo.convert %3828 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %3832 = stablehlo.broadcast_in_dim %3829, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3833 = stablehlo.subtract %3831, %3832 : tensor<256x7x7x512xf32>
      %cst_198 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3834 = stablehlo.broadcast_in_dim %cst_198, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %3835 = stablehlo.add %3830, %3834 : tensor<1x1x1x512xf32>
      %3836 = stablehlo.rsqrt %3835 : tensor<1x1x1x512xf32>
      %3837 = stablehlo.reshape %arg1583 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3838 = stablehlo.multiply %3836, %3837 : tensor<1x1x1x512xf32>
      %3839 = stablehlo.broadcast_in_dim %3838, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3840 = stablehlo.multiply %3833, %3839 : tensor<256x7x7x512xf32>
      %3841 = stablehlo.reshape %arg1582 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3842 = stablehlo.broadcast_in_dim %3841, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3843 = stablehlo.add %3840, %3842 : tensor<256x7x7x512xf32>
      %3844 = stablehlo.convert %3843 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %3845 = func.call @relu_321(%3844) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %3846 = stablehlo.convert %arg1588 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
      %3847 = stablehlo.convolution(%3845, %3846) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
      %3848 = stablehlo.broadcast_in_dim %arg2008, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3849 = stablehlo.broadcast_in_dim %arg2009, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3850 = stablehlo.convert %3847 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %3851 = stablehlo.broadcast_in_dim %3848, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3852 = stablehlo.subtract %3850, %3851 : tensor<256x7x7x2048xf32>
      %cst_199 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3853 = stablehlo.broadcast_in_dim %cst_199, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %3854 = stablehlo.add %3849, %3853 : tensor<1x1x1x2048xf32>
      %3855 = stablehlo.rsqrt %3854 : tensor<1x1x1x2048xf32>
      %3856 = stablehlo.reshape %arg1585 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3857 = stablehlo.multiply %3855, %3856 : tensor<1x1x1x2048xf32>
      %3858 = stablehlo.broadcast_in_dim %3857, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3859 = stablehlo.multiply %3852, %3858 : tensor<256x7x7x2048xf32>
      %3860 = stablehlo.reshape %arg1584 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3861 = stablehlo.broadcast_in_dim %3860, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3862 = stablehlo.add %3859, %3861 : tensor<256x7x7x2048xf32>
      %3863 = stablehlo.convert %3862 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %3864 = stablehlo.add %3807, %3863 : tensor<256x7x7x2048xf16>
      %3865 = func.call @relu_339(%3864) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
      %3866 = stablehlo.convert %arg1595 : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf16>
      %3867 = stablehlo.convolution(%3865, %3866) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x2048xf16>, tensor<1x1x2048x512xf16>) -> tensor<256x7x7x512xf16>
      %3868 = stablehlo.broadcast_in_dim %arg2010, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3869 = stablehlo.broadcast_in_dim %arg2011, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3870 = stablehlo.convert %3867 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %3871 = stablehlo.broadcast_in_dim %3868, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3872 = stablehlo.subtract %3870, %3871 : tensor<256x7x7x512xf32>
      %cst_200 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3873 = stablehlo.broadcast_in_dim %cst_200, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %3874 = stablehlo.add %3869, %3873 : tensor<1x1x1x512xf32>
      %3875 = stablehlo.rsqrt %3874 : tensor<1x1x1x512xf32>
      %3876 = stablehlo.reshape %arg1590 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3877 = stablehlo.multiply %3875, %3876 : tensor<1x1x1x512xf32>
      %3878 = stablehlo.broadcast_in_dim %3877, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3879 = stablehlo.multiply %3872, %3878 : tensor<256x7x7x512xf32>
      %3880 = stablehlo.reshape %arg1589 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3881 = stablehlo.broadcast_in_dim %3880, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3882 = stablehlo.add %3879, %3881 : tensor<256x7x7x512xf32>
      %3883 = stablehlo.convert %3882 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %3884 = func.call @relu_321(%3883) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %3885 = stablehlo.convert %arg1596 : (tensor<3x3x512x512xf32>) -> tensor<3x3x512x512xf16>
      %3886 = stablehlo.convolution(%3884, %3885) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<3x3x512x512xf16>) -> tensor<256x7x7x512xf16>
      %3887 = stablehlo.broadcast_in_dim %arg2012, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3888 = stablehlo.broadcast_in_dim %arg2013, dims = [3] : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3889 = stablehlo.convert %3886 : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf32>
      %3890 = stablehlo.broadcast_in_dim %3887, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3891 = stablehlo.subtract %3889, %3890 : tensor<256x7x7x512xf32>
      %cst_201 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3892 = stablehlo.broadcast_in_dim %cst_201, dims = [] : (tensor<f32>) -> tensor<1x1x1x512xf32>
      %3893 = stablehlo.add %3888, %3892 : tensor<1x1x1x512xf32>
      %3894 = stablehlo.rsqrt %3893 : tensor<1x1x1x512xf32>
      %3895 = stablehlo.reshape %arg1592 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3896 = stablehlo.multiply %3894, %3895 : tensor<1x1x1x512xf32>
      %3897 = stablehlo.broadcast_in_dim %3896, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3898 = stablehlo.multiply %3891, %3897 : tensor<256x7x7x512xf32>
      %3899 = stablehlo.reshape %arg1591 : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
      %3900 = stablehlo.broadcast_in_dim %3899, dims = [0, 1, 2, 3] : (tensor<1x1x1x512xf32>) -> tensor<256x7x7x512xf32>
      %3901 = stablehlo.add %3898, %3900 : tensor<256x7x7x512xf32>
      %3902 = stablehlo.convert %3901 : (tensor<256x7x7x512xf32>) -> tensor<256x7x7x512xf16>
      %3903 = func.call @relu_321(%3902) : (tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16>
      %3904 = stablehlo.convert %arg1597 : (tensor<1x1x512x2048xf32>) -> tensor<1x1x512x2048xf16>
      %3905 = stablehlo.convolution(%3903, %3904) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x7x7x512xf16>, tensor<1x1x512x2048xf16>) -> tensor<256x7x7x2048xf16>
      %3906 = stablehlo.broadcast_in_dim %arg2014, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3907 = stablehlo.broadcast_in_dim %arg2015, dims = [3] : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3908 = stablehlo.convert %3905 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %3909 = stablehlo.broadcast_in_dim %3906, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3910 = stablehlo.subtract %3908, %3909 : tensor<256x7x7x2048xf32>
      %cst_202 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %3911 = stablehlo.broadcast_in_dim %cst_202, dims = [] : (tensor<f32>) -> tensor<1x1x1x2048xf32>
      %3912 = stablehlo.add %3907, %3911 : tensor<1x1x1x2048xf32>
      %3913 = stablehlo.rsqrt %3912 : tensor<1x1x1x2048xf32>
      %3914 = stablehlo.reshape %arg1594 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3915 = stablehlo.multiply %3913, %3914 : tensor<1x1x1x2048xf32>
      %3916 = stablehlo.broadcast_in_dim %3915, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3917 = stablehlo.multiply %3910, %3916 : tensor<256x7x7x2048xf32>
      %3918 = stablehlo.reshape %arg1593 : (tensor<2048xf32>) -> tensor<1x1x1x2048xf32>
      %3919 = stablehlo.broadcast_in_dim %3918, dims = [0, 1, 2, 3] : (tensor<1x1x1x2048xf32>) -> tensor<256x7x7x2048xf32>
      %3920 = stablehlo.add %3917, %3919 : tensor<256x7x7x2048xf32>
      %3921 = stablehlo.convert %3920 : (tensor<256x7x7x2048xf32>) -> tensor<256x7x7x2048xf16>
      %3922 = stablehlo.add %3865, %3921 : tensor<256x7x7x2048xf16>
      %3923 = func.call @relu_339(%3922) : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16>
      %3924 = stablehlo.convert %3923 : (tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf32>
      %cst_203 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3925 = stablehlo.reduce(%3924 init: %cst_203) applies stablehlo.add across dimensions = [1, 2] : (tensor<256x7x7x2048xf32>, tensor<f32>) -> tensor<256x2048xf32>
      %cst_204 = stablehlo.constant dense<4.900000e+01> : tensor<f32>
      %3926 = stablehlo.broadcast_in_dim %cst_204, dims = [] : (tensor<f32>) -> tensor<256x2048xf32>
      %3927 = stablehlo.divide %3925, %3926 : tensor<256x2048xf32>
      %3928 = stablehlo.convert %3927 : (tensor<256x2048xf32>) -> tensor<256x2048xf16>
      %3929 = stablehlo.convert %arg1626 : (tensor<2048x1000xf32>) -> tensor<2048x1000xf16>
      %3930 = stablehlo.convert %arg1625 : (tensor<1000xf32>) -> tensor<1000xf16>
      %3931 = stablehlo.dot_general %3928, %3929, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf16>, tensor<2048x1000xf16>) -> tensor<256x1000xf16>
      %3932 = stablehlo.reshape %3930 : (tensor<1000xf16>) -> tensor<1x1000xf16>
      %3933 = stablehlo.broadcast_in_dim %3932, dims = [0, 1] : (tensor<1x1000xf16>) -> tensor<256x1000xf16>
      %3934 = stablehlo.add %3931, %3933 : tensor<256x1000xf16>
      %3935 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
      %3936 = stablehlo.iota dim = 0 : tensor<1000xi32>
      %3937 = stablehlo.reshape %3936 : (tensor<1000xi32>) -> tensor<1x1000xi32>
      %3938 = stablehlo.broadcast_in_dim %3935, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1000xi32>
      %3939 = stablehlo.broadcast_in_dim %3937, dims = [0, 1] : (tensor<1x1000xi32>) -> tensor<256x1000xi32>
      %3940 = stablehlo.compare  EQ, %3938, %3939,  SIGNED : (tensor<256x1000xi32>, tensor<256x1000xi32>) -> tensor<256x1000xi1>
      %cst_205 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %3941 = stablehlo.broadcast_in_dim %cst_205, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %cst_206 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3942 = stablehlo.broadcast_in_dim %cst_206, dims = [] : (tensor<f32>) -> tensor<256x1000xf32>
      %3943 = stablehlo.select %3940, %3941, %3942 : tensor<256x1000xi1>, tensor<256x1000xf32>
      %3944 = stablehlo.convert %3943 : tensor<256x1000xf32>
      %3945 = func.call @log_softmax(%3934) : (tensor<256x1000xf16>) -> tensor<256x1000xf16>
      %3946 = stablehlo.convert %3945 : (tensor<256x1000xf16>) -> tensor<256x1000xf32>
      %3947 = stablehlo.multiply %3944, %3946 : tensor<256x1000xf32>
      %cst_207 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3948 = stablehlo.reduce(%3947 init: %cst_207) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
      %3949 = stablehlo.negate %3948 : tensor<256xf32>
      %cst_208 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3950 = stablehlo.reduce(%3949 init: %cst_208) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_209 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %3951 = stablehlo.divide %3950, %cst_209 : tensor<f32>
      %3952 = func.call @argmax(%3934) : (tensor<256x1000xf16>) -> tensor<256xi32>
      %3953 = stablehlo.compare  EQ, %3952, %2,  SIGNED : (tensor<256xi32>, tensor<256xi32>) -> tensor<256xi1>
      %3954 = stablehlo.convert %3953 : (tensor<256xi1>) -> tensor<256xi32>
      %3955 = stablehlo.convert %3954 : (tensor<256xi32>) -> tensor<256xf32>
      %cst_210 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3956 = stablehlo.reduce(%3955 init: %cst_210) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
      %cst_211 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
      %3957 = stablehlo.divide %3956, %cst_211 : tensor<f32>
      %3958 = "stablehlo.all_reduce"(%3957) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg2038: tensor<f32>, %arg2039: tensor<f32>):
        %3964 = stablehlo.add %arg2038, %arg2039 : tensor<f32>
        stablehlo.return %3964 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %3959 = "stablehlo.all_reduce"(%3951) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
      ^bb0(%arg2038: tensor<f32>, %arg2039: tensor<f32>):
        %3964 = stablehlo.add %arg2038, %arg2039 : tensor<f32>
        stablehlo.return %3964 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_212 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3960 = stablehlo.divide %3958, %cst_212 : tensor<f32>
      %cst_213 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %3961 = stablehlo.divide %3959, %cst_213 : tensor<f32>
      %3962 = stablehlo.broadcast_in_dim %3960, dims = [] : (tensor<f32>) -> tensor<1xf32>
      %3963 = stablehlo.broadcast_in_dim %3961, dims = [] : (tensor<f32>) -> tensor<1xf32>
      sdy.return %3962, %3963 : tensor<1xf32>, tensor<1xf32>
    } : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x1x64x64xf32>, tensor<3x3x64x64xf32>, tensor<1x1x64x256xf32>, tensor<1x1x64x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x1x256x64xf32>, tensor<3x3x64x64xf32>, tensor<1x1x64x256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x1x256x64xf32>, tensor<3x3x64x64xf32>, tensor<1x1x64x256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x512x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<1x1x512x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x256x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<1x1x256x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1x1x1024x256xf32>, tensor<3x3x256x256xf32>, tensor<1x1x256x1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x1x1024x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x512x2048xf32>, tensor<1x1x1024x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x1x2048x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x512x2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1x1x2048x512xf32>, tensor<3x3x512x512xf32>, tensor<1x1x512x2048xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x1x512x128xf32>, tensor<3x3x128x128xf32>, tensor<1x1x128x512xf32>, tensor<1000xf32>, tensor<2048x1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<7x7x3x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4x256x224x224x3xf16>, tensor<4x256xi32>) -> (tensor<4xf32>, tensor<4xf32>)
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
  func.func private @relu_165(%arg0: tensor<256x28x28x256xf16>) -> tensor<256x28x28x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x28x28x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x28x28x256xf16>
    return %1 : tensor<256x28x28x256xf16>
  }
  func.func private @relu_175(%arg0: tensor<256x14x14x256xf16>) -> tensor<256x14x14x256xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x256xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x256xf16>
    return %1 : tensor<256x14x14x256xf16>
  }
  func.func private @relu_193(%arg0: tensor<256x14x14x1024xf16>) -> tensor<256x14x14x1024xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x1024xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x1024xf16>
    return %1 : tensor<256x14x14x1024xf16>
  }
  func.func private @relu_311(%arg0: tensor<256x14x14x512xf16>) -> tensor<256x14x14x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x14x14x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x14x14x512xf16>
    return %1 : tensor<256x14x14x512xf16>
  }
  func.func private @relu_321(%arg0: tensor<256x7x7x512xf16>) -> tensor<256x7x7x512xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<256x7x7x512xf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<256x7x7x512xf16>
    return %1 : tensor<256x7x7x512xf16>
  }
  func.func private @relu_339(%arg0: tensor<256x7x7x2048xf16>) -> tensor<256x7x7x2048xf16> {
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
